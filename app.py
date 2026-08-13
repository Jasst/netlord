# app.py
import asyncio
import json
import os
import time
import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from typing import Optional, List
import atexit
import signal
import sys
from openai import OpenAI
# ---------- ГРАФ ДЛЯ ВИЗУАЛИЗАЦИИ (API) ----------
import torch.nn as nn
from brain.graph import NodeType
from brain import CognitiveBrain, BrainConfig
from brain.teacher import Teacher
from agent import BrainAgent
import uuid
import subprocess
import threading
import tempfile
import os

torch.set_default_dtype(torch.float32)

LM_STUDIO_BASE_URL = os.environ.get("LM_STUDIO_BASE_URL", "http://192.168.0.13:1234/v1")

config = BrainConfig(
    llm_base_url=LM_STUDIO_BASE_URL,
)

brain = CognitiveBrain(config)
brain.load()
brain.load_dialog_history()

_llm_client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key="not-needed")
_teacher = Teacher(llm_client=_llm_client)
agent = BrainAgent(
    brain=brain,
    teacher=_teacher,
    llm_client=_llm_client,
    interactive_mode=False,
    enabled=False,
)

app = FastAPI(title="Smart Brain v10")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------- МОДЕЛИ PYDANTIC ----------
class AskRequest(BaseModel):
    question: str
    temperature: float = 0.7
    use_search: bool = False
    allow_clarifying: bool = True

class LearnRequest(BaseModel):
    question: str
    answer: str

class TrainTopicRequest(BaseModel):
    topic: str
    num_pairs: int = 30
    negative_ratio: float = 0.2
    epochs: int = 1

class AgentToggleRequest(BaseModel):
    enabled: bool

class AgentConfigRequest(BaseModel):
    topics: Optional[List[str]] = None
    interval: Optional[int] = None
    questions_per_cycle: Optional[int] = None
    interactive_mode: Optional[bool] = None
    user_question_timeout: Optional[int] = None
    # НОВЫЕ ПОЛЯ
    teacher_threshold: Optional[float] = None
    use_dynamic_threshold: Optional[bool] = None

class TrainPairRequest(BaseModel):
    question: str
    answer: str
    epochs: int = 3


# ---------- ЭНДПОИНТЫ ----------
@app.post("/ask")
async def ask(req: AskRequest):
    try:
        result = await asyncio.to_thread(
            brain.step, req.question, use_search=req.use_search, temperature=req.temperature
        )
        answer = result["answer"]
        thoughts = result.get("thoughts", answer)
        await asyncio.to_thread(brain.save_dialog_history)
        return {
            "question": req.question,
            "answer": answer,
            "thoughts": thoughts,
            "facts": result.get("memory_results", []),
            "known": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask_stream")
async def ask_stream(req: AskRequest):
    try:
        result = await asyncio.to_thread(
            brain.step, req.question, use_search=req.use_search, temperature=req.temperature
        )
        answer = result["answer"]
        thoughts = result.get("thoughts", answer)

        async def generate():
            full = ""
            for token in answer.split():
                full += token
                yield f"data: {json.dumps({'token': token + ' '})}\n\n"
            yield f"data: {json.dumps({'done': True, 'full_answer': answer, 'thoughts': thoughts})}\n\n"
            await asyncio.to_thread(brain.save_dialog_history)

        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Хранилище для заданий bulk_train
bulk_jobs = {}  # job_id -> {'status': 'running'|'completed'|'error', 'log': [], 'total_pairs': 0}

@app.post("/upload_jsonl")
async def upload_jsonl(request: Request):
    """
    Принимает JSONL-файл и запускает bulk_train в фоновом потоке.
    Поддерживает параметры валидации.
    """
    form = await request.form()
    file = form.get("file")
    if not file:
        raise HTTPException(400, "No file uploaded")

    sleep_every = int(form.get("sleep_every", 200))
    reward = float(form.get("reward", 1.0))
    validate = form.get("validate", "false").lower() == "true"
    validation_threshold = float(form.get("validation_threshold", 0.6))
    validation_retries = int(form.get("validation_retries", 1))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    job_id = str(uuid.uuid4())[:8]
    bulk_jobs[job_id] = {"status": "running", "log": [], "total_pairs": 0}

    def run_bulk_train():
        try:
            cmd = [
                "python", "bulk_train.py",
                "--data", tmp_path,
                "--sleep-every", str(sleep_every),
                "--reward", str(reward),
                "--model-dir", brain.config.model_dir
            ]
            if validate:
                cmd.append("--validate")
                cmd.append("--validation-threshold")
                cmd.append(str(validation_threshold))
                cmd.append("--validation-retries")
                cmd.append(str(validation_retries))
                # Можно добавить --sample-rate, если нужно
                # cmd.extend(["--sample-rate", "1.0"])

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            for line in iter(process.stdout.readline, ''):
                if line:
                    bulk_jobs[job_id]["log"].append(line.strip())
                    if "Готово. Загружено" in line:
                        import re
                        m = re.search(r"Загружено (\d+) пар", line)
                        if m:
                            bulk_jobs[job_id]["total_pairs"] = int(m.group(1))
            process.stdout.close()
            return_code = process.wait()
            if return_code == 0:
                bulk_jobs[job_id]["status"] = "completed"
            else:
                bulk_jobs[job_id]["status"] = "error"
                bulk_jobs[job_id]["log"].append(f"Процесс завершился с кодом {return_code}")
        except Exception as e:
            bulk_jobs[job_id]["status"] = "error"
            bulk_jobs[job_id]["log"].append(f"Ошибка: {str(e)}")
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass

    thread = threading.Thread(target=run_bulk_train)
    thread.start()

    return {"job_id": job_id, "status": "started"}


@app.get("/upload_jsonl/logs/{job_id}")
async def get_bulk_logs(job_id: str, last: int = 0):
    """
    Возвращает новые строки лога для задания.
    """
    job = bulk_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    log = job.get("log", [])
    new_lines = log[last:]
    return {
        "lines": new_lines,
        "last_line": len(log),
        "done": job["status"] != "running",
        "status": job["status"],
        "total_pairs": job.get("total_pairs", 0)
    }

@app.get("/graph-editor", response_class=HTMLResponse)
async def graph_editor():
    with open("templates/graph_editor.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/learn")
async def learn(req: LearnRequest):
    try:
        await asyncio.to_thread(brain.learn_pair, req.question, req.answer)
        await asyncio.to_thread(brain.save)
        return {"status": "learned"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/brain/save")
async def save_brain():
    brain.save()
    return {"status": "saved"}

@app.post("/learn_neg")
async def learn_neg(req: LearnRequest):
    try:
        await asyncio.to_thread(brain.learn_negative_pair, req.question, req.answer)
        await asyncio.to_thread(brain.save)
        return {"status": "learned_negative"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/train_pair")
async def train_pair(req: TrainPairRequest):
    try:
        await asyncio.to_thread(brain.learn_pair, req.question, req.answer, epochs=req.epochs)
        await asyncio.to_thread(brain.save)
        return {"status": "ok", "epochs": req.epochs}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/train_topic")
async def train_topic(req: TrainTopicRequest):
    try:
        pairs = await asyncio.to_thread(_generate_training_pairs, req.topic, req.num_pairs)
        for q, a in pairs:
            brain.learn_pair(q, a, epochs=req.epochs)
        brain.sleep()
        brain.save()
        return {"status": "ok", "pairs_learned": len(pairs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _generate_training_pairs(topic: str, num_pairs: int) -> List[tuple]:
    prompt = f"Сгенерируй {num_pairs} пар 'вопрос|ответ' по теме '{topic}'. Формат: каждая пара на новой строке, разделённая '|'."
    response = brain.llm.generate(prompt, max_tokens=num_pairs * 30, temperature=0.9)
    pairs = []
    for line in response.splitlines():
        if '|' in line:
            q, a = line.split('|', 1)
            pairs.append((q.strip(), a.strip()))
    return pairs[:num_pairs]


@app.get("/stats")
async def stats():
    return brain.get_stats()


@app.post("/sleep")
async def sleep_brain():
    await asyncio.to_thread(brain.sleep)
    return {"status": "sleep_done"}


@app.post("/chat/clear")
async def clear_chat():
    brain.dialog_memory.clear()
    await asyncio.to_thread(brain.save_dialog_history)
    return {"status": "cleared"}


@app.get("/chat/messages")
async def get_messages(limit: int = 50):
    return {"messages": brain.dialog_memory[-limit:]}


# ---------- УПРАВЛЕНИЕ АГЕНТОМ ----------
@app.get("/agent/status")
async def agent_status():
    return {
        "enabled": agent.enabled,
        "topic_confidence": agent.topic_confidence,
        "interactive_mode": agent.interactive_mode,
    }


@app.post("/agent/toggle")
async def agent_toggle(req: AgentToggleRequest):
    agent.enabled = req.enabled
    return {"enabled": agent.enabled}


@app.post("/agent/start")
async def agent_start():
    agent.enabled = True
    agent.start()
    return {"status": "started"}


@app.post("/agent/stop")
async def agent_stop():
    agent.enabled = False
    agent.stop()
    return {"status": "stopped"}


@app.get("/agent/next_question")
async def agent_next_question():
    q = agent.get_next_question()
    if q:
        return {"question": q}
    return {"question": None}


@app.post("/agent/submit_answer")
async def agent_submit_answer(req: dict):
    question = req.get("question")
    answer = req.get("answer")
    if not question or not answer:
        raise HTTPException(400, "Missing question or answer")
    agent.submit_answer(question, answer)
    return {"status": "accepted"}


@app.post("/agent/config")
async def agent_config(req: AgentConfigRequest):
    if req.topics is not None:
        agent.topics = req.topics
        agent.topic_confidence = {t: 0.5 for t in req.topics}
        agent.asked_questions = {t: [] for t in req.topics}
    if req.interval is not None:
        agent.interval = req.interval
    if req.questions_per_cycle is not None:
        agent.questions_per_cycle = req.questions_per_cycle
    if req.interactive_mode is not None:
        agent.interactive_mode = req.interactive_mode
    if req.user_question_timeout is not None:
        agent.user_question_timeout = req.user_question_timeout
    # НОВЫЕ ПАРАМЕТРЫ
    if req.teacher_threshold is not None:
        agent.teacher_threshold = req.teacher_threshold
    if req.use_dynamic_threshold is not None:
        agent.use_dynamic_threshold = req.use_dynamic_threshold
    return {"status": "updated", "topics": agent.topics}


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/brain/last_thought")
async def last_thought():
    if hasattr(brain, "proactive_thoughts") and brain.proactive_thoughts:
        return {"thought": brain.proactive_thoughts[-1]["thought"]}
    return {"thought": None}


def _get_graph_data(limit: int = 500):
    """Возвращает узлы и рёбра графа для визуализации."""
    graph = brain.graph
    if hasattr(graph, 'levels'):
        g = graph.levels[0]
    else:
        g = graph

    nodes = []
    n_nodes = g.node_emb.shape[0]
    if n_nodes == 0:
        return {"nodes": [], "edges": []}

    if n_nodes > limit:
        degree = {}
        for f, t in g._edges:
            f = int(f)
            t = int(t)
            if f <= n_nodes: degree[f] = degree.get(f, 0) + 1
            if t <= n_nodes: degree[t] = degree.get(t, 0) + 1
        top = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:limit]
        selected = set(nid for nid, _ in top)
    else:
        selected = set(range(1, n_nodes+1))

    for nid in selected:
        label = g.node_labels.get(nid, f"Нейрон {nid}")
        short_label = label[:20] + ("..." if len(label) > 20 else "")
        ntype = g.node_types.get(nid, NodeType.CONCEPT)
        cluster = g.node_clusters.get(nid, "hidden")
        color = {
            NodeType.SENSORY: "#FF6B6B",
            NodeType.CONCEPT: "#4ECDC4",
            NodeType.MOTOR: "#45B7D1",
            NodeType.EMOTIONAL: "#FFA07A",
            NodeType.ATTENTION: "#DDA0DD",
        }.get(ntype, "#AAAAAA")
        nodes.append({
            "id": nid,
            "label": short_label,
            "full_label": label,
            "node_type": ntype.name,
            "cluster": cluster,
            "title": f"ID: {nid}\nТип: {ntype.name}\nМетка: {label}",
            "color": color,
            "shape": "dot" if ntype == NodeType.CONCEPT else "box"
        })

    edges = []
    for (f, t), w in zip(g._edges, g._edge_weights):
        f = int(f)
        t = int(t)
        if f in selected and t in selected:
            w_val = w.item()
            width = max(1, abs(w_val) * 5)
            color = "#00FF00" if w > 0 else "#FF4444"
            edges.append({
                "from": f,
                "to": t,
                "value": abs(w_val),
                "weight": round(w_val, 4),
                "title": f"вес: {w_val:.3f}",
                "color": color,
                "width": width
            })

    return {"nodes": nodes, "edges": edges}

@app.get("/graph/data")
async def graph_data(limit: int = 500):
    return _get_graph_data(limit)

@app.post("/graph/add_node")
async def add_node(req: dict):
    label = (req.get("label") or "").strip()
    node_type_str = (req.get("node_type") or "CONCEPT").strip().upper()
    if not label:
        raise HTTPException(400, "Missing label")
    try:
        node_type = NodeType[node_type_str]
    except KeyError:
        node_type = NodeType.CONCEPT
    with brain.lock:
        emb = brain.text_to_embedding(label, is_query=False)
        nid = brain.graph.add_node(
            emb, label=label, cluster="manual", node_type=node_type,
            optimizer=brain.optimizer
        )
        brain.save()
    return {"status": "added", "id": nid}


@app.post("/graph/delete_node")
async def delete_node(req: dict):
    node_id = req.get("id")
    if not node_id:
        raise HTTPException(400, "Missing id")
    node_id = int(node_id)
    with brain.lock:
        graph = brain.graph
        if hasattr(graph, 'levels'):
            g = graph.levels[0]
        else:
            g = graph

        # Проверяем, что узел существует
        if node_id < 1 or node_id > g.node_emb.shape[0]:
            raise HTTPException(400, f"Node {node_id} does not exist")

        # 1. Удаляем все рёбра, связанные с узлом
        new_edges = []
        new_weights = []
        for (f, t), w in zip(g._edges, g._edge_weights):
            f = int(f)
            t = int(t)
            if f != node_id and t != node_id:
                new_edges.append((f, t))
                new_weights.append(w)
        g._edges = new_edges
        g._edge_weights = nn.ParameterList(new_weights)

        # 2. Удаляем сам узел из тензора эмбеддингов
        old_emb = g.node_emb
        mask = torch.ones(old_emb.shape[0], dtype=torch.bool)
        mask[node_id-1] = False
        new_emb = nn.Parameter(old_emb.data[mask])
        g.node_emb = new_emb

        # 3. Обновляем оптимизатор (заменяем параметр)
        for group in brain.optimizer.param_groups:
            for i, p in enumerate(group["params"]):
                if p is old_emb:
                    group["params"][i] = new_emb
                    state = brain.optimizer.state.pop(old_emb, None)
                    if state:
                        brain.optimizer.state[new_emb] = state
                    break

        # 4. Удаляем метаданные для этого узла
        g.node_labels.pop(node_id, None)
        g.node_types.pop(node_id, None)
        g.node_clusters.pop(node_id, None)

        # 5. Сдвигаем индексы всех оставшихся узлов (для рёбер и метаданных)
        #    Все ID больше node_id уменьшаем на 1
        def shift_dict(d: dict):
            new_d = {}
            for k, v in d.items():
                if k == node_id:
                    continue
                new_k = k if k < node_id else k - 1
                new_d[new_k] = v
            return new_d

        g.node_labels = shift_dict(g.node_labels)
        g.node_types = shift_dict(g.node_types)
        g.node_clusters = shift_dict(g.node_clusters)

        # 6. Сдвигаем индексы в рёбрах
        new_edges2 = []
        for f, t in g._edges:
            f = int(f)
            t = int(t)
            f2 = f if f < node_id else f-1
            t2 = t if t < node_id else t-1
            new_edges2.append((f2, t2))
        g._edges = new_edges2
        g._rebuild_edges()

        # 7. Удаляем ссылки из concept_index (и сдвигаем ID)
        new_concept_index = {}
        for key, val in brain.concept_index.items():
            if val == node_id:
                continue
            new_val = val if val < node_id else val - 1
            new_concept_index[key] = new_val
        brain.concept_index = new_concept_index

        # 8. Если граф иерархический, удаляем ссылки в cluster_of (уровень 0)
        if hasattr(graph, 'levels') and len(graph.cluster_of) > 0:
            # cluster_of[0] – словарь для первого уровня (нижний -> верхний)
            cluster_dict = graph.cluster_of[0]
            new_cluster = {}
            for member, cluster_id in cluster_dict.items():
                if member == node_id:
                    continue
                new_member = member if member < node_id else member - 1
                # cluster_id (ID на верхнем уровне) тоже может сдвинуться?
                # Но верхний уровень не затрагивается, и ID кластеров не меняются,
                # поэтому оставляем как есть.
                new_cluster[new_member] = cluster_id
            graph.cluster_of[0] = new_cluster

            # Перестраиваем иерархию, чтобы синхронизировать верхние уровни
            graph.rebuild_hierarchy(optimizer=brain.optimizer)

        # 9. Сохраняем изменения
        brain.save()
    return {"status": "deleted"}

@app.post("/graph/update_label")
async def update_label(req: dict):
    node_id = req.get("id")
    new_label = req.get("label")
    node_type_str = req.get("node_type")
    cluster = req.get("cluster")
    reembed = req.get("reembed", True)
    if not node_id:
        raise HTTPException(400, "Missing id")
    node_id = int(node_id)
    with brain.lock:
        graph = brain.graph
        if hasattr(graph, 'levels'):
            g = graph.levels[0]
        else:
            g = graph

        if new_label is not None and new_label.strip():
            g.node_labels[node_id] = new_label.strip()
            if reembed:
                new_emb = brain.text_to_embedding(new_label.strip(), is_query=False)
                with torch.no_grad():
                    g.node_emb.data[node_id - 1] = new_emb.to(
                        device=g.node_emb.data.device, dtype=g.node_emb.data.dtype
                    )

        if node_type_str:
            try:
                g.node_types[node_id] = NodeType[node_type_str.upper()]
            except KeyError:
                pass

        if cluster is not None and cluster.strip():
            g.node_clusters[node_id] = cluster.strip()

        brain.save()
    return {"status": "updated"}

@app.post("/graph/add_edge")
async def add_edge(req: dict):
    from_id = req.get("from")
    to_id = req.get("to")
    weight = req.get("weight", 0.5)
    if from_id is None or to_id is None:
        raise HTTPException(400, "Missing from/to")
    from_id = int(from_id)
    to_id = int(to_id)
    with brain.lock:
        brain.graph.add_synapse(from_id, to_id, weight, optimizer=brain.optimizer)
        brain.save()
    return {"status": "added"}

@app.post("/graph/delete_edge")
async def delete_edge(req: dict):
    from_id = req.get("from")
    to_id = req.get("to")
    if not from_id or not to_id:
        raise HTTPException(400, "Missing from/to")
    from_id = int(from_id)
    to_id = int(to_id)
    with brain.lock:
        graph = brain.graph
        if hasattr(graph, 'levels'):
            g = graph.levels[0]
        else:
            g = graph
        # удаляем ребро
        new_edges = []
        new_weights = []
        for (f, t), w in zip(g._edges, g._edge_weights):
            f = int(f)
            t = int(t)
            if not (f == from_id and t == to_id):
                new_edges.append((f, t))
                new_weights.append(w)
        g._edges = new_edges
        g._edge_weights = nn.ParameterList(new_weights)
        g._rebuild_edges()
        brain.save()
    return {"status": "deleted"}

@app.get("/graph/refresh")
async def refresh_graph():
    brain.load()
    brain.load_dialog_history()
    return {"status": "refreshed"}

@app.post("/graph/update_edge")
async def update_edge(req: dict):
    from_id = req.get("from")
    to_id = req.get("to")
    weight = req.get("weight")
    if from_id is None or to_id is None or weight is None:
        raise HTTPException(400, "Missing fields")
    with brain.lock:
        g = brain.graph.levels[0] if hasattr(brain.graph, 'levels') else brain.graph
        for i, (f, t) in enumerate(g._edges):
            if f == from_id and t == to_id:
                g._edge_weights[i] = nn.Parameter(torch.tensor(weight, dtype=torch.float))
                g._rebuild_edges()
                brain.save()
                return {"status": "updated"}
        raise HTTPException(404, "Edge not found")


# ---------- ЗАВЕРШЕНИЕ ----------
def save_brain():
    print("\n💾 Сохраняем модель...")
    agent.stop()
    brain.save()
    brain.save_dialog_history()


atexit.register(save_brain)
signal.signal(signal.SIGINT, lambda s, f: (save_brain(), sys.exit(0)))
signal.signal(signal.SIGTERM, lambda s, f: (save_brain(), sys.exit(0)))

# ===== НОВЫЕ ЭНДПОИНТЫ ДЛЯ КОГНИТИВНЫХ МОДУЛЕЙ =====

@app.get("/world_model/stats")
async def world_model_stats():
    """Статистика модели мира."""
    if brain.world_model is None:
        return {"error": "World model disabled"}
    return brain.world_model.get_stats()

@app.get("/world_model/facts")
async def world_model_facts(limit: int = 20, min_confidence: float = 0.0):
    """Получить последние факты из модели мира."""
    if brain.world_model is None:
        return {"facts": []}
    facts = brain.world_model.get_facts(min_confidence=min_confidence)
    # Сортируем по времени (свежие сверху)
    facts = sorted(facts, key=lambda f: f.timestamp, reverse=True)[:limit]
    return {
        "facts": [
            {
                "subject": f.subject,
                "relation": f.relation,
                "object": f.object,
                "confidence": f.confidence,
                "source": f.source,
                "timestamp": f.timestamp,
                "is_hypothesis": f.is_hypothesis,
                "evidence": f.evidence[:3],
            }
            for f in facts
        ]
    }

@app.get("/world_model/hypotheses")
async def world_model_hypotheses():
    """Получить активные гипотезы."""
    if brain.world_model is None:
        return {"hypotheses": []}
    hyps = brain.world_model.hypotheses[-20:]  # последние 20
    return {
        "hypotheses": [
            {
                "content": h.content,
                "confidence": h.confidence,
                "generated_from": h.generated_from,
                "verified": h.verified,
                "result": h.result,
                "tests": h.tests,
            }
            for h in hyps
        ]
    }

@app.get("/self_model/status")
async def self_model_status():
    """Состояние Self-модели."""
    if brain.self_model is None:
        return {"error": "Self model disabled"}
    return {
        "capabilities": brain.self_model.capabilities,
        "recent_errors": brain.self_model.recent_errors[-5:],
        "successful_strategies": brain.self_model.successful_strategies[-5:],
        "unresolved_questions": brain.self_model.unresolved_questions[-5:],
        "knowledge": [
            {"aspect": k.aspect, "description": k.description, "confidence": k.confidence}
            for k in brain.self_model.knowledge[-10:]
        ],
    }

@app.get("/planner/goals")
async def planner_goals():
    """Текущие цели и планы."""
    if brain.planner is None:
        return {"goals": []}
    goals = brain.planner.goals[-10:]
    return {
        "goals": [
            {
                "description": g.description,
                "priority": g.priority,
                "status": g.status,
                "plan": [
                    {"type": a.type, "params": a.params, "status": a.status}
                    for a in g.plan
                ],
                "created": g.created,
                "completed": g.completed,
            }
            for g in goals
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)