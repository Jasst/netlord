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

from brain import CognitiveBrain, BrainConfig
from brain.teacher import Teacher
from agent import BrainAgent

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
        thoughts = result.get("thoughts", answer)  # если не включен two_level_answer, будет равно answer
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
        # Чтобы получить и мысли, и ответ, делаем обычный step, но стримим только краткий ответ.
        # Можно сделать два вызова, но проще сначала получить всё, а потом стримить краткий.
        result = await asyncio.to_thread(
            brain.step, req.question, use_search=req.use_search, temperature=req.temperature
        )
        answer = result["answer"]
        thoughts = result.get("thoughts", answer)

        # Стримим краткий ответ
        async def generate():
            full = ""
            for token in answer.split():
                full += token
                yield f"data: {json.dumps({'token': token + ' '})}\n\n"
            # В конце отправляем полные мысли (для отображения в интерфейсе)
            yield f"data: {json.dumps({'done': True, 'full_answer': answer, 'thoughts': thoughts})}\n\n"
            await asyncio.to_thread(brain.save_dialog_history)

        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/learn")
async def learn(req: LearnRequest):
    try:
        await asyncio.to_thread(brain.learn_pair, req.question, req.answer)
        await asyncio.to_thread(brain.save)
        return {"status": "learned"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    return {"status": "updated", "topics": agent.topics}


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


# ---------- ЗАВЕРШЕНИЕ ----------
def save_brain():
    print("\n💾 Сохраняем модель...")
    agent.stop()
    brain.save()
    brain.save_dialog_history()


atexit.register(save_brain)
signal.signal(signal.SIGINT, lambda s, f: (save_brain(), sys.exit(0)))
signal.signal(signal.SIGTERM, lambda s, f: (save_brain(), sys.exit(0)))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)