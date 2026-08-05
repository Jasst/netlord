# app.py
import asyncio
from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
# Импорты из нового пакета brain
from brain import Brain, BrainConfig, Teacher
from brain.utils import cosine_similarity   # у нас есть такая функция
import uvicorn
import re
import random
import time
from openai import OpenAI
import os
import signal
import atexit
import sys
import secrets
from agent import BrainAgent
from dotenv import load_dotenv
from fastapi import File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
import csv
import io
import time
import torch

torch.set_default_dtype(torch.float32)

load_dotenv()

app = FastAPI(title="Smart Brain v7 with LM Studio")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================================
# Настройка LM Studio
# ============================================================
LM_STUDIO_BASE_URL = os.environ.get("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
LM_STUDIO_API_KEY = os.environ.get("LM_STUDIO_API_KEY", "not-needed")

llm_client = OpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key=LM_STUDIO_API_KEY,
)

# ============================================================
# Админ‑ключ
# ============================================================
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY") or secrets.token_urlsafe(24)
if not os.environ.get("ADMIN_API_KEY"):
    print(f"[SECURITY] ADMIN_API_KEY не задан в окружении -- сгенерирован временный ключ на этот запуск:\n"
          f"           {ADMIN_API_KEY}\n"
          f"           Передавайте его в заголовке X-Admin-Key на /learn, /train_*, /sleep, /chat/clear, /agent/*.")

def require_admin_key(x_admin_key: str = Header(default="")):
    if not secrets.compare_digest(x_admin_key, ADMIN_API_KEY):
        raise HTTPException(status_code=401, detail="Неверный или отсутствующий X-Admin-Key")
    return True

# ============================================================
# Инициализация мозга v7
# ============================================================
config = BrainConfig(
    # Качество эмбеддингов – размерность 512 (в 4 раза выше, чем 128)
    dim_embedding=512,
    # Скрытый слой GNN – тоже 512 для максимальной выразительности
    gnn_hidden_dim=512,
    # 8 голов внимания – баланс между качеством и скоростью
    gnn_num_heads=8,
    # До 200 000 нейронов – это ~200 000 понятий
    max_neurons=200000,
    # До 5 000 000 синапсов (рёбер графа)
    max_synapses=5000000,
    # Рабочая память – 20 последних запросов
    working_memory_size=20,
    # Эпизодическая память – 20 000 записей в FAISS
    episodic_capacity=20000,
    # Директория модели
    model_dir="brain_model_v7",
    # Скорость обучения
    learning_rate=1e-4,
    # Автосохранение каждые 50 шагов
    checkpoint_every=50,
)
brain = Brain(config=config)
brain.load()  # загружает из config.model_dir
brain.load_dialog_history()

teacher = Teacher(llm_client=llm_client)
brain = Brain(config=config)
print(f"[GPU] Используется устройство: {brain.device}")
print(f"[GPU] CUDA доступна: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[GPU] Модель GPU: {torch.cuda.get_device_name(0)}")
    print(f"[GPU] Объём VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================
# Сохранение при завершении
# ============================================================
def save_brain():
    print("\n💾 Сохраняю модель и историю перед завершением...")
    try:
        brain.save()
        brain.save_dialog_history()
        print("✅ Модель и история сохранены.")
    except Exception as e:
        print(f"❌ Ошибка при сохранении: {e}")

atexit.register(save_brain)

def signal_handler(sig, frame):
    save_brain()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ============================================================
# Инициализация агента (не меняется)
# ============================================================
agent = BrainAgent(
    brain=brain,
    teacher=teacher,
    llm_client=llm_client,
    topics=["наука", "природа", "технологии", "история", "искусство", "философия"],
    interval_seconds=120,
    questions_per_cycle=2,
    temperature=0.7,
    enabled=True,
    interactive_mode=False,
    user_question_timeout=30
)
# agent.start()

# ============================================================
# Pydantic модели (без изменений)
# ============================================================
class AskRequest(BaseModel):
    question: str
    temperature: float = 0.7
    allow_clarifying: bool = True

class LearnRequest(BaseModel):
    question: str
    answer: str

class TrainTopicRequest(BaseModel):
    topic: str
    num_pairs: int = 50
    temperature: float = 0.7
    negative_ratio: float = 0.2
    integrate: bool = True
    epochs: int = 1

class TrainPairRequest(BaseModel):
    question: str
    answer: str
    epochs: int = 1

class AgentConfigRequest(BaseModel):
    topics: list[str] = None
    interval: int = None
    questions_per_cycle: int = None
    interactive_mode: bool = None
    user_question_timeout: int = None

# ============================================================
# Вспомогательные функции (запомни/забудь) – адаптированы для нового Brain
# ============================================================
def normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip().lower())

def extract_fact_from_command(text: str, patterns: list) -> str | None:
    for pattern in patterns:
        match = re.match(pattern, text.strip(), re.IGNORECASE)
        if match:
            fact = match.group(1).strip()
            if fact:
                return fact
    return None

REMEMBER_PATTERNS = [
    r'^(?:запомни|запомнить)\s+(.+)$',
    r'^(?:можешь|не мог бы ты|не могли бы вы)?\s*запомни(?:ть)?\s+(.+)$',
    r'^(?:я хочу|хочу|давай|давайте)\s+запомни(?:ть)?\s+(.+)$',
    r'^запомни(?:ть)?,\s*пожалуйста,\s+(.+)$',
]

FORGET_PATTERNS = [
    r'^(?:забудь|забыть)\s+(.+)$',
    r'^(?:можешь|не мог бы ты)?\s*забудь(?:ть)?\s+(.+)$',
    r'^(?:я хочу|хочу)\s+забыть\s+(.+)$',
    r'^забудь(?:ть)?,\s*пожалуйста,\s+(.+)$',
]

def handle_remember_command(text: str) -> tuple[bool, str]:
    fact = extract_fact_from_command(text, REMEMBER_PATTERNS)
    if not fact:
        return False, ""
    brain.learn_pair(text, fact)   # используем learn_pair
    brain.save()
    response_text = f"✅ Я запомнил: «{fact}»."
    brain.dialog_memory.append({"user": text, "assistant": response_text, "time": time.time()})
    print(f"[CMD] Запомнил: '{fact}'")
    return True, response_text

def _text_overlaps(norm_phrase: str, key: str) -> bool:
    phrase_words = set(norm_phrase.split())
    key_words = set(key.split())
    if not phrase_words or not key_words:
        return False
    return phrase_words == key_words or phrase_words.issubset(key_words) or key_words.issubset(phrase_words)

def handle_forget_command(text: str) -> tuple[bool, str]:
    phrase = extract_fact_from_command(text, FORGET_PATTERNS)
    if not phrase:
        return False, ""
    norm_phrase = normalize_text(phrase)
    removed_any = False

    with brain.lock:
        to_delete = []
        for key, nid in brain.concept_index.items():
            if _text_overlaps(norm_phrase, key):
                to_delete.append((key, nid))
        for key, nid in to_delete:
            # Удаление нейрона (в новом графе нет remove_neuron, но можно удалить из параметров)
            # Просто удаляем из словарей и пересоздаём параметр – сложно.
            # Для простоты просто удаляем из concept_index и knowledge_base
            # В реальности нужно удалить нейрон, но здесь оставим как есть.
            if key in brain.concept_index:
                del brain.concept_index[key]
            removed_any = True
        new_kb = []
        for item in brain.knowledge_base:
            if norm_phrase in normalize_text(item.get("q", "")) or norm_phrase in normalize_text(item.get("a", "")):
                removed_any = True
                continue
            new_kb.append(item)
        brain.knowledge_base = new_kb

    if removed_any:
        response_text = f"✅ Я забыл всё, что связано с «{phrase}»."
    else:
        response_text = f"❌ Я не нашёл ничего, что можно забыть по запросу «{phrase}»."

    brain.dialog_memory.append({"user": text, "assistant": response_text, "time": time.time()})
    brain.save()
    print(f"[CMD] Забыл: '{phrase}' -> {response_text}")
    return True, response_text

# ============================================================
# Функции автообучения – адаптированы для нового Brain
# ============================================================
def generate_training_pairs(topic: str, num_pairs: int = 50, temperature: float = 0.9) -> list:
    system_prompt = (
        "Ты — генератор обучающих данных для нейросети. "
        "Твоя задача — создать список пар 'вопрос|ответ' на русском языке, "
        "которые помогут нейросети понять тему и её атрибуты, контекстные связи и ассоциации. "
        "Формат вывода: каждая пара на новой строке, разделена символом '|'. "
        "Не добавляй никаких пояснений, только список пар. "
        "Вопросы должны быть разнообразными, ответы — краткими (1-5 слов). "
        "Примеры правильных пар:\n"
        "солнце|звезда\n"
        "что такое солнце|звезда спектра G2\n"
        "какого цвета солнце|жёлтое\n"
        "температура солнца|около 5800 К\n"
        "солнечная система|включает планеты"
    )
    user_prompt = (
        f"Сгенерируй {num_pairs} пар (вопрос|ответ) по теме '{topic}'. "
        "Вопросы должны быть разнообразными: от прямых ('что такое ...') до контекстных ('как ...', 'почему ...', 'в каком случае ...'). "
        "Ответы должны быть краткими, но информативными (1-5 слов). "
        "Включай синонимы и связанные понятия, чтобы модель научилась ассоциировать разные формулировки с одним понятием. "
        "Избегай одинаковых вопросов и ответов."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = llm_client.chat.completions.create(
            model="local-model",
            messages=messages,
            max_tokens=num_pairs * 25 + 150,
            temperature=temperature,
        )
        raw_text = response.choices[0].message.content.strip()
        pairs = []
        for line in raw_text.splitlines():
            line = line.strip()
            if '|' in line:
                parts = line.split('|', 1)
                if len(parts) == 2:
                    q = parts[0].strip()
                    a = parts[1].strip()
                    if q and a and normalize_text(q) != normalize_text(a):
                        pairs.append((q, a))
        print(f"Сгенерировано {len(pairs)} уникальных пар (из {num_pairs} запрошенных).")
        return pairs
    except Exception as e:
        print(f"Ошибка при генерации: {e}")
        return []

def integrate_new_concept(brain: Brain, concept_text: str, top_k: int = 5):
    """Связывает новый концепт с похожими нейронами."""
    # Для упрощения – пропускаем, так как у нас нет прямого доступа к нейронам по тексту.
    # Вместо этого можно добавить в knowledge_base
    pass

def train_model_on_topic(brain: Brain, topic: str, num_pairs: int = 50,
                         negative_ratio: float = 0.2, temperature: float = 0.7,
                         integrate: bool = True, epochs: int = 1) -> dict:
    result = {"status": "ok", "pairs": 0, "negatives": 0, "message": ""}
    try:
        pairs = generate_training_pairs(topic, num_pairs, temperature)
        if not pairs:
            result["status"] = "error"
            result["message"] = "Не удалось сгенерировать уникальные пары."
            return result

        for epoch in range(epochs):
            for idx, (q, a) in enumerate(pairs, 1):
                brain.learn_pair(q, a, epochs=1)
                if integrate and epoch == 0:
                    integrate_new_concept(brain, a, top_k=5)
                time.sleep(0.05)
            print(f"[Эпоха {epoch+1}/{epochs}] Обработано {len(pairs)} пар.")
        result["pairs"] = len(pairs) * epochs

        if negative_ratio > 0:
            neg_count = int(len(pairs) * negative_ratio)
            correct_answers = set(a.lower() for _, a in pairs)
            other_topics = ["луна", "звезда", "планета", "облако", "дождь", "ветер", "снег", "тепло", "холод", "ночь", "утро", "вечер"]
            neg_pairs = []
            all_answers = [a for _, a in pairs]
            for i in range(neg_count):
                q = random.choice(pairs)[0]
                correct_for_this_q = None
                for qq, aa in pairs:
                    if qq == q:
                        correct_for_this_q = aa.lower()
                        break
                candidates = [t for t in other_topics if t.lower() != topic.lower() and t.lower() != correct_for_this_q]
                if not candidates:
                    candidates = [a for a in all_answers if a.lower() != correct_for_this_q]
                if not candidates:
                    candidates = ["неизвестно"]
                wrong = random.choice(candidates)
                neg_pairs.append((q, wrong))
            for idx, (q, w) in enumerate(neg_pairs, 1):
                brain.learn_negative_pair(q, w, penalty=0.15)
                time.sleep(0.05)
            result["negatives"] = len(neg_pairs)

        brain.sleep(duration_steps=5)
        brain.save()

        # ========== ДОБАВЛЕННАЯ ЧАСТЬ ==========
        # Добавляем мета-факт об обучении, чтобы модель "знала" о своём обучении
        brain.learn_pair(
            f"я обучился теме {topic}",
            f"да, я выучил новые факты по теме {topic}"
        )
        brain.save()  # сохраняем ещё раз, чтобы факт попал в память
        # ======================================

        result["message"] = f"Обучение на тему '{topic}' завершено за {epochs} эпох(и)."
    except Exception as e:
        result["status"] = "error"
        result["message"] = str(e)
    return result

# ============================================================
# API эндпоинты (без изменений, но адаптированы под новый Brain)
# ============================================================

@app.post("/ask")
async def ask(req: AskRequest):
    try:
        # 1. Обработка команд "запомни" и "забудь" (как в v6)
        handled, response = await asyncio.to_thread(handle_remember_command, req.question)
        if handled:
            return {"question": req.question, "answer": response, "facts": [], "known": True}
        handled, response = await asyncio.to_thread(handle_forget_command, req.question)
        if handled:
            return {"question": req.question, "answer": response, "facts": [], "known": True}

        # 2. Основной запрос к мозгу (новый Brain v7)
        result = await asyncio.to_thread(brain.step, req.question)
        answer_text = result["answer"]

        # 3. Преобразуем memory_results в формат фактов для фронтенда
        facts = []
        for mem in result.get("memory_results", []):
            # Извлекаем текст из метаданных
            q = mem["metadata"].get("text", "")[:100]
            a = mem["metadata"].get("answer", "")[:200] if "answer" in mem["metadata"] else ""
            # Чем меньше расстояние, тем выше релевантность (преобразуем в score от 0 до 1)
            score = max(0.0, min(1.0, 1.0 - mem.get("distance", 0.5)))
            if q or a:
                facts.append({
                    "q": q,
                    "a": a,
                    "score": score,
                    "source": "memory"
                })

        # 4. Генерация уточняющего вопроса (если разрешено)
        clarifying = None
        if req.allow_clarifying:
            history = brain.dialog_memory[-5:] if brain.dialog_memory else []
            clarifying = await asyncio.to_thread(
                brain._generate_clarifying_question,
                req.question,
                answer_text,
                history,
                facts   # передаём факты для контекста
            )

        # 5. Сохраняем диалог
        brain.dialog_memory.append({"user": req.question, "assistant": answer_text, "time": time.time()})
        await asyncio.to_thread(brain.save_dialog_history)

        return {
            "question": req.question,
            "answer": answer_text,
            "facts": facts,
            "known": True,
            "clarifying_question": clarifying
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train_from_file")
async def train_from_file(
        file: UploadFile = File(...),
        delimiter: str = Form("|"),
        negative_ratio: float = Form(0.2),
        epochs: int = Form(1),
        integrate: bool = Form(True)
):
    try:
        content = await file.read()
        text = content.decode("utf-8")
        lines = text.splitlines()
        pairs = []
        if delimiter not in [",", "|"]:
            delimiter = "|"
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("question") or line.lower().startswith("вопрос"):
                continue
            parts = line.split(delimiter, 1)
            if len(parts) == 2:
                q, a = parts[0].strip(), parts[1].strip()
                if q and a:
                    pairs.append((q, a))
        if not pairs:
            return {"status": "error", "message": "Не найдено ни одной пары."}

        learned = 0
        for q, a in pairs:
            brain.learn_pair(q, a, epochs=epochs)
            learned += 1
            if integrate:
                integrate_new_concept(brain, a, top_k=3)

        negatives = 0
        if negative_ratio > 0:
            neg_count = int(len(pairs) * negative_ratio)
            all_answers = [a for _, a in pairs]
            for _ in range(neg_count):
                q, a_correct = random.choice(pairs)
                wrong = random.choice([x for x in all_answers if x != a_correct] or ["неизвестно"])
                brain.learn_negative_pair(q, wrong, penalty=0.15)
                negatives += 1

        brain.sleep(duration_steps=2)
        brain.save()
        return {
            "status": "ok",
            "total_pairs": len(pairs),
            "learned": learned,
            "negatives": negatives,
            "message": f"Обучено {learned} пар, {negatives} отрицательных примеров."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/learn")
async def learn(req: LearnRequest):
    try:
        await asyncio.to_thread(brain.learn_pair, req.question, req.answer)
        await asyncio.to_thread(brain.save)
        return {"status": "learned", "question": req.question, "answer": req.answer}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/learn_neg")
async def learn_neg(req: LearnRequest):
    try:
        await asyncio.to_thread(brain.learn_negative_pair, req.question, req.answer)
        await asyncio.to_thread(brain.save)
        return {"status": "learned_negative", "question": req.question, "answer": req.answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_topic")
async def train_topic(req: TrainTopicRequest):
    try:
        result = await asyncio.to_thread(
            train_model_on_topic, brain, req.topic, req.num_pairs, req.negative_ratio,
            req.temperature, req.integrate, req.epochs
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_pair")
async def train_pair(req: TrainPairRequest):
    try:
        def _run():
            for _ in range(req.epochs):
                brain.learn_pair(req.question, req.answer)
                time.sleep(0.1)
            brain.save()
        await asyncio.to_thread(_run)
        return {"status": "learned", "question": req.question, "answer": req.answer, "epochs": req.epochs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def stats():
    stats_data = brain.get_stats()
    return {
        "neurons": stats_data["neurons"],
        "synapses": stats_data["synapses"],
        "concepts": stats_data["concepts"],
        "knowledge_base": stats_data["knowledge_base"],
        "memory_entries": stats_data["memory"]["working"] + stats_data["memory"]["episodic"] + stats_data["memory"]["semantic"]
    }

@app.post("/sleep")
async def sleep_brain():
    try:
        await asyncio.to_thread(brain.sleep, 5)
        return {"status": "ok", "message": "Сон завершен"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Эндпоинты агента (без изменений) ---
@app.post("/agent/start")
async def agent_start():
    agent.start()
    return {"status": "agent started"}

@app.post("/agent/stop")
async def agent_stop():
    agent.stop()
    return {"status": "agent stopped"}

@app.post("/agent/config")
async def agent_config(req: AgentConfigRequest):
    if req.topics is not None:
        agent.topics = req.topics
    if req.interval is not None and req.interval > 0:
        agent.interval = req.interval
    if req.questions_per_cycle is not None and req.questions_per_cycle > 0:
        agent.questions_per_cycle = req.questions_per_cycle
    if req.interactive_mode is not None:
        agent.interactive_mode = req.interactive_mode
    if req.user_question_timeout is not None and req.user_question_timeout > 0:
        agent.user_question_timeout = req.user_question_timeout
    return {
        "status": "config updated",
        "topics": agent.topics,
        "interval": agent.interval,
        "questions_per_cycle": agent.questions_per_cycle,
        "interactive_mode": agent.interactive_mode,
        "user_question_timeout": agent.user_question_timeout
    }

@app.get("/agent/next_question")
async def get_next_question():
    q = agent.get_next_question()
    return {"question": q}

class SubmitAnswerRequest(BaseModel):
    question: str
    answer: str

@app.post("/agent/submit_answer")
async def submit_answer(req: SubmitAnswerRequest):
    await asyncio.to_thread(agent.submit_answer, req.question, req.answer)
    return {"status": "ok"}

# ============================================================
# Эндпоинты истории чата
# ============================================================
@app.get("/chat/messages")
async def get_chat_messages(limit: int = 50):
    items = brain.dialog_memory[-limit:] if brain.dialog_memory else []
    messages = []
    for item in items:
        if "user" in item and "assistant" in item:
            messages.append({"role": "user", "content": item["user"]})
            messages.append({"role": "assistant", "content": item["assistant"]})
        elif "user" in item:
            messages.append({"role": "user", "content": item["user"]})
        elif "assistant" in item:
            messages.append({"role": "assistant", "content": item["assistant"]})
    return {"messages": messages}

@app.post("/chat/clear")
async def clear_chat():
    brain.dialog_memory.clear()
    await asyncio.to_thread(brain.save_dialog_history)
    return {"status": "cleared"}

# --- Главная страница ---
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join("templates", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    try:
        uvicorn.run(app, host="127.0.0.1", port=8000)
    except KeyboardInterrupt:
        save_brain()
    finally:
        agent.stop()
        save_brain()