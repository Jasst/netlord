from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from smart_brain_v4 import Brain, cosine_similarity
import uvicorn
import re
import random
import time
from openai import OpenAI
import os

app = FastAPI(title="Smart Brain v4")

# Инициализация мозга
brain = Brain(
    dim_embedding=128,
    input_neurons=40,
    output_neurons=40,
    hidden_layers=[100, 80, 60],
    model_path="brain_model_trained.json",
    max_neurons=800,
    max_synapses=8000
)
brain.load()

# Настройка LLM клиента (для генерации пар)
llm_client = OpenAI(
    base_url="http://192.168.0.13:1234/v1",
    api_key="not-needed",
)

class AskRequest(BaseModel):
    question: str
    temperature: float = 0.7

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

# --- Вспомогательные функции для команд "запомни/забудь" ---

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
    brain.learn_pair(text, fact)
    brain.save()
    response_text = f"✅ Я запомнил: «{fact}»."
    brain.dialog_memory.add_turn(text, response_text, brain.text_to_embedding(text))
    print(f"[CMD] Запомнил: '{fact}'")
    return True, response_text

def handle_forget_command(text: str) -> tuple[bool, str]:
    phrase = extract_fact_from_command(text, FORGET_PATTERNS)
    if not phrase:
        return False, ""

    norm_phrase = normalize_text(phrase)
    removed_any = False

    to_delete = []
    for key, nid in brain.concept_index.items():
        if norm_phrase in key or key in norm_phrase:
            to_delete.append((key, nid))

    for key, nid in to_delete:
        if nid in brain.neurons:
            neuron = brain.neurons[nid]
            for syn_id in list(neuron.incoming_synapses) + list(neuron.outgoing_synapses):
                brain._remove_synapse(syn_id)
            del brain.neurons[nid]
        del brain.concept_index[key]
        removed_any = True

    new_kb = []
    for item in brain.knowledge_base:
        if norm_phrase in normalize_text(item["q"]) or norm_phrase in normalize_text(item["a"]):
            removed_any = True
            continue
        new_kb.append(item)
    brain.knowledge_base = new_kb

    if removed_any:
        response_text = f"✅ Я забыл всё, что связано с «{phrase}»."
    else:
        response_text = f"❌ Я не нашёл ничего, что можно забыть по запросу «{phrase}»."

    brain.dialog_memory.add_turn(text, response_text, brain.text_to_embedding(text))
    brain.save()
    print(f"[CMD] Забыл: '{phrase}' -> {response_text}")
    return True, response_text

# --- Функции автообучения ---

def generate_training_pairs(topic: str, num_pairs: int = 50, temperature: float = 0.7) -> list:
    system_prompt = (
        "Ты — генератор обучающих данных для нейросети. "
        "Твоя задача — создать список пар 'вопрос|ответ' на русском языке, "
        "которые помогут нейросети понять тему и её атрибуты, контекстные связи и ассоциации. "
        "Формат вывода: каждая пара на новой строке, разделена символом '|'. "
        "Не добавляй никаких пояснений, только список пар."
    )
    user_prompt = (
        f"Сгенерируй {num_pairs} пар (вопрос|ответ) по теме '{topic}'. "
        "Вопросы должны быть разнообразными: от прямых ('что такое ...') до контекстных ('как ...', 'почему ...', 'в каком случае ...'). "
        "Ответы должны быть краткими, но информативными (1-5 слов). "
        "Включай синонимы и связанные понятия, чтобы модель научилась ассоциировать разные формулировки с одним понятием. "
        "Примеры: 'солнце|звезда', 'солнечный день|ясно и жарко', 'что дает солнце|свет и тепло'."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = llm_client.chat.completions.create(
            model="local-model",
            messages=messages,
            max_tokens=num_pairs * 20 + 100,
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
                    if q and a:
                        pairs.append((q, a))
        return pairs
    except Exception as e:
        print(f"Ошибка при генерации: {e}")
        return []

def integrate_new_concept(brain: Brain, concept_text: str, top_k: int = 5):
    nid = brain.concept_index.get(concept_text.strip().lower())
    if nid is None:
        return
    emb = brain.neurons[nid].embedding
    similarities = []
    for other_nid, neuron in brain.neurons.items():
        if other_nid == nid:
            continue
        if neuron.cluster == 'output' and neuron.label:
            sim = cosine_similarity(emb, neuron.embedding)
            if sim > 0.3:
                similarities.append((sim, other_nid))
    similarities.sort(reverse=True)
    for sim, other_nid in similarities[:top_k]:
        brain._create_synapse(nid, other_nid, weight=0.1 + 0.2 * sim)
        brain._create_synapse(other_nid, nid, weight=0.1 + 0.2 * sim)

def train_model_on_topic(brain: Brain, topic: str, num_pairs: int = 50,
                         negative_ratio: float = 0.2, temperature: float = 0.7,
                         integrate: bool = True, epochs: int = 1) -> dict:
    result = {"status": "ok", "pairs": 0, "negatives": 0, "message": ""}
    try:
        pairs = generate_training_pairs(topic, num_pairs, temperature)
        if not pairs:
            result["status"] = "error"
            result["message"] = "Не удалось сгенерировать пары."
            return result

        for epoch in range(epochs):
            for idx, (q, a) in enumerate(pairs, 1):
                brain.learn_pair(q, a)
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
                brain.learn_negative_pair(q, w)
                time.sleep(0.05)
            result["negatives"] = len(neg_pairs)

        brain.sleep(duration_steps=5)
        brain.save()
        result["message"] = f"Обучение на тему '{topic}' завершено за {epochs} эпох(и)."
    except Exception as e:
        result["status"] = "error"
        result["message"] = str(e)
    return result

# --- API эндпоинты ---

@app.post("/ask")
async def ask(req: AskRequest):
    try:
        handled, response = handle_remember_command(req.question)
        if handled:
            return {"question": req.question, "answer": response, "facts": [], "known": True}

        handled, response = handle_forget_command(req.question)
        if handled:
            return {"question": req.question, "answer": response, "facts": [], "known": True}

        result = brain.generate_answer(req.question, temperature=req.temperature, use_rag=True)
        answer_text = result["text"]
        brain.dialog_memory.add_turn(req.question, answer_text, brain.text_to_embedding(req.question))

        return {
            "question": req.question,
            "answer": answer_text,
            "facts": result.get("facts", []),
            "known": result.get("known", False)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/learn")
async def learn(req: LearnRequest):
    try:
        brain.learn_pair(req.question, req.answer)
        brain.save()
        return {"status": "learned", "question": req.question, "answer": req.answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/learn_neg")
async def learn_neg(req: LearnRequest):
    try:
        brain.learn_negative_pair(req.question, req.answer)
        brain.save()
        return {"status": "learned_negative", "question": req.question, "answer": req.answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_topic")
async def train_topic(req: TrainTopicRequest):
    try:
        result = train_model_on_topic(
            brain, req.topic, req.num_pairs, req.negative_ratio,
            req.temperature, req.integrate, req.epochs
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_pair")
async def train_pair(req: TrainPairRequest):
    try:
        for epoch in range(req.epochs):
            brain.learn_pair(req.question, req.answer)
            time.sleep(0.1)
        brain.save()
        return {
            "status": "learned",
            "question": req.question,
            "answer": req.answer,
            "epochs": req.epochs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def stats():
    return {
        "neurons": len(brain.neurons),
        "synapses": len(brain.synapses),
        "concepts": len(brain.concept_index),
        "knowledge_base": len(brain.knowledge_base),
        "memory_entries": len(brain.long_memory.items) + len(brain.short_memory.items)
    }

# --- Главная страница ---
@app.get("/", response_class=HTMLResponse)
async def index():
    # Читаем HTML из файла
    html_path = os.path.join("templates", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)