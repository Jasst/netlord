from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from smart_brain_v4 import Brain
import uvicorn
import re

app = FastAPI(title="Smart Brain v4")

# Инициализация мозга (используем улучшенную версию, где есть forget_concept)
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

class AskRequest(BaseModel):
    question: str
    temperature: float = 0.7

class LearnRequest(BaseModel):
    question: str
    answer: str

# --- Вспомогательные функции ---

def normalize_text(text: str) -> str:
    """Приводит текст к единому формату для сравнения."""
    return re.sub(r'\s+', ' ', text.strip().lower())

def extract_fact_from_command(text: str, patterns: list) -> str | None:
    """
    Пытается извлечь факт (то, что нужно запомнить/забыть) из текста,
    используя список шаблонов. Возвращает извлечённую строку или None.
    """
    for pattern in patterns:
        match = re.match(pattern, text.strip(), re.IGNORECASE)
        if match:
            fact = match.group(1).strip()
            if fact:
                return fact
    return None

# Расширенные шаблоны для запоминания
REMEMBER_PATTERNS = [
    r'^(?:запомни|запомнить)\s+(.+)$',                     # запомни ...
    r'^(?:можешь|не мог бы ты|не могли бы вы)?\s*запомни(?:ть)?\s+(.+)$',  # можешь запомнить ...
    r'^(?:я хочу|хочу|давай|давайте)\s+запомни(?:ть)?\s+(.+)$',          # я хочу запомнить ...
    r'^запомни(?:ть)?,\s*пожалуйста,\s+(.+)$',            # запомни, пожалуйста, ...
]

# Шаблоны для забывания (аналогично)
FORGET_PATTERNS = [
    r'^(?:забудь|забыть)\s+(.+)$',
    r'^(?:можешь|не мог бы ты)?\s*забудь(?:ть)?\s+(.+)$',
    r'^(?:я хочу|хочу)\s+забыть\s+(.+)$',
    r'^забудь(?:ть)?,\s*пожалуйста,\s+(.+)$',
]

def handle_remember_command(text: str) -> tuple[bool, str]:
    """
    Обрабатывает команду "запомни ..." (в разных формулировках).
    Возвращает (была_обработана, ответ_бота)
    """
    fact = extract_fact_from_command(text, REMEMBER_PATTERNS)
    if not fact:
        return False, ""

    # Сохраняем в память (в knowledge_base и нейросеть)
    brain.learn_pair(text, fact)
    brain.save()

    # Сохраняем диалог в историю, чтобы следующий вопрос знал о запоминании
    response_text = f"✅ Я запомнил: «{fact}»."
    brain.dialog_memory.add_turn(text, response_text, brain.text_to_embedding(text))

    print(f"[CMD] Запомнил: '{fact}'")
    return True, response_text

def handle_forget_command(text: str) -> tuple[bool, str]:
    """
    Обрабатывает команду "забудь ..." (в разных формулировках).
    Удаляет все понятия, которые содержат искомую фразу (частичное совпадение).
    Возвращает (была_обработана, ответ_бота)
    """
    phrase = extract_fact_from_command(text, FORGET_PATTERNS)
    if not phrase:
        return False, ""

    norm_phrase = normalize_text(phrase)
    removed_any = False

    # 1. Ищем в concept_index по частичному совпадению
    to_delete = []
    for key, nid in brain.concept_index.items():
        if norm_phrase in key or key in norm_phrase:
            to_delete.append((key, nid))

    # Удаляем найденные нейроны и записи из concept_index
    for key, nid in to_delete:
        if nid in brain.neurons:
            neuron = brain.neurons[nid]
            for syn_id in list(neuron.incoming_synapses) + list(neuron.outgoing_synapses):
                brain._remove_synapse(syn_id)
            del brain.neurons[nid]
        del brain.concept_index[key]
        removed_any = True

    # 2. Удаляем из knowledge_base записи, где вопрос или ответ содержат norm_phrase
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

# --- API ---

@app.post("/ask")
async def ask(req: AskRequest):
    try:
        # Проверяем команды (сначала запомнить, потом забыть)
        handled, response = handle_remember_command(req.question)
        if handled:
            return {
                "question": req.question,
                "answer": response,
                "facts": [],
                "known": True
            }

        handled, response = handle_forget_command(req.question)
        if handled:
            return {
                "question": req.question,
                "answer": response,
                "facts": [],
                "known": True
            }

        # Обычный вопрос
        result = brain.generate_answer(req.question, temperature=req.temperature, use_rag=True)
        answer_text = result["text"]

        # Сохраняем диалог в историю
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

@app.get("/stats")
async def stats():
    return {
        "neurons": len(brain.neurons),
        "synapses": len(brain.synapses),
        "concepts": len(brain.concept_index),
        "knowledge_base": len(brain.knowledge_base),
        "memory_entries": len(brain.long_memory.items) + len(brain.short_memory.items)
    }

# --- Веб-интерфейс ---
HTML_PAGE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Brain v4 - Чат</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #eef2f7;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .header {
            background: #2c3e50;
            color: white;
            padding: 15px 25px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            flex-shrink: 0;
            z-index: 10;
        }
        .header h1 {
            font-size: 22px;
            font-weight: 400;
            letter-spacing: 1px;
        }
        .header .controls {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .header .controls label {
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .header .controls input[type="range"] {
            width: 80px;
            cursor: pointer;
        }
        .header .controls span {
            font-weight: bold;
            min-width: 30px;
            text-align: center;
        }
        .menu-btn {
            background: none;
            border: none;
            color: white;
            font-size: 28px;
            cursor: pointer;
            padding: 0 10px;
            transition: 0.2s;
        }
        .menu-btn:hover { opacity: 0.7; }

        .app-container {
            display: flex;
            flex: 1;
            overflow: hidden;
        }

        .sidebar {
            width: 320px;
            background: #ffffff;
            border-right: 1px solid #d0d7de;
            padding: 20px;
            overflow-y: auto;
            transition: transform 0.3s ease, width 0.3s;
            flex-shrink: 0;
            box-shadow: 2px 0 8px rgba(0,0,0,0.05);
        }
        .sidebar.closed {
            transform: translateX(-100%);
            width: 0;
            padding: 0;
            border: none;
            overflow: hidden;
        }
        .sidebar h3 {
            font-size: 16px;
            color: #2c3e50;
            margin-bottom: 12px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 8px;
        }
        .sidebar .stat-item {
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            border-bottom: 1px solid #f0f0f0;
            font-size: 14px;
        }
        .sidebar .stat-item .label { color: #555; }
        .sidebar .stat-item .value { font-weight: 600; color: #2c3e50; }

        .sidebar .facts-section {
            margin-top: 20px;
        }
        .sidebar .fact-item {
            background: #f8f9fa;
            border-left: 3px solid #3498db;
            padding: 8px 12px;
            margin-bottom: 8px;
            border-radius: 4px;
            font-size: 13px;
            line-height: 1.4;
        }
        .sidebar .fact-item .q { font-weight: 600; color: #2c3e50; }
        .sidebar .fact-item .a { color: #555; }
        .sidebar .fact-item .score { font-size: 12px; color: #888; float: right; }

        .chat {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #f8fafc;
            padding: 20px 30px;
            overflow: hidden;
        }
        .messages {
            flex: 1;
            overflow-y: auto;
            padding-right: 10px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            margin-bottom: 15px;
        }
        .message {
            max-width: 80%;
            padding: 12px 18px;
            border-radius: 18px;
            line-height: 1.5;
            word-wrap: break-word;
            animation: fadeIn 0.3s ease;
        }
        .message.user {
            align-self: flex-end;
            background: #3498db;
            color: white;
            border-bottom-right-radius: 6px;
        }
        .message.bot {
            align-self: flex-start;
            background: #ffffff;
            border: 1px solid #d0d7de;
            border-bottom-left-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .message .time {
            font-size: 11px;
            opacity: 0.7;
            margin-top: 4px;
            text-align: right;
        }
        .message .facts-indicator {
            font-size: 12px;
            background: #ecf0f1;
            padding: 2px 8px;
            border-radius: 12px;
            display: inline-block;
            margin-top: 6px;
            color: #2c3e50;
        }

        .input-area {
            display: flex;
            gap: 10px;
            background: white;
            padding: 12px 18px;
            border-radius: 30px;
            box-shadow: 0 1px 6px rgba(0,0,0,0.08);
            border: 1px solid #d0d7de;
        }
        .input-area input {
            flex: 1;
            border: none;
            padding: 10px 0;
            font-size: 15px;
            outline: none;
            background: transparent;
        }
        .input-area button {
            background: #3498db;
            color: white;
            border: none;
            border-radius: 30px;
            padding: 8px 20px;
            font-size: 15px;
            cursor: pointer;
            transition: 0.2s;
            white-space: nowrap;
        }
        .input-area button:hover { background: #2980b9; }
        .input-area button:disabled { opacity: 0.5; cursor: not-allowed; }

        .learn-btn-container {
            display: flex;
            justify-content: flex-end;
            margin-top: 6px;
            gap: 8px;
        }
        .learn-btn-container button {
            background: #27ae60;
            color: white;
            border: none;
            border-radius: 20px;
            padding: 4px 14px;
            font-size: 12px;
            cursor: pointer;
            transition: 0.2s;
        }
        .learn-btn-container button:hover { background: #1e8449; }
        .learn-btn-container button.neg {
            background: #e74c3c;
        }
        .learn-btn-container button.neg:hover { background: #c0392b; }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 768px) {
            .sidebar { width: 260px; }
            .chat { padding: 10px; }
            .message { max-width: 90%; }
            .header h1 { font-size: 18px; }
            .header .controls label { font-size: 12px; }
        }
    </style>
</head>
<body>

<div class="header">
    <div style="display:flex; align-items:center; gap:10px;">
        <button class="menu-btn" onclick="toggleSidebar()">☰</button>
        <h1>🧠 Smart Brain v4</h1>
    </div>
    <div class="controls">
        <label>Температура <input type="range" id="tempSlider" min="0.1" max="1.5" step="0.1" value="0.7"> <span id="tempDisplay">0.7</span></label>
    </div>
</div>

<div class="app-container">
    <div class="sidebar" id="sidebar">
        <h3>📊 Статистика</h3>
        <div id="statsContainer">
            <div class="stat-item"><span class="label">Нейроны</span><span class="value" id="statNeurons">-</span></div>
            <div class="stat-item"><span class="label">Синапсы</span><span class="value" id="statSynapses">-</span></div>
            <div class="stat-item"><span class="label">Понятия</span><span class="value" id="statConcepts">-</span></div>
            <div class="stat-item"><span class="label">Факты в KB</span><span class="value" id="statKB">-</span></div>
            <div class="stat-item"><span class="label">Записей памяти</span><span class="value" id="statMemory">-</span></div>
        </div>

        <div class="facts-section" id="factsSection">
            <h3>📚 Факты из памяти</h3>
            <div id="factsList"><p style="color:#999; font-size:14px;">Факты появятся здесь после ответа.</p></div>
        </div>
    </div>

    <div class="chat">
        <div class="messages" id="messages">
            <div class="message bot">Привет! Я Smart Brain. Задай мне вопрос.</div>
        </div>

        <div class="input-area">
            <input type="text" id="questionInput" placeholder="Введите вопрос..." onkeydown="if(event.key==='Enter') sendMessage()">
            <button id="sendBtn" onclick="sendMessage()">Отправить</button>
        </div>
        <div class="learn-btn-container">
            <button onclick="learnLastPair()" id="learnPosBtn" style="display:none;">✅ Обучить ответу</button>
            <button onclick="learnLastPairNeg()" id="learnNegBtn" style="display:none;">❌ Обучить отрицанию</button>
        </div>
    </div>
</div>

<script>
    let lastQuestion = '';
    let lastAnswer = '';
    let lastFacts = [];

    const tempSlider = document.getElementById('tempSlider');
    const tempDisplay = document.getElementById('tempDisplay');
    tempSlider.addEventListener('input', () => {
        tempDisplay.textContent = parseFloat(tempSlider.value).toFixed(1);
    });

    let sidebarOpen = true;
    function toggleSidebar() {
        sidebarOpen = !sidebarOpen;
        document.getElementById('sidebar').classList.toggle('closed', !sidebarOpen);
    }

    async function loadStats() {
        try {
            const res = await fetch('/stats');
            const data = await res.json();
            document.getElementById('statNeurons').textContent = data.neurons;
            document.getElementById('statSynapses').textContent = data.synapses;
            document.getElementById('statConcepts').textContent = data.concepts;
            document.getElementById('statKB').textContent = data.knowledge_base;
            document.getElementById('statMemory').textContent = data.memory_entries || 0;
        } catch(e) { console.error('Stats error', e); }
    }

    function renderFacts(facts) {
        const container = document.getElementById('factsList');
        if (!facts || facts.length === 0) {
            container.innerHTML = '<p style="color:#999; font-size:14px;">Нет фактов.</p>';
            return;
        }
        let html = '';
        facts.forEach(f => {
            html += `<div class="fact-item"><div class="q">${f.q}</div><div class="a">${f.a}</div><div class="score">score: ${f.score?.toFixed(2) || '?'}</div></div>`;
        });
        container.innerHTML = html;
    }

    async function sendMessage() {
        const input = document.getElementById('questionInput');
        const question = input.value.trim();
        if (!question) return;
        input.value = '';

        addMessage(question, 'user');
        const sendBtn = document.getElementById('sendBtn');
        sendBtn.disabled = true;

        try {
            const temp = parseFloat(tempSlider.value);
            const res = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, temperature: temp })
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Ошибка');

            const answer = data.answer || 'Нет ответа.';
            lastQuestion = question;
            lastAnswer = answer;
            lastFacts = data.facts || [];

            addMessage(answer, 'bot', lastFacts);
            renderFacts(lastFacts);
            loadStats();

            document.getElementById('learnPosBtn').style.display = 'inline-block';
            document.getElementById('learnNegBtn').style.display = 'inline-block';
        } catch(e) {
            addMessage('❌ Ошибка: ' + e.message, 'bot');
        } finally {
            sendBtn.disabled = false;
        }
    }

    function addMessage(text, sender, facts = null) {
        const container = document.getElementById('messages');
        const div = document.createElement('div');
        div.className = 'message ' + sender;
        let html = text;
        if (sender === 'bot' && facts && facts.length > 0) {
            html += `<div class="facts-indicator">📚 использовано фактов: ${facts.length}</div>`;
        }
        html += `<div class="time">${new Date().toLocaleTimeString()}</div>`;
        div.innerHTML = html;
        container.appendChild(div);
        container.scrollTop = container.scrollHeight;
    }

    async function learnLastPair() {
        if (!lastQuestion || !lastAnswer) return alert('Нет пары для обучения.');
        await doLearn(lastQuestion, lastAnswer, 'positive');
    }

    async function learnLastPairNeg() {
        if (!lastQuestion || !lastAnswer) return alert('Нет пары для обучения.');
        await doLearn(lastQuestion, lastAnswer, 'negative');
    }

    async function doLearn(q, a, type) {
        try {
            let url = '/learn';
            if (type === 'negative') url = '/learn_neg';
            const res = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: q, answer: a })
            });
            const data = await res.json();
            if (res.ok) {
                alert(`✅ ${type === 'positive' ? 'Обучено' : 'Отрицание обучено'}: ${q} → ${a}`);
                loadStats();
                document.getElementById('learnPosBtn').style.display = 'none';
                document.getElementById('learnNegBtn').style.display = 'none';
            } else {
                alert('Ошибка: ' + (data.detail || 'неизвестная'));
            }
        } catch(e) { alert('Ошибка: ' + e.message); }
    }

    loadStats();
    if (window.innerWidth < 768) toggleSidebar();
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)