from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from smart_brain_v4 import Brain
import uvicorn
import time

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

# Модели данных
class AskRequest(BaseModel):
    question: str
    temperature: float = 0.7

class LearnRequest(BaseModel):
    question: str
    answer: str

# --- API ---
@app.post("/ask")
async def ask(req: AskRequest):
    try:
        result = brain.generate_answer(req.question, temperature=req.temperature, use_rag=True)
        return {
            "question": req.question,
            "answer": result["text"],
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
        /* Хедер */
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

        /* Основной контейнер */
        .app-container {
            display: flex;
            flex: 1;
            overflow: hidden;
        }

        /* Боковая панель */
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

        /* Чат */
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

        /* Адаптив */
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
    <!-- Боковая панель -->
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

    <!-- Чат -->
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

    // Температура
    const tempSlider = document.getElementById('tempSlider');
    const tempDisplay = document.getElementById('tempDisplay');
    tempSlider.addEventListener('input', () => {
        tempDisplay.textContent = parseFloat(tempSlider.value).toFixed(1);
    });

    // Переключение панели
    let sidebarOpen = true;
    function toggleSidebar() {
        sidebarOpen = !sidebarOpen;
        document.getElementById('sidebar').classList.toggle('closed', !sidebarOpen);
    }

    // Загрузка статистики
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

    // Отображение фактов в панели
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

    // Отправить сообщение
    async function sendMessage() {
        const input = document.getElementById('questionInput');
        const question = input.value.trim();
        if (!question) return;
        input.value = '';

        // Добавляем сообщение пользователя
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

            // Показываем кнопки обучения
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

    // Обучение на последнем ответе (положительное)
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
            const endpoint = '/learn';
            const body = { question: q, answer: a };
            // Для отрицания мы не передаём отдельный эндпоинт, используем learn_negative_pair
            // Поэтому добавим параметр? Но в API у нас только learn. Модифицируем? 
            // Сделаем два эндпоинта: /learn и /learn_neg. Добавим ниже.
            // Временно: для отрицания вызовем /learn_neg (добавим позже)
            // Для простоты здесь используем отдельный вызов.
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
                // Скрываем кнопки
                document.getElementById('learnPosBtn').style.display = 'none';
                document.getElementById('learnNegBtn').style.display = 'none';
            } else {
                alert('Ошибка: ' + (data.detail || 'неизвестная'));
            }
        } catch(e) { alert('Ошибка: ' + e.message); }
    }

    // Добавим эндпоинт /learn_neg в Python позже

    // Инициализация
    loadStats();
    // Автоматически открыть панель на десктопе
    if (window.innerWidth < 768) toggleSidebar();
</script>
</body>
</html>
"""

# Добавляем эндпоинт для отрицательного обучения
@app.post("/learn_neg")
async def learn_neg(req: LearnRequest):
    try:
        brain.learn_negative_pair(req.question, req.answer)
        brain.save()
        return {"status": "learned_negative", "question": req.question, "answer": req.answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)