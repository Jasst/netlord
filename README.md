# 🧠 Smart Brain v7

### Differentiable Neural Graph + FAISS Memory + LM Studio

> **A hybrid memory system that learns from conversations in real time.**

Smart Brain v7 is a complete rewrite of the v6 architecture.  
Instead of a static graph with hand‑crafted Hebbian rules, it uses a **differentiable graph neural network (GNN)** that learns via gradient descent.  
Memory is now served by **FAISS** (fast approximate nearest neighbour search) instead of SQLite full scans, making retrieval instant even with thousands of entries.  
The user interface remains the same – but the brain under the hood is now **orders of magnitude faster, more adaptive, and easier to extend.**

---

## ✨ What’s New in v7

| Feature | v6 | v7 |
|---------|----|----|
| Graph type | Static NetworkX + hand‑tuned Hebbian updates | Differentiable GNN (PyTorch Geometric) – trained with `loss.backward()` |
| Memory search | Full SQLite scan (O(N)) | FAISS index – logarithmic time |
| LLM integration | Scattered calls across many methods | Centralised `LMStudioLLM` class |
| Modularity | Monolithic `smart_brain_v6.py` | Clean package `brain/` with separate modules |
| Training | Heuristic weight adjustment | Contrastive loss + gradient descent |
| Sleep | Simple replay and pruning | Consolidation + associative bridge building (still optional) |
| Agent | Relies on v6 methods | Rewritten to use new `Brain.step()` and standardised API |

> **Bottom line:** v7 learns faster, scales better, and is much easier to maintain and extend.

---

## 🧠 Architecture Overview

```
User question (Web UI)
       │
       ▼
FastAPI (app.py)
       │
       ▼
Brain (brain.brain)
   ├── Embedding (sentence‑transformers)
   ├── Graph (GNN) – forward pass with message passing
   ├── FAISS memory retrieval (episodic + working)
   ├── LM Studio LLM (or any OpenAI‑compatible endpoint)
   └── Learning – contrastive loss on graph parameters
       │
       ▼
Answer returned to UI
       │
       ▼
Teacher (optional) evaluates and triggers learning
```

---

## 🚀 Key Features

- **Differentiable Neural Graph** – each neuron has a trainable embedding; synapses are learned via backprop.
- **FAISS‑powered memory** – ultra‑fast retrieval of working and episodic memories.
- **Full‑featured web interface** – chat, settings, training controls, stats, and chat history.
- **Autonomous BrainAgent** – asks questions, learns from answers, works in interactive or background mode.
- **Teacher & Self‑Reflection** – evaluates answer quality and improves the graph accordingly.
- **Sleep mode** – consolidates memory and prunes weak connections.
- **Persistence** – graph weights and memory indexes are saved to disk.
- **LM Studio integration** – use any local LLM with an OpenAI‑compatible API.

---

## 📦 Technology Stack

- **Backend**: Python 3.10+, FastAPI, Uvicorn  
- **Machine Learning**: PyTorch, PyTorch Geometric, sentence‑transformers  
- **Vector Search**: FAISS (CPU)  
- **LLM**: OpenAI SDK (works with LM Studio, Ollama, or any OpenAI‑compatible server)  
- **Frontend**: Vanilla HTML5, CSS3, JavaScript (no framework)  
- **Environment**: python‑dotenv, standard library  

---

## 🛠 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Jasst/netlord.git
cd netlord
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
# or
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

A typical `requirements.txt`:

```txt
fastapi==0.115.0
uvicorn[standard]==0.30.0
torch==2.4.0
torch-geometric==2.5.0
faiss-cpu==1.8.0
sentence-transformers==2.2.2
openai==1.35.0
python-dotenv==1.0.0
```

> **Note:** If you have a CUDA GPU, you may replace `faiss-cpu` with `faiss-gpu` and install PyTorch with CUDA support.

### 4. Configure environment

Create a `.env` file:

```env
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
LM_STUDIO_API_KEY=not-needed
ADMIN_API_KEY=your-secure-key   # optional, auto‑generated if not set
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

Make sure LM Studio (or any other OpenAI‑compatible server) is running and accessible at the configured URL.

### 5. Run the application

```bash
python app.py
```

Open your browser at `http://127.0.0.1:8000` and start chatting.

---

## 💬 Usage

### Web Interface

- **Chat**: type a question in the input area and press Enter.
- **Learn a pair**: after each answer, two buttons appear – ✅ (positive) and ❌ (negative). Use them to reinforce or inhibit the last question‑answer pair.
- **Chat history**: left sidebar lists all chats; you can rename or delete them.
- **Settings** (⚙️): adjust temperature, enable clarifying questions, configure the BrainAgent, and run batch training.
- **Stats & facts**: see live statistics (neurons, synapses, memory entries) and the facts used in the last answer.

### Commands in the Chat

You can also type special commands:

- `learn <question> => <answer>` – explicit positive training
- `neg <question> => <answer>` – negative training (inhibitory synapse)
- `links <concept>` – show outgoing connections of a concept
- `forget <phrase>` – remove neurons and facts related to that phrase
- `stats` – summary statistics
- `sleep` – trigger a consolidation cycle
- `save` – save current state
- `exit` – save and quit

### REST API

The FastAPI backend exposes the same endpoints used by the UI. You can call them programmatically:

- `POST /ask` – send a question, get an answer with facts and clarifying question
- `POST /learn` / `/learn_neg` – learn positive / negative pairs
- `POST /train_topic` – auto‑generate and learn many pairs on a given topic
- `POST /train_pair` – learn a single pair with multiple epochs
- `POST /sleep` – run memory consolidation
- `GET /stats` – get current statistics
- `POST /agent/start` / `/agent/stop` – control the autonomous BrainAgent
- `POST /chat/clear` – clear dialog history

---

## 📁 Project Structure

```
.
├── app.py                  # FastAPI server (main entry point)
├── agent.py                # BrainAgent – autonomous learner & question‑asker
├── brain/                  # Core package (v7)
│   ├── __init__.py
│   ├── brain.py            # Brain class – orchestrator
│   ├── config.py           # Configuration dataclass
│   ├── graph.py            # DifferentiableNeuralGraph (GNN)
│   ├── memory.py           # HierarchicalMemory with FAISS
│   ├── llm.py              # LMStudioLLM wrapper
│   ├── teacher.py          # Teacher – LLM‑based evaluator
│   └── utils.py            # EmbeddingProvider, helpers
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
├── templates/
│   └── index.html
├── brain_model_v7/         # Default persistence folder (auto‑created)
│   ├── graph.pth           # GNN weights
│   ├── meta.json           # metadata (step counter, concept index, etc.)
│   └── dialog_history.json
├── .env                    # Environment variables
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🔬 How It Works – Internally

### Differentiable Neural Graph

- **Neurons** are trainable embeddings (parameters) with optional metadata (`label`, `cluster`, `layer`).
- **Synapses** are edges in the graph; they are used by the GNN layers to propagate messages.
- The graph uses **GATConv** layers (Graph Attention Networks) to update neuron states.
- An update scale parameter controls the magnitude of change, allowing smooth adaptation.

### Hierarchical Memory (FAISS)

- **Working memory** stores the most recent inputs (vectors) in a small deque.
- **Episodic memory** uses a FAISS index with HNSW for fast approximate nearest neighbour search.
- On `consolidate()`, working memory items are moved to episodic memory.
- Retrieval returns the top‑k most similar items with their metadata.

### LM Studio LLM

- All LLM calls go through `LMStudioLLM.generate(prompt, max_tokens, temperature)`.
- This makes it easy to switch to a different provider (Ollama, OpenAI, etc.) by changing the `base_url`.

### Teacher

- Uses the same LLM to evaluate answers on four criteria: relevance, accuracy, completeness, clarity.
- Returns a score (0–1) and an improved version of the answer.
- The score is used to decide whether to reinforce or inhibit the connection.

### BrainAgent

- Runs in a background thread.
- **Autonomous mode**: picks a topic, generates questions, answers them via the brain, evaluates with Teacher, and learns.
- **Interactive mode**: asks the user a question, waits for a reply, and learns from it.
- Periodically saves the model.

### Sleep

- Consolidates memory (moves working → episodic).
- Prunes weak neurons and synapses (by threshold or age).
- (Optional) builds associative bridges between similar concepts.

---

## 🧪 Use Cases

1. **Personal assistant with persistent memory** – remember facts about the user across sessions without retraining the LLM.
2. **Offline knowledge base** – run on a Raspberry Pi without internet using only the graph (fallback mode).
3. **Game NPC with evolving personality** – the NPC remembers player interactions and adapts its behaviour.
4. **RAG extension** – use Smart Brain as a dynamic, self‑learning retrieval system that improves over time.
5. **Research platform** – experiment with different GNN architectures, memory consolidation strategies, and learning rules.

---

## 🔮 Future Directions

- Support for spiking neural networks and STDP
- Multi‑agent shared memory graphs
- Reinforcement learning integration
- Interactive graph visualisation
- Memory compression via community detection
- Distributed / federated learning

---

## 📄 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute it, provided that the copyright notice and permission notice are included in all copies or substantial portions of the Software.

---

## 🌐 Repository

GitHub: [https://github.com/Jasst/netlord](https://github.com/Jasst/netlord)

If you find this project useful, please consider giving it a star ⭐ and sharing your experiments. Issues and pull requests are always welcome!

---

# 🧠 Smart Brain v7 (Русская версия)

### Дифференцируемый нейрограф + FAISS-память + LM Studio

> **Гибридная система памяти, которая учится в реальном времени.**

Smart Brain v7 — это полный пересмотр архитектуры v6.  
Вместо статического графа с ручными эвристиками Хебба здесь используется **дифференцируемая графовая нейросеть (GNN)**, обучаемая через градиентный спуск.  
Память теперь работает на **FAISS** (мгновенный приближённый поиск), а не на полном сканировании SQLite.  
Интерфейс остался прежним, но «мозг» внутри стал на **порядок быстрее, адаптивнее и проще в расширении**.

---

## ✨ Что нового в v7

| Возможность | v6 | v7 |
|-------------|----|----|
| Граф | Статический NetworkX + ручные Хеббовские обновления | Дифференцируемый GNN (PyTorch Geometric) – обучение через `loss.backward()` |
| Поиск в памяти | Полный перебор SQLite (O(N)) | FAISS – логарифмическая сложность |
| Интеграция LLM | Вызовы разбросаны по коду | Централизованный класс `LMStudioLLM` |
| Модульность | Монолитный `smart_brain_v6.py` | Чистый пакет `brain/` с отдельными модулями |
| Обучение | Эвристическая корректировка весов | Контрастная потеря + градиентный спуск |
| Сон | Простой реплей и обрезка | Консолидация + ассоциативные мостики (опционально) |
| Агент | Опирался на методы v6 | Переписан под `Brain.step()` и унифицированный API |

> **Итог:** v7 учится быстрее, масштабируется лучше и намного проще в поддержке и развитии.

---

## 🧠 Архитектура

```
Вопрос пользователя (Web UI)
       │
       ▼
FastAPI (app.py)
       │
       ▼
Brain (brain.brain)
   ├── Эмбеддинг (sentence‑transformers)
   ├── Граф (GNN) – прямой проход с передачей сообщений
   ├── Поиск в FAISS (эпизодическая + рабочая память)
   ├── LM Studio LLM (или любой OpenAI‑совместимый сервер)
   └── Обучение – контрастная потеря на параметрах графа
       │
       ▼
Ответ возвращается в UI
       │
       ▼
Teacher (опционально) оценивает и запускает обучение
```

---

## 🚀 Ключевые возможности

- **Дифференцируемый нейрограф** – каждый нейрон имеет обучаемый эмбеддинг; синапсы обучаются через обратное распространение.
- **FAISS-память** – сверхбыстрый поиск в рабочей и эпизодической памяти.
- **Полнофункциональный веб-интерфейс** – чат, настройки, обучение, статистика, история.
- **Автономный BrainAgent** – задаёт вопросы, учится на ответах, работает в интерактивном или фоновом режиме.
- **Teacher и саморефлексия** – оценивает качество ответов и улучшает граф.
- **Режим сна** – консолидирует память и удаляет слабые связи.
- **Сохранение состояния** – веса графа и индексы памяти сохраняются на диск.
- **Интеграция с LM Studio** – используйте любую локальную LLM с OpenAI‑совместимым API.

---

## 📦 Технологический стек

- **Бэкенд**: Python 3.10+, FastAPI, Uvicorn  
- **Машинное обучение**: PyTorch, PyTorch Geometric, sentence‑transformers  
- **Векторный поиск**: FAISS (CPU)  
- **LLM**: OpenAI SDK (работает с LM Studio, Ollama или любым OpenAI‑совместимым сервером)  
- **Фронтенд**: чистый HTML5, CSS3, JavaScript (без фреймворков)  
- **Окружение**: python‑dotenv, стандартная библиотека  

---

## 🛠 Установка и настройка

### 1. Клонируйте репозиторий

```bash
git clone https://github.com/Jasst/netlord.git
cd netlord
```

### 2. Создайте виртуальное окружение

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
# или
venv\Scripts\activate      # Windows
```

### 3. Установите зависимости

```bash
pip install -r requirements.txt
```

Типичный `requirements.txt`:

```txt
fastapi==0.115.0
uvicorn[standard]==0.30.0
torch==2.4.0
torch-geometric==2.5.0
faiss-cpu==1.8.0
sentence-transformers==2.2.2
openai==1.35.0
python-dotenv==1.0.0
```

> **Примечание:** Если у вас есть CUDA-видеокарта, вы можете заменить `faiss-cpu` на `faiss-gpu` и установить PyTorch с поддержкой CUDA.

### 4. Настройте окружение

Создайте файл `.env`:

```env
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
LM_STUDIO_API_KEY=not-needed
ADMIN_API_KEY=your-secure-key   # опционально, генерируется автоматически
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

Убедитесь, что LM Studio (или другой OpenAI‑совместимый сервер) запущен и доступен по указанному URL.

### 5. Запустите приложение

```bash
python app.py
```

Откройте браузер по адресу `http://127.0.0.1:8000` и начинайте общаться.

---

## 💬 Использование

### Веб-интерфейс

- **Чат**: введите вопрос в нижнем поле и нажмите Enter.
- **Обучение пары**: после каждого ответа появляются две кнопки – ✅ (положительное) и ❌ (отрицательное). Используйте их для усиления или подавления последней пары вопрос-ответ.
- **История чатов**: левая боковая панель показывает все чаты; их можно переименовывать или удалять.
- **Настройки** (⚙️): регулируйте температуру, включайте уточняющие вопросы, настраивайте BrainAgent и запускайте пакетное обучение.
- **Статистика и факты**: в реальном времени показываются количество нейронов, синапсов, записей памяти и факты, использованные в последнем ответе.

### Команды в чате

В поле ввода можно вводить специальные команды:

- `learn <вопрос> => <ответ>` – явное положительное обучение
- `neg <вопрос> => <ответ>` – отрицательное обучение (тормозной синапс)
- `links <понятие>` – показать все исходящие связи понятия
- `forget <фраза>` – удалить нейроны и факты, связанные с этой фразой
- `stats` – краткая статистика
- `sleep` – запустить цикл консолидации
- `save` – сохранить текущее состояние
- `exit` – сохранить и выйти

### REST API

Бэкенд FastAPI предоставляет те же эндпоинты, что используются в интерфейсе. Вы можете вызывать их программно:

- `POST /ask` – отправить вопрос, получить ответ с фактами и уточняющим вопросом
- `POST /learn` / `/learn_neg` – обучить положительную / отрицательную пару
- `POST /train_topic` – автоматически сгенерировать и выучить много пар по теме
- `POST /train_pair` – обучить одну пару за несколько эпох
- `POST /sleep` – запустить консолидацию памяти
- `GET /stats` – получить статистику
- `POST /agent/start` / `/agent/stop` – управлять автономным BrainAgent
- `POST /chat/clear` – очистить историю диалога

---

## 📁 Структура проекта

```
.
├── app.py                  # FastAPI сервер (главный файл)
├── agent.py                # BrainAgent – автономный исследователь
├── brain/                  # Основной пакет (v7)
│   ├── __init__.py
│   ├── brain.py            # Класс Brain – оркестратор
│   ├── config.py           # Конфигурация
│   ├── graph.py            # DifferentiableNeuralGraph (GNN)
│   ├── memory.py           # HierarchicalMemory с FAISS
│   ├── llm.py              # LMStudioLLM – обёртка для LM Studio
│   ├── teacher.py          # Teacher – оценка через LLM
│   └── utils.py            # EmbeddingProvider, утилиты
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
├── templates/
│   └── index.html
├── brain_model_v7/         # Папка для сохранения (создаётся автоматически)
│   ├── graph.pth           # Веса GNN
│   ├── meta.json           # Метаданные (счётчик, индекс понятий и т.д.)
│   └── dialog_history.json # История диалогов
├── .env                    # Переменные окружения
├── requirements.txt        # Зависимости Python
└── README.md               # Этот файл
```

---

## 🔬 Как это работает – внутреннее устройство

### Дифференцируемый нейрограф

- **Нейроны** – это обучаемые эмбеддинги (параметры) с дополнительными метаданными (`label`, `cluster`, `layer`).
- **Синапсы** – рёбра графа, по которым GNN передаёт сообщения.
- Используются слои **GATConv** (Graph Attention Networks) для обновления состояний нейронов.
- Параметр масштаба обновления управляет величиной изменения, обеспечивая плавную адаптацию.

### Иерархическая память (FAISS)

- **Рабочая память** хранит последние входные векторы в небольшой очереди.
- **Эпизодическая память** использует FAISS-индекс HNSW для быстрого приближённого поиска.
- При вызове `consolidate()` элементы рабочей памяти переносятся в эпизодическую.
- Поиск возвращает топ‑k наиболее похожих элементов с их метаданными.

### LM Studio LLM

- Все вызовы к LLM проходят через `LMStudioLLM.generate(prompt, max_tokens, temperature)`.
- Это позволяет легко переключиться на другого провайдера (Ollama, OpenAI и т.д.) простой сменой `base_url`.

### Teacher

- Использует ту же LLM для оценки ответов по четырём критериям: релевантность, точность, полнота, ясность.
- Возвращает оценку (0–1) и улучшенную версию ответа.
- Оценка используется для решения, усиливать или подавлять связь.

### BrainAgent

- Работает в фоновом потоке.
- **Автономный режим**: выбирает тему, генерирует вопросы, отвечает через мозг, оценивает через Teacher и учится.
- **Интерактивный режим**: задаёт вопрос пользователю, ждёт ответа и учится на нём.
- Периодически сохраняет модель.

### Сон

- Консолидирует память (рабочая → эпизодическая).
- Удаляет слабые нейроны и синапсы (по порогу или возрасту).
- (Опционально) строит ассоциативные мостики между похожими концептами.

---

## 🧪 Примеры использования

1. **Персональный ассистент с постоянной памятью** – запоминает факты о пользователе между сессиями без переобучения LLM.
2. **Офлайн-база знаний** – запуск на Raspberry Pi без интернета, используя только граф (режим fallback).
3. **Игровой NPC с эволюционирующей личностью** – NPC запоминает взаимодействия с игроком и адаптирует поведение.
4. **Расширение RAG** – динамическая самообучающаяся система поиска, улучшающаяся со временем.
5. **Исследовательская платформа** – эксперименты с различными архитектурами GNN, стратегиями консолидации памяти и правилами обучения.

---

## 🔮 Планы на будущее

- Поддержка спайковых нейронных сетей и STDP
- Мультиагентные разделяемые графы памяти
- Интеграция с обучением с подкреплением
- Интерактивная визуализация графа
- Сжатие памяти через обнаружение сообществ
- Распределённое / федеративное обучение

---

## 📄 Лицензия

Этот проект распространяется под лицензией **MIT**.  
Вы можете свободно использовать, изменять и распространять его при условии сохранения уведомления об авторских правах и разрешении во всех копиях или значимых частях программного обеспечения.

---

## 🌐 Репозиторий

GitHub: [https://github.com/Jasst/netlord](https://github.com/Jasst/netlord)

Если проект оказался полезным, поставьте звёздочку ⭐ и поделитесь своими экспериментами. Issues и pull request’ы всегда приветствуются!