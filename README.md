
# Smart Brain: Гибридная память для LLM и автономный ассоциативный граф

Два эксперимента, объединённых идеей: *«А что, если нейросеть не просто генерирует, а ещё и *помнит* так, как это делаем мы — через ассоциации, укрепление связей и сон?»*

---

## О чём вообще речь?

Это не очередной чат-бот. Это **две реализации ассоциативной долговременной памяти** для языковых моделей.

- **`smart_brain_v4.py`** — гибрид: локальная LLM (через LM Studio) + маленькая эмуляция нейронной сети, которая хранит выученные факты и подставляет их в промпт как RAG.
- **`smart_brain.py`** — полностью автономный ассоциативный граф без LLM. Генерирует ответы только на основе связей между понятиями (цепочки, похожие на работу ассоциативной памяти человека).

Оба кода — один концепт, два уровня «интеллекта».

---

## Как это работает (на пальцах)

### Внутри мозга живут:

- **Нейроны** — понятия. У каждого есть векторное представление (эмбеддинг) и метка-текст.
- **Синапсы** — связи между понятиями. У них есть:
  - вес (положительный = усиливает, отрицательный = тормозит),
  - уверенность,
  - частота использования,
  - собственные векторы для внимания (в v4).
- **Память** — рабочая, краткосрочная и долгосрочная (сон консолидирует).
- **Хеббовское обучение** — «вместе возбудившиеся нейроны укрепляют свои связи».
- **Учитель (Teacher)** — оценивает ответы через ту же LLM и решает, усилить связь или создать тормозную.

---

### Версия v4 (с LLM)

1. Вы задаёте вопрос.
2. Маленькая сеть ищет в своей базе факты (пары «вопрос → ответ») и активирует связанные нейроны.
3. Эти факты подставляются в промпт к большой модели (LM Studio).
4. LLM генерирует связный ответ, опираясь на найденные факты.
5. Учитель оценивает ответ и автоматически обучает сеть: если ответ хороший — укрепляет связь; если плохой — создаёт тормозной синапс.

**Что даёт:** LLM перестаёт быть «рыбой без памяти» — она получает персонализированные, запоминаемые знания, которые не улетают из контекста.

---

### Версия без LLM (`smart_brain.py`)

1. Вопрос преобразуется в вектор (локально, без API).
2. Активируются ближайшие входные нейроны.
3. Сигнал распространяется по сети, активируя выходные нейроны с метками.
4. Ответ — это цепочка понятий: «Солнце → звезда → жёлтый карлик».

**Что даёт:** полностью автономный ассоциативный движок, не требующий интернета или мощного GPU. Можно использовать как «запоминающую базу знаний» для встраиваемых систем.

---

## Какие преимущества у такого подхода?

- **Долгосрочная память без тонкой настройки.**  
  Не нужно дообучать LLM — достаточно добавить пару `learn "вопрос" => "ответ"`.
- **Персонализация.**  
  Каждый экземпляр мозга учится на своих диалогах и не забывает.
- **Объяснимость.**  
  Можно посмотреть связи: `links Солнце` покажет, с чем ассоциируется понятие.
- **Гибкость.**  
  Можно работать в режиме «чистой ассоциативной сети» или как RAG-провайдер для любой LLM.
- **Регуляризация.**  
  Есть автоматическая «чистка» — удаление слабых нейронов и синапсов, энергетический распад, сон для консолидации.

---

## Где это можно применить? (пара идей)

### 1. Личный помощник с памятью на годы
Вы говорите: *«Запомни, мой любимый цвет — синий»*.  
Через неделю спрашиваете: *«Какой у меня любимый цвет?»* — и получаете точный ответ, даже если LLM «забыла» контекст.

### 2. Экспертная система без интернета
Можно загрузить набор правил или фактов (через `learn`), и сеть будет отвечать на вопросы, используя только свои ассоциации. Отлично для закрытых контуров.

### 3. Игровой NPC с обучаемой личностью
Персонаж запоминает поступки игрока, строит ассоциации и меняет реплики в зависимости от накопленного опыта. Без дообучения большой модели.

### 4. Гибридный RAG для локальной LLM
Вы поднимаете LM Studio на своём компьютере, а Smart Brain служит внешней памятью, которая не ограничена размером контекстного окна.

---

## Как запустить?

Установите зависимости:
```bash
pip install openai numpy
```

Запустите LM Studio на том же или другом хосте, укажите IP в коде:
```python
base_url="http://192.168.0.13:1234/v1"
```

Запустите нужный файл:
```bash
python smart_brain_v4.py   # с LLM
python smart_brain.py      # без LLM
```

Команды в интерактивном режиме:
- `learn вопрос => ответ` — выучить связь.
- `neg вопрос => ответ` — отрицательное обучение.
- `links понятие` — посмотреть связи.
- `stats` — статистика.
- `sleep` — запустить сон (очистка, укрепление связей).
- `save` / `exit` — сохранить модель.

---

## Что внутри кода важного?

- **Тормозные синапсы** — блокируют распространение сигнала по ошибочному пути. Это помогает сети не закреплять плохие ответы.
- **Attention в синапсах (v4)** — каждый синапс имеет несколько «ключей» для сравнения с текущим контекстом, что позволяет динамически менять вес связи в зависимости от вопроса.
- **Энергетический распад** — всё живое умирает, если не используется. Нейроны и синапсы со временем слабеют, что предотвращает переполнение памяти.
- **Автосохранение** — после каждых 3 обучений модель сохраняется в JSON.

---

## Две версии — два подхода

| | `smart_brain.py` | `smart_brain_v4.py` |
|---|---|---|
| **Генерация ответа** | Ассоциативные цепочки | LLM (LM Studio) |
| **Память** | Встроенная сеть | Сеть + RAG-контекст |
| **Зависимости** | Только numpy | Нужен запущенный LM Studio |
| **Размер модели** | Несколько МБ | Вес LLM + несколько МБ сети |
| **Качество текста** | Сухие цепочки | Естественные предложения |

---

## Почему это не «ещё один RAG»?

Потому что здесь память **обучается** через взаимодействие с учителем, а не просто ищет по косинусной близости. Сеть сама решает, какие связи укреплять, а какие делать тормозными. И делает это в реальном времени, без переобучения.

Это похоже на то, как человек запоминает: повторяет правильные ассоциации и избегает ошибочных.

---

## И в конце

Я писал этот код как эксперимент — можно ли соединить «сырую» ассоциативную память с современными LLM, чтобы они стали не просто умными, а ещё и памятливыми. Получилось два рабочих инструмента, каждый из которых можно использовать по‑своему.

Если у вас есть идея или сценарий, где такое может пригодиться — попробуйте, код открыт. И не бойтесь заглядывать внутрь: там нет магии, только нейроны, синапсы и немного случайности.


# 🧠 Smart Brain
### Hybrid Associative Memory for LLMs and an Autonomous Knowledge Graph

> **A research project exploring how Large Language Models can acquire long-term associative memory without fine-tuning.**

Two implementations built around the same idea:

> *"What if an AI could not only generate text, but also remember, reinforce, forget, and build associations like the human brain?"*

---

# Overview

Smart Brain is an experimental memory architecture inspired by biological neural systems.

Instead of treating every conversation as isolated, Smart Brain continuously builds an associative graph of concepts, strengthens useful connections, suppresses incorrect ones, consolidates memories during "sleep", and gradually forgets unused information.

The project currently contains **two independent implementations**:

| Version | Description |
|---------|-------------|
| **smart_brain.py** | Fully autonomous associative neural graph. Generates answers without any LLM. |
| **smart_brain_v4.py** | Hybrid memory system that extends any LLM (LM Studio/OpenAI compatible) with long-term associative memory. |

Repository:

**GitHub**
https://github.com/Jasst/netlord

---

# Why?

Traditional LLMs have excellent reasoning abilities...

…but almost no persistent memory.

Even Retrieval-Augmented Generation (RAG) is mostly passive:

- store vectors
- retrieve vectors
- inject into prompt

Nothing is actually **learned**.

Smart Brain explores a different direction:

Instead of retrieving documents...

**the memory itself learns.**

Connections become stronger after successful use.

Wrong associations become inhibitory.

Unused memories decay.

The graph reorganizes itself continuously.

---

# Main Features

## 🧠 Associative Memory

Every concept becomes a neuron.

Every learned relationship becomes a synapse.

The system remembers through associations instead of static embeddings.

---

## 🔗 Hebbian Learning

Inspired by neuroscience:

> "Neurons that fire together wire together."

Frequently co-activated concepts automatically reinforce their connections.

---

## 🚫 Inhibitory Synapses

Not every learned connection should survive.

Incorrect answers produce **negative synapses** that suppress future activation.

This allows the network to learn both:

- what is correct
- what should be avoided

---

## 😴 Sleep & Consolidation

The network periodically enters a "sleep" phase.

During sleep it:

- consolidates short-term memory
- reinforces useful pathways
- removes weak neurons
- removes weak synapses
- performs energy decay
- compresses memory

Exactly like biological consolidation.

---

## ⚡ Energy Model

Every neuron and synapse has energy.

Unused structures slowly decay.

Useful structures survive.

This prevents unlimited memory growth.

---

## 🧩 Explainable AI

Unlike transformers, Smart Brain can explain itself.

Example:

```
links Sun
```

Output:

```
Sun
 ├── Star
 ├── Yellow dwarf
 ├── Solar System
 └── Fusion
```

Every learned association is visible.

---

# Architecture

## Version 1 — Autonomous Brain

```
Question

↓

Embedding

↓

Input neurons

↓

Associative graph

↓

Propagation

↓

Activated concepts

↓

Generated concept chain
```

Example:

```
Sun

↓

Star

↓

Yellow Dwarf

↓

Main Sequence
```

No LLM required.

No internet required.

No API required.

---

## Version 2 — Hybrid Brain (LLM)

```
Question

↓

Embedding

↓

Associative Memory

↓

Relevant Facts

↓

Prompt Construction (RAG)

↓

LLM (LM Studio)

↓

Teacher Evaluation

↓

Graph Learning
```

The LLM becomes only the language generator.

Memory exists independently.

---

# Memory Layers

The architecture contains multiple memory systems.

### Working Memory

Current reasoning context.

---

### Dialog Memory

Recent conversation history.

---

### Short-Term Memory

Recently activated knowledge.

---

### Long-Term Memory

Persistent learned knowledge.

---

# Attention inside Synapses

Unlike ordinary graphs...

Each synapse stores multiple semantic vectors:

- semantic vector
- episodic vector
- context vector
- attention keys

Therefore the same connection may become stronger or weaker depending on the current context.

The graph is dynamic rather than static.

---

# Automatic Teacher

The hybrid version contains a second LLM acting as a teacher.

Workflow:

1. User asks a question.
2. Smart Brain generates an answer.
3. Teacher evaluates it.
4. Score > threshold → reinforce.
5. Score < threshold → create inhibitory learning.

This enables continuous online learning.

No retraining required.

---

# Commands

Learning:

```
learn Question => Answer
```

Negative learning:

```
neg Question => Wrong Answer
```

Show graph connections:

```
links Concept
```

Statistics:

```
stats
```

Sleep:

```
sleep
```

Save:

```
save
```

Exit:

```
exit
```

---

# Installation

Install dependencies:

```bash
pip install numpy openai
```

Configure LM Studio:

```python
client = OpenAI(
    base_url="http://192.168.0.13:1234/v1",
    api_key="not-needed"
)
```

Run:

```bash
python smart_brain_v4.py
```

or

```bash
python smart_brain.py
```

---

# Example

Teach:

```
learn Earth => Planet
learn Planet => Solar System
learn Solar System => Milky Way
```

Ask:

```
What is Earth?
```

Possible output:

> Earth is a planet in the Solar System.

Without LLM:

```
Earth

↓

Planet

↓

Solar System

↓

Milky Way
```

---

# Research Ideas

This project explores several research directions.

## 1. Long-Term Memory for LLMs

Replace repeated prompt engineering with an adaptive memory that evolves over months or years.

Instead of constantly injecting documents into prompts, the model develops persistent semantic structures.

---

## 2. Artificial Associative Cortex

The graph resembles a simplified associative cortex.

Concepts become distributed memories connected through reinforcement.

Activation spreads through the network rather than through token prediction.

---

## 3. Online Continual Learning

Unlike classical neural networks, Smart Brain learns continuously.

No gradient updates.

No retraining.

No expensive fine-tuning.

Knowledge is acquired through interaction.

---

## 4. Cognitive Digital Twin

The memory graph can gradually become a digital representation of a user's knowledge, preferences, habits, and reasoning style.

Over time it evolves into a personalized cognitive layer that can be attached to any language model.

---

## 5. Multi-Agent Shared Memory

Multiple LLM agents can share the same associative graph.

Instead of exchanging prompts, they exchange learned memories.

This creates a collaborative long-term knowledge network.

---

## 6. Robotics

Autonomous robots require memory that survives reboots.

Smart Brain can serve as:

- persistent world model
- navigation memory
- object association graph
- learned behavioral memory

without retraining the underlying language model.

---

## 7. NPCs and Game AI

Game characters can permanently remember:

- player actions
- locations
- conversations
- alliances
- betrayals

forming evolving personalities without scripting.

---

## 8. Edge AI & Offline Systems

The autonomous version requires only NumPy.

No GPU.

No cloud.

No internet.

Suitable for:

- Raspberry Pi
- ESP-class companion systems
- industrial controllers
- offline assistants

---

# How is this different from RAG?

Traditional RAG:

```
Documents

↓

Vector Search

↓

Prompt

↓

LLM
```

Smart Brain:

```
Experience

↓

Associative Graph

↓

Hebbian Learning

↓

Teacher Feedback

↓

Memory Consolidation

↓

Adaptive Retrieval

↓

LLM
```

Memory changes after every interaction.

The graph reorganizes itself.

Knowledge evolves.

---

# Current Status

✔ Autonomous associative memory

✔ Hybrid LLM memory

✔ Hebbian learning

✔ Inhibitory synapses

✔ Attention inside synapses

✔ Sleep consolidation

✔ Automatic pruning

✔ Long-term memory

✔ Explainable graph

✔ Teacher-guided online learning

---

# Future Ideas

- Spiking neuron dynamics
- STDP learning rules
- hippocampus-inspired episodic memory
- semantic clustering
- emotional weighting
- multi-modal memory
- reinforcement learning integration
- distributed graph synchronization
- hierarchical concept abstraction
- graph visualization
- memory compression using graph communities
- multi-agent collective memory

---

# Project Goals

This project is not intended to replace transformers.

Instead, it explores how an external associative memory inspired by biological cognition can complement modern language models.

The long-term vision is a modular cognitive architecture where:

- the **LLM** performs reasoning,
- the **associative graph** stores experience,
- the **teacher** evaluates outcomes,
- and **sleep** continuously reorganizes knowledge.

---

# Repository

GitHub:

https://github.com/Jasst/netlord

If you use Smart Brain in your own research or projects, feel free to open an issue, submit improvements, or share your experiments.