# 🧠 Smart Brain v6

### Hybrid Associative Memory System with Web Interface

> **A production‑ready research platform that combines an associative neural graph, hierarchical memory, and LLM integration into a single interactive application.**

Smart Brain v6 is the next evolution of an experimental memory architecture inspired by biological neural systems. Instead of treating every conversation as isolated, it builds a persistent associative graph of concepts, reinforces useful connections, suppresses incorrect ones, consolidates memories during “sleep”, and gradually forgets unused information — all in real time.

Unlike earlier versions (v4, v5), this implementation is a **full‑stack web application** with:

- a PyTorch‑powered neural graph (with LSTM cells and multi‑head attention inside synapses),
- SQLite‑based hierarchical memory (working, episodic, semantic),
- a RESTful API (FastAPI),
- a modern, mobile‑friendly frontend,
- an autonomous **BrainAgent** that asks questions and learns from user feedback,
- and deep integration with any OpenAI‑compatible LLM (local LM Studio, remote APIs).

---

## Why Smart Brain?

Most LLMs are excellent at reasoning but lack persistent, long‑term memory. Even RAG is mostly passive — vectors are stored and retrieved, but nothing is actually *learned* over time.

Smart Brain explores a different direction:

> Instead of retrieving documents, the memory itself **learns**.

Connections become stronger after successful answers, weaker after failures. The graph reorganises itself continuously, and during “sleep” it consolidates short‑term experiences into long‑term semantic structures — just like the human brain.

---

## Key Features

- 🧠 **Associative Neural Graph** — built with PyTorch and NetworkX, every concept is a neuron, every relationship is a synapse with weight, plasticity, confidence, and attention vectors.
- 🔁 **Continuous Online Learning** — Hebbian learning, inhibitory synapses, and teacher‑guided reinforcement happen in real time.
- 💬 **Full‑Featured Web Interface** — chat history, temperature control, clarifying questions, and a detailed settings panel.
- 🤖 **BrainAgent** — an autonomous researcher that periodically generates questions, evaluates answers via an internal “Teacher”, and learns from the results, optionally in interactive mode where it asks the user.
- 🧩 **Hierarchical Memory (SQLite)** — working memory (current context), episodic (short‑term), and semantic (long‑term) with automatic consolidation and decay.
- 🧠 **Self‑Reflection** — each answer is evaluated; low‑confidence replies trigger clarifying questions.
- 📊 **Explainable** — you can inspect neuron connections with the `links` command or via the UI.
- 💾 **Full State Persistence** — graph topology, tensors, memory entries, dialog history, and meta‑parameters are saved automatically.

---

## Technology Stack

- **Backend**: Python 3.10+, FastAPI, Uvicorn
- **Machine Learning**: PyTorch, NetworkX, sentence‑transformers
- **Memory**: SQLite (hierarchical memory)
- **LLM Integration**: OpenAI SDK (compatible with LM Studio, Ollama, or any OpenAI API)
- **Frontend**: Vanilla HTML5, CSS3, JavaScript (no framework), Highlight.js for code blocks
- **Environment**: python‑dotenv, standard library

---

## Architecture at a Glance

```
User Question
       ↓
[Web UI] → FastAPI → Brain
       ↓
Embedding (sentence‑transformers)
       ↓
NeuralGraph (PyTorch + NetworkX)
   ├── Propagate signal through associative graph
   ├── Apply Hebbian updates & attention
   ├── Retrieve facts from HierarchicalMemory (SQLite)
   └── Return activated neurons + facts
       ↓
Generate answer using LLM (with RAG)
       ↓
Teacher evaluates (quality score)
       ↓
Learn: reinforce or inhibit synapses
       ↓
Sleep (consolidation, pruning, decay)
```

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Jasst/netlord.git
cd netlord
```

### 2. Create a virtual environment (recommended)

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

```
fastapi==0.115.0
uvicorn[standard]==0.30.0
torch==2.4.0
networkx==3.3
sentence-transformers==2.2.2
openai==1.35.0
python-dotenv==1.0.0
```

### 4. Configure environment

Create a `.env` file:

```env
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
LM_STUDIO_API_KEY=not-needed
ADMIN_API_KEY=your-secure-key   # optional, auto‑generated if not set
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

If you use LM Studio, start it and load a model. If you use another OpenAI‑compatible endpoint, adjust the URL accordingly.

### 5. Run the application

```bash
python app.py
```

The server will start at `http://127.0.0.1:8000`. Open that address in your browser.

---

## Usage

### Web Interface

- **Chat**: type a question in the bottom text area and press Enter.
- **Learn a pair**: after an answer, two buttons appear: ✅ (positive) and ❌ (negative) — use them to reinforce or suppress the last question‑answer pair.
- **Chat history**: left sidebar shows all chats; create, rename, or delete them.
- **Settings** (⚙️): adjust temperature, enable clarifying questions, configure the BrainAgent, and run batch training.
- **Stats & facts**: view real‑time statistics (neurons, synapses, memory entries) and the facts used in the last answer.

### Commands in the Chat

You can also type these special commands:

- `learn <question> => <answer>` — explicitly teach a pair.
- `neg <question> => <answer>` — negative learning (inhibitory synapse).
- `links <concept>` — show all outgoing connections of a concept.
- `forget <phrase>` — remove all neurons and facts related to that phrase.
- `stats` — show summary statistics.
- `sleep` — trigger a consolidation cycle.
- `save` — save the current state.
- `exit` — save and quit.

### REST API

The FastAPI backend exposes endpoints that the UI consumes, but you can also use them programmatically:

- `POST /ask` — send a question and get an answer with facts.
- `POST /learn` / `/learn_neg` — learn a positive or negative pair.
- `POST /train_topic` — automatically generate and learn many pairs on a given topic.
- `POST /train_pair` — learn a single pair with multiple epochs.
- `POST /sleep` — run memory consolidation.
- `GET /stats` — get current statistics.
- `POST /agent/start` / `/agent/stop` — control the autonomous BrainAgent.
- `POST /chat/clear` — clear dialog history.

---

## Project Structure

```
.
├── app.py                  # Main FastAPI application (includes all routes and static file serving)
├── auto_train.py           # Simplified version (API only, without frontend)
├── smart_brain_v6.py       # The core: Brain, NeuralGraph, HierarchicalMemory, Teacher, etc.
├── agent.py                # BrainAgent class (autonomous question‑asker & learner)
├── train_from_file.py      # Script to batch‑train from a .txt file (Q|A pairs)
├── static/
│   ├── css/
│   │   └── style.css       # All styles
│   └── js/
│       └── app.js          # Frontend logic (chat, settings, agent, training)
├── templates/
│   └── index.html          # Main page
├── brain_model_v6/         # Default persistence directory (auto‑created)
│   ├── graph/              # NeuralGraph data: tensors.pt + graph.pkl.gz
│   ├── memory.db           # SQLite database for hierarchical memory
│   ├── meta.json           # Metadata, concept index, knowledge base
│   └── dialog_history.json # Chat history
├── .env                    # Environment variables
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## How It Works — Internally

### NeuralGraph

- **Neurons** — each has an embedding (vector), activation, potential, energy, importance, a cluster label (input/hidden/output), an LSTM cell state, and a hidden state.
- **Synapses** — each has weight, plasticity, confidence, frequency, energy, learning rate, momentum, and three independent vectors (semantic, episodic, context) used for dynamic attention.
- **Propagation** — signals spread through the graph based on synaptic weights and attention scores. LSTM cells update per neuron, allowing context‑sensitive behaviour.

### HierarchicalMemory (SQLite)

- Stores items with content, embedding BLOB, memory level (working/episodic/semantic), importance, tags, and usage statistics.
- Retrieval uses dot‑product similarity (cosine) over all stored items, with recency and importance bonuses.
- Automatic consolidation: when episodic memory exceeds capacity, frequently accessed items are promoted to semantic memory.

### Teacher & Self‑Reflection

- The Teacher uses the same LLM to evaluate answers on relevance, accuracy, completeness, and clarity.
- Low scores trigger negative learning; high scores reinforce the connection.
- Self‑Reflection produces a confidence score and can generate clarifying questions.

### BrainAgent

- Runs in a background thread.
- In autonomous mode, it picks a topic, generates questions, answers them via the brain, evaluates with Teacher, and learns.
- In interactive mode, it asks the user a question, waits for an answer, and learns from the reply.
- Periodically saves the model to disk.

### Sleep & Consolidation

- Replays recent memories to strengthen circuits.
- Runs experience replay from the buffer.
- Decays memory importance over time.
- Prunes neurons and synapses with low energy/usage.
- Creates associative bridges between highly similar concepts (if `use_sleep_associations` is enabled).

---

## Use Cases

### 1. Personal Assistant with Permanent Memory

Teach it your preferences, schedule, and facts. It will remember them across sessions without needing to retrain the LLM.

### 2. Offline Knowledge Base

Run the autonomous version (without LLM) on a Raspberry Pi or edge device to answer questions using its associative graph only — no internet required.

### 3. Game NPC with Evolving Personality

An NPC can remember player interactions and adjust its behaviour through learned associations, creating a unique experience for each player.

### 4. RAG Extension for Any LLM

Use Smart Brain as a dynamic, self‑learning retrieval system that doesn't just pull from a fixed vector store but actually learns from each interaction.

### 5. Research Platform

Experiment with different learning rules, memory consolidation algorithms, and attention mechanisms inside the graph, all while having a full‑featured UI to test your ideas.

---

## Future Directions

- Spiking neural dynamics and STDP
- Multi‑agent shared memory graphs
- Reinforcement learning integration
- Graph visualisation (interactive network view)
- Memory compression via community detection
- Distributed / federated learning

---

## License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute it, provided that the copyright notice and permission notice are included in all copies or substantial portions of the Software.

---

## Repository

GitHub: [https://github.com/Jasst/netlord](https://github.com/Jasst/netlord)

If you find this project useful, please consider giving it a star ⭐ and sharing your experiments. Issues and pull requests are always welcome!