# brain/brain.py
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import time
import re
import threading
from collections import deque
from typing import List, Dict, Optional, Any

from brain.config import BrainConfig
from brain.graph import DifferentiableNeuralGraph
from brain.memory import HierarchicalMemory
from brain.llm import LMStudioLLM
from brain.utils import EmbeddingProvider, random_vector, cosine_similarity

class Brain(nn.Module):
    def __init__(self, config: BrainConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim_embedding
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Компоненты
        self.embedder = EmbeddingProvider(dim=self.dim)
        self.graph = DifferentiableNeuralGraph(
            dim=self.dim,
            max_nodes=self.config.max_neurons,
            hidden_dim=self.config.gnn_hidden_dim,
            num_heads=self.config.gnn_num_heads
        ).to(self.device)
        self.memory = HierarchicalMemory(
            dim=self.dim,
            working_size=self.config.working_memory_size,
            episodic_capacity=self.config.episodic_capacity
        )
        self.llm = LMStudioLLM()

        self.optimizer = optim.Adam(self.graph.parameters(), lr=self.config.learning_rate)

        self.step_counter = 0
        self._learn_counter = 0

        # Совместимость с v6
        self.dialog_memory = []          # список диалогов
        self.concept_index = {}          # понятия -> nid
        self.knowledge_base = []         # факты
        self.lock = threading.RLock()    # для потокобезопасности
        self.short_memory = deque(maxlen=100)  # для sleep (не используется, но для совместимости)

        self._init_architecture()

    def _init_architecture(self):
        for i in range(10):
            emb = random_vector(self.dim)
            self.graph.add_node(emb, label=f"init_{i}", cluster="hidden", layer=0)

    def forward(self, input_vec: torch.Tensor) -> torch.Tensor:
        x = self.graph.get_node_embeddings()
        if x.shape[0] == 0:
            return x
        x = self.graph(x)
        return x

    def text_to_embedding(self, text: str) -> torch.Tensor:
        return self.embedder.get_embedding(text).to(self.device)

    # ---------- Основной шаг (step) ----------
    def step(self, input_text: str) -> Dict[str, Any]:
        self.step_counter += 1
        query_vec = self.text_to_embedding(input_text)
        memory_results = self.memory.retrieve(query_vec, k=3)

        start_nid = self.graph.find_most_similar(query_vec, threshold=0.5)
        if start_nid is None:
            start_nid = self.graph.add_node(query_vec, label=input_text[:30], cluster="input", layer=0)

        self.forward(query_vec)   # обновляем состояния графа

        context = f"Вопрос: {input_text}\n"
        if memory_results:
            context += "Из памяти:\n"
            for res in memory_results:
                context += f"- {res['metadata'].get('text', '')}\n"
        label = self.graph.neuron_labels.get(start_nid, "")
        context += f"Ассоциация: {label}\n"

        answer = self.llm.generate(context, max_tokens=256, temperature=0.7)
        self.memory.add_working(query_vec, {"text": input_text, "answer": answer})

        return {
            "input": input_text,
            "answer": answer,
            "activated_neurons": [start_nid],
            "memory_results": memory_results
        }

    # ---------- Уточняющий вопрос ----------
    def _generate_clarifying_question(self, user_input: str, answer: str,
                                      history: List[Dict], facts: List[Dict]) -> Optional[str]:
        if len(facts) >= 3:
            return None
        history_text = ""
        for turn in history[-3:]:
            history_text += f"Пользователь: {turn.get('user', '')}\nАссистент: {turn.get('assistant', '')}\n"
        system_prompt = (
            "Ты – ассистент, который помогает пользователю уточнить или расширить его запрос. "
            "На основе истории диалога и только что данного ответа, сформулируй один уточняющий вопрос. "
            "Если уточнение не требуется – ответь 'НЕТ'."
        )
        user_prompt = (
            f"История:\n{history_text}\n\n"
            f"Вопрос: {user_input}\nОтвет: {answer}\n"
            f"Факты: {[f['q'] + ' -> ' + f['a'] for f in facts]}\n\n"
            f"Уточняющий вопрос:"
        )
        try:
            response = self.llm.generate(
                prompt=system_prompt + "\n" + user_prompt,
                max_tokens=60,
                temperature=0.6
            )
            reply = response.strip()
            if reply.upper() == "НЕТ" or not reply:
                return None
            return reply
        except Exception as e:
            print(f"Ошибка генерации уточняющего вопроса: {e}")
            return None

    # ---------- Обучение графа (feedback) ----------
    def learn_from_feedback(self, input_text: str, answer_text: str, reward: float):
        query_vec = self.text_to_embedding(input_text)
        answer_vec = self.text_to_embedding(answer_text)

        q_nid = self.graph.find_most_similar(query_vec, threshold=0.0)
        a_nid = self.graph.find_most_similar(answer_vec, threshold=0.0)
        if q_nid is not None and a_nid is not None:
            emb_q = self.graph.get_embedding_by_id(q_nid)
            emb_a = self.graph.get_embedding_by_id(a_nid)
            sim = cosine_similarity(emb_q, emb_a)
            loss = -torch.log(torch.sigmoid(torch.tensor(sim * 10.0)))
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.graph.add_synapse(q_nid, a_nid, weight=0.2)

        self.memory.add_episodic(query_vec + answer_vec, {"q": input_text, "a": answer_text, "reward": reward})
        if self.step_counter % 50 == 0:
            self.memory.consolidate()

    # ---------- Совместимые методы для app.py ----------
    def learn_pair(self, input_text: str, output_text: str, reinforce_boost: float = 0.15, epochs: int = 1):
        with self.lock:
            for _ in range(epochs):
                self.learn_from_feedback(input_text, output_text, reward=1.0)
                time.sleep(0.05)
            self._learn_counter += epochs
            if self._learn_counter % self.config.checkpoint_every == 0:
                self.save()

    def learn_negative_pair(self, input_text: str, output_text: str, penalty: float = 0.15):
        with self.lock:
            # Создаём тормозной синапс
            q_emb = self.text_to_embedding(input_text)
            a_emb = self.text_to_embedding(output_text)
            q_nid = self.graph.find_most_similar(q_emb, threshold=0.0)
            a_nid = self.graph.find_most_similar(a_emb, threshold=0.0)
            if q_nid is None:
                q_nid = self.graph.add_node(q_emb, label=input_text[:30], cluster="output", layer=0)
            if a_nid is None:
                a_nid = self.graph.add_node(a_emb, label=output_text[:30], cluster="output", layer=0)
            self.graph.add_synapse(q_nid, a_nid, weight=-penalty)
            self._learn_counter += 1
            if self._learn_counter % self.config.checkpoint_every == 0:
                self.save()

    # ---------- Сон ----------
    def sleep(self, duration_steps: int = 10):
        with self.lock:
            print("💤 Сон... (консолидация памяти)")
            # Консолидируем память
            self.memory.consolidate()
            # Можно добавить реплей опыта, но для простоты оставим только консолидацию
            # И уменьшаем важность старых эпизодов (эмуляция забывания)
            # В реальном FAISS сложно удалять, поэтому просто очистим рабочие
            self.memory.working.clear()
            print("😴 Сон завершён")

    # ---------- Сохранение / загрузка ----------
    def save(self, model_dir: str = None):
        path = model_dir or self.config.model_dir
        os.makedirs(path, exist_ok=True)
        torch.save(self.graph.state_dict(), f"{path}/graph.pth")
        meta = {
            "step_counter": self.step_counter,
            "learn_counter": self._learn_counter,
            "concept_index": self.concept_index,
            "knowledge_base": self.knowledge_base,
        }
        with open(f"{path}/meta.json", "w") as f:
            json.dump(meta, f)
        self.save_dialog_history(os.path.join(path, "dialog_history.json"))
        print(f"[Brain] Модель сохранена в {path}")

    def load(self, model_dir: str = None):
        path = model_dir or self.config.model_dir
        if not os.path.exists(path):
            print(f"[Brain] Директория {path} не найдена, запуск с нуля.")
            return
        graph_path = f"{path}/graph.pth"
        if os.path.exists(graph_path):
            self.graph.load_state_dict(torch.load(graph_path, map_location=self.device))
        meta_path = f"{path}/meta.json"
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self.step_counter = meta.get("step_counter", 0)
            self._learn_counter = meta.get("learn_counter", 0)
            self.concept_index = meta.get("concept_index", {})
            self.knowledge_base = meta.get("knowledge_base", [])
        self.load_dialog_history(os.path.join(path, "dialog_history.json"))
        print(f"[Brain] Модель загружена из {path}")

    # ---------- История диалогов ----------
    def save_dialog_history(self, filename: str = None):
        if filename is None:
            filename = os.path.join(self.config.model_dir, "dialog_history.json")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        data = []
        for item in self.dialog_memory:
            data.append({
                "user": item.get("user", ""),
                "assistant": item.get("assistant", ""),
                "time": item.get("time", 0)
            })
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_dialog_history(self, filename: str = None):
        if filename is None:
            filename = os.path.join(self.config.model_dir, "dialog_history.json")
        if os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.dialog_memory = data
            except Exception as e:
                print(f"Ошибка загрузки истории: {e}")

    # ---------- Статистика ----------
    def get_stats(self) -> Dict[str, Any]:
        return {
            "neurons": self.graph.node_emb.shape[0],
            "synapses": len(self.graph._edges),
            "concepts": len(self.concept_index),
            "knowledge_base": len(self.knowledge_base),
            "memory": {
                "working": len(self.memory.working),
                "episodic": self.memory.episodic.index.ntotal,
                "semantic": 0,
            },
            "step_counter": self.step_counter,
            "meta_lr": 0.0,
            "embedding_cache": len(self.embedder._cache),
        }

    # ---------- Дополнительные утилиты (для совместимости) ----------
    def normalize_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text.strip().lower())

    def text_to_signal(self, text: str, energy: float = 1.0):
        # Заглушка для совместимости с v6
        class FakeSignal:
            def __init__(self, emb, energy, text):
                self.embedding = emb
                self.energy = energy
                self.text = text
        return FakeSignal(self.text_to_embedding(text), energy, text)