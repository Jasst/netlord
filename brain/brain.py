# brain/brain.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import os
import time
import re
import threading
import pickle
import requests
from collections import deque
from typing import List, Dict, Optional, Any, Tuple

from brain.config import BrainConfig
from brain.graph import DifferentiableNeuralGraph
from brain.memory import HierarchicalMemory
from brain.llm import LLMInterface
from brain.utils import EmbeddingProvider, random_vector, cosine_similarity
from brain.search import WebSearcher

class Brain(nn.Module):
    def __init__(self, config: BrainConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim_embedding
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Компоненты
        self.embedder = EmbeddingProvider(dim=self.dim, model_name=config.embedding_model)
        self.graph = DifferentiableNeuralGraph(
            dim=self.dim,
            max_nodes=config.max_neurons,
            hidden_dim=config.gnn_hidden_dim,
            num_heads=config.gnn_num_heads,
            num_layers=config.gnn_num_layers
        ).to(self.device)
        self.memory = HierarchicalMemory(
            dim=self.dim,
            working_size=config.working_memory_size,
            episodic_capacity=config.episodic_capacity
        )
        self.llm = LLMInterface(
            model_name=config.llm_model,
            use_openai_api=config.use_openai_api,
            api_key=config.openai_api_key
        )
        self.searcher = WebSearcher(max_results=5)

        self.optimizer = optim.Adam(self.graph.parameters(), lr=config.learning_rate)
        self.step_counter = 0
        self._learn_counter = 0

        # Совместимость
        self.dialog_memory = []
        self.concept_index = {}
        self.knowledge_base = []
        self.lock = threading.RLock()
        self.short_memory = deque(maxlen=100)

        self._init_architecture()

    def _init_architecture(self):
        # Инициализируем граф несколькими случайными узлами
        for i in range(10):
            emb = random_vector(self.dim)
            self.graph.add_node(emb, label=f"init_{i}", cluster="hidden", layer=0)

    def forward(self, input_vec: torch.Tensor) -> torch.Tensor:
        return self.graph(input_vec)

    def text_to_embedding(self, text: str) -> torch.Tensor:
        return self.embedder.get_embedding(text).to(self.device)

    # ---------- Обучение пары (положительный пример) ----------
    def learn_pair(self, input_text: str, output_text: str, reinforce_boost: float = 0.15, epochs: int = 1):
        with self.lock:
            for _ in range(epochs):
                self._learn_from_pair(input_text, output_text, reward=1.0)
                time.sleep(0.05)
            self._learn_counter += epochs
            if self._learn_counter % self.config.checkpoint_every == 0:
                self.save()

    def _learn_from_pair(self, q: str, a: str, reward: float):
        q_vec = self.text_to_embedding(q)
        a_vec = self.text_to_embedding(a)

        # Находим или создаём узлы
        q_nid = self.graph.find_most_similar(q_vec, threshold=0.6)
        if q_nid is None:
            q_nid = self.graph.add_node(q_vec, label=q[:30], cluster="concept", layer=0)
            self.concept_index[self._normalize(q)] = q_nid

        a_nid = self.graph.find_most_similar(a_vec, threshold=0.6)
        if a_nid is None:
            a_nid = self.graph.add_node(a_vec, label=a[:30], cluster="concept", layer=0)
            self.concept_index[self._normalize(a)] = a_nid

        # Добавляем синапс
        self.graph.add_synapse(q_nid, a_nid, weight=0.2 * reward)

        # Контрастивная потеря
        loss = self._contrastive_loss(q_nid, a_nid)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Сохраняем в knowledge_base
        self._add_to_knowledge_base(q, a, q_vec, a_vec)

    def _contrastive_loss(self, q_nid: int, a_nid: int) -> torch.Tensor:
        updated = self.graph.forward()  # [N, dim]
        emb_q = updated[q_nid - 1]
        emb_a = updated[a_nid - 1]
        # Нормализуем
        emb_q = F.normalize(emb_q.unsqueeze(0), p=2, dim=1)
        emb_a = F.normalize(emb_a.unsqueeze(0), p=2, dim=1)
        sim = F.cosine_similarity(emb_q, emb_a, dim=1)
        loss = -torch.log(torch.sigmoid(sim * 10.0)).mean()
        # Регуляризация весов
        edge_w = self.graph.get_edge_weights()
        if edge_w.numel() > 0:
            loss += 1e-4 * torch.norm(edge_w, p=2)
        return loss

    def learn_negative_pair(self, input_text: str, output_text: str, penalty: float = 0.15):
        with self.lock:
            q_vec = self.text_to_embedding(input_text)
            a_vec = self.text_to_embedding(output_text)
            q_nid = self.graph.find_most_similar(q_vec, threshold=0.0)
            a_nid = self.graph.find_most_similar(a_vec, threshold=0.0)
            if q_nid is None:
                q_nid = self.graph.add_node(q_vec, label=input_text[:30], cluster="output", layer=0)
            if a_nid is None:
                a_nid = self.graph.add_node(a_vec, label=output_text[:30], cluster="output", layer=0)
            self.graph.add_synapse(q_nid, a_nid, weight=-penalty)
            self._learn_counter += 1
            if self._learn_counter % self.config.checkpoint_every == 0:
                self.save()

    # ---------- Основной шаг (с гибридным поиском) ----------
    def step(self, input_text: str, use_search: bool = False) -> Dict[str, Any]:
        self.step_counter += 1
        # Обработка команд "запомни/забудь" перенесена в app.py, здесь не дублируем.

        # 1. Проверка на запросы о криптовалютах (пример)
        lower = input_text.lower()
        if any(kw in lower for kw in ["биткоин", "btc", "курс биткоина"]):
            price = self._get_crypto_price("bitcoin", "usd")
            if price is not None:
                return {
                    "input": input_text,
                    "answer": f"Текущий курс BTC/USD: ${price:.2f}",
                    "activated_neurons": [],
                    "memory_results": []
                }

        # 2. Принудительный интернет-поиск
        if use_search:
            enhanced = self._enhance_search_query(input_text)
            results = self.searcher.search(enhanced)
            if results:
                context = self._build_search_context(input_text, enhanced, results)
                answer = self.llm.generate(context, max_tokens=300, temperature=0.3)
            else:
                answer = "Не удалось найти информацию."
            return {"input": input_text, "answer": answer, "activated_neurons": [], "memory_results": []}

        # 3. Гибридный поиск: FAISS + граф
        query_vec = self.text_to_embedding(input_text)
        # Поиск в эпизодической памяти
        memory_results = self.memory.retrieve(query_vec, k=5)

        # Поиск в графе (ассоциации)
        start_nid = self.graph.find_most_similar(query_vec, threshold=0.5)
        if start_nid is None:
            start_nid = self.graph.add_node(query_vec, label=input_text[:30], cluster="input", layer=0)

        # Активируем граф (forward pass)
        updated = self.graph.forward()

        # Формируем контекст для LLM
        context = self._build_context(input_text, memory_results, start_nid)
        answer = self.llm.generate(context, max_tokens=300, temperature=0.7)

        # Сохраняем в рабочую память
        self.memory.add_working(query_vec, {"text": input_text, "answer": answer})

        return {
            "input": input_text,
            "answer": answer,
            "activated_neurons": [start_nid],
            "memory_results": memory_results
        }

    def step_stream(self, input_text: str, use_search: bool = False):
        # Аналогично step, но с генерацией токенов
        lower = input_text.lower()
        if use_search:
            enhanced = self._enhance_search_query(input_text)
            results = self.searcher.search(enhanced)
            if results:
                context = self._build_search_context(input_text, enhanced, results)
                for token in self.llm.generate_stream(context, max_tokens=300, temperature=0.3):
                    yield token
                return
            else:
                yield "Не удалось найти информацию."
                return

        query_vec = self.text_to_embedding(input_text)
        memory_results = self.memory.retrieve(query_vec, k=5)
        start_nid = self.graph.find_most_similar(query_vec, threshold=0.5)
        if start_nid is None:
            start_nid = self.graph.add_node(query_vec, label=input_text[:30], cluster="input", layer=0)
        self.graph.forward()

        context = self._build_context(input_text, memory_results, start_nid)
        for token in self.llm.generate_stream(context, max_tokens=300, temperature=0.7):
            yield token

    # ---------- Вспомогательные методы ----------
    def _normalize(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text.strip().lower())

    def _add_to_knowledge_base(self, q: str, a: str, q_vec: torch.Tensor, a_vec: torch.Tensor):
        for item in self.knowledge_base:
            if item["q"] == q and item["a"] == a:
                return
        self.knowledge_base.append({
            "q": q,
            "a": a,
            "emb": (q_vec + a_vec) / 2,  # усреднённый эмбеддинг
            "time": time.time(),
            "confidence": 0.5,
            "access_count": 0
        })
        if len(self.knowledge_base) > self.config.max_kb_size:
            self.knowledge_base.pop(0)

    def _build_context(self, query: str, memory_results: List[Dict], start_nid: int) -> str:
        context = f"Вопрос: {query}\n"
        if memory_results:
            context += "Из эпизодической памяти:\n"
            for res in memory_results[:3]:
                meta = res.get("metadata", {})
                context += f"- {meta.get('text', '')}\n"
        # Добавляем знания из базы знаний (поиск по сходству)
        kb_facts = self._search_knowledge_base(query, top_k=3)
        if kb_facts:
            context += "Из базы знаний:\n" + "\n".join(kb_facts) + "\n"
        # Ассоциация из графа
        label = self.graph.neuron_labels.get(start_nid, "")
        if label:
            context += f"Ассоциация: {label}\n"
        context += "Отвечай кратко и по существу. Если не знаешь, скажи: 'Я не знаю'."
        return context

    def _build_search_context(self, query: str, enhanced: str, results: List[Dict]) -> str:
        context = f"Вопрос: {query}\nУлучшенный запрос: {enhanced}\n"
        context += "Результаты поиска:\n"
        for r in results:
            context += f"- {r.get('title', '')}: {r.get('body', '')[:200]}\n"
        context += "На основе этих данных дай точный ответ. Если данных недостаточно, скажи: 'Не удалось найти'."
        return context

    def _search_knowledge_base(self, query: str, top_k: int = 3) -> List[str]:
        results = []
        if not self.knowledge_base:
            return results
        q_vec = self.text_to_embedding(query)
        scored = []
        for item in self.knowledge_base:
            emb = item.get("emb")
            if emb is None:
                continue
            sim = cosine_similarity(q_vec, emb)
            scored.append((sim, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        for sim, item in scored[:top_k]:
            if sim > 0.4:
                results.append(f"Q: {item['q']} -> A: {item['a']}")
        return results

    def _get_crypto_price(self, crypto_id: str = "bitcoin", vs_currency: str = "usd") -> Optional[float]:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={crypto_id}&vs_currencies={vs_currency}"
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return data.get(crypto_id, {}).get(vs_currency)
        except Exception:
            return None

    def _enhance_search_query(self, query: str) -> str:
        # Простое улучшение (добавляем дату)
        return query + " " + time.strftime("%d.%m.%Y")

    # ---------- Сон (консолидация) ----------
    def sleep(self, duration_steps: int = 10):
        with self.lock:
            print("💤 Сон... (консолидация памяти)")
            self.memory.consolidate()
            # Также можно обучить GNN на эпизодах (здесь пропускаем)
            print("😴 Сон завершён")

    # ---------- Сохранение / загрузка ----------
    def save(self, model_dir: str = None):
        path = model_dir or self.config.model_dir
        os.makedirs(path, exist_ok=True)
        torch.save(self.graph.state_dict(), f"{path}/graph.pth")
        with open(f"{path}/edges.pkl", "wb") as f:
            pickle.dump((self.graph._edges, [w.detach().cpu().numpy() for w in self.graph._edge_weights]), f)
        # Сохраняем метаданные
        meta = {
            "step_counter": self.step_counter,
            "learn_counter": self._learn_counter,
            "concept_index": self.concept_index,
            "knowledge_base": self.knowledge_base,
            "num_neurons": self.graph.node_emb.shape[0],
        }
        with open(f"{path}/meta.json", "w") as f:
            json.dump(meta, f, default=lambda o: o.tolist() if isinstance(o, torch.Tensor) else o)
        self.save_dialog_history(os.path.join(path, "dialog_history.json"))
        print(f"[Brain] Модель сохранена в {path}")

    def load(self, model_dir: str = None):
        path = model_dir or self.config.model_dir
        if not os.path.exists(path):
            return
        # Загружаем state_dict графа
        graph_path = f"{path}/graph.pth"
        if os.path.exists(graph_path):
            state_dict = torch.load(graph_path, map_location=self.device)
            self.graph.load_state_dict(state_dict, strict=False)
        edges_path = f"{path}/edges.pkl"
        if os.path.exists(edges_path):
            with open(edges_path, "rb") as f:
                edges, weights = pickle.load(f)
            self.graph._edges = edges
            self.graph._edge_weights = nn.ParameterList([nn.Parameter(torch.tensor(w)) for w in weights])
            self.graph._rebuild_edges()
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

    def save_dialog_history(self, filename: str = None):
        if filename is None:
            filename = os.path.join(self.config.model_dir, "dialog_history.json")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.dialog_memory, f, ensure_ascii=False, indent=2)

    def load_dialog_history(self, filename: str = None):
        if filename is None:
            filename = os.path.join(self.config.model_dir, "dialog_history.json")
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                self.dialog_memory = json.load(f)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "neurons": self.graph.node_emb.shape[0],
            "synapses": len(self.graph._edges),
            "concepts": len(self.concept_index),
            "knowledge_base": len(self.knowledge_base),
            "memory": {
                "working": len(self.memory.working),
                "episodic": self.memory.episodic.index.ntotal,
            },
            "step_counter": self.step_counter,
        }