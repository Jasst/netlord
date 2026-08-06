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
from brain.graph import DifferentiableNeuralGraph, HierarchicalGraph, NodeType
from brain.memory import HierarchicalMemory
from brain.llm import LLMInterface
from brain.utils import EmbeddingProvider, random_vector, cosine_similarity
from brain.search import WebSearcher


# ----------------------------------------------------------------------
# Вспомогательные модули
# ----------------------------------------------------------------------
class CuriosityModule:
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def compute_reward(self, query_vec: torch.Tensor, predicted_vec: torch.Tensor, actual_vec: torch.Tensor) -> float:
        error = F.mse_loss(predicted_vec, actual_vec)
        return error.item()


class Planner:
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def plan(self, question: str, context: str) -> List[str]:
        prompt = f"Составь план действий для ответа на вопрос: {question}\nКонтекст: {context}\nПлан (каждый пункт с новой строки):"
        plan_text = self.llm.generate(prompt, max_tokens=60, temperature=0.5)
        lines = [line.strip() for line in plan_text.split('\n') if line.strip()]
        return lines if lines else ["answer_directly"]


class Reflector:
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def should_reflect(self, answer: str) -> bool:
        if len(answer.split()) < 3 or "не знаю" in answer.lower():
            return True
        return False

    def reflect(self, question: str, answer: str) -> str:
        prompt = f"Исправь и улучши ответ на вопрос '{question}'. Текущий ответ: '{answer}'. Улучшенный ответ:"
        improved = self.llm.generate(prompt, max_tokens=150, temperature=0.3)
        return improved if improved.strip() else answer


class EWC:
    def __init__(self, model: nn.Module, lambda_: float = 0.1):
        self.model = model
        self.lambda_ = lambda_
        self.fisher = {}
        self.opt_params = {}

    def compute_fisher(self, dataset_loader):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher[name] = torch.zeros_like(param)
        self.model.train()
        for inputs, targets in dataset_loader:
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.fisher[name] += param.grad.data ** 2
        for name in self.fisher:
            self.fisher[name] /= len(dataset_loader)
        self.opt_params = {n: p.clone() for n, p in self.model.named_parameters() if p.requires_grad}

    def penalty(self) -> torch.Tensor:
        loss = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.fisher:
                loss += (self.fisher[name] * (param - self.opt_params[name]) ** 2).sum()
        return self.lambda_ * loss


# ----------------------------------------------------------------------
# Основной класс CognitiveBrain
# ----------------------------------------------------------------------
class CognitiveBrain(nn.Module):
    def __init__(self, config: BrainConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim_embedding
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Эмбеддер
        self.embedder = EmbeddingProvider(dim=self.dim, model_name=config.embedding_model)

        # Граф (иерархический или обычный)
        if config.use_hierarchical_graph:
            self.graph = HierarchicalGraph(
                dims=config.graph_levels,
                num_heads=config.gnn_num_heads,
                num_layers=config.gnn_num_layers,
                attn_heads=config.attention_heads
            ).to(self.device)
        else:
            self.graph = DifferentiableNeuralGraph(
                dim=self.dim,
                max_nodes=config.max_neurons,
                hidden_dim=config.gnn_hidden_dim,
                num_heads=config.gnn_num_heads,
                num_layers=config.gnn_num_layers
            ).to(self.device)

        # Память
        self.memory = HierarchicalMemory(
            dim=self.dim,
            working_size=config.working_memory_size,
            episodic_capacity=config.episodic_capacity
        )

        # LLM с поддержкой локального сервера
        self.llm = LLMInterface(
            model_name=config.llm_model,
            use_openai_api=config.use_openai_api,
            api_key=config.openai_api_key,
            base_url=config.llm_base_url
        )

        # Поиск
        self.searcher = WebSearcher(max_results=5)

        # Оптимизатор
        self.optimizer = optim.Adam(self.graph.parameters(), lr=config.learning_rate)

        # Счётчики
        self.step_counter = 0
        self._learn_counter = 0

        # Диалоговая история и база знаний
        self.dialog_memory = []
        self.concept_index = {}
        self.knowledge_base = []
        self.lock = threading.RLock()
        self.short_memory = deque(maxlen=100)

        # Улучшенные модули (опционально)
        self.curiosity = CuriosityModule(lr=config.curiosity_lr) if config.enable_curiosity else None
        self.planner = Planner(self.llm) if config.enable_planning else None
        self.reflector = Reflector(self.llm) if config.enable_reflection else None
        self.ewc = EWC(self.graph, lambda_=config.ewc_lambda) if config.enable_ewc else None

        # Инициализация графа
        self._init_architecture()

    def _init_architecture(self):
        for i in range(10):
            emb = random_vector(self.dim)
            self.graph.add_node(emb, label=f"init_{i}", cluster="hidden", layer=0, node_type=NodeType.CONCEPT)

    def forward(self, input_vec: torch.Tensor) -> torch.Tensor:
        return self.graph(input_vec)

    def text_to_embedding(self, text: str) -> torch.Tensor:
        return self.embedder.get_embedding(text).to(self.device)

    # ---------- Обучение ----------
    def learn_pair(self, input_text: str, output_text: str, reward: float = 1.0, epochs: int = 1):
        with self.lock:
            for _ in range(epochs):
                self._learn_from_pair(input_text, output_text, reward=reward)
                time.sleep(0.05)
            self._learn_counter += epochs
            if self._learn_counter % self.config.checkpoint_every == 0:
                self.save()

    def _learn_from_pair(self, q: str, a: str, reward: float):
        q_vec = self.text_to_embedding(q)
        a_vec = self.text_to_embedding(a)

        # Поиск или создание узла для вопроса
        if hasattr(self.graph, 'levels'):
            q_nid = self.graph.find_most_similar(q_vec, level_idx=0, threshold=0.6)
        else:
            q_nid = self.graph.find_most_similar(q_vec, threshold=0.6)

        if q_nid is None:
            q_nid = self.graph.add_node(q_vec, label=q[:30], cluster="concept", layer=0, node_type=NodeType.CONCEPT)
            self.concept_index[self._normalize(q)] = q_nid

        # Поиск или создание узла для ответа
        if hasattr(self.graph, 'levels'):
            a_nid = self.graph.find_most_similar(a_vec, level_idx=0, threshold=0.6)
        else:
            a_nid = self.graph.find_most_similar(a_vec, threshold=0.6)

        if a_nid is None:
            a_nid = self.graph.add_node(a_vec, label=a[:30], cluster="concept", layer=0, node_type=NodeType.CONCEPT)
            self.concept_index[self._normalize(a)] = a_nid

        self.graph.add_synapse(q_nid, a_nid, weight=0.2 * reward)

        loss = self._contrastive_loss(q_nid, a_nid)
        if self.ewc is not None:
            loss += self.ewc.penalty()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self._add_to_knowledge_base(q, a, q_vec, a_vec)
        if hasattr(self, 'memory'):
            self.memory.add_semantic_triple(q, "has_answer", a, confidence=reward)

    def _contrastive_loss(self, q_nid: int, a_nid: int) -> torch.Tensor:
        # Берём эмбеддинги узлов напрямую с уровня 0 (гарантированно размерность dim_embedding)
        if hasattr(self.graph, 'levels'):
            emb_q = self.graph.levels[0].node_emb[q_nid - 1]
            emb_a = self.graph.levels[0].node_emb[a_nid - 1]
        else:
            emb_q = self.graph.node_emb[q_nid - 1]
            emb_a = self.graph.node_emb[a_nid - 1]
        emb_q = F.normalize(emb_q.unsqueeze(0), p=2, dim=1)
        emb_a = F.normalize(emb_a.unsqueeze(0), p=2, dim=1)
        sim = F.cosine_similarity(emb_q, emb_a, dim=1)
        loss = -torch.log(torch.sigmoid(sim * 10.0)).mean()
        # регуляризация весов рёбер
        if hasattr(self.graph, 'get_edge_weights'):
            edge_w = self.graph.get_edge_weights()
            if edge_w.numel() > 0:
                loss += 1e-4 * torch.norm(edge_w, p=2)
        return loss

    def learn_negative_pair(self, input_text: str, output_text: str, penalty: float = 0.15):
        with self.lock:
            q_vec = self.text_to_embedding(input_text)
            a_vec = self.text_to_embedding(output_text)

            if hasattr(self.graph, 'levels'):
                q_nid = self.graph.find_most_similar(q_vec, level_idx=0, threshold=0.0)
                a_nid = self.graph.find_most_similar(a_vec, level_idx=0, threshold=0.0)
            else:
                q_nid = self.graph.find_most_similar(q_vec, threshold=0.0)
                a_nid = self.graph.find_most_similar(a_vec, threshold=0.0)

            if q_nid is None:
                q_nid = self.graph.add_node(q_vec, label=input_text[:30], cluster="output", layer=0, node_type=NodeType.CONCEPT)
            if a_nid is None:
                a_nid = self.graph.add_node(a_vec, label=output_text[:30], cluster="output", layer=0, node_type=NodeType.CONCEPT)

            self.graph.add_synapse(q_nid, a_nid, weight=-penalty)
            self._learn_counter += 1
            if self._learn_counter % self.config.checkpoint_every == 0:
                self.save()

    # ---------- Основной шаг ----------
    def step(self, input_text: str, use_search: bool = False) -> Dict[str, Any]:
        self.step_counter += 1

        # Планирование
        if self.planner is not None:
            context = self._build_context(input_text, [], None)
            plan = self.planner.plan(input_text, context)
            if "search" in plan or "поиск" in plan:
                use_search = True

        # Быстрый ответ на криптовалюту
        lower = input_text.lower()
        if any(kw in lower for kw in ["биткоин", "btc", "курс биткоина"]):
            price = self._get_crypto_price("bitcoin", "usd")
            if price is not None:
                answer = f"Текущий курс BTC/USD: ${price:.2f}"
                self._update_after_step(input_text, answer)
                return {"input": input_text, "answer": answer, "activated_neurons": [], "memory_results": []}

        # Поиск в интернете
        if use_search:
            enhanced = self._enhance_search_query(input_text)
            results = self.searcher.search(enhanced)
            if results:
                context = self._build_search_context(input_text, enhanced, results)
                answer = self.llm.generate(context, max_tokens=300, temperature=0.3)
            else:
                answer = "Не удалось найти информацию."
            self._update_after_step(input_text, answer)
            return {"input": input_text, "answer": answer, "activated_neurons": [], "memory_results": []}

        # Основной поток
        query_vec = self.text_to_embedding(input_text)
        memory_results = self.memory.retrieve(query_vec, k=5)

        if hasattr(self.graph, 'levels'):
            start_nid = self.graph.find_most_similar(query_vec, level_idx=0, threshold=0.5)
        else:
            start_nid = self.graph.find_most_similar(query_vec, threshold=0.5)

        if start_nid is None:
            start_nid = self.graph.add_node(query_vec, label=input_text[:30], cluster="input", layer=0, node_type=NodeType.SENSORY)

        # Обновление графа (вызов forward для распространения сигналов)
        _ = self.graph.forward()

        # Генерация ответа
        context = self._build_context(input_text, memory_results, start_nid)
        answer = self.llm.generate(context, max_tokens=300, temperature=0.7)

        # Рефлексия
        if self.reflector is not None and self.reflector.should_reflect(answer):
            improved = self.reflector.reflect(input_text, answer)
            if improved != answer:
                answer = improved
                self.learn_pair(input_text, answer, reward=0.9)

        # Внутреннее вознаграждение (любопытство)
        # ИСПРАВЛЕНО: используем query_vec как предсказание (размерность совпадает с actual_vec)
        if self.curiosity is not None:
            predicted_vec = query_vec  # предсказание – сам запрос (или можно использовать эмбеддинг ответа)
            actual_vec = self.text_to_embedding(answer)
            reward = self.curiosity.compute_reward(query_vec, predicted_vec, actual_vec)
            if reward > 0.1:
                self.learn_pair(input_text, answer, reward=min(reward, 1.0))

        # Сохранение в память
        self.memory.add_working(query_vec, {"text": input_text, "answer": answer})
        self._update_after_step(input_text, answer)

        return {
            "input": input_text,
            "answer": answer,
            "activated_neurons": [start_nid] if start_nid else [],
            "memory_results": memory_results
        }

    def _update_after_step(self, question: str, answer: str):
        self.dialog_memory.append({"user": question, "assistant": answer, "time": time.time()})
        if len(self.dialog_memory) > 1000:
            self.dialog_memory = self.dialog_memory[-1000:]

    def step_stream(self, input_text: str, use_search: bool = False):
        result = self.step(input_text, use_search)
        answer = result["answer"]
        for token in answer.split():
            yield token + " "

    # ---------- Вспомогательные ----------
    def _normalize(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text.strip().lower())

    def _add_to_knowledge_base(self, q: str, a: str, q_vec: torch.Tensor, a_vec: torch.Tensor):
        for item in self.knowledge_base:
            if item["q"] == q and item["a"] == a:
                return
        self.knowledge_base.append({
            "q": q,
            "a": a,
            "emb": (q_vec + a_vec) / 2,
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
        kb_facts = self._search_knowledge_base(query, top_k=3)
        if kb_facts:
            context += "Из базы знаний:\n" + "\n".join(kb_facts) + "\n"
        if hasattr(self.graph, 'node_labels') and start_nid in self.graph.node_labels:
            label = self.graph.node_labels.get(start_nid, "")
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
        return query + " " + time.strftime("%d.%m.%Y")

    def sleep(self, duration_steps: int = 10):
        with self.lock:
            print("💤 Сон... (консолидация памяти)")
            self.memory.consolidate(threshold=0.05)
            print("😴 Сон завершён")

    # ---------- Сохранение / загрузка ----------
    def save(self, model_dir: str = None):
        path = model_dir or self.config.model_dir
        os.makedirs(path, exist_ok=True)
        torch.save(self.graph.state_dict(), f"{path}/graph.pth")
        with open(f"{path}/edges.pkl", "wb") as f:
            if hasattr(self.graph, '_edges') and hasattr(self.graph, '_edge_weights'):
                edges = self.graph._edges
                weights = [w.detach().cpu().numpy() for w in self.graph._edge_weights]
                pickle.dump((edges, weights), f)
        meta = {
            "step_counter": self.step_counter,
            "learn_counter": self._learn_counter,
            "concept_index": self.concept_index,
            "knowledge_base": self.knowledge_base,
        }
        with open(f"{path}/meta.json", "w") as f:
            json.dump(meta, f, default=lambda o: o.tolist() if isinstance(o, torch.Tensor) else o)
        self.save_dialog_history(os.path.join(path, "dialog_history.json"))
        print(f"[Brain] Модель сохранена в {path}")

    def load(self, model_dir: str = None):
        path = model_dir or self.config.model_dir
        if not os.path.exists(path):
            return
        graph_path = f"{path}/graph.pth"
        if os.path.exists(graph_path):
            state_dict = torch.load(graph_path, map_location=self.device)
            self.graph.load_state_dict(state_dict, strict=False)
        edges_path = f"{path}/edges.pkl"
        if os.path.exists(edges_path):
            with open(edges_path, "rb") as f:
                edges, weights = pickle.load(f)
            if hasattr(self.graph, '_edges') and hasattr(self.graph, '_edge_weights'):
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
        num_nodes = self.graph.node_emb.shape[0] if hasattr(self.graph, 'node_emb') else sum(g.node_emb.shape[0] for g in self.graph.levels)
        num_edges = len(self.graph._edges) if hasattr(self.graph, '_edges') else 0
        return {
            "neurons": num_nodes,
            "synapses": num_edges,
            "concepts": len(self.concept_index),
            "knowledge_base": len(self.knowledge_base),
            "memory": {
                "working": len(self.memory.working),
                "episodic": self.memory.episodic.index.ntotal,
                "semantic": len(self.memory.semantic_memory.triples),
            },
            "step_counter": self.step_counter,
        }