# brain/brain.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import os
import time
import re
import random
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
# Вспомогательные модули (без изменений)
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
    def __init__(self, model: nn.Module, lambda_: float = 0.1, decay: float = 0.99):
        self.model = model
        self.lambda_ = lambda_
        self.decay = decay
        self.fisher: Dict[str, torch.Tensor] = {}
        self.anchor: Dict[str, torch.Tensor] = {}
    def accumulate(self):
        for name, param in self.model.named_parameters():
            if not (param.requires_grad and param.grad is not None):
                continue
            g2 = param.grad.data.detach() ** 2
            if name not in self.fisher or self.fisher[name].shape != g2.shape:
                self.fisher[name] = g2.clone()
                self.anchor[name] = param.data.clone()
            else:
                self.fisher[name].mul_(self.decay).add_(g2, alpha=1 - self.decay)
    def set_anchor(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.anchor[name] = param.data.clone()
    def penalty(self) -> torch.Tensor:
        params = list(self.model.parameters())
        device = params[0].device if params else torch.device("cpu")
        loss = torch.zeros((), device=device)
        for name, param in self.model.named_parameters():
            if not param.requires_grad or name not in self.fisher or name not in self.anchor:
                continue
            if self.fisher[name].shape != param.shape:
                continue
            loss = loss + (self.fisher[name] * (param - self.anchor[name]) ** 2).sum()
        return self.lambda_ * loss


# ----------------------------------------------------------------------
# Основной класс CognitiveBrain
# ----------------------------------------------------------------------
class CognitiveBrain(nn.Module):
    SYSTEM_PROMPT = (
        "Ты — речевой аппарат когнитивного графа Smart Brain, а не самостоятельный источник знаний. "
        "Ниже дан 'поток ассоциаций' — концепты и факты, которые граф активировал в ответ на вопрос "
        "через распространение активации по своим связям (это и есть его мышление и память). "
        "Твоя задача — связно, естественно и кратко сформулировать ответ, опираясь ТОЛЬКО на эти "
        "активированные ассоциации и данные из памяти/базы знаний ниже. Не добавляй факты, которых "
        "там нет, и не рассуждай от себя. Если активированных ассоциаций недостаточно для ответа — "
        "прямо скажи, что граф пока не связал с этим вопросом ничего релевантного."
    )

    def __init__(self, config: BrainConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim_embedding
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embedder = EmbeddingProvider(dim=self.dim, model_name=config.embedding_model)

        if config.use_hierarchical_graph:
            self.graph = HierarchicalGraph(
                dims=config.graph_levels,
                num_heads=config.gnn_num_heads,
                num_layers=config.gnn_num_layers,
                cluster_threshold=config.hierarchy_cluster_threshold,
                min_cluster_size=config.hierarchy_min_cluster_size,
            ).to(self.device)
        else:
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
            api_key=config.openai_api_key,
            base_url=config.llm_base_url
        )

        self.searcher = WebSearcher(max_results=5)

        self.optimizer = optim.Adam(self.graph.parameters(), lr=config.learning_rate)

        self.step_counter = 0
        self._learn_counter = 0

        self.dialog_memory = []
        self.concept_index = {}
        self.knowledge_base = []
        self.lock = threading.RLock()
        self.short_memory = deque(maxlen=100)
        self.proactive_thoughts = []  # отдельное хранилище для внутренних мыслей

        self.curiosity = CuriosityModule(lr=config.curiosity_lr) if config.enable_curiosity else None
        self.planner = Planner(self.llm) if config.enable_planning else None
        self.reflector = Reflector(self.llm) if config.enable_reflection else None
        self.ewc = EWC(self.graph, lambda_=config.ewc_lambda) if config.enable_ewc else None

        self._init_architecture()

    def _init_architecture(self):
        for i in range(10):
            emb = random_vector(self.dim)
            self.graph.add_node(emb, label=f"init_{i}", cluster="hidden", layer=0,
                                 node_type=NodeType.CONCEPT, optimizer=self.optimizer)

    def forward(self, input_vec: torch.Tensor) -> torch.Tensor:
        return self.graph(input_vec)

    def text_to_embedding(self, text: str, is_query: bool = True) -> torch.Tensor:
        return self.embedder.get_embedding(text, is_query=is_query).to(self.device)

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
        q_vec = self.text_to_embedding(q, is_query=True)
        a_vec = self.text_to_embedding(a, is_query=False)
        merge_threshold = self.config.node_merge_threshold

        if hasattr(self.graph, 'levels'):
            q_nid = self.graph.find_most_similar(q_vec, level_idx=0, threshold=merge_threshold)
        else:
            q_nid = self.graph.find_most_similar(q_vec, threshold=merge_threshold)
        if q_nid is None:
            q_nid = self.graph.add_node(q_vec, label=q[:30], cluster="concept", layer=0,
                                         node_type=NodeType.CONCEPT, optimizer=self.optimizer)
            self.concept_index[self._normalize(q)] = q_nid

        if hasattr(self.graph, 'levels'):
            a_nid = self.graph.find_most_similar(a_vec, level_idx=0, threshold=merge_threshold)
        else:
            a_nid = self.graph.find_most_similar(a_vec, threshold=merge_threshold)
        if a_nid is None:
            a_nid = self.graph.add_node(a_vec, label=a[:30], cluster="concept", layer=0,
                                         node_type=NodeType.CONCEPT, optimizer=self.optimizer)
            self.concept_index[self._normalize(a)] = a_nid

        self.graph.add_synapse(q_nid, a_nid, weight=0.2 * reward, optimizer=self.optimizer)

        loss = self._contrastive_loss(q_nid, a_nid)
        if self.ewc is not None:
            loss = loss + self.ewc.penalty()
        loss.backward()
        if self.ewc is not None:
            self.ewc.accumulate()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self._add_to_knowledge_base(q, a, q_vec, a_vec)
        if hasattr(self, 'memory'):
            self.memory.add_semantic_triple(q, "has_answer", a, confidence=reward)

    def _contrastive_loss(self, q_nid: int, a_nid: int) -> torch.Tensor:
        h = self.graph.forward()
        n = h.shape[0]
        emb_q = F.normalize(h[q_nid - 1].unsqueeze(0), p=2, dim=1)
        emb_a = F.normalize(h[a_nid - 1].unsqueeze(0), p=2, dim=1)
        pos_sim = (emb_q * emb_a).sum(dim=1) * 10.0
        num_negatives = self.config.contrastive_num_negatives
        candidates = [i for i in range(n) if i not in (q_nid - 1, a_nid - 1)]
        k = min(num_negatives, len(candidates))
        if k > 0:
            neg_idx = torch.tensor(random.sample(candidates, k), device=h.device)
            emb_neg = F.normalize(h[neg_idx], p=2, dim=1)
            neg_sim = (emb_q @ emb_neg.T).squeeze(0) * 10.0
            logits = torch.cat([pos_sim, neg_sim]).unsqueeze(0)
            labels = torch.zeros(1, dtype=torch.long, device=h.device)
            loss = F.cross_entropy(logits, labels)
        else:
            loss = -torch.log(torch.sigmoid(pos_sim)).mean()
        if hasattr(self.graph, 'get_edge_weights'):
            edge_w = self.graph.get_edge_weights()
            if edge_w.numel() > 0:
                loss = loss + 1e-4 * torch.norm(edge_w, p=2)
        return loss

    def learn_negative_pair(self, input_text: str, output_text: str, penalty: float = 0.15):
        with self.lock:
            q_vec = self.text_to_embedding(input_text, is_query=True)
            a_vec = self.text_to_embedding(output_text, is_query=False)
            if hasattr(self.graph, 'levels'):
                q_nid = self.graph.find_most_similar(q_vec, level_idx=0, threshold=0.0)
                a_nid = self.graph.find_most_similar(a_vec, level_idx=0, threshold=0.0)
            else:
                q_nid = self.graph.find_most_similar(q_vec, threshold=0.0)
                a_nid = self.graph.find_most_similar(a_vec, threshold=0.0)
            if q_nid is None:
                q_nid = self.graph.add_node(q_vec, label=input_text[:30], cluster="output", layer=0,
                                             node_type=NodeType.CONCEPT, optimizer=self.optimizer)
            if a_nid is None:
                a_nid = self.graph.add_node(a_vec, label=output_text[:30], cluster="output", layer=0,
                                             node_type=NodeType.CONCEPT, optimizer=self.optimizer)
            self.graph.add_synapse(q_nid, a_nid, weight=-penalty, optimizer=self.optimizer)
            self._learn_counter += 1
            if self._learn_counter % self.config.checkpoint_every == 0:
                self.save()

    # ---------- Основной шаг ----------
    def step(self, input_text: str, use_search: bool = False, temperature: Optional[float] = None) -> Dict[str, Any]:
        self.step_counter += 1

        # Планирование
        if self.planner is not None:
            context = self._build_context(input_text, [], None)
            plan = self.planner.plan(input_text, context)
            if "search" in plan or "поиск" in plan:
                use_search = True

        # Обработка крипто
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
                full_answer = self.llm.generate(
                    context,
                    system=self.SYSTEM_PROMPT,
                    history=self._recent_history(),
                    max_tokens=2000,
                    temperature=temperature if temperature is not None else 0.3,
                )
            else:
                full_answer = "Не удалось найти информацию."
            self._update_after_step(input_text, full_answer)
            if self.config.two_level_answer:
                summary = self._summarize_answer(full_answer)
            else:
                summary = full_answer
            return {
                "input": input_text,
                "answer": summary,
                "thoughts": full_answer,
                "activated_neurons": [],
                "memory_results": []
            }

        # Основной путь: граф + LLM
        query_vec = self.text_to_embedding(input_text, is_query=True)
        memory_results = self.memory.retrieve(query_vec, k=10)

        if hasattr(self.graph, 'levels'):
            start_nid = self.graph.find_most_similar_contextual(query_vec, level_idx=0, threshold=0.5)
        else:
            start_nid = self.graph.find_most_similar_contextual(query_vec, threshold=0.5)

        if start_nid is None:
            start_nid = self.graph.add_node(query_vec, label=input_text[:30], cluster="input", layer=0,
                                             node_type=NodeType.SENSORY, optimizer=self.optimizer)

        thought_stream = self.graph.spreading_activation(start_nid, steps=3, decay=0.6, top_k=6)

        context = self._build_context(input_text, memory_results, start_nid, thought_stream)
        full_answer = self.llm.generate(
            context,
            system=self.SYSTEM_PROMPT,
            history=self._recent_history(),
            max_tokens=2000,
            temperature=temperature if temperature is not None else 0.7,
        )

        # Рефлексия и любопытство
        if self.reflector is not None and self.reflector.should_reflect(full_answer):
            improved = self.reflector.reflect(input_text, full_answer)
            if improved != full_answer:
                full_answer = improved
                self.learn_pair(input_text, full_answer, reward=0.9)

        if self.curiosity is not None:
            predicted_vec = query_vec
            actual_vec = self.text_to_embedding(full_answer, is_query=False)
            reward = self.curiosity.compute_reward(query_vec, predicted_vec, actual_vec)
            if reward > 0.1:
                self.learn_pair(input_text, full_answer, reward=min(reward, 1.0))

        self.memory.add_working(query_vec, {"text": input_text, "answer": full_answer})
        self._update_after_step(input_text, full_answer)

        if self.config.two_level_answer:
            summary = self._summarize_answer(full_answer)
        else:
            summary = full_answer

        return {
            "input": input_text,
            "answer": summary,
            "thoughts": full_answer,
            "activated_neurons": [start_nid] + [nid for nid, _ in thought_stream] if start_nid else [],
            "thought_stream": thought_stream,
            "memory_results": memory_results
        }

    def _summarize_answer(self, full_answer: str) -> str:
        if not self.config.two_level_answer:
            return full_answer
        prompt = (
            "Сократи следующий текст до 2–3 предложений, обращаясь прямо к пользователю. "
            "Сохрани суть, убери воду. Текст:\n" + full_answer
        )
        summary = self.llm.generate(
            prompt,
            max_tokens=self.config.summary_max_tokens,
            temperature=self.config.summary_temperature
        )
        return summary if summary.strip() else full_answer

    # ---------- ПРОАКТИВНЫЙ РЕЖИМ ----------
    def proactive_thought(self):
        if not self.config.proactive_enabled:
            return
        with self.lock:
            # Получаем граф уровня 0 (для иерархического) или сам граф
            if hasattr(self.graph, 'levels'):
                g = self.graph.levels[0]
            else:
                g = self.graph

            n_nodes = g.node_emb.shape[0]
            if n_nodes < 2:
                return

            # Выбор узла по степени (топ-10)
            degrees = {nid: len(g._adjacency.get(nid, [])) for nid in range(1, n_nodes + 1)}
            top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:10]
            start_id = random.choice(top_nodes) if top_nodes else random.randint(1, n_nodes)

            stream = g.spreading_activation(
                start_id,
                steps=self.config.proactive_steps,
                decay=0.6,
                top_k=self.config.proactive_top_k,
                min_activation=0.05
            )
            if not stream:
                return

            node_labels = g.node_labels  # метки узлов уровня 0
            associations = "\n".join(
                f"- {node_labels.get(nid, f'нейрон_{nid}')} (сила: {strength:.2f})"
                for nid, strength in stream
            )

            # Контекст диалога (без изменений)
            recent_context = ""
            if self.dialog_memory:
                last_turns = self.dialog_memory[-6:]
                for turn in last_turns:
                    if turn.get("user") and not turn["user"].startswith("[внутренняя мысль]"):
                        recent_context += f"Пользователь: {turn['user']}\n"
                    if turn.get("assistant") and not turn["assistant"].startswith("[внутренняя мысль]"):
                        recent_context += f"Ассистент: {turn['assistant']}\n"

            prompt = (
                f"Ты – внутренний голос когнитивного графа.\n"
                f"Недавний диалог:\n{recent_context}\n"
                f"Поток ассоциаций (концепты с силой активации):\n{associations}\n"
                "Сформулируй одну связную мысль, которая естественно вытекает из этих ассоциаций и развивает тему диалога."
            )
            thought = self.llm.generate(prompt, max_tokens=100, temperature=0.8)
            if thought and len(thought.strip()) > 20:
                self.learn_pair("внутренняя мысль", thought, reward=self.config.proactive_reward)
                self.proactive_thoughts.append({"thought": thought, "time": time.time()})
                print(f"[Proactive] Сгенерирована мысль: {thought[:80]}...")

    # ---------- Вспомогательные методы ----------
    def _recent_history(self, n_pairs: int = 4) -> List[Dict[str, str]]:
        msgs = []
        for turn in self.dialog_memory[-n_pairs:]:
            if turn.get("user"):
                msgs.append({"role": "user", "content": turn["user"]})
            if turn.get("assistant"):
                msgs.append({"role": "assistant", "content": turn["assistant"]})
        return msgs

    def _update_after_step(self, question: str, answer: str):
        self.dialog_memory.append({"user": question, "assistant": answer, "time": time.time()})
        if len(self.dialog_memory) > 1000:
            self.dialog_memory = self.dialog_memory[-1000:]

    def step_stream(self, input_text: str, use_search: bool = False):
        result = self.step(input_text, use_search)
        answer = result["answer"]
        for token in answer.split():
            yield token + " "

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

    def _build_context(self, query: str, memory_results: List[Dict], start_nid: int,
                        thought_stream: Optional[List[Tuple[int, float]]] = None) -> str:
        context = f"Вопрос: {query}\n"
        node_labels = getattr(self.graph, 'node_labels', {})
        start_label = node_labels.get(start_nid, "")
        if start_label:
            context += f"Отправная ассоциация графа: {start_label}\n"
        if thought_stream:
            context += "Поток ассоциаций графа (концепт — сила активации):\n"
            for nid, strength in thought_stream:
                label = node_labels.get(nid, "")
                if label:
                    context += f"- {label} (активация: {strength:.2f})\n"
        if memory_results:
            context += "Из эпизодической памяти:\n"
            for res in memory_results[:10]:
                meta = res.get("metadata", {})
                context += f"- {meta.get('text', '')}\n"
        kb_facts = self._search_knowledge_base(query, top_k=10)
        if kb_facts:
            context += "Из базы знаний:\n" + "\n".join(kb_facts) + "\n"
        context += "Сформулируй ответ на основе потока ассоциаций и данных выше. Если ничего релевантного не активировано, скажи: 'Я не знаю'."
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
        q_vec = self.text_to_embedding(query, is_query=True)
        scored = []
        for item in self.knowledge_base:
            emb = item.get("emb")
            if emb is None:
                continue
            if isinstance(emb, list):
                emb = torch.tensor(emb, dtype=torch.float32)
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

    # ---------- Сон и консолидация ----------
    def sleep(self, duration_steps: int = 10):
        with self.lock:
            print("💤 Сон... (консолидация памяти)")
            self.memory.consolidate(threshold=0.05)
            if hasattr(self.graph, "rebuild_hierarchy"):
                self.graph.rebuild_hierarchy(optimizer=self.optimizer)
            if self.ewc is not None:
                self.ewc.set_anchor()
            for _ in range(3):
                self.proactive_thought()
                time.sleep(0.5)
            print("😴 Сон завершён")

    # ---------- Сохранение / загрузка (ИСПРАВЛЕНЫ) ----------
    def save(self, model_dir: str = None):
        path = model_dir or self.config.model_dir
        os.makedirs(path, exist_ok=True)

        # 1. Сохраняем параметры графа (веса)
        torch.save(self.graph.state_dict(), f"{path}/graph.pth")
        with open(f"{path}/edges.pkl", "wb") as f:
            if hasattr(self.graph, '_edges') and hasattr(self.graph, '_edge_weights'):
                edges = self.graph._edges
                weights = [w.detach().cpu().numpy() for w in self.graph._edge_weights]
                pickle.dump((edges, weights), f)

        # 2. Сохраняем метаданные узлов (метки, типы, кластеры) — ЭТО НОВОЕ
        if hasattr(self.graph, 'levels'):
            # Иерархический граф — сохраняем все уровни
            meta_data = {}
            for idx, level in enumerate(self.graph.levels):
                meta_data[f'level_{idx}'] = {
                    'labels': level.node_labels,
                    'types': level.node_types,
                    'clusters': level.node_clusters,
                }
        else:
            meta_data = {
                'labels': self.graph.node_labels,
                'types': self.graph.node_types,
                'clusters': self.graph.node_clusters,
            }
        with open(f"{path}/nodes_meta.pkl", "wb") as f:
            pickle.dump(meta_data, f)

        # 3. База знаний
        kb_temp = f"{path}/knowledge_base.pkl.tmp"
        with open(kb_temp, "wb") as f:
            pickle.dump(self.knowledge_base, f)
        os.replace(kb_temp, f"{path}/knowledge_base.pkl")

        # 4. Остальные метаданные (счётчики, concept_index)
        meta = {
            "step_counter": self.step_counter,
            "learn_counter": self._learn_counter,
            "concept_index": self.concept_index,
        }
        meta_temp = f"{path}/meta.json.tmp"
        with open(meta_temp, "w") as f:
            json.dump(meta, f, default=lambda o: o.tolist() if isinstance(o, torch.Tensor) else o)
        os.replace(meta_temp, f"{path}/meta.json")

        # 5. История диалогов
        self.save_dialog_history(os.path.join(path, "dialog_history.json"))
        print(f"[Brain] Модель сохранена в {path}")

    def _resize_parameter(self, param: nn.Parameter, new_shape: tuple) -> nn.Parameter:
        if param.shape == new_shape:
            return param
        with torch.no_grad():
            new_data = torch.randn(new_shape, dtype=param.dtype, device=param.device) * 0.01
            if param.dim() >= 2:
                min_rows = min(param.shape[0], new_shape[0])
                new_data[:min_rows] = param.data[:min_rows]
            else:
                min_len = min(param.numel(), new_shape[0])
                new_data[:min_len] = param.data[:min_len]
        return nn.Parameter(new_data)

    def load(self, model_dir: str = None):
        path = model_dir or self.config.model_dir
        if not os.path.exists(path):
            print(f"[Brain] Папка модели {path} не найдена, начинаем с нуля.")
            return

        # 1. Загружаем параметры графа
        graph_path = f"{path}/graph.pth"
        if os.path.exists(graph_path):
            state_dict = torch.load(graph_path, map_location=self.device)

            def adapt_params(module, prefix=""):
                for name, param in list(module.named_parameters(recurse=False)):
                    full_name = prefix + name if prefix else name
                    if full_name in state_dict:
                        saved_shape = state_dict[full_name].shape
                        if param.shape != saved_shape:
                            print(f"[Brain] Адаптация {full_name}: {param.shape} -> {saved_shape}")
                            new_param = self._resize_parameter(param, saved_shape)
                            setattr(module, name, new_param)
                for child_name, child in module.named_children():
                    adapt_params(child, prefix + child_name + ".")

            adapt_params(self.graph)
            self.graph.load_state_dict(state_dict, strict=False)

        # 2. Загружаем рёбра
        edges_path = f"{path}/edges.pkl"
        if os.path.exists(edges_path):
            with open(edges_path, "rb") as f:
                edges, weights = pickle.load(f)
            if hasattr(self.graph, 'levels'):
                level0 = self.graph.levels[0]
                level0._edges = edges
                level0._edge_weights = nn.ParameterList([nn.Parameter(torch.tensor(w)) for w in weights])
                level0._rebuild_edges()
            else:
                self.graph._edges = edges
                self.graph._edge_weights = nn.ParameterList([nn.Parameter(torch.tensor(w)) for w in weights])
                self.graph._rebuild_edges()

        # 3. Загружаем метаданные узлов (ЭТО НОВОЕ)
        meta_path = f"{path}/nodes_meta.pkl"
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta_data = pickle.load(f)
            if hasattr(self.graph, 'levels'):
                for idx, level in enumerate(self.graph.levels):
                    level_key = f'level_{idx}'
                    if level_key in meta_data:
                        level.node_labels = meta_data[level_key]['labels']
                        level.node_types = meta_data[level_key]['types']
                        level.node_clusters = meta_data[level_key]['clusters']
            else:
                self.graph.node_labels = meta_data['labels']
                self.graph.node_types = meta_data['types']
                self.graph.node_clusters = meta_data['clusters']
        else:
            # Если файла с метаданными нет (старая модель) — попробуем восстановить метки из concept_index
            print("[Brain] nodes_meta.pkl не найден, восстанавливаем метки из concept_index (если возможно).")
            reverse_index = {v: k for k, v in self.concept_index.items()}
            if hasattr(self.graph, 'levels'):
                level = self.graph.levels[0]
                for nid in range(1, level.node_emb.shape[0] + 1):
                    if nid not in level.node_labels and nid in reverse_index:
                        level.node_labels[nid] = reverse_index[nid]
                        # типы и кластеры оставляем по умолчанию
            else:
                for nid in range(1, self.graph.node_emb.shape[0] + 1):
                    if nid not in self.graph.node_labels and nid in reverse_index:
                        self.graph.node_labels[nid] = reverse_index[nid]

        # 4. Загружаем meta.json
        meta_path_json = f"{path}/meta.json"
        if os.path.exists(meta_path_json):
            with open(meta_path_json, "r") as f:
                meta = json.load(f)
            self.step_counter = meta.get("step_counter", 0)
            self._learn_counter = meta.get("learn_counter", 0)
            self.concept_index = meta.get("concept_index", {})
            if "knowledge_base" in meta:
                self.knowledge_base = meta["knowledge_base"]
                for item in self.knowledge_base:
                    if "emb" in item and isinstance(item["emb"], list):
                        item["emb"] = torch.tensor(item["emb"], dtype=torch.float32)
                with open(f"{path}/knowledge_base.pkl", "wb") as f:
                    pickle.dump(self.knowledge_base, f)
                print("[Brain] Старый формат knowledge_base перенесён в отдельный файл.")
            else:
                kb_path = f"{path}/knowledge_base.pkl"
                if os.path.exists(kb_path):
                    with open(kb_path, "rb") as f:
                        self.knowledge_base = pickle.load(f)
                else:
                    self.knowledge_base = []

        # 5. История диалогов
        self.load_dialog_history(os.path.join(path, "dialog_history.json"))

        # 6. Пересоздаём оптимизатор
        self.optimizer = optim.Adam(self.graph.parameters(), lr=self.config.learning_rate)
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