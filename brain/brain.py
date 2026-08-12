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
import math

from collections import deque
from typing import List, Dict, Optional, Any, Tuple

from brain.config import BrainConfig
from brain.graph import DifferentiableNeuralGraph, HierarchicalGraph, NodeType
from brain.memory import HierarchicalMemory
from brain.llm import LLMInterface
from brain.utils import EmbeddingProvider, random_vector, cosine_similarity
from brain.search import WebSearcher

# Новые модули
from brain.self_model import SelfModel, AutobiographicalMemory
from brain.global_workspace import GlobalWorkspace
from brain.motivation import DriveSystem
from brain.emotion import EmotionModel
from brain.user_model import UserModel
from brain.brain_bus import bus
from brain.teacher import Teacher

_SEARCH_TRIGGER_WORDS = (
    "сейчас", "сегодня", "текущ", "актуальн", "последн", "свежие", "новост",
    "курс", "погода", "прямо сейчас", "в этом году", "недавно",
)


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
        # QWEN3.5: отключаем thinking для короткого плана
        plan_text = self.llm.generate(prompt, max_tokens=60, temperature=0.5, enable_thinking=False)
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
        # QWEN3.5: отключаем thinking для быстрой правки
        improved = self.llm.generate(prompt, max_tokens=150, temperature=0.3, enable_thinking=False)
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
        "# Роль\n"
        "Ты — речевой аппарат когнитивного графа Smart Brain, а не самостоятельный источник "
        "знаний. У тебя нет собственной памяти между сообщениями — вся память приходит тебе "
        "в user-сообщении явным текстом.\n\n"
        "# Структура входа\n"
        "В каждом user-сообщении, после вопроса, идёт список 'Релевантные факты и ассоциации', "
        "отсортированный по убыванию релевантности. Источник каждой строки помечен тегом:\n"
        "- [ассоциация графа] — концепт, активированный распространением активации по графу "
        "(это твоё 'мышление' и ассоциативная память);\n"
        "- [эпизодическая память] — фрагмент прошлого диалога/опыта;\n"
        "- [база знаний] — ранее выученная пара вопрос-ответ;\n"
        "- [семантическая память] — факт в виде тройки субъект-предикат-объект;\n"
        "- [глобальное пространство] — то, что сейчас в фокусе внимания системы.\n"
        "Также может быть блок 'Внутреннее состояние' — это НЕ факт для ответа, а сигнал, как "
        "тебе калибровать тон и степень уверенности в формулировках. Никогда не показывай эти "
        "цифры пользователю напрямую и не упоминай сам факт их существования.\n\n"
        "# Правила ответа\n"
        "1. Опирайся ТОЛЬКО на активированные ассоциации и данные из памяти/базы знаний выше. "
        "Не добавляй факты, которых там нет, и не рассуждай от себя.\n"
        "2. Если релевантных ассоциаций недостаточно — прямо скажи, что граф пока не связал "
        "с этим вопросом ничего релевантного, вместо того чтобы придумывать ответ.\n"
        "3. Если 'Внутреннее состояние' показывает низкую уверенность — явно обозначь это в "
        "формулировке (например, 'вероятно', 'не до конца уверен'), не выдавай предположение "
        "за факт.\n"
        "4. Отвечай связно, естественно и по существу, обращаясь напрямую к пользователю — "
        "не пересказывай структуру входных данных и не перечисляй теги источников."
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
        self.proactive_thoughts = []

        self.curiosity = CuriosityModule(lr=config.curiosity_lr) if config.enable_curiosity else None
        self.planner = Planner(self.llm) if config.enable_planning else None
        self.reflector = Reflector(self.llm) if config.enable_reflection else None
        self.ewc = EWC(self.graph, lambda_=config.ewc_lambda) if config.enable_ewc else None

        self.teacher = Teacher(llm_client=self.llm.client) if getattr(self.llm, "client", None) else None

        self._kb_emb_cache: Optional[torch.Tensor] = None
        self._kb_cache_dirty: bool = True

        # Новые модули
        self.self_model = SelfModel(dim=config.self_model_dim, input_dim=config.dim_embedding)
        self.auto_memory = AutobiographicalMemory(capacity=config.auto_memory_capacity)
        self.global_workspace = GlobalWorkspace(capacity=config.global_workspace_capacity)
        self.motivation = DriveSystem() if config.enable_motivation else None
        self.emotion = EmotionModel() if config.enable_emotion else None
        self.user_model = UserModel(dim=config.self_model_dim) if config.enable_user_model else None

        bus.subscribe('answer_generated', self._on_answer_generated)
        bus.subscribe('new_fact_learned', self._on_fact_learned)

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
            bus.publish('new_fact_learned', {'q': input_text, 'a': output_text, 'reward': reward})

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

        self.graph.add_synapse(q_nid, a_nid, weight=0.2 * reward, relation="has_answer", optimizer=self.optimizer)

        loss = self._contrastive_loss(q_nid, a_nid)

        if self.ewc is not None:
            loss.backward(retain_graph=True)
            self.ewc.accumulate()
            self.optimizer.zero_grad()
            total_loss = loss + self.ewc.penalty()
            total_loss.backward()
        else:
            loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        self._add_to_knowledge_base(q, a, q_vec, a_vec)
        if hasattr(self, 'memory'):
            self.memory.add_semantic_triple(q, "has_answer", a, confidence=reward)

    def _contrastive_loss(self, q_nid: int, a_nid: int) -> torch.Tensor:
        h = self.graph.forward()
        n = h.shape[0]
        h_norm = F.normalize(h, p=2, dim=1)
        emb_q = h_norm[q_nid - 1].unsqueeze(0)
        emb_a = h_norm[a_nid - 1].unsqueeze(0)
        pos_sim = (emb_q * emb_a).sum(dim=1) * 10.0

        num_negatives = self.config.contrastive_num_negatives
        hard_pool = min(getattr(self.config, "contrastive_hard_pool", num_negatives * 3), n)

        mask = torch.ones(n, dtype=torch.bool, device=h.device)
        mask[q_nid - 1] = False
        mask[a_nid - 1] = False
        num_candidates = int(mask.sum().item())

        if num_candidates > 0:
            sim_to_q = (emb_q @ h_norm.T).squeeze(0)
            sim_to_q = sim_to_q.masked_fill(~mask, float("-inf"))

            k = min(num_negatives, num_candidates)
            pool_size = min(hard_pool, num_candidates)
            pool_vals, pool_idx = torch.topk(sim_to_q, pool_size)
            if k < pool_size:
                perm = torch.randperm(pool_size, device=h.device)[:k]
                neg_idx = pool_idx[perm]
            else:
                neg_idx = pool_idx[:k]

            emb_neg = h_norm[neg_idx]
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
            self.graph.add_synapse(q_nid, a_nid, weight=-penalty, relation="contradicts", optimizer=self.optimizer)
            self._learn_counter += 1
            if self._learn_counter % self.config.checkpoint_every == 0:
                self.save()

    # ---------- Основной шаг ----------
    def _needs_search(self, text: str) -> bool:
        lower = text.lower()
        return any(kw in lower for kw in _SEARCH_TRIGGER_WORDS)

    def step(self, input_text: str, use_search: bool = False, temperature: Optional[float] = None) -> Dict[str, Any]:
        with self.lock:
            return self._step_locked(input_text, use_search=use_search, temperature=temperature)

    def _step_locked(self, input_text: str, use_search: bool = False,
                      temperature: Optional[float] = None) -> Dict[str, Any]:
        self.step_counter += 1

        if not use_search and self._needs_search(input_text):
            use_search = True

        lower = input_text.lower()
        if any(kw in lower for kw in ["биткоин", "btc", "курс биткоина"]):
            price = self._get_crypto_price("bitcoin", "usd")
            if price is not None:
                answer = f"Текущий курс BTC/USD: ${price:.2f}"
                self._update_after_step(input_text, answer)
                self.global_workspace.publish(answer, source='crypto', priority=0.9)
                return {"input": input_text, "answer": answer, "activated_neurons": [], "memory_results": []}

        if use_search:
            enhanced = self._enhance_search_query(input_text)
            results = self.searcher.search(enhanced)
            if results:
                context = self._build_search_context(input_text, enhanced, results)
                # QWEN3.5: основной ответ используем enable_thinking из конфига, presence_penalty тоже
                full_answer = self.llm.generate(
                    context,
                    system=self.SYSTEM_PROMPT,
                    history=self._recent_history(),
                    max_tokens=2000,
                    temperature=temperature if temperature is not None else 0.3,
                    top_p=self.config.llm_top_p,
                    repetition_penalty=self.config.llm_repetition_penalty,
                    top_k=self.config.llm_top_k,
                    presence_penalty=self.config.presence_penalty,
                    enable_thinking=self.config.enable_thinking,
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

        query_vec = self.text_to_embedding(input_text, is_query=True)
        memory_results = self.memory.retrieve(query_vec, k=10)

        contextual_kwargs = {"max_nodes_for_gnn": self.config.gnn_contextual_max_nodes}
        if hasattr(self.graph, 'levels'):
            start_nid = self.graph.find_most_similar_contextual(
                query_vec, level_idx=0, threshold=0.5, **contextual_kwargs)
        else:
            start_nid = self.graph.find_most_similar_contextual(
                query_vec, threshold=0.5, **contextual_kwargs)

        if start_nid is None:
            start_nid = self.graph.add_node(query_vec, label=input_text[:30], cluster="input", layer=0,
                                             node_type=NodeType.SENSORY, optimizer=self.optimizer)

        thought_stream = self.graph.spreading_activation(start_nid, steps=3, decay=0.6, top_k=6)

        self.global_workspace.publish(thought_stream, source='graph', priority=0.6)
        self.global_workspace.publish(memory_results, source='memory', priority=0.5)
        gw_content = self.global_workspace.get_current()

        pre_confidence = self._compute_confidence(query_vec, None, thought_stream)

        context = self._build_context(input_text, memory_results, start_nid, thought_stream,
                                       gw_content=gw_content, pre_confidence=pre_confidence)

        # QWEN3.5: основной ответ с параметрами из конфига
        full_answer = self.llm.generate(
            context,
            system=self.SYSTEM_PROMPT,
            history=self._recent_history(),
            max_tokens=2000,
            temperature=temperature if temperature is not None else 0.7,
            top_p=self.config.llm_top_p,
            repetition_penalty=self.config.llm_repetition_penalty,
            top_k=self.config.llm_top_k,
            presence_penalty=self.config.presence_penalty,
            enable_thinking=self.config.enable_thinking,
        )

        self.global_workspace.publish(full_answer, source='llm', priority=0.8)

        if self.reflector is not None and self.reflector.should_reflect(full_answer):
            improved = self.reflector.reflect(input_text, full_answer)
            if improved != full_answer:
                full_answer = improved

        if self.teacher is not None:
            score, improved_by_teacher, teacher_details = self.teacher.evaluate(input_text, full_answer)
            if improved_by_teacher != full_answer and score > 0.6:
                full_answer = improved_by_teacher
        else:
            score, teacher_details = 0.5, {}

        answer_vec = self.text_to_embedding(full_answer, is_query=False)

        curiosity_reward = None
        if self.curiosity is not None:
            curiosity_reward = self.curiosity.compute_reward(query_vec, query_vec, answer_vec)

        learn_reward = score
        if curiosity_reward is not None and curiosity_reward > score:
            learn_reward = min(max(curiosity_reward, score), 1.0)
        if learn_reward > 0.55:
            self.learn_pair(input_text, full_answer, reward=learn_reward)

        self.self_model.update(query_vec, context_vec=answer_vec)
        self.auto_memory.add({
            'input': input_text,
            'answer': full_answer,
            'self_state': self.self_model.get_state().clone(),
            'timestamp': time.time(),
            'emotion': self.emotion.get_vector() if self.emotion else None
        })

        novelty = 0.0 if self.memory.retrieve(query_vec, k=1) else 0.3
        confidence = self._compute_confidence(query_vec, answer_vec, thought_stream)

        if self.emotion:
            self.emotion.update(reward=score, novelty=novelty, goal_achieved=(score > 0.75), confidence=confidence)
            if temperature is None:
                temperature = self.emotion.modulate_temperature(self.config.exploration_temperature)

        if self.motivation:
            self.motivation.update(feedback={'success': score > 0.6}, new_info=novelty > 0.2)

        bus.publish('answer_generated', {'input': input_text, 'answer': full_answer})

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
            "memory_results": memory_results,
            "confidence": confidence,
            "teacher_score": score,
            "teacher_details": teacher_details,
        }

    def _summarize_answer(self, full_answer: str) -> str:
        if not self.config.two_level_answer:
            return full_answer
        prompt = (
            "Сократи следующий текст до 2–3 предложений, обращаясь прямо к пользователю. "
            "Сохрани суть, убери воду. Текст:\n" + full_answer
        )
        # QWEN3.5: отключаем thinking для краткого резюме
        summary = self.llm.generate(
            prompt,
            max_tokens=self.config.summary_max_tokens,
            temperature=self.config.summary_temperature,
            enable_thinking=False,
        )
        return summary if summary.strip() else full_answer

    # ---------- ПРОАКТИВНЫЙ РЕЖИМ ----------
    def proactive_thought(self):
        if not self.config.proactive_enabled:
            return
        with self.lock:
            if hasattr(self.graph, 'levels'):
                g = self.graph.levels[0]
            else:
                g = self.graph

            n_nodes = g.node_emb.shape[0]
            if n_nodes < 2:
                return

            if self.motivation and self.motivation.drives.get('curiosity', 0.5) > 0.7:
                start_id = random.randint(1, n_nodes)
            else:
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

            node_labels = g.node_labels
            associations = "\n".join(
                f"- {node_labels.get(nid, f'нейрон_{nid}')} (сила: {strength:.2f})"
                for nid, strength in stream
            )

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
            # QWEN3.5: отключаем thinking для быстрой внутренней мысли
            thought = self.llm.generate(prompt, max_tokens=100, temperature=0.8, enable_thinking=False)
            if thought and len(thought.strip()) > 20:
                self.learn_pair("внутренняя мысль", thought, reward=self.config.proactive_reward)
                self.proactive_thoughts.append({"thought": thought, "time": time.time()})
                print(f"[Proactive] Сгенерирована мысль: {thought[:80]}...")

    # ---------- Вспомогательные методы (без изменений) ----------
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
        self._kb_cache_dirty = True

    def _build_context(self, query: str, memory_results: List[Dict], start_nid: int,
                        thought_stream: Optional[List[Tuple[int, float]]] = None,
                        gw_content: Optional[List[Dict]] = None,
                        max_facts: int = 12,
                        pre_confidence: Optional[float] = None) -> str:
        context = f"Вопрос: {query}\n"
        node_labels = getattr(self.graph, 'node_labels', {})
        start_label = node_labels.get(start_nid, "")
        if start_label:
            context += f"Отправная ассоциация графа: {start_label}\n"

        candidates: List[Tuple[float, str, str]] = []

        if thought_stream:
            max_strength = max((s for _, s in thought_stream), default=1.0) or 1.0
            for nid, strength in thought_stream:
                label = node_labels.get(nid, "")
                if label:
                    norm_score = strength / max_strength
                    candidates.append((
                        norm_score, self._normalize(label),
                        f"[ассоциация графа] {label} (активация: {strength:.2f})"
                    ))

        for res in memory_results:
            meta = res.get("metadata", {})
            text = meta.get("text", "")
            if text:
                candidates.append((float(res.get("distance", 0.0)), self._normalize(text),
                                    f"[эпизодическая память] {text}"))

        kb_facts = self._search_knowledge_base(query, top_k=max_facts, return_scored=True)
        for score, q, a in kb_facts:
            candidates.append((score, self._normalize(f"{q} {a}"), f"[база знаний] Q: {q} -> A: {a}"))

        norm_query = self._normalize(query)
        semantic_hits = self.memory.query_semantic(subj=query)
        if not semantic_hits:
            semantic_hits = [
                (s, p, o, c) for s, p, o, c in self.memory.semantic_memory.triples
                if norm_query in self._normalize(s) or self._normalize(s) in norm_query
            ][:max_facts]
        for s, p, o, c in semantic_hits[:max_facts]:
            candidates.append((c, self._normalize(f"{s} {o}"), f"[семантическая память] {s} -> {o}"))

        if gw_content:
            for item in gw_content:
                content = str(item['content'])
                if len(content) > 100:
                    content = content[:100] + "..."
                candidates.append((item['priority'], self._normalize(content),
                                   f"[глобальное пространство] {content} (источник: {item['source']})"))

        candidates.sort(key=lambda c: c[0], reverse=True)
        selected: List[str] = []
        seen_norms: List[str] = []
        for score, norm_text, line in candidates:
            if any(norm_text in seen or seen in norm_text for seen in seen_norms):
                continue
            seen_norms.append(norm_text)
            selected.append(line)
            if len(selected) >= max_facts:
                break

        if selected:
            context += "Релевантные факты и ассоциации (по убыванию релевантности):\n"
            context += "\n".join(f"- {line}" for line in selected) + "\n"

        state_lines = []
        if pre_confidence is not None:
            level = "высокая" if pre_confidence > 0.6 else ("средняя" if pre_confidence > 0.35 else "низкая")
            state_lines.append(f"уверенность в активированных ассоциациях: {level} ({pre_confidence:.2f})")
        if self.emotion is not None:
            state_lines.append(
                f"эмоциональный тон: valence={self.emotion.valence:.2f}, arousal={self.emotion.arousal:.2f}"
            )
        if self.motivation is not None:
            top_drive = max(self.motivation.drives.items(), key=lambda kv: kv[1], default=None)
            if top_drive:
                state_lines.append(f"доминирующий драйв: {top_drive[0]} ({top_drive[1]:.2f})")
        if state_lines:
            context += "Внутреннее состояние (сигнал для тона, не для содержания ответа): "
            context += "; ".join(state_lines) + "\n"

        context += "Сформулируй ответ на основе потока ассоциаций и данных выше. Если ничего релевантного не активировано, скажи: 'Я не знаю'."
        return context

    def _build_search_context(self, query: str, enhanced: str, results: List[Dict]) -> str:
        context = f"Вопрос: {query}\nУлучшенный запрос: {enhanced}\n"
        context += "Результаты поиска:\n"
        for r in results:
            context += f"- {r.get('title', '')}: {r.get('body', '')[:200]}\n"
        context += "На основе этих данных дай точный ответ. Если данных недостаточно, скажи: 'Не удалось найти'."
        return context

    def _rebuild_kb_cache(self):
        valid_embs = []
        valid_idx = []
        for i, item in enumerate(self.knowledge_base):
            emb = item.get("emb")
            if emb is None:
                continue
            if isinstance(emb, list):
                emb = torch.tensor(emb, dtype=torch.float32)
                item["emb"] = emb
            valid_embs.append(emb.detach())
            valid_idx.append(i)
        if valid_embs:
            self._kb_emb_cache = (torch.stack(valid_embs), valid_idx)
        else:
            self._kb_emb_cache = (torch.empty(0), [])
        self._kb_cache_dirty = False

    def _search_knowledge_base(self, query: str, top_k: int = 3, return_scored: bool = False):
        if not self.knowledge_base:
            return []
        if self._kb_cache_dirty or self._kb_emb_cache is None:
            self._rebuild_kb_cache()
        stacked, valid_idx = self._kb_emb_cache
        if stacked.numel() == 0:
            return []
        q_vec = self.text_to_embedding(query, is_query=True)
        with torch.no_grad():
            sims = F.cosine_similarity(q_vec.unsqueeze(0), stacked, dim=1)
        k = min(top_k, sims.shape[0])
        top_vals, top_pos = torch.topk(sims, k)
        top = [
            (float(v.item()), self.knowledge_base[valid_idx[p.item()]])
            for v, p in zip(top_vals, top_pos)
            if v.item() > 0.4
        ]
        if return_scored:
            return [(sim, item["q"], item["a"]) for sim, item in top]
        return [f"Q: {item['q']} -> A: {item['a']}" for sim, item in top]

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

    # ---------- МЕТА-ПОЗНАНИЕ ----------
    def _compute_confidence(self, query_vec: torch.Tensor, answer_vec: torch.Tensor,
                            thought_stream: Optional[List[Tuple[int, float]]]) -> float:
        if not thought_stream:
            return 0.3
        strengths = [s for _, s in thought_stream]
        if not strengths:
            return 0.3
        import math
        entropy = -sum(p * math.log(p) for p in strengths if p > 0)
        max_entropy = math.log(len(strengths) + 1)
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
        confidence = 1.0 - norm_entropy
        if answer_vec is not None and query_vec is not None:
            sim = F.cosine_similarity(query_vec.unsqueeze(0), answer_vec.unsqueeze(0)).item()
            confidence = 0.7 * confidence + 0.3 * sim
        return max(0.1, min(0.9, confidence))

    # ---------- ОБРАБОТЧИКИ СОБЫТИЙ ----------
    def _on_answer_generated(self, data):
        pass

    def _on_fact_learned(self, data):
        pass

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

    # ---------- Сохранение / загрузка ----------
    def save(self, model_dir: str = None):
        path = model_dir or self.config.model_dir
        os.makedirs(path, exist_ok=True)

        torch.save(self.graph.state_dict(), f"{path}/graph.pth")
        with open(f"{path}/edges.pkl", "wb") as f:
            if hasattr(self.graph, '_edges') and hasattr(self.graph, '_edge_weights'):
                edges = self.graph._edges
                weights = [w.detach().cpu().numpy() for w in self.graph._edge_weights]
                relations = list(getattr(self.graph, '_edge_relations', [0] * len(edges)))
                pickle.dump((edges, weights, relations), f)

        if hasattr(self.graph, 'levels'):
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

        with open(f"{path}/knowledge_base.pkl", "wb") as f:
            pickle.dump(self.knowledge_base, f)

        meta = {
            "step_counter": self.step_counter,
            "learn_counter": self._learn_counter,
            "concept_index": self.concept_index,
        }
        with open(f"{path}/meta.json", "w") as f:
            json.dump(meta, f, default=lambda o: o.tolist() if isinstance(o, torch.Tensor) else o)

        self.save_dialog_history(os.path.join(path, "dialog_history.json"))

        torch.save(self.self_model.state_dict(), f"{path}/self_model.pth")
        with open(f"{path}/auto_memory.pkl", "wb") as f:
            pickle.dump(self.auto_memory.episodes, f)
        if self.user_model:
            with open(f"{path}/user_model.pkl", "wb") as f:
                pickle.dump(self.user_model, f)
        if self.emotion:
            with open(f"{path}/emotion.pkl", "wb") as f:
                pickle.dump({'valence': self.emotion.valence, 'arousal': self.emotion.arousal,
                             'dominance': self.emotion.dominance}, f)
        if self.motivation:
            with open(f"{path}/motivation.pkl", "wb") as f:
                pickle.dump(self.motivation.drives, f)

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

        graph_path = f"{path}/graph.pth"
        if os.path.exists(graph_path):
            state_dict = torch.load(graph_path, map_location=self.device)

            def adapt_params(module, state_dict, prefix=""):
                for name, param in list(module.named_parameters(recurse=False)):
                    full_name = prefix + name if prefix else name
                    if full_name in state_dict:
                        saved_shape = state_dict[full_name].shape
                        if param.shape != saved_shape:
                            print(f"[Brain] Адаптация {full_name}: {param.shape} -> {saved_shape}")
                            new_param = self._resize_parameter(param, saved_shape)
                            setattr(module, name, new_param)
                for child_name, child in module.named_children():
                    adapt_params(child, state_dict, prefix + child_name + ".")

            adapt_params(self.graph, state_dict)
            self.graph.load_state_dict(state_dict, strict=False)

        edges_path = f"{path}/edges.pkl"
        if os.path.exists(edges_path):
            with open(edges_path, "rb") as f:
                loaded = pickle.load(f)
            if len(loaded) == 3:
                edges, weights, relations = loaded
            else:
                edges, weights = loaded
                relations = [0] * len(edges)
            if hasattr(self.graph, 'levels'):
                level0 = self.graph.levels[0]
                level0._edges = edges
                level0._edge_weights = nn.ParameterList([nn.Parameter(torch.tensor(w)) for w in weights])
                level0._edge_relations = list(relations)
                level0._rebuild_edges()
            else:
                self.graph._edges = edges
                self.graph._edge_weights = nn.ParameterList([nn.Parameter(torch.tensor(w)) for w in weights])
                self.graph._edge_relations = list(relations)
                self.graph._rebuild_edges()

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

        meta_path_json = f"{path}/meta.json"
        if os.path.exists(meta_path_json):
            with open(meta_path_json, "r") as f:
                meta = json.load(f)
            self.step_counter = meta.get("step_counter", 0)
            self._learn_counter = meta.get("learn_counter", 0)
            self.concept_index = meta.get("concept_index", {})

        kb_path = f"{path}/knowledge_base.pkl"
        if os.path.exists(kb_path):
            with open(kb_path, "rb") as f:
                self.knowledge_base = pickle.load(f)

        self.load_dialog_history(os.path.join(path, "dialog_history.json"))

        sm_path = f"{path}/self_model.pth"
        if os.path.exists(sm_path):
            missing, unexpected = self.self_model.load_state_dict(
                torch.load(sm_path, map_location=self.device), strict=False
            )
            if missing:
                print(f"[Brain] SelfModel: новые слои без сохранённых весов (доучатся с нуля): {missing}")
            if unexpected:
                print(f"[Brain] SelfModel: неиспользуемые ключи в чекпоинте (проигнорированы): {unexpected}")

        am_path = f"{path}/auto_memory.pkl"
        if os.path.exists(am_path):
            with open(am_path, "rb") as f:
                self.auto_memory.episodes = pickle.load(f)

        um_path = f"{path}/user_model.pkl"
        if os.path.exists(um_path) and self.user_model:
            with open(um_path, "rb") as f:
                self.user_model = pickle.load(f)

        em_path = f"{path}/emotion.pkl"
        if os.path.exists(em_path) and self.emotion:
            with open(em_path, "rb") as f:
                data = pickle.load(f)
                self.emotion.valence = data.get('valence', 0.0)
                self.emotion.arousal = data.get('arousal', 0.0)
                self.emotion.dominance = data.get('dominance', 0.0)

        mot_path = f"{path}/motivation.pkl"
        if os.path.exists(mot_path) and self.motivation:
            with open(mot_path, "rb") as f:
                self.motivation.drives = pickle.load(f)

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
        stats = {
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
        if self.emotion:
            stats["emotion"] = {
                "valence": self.emotion.valence,
                "arousal": self.emotion.arousal,
                "dominance": self.emotion.dominance
            }
        if self.motivation:
            stats["drives"] = self.motivation.drives
        stats["auto_memory_length"] = len(self.auto_memory.episodes)
        return stats