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
import requests                          # ===== НОВОЕ: для API =====
from collections import deque
from typing import List, Dict, Optional, Any
from brain.search import WebSearcher
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
        self.searcher = WebSearcher(max_results=10)
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
        return self.embedder.get_embedding(text).float().to(self.device)

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

        # Получаем или создаём нейроны (как раньше)
        q_nid = self.graph.find_most_similar(query_vec, threshold=0.7)
        a_nid = self.graph.find_most_similar(answer_vec, threshold=0.7)
        if q_nid is None:
            q_nid = self.graph.add_node(query_vec, label=input_text[:30], cluster="concept", layer=0)
            norm_q = self.normalize_text(input_text)
            self.concept_index[norm_q] = q_nid
        if a_nid is None:
            a_nid = self.graph.add_node(answer_vec, label=answer_text[:30], cluster="concept", layer=0)
            norm_a = self.normalize_text(answer_text)
            self.concept_index[norm_a] = a_nid

        # Добавляем синапс (вес положительный)
        if q_nid is not None and a_nid is not None:
            self.graph.add_synapse(q_nid, a_nid, weight=0.2)

        # ---- НОВАЯ ЧАСТЬ: обучение через GAT + регуляризация ----
        loss = self._update_graph_and_loss(q_nid, a_nid, input_text, answer_text)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Добавляем в knowledge_base и память (как раньше)
        self._add_to_knowledge_base(input_text, answer_text)
        self.memory.add_episodic(query_vec + answer_vec, {"q": input_text, "a": answer_text, "reward": reward})
        if self.step_counter % 50 == 0:
            self.memory.consolidate()

    def _add_to_knowledge_base(self, q: str, a: str):
        for item in self.knowledge_base:
            if item["q"] == q and item["a"] == a:
                return
        emb = self.text_to_embedding(q + " " + a)
        self.knowledge_base.append({
            "q": q,
            "a": a,
            "emb": emb,
            "time": time.time(),
            "confidence": 0.5,
            "access_count": 0
        })
        if len(self.knowledge_base) > self.config.max_kb_size:
            self.knowledge_base.pop(0)

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
            self.memory.consolidate()
            self.memory.working.clear()
            print("😴 Сон завершён")

    # ---------- Сохранение / загрузка ----------
    def save(self, model_dir: str = None):
        path = model_dir or self.config.model_dir
        os.makedirs(path, exist_ok=True)
        torch.save(self.graph.state_dict(), f"{path}/graph.pth")

        with open(f"{path}/edges.pkl", "wb") as f:
            pickle.dump(self.graph._edges, f)

        kb_serializable = []
        for item in self.knowledge_base:
            kb_serializable.append({
                "q": item["q"],
                "a": item["a"],
                "time": item.get("time", 0),
                "confidence": item.get("confidence", 0.5),
                "access_count": item.get("access_count", 0)
            })
        meta = {
            "step_counter": self.step_counter,
            "learn_counter": self._learn_counter,
            "concept_index": self.concept_index,
            "knowledge_base": kb_serializable,
            "num_neurons": self.graph.node_emb.shape[0],
            "num_edges": len(self.graph._edges),
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
        meta_path = f"{path}/meta.json"
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            num_neurons = meta.get("num_neurons", 10)
            current_num = self.graph.node_emb.shape[0]
            if current_num < num_neurons:
                for _ in range(num_neurons - current_num):
                    emb = random_vector(self.dim)
                    self.graph.add_node(emb, label="loaded", cluster="hidden", layer=0)
                print(f"[Brain] Добавлено {num_neurons - current_num} нейронов для загрузки.")
            graph_path = f"{path}/graph.pth"
            if os.path.exists(graph_path):
                state_dict = torch.load(graph_path, map_location=self.device)
                model_state = self.graph.state_dict()
                filtered_state = {}
                for key, value in state_dict.items():
                    if key in model_state and model_state[key].shape == value.shape:
                        filtered_state[key] = value
                    else:
                        print(f"[Brain] Пропуск загрузки {key}: размер {value.shape} -> {model_state[key].shape if key in model_state else 'not found'}")
                self.graph.load_state_dict(filtered_state, strict=False)
            edges_path = f"{path}/edges.pkl"
            if os.path.exists(edges_path):
                with open(edges_path, "rb") as f:
                    self.graph._edges = pickle.load(f)
                print(f"[Brain] Загружено {len(self.graph._edges)} синапсов.")
            else:
                self.graph._edges = []
            self.step_counter = meta.get("step_counter", 0)
            self._learn_counter = meta.get("learn_counter", 0)
            self.concept_index = meta.get("concept_index", {})
            self.knowledge_base = meta.get("knowledge_base", [])
        self.load_dialog_history(os.path.join(path, "dialog_history.json"))
        print(f"[Brain] Модель загружена из {path}")

    # =========================================================================
    # НОВЫЕ ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # =========================================================================

    # ----- Извлечение цены из текста (парсинг) -----
    @staticmethod
    def extract_price(text: str) -> Optional[float]:
        patterns = [
            r'(\d{1,3}(?:[ ,]\d{3})*(?:\.\d+)?)\s*[$€£]',   # 67,000 $
            r'[$€£]\s*(\d{1,3}(?:[ ,]\d{3})*(?:\.\d+)?)',    # $ 67,000
            r'(\d+)\s*(?:USD|BTC|USDT|RUB|руб)',             # 67000 USD
            r'(\d+\.\d+)\s*[$€£]',                           # 123.45 $
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                raw = m.group(1).replace(' ', '').replace(',', '')
                try:
                    return float(raw)
                except ValueError:
                    continue
        return None

    # ----- Запрос курса криптовалют через CoinGecko API -----
    @staticmethod
    def get_crypto_price(crypto_id: str = "bitcoin", vs_currency: str = "usd") -> Optional[float]:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={crypto_id}&vs_currencies={vs_currency}"
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return data.get(crypto_id, {}).get(vs_currency)
        except Exception as e:
            print(f"[API] Ошибка запроса к CoinGecko: {e}")
        return None

    # ----- Поиск фактов в knowledge_base по сходству -----
    def search_knowledge_base(self, query: str, top_k: int = 5) -> List[str]:
        """
        Возвращает строки фактов из knowledge_base, релевантных запросу.
        """
        results = []
        if not self.knowledge_base:
            return results
        query_vec = self.text_to_embedding(query)
        scored = []
        for item in self.knowledge_base:
            emb = item.get("emb")
            if emb is None:
                # если эмбеддинг не сохранён, вычисляем
                emb = self.text_to_embedding(item["q"] + " " + item["a"])
                item["emb"] = emb
            sim = cosine_similarity(query_vec, emb)
            scored.append((sim, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        threshold = 0.3
        for sim, item in scored[:top_k]:
            if sim >= threshold:
                results.append(f"Q: {item['q']} -> A: {item['a']}")
        return results

    # =========================================================================
    # ОСНОВНОЙ МЕТОД step (с изменениями)
    # =========================================================================

    def step(self, input_text: str, use_search: bool = False) -> Dict[str, Any]:
        self.step_counter += 1
        lower = input_text.lower()

        # ===== НОВОЕ: обработка запросов о криптовалютах через API (до поиска) =====
        crypto_keywords = ["биткоин", "btc", "bitcoin", "курс биткоина", "цена биткоина"]
        if any(kw in lower for kw in crypto_keywords):
            price = self.get_crypto_price("bitcoin", "usd")
            if price is not None:
                answer = f"Текущий курс биткоина (BTC/USD): ${price:.2f}"
                return {
                    "input": input_text,
                    "answer": answer,
                    "activated_neurons": [],
                    "memory_results": []
                }
            # если API не ответил – идём дальше, возможно поиск или LLM

        # ---- 0. ПРИНУДИТЕЛЬНЫЙ ПОИСК (если включена кнопка 🌐) ----
        if use_search:
            enhanced_query = self._enhance_search_query(input_text)
            results = self.searcher.search(enhanced_query)
            if results:
                # ---- НОВОЕ: пытаемся извлечь цену из сниппетов ----
                full_text = " ".join([r.get("title", "") + " " + r.get("body", "") for r in results])
                price = self.extract_price(full_text)
                if price is not None:
                    answer = f"Курс биткоина (по данным поиска): ${price:.2f}"
                    return {
                        "input": input_text,
                        "answer": answer,
                        "activated_neurons": [],
                        "memory_results": []
                    }

                # Если цена не найдена, передаём результаты в LLM
                context = (
                    f"Вопрос: {input_text}\n"
                    f"Улучшенный запрос: {enhanced_query}\n"
                    f"Результаты поиска в интернете:\n{self.searcher.format_results(results)}\n"
                    "Важно: используй информацию из результатов поиска как достоверный источник. "
                    "Если в них есть конкретные цифры, не сомневайся в них. Если данных нет, скажи: 'Я не знаю'."
                )
                answer = self.llm.generate(context, max_tokens=256, temperature=0.3)
            else:
                answer = "Не удалось найти информацию по вашему запросу."
            return {
                "input": input_text,
                "answer": answer,
                "activated_neurons": [],
                "memory_results": []
            }

        # ---- 1. БЛОК ВОПРОСОВ О ПАМЯТИ / ОБУЧЕНИИ (без изменений) ----
        memory_keywords = [
            "обучил", "запомнил", "факты", "выучил", "узнал",
            "какие данные", "чему научился", "что ты знаешь",
            "твои знания", "что ты помнишь", "что запомнил",
            "какие факты", "чему обучился", "какую тему"
        ]
        if any(phrase in lower for phrase in memory_keywords):
            if self.knowledge_base:
                sorted_kb = sorted(self.knowledge_base, key=lambda x: x.get('time', 0), reverse=True)
                facts = []
                for item in sorted_kb[:10]:
                    facts.append(f"  • {item['q']} → {item['a']}")
                answer = "Я запомнил следующие факты (последние 10):\n" + "\n".join(facts)
            elif self.concept_index:
                concepts = list(self.concept_index.keys())[:20]
                answer = "Я знаю понятия: " + ", ".join(concepts)
            else:
                answer = "Я пока ничего не выучил в этом сеансе."
            return {
                "input": input_text,
                "answer": answer,
                "activated_neurons": [],
                "memory_results": []
            }

        # ---- 2. БЛОК ВОПРОСОВ ОБ АРХИТЕКТУРЕ (без изменений) ----
        architecture_keywords = [
            "нейроны", "синапсы", "архитектура", "ассоциативная память",
            "как ты устроен", "граф", "gnn", "нейросеть", "обучаешься",
            "как работает твоя память"
        ]
        if any(phrase in lower for phrase in architecture_keywords):
            answer = (
                "Я — ассоциативная нейросеть с дифференцируемым графом (GNN).\n"
                "• Мои нейроны — это обучаемые эмбеддинги (векторы).\n"
                "• Синапсы — это рёбра графа, которые соединяют нейроны.\n"
                "• Я обучаюсь через контрастную потерю — градиенты обновляют эмбеддинги нейронов и веса связей.\n"
                "• У меня есть иерархическая память: рабочая (краткосрочная) и эпизодическая (FAISS).\n"
                "• Знания сохраняются в нейронах и синапсах, а также в базе знаний (knowledge_base)."
            )
            return {
                "input": input_text,
                "answer": answer,
                "activated_neurons": [],
                "memory_results": []
            }

        # ---- 3. АВТОМАТИЧЕСКИЙ ПОИСК ПО КЛЮЧЕВЫМ СЛОВАМ (без изменений) ----
        search_triggers = ["поищи", "найди", "актуальный", "сегодня", "новости", "интернет", "в сети"]
        if any(phrase in lower for phrase in search_triggers):
            enhanced_query = self._enhance_search_query(input_text)
            results = self.searcher.search(enhanced_query)
            if results:
                context = (
                    f"Вопрос: {input_text}\n"
                    f"Улучшенный запрос: {enhanced_query}\n"
                    f"Результаты поиска в интернете:\n{self.searcher.format_results(results)}\n"
                    "Важно: используй информацию из результатов поиска как достоверный источник. "
                    "Если в них есть конкретные цифры, не сомневайся в них. Если данных нет, скажи: 'Я не знаю'."
                )
                answer = self.llm.generate(context, max_tokens=256, temperature=0.3)
            else:
                answer = "Не удалось найти информацию по вашему запросу."
            return {
                "input": input_text,
                "answer": answer,
                "activated_neurons": [],
                "memory_results": []
            }

        # ---- 4. ОБЫЧНЫЙ ОБРАБОТЧИК (граф + FAISS + LLM) С ДОБАВЛЕНИЕМ KNOWLEDGE_BASE ----
        query_vec = self.text_to_embedding(input_text)
        memory_results = self.memory.retrieve(query_vec, k=3)

        start_nid = self.graph.find_most_similar(query_vec, threshold=0.5)
        if start_nid is None:
            start_nid = self.graph.add_node(query_vec, label=input_text[:30], cluster="input", layer=0)

       

        # Формируем контекст
        context = f"Вопрос: {input_text}\n"

        # ===== НОВОЕ: добавляем факты из knowledge_base =====
        kb_facts = self.search_knowledge_base(input_text, top_k=3)
        if kb_facts:
            context += "Из моей базы знаний:\n" + "\n".join(kb_facts) + "\n"

        if memory_results:
            context += "Из памяти:\n"
            for res in memory_results:
                context += f"- {res['metadata'].get('text', '')}\n"

        label = self.graph.neuron_labels.get(start_nid, "")
        context += f"Ассоциация: {label}\n"
        context += "\nВажно: не выдумывай факты. Если в памяти нет достоверной информации, скажи: 'Я не знаю'."

        # ===== НОВОЕ: логирование контекста (для отладки) =====
        print("\n" + "="*50)
        print("🧠 КОНТЕКСТ ДЛЯ LLM:")
        print(context)
        print("="*50 + "\n")

        answer = self.llm.generate(context, max_tokens=256, temperature=0.7)
        self.memory.add_working(query_vec, {"text": input_text, "answer": answer})

        return {
            "input": input_text,
            "answer": answer,
            "activated_neurons": [start_nid],
            "memory_results": memory_results
        }

    # ---------- step_stream (без изменений, кроме добавления KB – опционально) ----------
    def step_stream(self, input_text: str, use_search: bool = False):
        lower = input_text.lower()

        # ---- ПРОВЕРКА КОМАНД (память, архитектура) ----
        memory_keywords = [
            "обучил", "запомнил", "факты", "выучил", "узнал",
            "какие данные", "чему научился", "что ты знаешь",
            "твои знания", "что ты помнишь", "что запомнил",
            "какие факты", "чему обучился", "какую тему"
        ]
        if any(phrase in lower for phrase in memory_keywords):
            if self.knowledge_base:
                sorted_kb = sorted(self.knowledge_base, key=lambda x: x.get('time', 0), reverse=True)
                facts = []
                for item in sorted_kb[:10]:
                    facts.append(f"  • {item['q']} → {item['a']}")
                answer = "Я запомнил следующие факты (последние 10):\n" + "\n".join(facts)
            elif self.concept_index:
                concepts = list(self.concept_index.keys())[:20]
                answer = "Я знаю понятия: " + ", ".join(concepts)
            else:
                answer = "Я пока ничего не выучил в этом сеансе."
            yield answer
            return

        architecture_keywords = [
            "нейроны", "синапсы", "архитектура", "ассоциативная память",
            "как ты устроен", "граф", "gnn", "нейросеть", "обучаешься",
            "как работает твоя память"
        ]
        if any(phrase in lower for phrase in architecture_keywords):
            answer = (
                "Я — ассоциативная нейросеть с дифференцируемым графом (GNN).\n"
                "• Мои нейроны — это обучаемые эмбеддинги (векторы).\n"
                "• Синапсы — это рёбра графа, которые соединяют нейроны.\n"
                "• Я обучаюсь через контрастную потерю — градиенты обновляют эмбеддинги нейронов и веса связей.\n"
                "• У меня есть иерархическая память: рабочая (краткосрочная) и эпизодическая (FAISS).\n"
                "• Знания сохраняются в нейронах и синапсах, а также в базе знаний (knowledge_base)."
            )
            yield answer
            return

        # ---- ОСНОВНАЯ ЛОГИКА (поиск + граф + LLM) ----
        enhanced_query = None
        results = None
        if use_search:
            enhanced_query = self._enhance_search_query(input_text)
            results = self.searcher.search(enhanced_query)

        query_vec = self.text_to_embedding(input_text)
        memory_results = self.memory.retrieve(query_vec, k=3)

        start_nid = self.graph.find_most_similar(query_vec, threshold=0.5)
        if start_nid is None:
            start_nid = self.graph.add_node(query_vec, label=input_text[:30], cluster="input", layer=0)

        self.forward(query_vec)

        context = f"Вопрос: {input_text}\n"

        # ===== НОВОЕ: добавляем факты из knowledge_base и в stream =====
        kb_facts = self.search_knowledge_base(input_text, top_k=3)
        if kb_facts:
            context += "Из моей базы знаний:\n" + "\n".join(kb_facts) + "\n"

        if use_search and results:
            context += f"Улучшенный запрос: {enhanced_query}\n"
            context += f"Результаты поиска в интернете:\n{self.searcher.format_results(results)}\n"
            context += "Важно: используй информацию из результатов поиска как достоверный источник. "
            context += "Если в них есть конкретные цифры, не сомневайся в них. Если данных нет, скажи: 'Я не знаю'.\n"
        else:
            if memory_results:
                context += "Из памяти:\n"
                for res in memory_results:
                    context += f"- {res['metadata'].get('text', '')}\n"
            label = self.graph.neuron_labels.get(start_nid, "")
            context += f"Ассоциация: {label}\n"
            context += "Важно: не выдумывай факты. Если в памяти нет достоверной информации, скажи: 'Я не знаю'.\n"

        # ===== НОВОЕ: логирование (опционально) =====
        print("\n" + "="*50)
        print("🧠 КОНТЕКСТ ДЛЯ LLM (stream):")
        print(context)
        print("="*50 + "\n")

        for token in self.llm.generate_stream(
                context,
                max_tokens=256,
                temperature=0.7
        ):
            yield token

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

    def _update_graph_and_loss(self, q_nid: int, a_nid: int, input_text: str, answer_text: str) -> torch.Tensor:
        # 1. Пропускаем все эмбеддинги через GAT – получаем обновлённые
        updated_emb = self.graph.forward()  # без аргументов, использует self.graph.node_emb

        # 2. Берём эмбеддинги нужных узлов из обновлённого тензора (НЕ из node_emb!)
        emb_q = updated_emb[q_nid - 1]  # индексы с 0
        emb_a = updated_emb[a_nid - 1]

        # 3. Контрастная потеря (максимизируем косинусное сходство)
        emb_q = F.normalize(emb_q.unsqueeze(0), p=2, dim=1)
        emb_a = F.normalize(emb_a.unsqueeze(0), p=2, dim=1)
        sim = F.cosine_similarity(emb_q, emb_a, dim=1)
        contrast_loss = -torch.log(torch.sigmoid(sim * 10.0)).mean()

        # 4. Регуляризация весов синапсов (L2)
        edge_weights = self.graph.get_edge_weights()
        if edge_weights.numel() > 0:
            reg_loss = 1e-4 * torch.norm(edge_weights, p=2)
        else:
            reg_loss = torch.tensor(0.0, device=self.device)

        total_loss = contrast_loss + reg_loss
        return total_loss

    # ---------- Дополнительные утилиты ----------
    def normalize_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text.strip().lower())

    def _enhance_search_query(self, query: str) -> str:
        try:
            today = time.strftime("%d.%m.%Y")
            lower = query.lower()

            currency_pairs = {
                "доллар": "USD/RUB",
                "евро": "EUR/RUB",
                "юань": "CNY/RUB",
                "йена": "JPY/RUB",
                "ена": "JPY/RUB",
                "фунт": "GBP/RUB",
            }
            pair = None
            for key, val in currency_pairs.items():
                if key in lower:
                    pair = val
                    break

            if pair:
                if "актуальн" in lower or "сегодня" in lower or "сейчас" in lower:
                    return f"курс {pair} на {today}"
                else:
                    return f"курс {pair}"

            if "погода" in lower:
                return f"{query} {today}"
            if "новости" in lower:
                return f"{query} за {today}"

            if "?" in query and "сегодня" not in lower:
                return f"{query} сегодня"

            return query
        except Exception as e:
            print(f"[Brain] Ошибка улучшения запроса: {e}")
            return query

    def text_to_signal(self, text: str, energy: float = 1.0):
        class FakeSignal:
            def __init__(self, emb, energy, text):
                self.embedding = emb
                self.energy = energy
                self.text = text
        return FakeSignal(self.text_to_embedding(text), energy, text)