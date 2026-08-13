# brain/agent.py
import time
import threading
import random
import datetime
from typing import List, Optional, Dict, Any
from openai import OpenAI

from brain import CognitiveBrain
from brain.teacher import Teacher
from brain.controller import CognitiveController
from brain.world_model import WorldModel
from brain.reasoner import Reasoner
from brain.planner import Planner
from brain.self_model import SelfModel


class BrainAgent:
    def __init__(
        self,
        brain: CognitiveBrain,
        teacher: Teacher,
        llm_client: OpenAI,
        topics: Optional[List[str]] = None,
        interval_seconds: int = 120,
        questions_per_cycle: int = 2,
        temperature: float = 0.7,
        enabled: bool = True,
        interactive_mode: bool = False,
        user_question_timeout: int = 30,
        self_play_rounds: int = 3,
        exploration_factor: float = 0.2,
        history_per_topic: int = 15,
        teacher_threshold: float = 0.7,
        use_dynamic_threshold: bool = True,
    ):
        self.brain = brain
        self.teacher = teacher
        self.llm = llm_client
        self.topics = topics or ["наука", "природа", "технологии", "история", "искусство", "философия"]
        self.interval = interval_seconds
        self.questions_per_cycle = questions_per_cycle
        self.temperature = temperature
        self.enabled = enabled
        self.interactive_mode = interactive_mode
        self.user_question_timeout = user_question_timeout
        self.self_play_rounds = self_play_rounds
        self.exploration_factor = exploration_factor
        self.history_per_topic = history_per_topic
        self.teacher_threshold = teacher_threshold
        self.use_dynamic_threshold = use_dynamic_threshold

        self._stop_flag = False
        self._thread = None

        self.pending_question = None
        self.active_question = None
        self.waiting_for_answer = False

        self.topic_confidence = {t: 0.5 for t in self.topics}
        self.asked_questions: Dict[str, List[str]] = {t: [] for t in self.topics}

        self._proactive_counter = 0
        self.accepted_count = 0
        self.improved_count = 0
        self.rejected_count = 0
        self.negative_count = 0

        # Время последнего взаимодействия для проактивности
        self.last_interaction_time = time.time()

        # Если мозг имеет контроллер – используем его
        self.controller = getattr(brain, "controller", None)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_flag = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print("[Agent] Запущен.")

    def stop(self):
        self._stop_flag = True
        if self._thread:
            self._thread.join(timeout=5)
        print("[Agent] Остановлен.")

    def _run(self):
        cycle_counter = 0
        while not self._stop_flag:
            if self.enabled:
                self._cycle()
                cycle_counter += 1

                # Проактивные мысли
                if self.brain.config.proactive_enabled:
                    self._proactive_counter += 1
                    interval_seconds = self.brain.config.proactive_interval_seconds
                    if self._proactive_counter % max(1, int(interval_seconds / 30)) == 0:
                        self.brain.proactive_thought()
                        self.brain.save()

                # Автосохранение
                if cycle_counter % 10 == 0:
                    self.brain.save()
                    self.brain.save_dialog_history()
                    print("[Agent] Автосохранение выполнено.")

                # Фоновый синтез (с учётом интереса)
                if cycle_counter % 5 == 0:
                    self._synthesize_curious_pair()

                # Периодическая статистика обучения
                if cycle_counter % 20 == 0:
                    total = self.accepted_count + self.improved_count + self.rejected_count + self.negative_count
                    if total > 0:
                        print(f"[Agent] Автообучение: принято={self.accepted_count}, улучшено={self.improved_count}, "
                              f"отклонено={self.rejected_count}, отрицательных={self.negative_count}")

                # Проактивный запуск на основе времени и Self-модели
                if cycle_counter % 3 == 0:
                    self._time_based_proactive()

            for _ in range(self.interval):
                if self._stop_flag:
                    break
                time.sleep(1)

    def _synthesize_curious_pair(self):
        """Синтез не случайный, а на основе «интереса»: ищем слабые или противоречивые связи."""
        graph = self.brain.graph
        if hasattr(graph, 'levels'):
            g = graph.levels[0]
        else:
            g = graph

        n_nodes = g.node_emb.shape[0]
        if n_nodes < 5:
            return

        # Найдём узлы с низкой степенью (неисследованные)
        degree = {nid: len(g._adjacency.get(nid, [])) for nid in range(1, n_nodes+1)}
        if not degree:
            return

        # Выберем пару: один узел с высокой степенью, другой – с низкой (любопытство)
        high_deg = sorted(degree.items(), key=lambda x: -x[1])[:10]
        low_deg = sorted(degree.items(), key=lambda x: x[1])[:10]

        if high_deg and low_deg:
            nid1 = random.choice(high_deg)[0]
            nid2 = random.choice(low_deg)[0]
        else:
            return

        # Проверим, нет ли между ними ребра
        if g.get_edges_between(nid1, nid2):
            return

        new_nid = self.brain.synthesize_concepts(nid1, nid2, optimizer=self.brain.optimizer)
        if new_nid != -1:
            print(f"[Agent] Любопытный синтез: создан узел {new_nid} из {nid1} и {nid2}")

    def _time_based_proactive(self):
        """Инициировать диалог на основе времени и Self-модели."""
        now = datetime.datetime.now()
        hour = now.hour
        # Если давно не было диалога и сейчас день
        if time.time() - self.last_interaction_time > 1800 and 8 <= hour <= 22:
            # Используем Self-модель для генерации вопроса о нерешённых вопросах
            if hasattr(self.brain, "self_model"):
                unresolved = self.brain.self_model.unresolved_questions
                if unresolved:
                    question = f"У меня есть нерешённый вопрос: {unresolved[-1]}. Хотите обсудить?"
                    self.pending_question = question
                    self.waiting_for_answer = True
                    print(f"[Agent] Проактивный вопрос (нерешённый): {question}")
                    self.last_interaction_time = time.time()
                    return

            # Иначе – вопрос о времени суток
            if 5 <= hour < 12:
                topic = "планы на день"
            elif 12 <= hour < 18:
                topic = "рабочие задачи"
            else:
                topic = "вечерний отдых"
            question = self._generate_question_for_topic(topic)
            if question:
                self.pending_question = question
                self.waiting_for_answer = True
                print(f"[Agent] Проактивный вопрос о {topic}: {question}")
                self.last_interaction_time = time.time()

    def _cycle(self):
        if self.interactive_mode:
            self._interactive_cycle()
        else:
            self._autonomous_cycle()

    def _interactive_cycle(self):
        if self.active_question is not None:
            return

        if self.brain.motivation is not None:
            actions = [
                {'name': 'ask_user', 'expected_impact': {'social': 0.7, 'curiosity': 0.3}},
                {'name': 'search_web', 'expected_impact': {'curiosity': 0.9, 'mastery': 0.4}},
                {'name': 'propose_topic', 'expected_impact': {'social': 0.5, 'coherence': 0.4}},
            ]
            chosen = self.brain.motivation.select_action(actions)
            if chosen and chosen['name'] == 'search_web':
                query = self._generate_search_query()
                if query:
                    self._ask_controller(query, use_search=True)
                return
            elif chosen and chosen['name'] == 'propose_topic':
                topic = self._select_topic_for_exploration()
                print(f"[Agent] Предлагаю обсудить тему: {topic}")
                question = self._generate_question_for_topic(topic)
                if question:
                    self.pending_question = question
                    self.waiting_for_answer = True
                    print(f"[Agent] Вопрос для пользователя: {question} (тема: {topic})")
                    start = time.time()
                    while self.waiting_for_answer and (time.time() - start) < self.user_question_timeout:
                        if self._stop_flag:
                            break
                        time.sleep(1)
                    if self.pending_question == question:
                        self.pending_question = None
                        self.waiting_for_answer = False
                    elif self.active_question == question:
                        self.active_question = None
                        self.waiting_for_answer = False
                return

        topic = self._select_topic_for_exploration()
        question = self._generate_question_for_topic(topic)
        if not question:
            return
        self.pending_question = question
        self.waiting_for_answer = True
        print(f"[Agent] Вопрос для пользователя: {question} (тема: {topic})")

        start = time.time()
        while self.waiting_for_answer and (time.time() - start) < self.user_question_timeout:
            if self._stop_flag:
                break
            time.sleep(1)

        if self.pending_question == question:
            self.pending_question = None
            self.waiting_for_answer = False
            print(f"[Agent] Таймаут ответа на вопрос: {question}")
        elif self.active_question == question:
            self.active_question = None
            self.waiting_for_answer = False
            print(f"[Agent] Ответ получен на вопрос: {question}")

    def _autonomous_cycle(self):
        if self.brain.motivation is not None:
            actions = [
                {'name': 'study_topic', 'expected_impact': {'mastery': 0.8, 'coherence': 0.6}},
                {'name': 'explore_new', 'expected_impact': {'curiosity': 0.9, 'mastery': 0.3}},
                {'name': 'search_web', 'expected_impact': {'curiosity': 0.7, 'social': 0.1}},
            ]
            chosen = self.brain.motivation.select_action(actions)
            if chosen:
                if chosen['name'] == 'search_web':
                    query = self._generate_search_query()
                    if query:
                        self._ask_controller(query, use_search=True)
                    return
                elif chosen['name'] == 'explore_new':
                    topic = min(self.topic_confidence, key=self.topic_confidence.get)
                    self._ask_and_learn(topic)
                    return
                elif chosen['name'] == 'study_topic':
                    if self.topics:
                        topic = max(self.topic_confidence, key=self.topic_confidence.get)
                        self._ask_and_learn(topic)
                    return

        topic = self._select_topic_for_exploration()
        for _ in range(self.self_play_rounds):
            q = self._generate_question_for_topic(topic)
            if not q:
                break
            self._ask_and_learn(topic)

    def _ask_controller(self, question: str, use_search: bool = False):
        """Используем контроллер для обработки вопроса."""
        if self.controller:
            answer = self.controller.process_input(question)
        else:
            result = self.brain.step(question, use_search=use_search)
            answer = result["answer"]
        # Сохраняем в диалог
        self.brain.dialog_memory.append({"user": question, "assistant": answer, "time": time.time()})
        self.last_interaction_time = time.time()

    def _ask_and_learn(self, topic: str):
        q = self._generate_question_for_topic(topic)
        if not q:
            return
        if self.controller:
            answer = self.controller.process_input(q)
            thoughts = answer
            confidence = 0.5  # можно получить из контроллера, но пока упростим
        else:
            result = self.brain.step(q)
            thoughts = result.get("thoughts", result["answer"])
            confidence = result.get("confidence", 0.5)

        score, improved, _ = self.teacher.evaluate(q, thoughts)

        if self.use_dynamic_threshold:
            dynamic_threshold = 0.5 + 0.25 * (1 - confidence)
            threshold = dynamic_threshold
        else:
            threshold = self.teacher_threshold

        if score >= threshold:
            self.brain.learn_pair(q, thoughts, reward=score)
            self._update_topic_confidence(topic, score)
            self.accepted_count += 1
        else:
            if improved != thoughts and score > 0.3:
                self.brain.learn_pair(q, improved, reward=max(0.5, score + 0.1))
                self._update_topic_confidence(topic, max(0.5, score + 0.1))
                self.improved_count += 1
            else:
                if score < 0.3:
                    self.brain.learn_negative_pair(q, thoughts)
                    self.negative_count += 1
                else:
                    self.rejected_count += 1
        self.last_interaction_time = time.time()

    def _select_topic_for_exploration(self) -> str:
        if self.brain.user_model is not None:
            interests = self.brain.user_model.interests
            if interests:
                weighted = []
                for t in self.topics:
                    interest = interests.get(t, 0.0)
                    weight = interest + random.uniform(0, 0.3)
                    weighted.append((weight, t))
                weighted.sort(reverse=True)
                return weighted[0][1]

        if random.random() < self.exploration_factor:
            return random.choice(self.topics)
        return min(self.topic_confidence, key=self.topic_confidence.get)

    def _update_topic_confidence(self, topic: str, score: float):
        if topic in self.topic_confidence:
            self.topic_confidence[topic] = 0.8 * self.topic_confidence[topic] + 0.2 * score

    def _generate_question_for_topic(self, topic: str) -> Optional[str]:
        recent = self.asked_questions.setdefault(topic, [])[-self.history_per_topic:]
        avoid_block = "\n".join(f"- {q}" for q in recent) if recent else "(пока нет)"
        try:
            temp = self.temperature
            if self.brain.emotion is not None:
                temp = self.brain.emotion.modulate_temperature(temp)
            response = self.llm.chat.completions.create(
                model="local-model",
                messages=[
                    {"role": "system", "content": (
                        "Ты — исследовательский агент. Придумай ОДИН новый, содержательный вопрос "
                        "по заданной теме. Вопрос НЕ должен повторять и не должен быть перефразировкой "
                        "уже заданных вопросов из списка ниже."
                    )},
                    {"role": "user", "content": (
                        f"Тема: {topic}.\nУже заданные вопросы по этой теме:\n{avoid_block}\n"
                        "Сформулируй один новый вопрос (только вопрос, без пояснений и нумерации)."
                    )}
                ],
                max_tokens=40,
                temperature=max(temp, 0.7),
            )
            q = response.choices[0].message.content.strip()
            if q and "?" in q and q not in recent:
                self.asked_questions[topic].append(q)
                if len(self.asked_questions[topic]) > self.history_per_topic * 4:
                    self.asked_questions[topic] = self.asked_questions[topic][-self.history_per_topic * 2:]
                return q
            return None
        except Exception as e:
            print(f"[Agent] Ошибка генерации вопроса: {e}")
            return None

    def _generate_search_query(self) -> Optional[str]:
        if self.brain.user_model is not None and self.brain.user_model.interests:
            topic = max(self.brain.user_model.interests, key=self.brain.user_model.interests.get)
        else:
            topic = random.choice(self.topics)
        try:
            response = self.llm.chat.completions.create(
                model="local-model",
                messages=[
                    {"role": "system", "content": "Ты — помощник. Сформулируй короткий поисковый запрос по теме."},
                    {"role": "user", "content": f"Тема: {topic}. Сформулируй один поисковый запрос (5-10 слов)."}
                ],
                max_tokens=20,
                temperature=0.5,
            )
            query = response.choices[0].message.content.strip()
            return query if query else None
        except Exception:
            return None

    def get_next_question(self) -> Optional[str]:
        if self.pending_question is not None:
            q = self.pending_question
            self.active_question = q
            self.pending_question = None
            return q
        return None

    def submit_answer(self, question: str, answer: str):
        if self.active_question == question:
            # Используем контроллер для обработки ответа пользователя
            if self.controller:
                self.controller.process_input(answer)  # ответ пользователя как новый ввод
            else:
                self.brain.learn_pair(question, answer)
            self.brain.save()
            self.brain.save_dialog_history()
            print(f"[Agent] Пользователь ответил на '{question}' -> '{answer}', выучено.")
            self.active_question = None
            self.waiting_for_answer = False
            self.brain.dialog_memory.append({"user": question, "assistant": answer, "time": time.time()})
            self.last_interaction_time = time.time()
        else:
            print(f"[Agent] Ответ на неактивный вопрос: {question} (активный: {self.active_question})")