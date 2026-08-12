# agent.py
import time
import threading
import random
from typing import List, Optional, Dict, Any
from openai import OpenAI

from brain import CognitiveBrain
from brain.teacher import Teacher


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
        # НОВЫЕ ПАРАМЕТРЫ ДЛЯ АВТООБУЧЕНИЯ
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
        # Счётчики для статистики автообучения
        self.accepted_count = 0
        self.improved_count = 0
        self.rejected_count = 0
        self.negative_count = 0

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

                # --- ПРОАКТИВНЫЕ МЫСЛИ ---
                if self.brain.config.proactive_enabled:
                    self._proactive_counter += 1
                    interval_seconds = self.brain.config.proactive_interval_seconds
                    if self._proactive_counter % max(1, int(interval_seconds / 30)) == 0:
                        self.brain.proactive_thought()
                        self.brain.save()

                # --- АВТОСОХРАНЕНИЕ ---
                if cycle_counter % 10 == 0:
                    self.brain.save()
                    self.brain.save_dialog_history()
                    print("[Agent] Автосохранение выполнено.")

                # --- ФОНОВЫЙ СИНТЕЗ НОВЫХ УЗЛОВ ---
                if cycle_counter % 5 == 0:
                    self._synthesize_random_pair()

                # --- ПЕРИОДИЧЕСКАЯ СТАТИСТИКА АВТООБУЧЕНИЯ ---
                if cycle_counter % 20 == 0:
                    total = self.accepted_count + self.improved_count + self.rejected_count + self.negative_count
                    if total > 0:
                        print(f"[Agent] Автообучение: принято={self.accepted_count}, улучшено={self.improved_count}, "
                              f"отклонено={self.rejected_count}, отрицательных={self.negative_count}")

            for _ in range(self.interval):
                if self._stop_flag:
                    break
                time.sleep(1)

    def _synthesize_random_pair(self):
        graph = self.brain.graph
        if hasattr(graph, 'levels'):
            g = graph.levels[0]
        else:
            g = graph
        n_nodes = g.node_emb.shape[0]
        if n_nodes < 5:
            return
        degrees = {nid: len(g._adjacency.get(nid, [])) for nid in range(1, n_nodes+1)}
        if not degrees:
            return
        sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)[:20]
        if len(sorted_nodes) < 2:
            return
        nid1, nid2 = random.sample(sorted_nodes, 2)
        new_nid = self.brain.synthesize_concepts(nid1, nid2, optimizer=self.brain.optimizer)
        if new_nid != -1:
            print(f"[Agent] Фоновый синтез: создан узел {new_nid} из {nid1} и {nid2}")

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
                    self.brain.step(query, use_search=True)
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
                        self.brain.step(query, use_search=True)
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

    def _ask_and_learn(self, topic: str):
        q = self._generate_question_for_topic(topic)
        if not q:
            return
        result = self.brain.step(q)
        thoughts = result.get("thoughts", result["answer"])
        confidence = result.get("confidence", 0.5)

        # ----- НОВАЯ ЛОГИКА ОЦЕНКИ -----
        score, improved, _ = self.teacher.evaluate(q, thoughts)

        # Динамический порог
        if self.use_dynamic_threshold:
            # Чем выше уверенность графа, тем ниже порог
            dynamic_threshold = 0.5 + 0.25 * (1 - confidence)  # от 0.5 до 0.75
            threshold = dynamic_threshold
        else:
            threshold = self.teacher_threshold

        # Принятие решения
        if score >= threshold:
            # Хороший ответ – обучаем
            self.brain.learn_pair(q, thoughts, reward=score)
            self._update_topic_confidence(topic, score)
            self.accepted_count += 1
        else:
            # Пытаемся улучшить
            if improved != thoughts and score > 0.3:
                # Есть улучшенный вариант и оценка не совсем провальная
                self.brain.learn_pair(q, improved, reward=max(0.5, score + 0.1))
                self._update_topic_confidence(topic, max(0.5, score + 0.1))
                self.improved_count += 1
            else:
                # Плохо – либо отрицательное обучение, либо пропуск
                if score < 0.3:
                    self.brain.learn_negative_pair(q, thoughts)
                    self.negative_count += 1
                else:
                    # Пропускаем (средний ответ без улучшения)
                    self.rejected_count += 1

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
            self.brain.learn_pair(question, answer)
            self.brain.save()
            self.brain.save_dialog_history()
            print(f"[Agent] Пользователь ответил на '{question}' -> '{answer}', выучено.")
            self.active_question = None
            self.waiting_for_answer = False
            self.brain.dialog_memory.append({"user": question, "assistant": answer, "time": time.time()})
        else:
            print(f"[Agent] Ответ на неактивный вопрос: {question} (активный: {self.active_question})")


if __name__ == "__main__":
    import signal
    import sys
    from brain import BrainConfig

    config = BrainConfig()
    brain = CognitiveBrain(config)
    brain.load()
    brain.load_dialog_history()

    llm_client = OpenAI(base_url=config.llm_base_url, api_key="not-needed")
    teacher = Teacher(llm_client=llm_client)

    agent = BrainAgent(
        brain=brain,
        teacher=teacher,
        llm_client=llm_client,
        interactive_mode=False,
        teacher_threshold=0.7,
        use_dynamic_threshold=True,
    )

    def _shutdown():
        print("\n[Agent] Остановка и сохранение...")
        agent.stop()
        brain.save()
        brain.save_dialog_history()

    signal.signal(signal.SIGINT, lambda s, f: (_shutdown(), sys.exit(0)))
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda s, f: (_shutdown(), sys.exit(0)))

    agent.start()
    print("[Agent] Работает в фоне. Ctrl+C для остановки.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        _shutdown()