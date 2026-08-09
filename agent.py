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

        self._stop_flag = False
        self._thread = None

        self.pending_question = None
        self.active_question = None
        self.waiting_for_answer = False

        self.topic_confidence = {t: 0.5 for t in self.topics}
        self.asked_questions: Dict[str, List[str]] = {t: [] for t in self.topics}

        # Счётчик для проактивных мыслей
        self._proactive_counter = 0

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

                # --- ПРОАКТИВНЫЕ МЫСЛИ (каждые N циклов) ---
                if self.brain.config.proactive_enabled:
                    self._proactive_counter += 1
                    # Вызываем proactive_thought примерно каждые proactive_interval_seconds
                    # Например, раз в 30 секунд (интервал цикла * 0.25)
                    if self._proactive_counter % max(1, int(self.interval / 30)) == 0:
                        self.brain.proactive_thought()
                        self.brain.save()  # сохраняем после мыслей

                if cycle_counter % 10 == 0:
                    self.brain.save()
                    self.brain.save_dialog_history()
                    print("[Agent] Автосохранение выполнено.")

            for _ in range(self.interval):
                if self._stop_flag:
                    break
                time.sleep(1)

    def _cycle(self):
        if self.interactive_mode:
            self._interactive_cycle()
        else:
            self._autonomous_cycle()

    def _interactive_cycle(self):
        if self.active_question is not None:
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
        topic = self._select_topic_for_exploration()
        for _ in range(self.self_play_rounds):
            q = self._generate_question_for_topic(topic)
            if not q:
                break
            result = self.brain.step(q)
            # result содержит "answer" (краткий) и "thoughts" (полный)
            answer = result["answer"]
            thoughts = result.get("thoughts", answer)  # для обучения используем полный ответ
            score, improved, _ = self.teacher.evaluate(q, thoughts)
            if score >= 0.7:
                self.brain.learn_pair(q, thoughts, reward=score)
                self._update_topic_confidence(topic, score)
            else:
                if improved != thoughts:
                    self.brain.learn_pair(q, improved, reward=0.8)
                    self._update_topic_confidence(topic, 0.8)
                else:
                    self.brain.learn_negative_pair(q, thoughts)
                    search_result = self.brain.step(q, use_search=True)
                    if search_result["answer"] != "Не удалось найти информацию.":
                        self.brain.learn_pair(q, search_result["thoughts"], reward=0.6)
            time.sleep(0.5)
        self.brain.sleep()

    def _select_topic_for_exploration(self) -> str:
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
                temperature=max(self.temperature, 0.9),
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