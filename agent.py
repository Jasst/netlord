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

        self._stop_flag = False
        self._thread = None

        self.pending_question = None
        self.active_question = None
        self.waiting_for_answer = False

        # Для отслеживания тем с низкой уверенностью
        self.topic_confidence = {t: 0.5 for t in self.topics}

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
        # Выбираем тему с низкой уверенностью для исследования
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
        # Самообучение через self-play
        topic = self._select_topic_for_exploration()
        for _ in range(self.self_play_rounds):
            q = self._generate_question_for_topic(topic)
            if not q:
                break
            result = self.brain.step(q)
            answer = result["answer"]
            score, improved, _ = self.teacher.evaluate(q, answer)
            if score >= 0.7:
                self.brain.learn_pair(q, answer, reward=score)
                self._update_topic_confidence(topic, score)
            else:
                if improved != answer:
                    self.brain.learn_pair(q, improved, reward=0.8)
                    self._update_topic_confidence(topic, 0.8)
                else:
                    self.brain.learn_negative_pair(q, answer)
                    # ищем альтернативный ответ через поиск
                    search_result = self.brain.step(q, use_search=True)
                    if search_result["answer"] != "Не удалось найти информацию.":
                        self.brain.learn_pair(q, search_result["answer"], reward=0.6)
            time.sleep(0.5)
        # после цикла консолидация
        self.brain.sleep()

    def _select_topic_for_exploration(self) -> str:
        # Выбираем тему с наименьшей уверенностью + случайность
        if random.random() < self.exploration_factor:
            return random.choice(self.topics)
        # иначе выбираем тему с минимальной уверенностью
        return min(self.topic_confidence, key=self.topic_confidence.get)

    def _update_topic_confidence(self, topic: str, score: float):
        if topic in self.topic_confidence:
            self.topic_confidence[topic] = 0.8 * self.topic_confidence[topic] + 0.2 * score

    def _generate_question_for_topic(self, topic: str) -> Optional[str]:
        try:
            response = self.llm.chat.completions.create(
                model="local-model",
                messages=[
                    {"role": "system", "content": "Ты — исследовательский агент. Придумай один вопрос по теме для пользователя."},
                    {"role": "user", "content": f"Тема: {topic}. Сформулируй один вопрос (только вопрос, без пояснений)."}
                ],
                max_tokens=30,
                temperature=0.8,
            )
            q = response.choices[0].message.content.strip()
            if q and '?' in q:
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