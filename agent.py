# agent.py (полностью исправленный для v6)
import time
import threading
import random
from typing import List, Optional
from openai import OpenAI
from smart_brain_v6 import Brain, Teacher

class BrainAgent:
    def __init__(
        self,
        brain: Brain,
        teacher: Teacher,
        llm_client: OpenAI,
        topics: List[str] = None,
        interval_seconds: int = 120,
        questions_per_cycle: int = 2,
        temperature: float = 0.7,
        enabled: bool = True,
        interactive_mode: bool = False,
        user_question_timeout: int = 30
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

        self._stop_flag = False
        self._thread = None

        # Очередь вопроса для пользователя
        self.pending_question = None
        self.active_question = None
        self.waiting_for_answer = False

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
                # Автосохранение каждые 10 циклов (для надёжности)
                if cycle_counter % 10 == 0:
                    self.brain.save()
                    self.brain.save_dialog_history()
                    print("[Agent] Автосохранение выполнено.")
            for _ in range(self.interval):
                if self._stop_flag:
                    break
                time.sleep(1)

    def _cycle(self):
        """Один цикл: либо задаём вопрос пользователю, либо учимся сами."""
        if self.interactive_mode:
            # 1. Проверяем, есть ли активный вопрос (ожидает ответа)
            if self.active_question is not None:
                print("[Agent] Ожидаем ответ на активный вопрос")
                return

            # 2. Получаем контекст: темы пользователя и слабые места сети
            user_topics = self.brain.get_recent_user_topics(limit=3)
            uncertain = self.brain.get_uncertain_concepts(threshold=0.3, limit=3)

            # 3. Пытаемся сгенерировать контекстный вопрос
            question = self.brain.generate_contextual_question(user_topics, uncertain)

            # Если не получилось – генерируем случайный вопрос по теме
            if not question:
                if user_topics:
                    topic = random.choice(user_topics)
                else:
                    topic = random.choice(self.topics)
                question = self._generate_question_for_topic(topic)

            if not question:
                print("[Agent] Не удалось сгенерировать вопрос")
                return

            # 4. Отправляем вопрос пользователю
            self.pending_question = question
            self.waiting_for_answer = True
            print(f"[Agent] Вопрос для пользователя: {question}")

            # Ждём, пока фронтенд заберёт вопрос и пользователь ответит
            start = time.time()
            while self.waiting_for_answer and (time.time() - start) < self.user_question_timeout:
                if self._stop_flag:
                    break
                time.sleep(1)

            # Если таймаут и вопрос всё ещё ожидает – сбрасываем
            if self.pending_question == question:
                self.pending_question = None
                self.waiting_for_answer = False
                print(f"[Agent] Таймаут ответа на вопрос: {question}")
            elif self.active_question == question:
                self.active_question = None
                self.waiting_for_answer = False
                print(f"[Agent] Таймаут ответа на активный вопрос: {question}")

            return

        # ------------------------------------------------------------
        # НЕИНТЕРАКТИВНЫЙ РЕЖИМ (самообучение)
        # ------------------------------------------------------------
        topic = random.choice(self.topics)
        questions = self._generate_questions(topic, self.questions_per_cycle)
        for q in questions:
            if self._stop_flag:
                break
            result = self.brain.generate_answer(q, temperature=self.temperature, use_rag=True)
            answer = result["text"]
            score, improved, details = self.teacher.evaluate(q, answer)

            if score >= 0.7:
                target = improved if improved != answer else answer
                if target.lower() != q.lower():
                    self.brain.learn_pair(q, target)
                    self.brain.save()                     # <-- СОХРАНЕНИЕ
                    self.brain.save_dialog_history()      # <-- СОХРАНЕНИЕ ИСТОРИИ
                    print(f"[Agent] Выучено: {q} -> {target}")
            elif score <= 0.3:
                if improved != answer and improved.lower() != q.lower():
                    self.brain.learn_negative_pair(q, answer)
                    self.brain.learn_pair(q, improved)
                    self.brain.save()                     # <-- СОХРАНЕНИЕ
                    self.brain.save_dialog_history()
                    print(f"[Agent] Отрицание и обучение: {q} -> {improved}")
                else:
                    self.brain.learn_negative_pair(q, answer)
                    self.brain.save()                     # <-- СОХРАНЕНИЕ
                    self.brain.save_dialog_history()
                    print(f"[Agent] Отрицание: {q} -> {answer}")
            else:
                self.brain.dialog_memory.add_turn(q, answer, self.brain.text_to_embedding(q))
                self.brain.save_dialog_history()          # <-- СОХРАНЕНИЕ ТОЛЬКО ИСТОРИИ
                print(f"[Agent] Диалог сохранён: {q} -> {answer}")
            time.sleep(1)

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
            self.brain.save()                             # <-- СОХРАНЕНИЕ
            self.brain.save_dialog_history()              # <-- СОХРАНЕНИЕ ИСТОРИИ
            print(f"[Agent] Пользователь ответил на '{question}' -> '{answer}', выучено.")
            self.active_question = None
            self.waiting_for_answer = False
            self.brain.dialog_memory.add_turn(question, answer, self.brain.text_to_embedding(question))
        else:
            print(f"[Agent] Ответ на неактивный вопрос: {question} (активный: {self.active_question})")

    def _generate_question_for_topic(self, topic: str) -> Optional[str]:
        """Генерирует один вопрос по теме (используется как fallback)."""
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

    def _generate_questions(self, topic: str, count: int) -> List[str]:
        """Генерирует несколько вопросов по теме (для самообучения)."""
        system_prompt = (
            "Ты — исследовательский агент. Твоя задача — придумать вопросы, "
            "которые помогут глубже понять тему. Вопросы должны быть разнообразными: "
            "фактические, причинно-следственные, сравнительные, гипотетические. "
            "Ответы должны быть короткими (1-5 слов) и точными."
        )
        user_prompt = (
            f"Сгенерируй {count} вопросов по теме '{topic}'. "
            "Формат: каждый вопрос на новой строке, без номеров."
        )
        try:
            response = self.llm.chat.completions.create(
                model="local-model",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=count * 30,
                temperature=0.8,
            )
            raw = response.choices[0].message.content.strip()
            questions = [line.strip() for line in raw.splitlines() if line.strip() and '?' in line]
            return questions[:count]
        except Exception as e:
            print(f"[Agent] Ошибка генерации вопросов: {e}")
            fallbacks = [
                f"Что такое {topic}?",
                f"Зачем нужен {topic}?",
                f"Как работает {topic}?",
                f"Где используется {topic}?",
                f"Какие виды {topic} существуют?"
            ]
            return random.sample(fallbacks, min(count, len(fallbacks)))