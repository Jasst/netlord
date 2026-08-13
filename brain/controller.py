# brain/controller.py
from __future__ import annotations

import time
import datetime
from typing import TYPE_CHECKING, Optional, Dict, Any, List

# Импорты типов только для статической проверки (не выполняются в рантайме)
if TYPE_CHECKING:
    from brain.brain import CognitiveBrain
    from brain.world_model import WorldModel
    from brain.reasoner import Reasoner
    from brain.planner import Planner
    from brain.self_model import SelfModel
    from brain.llm import LLMInterface
    from brain.teacher import Teacher
    from brain.search import WebSearcher


class CognitiveController:
    def __init__(self, brain: CognitiveBrain, world_model: WorldModel,
                 reasoner: Reasoner, planner: Planner, self_model: SelfModel,
                 llm: LLMInterface, teacher: Teacher, searcher: WebSearcher):
        self.brain = brain
        self.world = world_model
        self.reasoner = reasoner
        self.planner = planner
        self.self_model = self_model
        self.llm = llm
        self.teacher = teacher
        self.searcher = searcher

        # Регистрируем обработчики действий в планировщике
        planner.register_action_handler("search", self._handle_search)
        planner.register_action_handler("ask_user", self._handle_ask_user)
        planner.register_action_handler("reason", self._handle_reason)
        planner.register_action_handler("learn", self._handle_learn)
        planner.register_action_handler("observe", self._handle_observe)
        planner.register_action_handler("simulate", self._handle_simulate)

        self.history = []  # история действий

    def process_input(self, user_input: str) -> str:
        """Основной метод: обработать ввод пользователя."""
        # 1. Восприятие – запоминаем ввод
        self.world.add_episode(user_input, importance=0.7)

        # 2. Извлечение – получаем релевантные факты и память
        query_vec = self.brain.text_to_embedding(user_input, is_query=True)
        memory_results = self.brain.memory.retrieve(query_vec, k=10)
        facts = self.world.query_by_text(user_input, k=5)

        # 3. Рассуждение – выполняем выводы на основе фактов
        self.reasoner.update_world_with_inferences()
        contradictions = self.reasoner.check_contradictions()
        if contradictions:
            self.self_model.add_error(f"Противоречие: {contradictions[0][0].subject} -> ...")

        # 4. Генерация гипотез – если вопрос сложный
        if self._is_complex_question(user_input):
            hypothesis = self._generate_hypothesis(user_input)
            if hypothesis:
                self.world.add_hypothesis(hypothesis, confidence=0.3, generated_from=user_input)

        # 5. Планирование – если нужно выполнить действие
        if self._needs_planning(user_input):
            goal_id = self.planner.add_goal(f"Ответить на: {user_input}", priority=5)
            plan = self.planner.generate_plan(user_input, context=str(facts))
            if plan:
                self.planner.goals[goal_id].plan = plan
                success = self.planner.execute_plan(self.planner.goals[goal_id])
                if success:
                    self.self_model.add_successful_strategy("Планирование сработало")

        # 6. Основной ответ через Brain (если не было выполнено действие)
        result = self.brain._legacy_step(user_input, use_search=False)
        answer = result["answer"]

        # 7. Оценка ответа (Teacher)
        score, improved, details = self.teacher.evaluate(user_input, answer)
        if improved != answer and score > 0.6:
            answer = improved
            self.world.add_fact("question", "improved_answer", user_input,
                                confidence=score, source="teacher")

        # 8. Обучение – если ответ хороший
        if score > 0.55:
            self.brain.learn_pair(user_input, answer, reward=score)
            self.world.add_fact("question", "answer", answer, confidence=score, source="brain")

        # 9. Рефлексия – если уверенность низкая
        if result.get("confidence", 0) < 0.4:
            self.self_model.add_unresolved_question(user_input)

        # 10. Обновление Self-model на основе оценки
        self.self_model.update_capability("answer_quality", score)
        if score < 0.4:
            self.self_model.add_error(f"Низкая оценка для: {user_input}")

        # 11. Запись эпизода
        self.world.add_episode(f"Вопрос: {user_input} -> Ответ: {answer}")

        # 12. Периодическая консолидация (сон)
        if self.brain.step_counter % 100 == 0:
            self.brain.sleep()

        return answer

    def _is_complex_question(self, text: str) -> bool:
        return len(text.split()) > 10 or any(w in text.lower() for w in ["почему", "как", "объясни"])

    def _needs_planning(self, text: str) -> bool:
        return "найди" in text.lower() or "поищи" in text.lower() or "сделай" in text.lower()

    def _generate_hypothesis(self, text: str) -> Optional[str]:
        prompt = f"Сформулируй гипотезу, которая может объяснить или ответить на вопрос: {text}"
        hyp = self.llm.generate(prompt, max_tokens=100, temperature=0.7)
        return hyp if hyp.strip() else None

    # --- Обработчики действий для планировщика ---
    def _handle_search(self, params: Dict) -> bool:
        query = params.get("query", "")
        if not query:
            return False
        results = self.searcher.search(query)
        if results:
            for r in results[:3]:
                self.world.add_fact("search", "result", r.get("body", ""),
                                    confidence=0.7, source="web", evidence=[r.get("url", "")])
            return True
        return False

    def _handle_ask_user(self, params: Dict) -> bool:
        question = params.get("question", "")
        if question:
            # В интерактивном режиме – отложить вопрос (используется через агента)
            self.brain.pending_question = question
            return True
        return False

    def _handle_reason(self, params: Dict) -> bool:
        problem = params.get("problem", "")
        if problem:
            reasoning = self.llm.generate(f"Рассуждай шаг за шагом: {problem}", max_tokens=200)
            self.world.add_episode(f"Рассуждение: {reasoning}")
            return True
        return False

    def _handle_learn(self, params: Dict) -> bool:
        q = params.get("question", "")
        a = params.get("answer", "")
        if q and a:
            self.brain.learn_pair(q, a, reward=0.8)
            return True
        return False

    def _handle_observe(self, params: Dict) -> bool:
        what = params.get("what", "")
        if "время" in what:
            now = datetime.datetime.now().strftime("%H:%M:%S")
            self.world.add_fact("observation", "time", now, confidence=1.0, source="system")
            return True
        return False

    def _handle_simulate(self, params: Dict) -> bool:
        scenario = params.get("scenario", "")
        if scenario:
            prediction = self.llm.generate(f"Что произойдёт, если {scenario}?", max_tokens=150)
            self.world.add_fact("simulation", "prediction", prediction, confidence=0.4, source="simulation")
            return True
        return False