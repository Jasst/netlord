# brain/self_model.py
"""
Модель себя: самооценка, знание о своих ограничениях, успешных стратегиях.
"""
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class SelfKnowledge:
    """Знание о себе."""
    aspect: str  # "capability", "limitation", "strategy", "error"
    description: str
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)


class SelfModel:
    def __init__(self):
        self.knowledge: List[SelfKnowledge] = []
        self.capabilities: Dict[str, float] = {}  # skill -> score (0..1)
        self.recent_errors: List[str] = []
        self.successful_strategies: List[str] = []
        self.unresolved_questions: List[str] = []

    def add_knowledge(self, aspect: str, description: str, confidence: float = 0.5):
        self.knowledge.append(SelfKnowledge(aspect, description, confidence, time.time()))
        # Ограничим количество
        if len(self.knowledge) > 100:
            self.knowledge = self.knowledge[-100:]

    def update_capability(self, skill: str, score: float):
        self.capabilities[skill] = score

    def add_error(self, error_description: str):
        self.recent_errors.append(error_description)
        if len(self.recent_errors) > 20:
            self.recent_errors = self.recent_errors[-20:]

    def add_successful_strategy(self, strategy: str):
        self.successful_strategies.append(strategy)
        if len(self.successful_strategies) > 20:
            self.successful_strategies = self.successful_strategies[-20:]

    def add_unresolved_question(self, question: str):
        self.unresolved_questions.append(question)
        if len(self.unresolved_questions) > 50:
            self.unresolved_questions = self.unresolved_questions[-50:]

    def get_self_description(self) -> str:
        """Сгенерировать текстовое описание себя (для использования в контексте)."""
        lines = []
        if self.capabilities:
            caps = sorted(self.capabilities.items(), key=lambda x: -x[1])[:5]
            lines.append("Сильные стороны: " + ", ".join(f"{k} ({v:.2f})" for k, v in caps))
        if self.recent_errors:
            lines.append("Недавние ошибки: " + "; ".join(self.recent_errors[-3:]))
        if self.successful_strategies:
            lines.append("Успешные стратегии: " + "; ".join(self.successful_strategies[-3:]))
        if self.unresolved_questions:
            lines.append("Нерешённые вопросы: " + "; ".join(self.unresolved_questions[-3:]))
        return "\n".join(lines)