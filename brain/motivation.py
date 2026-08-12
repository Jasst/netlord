import random
from typing import List, Dict, Any, Optional

class DriveSystem:
    """
    Система внутренних драйверов (любопытство, когерентность, социальность и др.).
    Выбирает действие на основе максимизации удовлетворения драйверов.
    """
    def __init__(self):
        self.drives = {
            'curiosity': 0.5,   # желание узнавать новое
            'coherence': 0.5,   # стремление к непротиворечивости знаний
            'social': 0.3,      # желание быть полезным/понятым
            'mastery': 0.4,     # стремление к компетентности
        }
        self.goals = []  # активные цели (строки)

    def update(self, feedback: Dict[str, float], new_info: bool = False):
        """Обновить драйверы на основе обратной связи."""
        if new_info:
            self.drives['curiosity'] = min(1.0, self.drives['curiosity'] + 0.05)
        if feedback.get('success', False):
            self.drives['mastery'] = min(1.0, self.drives['mastery'] + 0.02)
            self.drives['social'] = min(1.0, self.drives['social'] + 0.02)
        # постепенное затухание
        for k in self.drives:
            self.drives[k] = max(0.1, self.drives[k] * 0.995)

    def select_action(self, possible_actions: List[Dict]) -> Dict:
        """
        possible_actions: список {'name': str, 'expected_impact': {drive: float}}
        Выбирает действие с максимальной взвешенной суммой impact * drive.
        """
        best_action = None
        best_score = -float('inf')
        for action in possible_actions:
            impact = action.get('expected_impact', {})
            score = sum(self.drives.get(k, 0) * impact.get(k, 0) for k in self.drives)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action if best_action else possible_actions[0] if possible_actions else None

    def add_goal(self, goal: str):
        self.goals.append(goal)

    def get_active_goals(self) -> List[str]:
        return self.goals[:]