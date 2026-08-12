import math
import torch

class EmotionModel:
    """
    Трёхмерная модель эмоций: валентность (valence), возбуждение (arousal), доминирование (dominance).
    """
    def __init__(self):
        self.valence = 0.0      # -1..1
        self.arousal = 0.0      # 0..1
        self.dominance = 0.0    # 0..1

    def update(self, reward: float, novelty: float, goal_achieved: bool, confidence: float = 0.5):
        """Обновить эмоциональное состояние на основе событий."""
        # валентность зависит от награды и достижения цели
        valence_delta = (reward - 0.5) * 0.2 + (0.3 if goal_achieved else -0.1)
        self.valence = max(-1.0, min(1.0, self.valence + valence_delta))

        # возбуждение зависит от новизны и уверенности (чем меньше уверенность, тем выше возбуждение)
        arousal_delta = novelty * 0.2 + (1 - confidence) * 0.1
        self.arousal = max(0.0, min(1.0, self.arousal + arousal_delta))

        # доминирование растёт при успехах, падает при неудачах
        dom_delta = 0.05 if reward > 0.6 else -0.05
        self.dominance = max(0.0, min(1.0, self.dominance + dom_delta))

        # естественное затухание
        self.valence *= 0.99
        self.arousal *= 0.99
        self.dominance *= 0.99

    def get_vector(self) -> torch.Tensor:
        return torch.tensor([self.valence, self.arousal, self.dominance], dtype=torch.float32)

    def modulate_temperature(self, base_temperature: float) -> float:
        """Эмоционально модулировать температуру: возбуждение повышает, валентность тоже."""
        return base_temperature + 0.2 * self.arousal + 0.1 * (self.valence + 1) / 2

    def __str__(self):
        return f"Emotion(val={self.valence:.2f}, ar={self.arousal:.2f}, dom={self.dominance:.2f})"