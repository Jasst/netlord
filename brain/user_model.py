import torch
import torch.nn.functional as F
from collections import deque
from typing import List, Dict, Optional

class UserModel:
    """
    Модель собеседника: хранит эмбеддинги его вопросов, предполагаемый уровень знаний, интересы.
    """
    def __init__(self, dim: int = 512, history_len: int = 20):
        self.dim = dim
        self.history = deque(maxlen=history_len)  # (question_emb, answer_emb, time)
        self.knowledge_level = 0.5   # 0..1
        self.interests = {}          # topic -> score

    def update(self, question_emb: torch.Tensor, answer_emb: torch.Tensor, topic: Optional[str] = None):
        self.history.append((question_emb.clone(), answer_emb.clone()))
        if topic:
            self.interests[topic] = self.interests.get(topic, 0.0) + 0.1
            # нормализуем
            total = sum(self.interests.values())
            if total > 0:
                for k in self.interests:
                    self.interests[k] /= total

    def get_user_embedding(self) -> torch.Tensor:
        """Усреднённый эмбеддинг пользователя на основе его вопросов."""
        if not self.history:
            return torch.zeros(self.dim)
        emb = torch.stack([q for q, _ in self.history]).mean(dim=0)
        return F.normalize(emb, p=2, dim=0)

    def predict_question_difficulty(self, question_emb: torch.Tensor) -> float:
        """Оценивает, насколько вопрос сложен для данного пользователя."""
        if not self.history:
            return 0.5
        user_emb = self.get_user_embedding()
        sim = F.cosine_similarity(question_emb.unsqueeze(0), user_emb.unsqueeze(0)).item()
        # чем ниже схожесть с прошлыми вопросами, тем сложнее
        return 1.0 - sim