import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Dict, Any, Optional, List
import time

class SelfModel(nn.Module):
    """Сжатое представление «Я» — динамическое состояние, обновляемое после каждого события.

    ИСПРАВЛЕНО: раньше один и тот же параметр `dim` использовался и как размер
    внутреннего состояния self.state, и как input_size у GRUCell. На вход же
    update() подаются эмбеддинги предложений (query_vec/answer_vec из
    EmbeddingProvider, размерность config.dim_embedding = 1024), а SelfModel
    создавался с dim=config.self_model_dim = 512 -> GRUCell(512, 512) падал с
    "input has inconsistent input_size: got 1024 expected 512". Теперь размер
    входного вектора (input_dim) и размер внутреннего состояния (dim) заданы
    отдельно, а между ними стоит обучаемая проекция input_proj.
    """
    def __init__(self, dim: int = 512, hidden_dim: int = 256, input_dim: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.input_dim = input_dim if input_dim is not None else dim
        self.state = nn.Parameter(torch.randn(dim) * 0.01, requires_grad=True)  # основной вектор
        self.input_proj = (nn.Linear(self.input_dim, dim)
                            if self.input_dim != dim else nn.Identity())
        self.gru = nn.GRUCell(dim, dim)
        self.proj = nn.Linear(dim + dim, dim)  # для обновления от внешнего входа

    def update(self, input_vec: torch.Tensor, context_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Обновляет состояние на основе нового опыта."""
        combined = input_vec if context_vec is None else (input_vec + context_vec) / 2
        combined = self.input_proj(combined)
        combined = F.normalize(combined, p=2, dim=0)
        new_state = self.gru(combined.unsqueeze(0), self.state.unsqueeze(0)).squeeze(0)
        self.state.data = F.normalize(new_state, p=2, dim=0)
        return self.state

    def get_state(self) -> torch.Tensor:
        return self.state.detach().clone()

    def similarity_to(self, other: torch.Tensor) -> float:
        return float(F.cosine_similarity(self.state.unsqueeze(0), other.unsqueeze(0), dim=1).item())


class AutobiographicalMemory:
    """Хранит эпизоды с привязкой к состоянию SelfModel в тот момент."""
    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self.episodes = deque(maxlen=capacity)  # list of dict

    def add(self, event: Dict[str, Any]):
        """event должен содержать 'input', 'answer', 'self_state' (tensor), 'timestamp'."""
        self.episodes.append({
            'input': event.get('input', ''),
            'answer': event.get('answer', ''),
            'self_state': event.get('self_state', None),  # clone
            'timestamp': event.get('timestamp', time.time()),
            'emotion': event.get('emotion', None),
        })

    def get_recent(self, k: int = 10) -> List[Dict]:
        return list(self.episodes)[-k:]

    def get_by_time(self, start_time: float, end_time: float) -> List[Dict]:
        return [e for e in self.episodes if start_time <= e['timestamp'] <= end_time]