from collections import deque
from typing import List, Dict, Any, Optional
import heapq
import time

class GlobalWorkspace:
    """
    Глобальное рабочее пространство — буфер для интеграции информации из разных модулей.
    Элементы имеют приоритет, и только топ-N транслируются всем.
    """
    def __init__(self, capacity: int = 20, broadcast_top_k: int = 5):
        self.buffer = deque(maxlen=capacity)
        self.broadcast_top_k = broadcast_top_k
        self._last_broadcast = []

    def publish(self, content: Any, source: str, priority: float = 0.5, metadata: Optional[Dict] = None):
        """Опубликовать сообщение от модуля-источника."""
        self.buffer.append({
            'content': content,
            'source': source,
            'priority': priority,
            'metadata': metadata or {},
            'timestamp': time.time()
        })

    def get_current(self) -> List[Dict]:
        """Вернуть топ-K самых приоритетных сообщений."""
        sorted_items = sorted(self.buffer, key=lambda x: x['priority'], reverse=True)
        self._last_broadcast = sorted_items[:self.broadcast_top_k]
        return self._last_broadcast

    def clear(self):
        self.buffer.clear()
        self._last_broadcast.clear()

    def get_last_broadcast(self) -> List[Dict]:
        return self._last_broadcast

    def __len__(self):
        return len(self.buffer)