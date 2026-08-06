# brain/memory.py
import numpy as np
import faiss
import torch
import time
from collections import deque
from typing import List, Dict, Any, Optional

class VectorMemoryIndex:
    def __init__(self, dim: int = 384, capacity: int = 50000):
        self.dim = dim
        self.capacity = capacity
        # Используем HNSW для быстрого приближённого поиска
        self.index = faiss.IndexHNSWFlat(dim, 32)
        self.index.hnsw.efConstruction = 200
        self.metadata = []
        self.timestamps = []

    def add(self, vector: np.ndarray, meta: Dict) -> int:
        if len(self.metadata) >= self.capacity:
            # Удаляем самую старую запись (ID 0)
            self.index.remove_ids(np.array([0]))
            self.metadata.pop(0)
            self.timestamps.pop(0)
        self.index.add(vector.reshape(1, -1).astype('float32'))
        idx = len(self.metadata)
        self.metadata.append(meta)
        self.timestamps.append(time.time())
        return idx

    def search(self, query: np.ndarray, k: int = 5) -> List[Dict]:
        if self.index.ntotal == 0:
            return []
        distances, indices = self.index.search(query.reshape(1, -1).astype('float32'), min(k, self.index.ntotal))
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                results.append({
                    "metadata": self.metadata[idx],
                    "distance": float(distances[0][i]),
                    "timestamp": self.timestamps[idx]
                })
        return results

class HierarchicalMemory:
    def __init__(self, dim: int = 384, working_size: int = 10, episodic_capacity: int = 50000):
        self.dim = dim
        self.working = deque(maxlen=working_size)
        self.episodic = VectorMemoryIndex(dim, capacity=episodic_capacity)

    def add_working(self, vector: torch.Tensor, context: Any = None):
        self.working.append({"vector": vector.detach().cpu().numpy(), "context": context, "time": time.time()})

    def add_episodic(self, vector: torch.Tensor, meta: Dict):
        self.episodic.add(vector.detach().cpu().numpy(), meta)

    def retrieve(self, query: torch.Tensor, k: int = 5) -> List[Dict]:
        return self.episodic.search(query.detach().cpu().numpy(), k)

    def consolidate(self):
        # Переносим рабочую память в эпизодическую
        for item in list(self.working):
            self.add_episodic(torch.from_numpy(item["vector"]).float(), item["context"])
        self.working.clear()