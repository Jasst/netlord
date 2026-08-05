# brain/memory.py
import numpy as np
import faiss
import torch
import time
from collections import deque
from typing import List, Dict, Any, Optional

class VectorMemoryIndex:
    def __init__(self, dim: int = 128, capacity: int = 1000):
        self.dim = dim
        self.capacity = capacity
        self.index = faiss.IndexHNSWFlat(dim, 32)
        self.metadata = []
        self.timestamps = []

    def add(self, vector: np.ndarray, meta: Dict) -> int:
        if len(self.metadata) >= self.capacity:
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
        distances, indices = self.index.search(query.reshape(1, -1).astype('float32'), k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                results.append({
                    "metadata": self.metadata[idx],
                    "distance": distances[0][i],
                    "timestamp": self.timestamps[idx]
                })
        return results

class HierarchicalMemory:
    def __init__(self, dim: int = 128, working_size: int = 5, episodic_capacity: int = 500):
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
        if len(self.working) > 0:
            last = list(self.working)[-1]
            vec = last["vector"]
            self.add_episodic(torch.from_numpy(vec).float(), {"context": last["context"]})
            self.working.clear()