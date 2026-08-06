# brain/memory.py
import numpy as np
import faiss
import torch
import time
from collections import deque
from typing import List, Dict, Any, Optional, Tuple

class VectorMemoryIndex:
    def __init__(self, dim: int = 384, capacity: int = 50000):
        self.dim = dim
        self.capacity = capacity
        self.index = faiss.IndexHNSWFlat(dim, 32)
        self.index.hnsw.efConstruction = 200
        self.metadata = []          # list of dict
        self.timestamps = []
        self.access_counts = []     # для вычисления важности
        self.importance_scores = []

    def add(self, vector: np.ndarray, meta: Dict) -> int:
        if len(self.metadata) >= self.capacity:
            # Удаляем наименее важную запись
            min_imp_idx = np.argmin(self.importance_scores)
            self.index.remove_ids(np.array([min_imp_idx]))
            self.metadata.pop(min_imp_idx)
            self.timestamps.pop(min_imp_idx)
            self.access_counts.pop(min_imp_idx)
            self.importance_scores.pop(min_imp_idx)
        self.index.add(vector.reshape(1, -1).astype('float32'))
        idx = len(self.metadata)
        self.metadata.append(meta)
        self.timestamps.append(time.time())
        self.access_counts.append(0)
        self.importance_scores.append(0.5)  # начальная важность
        return idx

    def search(self, query: np.ndarray, k: int = 5) -> List[Dict]:
        if self.index.ntotal == 0:
            return []
        distances, indices = self.index.search(query.reshape(1, -1).astype('float32'), min(k, self.index.ntotal))
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                self.access_counts[idx] += 1
                self.importance_scores[idx] = self._compute_importance(idx)
                results.append({
                    "metadata": self.metadata[idx],
                    "distance": float(distances[0][i]),
                    "timestamp": self.timestamps[idx],
                    "importance": self.importance_scores[idx]
                })
        return results

    def _compute_importance(self, idx: int) -> float:
        # важность = частота доступа * (1 - возраст/жизнь)
        age = time.time() - self.timestamps[idx]
        age_factor = max(0, 1 - age / (30*24*3600))  # за месяц забываем
        return (self.access_counts[idx] + 1) * age_factor

    def consolidate(self, threshold: float = 0.1):
        """Удаляем записи с важностью ниже порога"""
        to_remove = [i for i, imp in enumerate(self.importance_scores) if imp < threshold]
        for idx in sorted(to_remove, reverse=True):
            self.index.remove_ids(np.array([idx]))
            del self.metadata[idx]
            del self.timestamps[idx]
            del self.access_counts[idx]
            del self.importance_scores[idx]


class HierarchicalMemory:
    def __init__(self, dim: int = 384, working_size: int = 10, episodic_capacity: int = 50000):
        self.dim = dim
        self.working = deque(maxlen=working_size)
        self.episodic = VectorMemoryIndex(dim, capacity=episodic_capacity)
        self.semantic_memory = SemanticGraph()  # новый семантический граф

    def add_working(self, vector: torch.Tensor, context: Any = None):
        self.working.append({"vector": vector.detach().cpu().numpy(), "context": context, "time": time.time()})

    def add_episodic(self, vector: torch.Tensor, meta: Dict):
        self.episodic.add(vector.detach().cpu().numpy(), meta)

    def retrieve(self, query: torch.Tensor, k: int = 5) -> List[Dict]:
        return self.episodic.search(query.detach().cpu().numpy(), k)

    def consolidate(self, threshold: float = 0.1):
        # Переносим рабочую память в эпизодическую
        for item in list(self.working):
            self.add_episodic(torch.from_numpy(item["vector"]).float(), item["context"])
        self.working.clear()
        self.episodic.consolidate(threshold)

    # --- Семантическая память ---
    def add_semantic_triple(self, subj: str, pred: str, obj: str, confidence: float = 1.0):
        self.semantic_memory.add_triple(subj, pred, obj, confidence)

    def query_semantic(self, subj: Optional[str] = None, pred: Optional[str] = None, obj: Optional[str] = None) -> List[Tuple]:
        return self.semantic_memory.query(subj, pred, obj)


class SemanticGraph:
    """Простой граф знаний (тройки) с весами уверенности"""
    def __init__(self, capacity: int = 10000):
        self.triples = []  # (subj, pred, obj, confidence)
        self.capacity = capacity

    def add_triple(self, subj: str, pred: str, obj: str, confidence: float = 1.0):
        # Проверка на дубли
        for i, (s, p, o, c) in enumerate(self.triples):
            if s == subj and p == pred and o == obj:
                self.triples[i] = (s, p, o, min(1.0, c + confidence * 0.1))
                return
        self.triples.append((subj, pred, obj, min(1.0, confidence)))
        if len(self.triples) > self.capacity:
            self.triples.pop(0)

    def query(self, subj: Optional[str] = None, pred: Optional[str] = None, obj: Optional[str] = None) -> List[Tuple]:
        results = []
        for s, p, o, c in self.triples:
            if (subj is None or s == subj) and (pred is None or p == pred) and (obj is None or o == obj):
                results.append((s, p, o, c))
        return results