# brain/memory.py
import numpy as np
import faiss
import torch
import time
from collections import deque
from typing import List, Dict, Any, Optional, Tuple


class VectorMemoryIndex:
    """
    ИСПРАВЛЕНО: IndexHNSWFlat НЕ поддерживает remove_ids() — вызов кидает исключение
    (или в некоторых версиях faiss тихо ничего не делает). Раньше это происходило
    и при вытеснении по capacity в add(), и в consolidate() — то есть после заполнения
    памяти новые записи фактически переставали попадать в индекс (если исключение
    гасилось выше по стеку) либо приложение падало. Итог — ретривер годами возвращал
    один и тот же старый набор воспоминаний, отсюда повторяющиеся ответы.

    Заменено на IndexIDMap2(IndexFlatIP) с явными устойчивыми id (не позиция в списке —
    она "плывёт" при удалениях). remove_ids на ID-mapped flat-индексе поддерживается
    корректно. Векторы уже L2-нормализованы в EmbeddingProvider, поэтому inner product
    эквивалентен косинусной близости.
    """
    def __init__(self, dim: int = 384, capacity: int = 50000):
        self.dim = dim
        self.capacity = capacity
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self._next_id = 0
        self.metadata: Dict[int, Dict] = {}
        self.timestamps: Dict[int, float] = {}
        self.access_counts: Dict[int, int] = {}
        self.importance_scores: Dict[int, float] = {}

    def add(self, vector: np.ndarray, meta: Dict) -> int:
        if len(self.metadata) >= self.capacity:
            self._evict_least_important()
        vid = self._next_id
        self._next_id += 1
        self.index.add_with_ids(vector.reshape(1, -1).astype('float32'), np.array([vid], dtype=np.int64))
        self.metadata[vid] = meta
        self.timestamps[vid] = time.time()
        self.access_counts[vid] = 0
        self.importance_scores[vid] = 0.5
        return vid

    def _evict_least_important(self):
        if not self.importance_scores:
            return
        worst_id = min(self.importance_scores, key=self.importance_scores.get)
        self._remove_ids([worst_id])

    def _remove_ids(self, ids: List[int]):
        if not ids:
            return
        self.index.remove_ids(np.array(ids, dtype=np.int64))
        for vid in ids:
            self.metadata.pop(vid, None)
            self.timestamps.pop(vid, None)
            self.access_counts.pop(vid, None)
            self.importance_scores.pop(vid, None)

    def search(self, query: np.ndarray, k: int = 5) -> List[Dict]:
        if self.index.ntotal == 0:
            return []
        similarities, ids = self.index.search(query.reshape(1, -1).astype('float32'), min(k, self.index.ntotal))
        results = []
        for i, vid in enumerate(ids[0]):
            vid = int(vid)
            if vid == -1 or vid not in self.metadata:
                continue
            self.access_counts[vid] += 1
            self.importance_scores[vid] = self._compute_importance(vid)
            results.append({
                "metadata": self.metadata[vid],
                "distance": float(similarities[0][i]),  # inner product / cosine similarity (больше = ближе)
                "timestamp": self.timestamps[vid],
                "importance": self.importance_scores[vid],
            })
        return results

    def _compute_importance(self, vid: int) -> float:
        age = time.time() - self.timestamps[vid]
        age_factor = max(0, 1 - age / (30 * 24 * 3600))  # за месяц забываем
        return (self.access_counts[vid] + 1) * age_factor

    def consolidate(self, threshold: float = 0.1):
        """Удаляем записи с важностью ниже порога"""
        to_remove = [vid for vid, imp in self.importance_scores.items() if imp < threshold]
        self._remove_ids(to_remove)


class HierarchicalMemory:
    def __init__(self, dim: int = 384, working_size: int = 10, episodic_capacity: int = 50000):
        self.dim = dim
        self.working = deque(maxlen=working_size)
        self.episodic = VectorMemoryIndex(dim, capacity=episodic_capacity)
        self.semantic_memory = SemanticGraph()

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
