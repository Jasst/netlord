# brain/memory.py
import numpy as np
import faiss
import torch
import time
from collections import deque
from typing import List, Dict, Any, Optional, Tuple

# --- VectorMemoryIndex (без изменений) ---
class VectorMemoryIndex:
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
                "distance": float(similarities[0][i]),
                "timestamp": self.timestamps[vid],
                "importance": self.importance_scores[vid],
            })
        return results

    def _compute_importance(self, vid: int) -> float:
        age = time.time() - self.timestamps[vid]
        age_factor = max(0, 1 - age / (30 * 24 * 3600))
        return (self.access_counts[vid] + 1) * age_factor

    def consolidate(self, threshold: float = 0.1):
        to_remove = [vid for vid, imp in self.importance_scores.items() if imp < threshold]
        self._remove_ids(to_remove)

# --- HierarchicalMemory (с исправленным query_semantic) ---
class HierarchicalMemory:
    def __init__(self, dim: int = 384, working_size: int = 10, episodic_capacity: int = 50000):
        self.dim = dim
        self.working = deque(maxlen=working_size)
        self.episodic = VectorMemoryIndex(dim, capacity=episodic_capacity)
        self.semantic_memory = SemanticGraph(dim=dim, capacity=10000)

    def add_working(self, vector: torch.Tensor, context: Any = None):
        self.working.append({"vector": vector.detach().cpu().numpy(), "context": context, "time": time.time()})

    def add_episodic(self, vector: torch.Tensor, meta: Dict):
        self.episodic.add(vector.detach().cpu().numpy(), meta)

    def retrieve(self, query: torch.Tensor, k: int = 5) -> List[Dict]:
        return self.episodic.search(query.detach().cpu().numpy(), k)

    def consolidate(self, threshold: float = 0.1):
        for item in list(self.working):
            self.add_episodic(torch.from_numpy(item["vector"]).float(), item["context"])
        self.working.clear()
        self.episodic.consolidate(threshold)

    # --- Семантическая память (обёртка) ---
    def add_semantic_triple(self, subj: str, pred: str, obj: str, confidence: float = 1.0, embedder=None):
        self.semantic_memory.add_triple(subj, pred, obj, confidence, embedder)

    def query_semantic(self, query_text: Optional[str] = None, embedder=None,
                       subj: Optional[str] = None, pred: Optional[str] = None, obj: Optional[str] = None,
                       k: int = 10) -> List[Tuple]:
        """
        Поиск по семантической памяти.
        Если заданы subj/pred/obj — точный фильтр.
        Иначе — векторный поиск по query_text (с использованием embedder).
        """
        if subj is not None or pred is not None or obj is not None:
            return self.semantic_memory.query_exact(subj, pred, obj)
        if query_text:
            return self.semantic_memory.query_vector(query_text, embedder, k=k)
        return []

# --- SemanticGraph (с векторным поиском) ---
class SemanticGraph:
    def __init__(self, dim: int = 384, capacity: int = 10000):
        self.dim = dim
        self.capacity = capacity
        self.triples = []  # (subj, pred, obj, confidence)
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self._next_id = 0
        self.id_to_triple = {}
        self.exact_index = {}

    def _get_embedding(self, text: str, embedder) -> np.ndarray:
        if embedder is None:
            return np.random.randn(self.dim).astype('float32')
        emb = embedder.get_embedding(text, is_query=False).cpu().numpy()
        return emb.astype('float32')

    def add_triple(self, subj: str, pred: str, obj: str, confidence: float = 1.0, embedder=None):
        key = (subj, pred, obj)
        if key in self.exact_index:
            self.exact_index[key] = max(self.exact_index[key], confidence)
            for i, (s, p, o, c) in enumerate(self.triples):
                if s == subj and p == pred and o == obj:
                    self.triples[i] = (s, p, o, max(c, confidence))
                    return

        combined = f"{subj} {pred} {obj}"
        vec = self._get_embedding(combined, embedder)
        vid = self._next_id
        self._next_id += 1
        self.index.add_with_ids(vec.reshape(1, -1), np.array([vid], dtype=np.int64))
        self.id_to_triple[vid] = (subj, pred, obj, confidence)
        self.triples.append((subj, pred, obj, confidence))
        self.exact_index[key] = confidence

        if len(self.triples) > self.capacity:
            oldest = self.triples.pop(0)
            for vid, (s, p, o, c) in self.id_to_triple.items():
                if s == oldest[0] and p == oldest[1] and o == oldest[2]:
                    self.index.remove_ids(np.array([vid], dtype=np.int64))
                    del self.id_to_triple[vid]
                    del self.exact_index[(s, p, o)]
                    break

    def query_exact(self, subj: Optional[str] = None, pred: Optional[str] = None, obj: Optional[str] = None) -> List[Tuple]:
        results = []
        for s, p, o, c in self.triples:
            if (subj is None or s == subj) and (pred is None or p == pred) and (obj is None or o == obj):
                results.append((s, p, o, c))
        return results

    def query_vector(self, query_text: str, embedder, k: int = 10) -> List[Tuple]:
        if self.index.ntotal == 0:
            return []
        vec = self._get_embedding(query_text, embedder)
        sims, ids = self.index.search(vec.reshape(1, -1), min(k, self.index.ntotal))
        results = []
        for i, vid in enumerate(ids[0]):
            if vid == -1:
                continue
            triple = self.id_to_triple.get(int(vid))
            if triple:
                s, p, o, c = triple
                results.append((s, p, o, float(sims[0][i])))
        return results