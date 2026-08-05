# brain/utils.py
import torch
import torch.nn.functional as F
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Optional

class EmbeddingProvider:
    def __init__(self, dim: int = 128, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        self.dim = dim
        self.model = SentenceTransformer(model_name)
        self._cache = {}
        self._projection = None

    def _project(self, vec: torch.Tensor) -> torch.Tensor:
        src_dim = vec.shape[-1]
        if src_dim == self.dim:
            return vec.float()  # <-- добавлено .float()
        if self._projection is None or self._projection.shape[0] != src_dim:
            rng = np.random.RandomState(42)
            proj = rng.randn(src_dim, self.dim).astype(np.float32)
            self._projection = torch.from_numpy(proj)
        return F.normalize(vec.float() @ self._projection, p=2, dim=-1)  # <-- .float()

    def get_embedding(self, text: str) -> torch.Tensor:
        if text in self._cache:
            return self._cache[text].clone()
        emb = torch.from_numpy(self.model.encode(text, normalize_embeddings=True)).float()  # <-- добавлено .float()
        emb = self._project(emb)
        self._cache[text] = emb.clone()
        return emb

    def get_embeddings_batch(self, texts: list) -> list:
        return [self.get_embedding(t) for t in texts]

def hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

def random_vector(dim: int) -> torch.Tensor:
    v = torch.randn(dim)
    return F.normalize(v.unsqueeze(0), p=2, dim=1).squeeze(0)

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a.flatten().float(), p=2, dim=0)
    b = F.normalize(b.flatten().float(), p=2, dim=0)
    return float((a @ b).item())