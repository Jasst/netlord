# brain/utils.py
import torch
import torch.nn.functional as F
import hashlib
import numpy as np
from typing import Optional, List
from transformers import AutoTokenizer, AutoModel

class EmbeddingProvider:
    def __init__(self, dim: int = 384, model_name: str = "intfloat/e5-large-v2"):
        self.dim = dim
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self._cache = {}

    def get_embedding(self, text: str) -> torch.Tensor:
        if text in self._cache:
            return self._cache[text].clone()
        if "e5" in self.model_name.lower():
            text = "query: " + text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
            emb = F.normalize(emb, p=2, dim=0)
        self._cache[text] = emb.clone()
        return emb

    def get_embeddings_batch(self, texts: List[str]) -> List[torch.Tensor]:
        return [self.get_embedding(t) for t in texts]

def random_vector(dim: int) -> torch.Tensor:
    v = torch.randn(dim)
    return F.normalize(v, p=2, dim=0)

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a.flatten(), p=2, dim=0)
    b = F.normalize(b.flatten(), p=2, dim=0)
    return float(torch.dot(a, b).item())

def hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

def compute_importance(access_count: int, age: float, max_age: float = 30*24*3600) -> float:
    # важность = частота доступа * (1 - возраст/макс_возраст)
    age_factor = max(0, 1 - age / max_age)
    return (access_count + 1) * age_factor