# brain/utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import numpy as np
from collections import OrderedDict
from typing import Optional, List
from transformers import AutoTokenizer, AutoModel


class EmbeddingProvider:
    """
    ИСПРАВЛЕНО:
    - e5-модели асимметричны: вопросы нужно кодировать с префиксом "query: ",
      а сохраняемый контент (ответы, факты, документы) — с префиксом "passage: ".
      Раньше везде стоял "query: ", из-за чего similarity между двумя "query"-векторами
      систематически занижалась и поиск по памяти/графу промахивался мимо релевантных узлов.
    - Кэш эмбеддингов был неограниченным (self._cache = {}) — при непрерывном обучении
      это неограниченно растущая утечка памяти. Теперь это LRU-кэш с ограничением размера.
    """
    def __init__(self, dim: int = 384, model_name: str = "intfloat/e5-large-v2", cache_size: int = 20000):
        self.dim = dim
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.cache_size = cache_size
        self._cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()

    def _cache_get(self, key: str) -> Optional[torch.Tensor]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key].clone()
        return None

    def _cache_put(self, key: str, value: torch.Tensor):
        self._cache[key] = value.clone()
        self._cache.move_to_end(key)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)  # вытесняем самый старый (LRU)

    def get_embedding(self, text: str, is_query: bool = True) -> torch.Tensor:
        cache_key = f"{'q' if is_query else 'p'}::{text}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        prefixed = text
        if "e5" in self.model_name.lower():
            prefixed = ("query: " if is_query else "passage: ") + text

        inputs = self.tokenizer(prefixed, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
            emb = F.normalize(emb, p=2, dim=0)

        self._cache_put(cache_key, emb)
        return emb

    def get_embeddings_batch(self, texts: List[str], is_query: bool = True) -> List[torch.Tensor]:
        return [self.get_embedding(t, is_query=is_query) for t in texts]


def random_vector(dim: int) -> torch.Tensor:
    v = torch.randn(dim)
    return F.normalize(v, p=2, dim=0)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a.flatten(), p=2, dim=0)
    b = F.normalize(b.flatten(), p=2, dim=0)
    return float(torch.dot(a, b).item())


def hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def compute_importance(access_count: int, age: float, max_age: float = 30 * 24 * 3600) -> float:
    # важность = частота доступа * (1 - возраст/макс_возраст)
    age_factor = max(0, 1 - age / max_age)
    return (access_count + 1) * age_factor


def grow_parameter_in_optimizer(optimizer, old_param: nn.Parameter, new_param: nn.Parameter,
                                 num_old_rows: Optional[int] = None) -> bool:
    """
    КЛЮЧЕВОЙ ФИКС continual learning.

    Проблема: graph.py при добавлении узла/ребра создавал НОВЫЙ объект nn.Parameter
    (torch.cat + переприсвоение). Но self.optimizer = Adam(self.graph.parameters(), ...)
    был создан один раз в CognitiveBrain.__init__ и держит СНИМОК списка параметров на
    момент создания. Новый параметр в этот список не попадает -> градиенты на него
    считаются (backward доходит), но optimizer.step() их никогда не применяет.
    Итог: node_emb (а значит и всё "содержимое" узлов графа) на практике НЕ обучается
    после самого первого шага — отсюда повторяющиеся/не меняющиеся ответы.

    Это находит old_param в param_groups оптимизатора, заменяет его на new_param,
    и переносит накопленную Adam-статистику (exp_avg, exp_avg_sq, step) для первых
    num_old_rows строк, чтобы не терять историю обучения уже существующих узлов/рёбер.
    Новые строки (только что добавленный узел) стартуют со свежей статистикой Adam.
    """
    for group in optimizer.param_groups:
        params = group["params"]
        for i, p in enumerate(params):
            if p is old_param:
                params[i] = new_param
                old_state = optimizer.state.pop(old_param, None)
                if old_state:
                    new_state = {"step": old_state.get("step", 0)}
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in old_state:
                            old_buf = old_state[key]
                            new_buf = torch.zeros_like(new_param.data)
                            n = num_old_rows if num_old_rows is not None else old_buf.shape[0]
                            n = min(n, old_buf.shape[0], new_buf.shape[0]) if new_buf.dim() > 0 else old_buf.numel()
                            if new_buf.dim() > 0:
                                new_buf[:n] = old_buf[:n]
                            else:
                                new_buf = old_buf.clone()
                            new_state[key] = new_buf
                    optimizer.state[new_param] = new_state
                return True
    # old_param ещё не был в оптимизаторе (например, самый первый add_node до его создания) —
    # просто добавляем новый параметр, чтобы он в принципе обучался.
    if optimizer.param_groups:
        optimizer.param_groups[0]["params"].append(new_param)
        return True
    return False
