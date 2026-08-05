"""
Smart Brain v6 — Рефакторинг на PyTorch + NetworkX + SQLite
=============================================================
Ключевые улучшения:
1. PyTorch tensors вместо numpy для всех векторных операций
2. NetworkX DiGraph для топологии нейронов/синапсов
3. SQLite для иерархической памяти (вместо списков в RAM)
4. Бинарное сохранение через torch.save + pickle + gzip (вместо JSON)
5. nn.MultiheadAttention для синаптического внимания
6. nn.LSTMCell для нейронных состояний
7. Потокобезопасность через RLock
8. Чёткое разделение: NeuralGraph (топология+тензоры), HierarchicalMemory (SQLite),
   Brain (оркестратор), Teacher (оценка)

Автор: AI Assistant (рефакторинг v5 -> v6)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import sqlite3
import pickle
import gzip
import json
import os
import time
import re
import threading
import logging
import hashlib
import random
import copy
import zlib
import warnings
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
from openai import OpenAI

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================
# Настройка логирования
# ============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("SmartBrainV6")

# ============================
# Конфигурация
# ============================
LM_STUDIO_BASE_URL = os.environ.get("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
LM_STUDIO_API_KEY = os.environ.get("LM_STUDIO_API_KEY", "not-needed")


@dataclass
class BrainConfig:
    """Конфигурация мозга v6."""
    dim_embedding: int = 128
    input_neurons: int = 40
    output_neurons: int = 40
    hidden_layers: List[int] = field(default_factory=lambda: [100, 80, 60])
    max_neurons: int = 2000
    max_synapses: int = 20000
    model_dir: str = "brain_model_v6"          # <-- директория вместо файла

    # STDP
    stdp_tau: float = 2.0
    stdp_a_plus: float = 0.03
    stdp_a_minus: float = 0.032

    # Гомеостаз
    homeostasis_target_rate: float = 0.15
    homeostasis_every: int = 20

    # Память
    working_memory_size: int = 50
    dialog_memory_turns: int = 20
    short_memory_size: int = 1000
    long_memory_size: int = 100000
    max_kb_size: int = 5000

    # Обучение
    experience_buffer_size: int = 5000
    meta_learning_rate: float = 0.01

    # Прунинг
    prune_every: int = 25
    similarity_threshold_new_neuron: float = 0.85
    synapse_weight_threshold: float = 0.005
    synapse_max_age: int = 3600 * 24 * 30

    # Propagation
    max_propagation_steps: int = 20
    propagation_threshold: float = 0.25
    attention_heads: int = 4

    # Сохранение
    auto_save_every: int = 3


# ============================
# Вспомогательные функции
# ============================
def random_vector_torch(dim: int, seed: Optional[int] = None) -> torch.Tensor:
    """Случайный нормализованный torch-вектор."""
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
        v = torch.randn(dim, generator=gen, dtype=torch.float32)
    else:
        v = torch.randn(dim, dtype=torch.float32)
    return F.normalize(v.unsqueeze(0), p=2, dim=1).squeeze(0)


def cosine_similarity_torch(a: torch.Tensor, b: torch.Tensor) -> float:
    """Косинусное сходство для torch тензоров."""
    if a is None or b is None:
        return 0.0
    a = a.flatten().float()
    b = b.flatten().float()
    na = torch.norm(a)
    nb = torch.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(torch.dot(a, b) / (na * nb))


def sigmoid_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(torch.clamp(x, -20, 20))


def softmax_torch(x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    x = x.float() / max(temperature, 1e-8)
    e = torch.exp(x - torch.max(x))
    return e / (e.sum() + 1e-10)


def gelu_torch(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
    ))


def hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ============================
# Embedding Provider (torch)
# ============================
class EmbeddingProvider:
    """
    Провайдер эмбеддингов на torch с LRU-кэшем.
    Возвращает torch.Tensor вместо numpy.ndarray.
    """
    def __init__(self, dim: int = 128, api_model: str = "local-model",
                 llm_client: Optional[OpenAI] = None, cache_ttl: int = 3600):
        self.dim = dim
        self.api_model = api_model
        self.llm_client = llm_client
        self._cache: Dict[str, Tuple[torch.Tensor, float]] = {}
        self._cache_lock = threading.Lock()
        self._api_available = True
        self._projection: Optional[torch.Tensor] = None
        self.cache_ttl = cache_ttl
        self._cache_hits = 0
        self._cache_misses = 0

    def _local_embedding(self, text: str) -> torch.Tensor:
        text = text.lower().strip()
        text = re.sub(r"[^\\w\\s]", " ", text)
        tokens = text.split()
        if not tokens:
            return random_vector_torch(self.dim)

        ngrams = tokens[:]
        for i in range(len(tokens) - 1):
            ngrams.append(f"{tokens[i]}_{tokens[i+1]}")
        for i in range(len(tokens) - 2):
            ngrams.append(f"{tokens[i]}_{tokens[i+1]}_{tokens[i+2]}")

        vec = torch.zeros(self.dim, dtype=torch.float32)
        for idx, token in enumerate(ngrams):
            h = zlib.adler32(token.encode("utf-8")) + idx * 31
            rng = np.random.RandomState(h % (2 ** 31))
            tok_vec = torch.from_numpy(rng.randn(self.dim).astype(np.float32))
            weight = 1.0 / (1.0 + idx * 0.05)
            vec += tok_vec * weight

        norm = torch.norm(vec)
        return vec / norm if norm > 0 else random_vector_torch(self.dim)

    def _project_to_dim(self, vec: torch.Tensor) -> torch.Tensor:
        src_dim = vec.shape[0]
        if src_dim == self.dim:
            return vec
        if self._projection is None or self._projection.shape[0] != src_dim:
            rng = np.random.RandomState(1234567)
            proj = rng.randn(src_dim, self.dim).astype(np.float32) / np.sqrt(self.dim)
            self._projection = torch.from_numpy(proj)
        projected = vec @ self._projection
        return F.normalize(projected.unsqueeze(0), p=2, dim=1).squeeze(0)

    def get_embedding(self, text: str) -> torch.Tensor:
        if not text:
            return random_vector_torch(self.dim)

        cache_key = hash_text(text)
        with self._cache_lock:
            if cache_key in self._cache:
                vec, ts = self._cache[cache_key]
                if time.time() - ts < self.cache_ttl:
                    self._cache_hits += 1
                    return vec.clone()
                else:
                    del self._cache[cache_key]

        self._cache_misses += 1

        if self._api_available and self.llm_client is not None:
            try:
                resp = self.llm_client.embeddings.create(
                    model=self.api_model,
                    input=[text]
                )
                vec = torch.tensor(resp.data[0].embedding, dtype=torch.float32)
                if vec.shape[0] != self.dim:
                    vec = self._project_to_dim(vec)
                with self._cache_lock:
                    self._cache[cache_key] = (vec.clone(), time.time())
                return vec
            except Exception as e:
                logger.warning(f"API embedding failed: {e}, switching to local")
                self._api_available = False

        vec = self._local_embedding(text)
        with self._cache_lock:
            self._cache[cache_key] = (vec.clone(), time.time())
        return vec

    def get_embeddings_batch(self, texts: List[str]) -> List[torch.Tensor]:
        return [self.get_embedding(t) for t in texts]

    def get_cache_stats(self) -> Dict[str, Any]:
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "size": len(self._cache)
        }



# ============================
# NeuralGraph — граф нейронов (NetworkX + PyTorch)
# ============================
class NeuralGraph:
    """
    Управляет топологией нейронной сети через NetworkX DiGraph
    и хранит все тензоры в словарях, индексируемых ID.
    """
    def __init__(self, dim: int, max_neurons: int, max_synapses: int, device: str = "cpu"):
        self.dim = dim
        self.max_neurons = max_neurons
        self.max_synapses = max_synapses
        self.device = device

        # Топология — NetworkX DiGraph
        self.graph = nx.DiGraph()

        # --- Нейроны (torch tensors в CPU-RAM, при необходимости .to(device)) ---
        self.neuron_embeddings: Dict[int, torch.Tensor] = {}
        self.neuron_activations: Dict[int, float] = {}
        self.neuron_potentials: Dict[int, float] = {}
        self.neuron_importance: Dict[int, float] = {}
        self.neuron_energy: Dict[int, float] = {}
        self.neuron_clusters: Dict[int, str] = {}
        self.neuron_layers: Dict[int, int] = {}
        self.neuron_labels: Dict[int, Optional[str]] = {}
        self.neuron_utility: Dict[int, float] = {}
        self.neuron_usage: Dict[int, int] = {}
        self.neuron_age: Dict[int, int] = {}
        self.neuron_last_activation: Dict[int, float] = {}
        self.neuron_refractory: Dict[int, int] = {}
        self.neuron_pred_error: Dict[int, float] = {}
        self.neuron_target_rate: Dict[int, float] = {}
        # LSTM-like states
        self.neuron_cell: Dict[int, torch.Tensor] = {}   # cell state
        self.neuron_hidden: Dict[int, torch.Tensor] = {}  # hidden state
        self.neuron_activation_history: Dict[int, deque] = {}

        # --- Синапсы ---
        self.synapse_weights: Dict[int, float] = {}
        self.synapse_plasticity: Dict[int, float] = {}
        self.synapse_confidence: Dict[int, float] = {}
        self.synapse_frequency: Dict[int, float] = {}
        self.synapse_energy: Dict[int, float] = {}
        self.synapse_learning_rate: Dict[int, float] = {}
        self.synapse_lr_momentum: Dict[int, float] = {}
        self.synapse_eligibility: Dict[int, float] = {}
        self.synapse_tag_strength: Dict[int, float] = {}
        self.synapse_structural_stability: Dict[int, float] = {}
        self.synapse_is_inhibitory: Dict[int, bool] = {}
        self.synapse_semantic: Dict[int, torch.Tensor] = {}
        self.synapse_episodic: Dict[int, torch.Tensor] = {}
        self.synapse_context: Dict[int, torch.Tensor] = {}
        self.synapse_reward: Dict[int, float] = {}
        self.synapse_pred_error: Dict[int, float] = {}
        self.synapse_usage_count: Dict[int, int] = {}
        self.synapse_creation_time: Dict[int, float] = {}
        self.synapse_last_used: Dict[int, float] = {}

        # edge -> synapse_id mapping
        self.edge_to_sid: Dict[Tuple[int, int], int] = {}

        # ID management
        self._next_nid = 1
        self._next_sid = 1
        self._lock = threading.RLock()

        # PyTorch modules (shared across neurons/synapses)
        self.lstm_cell = nn.LSTMCell(dim, dim).to(device)
        with torch.no_grad():
            for p in self.lstm_cell.parameters():
                p.data *= 0.01

        self.attention = nn.MultiheadAttention(
            dim, num_heads=4, batch_first=True, device=device
        )
        with torch.no_grad():
            for p in self.attention.parameters():
                p.data *= 0.01

    # ---------- ID allocation ----------
    def _alloc_nid(self) -> int:
        nid = self._next_nid
        self._next_nid += 1
        return nid

    def _alloc_sid(self) -> int:
        sid = self._next_sid
        self._next_sid += 1
        return sid

    # ---------- Neuron CRUD ----------
    def add_neuron(self, embedding: torch.Tensor, cluster: str = "hidden",
                   layer: int = 0, label: Optional[str] = None,
                   importance: float = 0.5, energy: float = 1.0) -> int:
        with self._lock:
            if len(self.neuron_embeddings) >= self.max_neurons:
                self._prune_neurons()
                if len(self.neuron_embeddings) >= self.max_neurons:
                    logger.warning("Neuron limit reached")
                    return random.choice(list(self.neuron_embeddings.keys()))

            nid = self._alloc_nid()
            self.graph.add_node(nid)

            emb = F.normalize(embedding.unsqueeze(0), p=2, dim=1).squeeze(0)
            self.neuron_embeddings[nid] = emb
            self.neuron_activations[nid] = 0.0
            self.neuron_potentials[nid] = 0.0
            self.neuron_importance[nid] = importance
            self.neuron_energy[nid] = energy
            self.neuron_clusters[nid] = cluster
            self.neuron_layers[nid] = layer
            self.neuron_labels[nid] = label[:300] if label and len(label) > 300 else label
            self.neuron_utility[nid] = 0.0
            self.neuron_usage[nid] = 0
            self.neuron_age[nid] = 0
            self.neuron_last_activation[nid] = 0.0
            self.neuron_refractory[nid] = 0
            self.neuron_pred_error[nid] = 0.0
            self.neuron_target_rate[nid] = 0.15
            self.neuron_cell[nid] = torch.zeros(self.dim, device=self.device, dtype=torch.float32)
            self.neuron_hidden[nid] = torch.zeros(self.dim, device=self.device, dtype=torch.float32)
            self.neuron_activation_history[nid] = deque(maxlen=50)
            return nid

    def remove_neuron(self, nid: int):
        with self._lock:
            if nid not in self.neuron_embeddings:
                return
            # remove all connected synapses
            edges = list(self.graph.edges(data=True))
            for u, v, d in edges:
                if u == nid or v == nid:
                    sid = d.get("sid")
                    if sid is not None:
                        self._remove_synapse_by_sid(sid)
            self.graph.remove_node(nid)
            for d in [self.neuron_embeddings, self.neuron_activations, self.neuron_potentials,
                      self.neuron_importance, self.neuron_energy, self.neuron_clusters,
                      self.neuron_layers, self.neuron_labels, self.neuron_utility,
                      self.neuron_usage, self.neuron_age, self.neuron_last_activation,
                      self.neuron_refractory, self.neuron_pred_error, self.neuron_target_rate,
                      self.neuron_cell, self.neuron_hidden, self.neuron_activation_history]:
                d.pop(nid, None)

    # ---------- Synapse CRUD ----------
    def add_synapse(self, from_id: int, to_id: int, weight: float = 0.1,
                    plasticity: float = 0.5, is_inhibitory: bool = False) -> Optional[int]:
        with self._lock:
            if from_id not in self.neuron_embeddings or to_id not in self.neuron_embeddings:
                return None
            if len(self.synapse_weights) >= self.max_synapses:
                self._prune_synapses()
                if len(self.synapse_weights) >= self.max_synapses:
                    logger.warning("Synapse limit reached")
                    return None

            key = (from_id, to_id)
            if key in self.edge_to_sid:
                sid = self.edge_to_sid[key]
                self.synapse_weights[sid] = weight
                return sid

            sid = self._alloc_sid()
            self.graph.add_edge(from_id, to_id, sid=sid)
            self.edge_to_sid[key] = sid

            self.synapse_weights[sid] = weight
            self.synapse_plasticity[sid] = plasticity
            self.synapse_confidence[sid] = 0.1
            self.synapse_frequency[sid] = 0.0
            self.synapse_energy[sid] = 1.0
            self.synapse_learning_rate[sid] = 0.01
            self.synapse_lr_momentum[sid] = 0.0
            self.synapse_eligibility[sid] = 0.0
            self.synapse_tag_strength[sid] = 0.0
            self.synapse_structural_stability[sid] = 1.0
            self.synapse_is_inhibitory[sid] = is_inhibitory
            self.synapse_reward[sid] = 0.0
            self.synapse_pred_error[sid] = 0.0
            self.synapse_usage_count[sid] = 0
            self.synapse_creation_time[sid] = time.time()
            self.synapse_last_used[sid] = time.time()
            self.synapse_semantic[sid] = random_vector_torch(self.dim)
            self.synapse_episodic[sid] = random_vector_torch(self.dim)
            self.synapse_context[sid] = random_vector_torch(self.dim)
            return sid

    def _remove_synapse_by_sid(self, sid: int):
        if sid not in self.synapse_weights:
            return
        # find edge
        edge = None
        for u, v, d in self.graph.edges(data=True):
            if d.get("sid") == sid:
                edge = (u, v)
                break
        if edge:
            self.graph.remove_edge(*edge)
            self.edge_to_sid.pop(edge, None)
        for d in [self.synapse_weights, self.synapse_plasticity, self.synapse_confidence,
                  self.synapse_frequency, self.synapse_energy, self.synapse_learning_rate,
                  self.synapse_lr_momentum, self.synapse_eligibility, self.synapse_tag_strength,
                  self.synapse_structural_stability, self.synapse_is_inhibitory,
                  self.synapse_reward, self.synapse_pred_error, self.synapse_usage_count,
                  self.synapse_creation_time, self.synapse_last_used,
                  self.synapse_semantic, self.synapse_episodic, self.synapse_context]:
            d.pop(sid, None)

    def remove_synapse(self, sid: int):
        with self._lock:
            self._remove_synapse_by_sid(sid)

    # ---------- Pruning ----------
    def _prune_neurons(self):
        with self._lock:
            if len(self.neuron_embeddings) < int(self.max_neurons * 0.9):
                return
            scores = []
            for nid in self.neuron_embeddings:
                if self.neuron_clusters[nid] in ("input", "output") and self.neuron_labels[nid]:
                    scores.append((float("inf"), nid))
                else:
                    hist = self.neuron_activation_history.get(nid, deque(maxlen=50))
                    temporal = float(np.mean(list(hist)[-5:])) if len(hist) >= 5 else 0.5
                    score = (self.neuron_energy[nid] * 0.25 +
                             self.neuron_importance[nid] * 0.25 +
                             self.neuron_utility[nid] * 0.25 +
                             temporal * 0.15 +
                             (self.neuron_usage[nid] / (self.neuron_age[nid] + 1)) * 0.1)
                    scores.append((score, nid))
            scores.sort(key=lambda x: x[0])
            target = int(self.max_neurons * 0.8)
            to_remove = [nid for sc, nid in scores if sc != float("inf")]
            to_remove = to_remove[:max(0, len(scores) - target)]
            for nid in to_remove:
                self.remove_neuron(nid)
            if to_remove:
                logger.info(f"Pruned {len(to_remove)} neurons")

    def _prune_synapses(self, force: bool = False):
        with self._lock:
            if not force and len(self.synapse_weights) < int(self.max_synapses * 0.9):
                to_remove = []
                now = time.time()
                for sid in list(self.synapse_weights.keys()):
                    w = self.synapse_weights[sid]
                    if w > 0:
                        self.synapse_weights[sid] = max(0.0, w - 0.0005)
                    elif w < 0:
                        self.synapse_weights[sid] = min(0.0, w + 0.0005)
                    self.synapse_confidence[sid] = max(0.0, self.synapse_confidence[sid] - 0.00025)
                    self.synapse_energy[sid] = max(0.0, self.synapse_energy[sid] - 0.0001)
                    self.synapse_frequency[sid] *= 0.999
                    self.synapse_eligibility[sid] *= 0.9
                    self.synapse_tag_strength[sid] *= 0.95
                    self.synapse_structural_stability[sid] *= 0.9999

                    # find source neuron for age check
                    src = None
                    for u, v, d in self.graph.edges(data=True):
                        if d.get("sid") == sid:
                            src = u
                            break
                    age = now - self.synapse_last_used.get(sid, now)
                    if (self.synapse_structural_stability[sid] < 0.1 or
                        (abs(self.synapse_weights[sid]) < 0.01 and age > 3600 * 24 * 30) or
                        (self.synapse_usage_count[sid] == 0 and
                         now - self.synapse_creation_time.get(sid, now) > 3600 * 24 * 30)):
                        to_remove.append(sid)
                for sid in to_remove:
                    self._remove_synapse_by_sid(sid)
                if to_remove:
                    logger.info(f"Pruned {len(to_remove)} dead synapses")
                return

            target = int(self.max_synapses * 0.7)
            scored = []
            for sid in self.synapse_weights:
                sc = (abs(self.synapse_weights[sid]) * 0.3 +
                      self.synapse_frequency[sid] * 0.2 +
                      self.synapse_confidence[sid] * 0.15 +
                      self.synapse_tag_strength[sid] * 0.15 +
                      (self.synapse_usage_count[sid] /
                       (time.time() - self.synapse_creation_time.get(sid, time.time()) + 1)) * 0.1 +
                      self.synapse_structural_stability[sid] * 0.1)
                scored.append((sc, sid))
            scored.sort(key=lambda x: x[0])
            to_remove = []
            for sc, sid in scored:
                if len(self.synapse_weights) - len(to_remove) <= target:
                    break
                to_remove.append(sid)
            for sid in to_remove:
                self._remove_synapse_by_sid(sid)
            if to_remove:
                logger.info(f"Pruned {len(to_remove)} synapses (forced)")

    # ---------- Queries ----------
    def get_outgoing(self, nid: int) -> List[Tuple[int, int, float]]:
        """Returns list of (target_nid, synapse_id, weight)."""
        if nid not in self.graph:
            return []
        result = []
        for _, target, d in self.graph.out_edges(nid, data=True):
            sid = d.get("sid")
            if sid is not None and sid in self.synapse_weights:
                result.append((target, sid, self.synapse_weights[sid]))
        return result

    def get_incoming(self, nid: int) -> List[Tuple[int, int, float]]:
        """Returns list of (source_nid, synapse_id, weight)."""
        if nid not in self.graph:
            return []
        result = []
        for source, _, d in self.graph.in_edges(nid, data=True):
            sid = d.get("sid")
            if sid is not None and sid in self.synapse_weights:
                result.append((source, sid, self.synapse_weights[sid]))
        return result

    def find_most_similar(self, embedding: torch.Tensor, cluster: Optional[str] = None,
                          threshold: float = 0.85) -> Optional[int]:
        best_id = None
        best_sim = threshold
        for nid, emb in self.neuron_embeddings.items():
            if cluster is not None and self.neuron_clusters.get(nid) != cluster:
                continue
            sim = cosine_similarity_torch(emb, embedding)
            if sim > best_sim:
                best_sim = sim
                best_id = nid
        return best_id

    # ---------- Attention (PyTorch native) ----------
    def compute_attention(self, query: torch.Tensor, keys: List[torch.Tensor],
                          values: List[torch.Tensor]) -> torch.Tensor:
        if not keys or not values:
            return torch.zeros(self.dim, device=self.device, dtype=torch.float32)
        q = query.unsqueeze(0).unsqueeze(0)          # (1, 1, dim)
        k = torch.stack(keys).unsqueeze(0)           # (1, n, dim)
        v = torch.stack(values).unsqueeze(0)         # (1, n, dim)
        out, _ = self.attention(q, k, v)
        return out.squeeze(0).squeeze(0)

    def attention_score(self, query: torch.Tensor, key: torch.Tensor) -> float:
        """Упрощённый скор через dot-product."""
        q = F.normalize(query.unsqueeze(0), p=2, dim=1).squeeze(0)
        k = F.normalize(key.unsqueeze(0), p=2, dim=1).squeeze(0)
        return float(torch.dot(q, k))

    # ---------- LSTM update ----------
    def update_lstm(self, nid: int, input_vec: torch.Tensor) -> torch.Tensor:
        if nid not in self.neuron_cell:
            return torch.zeros(self.dim, device=self.device, dtype=torch.float32)
        h = self.neuron_hidden[nid].unsqueeze(0)
        c = self.neuron_cell[nid].unsqueeze(0)
        x = input_vec.unsqueeze(0)
        h_new, c_new = self.lstm_cell(x, (h, c))
        self.neuron_hidden[nid] = h_new.squeeze(0)
        self.neuron_cell[nid] = c_new.squeeze(0)
        return self.neuron_hidden[nid]

    # ---------- Save / Load ----------
    def save(self, path: str):
        """
        Бинарное сохранение:
        - tensors.pt   : torch tensors (эффективно, компактно)
        - graph.pkl.gz : NetworkX graph + scalar атрибуты (gzip-сжатие)
        """
        os.makedirs(path, exist_ok=True)

        # 1. Torch tensors
        tensor_state = {
            "neuron_embeddings": {k: v.cpu() for k, v in self.neuron_embeddings.items()},
            "neuron_cell":      {k: v.cpu() for k, v in self.neuron_cell.items()},
            "neuron_hidden":    {k: v.cpu() for k, v in self.neuron_hidden.items()},
            "synapse_semantic": {k: v.cpu() for k, v in self.synapse_semantic.items()},
            "synapse_episodic": {k: v.cpu() for k, v in self.synapse_episodic.items()},
            "synapse_context":  {k: v.cpu() for k, v in self.synapse_context.items()},
            "lstm_state_dict":  self.lstm_cell.state_dict(),
            "attn_state_dict":  self.attention.state_dict(),
            "next_nid":         self._next_nid,
            "next_sid":         self._next_sid,
        }
        torch.save(tensor_state, os.path.join(path, "tensors.pt"))

        # 2. Graph topology + scalars via pickle+gzip
        graph_data = {
            "graph_dict": nx.to_dict_of_dicts(self.graph),
            "neuron_activations": self.neuron_activations,
            "neuron_potentials": self.neuron_potentials,
            "neuron_importance": self.neuron_importance,
            "neuron_energy": self.neuron_energy,
            "neuron_clusters": self.neuron_clusters,
            "neuron_layers": self.neuron_layers,
            "neuron_labels": self.neuron_labels,
            "neuron_utility": self.neuron_utility,
            "neuron_usage": self.neuron_usage,
            "neuron_age": self.neuron_age,
            "neuron_last_activation": self.neuron_last_activation,
            "neuron_refractory": self.neuron_refractory,
            "neuron_pred_error": self.neuron_pred_error,
            "neuron_target_rate": self.neuron_target_rate,
            "synapse_weights": self.synapse_weights,
            "synapse_plasticity": self.synapse_plasticity,
            "synapse_confidence": self.synapse_confidence,
            "synapse_frequency": self.synapse_frequency,
            "synapse_energy": self.synapse_energy,
            "synapse_learning_rate": self.synapse_learning_rate,
            "synapse_lr_momentum": self.synapse_lr_momentum,
            "synapse_eligibility": self.synapse_eligibility,
            "synapse_tag_strength": self.synapse_tag_strength,
            "synapse_structural_stability": self.synapse_structural_stability,
            "synapse_is_inhibitory": self.synapse_is_inhibitory,
            "synapse_reward": self.synapse_reward,
            "synapse_pred_error": self.synapse_pred_error,
            "synapse_usage_count": self.synapse_usage_count,
            "synapse_creation_time": self.synapse_creation_time,
            "synapse_last_used": self.synapse_last_used,
            "edge_to_sid": self.edge_to_sid,
        }
        with gzip.open(os.path.join(path, "graph.pkl.gz"), "wb") as f:
            pickle.dump(graph_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"NeuralGraph saved: {len(self.neuron_embeddings)} neurons, "
                    f"{len(self.synapse_weights)} synapses -> {path}")

    def load(self, path: str):
        t_path = os.path.join(path, "tensors.pt")
        g_path = os.path.join(path, "graph.pkl.gz")
        if not (os.path.exists(t_path) and os.path.exists(g_path)):
            logger.info("Model files not found, starting fresh")
            return

        # 1. Load tensors
        t_state = torch.load(t_path, map_location=self.device, weights_only=False)
        self.neuron_embeddings = {k: v for k, v in t_state["neuron_embeddings"].items()}
        self.neuron_cell = {k: v for k, v in t_state["neuron_cell"].items()}
        self.neuron_hidden = {k: v for k, v in t_state["neuron_hidden"].items()}
        self.synapse_semantic = {k: v for k, v in t_state["synapse_semantic"].items()}
        self.synapse_episodic = {k: v for k, v in t_state["synapse_episodic"].items()}
        self.synapse_context = {k: v for k, v in t_state["synapse_context"].items()}
        self.lstm_cell.load_state_dict(t_state["lstm_state_dict"])
        self.attention.load_state_dict(t_state["attn_state_dict"])
        self._next_nid = t_state["next_nid"]
        self._next_sid = t_state["next_sid"]

        # 2. Load graph
        with gzip.open(g_path, "rb") as f:
            g_data = pickle.load(f)

        self.graph = nx.from_dict_of_dicts(g_data["graph_dict"], create_using=nx.DiGraph())
        self.neuron_activations = g_data["neuron_activations"]
        self.neuron_potentials = g_data["neuron_potentials"]
        self.neuron_importance = g_data["neuron_importance"]
        self.neuron_energy = g_data["neuron_energy"]
        self.neuron_clusters = g_data["neuron_clusters"]
        self.neuron_layers = g_data["neuron_layers"]
        self.neuron_labels = g_data["neuron_labels"]
        self.neuron_utility = g_data["neuron_utility"]
        self.neuron_usage = g_data["neuron_usage"]
        self.neuron_age = g_data["neuron_age"]
        self.neuron_last_activation = g_data["neuron_last_activation"]
        self.neuron_refractory = g_data.get("neuron_refractory", {})
        self.neuron_pred_error = g_data.get("neuron_pred_error", {})
        self.neuron_target_rate = g_data.get("neuron_target_rate", {})
        self.synapse_weights = g_data["synapse_weights"]
        self.synapse_plasticity = g_data["synapse_plasticity"]
        self.synapse_confidence = g_data["synapse_confidence"]
        self.synapse_frequency = g_data["synapse_frequency"]
        self.synapse_energy = g_data["synapse_energy"]
        self.synapse_learning_rate = g_data["synapse_learning_rate"]
        self.synapse_lr_momentum = g_data.get("synapse_lr_momentum", {})
        self.synapse_eligibility = g_data["synapse_eligibility"]
        self.synapse_tag_strength = g_data["synapse_tag_strength"]
        self.synapse_structural_stability = g_data["synapse_structural_stability"]
        self.synapse_is_inhibitory = g_data["synapse_is_inhibitory"]
        self.synapse_reward = g_data.get("synapse_reward", {})
        self.synapse_pred_error = g_data.get("synapse_pred_error", {})
        self.synapse_usage_count = g_data["synapse_usage_count"]
        self.synapse_creation_time = g_data["synapse_creation_time"]
        self.synapse_last_used = g_data["synapse_last_used"]
        self.edge_to_sid = g_data["edge_to_sid"]

        # Rebuild activation histories
        self.neuron_activation_history = {}
        for nid in self.neuron_embeddings:
            self.neuron_activation_history[nid] = deque(maxlen=50)

        logger.info(f"NeuralGraph loaded: {len(self.neuron_embeddings)} neurons, "
                    f"{len(self.synapse_weights)} synapses")



# ============================
# Hierarchical Memory (SQLite)
# ============================
class MemoryLevel(Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class HierarchicalMemory:
    """
    Иерархическая память на SQLite.
    Эмбеддинги хранятся как BLOB (torch tensor -> numpy bytes).
    Ретрив через numpy dot product (вместо FAISS — нет в окружении).
    """
    def __init__(self, db_path: str, config: BrainConfig):
        self.db_path = db_path
        self.config = config
        self._lock = threading.RLock()
        self._init_db()
        self.context_embedding: Optional[torch.Tensor] = None

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    memory_level TEXT NOT NULL,
                    importance REAL DEFAULT 0.5,
                    tags TEXT DEFAULT '[]',
                    timestamp REAL DEFAULT 0,
                    access_count INTEGER DEFAULT 0,
                    last_accessed REAL DEFAULT 0,
                    consolidation_score REAL DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_level ON memory_items(memory_level)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memory_items(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_importance ON memory_items(importance)")
            conn.commit()

    def _encode_embedding(self, emb: torch.Tensor) -> bytes:
        return emb.detach().cpu().numpy().tobytes()

    def _decode_embedding(self, blob: bytes) -> torch.Tensor:
        arr = np.frombuffer(blob, dtype=np.float32)
        return torch.from_numpy(arr)

    def add(self, content: Any, embedding: torch.Tensor,
            memory_level: MemoryLevel = MemoryLevel.WORKING,
            importance: float = 0.5, tags: List[str] = None):
        with self._lock:
            tags_json = json.dumps(tags or [], ensure_ascii=False)
            emb_blob = self._encode_embedding(embedding)
            ts = time.time()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO memory_items
                       (content, embedding, memory_level, importance, tags, timestamp, last_accessed)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (json.dumps(content, ensure_ascii=False), emb_blob, memory_level.value,
                     importance, tags_json, ts, ts)
                )
                conn.commit()

            # Consolidation: если эпизодическая память переполнена
            if memory_level == MemoryLevel.EPISODIC:
                self._maybe_consolidate()

    def _maybe_consolidate(self):
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM memory_items WHERE memory_level=?",
                (MemoryLevel.EPISODIC.value,)
            ).fetchone()[0]
            if count <= self.config.short_memory_size:
                return
            # Переносим старые, часто используемые элементы в семантическую
            rows = conn.execute(
                """SELECT id, content, embedding, importance, tags, timestamp, access_count
                   FROM memory_items
                   WHERE memory_level=? AND timestamp < ? AND access_count > 2
                   ORDER BY timestamp ASC LIMIT ?""",
                (MemoryLevel.EPISODIC.value, time.time() - 3600,
                 count - self.config.short_memory_size)
            ).fetchall()
            for row in rows:
                rid, content, emb_blob, importance, tags_json, ts, access_count = row
                conn.execute(
                    "UPDATE memory_items SET memory_level=?, consolidation_score=consolidation_score+0.1 WHERE id=?",
                    (MemoryLevel.SEMANTIC.value, rid)
                )
            conn.commit()
            if rows:
                logger.info(f"Consolidated {len(rows)} episodic items to semantic memory")

    def retrieve(self, query_embedding: torch.Tensor, top_k: int = 5,
                 memory_levels: List[MemoryLevel] = None,
                 min_importance: float = 0.0) -> List[Dict[str, Any]]:
        with self._lock:
            levels = memory_levels or [MemoryLevel.WORKING, MemoryLevel.EPISODIC, MemoryLevel.SEMANTIC]
            level_names = [l.value for l in levels]
            placeholders = ",".join("?" * len(level_names))

            candidates = []
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    f"""SELECT id, content, embedding, importance, memory_level,
                               timestamp, access_count
                        FROM memory_items
                        WHERE memory_level IN ({placeholders}) AND importance >= ?""",
                    level_names + [min_importance]
                ).fetchall()

            q_emb = F.normalize(query_embedding.unsqueeze(0), p=2, dim=1).squeeze(0)
            for row in rows:
                rid, content, emb_blob, importance, level, ts, access_count = row
                emb = self._decode_embedding(emb_blob)
                emb = F.normalize(emb.unsqueeze(0), p=2, dim=1).squeeze(0)
                sim = float(torch.dot(q_emb, emb))
                recency_bonus = 1.0 / (1.0 + (time.time() - ts) / 3600)
                score = sim * (0.5 + 0.5 * importance) * recency_bonus
                candidates.append((score, rid, content, importance, level, access_count))

            candidates.sort(key=lambda x: x[0], reverse=True)
            results = []
            with sqlite3.connect(self.db_path) as conn:
                for score, rid, content, importance, level, access_count in candidates[:top_k]:
                    conn.execute(
                        "UPDATE memory_items SET access_count=access_count+1, last_accessed=? WHERE id=?",
                        (time.time(), rid)
                    )
                    results.append({
                        "id": rid,
                        "content": json.loads(content),
                        "score": score,
                        "importance": importance,
                        "level": level,
                        "access_count": access_count + 1
                    })
                conn.commit()
            return results

    def decay_all(self, rate: float = 0.001):
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                now = time.time()
                conn.execute(
                    """UPDATE memory_items
                       SET importance = importance * (1.0 - ?) * exp(-( ? - last_accessed) / (3600*24))
                       WHERE memory_level = ?""",
                    (rate, now, MemoryLevel.EPISODIC.value)
                )
                conn.execute(
                    """UPDATE memory_items
                       SET importance = importance * (1.0 - ?) * exp(-( ? - last_accessed) / (3600*24*7))
                       WHERE memory_level = ?""",
                    (rate * 0.5, now, MemoryLevel.SEMANTIC.value)
                )
                conn.execute("DELETE FROM memory_items WHERE importance < 0.01 AND memory_level != ?",
                             (MemoryLevel.WORKING.value,))
                conn.commit()

    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                stats = {}
                for level in MemoryLevel:
                    count = conn.execute(
                        "SELECT COUNT(*) FROM memory_items WHERE memory_level=?",
                        (level.value,)
                    ).fetchone()[0]
                    stats[level.value] = count
                return stats

    def clear(self):
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM memory_items")
                conn.commit()


# ============================
# Experience Buffer (torch-based)
# ============================
class Experience:
    def __init__(self, state: torch.Tensor, action: str, reward: float,
                 next_state: torch.Tensor, priority: float = 1.0,
                 metadata: Optional[Dict] = None):
        self.state = state.detach().cpu().clone()
        self.action = action
        self.reward = reward
        self.next_state = next_state.detach().cpu().clone()
        self.priority = priority
        self.timestamp = time.time()
        self.metadata = metadata or {}
        self.visit_count = 0


class ExperienceBuffer:
    """Prioritized Experience Replay с torch tensors."""
    def __init__(self, capacity: int = 5000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer: List[Experience] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self._lock = threading.Lock()

    def add(self, experience: Experience):
        with self._lock:
            max_prio = self.priorities.max() if len(self.buffer) > 0 else 1.0
            if len(self.buffer) < self.capacity:
                self.buffer.append(experience)
            else:
                self.buffer[self.position] = experience
            self.priorities[self.position] = max_prio
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        with self._lock:
            if len(self.buffer) == 0:
                return [], np.array([]), np.array([])
            n = len(self.buffer)
            probs = self.priorities[:n] ** self.alpha
            probs /= probs.sum()
            indices = np.random.choice(n, size=min(batch_size, n), replace=False, p=probs)
            weights = (n * probs[indices]) ** (-self.beta)
            weights /= weights.max()
            experiences = [self.buffer[i] for i in indices]
            for exp in experiences:
                exp.visit_count += 1
            return experiences, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        with self._lock:
            for idx, prio in zip(indices, priorities):
                if idx < len(self.buffer):
                    self.priorities[idx] = prio

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            if not self.buffer:
                return {"size": 0}
            rewards = [e.reward for e in self.buffer]
            return {
                "size": len(self.buffer),
                "mean_reward": float(np.mean(rewards)),
                "max_reward": float(np.max(rewards)),
                "min_reward": float(np.min(rewards)),
                "mean_priority": float(np.mean(self.priorities[:len(self.buffer)])),
            }


# ============================
# Signal
# ============================
class Signal:
    def __init__(self, embedding: torch.Tensor, energy: float = 1.0, importance: float = 0.5,
                 context: Optional[torch.Tensor] = None, source=None, destination=None,
                 confidence: float = 0.5, text: str = "",
                 signal_type: str = "standard",
                 metadata: Optional[Dict] = None):
        self.id = id(self)
        self.embedding = embedding.detach().cpu().clone().float()
        self.energy = energy
        self.importance = importance
        self.timestamp = time.time()
        self.context = context.detach().cpu().clone().float() if context is not None else None
        self.source = source
        self.destination = destination
        self.confidence = confidence
        self.history = []
        self.text = text
        self.signal_type = signal_type
        self.metadata = metadata or {}
        self.hop_count = 0

    def __repr__(self):
        return f"Signal(id={self.id}, energy={self.energy:.2f}, type={self.signal_type}, text={self.text[:30]!r})"


# ============================
# Self-Reflection
# ============================
class SelfReflection:
    def __init__(self, llm_client: Optional[OpenAI] = None):
        self.llm_client = llm_client
        self.reflection_history: List[Dict] = []
        self.confidence_threshold = 0.6

    def reflect_on_answer(self, question: str, answer: str,
                          facts: List[Dict], brain_state: Dict) -> Dict:
        reflection = {
            "confidence": 0.5,
            "is_hallucination": False,
            "gaps": [],
            "suggestion": None,
            "needs_clarification": False,
        }
        if facts:
            fact_scores = [f.get("score", 0) for f in facts]
            avg_fact_score = np.mean(fact_scores) if fact_scores else 0
            reflection["confidence"] = 0.3 + 0.7 * avg_fact_score
        else:
            reflection["confidence"] = 0.3
            reflection["needs_clarification"] = True

        if len(facts) >= 2:
            for i in range(len(facts)):
                for j in range(i + 1, len(facts)):
                    a1, a2 = facts[i].get("a", ""), facts[j].get("a", "")
                    if a1 and a2 and a1 != a2:
                        reflection["confidence"] *= 0.8
                        reflection["gaps"].append(f"Противоречие: {a1} vs {a2}")

        if self.llm_client:
            try:
                system_msg = (
                    "Ты — система саморефлексии ИИ. Оцени достоверность ответа. "
                    "Ответь только числом от 0 до 1."
                )
                response = self.llm_client.chat.completions.create(
                    model="local-model",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": f"Вопрос: {question}\\nОтвет: {answer}"}
                    ],
                    max_tokens=10,
                    temperature=0.1,
                )
                llm_confidence = float(response.choices[0].message.content.strip()[:5])
                reflection["confidence"] = 0.5 * reflection["confidence"] + 0.5 * llm_confidence
            except Exception:
                pass

        if reflection["confidence"] < self.confidence_threshold:
            reflection["needs_clarification"] = True
            reflection["suggestion"] = self._generate_clarifying_question(question, answer, reflection["gaps"])

        self.reflection_history.append({
            "question": question,
            "answer": answer,
            "reflection": reflection,
            "timestamp": time.time(),
        })
        return reflection

    def _generate_clarifying_question(self, question: str, answer: str, gaps: List[str]) -> Optional[str]:
        if not gaps:
            return f"Можете уточнить, что именно вы имели в виду в вопросе '{question}'?"
        return f"Я не уверен в следующем: {gaps[0]}. Можете уточнить?"

    def get_reflection_stats(self) -> Dict:
        if not self.reflection_history:
            return {}
        confidences = [r["reflection"]["confidence"] for r in self.reflection_history]
        return {
            "total_reflections": len(self.reflection_history),
            "mean_confidence": float(np.mean(confidences)),
            "hallucination_rate": sum(1 for r in self.reflection_history if r["reflection"]["is_hallucination"]) / len(self.reflection_history),
        }



# ============================
# Legacy Memory (для обратной совместимости)
# ============================
class DialogMemory:
    def __init__(self, max_turns: int = 20):
        self.items = []
        self.max_size = max_turns
        self.turn_embeddings = []

    def add_turn(self, user_text: str, assistant_text: str, embedding: torch.Tensor):
        self.items.append({"user": user_text, "assistant": assistant_text, "time": time.time()})
        self.turn_embeddings.append(embedding.detach().cpu().clone())
        if len(self.items) > self.max_size:
            self.items.pop(0)
            self.turn_embeddings.pop(0)

    def get_context_string(self, n: int = 3) -> str:
        recent = self.items[-n:] if self.items else []
        parts = []
        for turn in recent:
            parts.append(f"Пользователь: {turn['user']}")
            parts.append(f"Ассистент: {turn['assistant']}")
        return "\\n".join(parts)

    def get_context_embedding(self) -> Optional[torch.Tensor]:
        if not self.turn_embeddings:
            return None
        vec = torch.stack(self.turn_embeddings).mean(dim=0)
        return F.normalize(vec.unsqueeze(0), p=2, dim=1).squeeze(0)

    def clear(self):
        self.items.clear()
        self.turn_embeddings.clear()


class ShortMemory:
    def __init__(self, max_size: int = 1000):
        self.items = deque(maxlen=max_size)

    def add(self, item):
        self.items.append(item)

    def get_recent(self, n: int = 10):
        return list(self.items)[-n:]

    def clear(self):
        self.items.clear()


class LongMemory:
    def __init__(self, max_size: int = 100000):
        self.items = deque(maxlen=max_size)

    def add(self, item):
        self.items.append(item)


# ============================
# Brain v6 — Основной класс
# ============================
class Brain:
    def __init__(self, config: Optional[BrainConfig] = None,
                 llm_client: Optional[OpenAI] = None,
                 device: str = "cpu"):
        self.config = config or BrainConfig()
        self.llm_client = llm_client or OpenAI(
            base_url=LM_STUDIO_BASE_URL,
            api_key=LM_STUDIO_API_KEY,
        )
        self.device = device
        self.dim = self.config.dim_embedding
        self.embedder = EmbeddingProvider(dim=self.dim, llm_client=self.llm_client)

        # Neural graph
        self.graph = NeuralGraph(
            dim=self.dim,
            max_neurons=self.config.max_neurons,
            max_synapses=self.config.max_synapses,
            device=device
        )

        # Hierarchical memory (SQLite)
        os.makedirs(self.config.model_dir, exist_ok=True)
        self.memory = HierarchicalMemory(
            db_path=os.path.join(self.config.model_dir, "memory.db"),
            config=self.config
        )

        # Experience replay
        self.experience_buffer = ExperienceBuffer(
            capacity=self.config.experience_buffer_size
        )

        # Self-reflection
        self.reflection = SelfReflection(self.llm_client)

        # Legacy memory
        self.dialog_memory = DialogMemory(max_turns=self.config.dialog_memory_turns)
        self.short_memory = ShortMemory()
        self.long_memory = LongMemory()

        # Knowledge base (in-memory list for fast retrieval)
        self.knowledge_base: List[Dict[str, Any]] = []
        self.concept_index: Dict[str, int] = {}

        # Counters
        self.step_counter = 0
        self.global_time = time.time()
        self.coactivation_counter = defaultdict(int)
        self.coactivation_window = deque(maxlen=300)

        # Meta-learning
        self.meta_lr = self.config.meta_learning_rate
        self.performance_history = deque(maxlen=100)
        self._learn_counter = 0

        # Lock
        self.lock = self.graph._lock  # reuse graph lock

        # Init architecture
        self._init_architecture(
            self.config.input_neurons,
            self.config.hidden_layers,
            self.config.output_neurons
        )
        logger.info(f"Brain v6 initialized: {len(self.graph.neuron_embeddings)} neurons, "
                    f"{len(self.graph.synapse_weights)} synapses")

    # ---------- Architecture init ----------
    def _init_architecture(self, input_n: int, hidden_layers: List[int], output_n: int):
        # Input layer
        for _ in range(input_n):
            emb = random_vector_torch(self.dim)
            nid = self.graph.add_neuron(emb, cluster="input", layer=0)

        # Hidden layers
        layer_id = 1
        for hsize in hidden_layers:
            for _ in range(hsize):
                emb = random_vector_torch(self.dim)
                nid = self.graph.add_neuron(emb, cluster="hidden", layer=layer_id)
            layer_id += 1

        # Output layer
        for _ in range(output_n):
            emb = random_vector_torch(self.dim)
            nid = self.graph.add_neuron(emb, cluster="output", layer=layer_id)

        # Dense inter-layer connections
        layer_map = defaultdict(list)
        for nid, lyr in self.graph.neuron_layers.items():
            layer_map[lyr].append(nid)

        for layer_idx in range(layer_id):
            from_ids = layer_map[layer_idx]
            to_ids = layer_map.get(layer_idx + 1, [])
            if not to_ids:
                continue
            for from_id in from_ids:
                k = min(len(to_ids), max(8, len(to_ids) // 3))
                targets = random.sample(to_ids, k)
                for to_id in targets:
                    w = random.uniform(0.01, 0.15)
                    self.graph.add_synapse(from_id, to_id, weight=w)

        # Recurrent connections in hidden layers
        for layer_idx in range(1, layer_id):
            ids = layer_map[layer_idx]
            for _ in range(len(ids) // 4):
                a, b = random.sample(ids, 2)
                self.graph.add_synapse(a, b, weight=random.uniform(0.005, 0.05))
                self.graph.add_synapse(b, a, weight=random.uniform(0.005, 0.05))

        # Skip connections (input -> output)
        input_ids = [nid for nid, c in self.graph.neuron_clusters.items() if c == "input"]
        output_ids = [nid for nid, c in self.graph.neuron_clusters.items() if c == "output"]
        for _ in range(min(50, len(input_ids) * len(output_ids) // 3)):
            a = random.choice(input_ids)
            b = random.choice(output_ids)
            self.graph.add_synapse(a, b, weight=random.uniform(0.005, 0.02))

    # ---------- STDP & Hebbian ----------
    def _stdp_delta(self, pre_time: float, post_time: float) -> float:
        dt = post_time - pre_time
        if dt >= 0:
            return self.config.stdp_a_plus * np.exp(-dt / self.config.stdp_tau)
        else:
            return -self.config.stdp_a_minus * np.exp(dt / self.config.stdp_tau)

    def _hebbian_update(self, activated_ids: List[int], query_vec: torch.Tensor):
        for i in range(len(activated_ids)):
            for j in range(i + 1, len(activated_ids)):
                a, b = activated_ids[i], activated_ids[j]
                # Check both directions
                for u, v in [(a, b), (b, a)]:
                    sid = self.graph.edge_to_sid.get((u, v))
                    if sid is None or sid not in self.graph.synapse_weights:
                        continue
                    if self.graph.synapse_is_inhibitory.get(sid, False):
                        continue
                    syn_sem = self.graph.synapse_semantic.get(sid)
                    if syn_sem is None:
                        continue
                    att = self.graph.attention_score(query_vec, syn_sem)
                    pre_t = self.graph.neuron_last_activation.get(u, 0)
                    post_t = self.graph.neuron_last_activation.get(v, 0)
                    stdp = self._stdp_delta(pre_t, post_t)
                    w = self.graph.synapse_weights[sid]
                    delta = stdp * (1.0 - min(1.0, abs(w))) * (1.0 + att)
                    meta_factor = 1.0 + self.meta_lr * np.log(1 + self.graph.synapse_usage_count.get(sid, 0))
                    delta *= meta_factor
                    lr = self.graph.synapse_learning_rate.get(sid, 0.01)
                    pl = self.graph.synapse_plasticity.get(sid, 0.5)
                    effective = delta * lr * pl
                    self.graph.synapse_weights[sid] = float(np.clip(w + effective, -2.0, 2.0))
                    self.graph.synapse_plasticity[sid] = float(np.clip(
                        self.graph.synapse_plasticity.get(sid, 0.5) + 0.001, 0.0, 1.0))
                    self.graph.synapse_confidence[sid] = float(np.clip(
                        self.graph.synapse_confidence.get(sid, 0.1) + 0.001, 0.0, 1.0))
                    self.graph.synapse_frequency[sid] = self.graph.synapse_frequency.get(sid, 0) + 0.001
                    self.graph.synapse_usage_count[sid] = self.graph.synapse_usage_count.get(sid, 0) + 1
                    self.graph.synapse_last_used[sid] = time.time()
                    self.graph.synapse_energy[sid] = min(1.0, self.graph.synapse_energy.get(sid, 1.0) + 0.01)
                else:
                    pair = tuple(sorted((a, b)))
                    self.coactivation_counter[pair] += 1
                    self.coactivation_window.append((self.step_counter, a, b))

    def _homeostatic_scaling(self):
        for nid in self.graph.neuron_embeddings:
            age = self.graph.neuron_age.get(nid, 0)
            if age <= 0:
                continue
            usage = self.graph.neuron_usage.get(nid, 0)
            actual_rate = usage / age
            if actual_rate <= 1e-6:
                continue
            scale = self.config.homeostasis_target_rate / actual_rate
            scale = float(np.clip(scale, 0.9, 1.1))
            if abs(scale - 1.0) < 1e-4:
                continue
            for source, sid, _ in self.graph.get_incoming(nid):
                if not self.graph.synapse_is_inhibitory.get(sid, False):
                    w = self.graph.synapse_weights.get(sid, 0)
                    self.graph.synapse_weights[sid] = float(np.clip(w * scale, -2.0, 2.0))

    def _create_new_synapses_from_coactivation(self):
        for pair, count in list(self.coactivation_counter.items()):
            if count >= 5:  # threshold
                a, b = pair
                if a in self.graph.neuron_embeddings and b in self.graph.neuron_embeddings:
                    emb_a = self.graph.neuron_embeddings[a]
                    emb_b = self.graph.neuron_embeddings[b]
                    sim = cosine_similarity_torch(emb_a, emb_b)
                    if sim > 0.2:
                        self.graph.add_synapse(a, b, weight=0.05)
                        self.graph.add_synapse(b, a, weight=0.05)
                        self.coactivation_counter[pair] = 0

    # ---------- Propagation ----------
    def propagate_signal(self, input_signal: Signal, max_steps: int = None) -> Tuple[List[int], Dict[int, float]]:
        max_steps = max_steps or self.config.max_propagation_steps
        with self.lock:
            return self._propagate_signal_impl(input_signal, max_steps)

    def _propagate_signal_impl(self, input_signal: Signal, max_steps: int) -> Tuple[List[int], Dict[int, float]]:
        # Context
        dialog_ctx = self.dialog_memory.get_context_embedding()
        if dialog_ctx is not None:
            ctx = 0.7 * input_signal.embedding + 0.3 * dialog_ctx
            ctx = F.normalize(ctx.unsqueeze(0), p=2, dim=1).squeeze(0)
        else:
            ctx = input_signal.embedding

        # Start neurons (input layer)
        similarities = []
        for nid, emb in self.graph.neuron_embeddings.items():
            if self.graph.neuron_clusters.get(nid) == "input":
                sim = cosine_similarity_torch(emb, input_signal.embedding)
                similarities.append((sim, nid))
        similarities.sort(key=lambda x: x[0], reverse=True)
        start_neurons = [nid for sim, nid in similarities[:10] if sim > 0.05]
        if not start_neurons:
            existing = self.graph.find_most_similar(input_signal.embedding, cluster="input", threshold=0.2)
            if existing is not None:
                start_neurons = [existing]
            else:
                new_nid = self.graph.add_neuron(input_signal.embedding, cluster="input", layer=0)
                start_neurons = [new_nid]

        visited = set(start_neurons)
        queue = deque([(nid, 0, input_signal.energy) for nid in start_neurons])
        activated = []
        activation_map = {}
        residual = {nid: 0.0 for nid in start_neurons}

        while queue and len(visited) < 500:
            nid, steps, current_energy = queue.popleft()
            if steps > max_steps or current_energy < 0.01:
                continue

            outgoing = []
            for target, sid, weight in self.graph.get_outgoing(nid):
                if abs(weight) < 0.003 or weight < 0:
                    continue
                if self.graph.neuron_energy.get(target, 0) < 0.1:
                    continue
                if target in visited and target not in activated:
                    continue
                syn_sem = self.graph.synapse_semantic.get(sid)
                syn_epis = self.graph.synapse_episodic.get(sid)
                att = self.graph.attention_score(ctx, syn_sem) if syn_sem is not None else 0.0
                sem_sim = cosine_similarity_torch(syn_sem, ctx) if syn_sem is not None else 0.0
                epis_sim = cosine_similarity_torch(syn_epis, ctx) if syn_epis is not None else 0.0
                target_imp = self.graph.neuron_importance.get(target, 0.5)
                score = weight * 0.4 + att * 0.2 + sem_sim * 0.2 + epis_sim * 0.1 + target_imp * 0.1
                outgoing.append((score, sid, target, weight))

            if not outgoing:
                continue

            # Attention-based re-ranking
            if len(outgoing) > 1:
                attn_out = self.graph.update_lstm(nid, ctx)
                for idx, (score, sid, target, weight) in enumerate(outgoing):
                    tgt_emb = self.graph.neuron_embeddings.get(target)
                    if tgt_emb is not None:
                        attn_sim = cosine_similarity_torch(attn_out, tgt_emb)
                        outgoing[idx] = (score * (1.0 + attn_sim * 0.3), sid, target, weight)

            outgoing.sort(key=lambda x: x[0], reverse=True)

            energy_budget = current_energy * 0.8
            for score, sid, target, weight in outgoing[:10]:
                energy_cost = weight * (0.8 + 0.2 * score)
                if energy_cost > energy_budget:
                    continue
                energy_budget -= energy_cost

                # Update target neuron potential
                tgt_pot = self.graph.neuron_potentials.get(target, 0.0)
                sim = cosine_similarity_torch(self.graph.neuron_embeddings.get(target, torch.zeros(self.dim)),
                                               input_signal.embedding)
                combined = input_signal.energy * weight * input_signal.importance * (1.0 + sim * 0.3)
                self.graph.neuron_potentials[target] = tgt_pot + combined

                # Update LSTM
                self.graph.update_lstm(target, input_signal.embedding)

                # Surprise detection
                pred_err = self.graph.neuron_pred_error.get(target, 0.0)
                error = abs(combined - pred_err)
                self.graph.neuron_pred_error[target] = 0.9 * pred_err + 0.1 * error
                if error > 0.5:
                    self.graph.neuron_importance[target] = min(1.0, self.graph.neuron_importance.get(target, 0.5) + 0.05)

                # Create new signal
                new_sig = copy.deepcopy(input_signal)
                new_sig.energy *= energy_cost
                new_sig.source = nid
                new_sig.destination = target
                new_sig.confidence = self.graph.synapse_confidence.get(sid, 0.1)
                new_sig.hop_count += 1
                new_sig.history.append((nid, sid, target))

                if target not in visited:
                    visited.add(target)
                    queue.append((target, steps + 1, new_sig.energy))

                # Update synapse frequency
                self.graph.synapse_frequency[sid] = self.graph.synapse_frequency.get(sid, 0) + 0.001
                self.graph.synapse_last_used[sid] = time.time()

            # Residual
            if nid in residual:
                self.graph.neuron_potentials[nid] = self.graph.neuron_potentials.get(nid, 0.0) + residual[nid] * 0.1

            # Activate
            pot = self.graph.neuron_potentials.get(nid, 0.0)
            refr = self.graph.neuron_refractory.get(nid, 0)
            if refr > 0:
                self.graph.neuron_refractory[nid] = refr - 1
                self.graph.neuron_activations[nid] = self.graph.neuron_activations.get(nid, 0.0) * 0.5
                continue

            if pot > self.config.propagation_threshold:
                act = float(torch.clamp(
                    gelu_torch(torch.tensor(pot - self.config.propagation_threshold)),
                    0.0, 1.0))
                self.graph.neuron_activations[nid] = act
                self.graph.neuron_last_activation[nid] = time.time()
                self.graph.neuron_usage[nid] = self.graph.neuron_usage.get(nid, 0) + 1
                self.graph.neuron_age[nid] = self.graph.neuron_age.get(nid, 0) + 1
                self.graph.neuron_energy[nid] = max(0.0, self.graph.neuron_energy.get(nid, 1.0) - 0.01)
                self.graph.neuron_potentials[nid] = pot * 0.3
                self.graph.neuron_refractory[nid] = 2
                self.graph.neuron_utility[nid] = 0.7 * self.graph.neuron_utility.get(nid, 0.0) + 0.3 * act
                hist = self.graph.neuron_activation_history.get(nid)
                if hist is not None:
                    hist.append(act)
                activated.append(nid)
                activation_map[nid] = act
                residual[nid] = act
            else:
                self.graph.neuron_activations[nid] = self.graph.neuron_activations.get(nid, 0.0) * 0.9

        if activated:
            self._hebbian_update(activated, ctx)
            for nid in activated:
                self.graph.neuron_importance[nid] = min(1.0, self.graph.neuron_importance.get(nid, 0.5) + 0.01)

        return activated, activation_map

    # ---------- Meta-learning ----------
    def _meta_learn(self, performance: float):
        self.performance_history.append(performance)
        if len(self.performance_history) < 10:
            return
        recent_perf = np.mean(list(self.performance_history)[-10:])
        if recent_perf > 0.7:
            self.meta_lr = min(0.1, self.meta_lr * 1.05)
        elif recent_perf < 0.4:
            self.meta_lr = max(0.001, self.meta_lr * 0.95)
        if recent_perf < 0.3:
            self.config.propagation_threshold = max(0.1, self.config.propagation_threshold * 0.95)
        elif recent_perf > 0.8:
            self.config.propagation_threshold = min(0.5, self.config.propagation_threshold * 1.02)
        logger.debug(f"Meta-learn: perf={recent_perf:.3f}, meta_lr={self.meta_lr:.4f}, "
                     f"threshold={self.config.propagation_threshold:.3f}")

    # ---------- Step ----------
    def step(self, input_signal: Signal, target_neuron_id: Optional[int] = None,
             reward: float = 0.0) -> List[int]:
        with self.lock:
            self.step_counter += 1
            for nid in self.graph.neuron_embeddings:
                self.graph.neuron_energy[nid] = max(0.0, self.graph.neuron_energy.get(nid, 1.0) - 0.0005)
                if self.graph.neuron_energy.get(nid, 1.0) < 0.1:
                    self.graph.neuron_potentials[nid] = 0.0
                    self.graph.neuron_cell[nid] = torch.zeros(self.dim, device=self.device, dtype=torch.float32)
                    self.graph.neuron_hidden[nid] = torch.zeros(self.dim, device=self.device, dtype=torch.float32)

            activated, _ = self.propagate_signal(input_signal)

            if target_neuron_id is not None:
                reward = 1.0 if target_neuron_id in activated else 0.0

            if reward > 0:
                for nid in activated:
                    self.graph.neuron_importance[nid] = min(1.0, self.graph.neuron_importance.get(nid, 0.5) + reward * 0.02)
                    for _, sid, _ in self.graph.get_incoming(nid):
                        if not self.graph.synapse_is_inhibitory.get(sid, False):
                            self.graph.synapse_reward[sid] = self.graph.synapse_reward.get(sid, 0.0) + reward
                            w = self.graph.synapse_weights.get(sid, 0)
                            self.graph.synapse_weights[sid] = float(np.clip(w + reward * 0.005, -2.0, 2.0))
                            self.graph.synapse_confidence[sid] = min(1.0, self.graph.synapse_confidence.get(sid, 0.1) + reward * 0.005)
                            # meta-learning on synapse LR
                            lr = self.graph.synapse_learning_rate.get(sid, 0.01)
                            mom = self.graph.synapse_lr_momentum.get(sid, 0.0)
                            mom = 0.9 * mom + 0.1 * reward
                            self.graph.synapse_lr_momentum[sid] = mom
                            self.graph.synapse_learning_rate[sid] = np.clip(
                                lr * (1.0 + 0.001 * mom), 0.0001, 0.5)

                exp = Experience(
                    state=input_signal.embedding,
                    action=str(activated),
                    reward=reward,
                    next_state=input_signal.embedding,
                    priority=reward
                )
                self.experience_buffer.add(exp)

            if self.step_counter % 50 == 0:
                self._experience_replay()
            if reward != 0:
                self._meta_learn(reward)
            if self.step_counter % 10 == 0:
                self._create_new_synapses_from_coactivation()
            if self.step_counter % self.config.homeostasis_every == 0:
                self._homeostatic_scaling()
            if self.step_counter % self.config.prune_every == 0:
                self.graph._prune_synapses()
                self.graph._prune_neurons()

            self.short_memory.add((input_signal, activated))
            self.global_time = time.time()
            return activated

    def _experience_replay(self, batch_size: int = 10):
        experiences, indices, weights = self.experience_buffer.sample(batch_size)
        if not experiences:
            return
        new_priorities = []
        for exp, weight in zip(experiences, weights):
            replayed = Signal(
                embedding=exp.state,
                energy=0.5 * weight,
                importance=0.5,
                text=exp.action,
                signal_type="replay"
            )
            activated, _ = self._propagate_signal_impl(replayed, max_steps=10)
            if activated:
                new_priorities.append(exp.priority * 0.9)
            else:
                new_priorities.append(exp.priority * 1.1)
        self.experience_buffer.update_priorities(indices, np.array(new_priorities))

    # ---------- RAG Retrieval ----------
    def retrieve_facts(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_emb = self.text_to_embedding(query_text)

        # 1. Knowledge base
        scored_kb = []
        for fact in self.knowledge_base:
            sim = cosine_similarity_torch(query_emb, fact["emb"])
            scored_kb.append((sim, fact))
        scored_kb.sort(key=lambda x: x[0], reverse=True)

        # 2. Hierarchical memory
        mem_results = self.memory.retrieve(
            query_emb, top_k=top_k * 2,
            memory_levels=[MemoryLevel.SEMANTIC, MemoryLevel.EPISODIC]
        )

        # 3. Network propagation
        sig = self.text_to_signal(query_text)
        activated, act_map = self.propagate_signal(sig, max_steps=12)

        network_facts = []
        for nid in activated:
            if self.graph.neuron_clusters.get(nid) == "output" and self.graph.neuron_labels.get(nid):
                sources = []
                for src, sid, _ in self.graph.get_incoming(nid):
                    if self.graph.synapse_weights.get(sid, 0) > 0.1:
                        lbl = self.graph.neuron_labels.get(src)
                        if lbl:
                            sources.append(lbl)
                if sources:
                    network_facts.append({
                        "q": ", ".join(sources[:3]),
                        "a": self.graph.neuron_labels[nid],
                        "weight": act_map.get(nid, 0.5),
                        "source": "network"
                    })

        # 4. Aggregate & re-rank
        results = []
        for sim, fact in scored_kb[:top_k]:
            if sim > 0.25:
                results.append({
                    "source": "memory",
                    "q": fact["q"],
                    "a": fact["a"],
                    "score": sim,
                    "confidence": fact.get("confidence", 0.5)
                })
        for mem in mem_results[:top_k]:
            results.append({
                "source": f"hierarchical_{mem['level']}",
                "q": str(mem["content"])[:100],
                "a": str(mem["content"])[:200],
                "score": mem["score"] * mem["importance"],
                "confidence": mem["importance"]
            })
        for f in network_facts[:top_k]:
            results.append({
                "source": "network",
                "q": f["q"],
                "a": f["a"],
                "score": f["weight"],
                "confidence": f["weight"]
            })

        seen = set()
        unique = []
        for r in results:
            key = (r["q"], r["a"])
            if key not in seen:
                seen.add(key)
                r["score"] = r["score"] * (0.5 + 0.5 * r.get("confidence", 0.5))
                unique.append(r)
        unique.sort(key=lambda x: x["score"], reverse=True)
        return unique[:top_k]

    # ---------- Answer Generation ----------
    def generate_answer(self, input_text: str, temperature: float = 0.7,
                        use_rag: bool = True, use_reflection: bool = True) -> Dict[str, Any]:
        dialog_ctx = self.dialog_memory.get_context_string(n=3)
        facts = self.retrieve_facts(input_text, top_k=5) if use_rag else []

        system_msg = (
            "Ты - интеллектуальный ассистент с ассоциативной памятью и способностью к саморефлексии. "
            "Отвечай естественно, по-человечески, на языке вопроса. "
            "Если есть релевантные факты из памяти - используй их. "
            "Если фактов нет - отвечай на основе общих знаний. "
            "Ответ должен быть связным и развернутым (1-3 предложения). "
            "Если ответ должен содержать код — напиши полностью развернутый ответ. "
            "Если запрос пустой — ответь только символами пробела."
        )
        context_parts = []
        if dialog_ctx:
            context_parts.append("=== История диалога ===")
            context_parts.append(dialog_ctx)
        if facts:
            context_parts.append("=== Факты из памяти ===")
            for i, f in enumerate(facts, 1):
                context_parts.append(
                    f"{i}. Вопрос: {f['q']} -> Ответ: {f['a']} "
                    f"(источник: {f['source']}, релевантность: {f['score']:.2f})"
                )
        context_str = "\\n".join(context_parts)
        user_prompt = input_text
        if context_str:
            user_prompt = f"{context_str}\\n\\nТекущий вопрос: {input_text}"

        try:
            response = self.llm_client.chat.completions.create(
                model="local-model",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2048,
                temperature=temperature,
            )
            answer_text = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            answer_text = self._fallback_chain_answer(input_text, temperature=temperature)

        reflection_result = None
        if use_reflection:
            reflection_result = self.reflection.reflect_on_answer(input_text, answer_text, facts, {})

        clarifying = None
        if reflection_result and reflection_result.get("needs_clarification"):
            clarifying = reflection_result.get("suggestion")
        elif len(facts) < 2:
            history = self.dialog_memory.items[-5:] if self.dialog_memory.items else []
            clarifying = self._generate_clarifying_question(input_text, answer_text, history, facts)

        return {
            "text": answer_text,
            "facts": facts,
            "known": len(facts) > 0,
            "fallback": len(facts) == 0,
            "reflection": reflection_result,
            "clarifying_question": clarifying,
            "confidence": reflection_result["confidence"] if reflection_result else 0.5,
        }

    def _generate_clarifying_question(self, user_input: str, answer: str,
                                      history: List[Dict], facts: List[Dict]) -> Optional[str]:
        if len(facts) >= 3:
            return None
        history_text = ""
        for turn in history[-3:]:
            history_text += f"Пользователь: {turn.get('user', '')}\\nАссистент: {turn.get('assistant', '')}\\n"
        system_prompt = (
            "Ты – ассистент, который помогает пользователю уточнить или расширить его запрос. "
            "На основе истории диалога и только что данного ответа, сформулируй один уточняющий вопрос. "
            "Если уточнение не требуется – ответь 'НЕТ'."
        )
        user_prompt = (
            f"История:\\n{history_text}\\n\\n"
            f"Вопрос: {user_input}\\nОтвет: {answer}\\n"
            f"Факты: {[f['q'] + ' -> ' + f['a'] for f in facts]}\\n\\n"
            f"Уточняющий вопрос:"
        )
        try:
            response = self.llm_client.chat.completions.create(
                model="local-model",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=60,
                temperature=0.6,
            )
            reply = response.choices[0].message.content.strip()
            if reply.upper() == "НЕТ" or not reply:
                return None
            return reply
        except Exception as e:
            logger.error(f"ContextQ Error: {e}")
            return None

    def _fallback_chain_answer(self, input_text: str, temperature: float = 0.5) -> str:
        concepts = self.extract_concepts(input_text)
        chains = []
        for c in concepts:
            nid = self.concept_index.get(c)
            if nid:
                chain = self._generate_chain_from_node(nid, max_hops=4, temperature=temperature)
                if len(chain) > 1:
                    chains.append(" -> ".join(chain))
        if chains:
            return " ; ".join(chains)
        return f"Я пока не знаю про '{input_text}'. Скажи: learn {input_text} => <ответ>"

    def _generate_chain_from_node(self, start_id: int, max_hops: int = 6,
                                   min_weight: float = 0.02, temperature: float = 0.5) -> List[str]:
        if start_id not in self.graph.neuron_embeddings:
            return []
        chain = [self.graph.neuron_labels.get(start_id) or "?"]
        visited = {start_id}
        current = start_id
        for _ in range(max_hops):
            candidates = []
            for _, sid, weight in self.graph.get_outgoing(current):
                if weight <= 0:
                    continue
                target = None
                for u, v, d in self.graph.graph.edges(data=True):
                    if d.get("sid") == sid:
                        target = v
                        break
                if target is None or target in visited:
                    continue
                lbl = self.graph.neuron_labels.get(target)
                if not lbl:
                    continue
                sc = weight + self.graph.synapse_confidence.get(sid, 0) * 0.1 + self.graph.synapse_tag_strength.get(sid, 0) * 0.1
                if sc >= min_weight:
                    candidates.append((sid, sc, target, lbl))
            if not candidates:
                break
            if temperature > 0 and len(candidates) > 1:
                scores = np.array([s for _, s, _, _ in candidates])
                exp_scores = np.exp(scores / temperature - np.max(scores / temperature))
                probs = exp_scores / exp_scores.sum()
                idx = np.random.choice(len(candidates), p=probs)
                best_sid, _, target, lbl = candidates[idx]
            else:
                best_sid, _, target, lbl = max(candidates, key=lambda x: x[1])
            current = target
            visited.add(current)
            chain.append(lbl)
            self.graph.synapse_frequency[best_sid] = self.graph.synapse_frequency.get(best_sid, 0) + 0.002
        return chain

    # ---------- Text utilities ----------
    def normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\\w\\s]", " ", text)
        return " ".join(text.split())

    def text_to_embedding(self, text: str) -> torch.Tensor:
        return self.embedder.get_embedding(text)

    def text_to_signal(self, text: str, energy: float = 1.0) -> Signal:
        vec = self.text_to_embedding(text)
        return Signal(embedding=vec, energy=energy, importance=0.8, context=vec, text=text)

    def extract_concepts(self, text: str) -> List[str]:
        text_norm = self.normalize_text(text)
        found = []
        for concept in sorted(self.concept_index.keys(), key=len, reverse=True):
            if concept in text_norm:
                found.append(concept)
                text_norm = text_norm.replace(concept, " ")
        return found

    def get_or_create_concept_neuron(self, text: str) -> int:
        normalized = self.normalize_text(text)
        nid = self.concept_index.get(normalized)
        if nid is not None and nid in self.graph.neuron_embeddings:
            return nid
        label = text.strip()
        if len(label) > 300:
            label = label[:300]
        emb = self.text_to_embedding(text)
        nid = self.graph.add_neuron(emb, cluster="output",
                                     layer=len(self.config.hidden_layers) + 1, label=label)
        self.concept_index[normalized] = nid
        logger.info(f"New concept neuron: {nid} = '{label}'")
        return nid

    # ---------- Learning ----------
    def learn_pair(self, input_text: str, output_text: str,
                   reinforce_boost: float = 0.15, epochs: int = 1):
        with self.lock:
            for epoch in range(epochs):
                input_norm = self.normalize_text(input_text)
                output_norm = self.normalize_text(output_text)
                if input_norm == output_norm:
                    logger.info("Skipping: input == output")
                    return
                input_id = self.get_or_create_concept_neuron(input_text)
                output_id = self.get_or_create_concept_neuron(output_text)
                sim = cosine_similarity_torch(
                    self.graph.neuron_embeddings[input_id],
                    self.graph.neuron_embeddings[output_id]
                )
                sim_bonus = max(0.0, sim) * 0.05
                syn_id = self.graph.add_synapse(input_id, output_id, weight=0.2 + sim_bonus)
                if syn_id is None:
                    logger.error("Failed to create synapse (limit reached)")
                    return
                syn = syn_id  # just ID
                usage = self.graph.synapse_usage_count.get(syn, 0)
                if usage > 0:
                    w = self.graph.synapse_weights[syn]
                    self.graph.synapse_weights[syn] = float(np.clip(w + reinforce_boost, -2.0, 2.0))
                    self.graph.synapse_plasticity[syn] = min(1.0, self.graph.synapse_plasticity.get(syn, 0.5) + 0.02)
                    self.graph.synapse_confidence[syn] = min(1.0, self.graph.synapse_confidence.get(syn, 0.1) + 0.05)
                    self.graph.synapse_frequency[syn] = self.graph.synapse_frequency.get(syn, 0) + 0.02
                else:
                    self.graph.synapse_confidence[syn] = min(1.0, self.graph.synapse_confidence.get(syn, 0.1) + 0.05)

                if epoch == 0:
                    sig = self.text_to_signal(input_text)
                    activated, _ = self.propagate_signal(sig, max_steps=15)
                    if activated:
                        sorted_activated = sorted(
                            activated,
                            key=lambda nid: self.graph.neuron_importance.get(nid, 0),
                            reverse=True
                        )
                        for hid in sorted_activated[:5]:
                            if self.graph.neuron_clusters.get(hid) == "output":
                                continue
                            hsyn_id = self.graph.add_synapse(hid, output_id, weight=0.05)
                            if hsyn_id is not None:
                                self.graph.synapse_weights[hsyn_id] = float(np.clip(
                                    self.graph.synapse_weights.get(hsyn_id, 0) + 0.03, -2.0, 2.0))
                    self.step(sig, target_neuron_id=output_id)

                if epoch == 0:
                    self._add_to_knowledge_base(input_text, output_text)
                    combined_emb = self.text_to_embedding(input_text + " " + output_text)
                    self.memory.add(
                        content={"q": input_text, "a": output_text},
                        embedding=combined_emb,
                        memory_level=MemoryLevel.SEMANTIC,
                        importance=0.7,
                        tags=["learned", "fact"]
                    )
                logger.info(f"Learn epoch {epoch+1}/{epochs}: '{input_text}' -> '{output_text}' "
                            f"(weight: {self.graph.synapse_weights.get(syn_id, 0):.3f})")
            self._learn_counter += epochs
            if self._learn_counter % self.config.auto_save_every == 0:
                self.save()
                logger.info("Auto-saved")

    def _add_to_knowledge_base(self, q: str, a: str):
        for item in self.knowledge_base:
            if item["q"] == q and item["a"] == a:
                return
        emb = self.text_to_embedding(q + " " + a)
        self.knowledge_base.append({
            "q": q, "a": a, "emb": emb, "time": time.time(),
            "confidence": 0.5, "access_count": 0
        })
        if len(self.knowledge_base) > self.config.max_kb_size:
            self.knowledge_base.pop(0)

    def learn_negative_pair(self, input_text: str, output_text: str, penalty: float = 0.15):
        with self.lock:
            input_norm = self.normalize_text(input_text)
            output_norm = self.normalize_text(output_text)
            if input_norm == output_norm:
                return
            input_id = self.get_or_create_concept_neuron(input_text)
            output_id = self.get_or_create_concept_neuron(output_text)
            sid = self.graph.edge_to_sid.get((input_id, output_id))
            if sid is not None and sid in self.graph.synapse_weights:
                w = self.graph.synapse_weights[sid]
                new_w = max(-2.0, w - penalty)
                self.graph.synapse_weights[sid] = new_w
                if new_w < 0:
                    self.graph.synapse_is_inhibitory[sid] = True
                logger.info(f"Negative learn: '{input_text}' -> '{output_text}' = {new_w:.3f}")
            else:
                sid = self.graph.add_synapse(input_id, output_id,
                                              weight=-random.uniform(0.1, 0.3), is_inhibitory=True)
                if sid is not None:
                    logger.info(f"Inhibitory synapse: '{input_text}' -> '{output_text}'")
            self._learn_counter += 1
            if self._learn_counter % self.config.auto_save_every == 0:
                self.save()

    def forget_concept(self, text: str) -> bool:
        with self.lock:
            normalized = self.normalize_text(text)
            nid = self.concept_index.get(normalized)
            if nid is None or nid not in self.graph.neuron_embeddings:
                return False
            self.graph.remove_neuron(nid)
            del self.concept_index[normalized]
            new_kb = []
            for item in self.knowledge_base:
                if normalized not in self.normalize_text(item["q"]) and normalized not in self.normalize_text(item["a"]):
                    new_kb.append(item)
            self.knowledge_base = new_kb
            logger.info(f"Forgot concept: '{text}'")
            return True

    def show_links(self, text: str, top_k: int = 10):
        normalized = self.normalize_text(text)
        if not normalized:
            print("Пустой запрос.")
            return
        nid = self.concept_index.get(normalized)
        if nid is None:
            print(f"Понятие '{text}' не изучено.")
            return
        links = []
        for _, sid, weight in self.graph.get_outgoing(nid):
            target = None
            for u, v, d in self.graph.graph.edges(data=True):
                if d.get("sid") == sid:
                    target = v
                    break
            if target is None:
                continue
            lbl = self.graph.neuron_labels.get(target)
            if not lbl:
                continue
            links.append((
                lbl, weight,
                self.graph.synapse_usage_count.get(sid, 0),
                self.graph.synapse_confidence.get(sid, 0),
                self.graph.synapse_is_inhibitory.get(sid, False),
                self.graph.synapse_learning_rate.get(sid, 0.01),
                self.graph.synapse_tag_strength.get(sid, 0)
            ))
        links.sort(key=lambda x: x[1], reverse=True)
        if not links:
            print(f"У '{text}' нет исходящих связей.")
            return
        print(f"Связи '{text}':")
        for label, weight, count, conf, inhib, lr, tag in links[:top_k]:
            inhib_str = " (тормозная)" if inhib else ""
            print(f"  -> {label}   (w={weight:.3f}, use={count}, conf={conf:.2f}, "
                  f"lr={lr:.4f}, tag={tag:.2f}){inhib_str}")

    # ---------- Sleep ----------
    def _replay_recent_memory(self, n: int = 30, energy_scale: float = 0.5):
        recent = self.short_memory.get_recent(n)
        for signal, _activated_ids in recent:
            if signal is None:
                continue
            replayed = copy.deepcopy(signal)
            replayed.energy *= energy_scale
            replayed.signal_type = "replay"
            self._propagate_signal_impl(replayed, max_steps=10)

    def sleep(self, duration_steps: int = 10):
        with self.lock:
            logger.info("=== Сон ===")
            self._replay_recent_memory(n=30)
            self._experience_replay(batch_size=20)
            self.memory.decay_all(rate=0.001)
            for _ in range(duration_steps):
                to_remove = []
                for nid in self.graph.neuron_embeddings:
                    if (self.graph.neuron_energy.get(nid, 1.0) < 0.05 and
                        self.graph.neuron_importance.get(nid, 0.5) < 0.1 and
                        self.graph.neuron_clusters.get(nid) not in ("output", "input")):
                        to_remove.append(nid)
                for nid in to_remove:
                    self.graph.remove_neuron(nid)
                    logger.debug(f"Removed noise neuron {nid}")
                for sid in list(self.graph.synapse_weights.keys()):
                    if self.graph.synapse_frequency.get(sid, 0) > 0.5:
                        w = self.graph.synapse_weights.get(sid, 0)
                        self.graph.synapse_weights[sid] = float(np.clip(w + 0.01, -2.0, 2.0))
                        self.graph.synapse_confidence[sid] = min(1.0, self.graph.synapse_confidence.get(sid, 0.1) + 0.01)
                    if self.graph.synapse_reward.get(sid, 0) > 0.5:
                        w = self.graph.synapse_weights.get(sid, 0)
                        self.graph.synapse_weights[sid] = float(np.clip(w + 0.02, -2.0, 2.0))
                        self.graph.synapse_confidence[sid] = min(1.0, self.graph.synapse_confidence.get(sid, 0.1) + 0.02)
                    self.graph.synapse_reward[sid] = self.graph.synapse_reward.get(sid, 0) * 0.9
            self._homeostatic_scaling()
            for item in self.short_memory.get_recent(50):
                self.long_memory.add(item)
            self.short_memory.clear()
            logger.info("Sleep completed")
            self.save()

    # ---------- Save / Load (model dir) ----------
    def save(self, model_dir: str = None):
        with self.lock:
            model_dir = model_dir or self.config.model_dir
            os.makedirs(model_dir, exist_ok=True)

            # 1. NeuralGraph (tensors + graph)
            self.graph.save(os.path.join(model_dir, "graph"))

            # 2. Metadata (small JSON)
            meta = {
                "version": "6.0",
                "dim": self.dim,
                "step_counter": self.step_counter,
                "meta_lr": self.meta_lr,
                "learn_counter": self._learn_counter,
                "concept_index": self.concept_index,
                "knowledge_base": [
                    {"q": kb["q"], "a": kb["a"], "time": kb["time"],
                     "confidence": kb.get("confidence", 0.5)}
                    for kb in self.knowledge_base
                ],
                "config": {
                    "dim_embedding": self.config.dim_embedding,
                    "max_neurons": self.config.max_neurons,
                    "max_synapses": self.config.max_synapses,
                }
            }
            with open(os.path.join(model_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            # 3. Dialog history
            dialog_data = []
            for item in self.dialog_memory.items:
                dialog_data.append({
                    "user": item.get("user", ""),
                    "assistant": item.get("assistant", ""),
                    "time": item.get("time", 0)
                })
            with open(os.path.join(model_dir, "dialog_history.json"), "w", encoding="utf-8") as f:
                json.dump(dialog_data, f, ensure_ascii=False, indent=2)

            # 4. KB embeddings (separate torch file)
            kb_embs = {i: kb["emb"] for i, kb in enumerate(self.knowledge_base)}
            torch.save(kb_embs, os.path.join(model_dir, "kb_embeddings.pt"))

            logger.info(f"Brain v6 saved to {model_dir}")

    def load(self, model_dir: str = None):
        with self.lock:
            model_dir = model_dir or self.config.model_dir
            if not os.path.exists(model_dir):
                logger.info(f"Model dir {model_dir} not found, starting fresh")
                return

            # 1. Load graph
            graph_dir = os.path.join(model_dir, "graph")
            if os.path.exists(graph_dir):
                self.graph.load(graph_dir)

            # 2. Load metadata
            meta_path = os.path.join(model_dir, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.step_counter = meta.get("step_counter", 0)
                self.meta_lr = meta.get("meta_lr", self.config.meta_learning_rate)
                self._learn_counter = meta.get("learn_counter", 0)
                self.concept_index = meta.get("concept_index", {})
                self.knowledge_base = [
                    {"q": kb["q"], "a": kb["a"], "time": kb["time"],
                     "confidence": kb.get("confidence", 0.5)}
                    for kb in meta.get("knowledge_base", [])
                ]

            # 3. Load KB embeddings
            kb_path = os.path.join(model_dir, "kb_embeddings.pt")
            if os.path.exists(kb_path):
                kb_embs = torch.load(kb_path, map_location=self.device, weights_only=False)
                for i, kb in enumerate(self.knowledge_base):
                    if i in kb_embs:
                        kb["emb"] = kb_embs[i]

            # 4. Load dialog
            dialog_path = os.path.join(model_dir, "dialog_history.json")
            if os.path.exists(dialog_path):
                with open(dialog_path, "r", encoding="utf-8") as f:
                    dialog_data = json.load(f)
                for item in dialog_data:
                    self.dialog_memory.items.append(item)

            logger.info(f"Brain v6 loaded from {model_dir}")

    # ---------- Stats & helpers ----------
    def get_stats(self) -> Dict[str, Any]:
        return {
            "neurons": len(self.graph.neuron_embeddings),
            "synapses": len(self.graph.synapse_weights),
            "concepts": len(self.concept_index),
            "knowledge_base": len(self.knowledge_base),
            "memory": self.memory.get_stats(),
            "experience_buffer": self.experience_buffer.get_stats(),
            "reflection": self.reflection.get_reflection_stats(),
            "step_counter": self.step_counter,
            "meta_lr": self.meta_lr,
            "embedding_cache": self.embedder.get_cache_stats(),
        }

    def get_weak_concepts(self, threshold: float = 0.2, limit: int = 5) -> List[str]:
        weak = []
        for nid in self.graph.neuron_embeddings:
            if self.graph.neuron_clusters.get(nid) == "output" and self.graph.neuron_labels.get(nid):
                total = 0.0
                count = 0
                for _, sid, _ in self.graph.get_outgoing(nid):
                    total += abs(self.graph.synapse_weights.get(sid, 0))
                    count += 1
                avg = total / count if count > 0 else 0.0
                if avg < threshold and count < 3:
                    weak.append(self.graph.neuron_labels[nid])
        random.shuffle(weak)
        return weak[:limit]

    def get_uncertain_concepts(self, threshold: float = 0.3, limit: int = 5) -> List[Dict[str, Any]]:
        uncertain = []
        for nid in self.graph.neuron_embeddings:
            if self.graph.neuron_clusters.get(nid) != "output" or not self.graph.neuron_labels.get(nid):
                continue
            in_weights = []
            for _, sid, _ in self.graph.get_incoming(nid):
                in_weights.append(abs(self.graph.synapse_weights.get(sid, 0)))
            avg_in = sum(in_weights) / len(in_weights) if in_weights else 0.0
            out_count = len(list(self.graph.get_outgoing(nid)))
            if avg_in < threshold or out_count < 2:
                age = self.graph.neuron_age.get(nid, 0)
                usage = self.graph.neuron_usage.get(nid, 0)
                activation_freq = usage / (age + 1) if age > 0 else 0
                score = ((1.0 - avg_in) * 0.5 + (1.0 - min(1.0, out_count / 5)) * 0.3 +
                         (1.0 - activation_freq) * 0.2)
                uncertain.append({
                    "label": self.graph.neuron_labels[nid],
                    "score": score,
                    "avg_in": avg_in,
                    "out_count": out_count,
                    "activation_freq": activation_freq
                })
        uncertain.sort(key=lambda x: x["score"], reverse=True)
        return uncertain[:limit]

    def get_recent_user_topics(self, limit: int = 3) -> List[str]:
        if not self.dialog_memory.items:
            return []
        user_messages = []
        for turn in self.dialog_memory.items[-10:]:
            if "user" in turn and turn["user"]:
                user_messages.append(turn["user"])
        if not user_messages:
            return []
        combined = " ".join(user_messages)
        prompt = f"Из текста: '{combined}' выдели {limit} главных тем (существительные), через запятую."
        try:
            resp = self.llm_client.chat.completions.create(
                model="local-model",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=30,
                temperature=0.3,
            )
            raw = resp.choices[0].message.content.strip()
            topics = [t.strip() for t in raw.split(",") if t.strip()]
            return topics[:limit]
        except Exception as e:
            logger.error(f"Topics error: {e}")
            last_msg = user_messages[-1] if user_messages else ""
            words = [w for w in last_msg.split() if len(w) > 3]
            return words[:limit]

    def generate_contextual_question(self, user_topics: List[str],
                                     uncertain_concepts: List[Dict]) -> Optional[str]:
        if not user_topics and not uncertain_concepts:
            return None
        topics_str = ", ".join(user_topics) if user_topics else "общая тема"
        weak_str = ", ".join([c["label"] for c in uncertain_concepts[:3]]) if uncertain_concepts else "нет явных слабых мест"
        system_prompt = (
            "Ты — ассистент, который помогает пользователю углубиться в тему. "
            "Сформулируй один естественный вопрос. Ответь только вопросом."
        )
        user_prompt = (
            f"Темы: {topics_str}.\\n"
            f"Слабые места: {weak_str}.\\n"
            f"Сформулируй один вопрос."
        )
        try:
            resp = self.llm_client.chat.completions.create(
                model="local-model",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=50,
                temperature=0.7,
            )
            question = resp.choices[0].message.content.strip()
            if question and "?" in question:
                return question
            return None
        except Exception as e:
            logger.error(f"ContextualQ error: {e}")
            return None

    def save_dialog_history(self, filename: str = None):
        filename = filename or os.path.join(self.config.model_dir, "dialog_history.json")
        data = []
        for item in self.dialog_memory.items:
            data.append({
                "user": item.get("user", ""),
                "assistant": item.get("assistant", ""),
                "time": item.get("time", 0)
            })
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_dialog_history(self, filename: str = None):
        filename = filename or os.path.join(self.config.model_dir, "dialog_history.json")
        if os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    self.dialog_memory.items.append(item)
            except Exception as e:
                logger.error(f"Dialog load error: {e}")


# ============================
# Teacher v6
# ============================
class Teacher:
    def __init__(self, llm_client: Optional[OpenAI] = None):
        self.llm_client = llm_client
        self.evaluation_history = []

    def evaluate(self, question: str, brain_answer: str) -> Tuple[float, str, Dict]:
        system_prompt = (
            "Ты — критический эксперт. Оцени ответ по критериям (0-1 каждый):\\n"
            "1. Релевантность\\n2. Точность\\n3. Полнота\\n4. Ясность\\n"
            "Формат: <средняя_оценка>|<улучшенный_ответ>|\\n"
            "details: relevance=X, accuracy=Y, completeness=Z, clarity=W"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Вопрос: {question}\\nОтвет: {brain_answer}\\nОценка:"}
        ]
        try:
            response = self.llm_client.chat.completions.create(
                model="local-model", messages=messages, max_tokens=150, temperature=0.2,
            )
            raw = response.choices[0].message.content.strip()
            score = 0.5
            improved = brain_answer
            details = {"relevance": 0.5, "accuracy": 0.5, "completeness": 0.5, "clarity": 0.5}
            match = re.search(r'(\\d+\\.?\\d*)', raw)
            if match:
                score = float(match.group(1))
                rest = raw.replace(match.group(1), '').strip()
                if rest.startswith('|'):
                    rest = rest[1:].strip()
                parts = rest.split('|')
                if len(parts) >= 1 and parts[0] and len(parts[0]) > 2:
                    improved = parts[0]
                if len(parts) >= 2:
                    details_str = parts[1]
                    for aspect in ["relevance", "accuracy", "completeness", "clarity"]:
                        m = re.search(rf'{aspect}=(\\d+\\.?\\d*)', details_str, re.IGNORECASE)
                        if m:
                            details[aspect] = float(m.group(1))
            score = max(0.0, min(1.0, score))
            self.evaluation_history.append({
                "question": question,
                "score": score,
                "details": details,
                "timestamp": time.time(),
            })
            return score, improved, details
        except Exception as e:
            logger.error(f"Teacher error: {e}")
            return 0.5, brain_answer, {"relevance": 0.5, "accuracy": 0.5, "completeness": 0.5, "clarity": 0.5}

    def get_evaluation_stats(self) -> Dict:
        if not self.evaluation_history:
            return {}
        scores = [e["score"] for e in self.evaluation_history]
        return {
            "total_evaluations": len(self.evaluation_history),
            "mean_score": float(np.mean(scores)),
            "recent_mean": float(np.mean(scores[-10:])) if len(scores) >= 10 else float(np.mean(scores)),
        }


# ============================
# Интерактивный режим v6
# ============================
LEARN_PATTERN = re.compile(r'^learn\\s+(.+?)\\s*=>\\s*(.+)$', re.IGNORECASE)
NEG_PATTERN = re.compile(r'^neg\\s+(.+?)\\s*=>\\s*(.+)$', re.IGNORECASE)
LINKS_PATTERN = re.compile(r'^links\\s+(.+)$', re.IGNORECASE)
FORGET_PATTERN = re.compile(r'^forget\\s+(.+)$', re.IGNORECASE)
STATS_PATTERN = re.compile(r'^stats?$', re.IGNORECASE)
SLEEP_PATTERN = re.compile(r'^sleep$', re.IGNORECASE)
SAVE_PATTERN = re.compile(r'^save$', re.IGNORECASE)
EXIT_PATTERN = re.compile(r'^(exit|quit|выход)$', re.IGNORECASE)
REFLECT_PATTERN = re.compile(r'^reflect$', re.IGNORECASE)
META_PATTERN = re.compile(r'^meta$', re.IGNORECASE)


def interactive_with_teacher(brain: Brain):
    teacher = Teacher(llm_client=brain.llm_client)
    print("\\n" + "=" * 60)
    print("  Smart Brain v6 — PyTorch + NetworkX + SQLite")
    print("=" * 60)
    print("Команды:")
    print("  learn <вопрос> => <ответ>  — обучить")
    print("  neg <вопрос> => <ответ>    — отрицательное обучение")
    print("  links <понятие>            — показать связи")
    print("  forget <понятие>           — забыть понятие")
    print("  stats                      — статистика")
    print("  meta                       — мета-обучение статус")
    print("  reflect                    — статус саморефлексии")
    print("  save                       — сохранить")
    print("  sleep                      — консолидация памяти")
    print("  exit / quit                — выход")
    print("  (любой текст — вопрос к Brain)")
    print("=" * 60 + "\\n")

    try:
        while True:
            user_input = input("\\n> ").strip()
            if not user_input:
                continue
            lower = user_input.lower()

            if EXIT_PATTERN.match(lower):
                brain.save()
                print("💾 Сохранено. До свидания!")
                break

            if STATS_PATTERN.match(lower):
                stats = brain.get_stats()
                n_in = sum(1 for nid in brain.graph.neuron_embeddings
                           if brain.graph.neuron_clusters.get(nid) == "input")
                n_out = sum(1 for nid in brain.graph.neuron_embeddings
                            if brain.graph.neuron_clusters.get(nid) == "output")
                n_hid = sum(1 for nid in brain.graph.neuron_embeddings
                            if brain.graph.neuron_clusters.get(nid) == "hidden")
                print(f"📊 Нейронов: {stats['neurons']} (in:{n_in}, out:{n_out}, hid:{n_hid})")
                print(f"🔗 Синапсов: {stats['synapses']}")
                print(f"💡 Понятий: {stats['concepts']}")
                print(f"📚 Фактов: {stats['knowledge_base']}")
                print(f"🧠 Память: {stats['memory']}")
                print(f"🎯 Experience buffer: {stats['experience_buffer']}")
                print(f"🔄 Step: {stats['step_counter']}, meta_lr: {stats['meta_lr']:.4f}")
                print(f"🎭 Reflection: {stats['reflection']}")
                print(f"⚡ Embedding cache: {stats['embedding_cache']}")
                continue

            if META_PATTERN.match(lower):
                print(f"Meta-learning rate: {brain.meta_lr:.4f}")
                print(f"Performance history (last 10): {list(brain.performance_history)[-10:]}")
                print(f"Propagation threshold: {brain.config.propagation_threshold:.3f}")
                continue

            if REFLECT_PATTERN.match(lower):
                refl_stats = brain.reflection.get_reflection_stats()
                print(f"Reflection stats: {refl_stats}")
                teacher_stats = teacher.get_evaluation_stats()
                print(f"Teacher stats: {teacher_stats}")
                continue

            if SAVE_PATTERN.match(lower):
                brain.save()
                brain.save_dialog_history()
                print("💾 Сохранено.")
                continue

            if SLEEP_PATTERN.match(lower):
                brain.sleep(duration_steps=5)
                print("😴 Сон завершён.")
                continue

            m_learn = LEARN_PATTERN.match(user_input)
            if m_learn:
                brain.learn_pair(m_learn.group(1).strip(), m_learn.group(2).strip())
                continue

            m_neg = NEG_PATTERN.match(user_input)
            if m_neg:
                brain.learn_negative_pair(m_neg.group(1).strip(), m_neg.group(2).strip())
                continue

            m_links = LINKS_PATTERN.match(user_input)
            if m_links:
                brain.show_links(m_links.group(1).strip())
                continue

            m_forget = FORGET_PATTERN.match(user_input)
            if m_forget:
                concept = m_forget.group(1).strip()
                if brain.forget_concept(concept):
                    print(f"🗑️  Понятие '{concept}' забыто.")
                else:
                    print(f"❌ Понятие '{concept}' не найдено.")
                continue

            # Основной диалог
            temp = random.uniform(0.5, 1.0)
            result = brain.generate_answer(user_input, temperature=temp, use_rag=True, use_reflection=True)
            brain_answer = result['text']
            print(f"\\n🤖 [Brain] {brain_answer}")
            if result.get('facts'):
                print(f"📖 [Фактов: {len(result['facts'])}]")
            if result.get('reflection'):
                refl = result['reflection']
                print(f"🎭 [Reflection] confidence={refl['confidence']:.2f}, "
                      f"needs_clarification={refl['needs_clarification']}")
            if result.get('clarifying_question'):
                print(f"❓ [Уточнение] {result['clarifying_question']}")

            score, improved, details = teacher.evaluate(user_input, brain_answer)
            print(f"📊 [Teacher] Оценка: {score:.2f} ", end="")
            if details:
                print(f"(релевантность={details.get('relevance',0):.2f}, "
                      f"точность={details.get('accuracy',0):.2f}, "
                      f"полнота={details.get('completeness',0):.2f}, "
                      f"ясность={details.get('clarity',0):.2f})")
            else:
                print()

            if improved != brain_answer:
                print(f"✨ [Улучшение] {improved}")

            if score >= 0.75:
                target = improved if improved != brain_answer else brain_answer
                if target.lower() != user_input.lower():
                    brain.learn_pair(user_input, target)
                    brain.dialog_memory.add_turn(user_input, target, brain.text_to_embedding(user_input))
                    print("✅ Обучение: положительное")
            elif score <= 0.3:
                if improved and improved != brain_answer and improved.lower() != user_input.lower():
                    brain.learn_negative_pair(user_input, brain_answer)
                    brain.learn_pair(user_input, improved)
                    brain.dialog_memory.add_turn(user_input, improved, brain.text_to_embedding(user_input))
                    print("⚠️ Обучение: отрицательное + коррекция")
                else:
                    brain.learn_negative_pair(user_input, brain_answer)
                    print("⚠️ Обучение: отрицательное")
            else:
                brain.dialog_memory.add_turn(user_input, brain_answer, brain.text_to_embedding(user_input))
                print("➖ Нейтрально: диалог сохранён")

    except (KeyboardInterrupt, EOFError):
        print("\\n\\n⏹️  Прерывание — сохраняю...")
    finally:
        brain.save()
        brain.save_dialog_history()
        print("✅ Готово.")


# ============================
# Запуск
# ============================
if __name__ == "__main__":
    MODEL_DIR = "brain_model_v6"
    config = BrainConfig(
        dim_embedding=128,
        input_neurons=40,
        output_neurons=40,
        hidden_layers=[100, 80, 60],
        max_neurons=2000,
        max_synapses=20000,
        model_dir=MODEL_DIR,
    )
    brain = Brain(config=config)
    brain.load(MODEL_DIR)
    interactive_with_teacher(brain)

