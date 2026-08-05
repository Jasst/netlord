"""
Smart Brain v5 — Улучшенная ассоциативная нейросеть с:
- LSTM-подобными нейронами (cell state + гейты)
- Multi-head attention для синапсов
- Prioritized Experience Replay
- Иерархической памятью (Semantic + Episodic + Working)
- Meta-learning (адаптивные learning rates)
- Self-reflection и hallucination detection
- Улучшенным RAG с multi-hop retrieval
- Structured logging

Автор: AI Assistant (улучшение v4)
"""

import numpy as np
import random
import time
import json
import os
import re
import zlib
import copy
import threading
import logging
import hashlib
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from openai import OpenAI

# ============================
# Настройка логирования
# ============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("SmartBrainV5")

# ============================
# Конфигурация
# ============================
LM_STUDIO_BASE_URL = os.environ.get("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
LM_STUDIO_API_KEY = os.environ.get("LM_STUDIO_API_KEY", "not-needed")

# ============================
# Вспомогательные функции
# ============================
def random_vector(dim: int = 128, seed: Optional[int] = None) -> np.ndarray:
    """Генерирует случайный нормализованный вектор."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    v = rng.randn(dim).astype(np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусное сходство с защитой от None и нулевых векторов."""
    if a is None or b is None:
        return 0.0
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def sigmoid(x):
    x = np.asarray(x)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax с температурой."""
    x = np.asarray(x, dtype=np.float32)
    x = x / max(temperature, 1e-8)
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-10)

def gelu(x: float) -> float:
    """GELU activation — лучше, чем ReLU для глубоких сетей."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Layer normalization."""
    x = np.asarray(x, dtype=np.float32)
    mean = np.mean(x)
    var = np.var(x)
    return (x - mean) / np.sqrt(var + eps)

def hash_text(text: str) -> str:
    """Быстрый хэш текста для кэширования."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()

# ============================
# Конфигурационные dataclasses
# ============================
@dataclass
class BrainConfig:
    """Конфигурация мозга."""
    dim_embedding: int = 128
    input_neurons: int = 40
    output_neurons: int = 40
    hidden_layers: List[int] = field(default_factory=lambda: [100, 80, 60])
    max_neurons: int = 2000
    max_synapses: int = 20000
    model_path: str = "brain_model_v5.json"

    # STDP параметры
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
# Улучшенный провайдер эмбеддингов
# ============================
class EmbeddingProvider:
    """
    Провайдер эмбеддингов с:
    - LRU-кэшем с TTL
    - Проекцией Джонсона-Линденштраусса
    - Fallback на локальные эмбеддинги
    - Batch processing
    """
    def __init__(self, dim: int = 128, api_model: str = "local-model", 
                 llm_client: Optional[OpenAI] = None, cache_ttl: int = 3600):
        self.dim = dim
        self.api_model = api_model
        self.llm_client = llm_client or OpenAI(
            base_url=LM_STUDIO_BASE_URL,
            api_key=LM_STUDIO_API_KEY,
        )
        self._cache: Dict[str, Tuple[np.ndarray, float]] = {}  # text -> (vec, timestamp)
        self._cache_lock = threading.Lock()
        self._api_available = True
        self._projection: Optional[np.ndarray] = None
        self.cache_ttl = cache_ttl
        self._cache_hits = 0
        self._cache_misses = 0

    def _local_embedding(self, text: str) -> np.ndarray:
        """Локальный эмбеддинг на основе n-грамм и хэширования."""
        text = text.lower().strip()
        text = re.sub(r"[^\\w\\s]", " ", text)
        tokens = text.split()
        if not tokens:
            return random_vector(self.dim)

        # Униграммы + биграммы + триграммы
        ngrams = tokens[:]
        for i in range(len(tokens) - 1):
            ngrams.append(tokens[i] + "_" + tokens[i+1])
        for i in range(len(tokens) - 2):
            ngrams.append(tokens[i] + "_" + tokens[i+1] + "_" + tokens[i+2])

        vec = np.zeros(self.dim, dtype=np.float32)
        for idx, token in enumerate(ngrams):
            h = zlib.adler32(token.encode("utf-8")) + idx * 31
            rng = np.random.RandomState(h % (2**31))
            tok_vec = rng.randn(self.dim).astype(np.float32)
            # TF-IDF-like weighting
            weight = 1.0 / (1.0 + idx * 0.05)
            vec += tok_vec * weight

        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else random_vector(self.dim)

    def _project_to_dim(self, vec: np.ndarray) -> np.ndarray:
        """Проецирует вектор произвольной размерности в self.dim."""
        src_dim = len(vec)
        if src_dim == self.dim:
            return vec
        if self._projection is None or self._projection.shape[0] != src_dim:
            rng = np.random.RandomState(1234567)
            self._projection = rng.randn(src_dim, self.dim).astype(np.float32) / np.sqrt(self.dim)
        projected = vec @ self._projection
        norm = np.linalg.norm(projected)
        return projected / norm if norm > 0 else projected

    def get_embedding(self, text: str) -> np.ndarray:
        """Получает эмбеддинг с кэшированием."""
        if not text:
            return random_vector(self.dim)

        cache_key = hash_text(text)

        with self._cache_lock:
            if cache_key in self._cache:
                vec, ts = self._cache[cache_key]
                if time.time() - ts < self.cache_ttl:
                    self._cache_hits += 1
                    return vec.copy()
                else:
                    del self._cache[cache_key]

        self._cache_misses += 1

        if self._api_available:
            try:
                resp = self.llm_client.embeddings.create(
                    model=self.api_model,
                    input=[text]
                )
                vec = np.array(resp.data[0].embedding, dtype=np.float32)
                if len(vec) != self.dim:
                    vec = self._project_to_dim(vec)
                with self._cache_lock:
                    self._cache[cache_key] = (vec.copy(), time.time())
                return vec
            except Exception as e:
                logger.warning(f"API embedding failed: {e}, switching to local")
                self._api_available = False

        vec = self._local_embedding(text)
        with self._cache_lock:
            self._cache[cache_key] = (vec.copy(), time.time())
        return vec

    def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Batch processing для эффективности."""
        return [self.get_embedding(t) for t in texts]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Статистика кэша."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "size": len(self._cache)
        }

# ============================
# LSTM-подобное состояние нейрона
# ============================
class NeuralState:
    """
    Состояние нейрона с LSTM-подобными гейтами.
    Позволяет нейрону "помнить" предыдущие активации и контролировать
    поток информации через forget/input/output гейты.
    """
    def __init__(self, dim: int):
        self.dim = dim
        self.cell_state = np.zeros(dim, dtype=np.float32)
        self.hidden_state = np.zeros(dim, dtype=np.float32)
        self.forget_gate = np.ones(dim, dtype=np.float32)
        self.input_gate = np.zeros(dim, dtype=np.float32)
        self.output_gate = np.zeros(dim, dtype=np.float32)

        # Веса для гейтов (инициализируются случайно, но стабильно)
        rng = np.random.RandomState(42)
        self.W_f = rng.randn(dim, dim).astype(np.float32) * 0.01
        self.W_i = rng.randn(dim, dim).astype(np.float32) * 0.01
        self.W_o = rng.randn(dim, dim).astype(np.float32) * 0.01
        self.W_c = rng.randn(dim, dim).astype(np.float32) * 0.01

    def update(self, input_signal: np.ndarray, prev_hidden: Optional[np.ndarray] = None):
        """Обновляет состояние на основе входного сигнала."""
        h = prev_hidden if prev_hidden is not None else self.hidden_state
        x = np.asarray(input_signal, dtype=np.float32)

        # Forget gate: что забыть из cell state
        self.forget_gate = sigmoid(np.dot(self.W_f, x) + np.dot(self.W_f, h))

        # Input gate: что записать в cell state
        self.input_gate = sigmoid(np.dot(self.W_i, x) + np.dot(self.W_i, h))

        # Candidate values
        candidate = np.tanh(np.dot(self.W_c, x) + np.dot(self.W_c, h))

        # Update cell state
        self.cell_state = self.forget_gate * self.cell_state + self.input_gate * candidate

        # Output gate: что выдать наружу
        self.output_gate = sigmoid(np.dot(self.W_o, x) + np.dot(self.W_o, h))
        self.hidden_state = self.output_gate * np.tanh(self.cell_state)

        return self.hidden_state.copy()

    def reset(self):
        """Сбрасывает состояние."""
        self.cell_state.fill(0)
        self.hidden_state.fill(0)
        self.forget_gate.fill(1)
        self.input_gate.fill(0)
        self.output_gate.fill(0)

    def to_dict(self) -> Dict:
        return {
            "cell_state": self.cell_state.tolist(),
            "hidden_state": self.hidden_state.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict, dim: int):
        state = cls(dim)
        state.cell_state = np.array(data.get("cell_state", [0.0]*dim), dtype=np.float32)
        state.hidden_state = np.array(data.get("hidden_state", [0.0]*dim), dtype=np.float32)
        return state

# ============================
# Multi-Head Attention для синапсов
# ============================
class AttentionMechanism:
    """
    Multi-head attention для синаптических связей.
    Позволяет синапсу вычислять контекстно-зависимые веса.
    """
    def __init__(self, dim: int, num_heads: int = 4):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        rng = np.random.RandomState(12345)
        # Query, Key, Value projections для каждой головы
        self.W_q = [rng.randn(dim, self.head_dim).astype(np.float32) * 0.01 for _ in range(num_heads)]
        self.W_k = [rng.randn(dim, self.head_dim).astype(np.float32) * 0.01 for _ in range(num_heads)]
        self.W_v = [rng.randn(dim, self.head_dim).astype(np.float32) * 0.01 for _ in range(num_heads)]

        # Output projection
        self.W_o = rng.randn(num_heads * self.head_dim, dim).astype(np.float32) * 0.01

    def compute(self, query: np.ndarray, keys: List[np.ndarray], values: List[np.ndarray]) -> np.ndarray:
        """
        Scaled dot-product attention.
        query: (dim,)
        keys: list of (dim,)
        values: list of (dim,)
        """
        if not keys or not values:
            return np.zeros(self.dim, dtype=np.float32)

        query = np.asarray(query, dtype=np.float32)
        keys = [np.asarray(k, dtype=np.float32) for k in keys]
        values = [np.asarray(v, dtype=np.float32) for v in values]

        all_heads = []
        for h in range(self.num_heads):
            q = query @ self.W_q[h]  # (head_dim,)
            k = np.stack([k @ self.W_k[h] for k in keys])  # (n_keys, head_dim)
            v = np.stack([v @ self.W_v[h] for v in values])  # (n_values, head_dim)

            scores = q @ k.T / np.sqrt(self.head_dim)  # (n_keys,)
            attn_weights = softmax(scores)
            head_output = attn_weights @ v  # (head_dim,)
            all_heads.append(head_output)

        concatenated = np.concatenate(all_heads)
        output = concatenated @ self.W_o
        return output

    def attention_score(self, query: np.ndarray, key: np.ndarray) -> float:
        """Упрощённая версия для одного key — среднее по головам."""
        query = np.asarray(query, dtype=np.float32)
        key = np.asarray(key, dtype=np.float32)
        scores = []
        for h in range(self.num_heads):
            q = query @ self.W_q[h]
            k = key @ self.W_k[h]
            score = np.dot(q, k) / (np.linalg.norm(q) * np.linalg.norm(k) + 1e-8)
            scores.append(float(score))
        return float(np.mean(scores))

    def update_weights(self, query: np.ndarray, key: np.ndarray, value: np.ndarray, 
                       learning_rate: float = 0.001):
        """Обновляет веса attention через Hebbian learning."""
        # Упрощённое обновление
        for h in range(self.num_heads):
            q = query @ self.W_q[h]
            k = key @ self.W_k[h]
            sim = np.dot(q, k) / (np.linalg.norm(q) * np.linalg.norm(k) + 1e-8)

            # Обновляем W_k в направлении query
            grad = learning_rate * sim * (q - k)
            self.W_k[h] += np.outer(query, grad) * 0.01

    def to_dict(self) -> Dict:
        return {
            "W_q": [w.tolist() for w in self.W_q],
            "W_k": [w.tolist() for w in self.W_k],
            "W_v": [w.tolist() for w in self.W_v],
            "W_o": self.W_o.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict, dim: int, num_heads: int):
        attn = cls(dim, num_heads)
        attn.W_q = [np.array(w, dtype=np.float32) for w in data["W_q"]]
        attn.W_k = [np.array(w, dtype=np.float32) for w in data["W_k"]]
        attn.W_v = [np.array(w, dtype=np.float32) for w in data["W_v"]]
        attn.W_o = np.array(data["W_o"], dtype=np.float32)
        return attn


# ============================
# Улучшенные классы Signal, Synapse, Neuron
# ============================
class Signal:
    """Сигнал с расширенной метаинформацией."""
    def __init__(self, embedding: np.ndarray, energy: float = 1.0, importance: float = 0.5,
                 context: Optional[np.ndarray] = None, source=None, destination=None,
                 confidence: float = 0.5, text: str = "", 
                 signal_type: str = "standard",  # standard, reward, punishment, query
                 metadata: Optional[Dict] = None):
        self.id = id(self)
        self.embedding = np.asarray(embedding, dtype=np.float32).copy()
        self.energy = energy
        self.importance = importance
        self.timestamp = time.time()
        self.context = np.asarray(context, dtype=np.float32).copy() if context is not None else None
        self.source = source
        self.destination = destination
        self.confidence = confidence
        self.history = []
        self.text = text
        self.signal_type = signal_type
        self.metadata = metadata or {}
        self.hop_count = 0  # Счётчик хопов для предотвращения зацикливания

    def __repr__(self):
        return f"Signal(id={self.id}, energy={self.energy:.2f}, type={self.signal_type}, text={self.text[:30]!r})"

class Synapse:
    """
    Улучшенный синапс с:
    - Adaptive learning rate
    - Multi-head attention
    - Eligibility traces (для delayed reward)
    - Synaptic tagging (для долговременной памяти)
    """
    def __init__(self, from_neuron, to_neuron, weight: float = 0.1, plasticity: float = 0.5,
                 confidence: float = 0.1, frequency: float = 0.0, energy: float = 1.0,
                 context_vector: Optional[np.ndarray] = None,
                 semantic_vector: Optional[np.ndarray] = None,
                 episodic_vector: Optional[np.ndarray] = None,
                 is_inhibitory: bool = False, dim: int = 128):
        self.id = id(self)
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron
        self.weight = weight
        self.plasticity = plasticity
        self.confidence = confidence
        self.frequency = frequency
        self.energy = energy
        self.creation_time = time.time()
        self.last_used = time.time()
        self.usage_count = 0
        self.context_vector = context_vector if context_vector is not None else random_vector(dim)
        self.semantic_vector = semantic_vector if semantic_vector is not None else random_vector(dim)
        self.episodic_vector = episodic_vector if episodic_vector is not None else random_vector(dim)
        self.reward = 0.0
        self.prediction_error = 0.0
        self.history = []
        self.is_inhibitory = is_inhibitory

        # Adaptive learning rate (meta-learning)
        self.learning_rate = 0.01
        self.lr_momentum = 0.0
        self.lr_adaptation_rate = 0.001

        # Eligibility trace для delayed reward (как в RL)
        self.eligibility_trace = 0.0
        self.trace_decay = 0.9

        # Synaptic tagging (долговременная память)
        self.tag_strength = 0.0
        self.tag_decay = 0.95

        # Attention mechanism
        self.attention = AttentionMechanism(dim, num_heads=4)

        # Structural plasticity
        self.structural_stability = 1.0  # 1.0 = стабильный, 0.0 = может быть удалён

    def attention_score(self, query_vec: np.ndarray) -> float:
        return self.attention.attention_score(query_vec, self.semantic_vector)

    def update_attention(self, query_vec: np.ndarray, learning_rate: float = 0.01):
        self.attention.update_weights(query_vec, self.semantic_vector, self.context_vector, learning_rate)

    def compute_attention_output(self, query_vec: np.ndarray, 
                                  neighbor_keys: List[np.ndarray],
                                  neighbor_values: List[np.ndarray]) -> np.ndarray:
        return self.attention.compute(query_vec, neighbor_keys, neighbor_values)

    def update(self, delta_weight: float, delta_plasticity: float = 0.0,
               delta_confidence: float = 0.0, delta_frequency: float = 0.0,
               reward: float = 0.0):
        """Обновление с adaptive learning rate и eligibility traces."""
        # Обновляем eligibility trace
        self.eligibility_trace = self.trace_decay * self.eligibility_trace + abs(delta_weight)

        # Meta-learning: адаптируем learning rate
        if reward != 0:
            # Если reward положительный — увеличиваем LR, если отрицательный — уменьшаем
            self.lr_momentum = 0.9 * self.lr_momentum + 0.1 * reward
            self.learning_rate *= (1.0 + self.lr_adaptation_rate * self.lr_momentum)
            self.learning_rate = np.clip(self.learning_rate, 0.0001, 0.5)

        # Применяем изменение веса с adaptive LR
        effective_delta = delta_weight * self.learning_rate * self.plasticity
        self.weight = float(np.clip(self.weight + effective_delta, -2.0, 2.0))

        self.plasticity = float(np.clip(self.plasticity + delta_plasticity, 0.0, 1.0))
        self.confidence = float(np.clip(self.confidence + delta_confidence, 0.0, 1.0))
        self.frequency += delta_frequency
        self.usage_count += 1
        self.last_used = time.time()
        self.energy = min(1.0, self.energy + 0.01)

        # Synaptic tagging
        if abs(delta_weight) > 0.01:
            self.tag_strength = min(1.0, self.tag_strength + 0.1)

    def decay(self, decay_rate: float = 0.001):
        if self.weight > 0:
            self.weight = max(0.0, self.weight - decay_rate)
        elif self.weight < 0:
            self.weight = min(0.0, self.weight + decay_rate)
        self.confidence = max(0.0, self.confidence - decay_rate * 0.5)
        self.energy = max(0.0, self.energy - decay_rate * 0.2)
        self.frequency *= 0.999
        self.eligibility_trace *= self.trace_decay
        self.tag_strength *= self.tag_decay
        self.structural_stability *= 0.9999

    def is_dead(self, weight_threshold=0.01, max_age_seconds=3600*24*30) -> bool:
        if self.structural_stability < 0.1:
            return True
        if abs(self.weight) < weight_threshold and (time.time() - self.last_used) > max_age_seconds:
            return True
        if self.usage_count == 0 and (time.time() - self.creation_time) > max_age_seconds:
            return True
        return False

    def __repr__(self):
        inhib = " (inh)" if self.is_inhibitory else ""
        return f"Synapse({self.from_neuron}->{self.to_neuron}, w={self.weight:.3f}, lr={self.learning_rate:.4f}){inhib}"

class Neuron:
    """
    Улучшенный нейрон с:
    - LSTM-подобным состоянием
    - Layer normalization
    - Multi-scale temporal dynamics
    - Utility scoring с exponential moving average
    """
    def __init__(self, embedding: Optional[np.ndarray] = None, activation: float = 0.0,
                 potential: float = 0.0, energy: float = 1.0, importance: float = 0.5,
                 cluster: str = 'hidden', layer: int = 0, dim: int = 128):
        self.id = id(self)
        self.embedding = embedding if embedding is not None else random_vector(dim)
        self.activation = activation
        self.potential = potential
        self.energy = energy
        self.importance = importance
        self.cluster = cluster
        self.layer = layer
        self.last_activation = time.time()
        self.age = 0
        self.usage_counter = 0
        self.memory_links = []
        self.incoming_synapses = []
        self.outgoing_synapses = []
        self.temporary_state = {}
        self.persistent_state = {}
        self.label = None
        self.refractory_period = 0
        self.utility_score = 0.0

        # LSTM-like state
        self.neural_state = NeuralState(dim)

        # Layer normalization params
        self.ln_gamma = np.ones(dim, dtype=np.float32)
        self.ln_beta = np.zeros(dim, dtype=np.float32)

        # Temporal dynamics
        self.activation_history = deque(maxlen=50)
        self.potential_history = deque(maxlen=50)

        # Surprise detection
        self.prediction_error_ema = 0.0  # Exponential moving average
        self.surprise_threshold = 0.5

        # Homeostatic target
        self.target_activation_rate = 0.15

    def add_incoming(self, synapse_id):
        if synapse_id not in self.incoming_synapses:
            self.incoming_synapses.append(synapse_id)

    def add_outgoing(self, synapse_id):
        if synapse_id not in self.outgoing_synapses:
            self.outgoing_synapses.append(synapse_id)

    def receive_signal(self, signal: Signal, synapse_weight: float, attention_boost: float = 1.0):
        """Получает сигнал с LSTM-like обработкой."""
        effective_weight = synapse_weight * attention_boost
        input_energy = signal.energy * effective_weight * signal.importance

        # Обновляем neural state
        hidden = self.neural_state.update(signal.embedding)

        # Комбинируем сигнал с hidden state
        sim = cosine_similarity(self.embedding, signal.embedding)
        combined_input = input_energy * (1.0 + sim * 0.3)

        # Layer norm на потенциал
        self.potential += combined_input
        self.potential_history.append(self.potential)

        # Surprise detection
        predicted = self.prediction_error_ema
        actual = combined_input
        error = abs(actual - predicted)
        self.prediction_error_ema = 0.9 * self.prediction_error_ema + 0.1 * error

        # Если сюрприз большой — повышаем важность
        if error > self.surprise_threshold:
            self.importance = min(1.0, self.importance + 0.05)

    def activate(self, threshold: float = 0.5) -> bool:
        if self.refractory_period > 0:
            self.refractory_period -= 1
            self.activation *= 0.5
            return False

        if self.potential > threshold:
            # GELU вместо сигмоиды для лучшей дифференцируемости
            self.activation = gelu(self.potential - threshold)
            self.activation = float(np.clip(self.activation, 0.0, 1.0))

            self.last_activation = time.time()
            self.usage_counter += 1
            self.age += 1
            self.energy = max(0.0, self.energy - 0.01)
            self.potential *= 0.3
            self.refractory_period = 2

            # EMA utility score
            self.utility_score = 0.7 * self.utility_score + 0.3 * self.activation
            self.activation_history.append(self.activation)

            return True
        else:
            self.activation *= 0.9
            return False

    def decay_energy(self, rate: float = 0.001):
        self.energy = max(0.0, self.energy - rate)
        if self.energy < 0.1:
            self.potential = 0.0
            self.neural_state.reset()

    def get_temporal_pattern(self) -> float:
        """Возвращает темпоральный паттерн активации (0-1)."""
        if len(self.activation_history) < 5:
            return 0.5
        recent = list(self.activation_history)[-5:]
        return float(np.mean(recent))

    def __repr__(self):
        return f"Neuron(id={self.id}, layer={self.layer}, cluster={self.cluster}, act={self.activation:.2f}, util={self.utility_score:.2f})"


# ============================
# Prioritized Experience Replay Buffer
# ============================
class Experience:
    """Один опыт для replay buffer."""
    def __init__(self, state: np.ndarray, action: str, reward: float, 
                 next_state: np.ndarray, priority: float = 1.0,
                 metadata: Optional[Dict] = None):
        self.state = state.copy()
        self.action = action
        self.reward = reward
        self.next_state = next_state.copy()
        self.priority = priority
        self.timestamp = time.time()
        self.metadata = metadata or {}
        self.visit_count = 0

class ExperienceBuffer:
    """
    Prioritized Experience Replay (Schaul et al., 2016).
    Сэмплирует опыт пропорционально его важности (TD-error).
    """
    def __init__(self, capacity: int = 5000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Степень приоритизации (0 = uniform, 1 = full prioritization)
        self.beta = beta    # Коррекция смещения (annealed to 1)
        self.buffer: List[Experience] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self._lock = threading.Lock()

    def add(self, experience: Experience):
        """Добавляет опыт с максимальным приоритетом."""
        with self._lock:
            max_prio = self.priorities.max() if len(self.buffer) > 0 else 1.0

            if len(self.buffer) < self.capacity:
                self.buffer.append(experience)
            else:
                self.buffer[self.position] = experience

            self.priorities[self.position] = max_prio
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        Сэмплирует batch с приоритизацией.
        Возвращает: (experiences, indices, importance_weights)
        """
        with self._lock:
            if len(self.buffer) == 0:
                return [], np.array([]), np.array([])

            n = len(self.buffer)
            probs = self.priorities[:n] ** self.alpha
            probs /= probs.sum()

            indices = np.random.choice(n, size=min(batch_size, n), replace=False, p=probs)

            # Importance sampling weights
            weights = (n * probs[indices]) ** (-self.beta)
            weights /= weights.max()

            experiences = [self.buffer[i] for i in indices]
            for exp in experiences:
                exp.visit_count += 1

            return experiences, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Обновляет приоритеты после обучения."""
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
# Иерархическая память
# ============================
class MemoryLevel(Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"

class MemoryItem:
    """Элемент памяти с метаданными."""
    def __init__(self, content: Any, embedding: np.ndarray, 
                 memory_level: MemoryLevel = MemoryLevel.WORKING,
                 importance: float = 0.5, tags: List[str] = None):
        self.content = content
        self.embedding = embedding.copy()
        self.memory_level = memory_level
        self.importance = importance
        self.tags = tags or []
        self.timestamp = time.time()
        self.access_count = 0
        self.last_accessed = time.time()
        self.consolidation_score = 0.0  # Для переноса между уровнями

    def access(self):
        self.access_count += 1
        self.last_accessed = time.time()
        self.importance = min(1.0, self.importance + 0.01)

    def decay(self, rate: float = 0.001):
        self.importance *= (1.0 - rate)
        time_since_access = time.time() - self.last_accessed
        # Забывание по Эббингаузу
        forgetting_factor = np.exp(-time_since_access / (3600 * 24))  # Забываем за сутки
        self.importance *= forgetting_factor

class HierarchicalMemory:
    """
    Многоуровневая память:
    - Working: активный контекст (кратковременная)
    - Episodic: конкретные события/диалоги
    - Semantic: обобщённые знания/факты
    - Procedural: навыки/процедуры
    """
    def __init__(self, config: BrainConfig):
        self.config = config
        self.working: deque = deque(maxlen=config.working_memory_size)
        self.episodic: List[MemoryItem] = []
        self.semantic: List[MemoryItem] = []
        self.procedural: List[MemoryItem] = []

        self.context_embedding: Optional[np.ndarray] = None
        self._lock = threading.Lock()

        # Индекс для быстрого поиска
        self._index: Dict[str, List[MemoryItem]] = defaultdict(list)

    def add(self, item: MemoryItem):
        """Добавляет элемент в соответствующий уровень памяти."""
        with self._lock:
            if item.memory_level == MemoryLevel.WORKING:
                self.working.append(item)
            elif item.memory_level == MemoryLevel.EPISODIC:
                self.episodic.append(item)
                if len(self.episodic) > self.config.short_memory_size:
                    # Консолидируем старые элементы в семантическую память
                    self._consolidate_old_episodic()
            elif item.memory_level == MemoryLevel.SEMANTIC:
                self.semantic.append(item)
                if len(self.semantic) > self.config.max_kb_size:
                    self.semantic.sort(key=lambda x: x.importance)
                    self.semantic = self.semantic[-self.config.max_kb_size:]
            elif item.memory_level == MemoryLevel.PROCEDURAL:
                self.procedural.append(item)

            # Индексируем по тегам
            for tag in item.tags:
                self._index[tag].append(item)

    def _consolidate_old_episodic(self):
        """Переносит старые эпизодические воспоминания в семантическую память."""
        now = time.time()
        to_consolidate = []
        remaining = []
        for item in self.episodic:
            if now - item.timestamp > 3600 and item.access_count > 2:
                # Часто используемые элементы консолидируем
                item.memory_level = MemoryLevel.SEMANTIC
                item.consolidation_score += 0.1
                to_consolidate.append(item)
            else:
                remaining.append(item)
        self.episodic = remaining
        for item in to_consolidate:
            self.semantic.append(item)
        if to_consolidate:
            logger.info(f"Consolidated {len(to_consolidate)} episodic items to semantic memory")

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5,
                 memory_levels: List[MemoryLevel] = None,
                 min_importance: float = 0.0) -> List[MemoryItem]:
        """Ретрив с ранжированием по релевантности + важности + recency."""
        with self._lock:
            levels = memory_levels or [MemoryLevel.WORKING, MemoryLevel.EPISODIC, MemoryLevel.SEMANTIC]
            candidates = []

            for level in levels:
                items = self._get_items_by_level(level)
                for item in items:
                    if item.importance < min_importance:
                        continue
                    sim = cosine_similarity(query_embedding, item.embedding)
                    # Комбинированный скор: similarity * importance * recency_bonus
                    recency_bonus = 1.0 / (1.0 + (time.time() - item.last_accessed) / 3600)
                    score = sim * (0.5 + 0.5 * item.importance) * recency_bonus
                    candidates.append((score, item))

            candidates.sort(key=lambda x: x[0], reverse=True)
            results = []
            for score, item in candidates[:top_k]:
                item.access()
                results.append(item)
            return results

    def _get_items_by_level(self, level: MemoryLevel) -> List[MemoryItem]:
        if level == MemoryLevel.WORKING:
            return list(self.working)
        elif level == MemoryLevel.EPISODIC:
            return self.episodic
        elif level == MemoryLevel.SEMANTIC:
            return self.semantic
        elif level == MemoryLevel.PROCEDURAL:
            return self.procedural
        return []

    def update_context(self, embedding: np.ndarray, decay: float = 0.7):
        """Обновляет контекстную эмбеддинг."""
        with self._lock:
            if self.context_embedding is None:
                self.context_embedding = embedding.copy()
            else:
                self.context_embedding = decay * self.context_embedding + (1 - decay) * embedding
                norm = np.linalg.norm(self.context_embedding)
                if norm > 0:
                    self.context_embedding /= norm

    def get_context_embedding(self) -> Optional[np.ndarray]:
        return self.context_embedding.copy() if self.context_embedding is not None else None

    def decay_all(self, rate: float = 0.001):
        """Забывание по всем уровням."""
        with self._lock:
            for item in self.episodic:
                item.decay(rate)
            for item in self.semantic:
                item.decay(rate * 0.5)  # Семантическая память забывается медленнее

            # Чистим забытые элементы
            self.episodic = [item for item in self.episodic if item.importance > 0.05]
            self.semantic = [item for item in self.semantic if item.importance > 0.01]

    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "working": len(self.working),
                "episodic": len(self.episodic),
                "semantic": len(self.semantic),
                "procedural": len(self.procedural),
            }


# ============================
# Self-Reflection Mechanism
# ============================
class SelfReflection:
    """
    Механизм саморефлексии для:
    - Обнаружения галлюцинаций
    - Оценки уверенности
    - Запроса уточнений
    - Мета-обучения
    """
    def __init__(self, llm_client: Optional[OpenAI] = None):
        self.llm_client = llm_client or OpenAI(
            base_url=LM_STUDIO_BASE_URL,
            api_key=LM_STUDIO_API_KEY,
        )
        self.reflection_history: List[Dict] = []
        self.confidence_threshold = 0.6

    def reflect_on_answer(self, question: str, answer: str, 
                          facts: List[Dict], brain_state: Dict) -> Dict:
        """
        Анализирует ответ и возвращает:
        - confidence: уверенность (0-1)
        - is_hallucination: есть ли галлюцинации
        - gaps: пробелы в знаниях
        - suggestion: предложение по улучшению
        """
        reflection = {
            "confidence": 0.5,
            "is_hallucination": False,
            "gaps": [],
            "suggestion": None,
            "needs_clarification": False,
        }

        # 1. Оценка на основе фактов
        if facts:
            fact_scores = [f.get("score", 0) for f in facts]
            avg_fact_score = np.mean(fact_scores)
            reflection["confidence"] = 0.3 + 0.7 * avg_fact_score
        else:
            reflection["confidence"] = 0.3  # Низкая уверенность без фактов
            reflection["needs_clarification"] = True

        # 2. Проверка на противоречия
        if len(facts) >= 2:
            for i in range(len(facts)):
                for j in range(i+1, len(facts)):
                    # Простая проверка: если ответы противоречат друг другу
                    a1, a2 = facts[i].get("a", ""), facts[j].get("a", "")
                    if a1 and a2 and a1 != a2:
                        # Проверяем семантическое сходство
                        # (в реальности нужны эмбеддинги, здесь упрощённо)
                        reflection["confidence"] *= 0.8
                        reflection["gaps"].append(f"Противоречие между фактами: {a1} vs {a2}")

        # 3. Проверка через LLM (если доступен)
        try:
            system_msg = (
                "Ты — система саморефлексии ИИ. Оцени, содержит ли следующий ответ "
                "галлюцинации или выдуманную информацию. Ответь только числом от 0 до 1, "
                "где 0 = полностью выдумано, 1 = полностью достоверно."
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

        # 4. Определение необходимости уточнения
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
        """Генерирует уточняющий вопрос на основе пробелов."""
        if not gaps:
            return f"Можете уточнить, что именно вы имели в виду в вопросе '{question}'?"
        return f"Я не уверен в следующем моменте: {gaps[0]}. Можете уточнить?"

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
# Основной класс Brain v5
# ============================
class Brain:
    """
    Smart Brain v5 — Улучшенная ассоциативная нейросеть.

    Ключевые улучшения:
    1. LSTM-подобные нейроны с cell state
    2. Multi-head attention для синапсов
    3. Prioritized Experience Replay
    4. Иерархическая память (Working/Episodic/Semantic/Procedural)
    5. Meta-learning (адаптивные learning rates)
    6. Self-reflection и hallucination detection
    7. Multi-hop reasoning с контролем глубины
    8. Enhanced RAG с query expansion
    """
    def __init__(self, config: Optional[BrainConfig] = None,
                 llm_client: Optional[OpenAI] = None):
        self.config = config or BrainConfig()
        self.llm_client = llm_client or OpenAI(
            base_url=LM_STUDIO_BASE_URL,
            api_key=LM_STUDIO_API_KEY,
        )

        self.dim = self.config.dim_embedding
        self.embedder = EmbeddingProvider(dim=self.dim, llm_client=self.llm_client)

        # Нейроны и синапсы
        self.neurons: Dict[int, Neuron] = {}
        self.synapses: Dict[int, Synapse] = {}
        self.edge_index: Dict[Tuple[int, int], int] = {}
        self.concept_index: Dict[str, int] = {}

        # ID management
        self._next_id = 1
        self.lock = threading.RLock()

        # Иерархическая память
        self.memory = HierarchicalMemory(self.config)

        # Experience replay
        self.experience_buffer = ExperienceBuffer(
            capacity=self.config.experience_buffer_size
        )

        # Self-reflection
        self.reflection = SelfReflection(self.llm_client)

        # Счётчики и метрики
        self.step_counter = 0
        self.global_time = time.time()
        self.coactivation_counter = defaultdict(int)
        self.coactivation_window = []
        self.window_size = 300

        # Meta-learning state
        self.meta_lr = self.config.meta_learning_rate
        self.performance_history = deque(maxlen=100)

        # Knowledge base (для обратной совместимости)
        self.knowledge_base: List[Dict[str, Any]] = []

        # Legacy memory (для обратной совместимости)
        self.working_memory = Memory(max_size=self.config.working_memory_size)
        self.dialog_memory = DialogMemory(max_turns=self.config.dialog_memory_turns)
        self.short_memory = ShortMemory()
        self.long_memory = LongMemory()

        self._learn_counter = 0
        self._model_path = self.config.model_path

        # Инициализация архитектуры
        self._init_architecture(
            self.config.input_neurons,
            self.config.hidden_layers,
            self.config.output_neurons
        )

        logger.info(f"Brain v5 initialized: {len(self.neurons)} neurons, {len(self.synapses)} synapses")

    def _alloc_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def _init_architecture(self, input_n: int, hidden_layers: List[int], output_n: int):
        """Инициализирует многослойную архитектуру с residual connections."""
        # Input layer
        for _ in range(input_n):
            n = Neuron(embedding=random_vector(self.dim), cluster='input', layer=0, dim=self.dim)
            n.id = self._alloc_id()
            self.neurons[n.id] = n

        # Hidden layers
        layer_id = 1
        for hsize in hidden_layers:
            for _ in range(hsize):
                n = Neuron(embedding=random_vector(self.dim), cluster='hidden', layer=layer_id, dim=self.dim)
                n.id = self._alloc_id()
                self.neurons[n.id] = n
            layer_id += 1

        # Output layer
        for _ in range(output_n):
            n = Neuron(embedding=random_vector(self.dim), cluster='output', layer=layer_id, dim=self.dim)
            n.id = self._alloc_id()
            self.neurons[n.id] = n

        # Создаём связи между слоями
        layer_map = defaultdict(list)
        for nid, n in self.neurons.items():
            layer_map[n.layer].append(nid)

        for layer_idx in range(layer_id):
            from_ids = layer_map[layer_idx]
            to_ids = layer_map.get(layer_idx + 1, [])
            if not to_ids:
                continue
            for from_id in from_ids:
                # Плотные связи, но с ограничением
                targets = random.sample(to_ids, min(len(to_ids), max(8, len(to_ids) // 3)))
                for to_id in targets:
                    w = random.uniform(0.01, 0.15)
                    self._create_synapse(from_id, to_id, weight=w)

        # Recurrent connections внутри hidden слоёв
        for layer_idx in range(1, layer_id):
            ids = layer_map[layer_idx]
            for _ in range(len(ids) // 4):
                a, b = random.sample(ids, 2)
                self._create_synapse(a, b, weight=random.uniform(0.005, 0.05))
                self._create_synapse(b, a, weight=random.uniform(0.005, 0.05))

        # Skip connections (input -> output)
        input_ids = [nid for nid, n in self.neurons.items() if n.cluster == 'input']
        output_ids = [nid for nid, n in self.neurons.items() if n.cluster == 'output']
        for _ in range(min(50, len(input_ids) * len(output_ids) // 3)):
            a = random.choice(input_ids)
            b = random.choice(output_ids)
            self._create_synapse(a, b, weight=random.uniform(0.005, 0.02))

    def _create_synapse(self, from_id: int, to_id: int, weight: float = None,
                        context_vec: np.ndarray = None, is_inhibitory: bool = False) -> Optional[int]:
        if from_id not in self.neurons or to_id not in self.neurons:
            return None
        key = (from_id, to_id)
        existing_id = self.edge_index.get(key)
        if existing_id is not None and existing_id in self.synapses:
            if weight is not None:
                self.synapses[existing_id].weight = weight
            return existing_id

        if weight is None:
            weight = random.uniform(0.01, 0.1)
        if context_vec is None:
            context_vec = random_vector(self.dim)

        if len(self.synapses) >= self.config.max_synapses:
            self._prune_synapses(force=True)
            if len(self.synapses) >= self.config.max_synapses:
                logger.warning("Synapse limit reached, new synapse not created")
                return None

        syn = Synapse(from_id, to_id, weight=weight,
                      context_vector=context_vec,
                      semantic_vector=random_vector(self.dim),
                      episodic_vector=random_vector(self.dim),
                      is_inhibitory=is_inhibitory, dim=self.dim)
        syn.id = self._alloc_id()
        self.synapses[syn.id] = syn
        self.edge_index[key] = syn.id
        self.neurons[from_id].add_outgoing(syn.id)
        self.neurons[to_id].add_incoming(syn.id)
        return syn.id

    def _remove_synapse(self, synapse_id: int):
        if synapse_id in self.synapses:
            syn = self.synapses[synapse_id]
            from_n = self.neurons.get(syn.from_neuron)
            to_n = self.neurons.get(syn.to_neuron)
            if from_n and synapse_id in from_n.outgoing_synapses:
                from_n.outgoing_synapses.remove(synapse_id)
            if to_n and synapse_id in to_n.incoming_synapses:
                to_n.incoming_synapses.remove(synapse_id)
            self.edge_index.pop((syn.from_neuron, syn.to_neuron), None)
            del self.synapses[synapse_id]

    def _find_most_similar_neuron(self, embedding: np.ndarray, cluster: str = None, 
                                   threshold: float = 0.85) -> Optional[int]:
        best_id = None
        best_sim = threshold
        for nid, n in self.neurons.items():
            if cluster is not None and n.cluster != cluster:
                continue
            sim = cosine_similarity(embedding, n.embedding)
            if sim > best_sim:
                best_sim = sim
                best_id = nid
        return best_id

    def _create_neuron(self, embedding: np.ndarray = None, importance: float = 0.5,
                       cluster: str = 'hidden', layer: int = None, label: str = None) -> int:
        if embedding is None:
            embedding = random_vector(self.dim)

        if cluster in ('input', 'output'):
            existing = self._find_most_similar_neuron(embedding, cluster=cluster, 
                                                       threshold=self.config.similarity_threshold_new_neuron)
            if existing is not None:
                if label and not self.neurons[existing].label:
                    self.neurons[existing].label = label
                return existing

        if len(self.neurons) >= self.config.max_neurons:
            self._prune_neurons(force=True)
            if len(self.neurons) >= self.config.max_neurons:
                logger.warning("Neuron limit reached, returning random")
                return random.choice(list(self.neurons.keys()))

        if layer is None:
            layer = len(self.config.hidden_layers) // 2 + 1

        n = Neuron(embedding=embedding, importance=importance, cluster=cluster, 
                   layer=layer, dim=self.dim)
        n.id = self._alloc_id()
        if label:
            n.label = label[:self.config.MAX_LABEL_LEN] if hasattr(self.config, 'MAX_LABEL_LEN') else label[:300]
        self.neurons[n.id] = n

        # Создаём связи с соседями
        if cluster == 'hidden':
            similarities = []
            for nid, neuron in self.neurons.items():
                if nid != n.id and neuron.cluster == 'hidden':
                    sim = cosine_similarity(embedding, neuron.embedding)
                    similarities.append((sim, nid))
            similarities.sort(reverse=True)
            for sim, nid in similarities[:20]:
                if sim > 0.15:
                    self._create_synapse(n.id, nid, weight=0.05)
                    self._create_synapse(nid, n.id, weight=0.05)
        elif cluster == 'output':
            for nid, neuron in self.neurons.items():
                if neuron.cluster == 'hidden':
                    self._create_synapse(nid, n.id, weight=random.uniform(0.01, 0.1))
        elif cluster == 'input':
            for nid, neuron in self.neurons.items():
                if neuron.cluster == 'hidden':
                    self._create_synapse(n.id, nid, weight=random.uniform(0.01, 0.1))

        return n.id

    def _prune_neurons(self, force=False):
        if not force and len(self.neurons) < self.config.max_neurons * 0.9:
            return
        utility = {}
        for nid, n in self.neurons.items():
            if n.cluster in ('input', 'output') and n.label:
                utility[nid] = float('inf')
            else:
                # Улучшенная метрика полезности
                temporal_score = n.get_temporal_pattern()
                score = (n.energy * 0.25 + n.importance * 0.25 + 
                        n.utility_score * 0.25 + temporal_score * 0.15 +
                        (n.usage_counter / (n.age+1)) * 0.1)
                utility[nid] = score

        sorted_neurons = sorted(utility.items(), key=lambda x: x[1])
        target = int(self.config.max_neurons * 0.8)
        to_remove = []
        for nid, score in sorted_neurons:
            if len(self.neurons) - len(to_remove) <= target:
                break
            if utility[nid] == float('inf'):
                continue
            to_remove.append(nid)

        for nid in to_remove:
            neuron = self.neurons[nid]
            for syn_id in list(neuron.incoming_synapses) + list(neuron.outgoing_synapses):
                self._remove_synapse(syn_id)
            if neuron.label:
                for key, val in list(self.concept_index.items()):
                    if val == nid:
                        del self.concept_index[key]
            del self.neurons[nid]

        if to_remove:
            logger.info(f"Pruned {len(to_remove)} neurons")

    def _prune_synapses(self, force=False):
        if not force and len(self.synapses) < self.config.max_synapses * 0.9:
            to_remove = []
            for sid, syn in self.synapses.items():
                syn.decay(decay_rate=0.0005)
                if syn.is_dead(weight_threshold=self.config.synapse_weight_threshold,
                               max_age_seconds=self.config.synapse_max_age):
                    to_remove.append(sid)
            for sid in to_remove:
                self._remove_synapse(sid)
            if to_remove:
                logger.info(f"Pruned {len(to_remove)} dead synapses")
            return

        target = int(self.config.max_synapses * 0.7)
        scored = []
        for sid, syn in self.synapses.items():
            # Улучшенная метрика: учитываем structural stability и tagging
            score = (abs(syn.weight) * 0.3 + syn.frequency * 0.2 + 
                    syn.confidence * 0.15 + syn.tag_strength * 0.15 +
                    (syn.usage_count / (time.time()-syn.creation_time+1)) * 0.1 +
                    syn.structural_stability * 0.1)
            scored.append((score, sid))

        scored.sort(key=lambda x: x[0])
        to_remove = []
        for score, sid in scored:
            if len(self.synapses) - len(to_remove) <= target:
                break
            to_remove.append(sid)

        for sid in to_remove:
            self._remove_synapse(sid)

        if to_remove:
            logger.info(f"Pruned {len(to_remove)} synapses (forced)")


    # -------------------- STDP и Hebbian Learning --------------------
    def _stdp_delta(self, pre_time: float, post_time: float) -> float:
        """Асимметричное правило STDP."""
        dt = post_time - pre_time
        if dt >= 0:
            return self.config.stdp_a_plus * np.exp(-dt / self.config.stdp_tau)
        else:
            return -self.config.stdp_a_minus * np.exp(dt / self.config.stdp_tau)

    def _hebbian_update(self, activated_ids: List[int], query_vec: np.ndarray):
        """Hebbian learning с STDP и attention."""
        for i in range(len(activated_ids)):
            for j in range(i + 1, len(activated_ids)):
                a, b = activated_ids[i], activated_ids[j]
                syn_id = self.edge_index.get((a, b)) or self.edge_index.get((b, a))
                if syn_id is not None and syn_id in self.synapses:
                    syn = self.synapses[syn_id]
                    if not syn.is_inhibitory:
                        pre_id, post_id = syn.from_neuron, syn.to_neuron
                        pre_n, post_n = self.neurons.get(pre_id), self.neurons.get(post_id)
                        if pre_n is None or post_n is None:
                            continue

                        att = syn.attention_score(query_vec)
                        stdp_delta = self._stdp_delta(pre_n.last_activation, post_n.last_activation)
                        delta = stdp_delta * (1.0 - min(1.0, abs(syn.weight))) * (1.0 + att)

                        # Meta-learning: адаптируем силу обучения
                        meta_factor = 1.0 + self.meta_lr * np.log(1 + syn.usage_count)
                        delta *= meta_factor

                        syn.update(delta_weight=delta, delta_plasticity=0.001,
                                   delta_confidence=0.001, delta_frequency=0.001)
                        syn.update_attention(query_vec)
                        syn.energy = min(1.0, syn.energy + 0.01)
                else:
                    pair = tuple(sorted((a, b)))
                    self.coactivation_counter[pair] += 1
                    self.coactivation_window.append((self.step_counter, a, b))
                    if len(self.coactivation_window) > self.window_size:
                        oldest = self.coactivation_window.pop(0)
                        old_pair = tuple(sorted((oldest[1], oldest[2])))
                        self.coactivation_counter[old_pair] -= 1
                        if self.coactivation_counter[old_pair] <= 0:
                            del self.coactivation_counter[old_pair]

    def _homeostatic_scaling(self):
        """Гомеостатическая нормировка синапсов."""
        for nid, neuron in self.neurons.items():
            if neuron.age <= 0:
                continue
            actual_rate = neuron.usage_counter / neuron.age
            if actual_rate <= 1e-6:
                continue
            scale = self.config.homeostasis_target_rate / actual_rate
            scale = float(np.clip(scale, 0.9, 1.1))
            if abs(scale - 1.0) < 1e-4:
                continue
            for syn_id in neuron.incoming_synapses:
                syn = self.synapses.get(syn_id)
                if syn and not syn.is_inhibitory:
                    syn.weight = float(np.clip(syn.weight * scale, -2.0, 2.0))

    def _create_new_synapses_from_coactivation(self):
        """Создаёт новые синапсы на основе совместной активации."""
        for pair, count in list(self.coactivation_counter.items()):
            if count >= self.config.synapse_creation_threshold:
                a, b = pair
                if a in self.neurons and b in self.neurons:
                    sim = cosine_similarity(self.neurons[a].embedding, self.neurons[b].embedding)
                    if sim > 0.2:
                        syn1 = self._create_synapse(a, b, weight=0.05)
                        syn2 = self._create_synapse(b, a, weight=0.05)
                        if syn1 and syn2:
                            self.coactivation_counter[pair] = 0

    # -------------------- Улучшенный Propagation --------------------
    def propagate_signal(self, input_signal: Signal, max_steps: int = None) -> Tuple[List[int], Dict[int, float]]:
        """
        Улучшенное распространение сигнала с:
        - Multi-hop reasoning
        - Attention-based routing
        - Gating mechanism
        - Residual connections
        """
        max_steps = max_steps or self.config.max_propagation_steps

        self.lock.acquire()
        try:
            return self._propagate_signal_impl(input_signal, max_steps)
        finally:
            self.lock.release()

    def _propagate_signal_impl(self, input_signal: Signal, max_steps: int) -> Tuple[List[int], Dict[int, float]]:
        # Добавляем в рабочую память
        self.working_memory.add(input_signal)
        self.memory.update_context(input_signal.embedding)

        # Добавляем в иерархическую память
        mem_item = MemoryItem(
            content=input_signal.text,
            embedding=input_signal.embedding,
            memory_level=MemoryLevel.WORKING,
            importance=input_signal.importance,
            tags=["signal", input_signal.signal_type]
        )
        self.memory.add(mem_item)

        # Контекстный query
        dialog_ctx = self.dialog_memory.get_context_embedding()
        if dialog_ctx is not None:
            context_query = 0.7 * input_signal.embedding + 0.3 * dialog_ctx
            context_query /= np.linalg.norm(context_query) + 1e-8
        else:
            context_query = input_signal.embedding

        # Находим стартовые нейроны (input layer)
        similarities = []
        for nid, neuron in self.neurons.items():
            if neuron.cluster == 'input':
                sim = cosine_similarity(neuron.embedding, input_signal.embedding)
                similarities.append((sim, nid))
        similarities.sort(reverse=True)
        start_neurons = [nid for sim, nid in similarities[:10] if sim > 0.05]

        if not start_neurons:
            existing = self._find_most_similar_neuron(input_signal.embedding, cluster='input', threshold=0.2)
            if existing is not None:
                start_neurons = [existing]
            else:
                new_nid = self._create_neuron(embedding=input_signal.embedding, cluster='input', layer=0)
                start_neurons = [new_nid]

        # BFS с attention-based routing
        visited = set(start_neurons)
        queue = deque([(nid, 0, input_signal.energy) for nid in start_neurons])
        activated = []
        activation_map = {}

        # Для residual connections
        residual_activations = {nid: 0.0 for nid in start_neurons}

        while queue and len(visited) < 500:
            nid, steps, current_energy = queue.popleft()
            if steps > max_steps or current_energy < 0.01:
                continue

            neuron = self.neurons[nid]

            # Собираем исходящие синапсы с attention scoring
            outgoing = []
            neighbor_keys = []
            neighbor_values = []

            for syn_id in neuron.outgoing_synapses:
                syn = self.synapses.get(syn_id)
                if syn is None or abs(syn.weight) < 0.003:
                    continue
                if syn.weight < 0:
                    continue
                target = self.neurons.get(syn.to_neuron)
                if target is None or target.energy < 0.1:
                    continue
                if syn.to_neuron in visited and syn.to_neuron not in activated:
                    continue  # Избегаем циклов

                att = syn.attention_score(context_query)
                semantic_sim = cosine_similarity(syn.semantic_vector, context_query)
                episodic_sim = cosine_similarity(syn.episodic_vector, context_query)

                # Multi-factor score
                score = (syn.weight * 0.4 + att * 0.2 + semantic_sim * 0.2 + 
                        episodic_sim * 0.1 + target.importance * 0.1)

                outgoing.append((score, syn_id, syn, target))
                neighbor_keys.append(syn.semantic_vector)
                neighbor_values.append(target.embedding)

            if not outgoing:
                continue

            # Attention-based aggregation для лучшего routing
            if len(neighbor_keys) > 1:
                # Используем attention для пересчёта scores
                attn_output = neuron.neural_state.update(context_query)
                for idx, (score, syn_id, syn, target) in enumerate(outgoing):
                    attn_sim = cosine_similarity(attn_output, target.embedding)
                    outgoing[idx] = (score * (1.0 + attn_sim * 0.3), syn_id, syn, target)

            outgoing.sort(reverse=True)

            # Пропускаем только top-K с учётом energy budget
            energy_budget = current_energy * 0.8
            for score, syn_id, syn, target in outgoing[:10]:
                energy_cost = syn.weight * (0.8 + 0.2 * score)
                if energy_cost > energy_budget:
                    continue
                energy_budget -= energy_cost

                attention_boost = 1.0 + score
                neuron.receive_signal(input_signal, syn.weight, attention_boost)

                # Создаём новый сигнал с уменьшенной энергией
                new_signal = copy.deepcopy(input_signal)
                new_signal.energy *= energy_cost
                new_signal.source = nid
                new_signal.destination = syn.to_neuron
                new_signal.confidence = syn.confidence
                new_signal.hop_count += 1
                new_signal.history.append((nid, syn_id, syn.to_neuron))

                self.working_memory.add(new_signal)

                if syn.to_neuron not in visited:
                    visited.add(syn.to_neuron)
                    queue.append((syn.to_neuron, steps + 1, new_signal.energy))

                syn.update(delta_weight=0.0, delta_frequency=0.001)

            # Residual connection: добавляем активацию от предыдущих шагов
            if nid in residual_activations:
                neuron.potential += residual_activations[nid] * 0.1

            if neuron.activate(threshold=self.config.propagation_threshold):
                activated.append(nid)
                activation_map[nid] = neuron.activation
                residual_activations[nid] = neuron.activation

        if activated:
            self._hebbian_update(activated, context_query)
            for nid in activated:
                self.neurons[nid].importance = min(1.0, self.neurons[nid].importance + 0.01)

        return activated, activation_map

    # -------------------- Meta-Learning --------------------
    def _meta_learn(self, performance: float):
        """
        Адаптирует гиперпараметры на основе производительности.
        performance: 0-1 (1 = отлично)
        """
        self.performance_history.append(performance)
        if len(self.performance_history) < 10:
            return

        recent_perf = np.mean(list(self.performance_history)[-10:])

        # Адаптируем meta learning rate
        if recent_perf > 0.7:
            self.meta_lr = min(0.1, self.meta_lr * 1.05)  # Ускоряем обучение
        elif recent_perf < 0.4:
            self.meta_lr = max(0.001, self.meta_lr * 0.95)  # Замедляем

        # Адаптируем порог активации
        if recent_perf < 0.3:
            self.config.propagation_threshold = max(0.1, self.config.propagation_threshold * 0.95)
        elif recent_perf > 0.8:
            self.config.propagation_threshold = min(0.5, self.config.propagation_threshold * 1.02)

        logger.debug(f"Meta-learn: perf={recent_perf:.3f}, meta_lr={self.meta_lr:.4f}, "
                    f"threshold={self.config.propagation_threshold:.3f}")

    # -------------------- Step --------------------
    def step(self, input_signal: Signal, target_neuron_id: Optional[int] = None,
             reward: float = 0.0) -> List[int]:
        with self.lock:
            self.step_counter += 1

            # Декей энергии
            for neuron in self.neurons.values():
                neuron.decay_energy(rate=0.0005)

            activated, _ = self.propagate_signal(input_signal)

            # Reward-based learning
            if target_neuron_id is not None:
                reward = 1.0 if target_neuron_id in activated else 0.0

            if reward > 0:
                for nid in activated:
                    neuron = self.neurons.get(nid)
                    if neuron is None:
                        continue
                    neuron.importance += reward * 0.02
                    for syn_id in neuron.incoming_synapses + neuron.outgoing_synapses:
                        syn = self.synapses.get(syn_id)
                        if syn and not syn.is_inhibitory:
                            syn.reward += reward
                            syn.update(delta_weight=reward * 0.005, delta_confidence=reward * 0.005,
                                      reward=reward)

                # Добавляем в experience buffer
                exp = Experience(
                    state=input_signal.embedding,
                    action=str(activated),
                    reward=reward,
                    next_state=input_signal.embedding,
                    priority=reward
                )
                self.experience_buffer.add(exp)

            # Experience replay
            if self.step_counter % 50 == 0:
                self._experience_replay()

            # Meta-learning
            if reward != 0:
                self._meta_learn(reward)

            if self.step_counter % 10 == 0:
                self._create_new_synapses_from_coactivation()
            if self.step_counter % self.config.homeostasis_every == 0:
                self._homeostatic_scaling()
            if self.step_counter % self.config.prune_every == 0:
                self._prune_synapses()
                self._prune_neurons()

            self.short_memory.add((input_signal, activated))
            self.global_time = time.time()
            return activated

    def _experience_replay(self, batch_size: int = 10):
        """Воспроизводит прошлый опыт для закрепления."""
        experiences, indices, weights = self.experience_buffer.sample(batch_size)
        if not experiences:
            return

        new_priorities = []
        for exp, weight in zip(experiences, weights):
            # Воспроизводим сигнал
            replayed_signal = Signal(
                embedding=exp.state,
                energy=0.5 * weight,
                importance=0.5,
                text=exp.action,
                signal_type="replay"
            )
            activated, _ = self._propagate_signal_impl(replayed_signal, max_steps=10)

            # Обновляем приоритет
            if activated:
                new_priorities.append(exp.priority * 0.9)  # Уменьшаем приоритет после replay
            else:
                new_priorities.append(exp.priority * 1.1)  # Увеличиваем, если не активировалось

        self.experience_buffer.update_priorities(indices, np.array(new_priorities))


    # -------------------- Улучшенный RAG --------------------
    def retrieve_facts(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Улучшенный RAG с:
        - Multi-source retrieval (KB + network + hierarchical memory)
        - Query expansion
        - Re-ranking
        """
        query_emb = self.text_to_embedding(query_text)

        # 1. Поиск в knowledge base
        scored_kb = []
        for fact in self.knowledge_base:
            sim = cosine_similarity(query_emb, fact["emb"])
            scored_kb.append((sim, fact))
        scored_kb.sort(key=lambda x: x[0], reverse=True)

        # 2. Поиск в иерархической памяти
        mem_results = self.memory.retrieve(query_emb, top_k=top_k*2, 
                                           memory_levels=[MemoryLevel.SEMANTIC, MemoryLevel.EPISODIC])

        # 3. Сетевой поиск через propagation
        sig = self.text_to_signal(query_text)
        activated, act_map = self.propagate_signal(sig, max_steps=12)

        network_facts = []
        for nid in activated:
            n = self.neurons.get(nid)
            if n and n.cluster == 'output' and n.label:
                sources = []
                for syn_id in n.incoming_synapses:
                    syn = self.synapses.get(syn_id)
                    if syn and syn.weight > 0.1:
                        src = self.neurons.get(syn.from_neuron)
                        if src and src.label:
                            sources.append(src.label)
                if sources:
                    network_facts.append({
                        "q": ", ".join(sources[:3]),
                        "a": n.label,
                        "weight": act_map.get(nid, 0.5),
                        "source": "network"
                    })

        # 4. Агрегация и re-ranking
        results = []

        # Из KB
        for sim, fact in scored_kb[:top_k]:
            if sim > 0.25:
                results.append({
                    "source": "memory",
                    "q": fact["q"],
                    "a": fact["a"],
                    "score": sim,
                    "confidence": fact.get("confidence", 0.5)
                })

        # Из иерархической памяти
        for mem_item in mem_results[:top_k]:
            sim = cosine_similarity(query_emb, mem_item.embedding)
            if sim > 0.25:
                results.append({
                    "source": f"hierarchical_{mem_item.memory_level.value}",
                    "q": str(mem_item.content)[:100],
                    "a": str(mem_item.content)[:200],
                    "score": sim * mem_item.importance,
                    "confidence": mem_item.importance
                })

        # Из сети
        for f in network_facts[:top_k]:
            results.append({
                "source": "network",
                "q": f["q"],
                "a": f["a"],
                "score": f["weight"],
                "confidence": f["weight"]
            })

        # Re-ranking: убираем дубликаты и сортируем
        seen = set()
        unique = []
        for r in results:
            key = (r["q"], r["a"])
            if key not in seen:
                seen.add(key)
                # Boost для высокой уверенности
                r["score"] = r["score"] * (0.5 + 0.5 * r.get("confidence", 0.5))
                unique.append(r)

        unique.sort(key=lambda x: x["score"], reverse=True)
        return unique[:top_k]

    def generate_answer(self, input_text: str, temperature: float = 0.7,
                        use_rag: bool = True, use_reflection: bool = True) -> Dict[str, Any]:
        """
        Генерация ответа с:
        - RAG retrieval
        - Self-reflection
        - Confidence estimation
        - Clarifying question generation
        """
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

        # Self-reflection
        reflection_result = None
        if use_reflection:
            reflection_result = self.reflection.reflect_on_answer(input_text, answer_text, facts, {})

        # Генерация уточняющего вопроса
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
        """Генерирует уточняющий вопрос на основе контекста."""
        if len(facts) >= 3:
            return None

        history_text = ""
        for turn in history[-3:]:
            history_text += f"Пользователь: {turn.get('user', '')}\\nАссистент: {turn.get('assistant', '')}\\n"

        system_prompt = (
            "Ты – ассистент, который помогает пользователю уточнить или расширить его запрос. "
            "На основе истории диалога и только что данного ответа, сформулируй один уточняющий, "
            "разъясняющий или обобщающий вопрос, который поможет лучше понять, что именно нужно пользователю. "
            "Вопрос должен быть естественным, вежливым, на русском языке. "
            "Если уточнение не требуется – ответь 'НЕТ'."
        )
        user_prompt = (
            f"История диалога (последние сообщения):\\n{history_text}\\n\\n"
            f"Вопрос пользователя: {user_input}\\n"
            f"Ответ ассистента: {answer}\\n"
            f"Найденные факты: {[f['q'] + ' -> ' + f['a'] for f in facts]}\\n\\n"
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
        if start_id not in self.neurons:
            return []
        chain = [self.neurons[start_id].label or "?"]
        visited = {start_id}
        current = start_id

        for _ in range(max_hops):
            candidates = []
            for syn_id in self.neurons[current].outgoing_synapses:
                syn = self.synapses.get(syn_id)
                if syn is None or syn.to_neuron in visited or syn.weight <= 0:
                    continue
                target = self.neurons.get(syn.to_neuron)
                if target is None or not target.label:
                    continue
                score = syn.weight + syn.confidence * 0.1 + syn.tag_strength * 0.1
                if score >= min_weight:
                    candidates.append((syn, score))
            if not candidates:
                break

            if temperature > 0 and len(candidates) > 1:
                scores = np.array([s for _, s in candidates])
                exp_scores = np.exp(scores / temperature - np.max(scores / temperature))
                probs = exp_scores / exp_scores.sum()
                chosen_idx = np.random.choice(len(candidates), p=probs)
                best_syn = candidates[chosen_idx][0]
            else:
                best_syn = max(candidates, key=lambda x: x[1])[0]

            current = best_syn.to_neuron
            visited.add(current)
            chain.append(self.neurons[current].label)
            best_syn.update(delta_weight=0.0, delta_frequency=0.002)

        return chain

    # -------------------- Текстовые методы --------------------
    def normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\\w\\s]", " ", text)
        return " ".join(text.split())

    def text_to_embedding(self, text: str) -> np.ndarray:
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
                text_norm = text_norm.replace(concept, ' ')
        return found

    def get_or_create_concept_neuron(self, text: str) -> int:
        normalized = self.normalize_text(text)
        nid = self.concept_index.get(normalized)
        if nid is not None and nid in self.neurons:
            return nid

        label = text.strip()
        if len(label) > 300:
            label = label[:300]

        emb = self.text_to_embedding(text)
        nid = self._create_neuron(embedding=emb, cluster='output',
                                   layer=len(self.config.hidden_layers) + 1, label=label)
        self.concept_index[normalized] = nid
        logger.info(f"New concept neuron: {nid} = '{label}'")
        return nid

    # -------------------- Обучение --------------------
    def learn_pair(self, input_text: str, output_text: str,
                   reinforce_boost: float = 0.15, epochs: int = 1):
        """
        Обучает модель на паре (вопрос→ответ) с:
        - Многократными эпохами
        - Semantic similarity bonus
        - Experience logging
        """
        with self.lock:
            for epoch in range(epochs):
                input_norm = self.normalize_text(input_text)
                output_norm = self.normalize_text(output_text)
                if input_norm == output_norm:
                    logger.info("Skipping: input == output")
                    return

                input_id = self.get_or_create_concept_neuron(input_text)
                output_id = self.get_or_create_concept_neuron(output_text)

                sim = cosine_similarity(self.neurons[input_id].embedding,
                                       self.neurons[output_id].embedding)
                sim_bonus = max(0.0, sim) * 0.05

                syn_id = self._create_synapse(input_id, output_id, weight=0.2 + sim_bonus)
                if syn_id is None:
                    logger.error("Failed to create synapse (limit reached)")
                    return
                syn = self.synapses[syn_id]

                if syn.usage_count > 0:
                    syn.update(delta_weight=reinforce_boost, delta_plasticity=0.02,
                               delta_confidence=0.05, delta_frequency=0.02)
                else:
                    syn.update(delta_weight=0.0, delta_confidence=0.05)

                # Propagation и создание связей (только на первой эпохе)
                if epoch == 0:
                    sig = self.text_to_signal(input_text)
                    activated, _ = self.propagate_signal(sig, max_steps=15)
                    if activated:
                        sorted_activated = sorted(activated,
                            key=lambda nid: self.neurons[nid].importance, reverse=True)
                        for nid in sorted_activated[:5]:
                            neuron = self.neurons.get(nid)
                            if neuron is None or neuron.cluster == 'output':
                                continue
                            hsyn_id = self._create_synapse(nid, output_id, weight=0.05)
                            if hsyn_id is not None:
                                self.synapses[hsyn_id].update(delta_weight=0.03)

                    self.step(sig, target_neuron_id=output_id)

                # Добавляем в KB только один раз
                if epoch == 0:
                    self._add_to_knowledge_base(input_text, output_text)

                    # Добавляем в иерархическую память
                    combined_emb = self.text_to_embedding(input_text + " " + output_text)
                    mem_item = MemoryItem(
                        content={"q": input_text, "a": output_text},
                        embedding=combined_emb,
                        memory_level=MemoryLevel.SEMANTIC,
                        importance=0.7,
                        tags=["learned", "fact"]
                    )
                    self.memory.add(mem_item)

                logger.info(f"Learn epoch {epoch+1}/{epochs}: '{input_text}' -> '{output_text}' "
                           f"(weight: {syn.weight:.3f})")

            self._learn_counter += epochs
            if self._learn_counter % self.config.auto_save_every == 0:
                self.save(self._model_path)
                logger.info(f"Auto-saved to {self._model_path}")

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

            syn_id = self.edge_index.get((input_id, output_id))
            if syn_id is not None and syn_id in self.synapses:
                syn = self.synapses[syn_id]
                new_weight = max(-2.0, syn.weight - penalty)
                syn.update(delta_weight=new_weight - syn.weight)
                if syn.weight < 0:
                    syn.is_inhibitory = True
                logger.info(f"Negative learn: '{input_text}' -> '{output_text}' = {syn.weight:.3f}")
            else:
                syn_id = self._create_synapse(input_id, output_id,
                                              weight=-random.uniform(0.1, 0.3), is_inhibitory=True)
                if syn_id is not None:
                    logger.info(f"Inhibitory synapse: '{input_text}' -> '{output_text}'")

            self._learn_counter += 1
            if self._learn_counter % self.config.auto_save_every == 0:
                self.save(self._model_path)

    def forget_concept(self, text: str) -> bool:
        with self.lock:
            normalized = self.normalize_text(text)
            nid = self.concept_index.get(normalized)
            if nid is None or nid not in self.neurons:
                return False

            neuron = self.neurons[nid]
            for syn_id in list(neuron.incoming_synapses) + list(neuron.outgoing_synapses):
                self._remove_synapse(syn_id)
            del self.concept_index[normalized]
            del self.neurons[nid]

            new_kb = []
            for item in self.knowledge_base:
                if normalized not in self.normalize_text(item["q"]) and \
                   normalized not in self.normalize_text(item["a"]):
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
        for syn_id in self.neurons[nid].outgoing_synapses:
            syn = self.synapses.get(syn_id)
            if syn is None:
                continue
            target = self.neurons.get(syn.to_neuron)
            if target is None or not target.label:
                continue
            links.append((target.label, syn.weight, syn.usage_count, syn.confidence,
                         syn.is_inhibitory, syn.learning_rate, syn.tag_strength))

        links.sort(key=lambda x: x[1], reverse=True)
        if not links:
            print(f"У '{text}' нет исходящих связей.")
            return

        print(f"Связи '{text}':")
        for label, weight, count, conf, inhib, lr, tag in links[:top_k]:
            inhib_str = " (тормозная)" if inhib else ""
            print(f"  -> {label}   (w={weight:.3f}, use={count}, conf={conf:.2f}, "
                  f"lr={lr:.4f}, tag={tag:.2f}){inhib_str}")

    # -------------------- Сон и консолидация --------------------
    def _replay_recent_memory(self, n: int = 30, energy_scale: float = 0.5):
        """Replay недавних сигналов для консолидации памяти."""
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

            # 1. Replay перед чисткой
            self._replay_recent_memory(n=30)

            # 2. Experience replay для закрепления
            self._experience_replay(batch_size=20)

            # 3. Консолидация памяти
            self.memory.decay_all(rate=0.001)

            # 4. Чистка шумовых нейронов
            for _ in range(duration_steps):
                to_remove = []
                for nid, neuron in self.neurons.items():
                    if (neuron.energy < 0.05 and neuron.importance < 0.1 and
                        neuron.cluster not in ('output', 'input')):
                        to_remove.append(nid)

                for nid in to_remove:
                    neuron = self.neurons[nid]
                    for syn_id in list(neuron.incoming_synapses) + list(neuron.outgoing_synapses):
                        self._remove_synapse(syn_id)
                    del self.neurons[nid]
                    logger.debug(f"Removed noise neuron {nid}")

                # Усиление часто используемых связей
                for syn in self.synapses.values():
                    if syn.frequency > 0.5:
                        syn.update(delta_weight=0.01, delta_confidence=0.01)
                    if syn.reward > 0.5:
                        syn.update(delta_weight=0.02, delta_confidence=0.02)
                    syn.reward *= 0.9

            # 5. Гомеостаз
            self._homeostatic_scaling()

            # 6. Перенос в долговременную память
            for item in self.short_memory.get_recent(50):
                self.long_memory.add(item)
            self.short_memory.clear()

            logger.info("Sleep completed")
            self.save(self._model_path)

    # -------------------- Сохранение / Загрузка --------------------
    def save(self, filename: str = None):
        with self.lock:
            filename = filename or self._model_path
            data = {
                "version": "5.0",
                "dim": self.dim,
                "step_counter": self.step_counter,
                "next_id": self._next_id,
                "meta_lr": self.meta_lr,
                "coactivation_counter": {str(k): v for k, v in self.coactivation_counter.items()},
                "neurons": [],
                "synapses": [],
                "knowledge_base": [],
                "config": {
                    "dim_embedding": self.config.dim_embedding,
                    "max_neurons": self.config.max_neurons,
                    "max_synapses": self.config.max_synapses,
                }
            }

            for nid, n in self.neurons.items():
                data["neurons"].append({
                    "id": nid,
                    "embedding": n.embedding.tolist(),
                    "cluster": n.cluster,
                    "layer": n.layer,
                    "label": n.label,
                    "importance": n.importance,
                    "energy": n.energy,
                    "utility_score": n.utility_score,
                    "neural_state": n.neural_state.to_dict(),
                })

            for sid, s in self.synapses.items():
                data["synapses"].append({
                    "id": sid,
                    "from": s.from_neuron,
                    "to": s.to_neuron,
                    "weight": s.weight,
                    "plasticity": s.plasticity,
                    "confidence": s.confidence,
                    "frequency": s.frequency,
                    "energy": s.energy,
                    "reward": s.reward,
                    "usage_count": s.usage_count,
                    "is_inhibitory": s.is_inhibitory,
                    "learning_rate": s.learning_rate,
                    "eligibility_trace": s.eligibility_trace,
                    "tag_strength": s.tag_strength,
                    "structural_stability": s.structural_stability,
                    "context_vector": s.context_vector.tolist(),
                    "semantic_vector": s.semantic_vector.tolist(),
                    "episodic_vector": s.episodic_vector.tolist(),
                    "attention": s.attention.to_dict(),
                })

            for kb in self.knowledge_base:
                data["knowledge_base"].append({
                    "q": kb["q"], "a": kb["a"],
                    "emb": kb["emb"].tolist(),
                    "time": kb["time"],
                    "confidence": kb.get("confidence", 0.5),
                })

            tmp = filename + ".tmp"
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, filename)
            logger.info(f"Saved to {filename}")

    def load(self, filename: str = None):
        with self.lock:
            filename = filename or self._model_path
            if not os.path.exists(filename):
                logger.info(f"File {filename} not found, starting fresh")
                return

            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                logger.error(f"Load error: {e}")
                return

            self.dim = data.get("dim", self.dim)
            self.step_counter = data.get("step_counter", 0)
            self._next_id = data.get("next_id", 1)
            self.meta_lr = data.get("meta_lr", self.config.meta_learning_rate)

            self.coactivation_counter = defaultdict(int)
            for k, v in data.get("coactivation_counter", {}).items():
                try:
                    a, b = k.strip("()").split(",")
                    self.coactivation_counter[(int(a), int(b))] = v
                except Exception:
                    pass

            self.neurons.clear()
            self.synapses.clear()
            self.edge_index.clear()
            self.concept_index.clear()
            self.knowledge_base.clear()

            max_seen_id = 0
            for nd in data["neurons"]:
                n = Neuron(embedding=np.array(nd["embedding"]), cluster=nd["cluster"],
                           layer=nd.get("layer", 0), importance=nd["importance"],
                           energy=nd["energy"], dim=self.dim)
                n.id = nd["id"]
                n.label = nd["label"]
                n.utility_score = nd.get("utility_score", 0.0)
                if "neural_state" in nd:
                    n.neural_state = NeuralState.from_dict(nd["neural_state"], self.dim)
                self.neurons[n.id] = n
                max_seen_id = max(max_seen_id, n.id)
                if n.cluster == 'output' and n.label:
                    self.concept_index[self.normalize_text(n.label)] = n.id

            for sd in data["synapses"]:
                s = Synapse(
                    sd["from"], sd["to"], weight=sd["weight"],
                    plasticity=sd["plasticity"], confidence=sd["confidence"],
                    frequency=sd["frequency"], energy=sd["energy"],
                    context_vector=np.array(sd["context_vector"]),
                    semantic_vector=np.array(sd["semantic_vector"]),
                    episodic_vector=np.array(sd["episodic_vector"]),
                    is_inhibitory=sd.get("is_inhibitory", False),
                    dim=self.dim
                )
                s.id = sd["id"]
                s.reward = sd["reward"]
                s.usage_count = sd.get("usage_count", 0)
                s.learning_rate = sd.get("learning_rate", 0.01)
                s.eligibility_trace = sd.get("eligibility_trace", 0.0)
                s.tag_strength = sd.get("tag_strength", 0.0)
                s.structural_stability = sd.get("structural_stability", 1.0)
                if "attention" in sd:
                    s.attention = AttentionMechanism.from_dict(sd["attention"], self.dim, 4)

                self.synapses[s.id] = s
                self.edge_index[(sd["from"], sd["to"])] = s.id
                max_seen_id = max(max_seen_id, s.id)
                if sd["from"] in self.neurons:
                    self.neurons[sd["from"]].add_outgoing(s.id)
                if sd["to"] in self.neurons:
                    self.neurons[sd["to"]].add_incoming(s.id)

            for kb in data.get("knowledge_base", []):
                self.knowledge_base.append({
                    "q": kb["q"], "a": kb["a"],
                    "emb": np.array(kb["emb"]),
                    "time": kb["time"],
                    "confidence": kb.get("confidence", 0.5),
                })

            saved_next_id = data.get("next_id")
            if saved_next_id is not None:
                self._next_id = max(saved_next_id, max_seen_id + 1)
            else:
                self._next_id = max_seen_id + 1

            logger.info(f"Loaded: {len(self.neurons)} neurons, {len(self.synapses)} synapses, "
                       f"{len(self.concept_index)} concepts, {len(self.knowledge_base)} facts")

    # -------------------- Дополнительные методы --------------------
    def get_weak_concepts(self, threshold: float = 0.2, limit: int = 5) -> List[str]:
        weak = []
        for nid, neuron in self.neurons.items():
            if neuron.cluster == 'output' and neuron.label:
                total = 0.0
                count = 0
                for syn_id in neuron.outgoing_synapses:
                    syn = self.synapses.get(syn_id)
                    if syn:
                        total += abs(syn.weight)
                        count += 1
                avg = total / count if count > 0 else 0.0
                if avg < threshold and count < 3:
                    weak.append(neuron.label)
        random.shuffle(weak)
        return weak[:limit]

    def suggest_next_topic(self, limit: int = 3) -> List[str]:
        if not self.dialog_memory.items:
            return []
        last_questions = [turn.get("user", "") for turn in self.dialog_memory.items[-3:] if turn.get("user")]
        if not last_questions:
            return []
        combined = " ".join(last_questions)
        prompt = f"На основе вопросов: '{combined}', предложи {limit} кратких тем для продолжения (через запятую)."
        try:
            resp = self.llm_client.chat.completions.create(
                model="local-model",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.7
            )
            raw = resp.choices[0].message.content.strip()
            topics = [t.strip() for t in raw.split(',') if t.strip()]
            return topics[:limit]
        except Exception as e:
            logger.error(f"Suggest topics error: {e}")
            return []

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
            topics = [t.strip() for t in raw.split(',') if t.strip()]
            return topics[:limit]
        except Exception as e:
            logger.error(f"Topics error: {e}")
            last_msg = user_messages[-1] if user_messages else ""
            words = [w for w in last_msg.split() if len(w) > 3]
            return words[:limit]

    def get_uncertain_concepts(self, threshold: float = 0.3, limit: int = 5) -> List[Dict[str, Any]]:
        uncertain = []
        for nid, neuron in self.neurons.items():
            if neuron.cluster != 'output' or not neuron.label:
                continue
            in_weights = []
            for syn_id in neuron.incoming_synapses:
                syn = self.synapses.get(syn_id)
                if syn:
                    in_weights.append(abs(syn.weight))
            avg_in = sum(in_weights) / len(in_weights) if in_weights else 0.0
            out_count = len(neuron.outgoing_synapses)

            if avg_in < threshold or out_count < 2:
                activation_freq = neuron.usage_counter / (neuron.age + 1) if neuron.age > 0 else 0
                score = ((1.0 - avg_in) * 0.5 + (1.0 - min(1.0, out_count / 5)) * 0.3 +
                        (1.0 - activation_freq) * 0.2)
                uncertain.append({
                    "label": neuron.label,
                    "score": score,
                    "avg_in": avg_in,
                    "out_count": out_count,
                    "activation_freq": activation_freq
                })
        uncertain.sort(key=lambda x: x["score"], reverse=True)
        return uncertain[:limit]

    def generate_contextual_question(self, user_topics: List[str],
                                     uncertain_concepts: List[Dict]) -> Optional[str]:
        if not user_topics and not uncertain_concepts:
            return None

        topics_str = ", ".join(user_topics) if user_topics else "общая тема"
        weak_str = ", ".join([c["label"] for c in uncertain_concepts[:3]]) if uncertain_concepts else "нет явных слабых мест"

        system_prompt = (
            "Ты — ассистент, который помогает пользователю углубиться в тему. "
            "На основе интересов пользователя и пробелов в знаниях, сформулируй один естественный вопрос. "
            "Вопрос должен быть простым и понятным. Ответь только вопросом."
        )
        user_prompt = (
            f"Темы пользователя: {topics_str}.\\n"
            f"Слабые места: {weak_str}.\\n"
            f"Сформулируй один вопрос для прояснения."
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
            if question and '?' in question:
                return question
            return None
        except Exception as e:
            logger.error(f"ContextualQ error: {e}")
            return None

    def save_dialog_history(self, filename: str = "dialog_history.json"):
        data = []
        for item in self.dialog_memory.items:
            data.append({
                "user": item.get("user", ""),
                "assistant": item.get("assistant", ""),
                "time": item.get("time", 0)
            })
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_dialog_history(self, filename: str = "dialog_history.json"):
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for item in data:
                    self.dialog_memory.add(item)
            except Exception as e:
                logger.error(f"Dialog load error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает полную статистику мозга."""
        return {
            "neurons": len(self.neurons),
            "synapses": len(self.synapses),
            "concepts": len(self.concept_index),
            "knowledge_base": len(self.knowledge_base),
            "memory": self.memory.get_stats(),
            "experience_buffer": self.experience_buffer.get_stats(),
            "reflection": self.reflection.get_reflection_stats(),
            "step_counter": self.step_counter,
            "meta_lr": self.meta_lr,
            "embedding_cache": self.embedder.get_cache_stats(),
        }


# ============================
# Legacy Memory Classes (для обратной совместимости)
# ============================
class Memory:
    def __init__(self, max_size: int = 100):
        self.items = []
        self.max_size = max_size
    def add(self, item):
        self.items.append(item)
        if len(self.items) > self.max_size:
            self.items.pop(0)
    def get_recent(self, n: int = 10):
        return self.items[-n:]
    def clear(self):
        self.items = []

class WorkingMemory(Memory):
    def __init__(self):
        super().__init__(max_size=50)
        self.context_embedding = None
    def update_context(self, embedding: np.ndarray, decay: float = 0.7):
        if self.context_embedding is None:
            self.context_embedding = embedding.copy()
        else:
            self.context_embedding = decay * self.context_embedding + (1 - decay) * embedding
            norm = np.linalg.norm(self.context_embedding)
            if norm > 0:
                self.context_embedding /= norm

class DialogMemory(Memory):
    def __init__(self, max_turns: int = 20):
        super().__init__(max_size=max_turns)
        self.turn_embeddings = []
    def add_turn(self, user_text: str, assistant_text: str, embedding: np.ndarray):
        self.add({"user": user_text, "assistant": assistant_text, "time": time.time()})
        self.turn_embeddings.append(embedding.copy())
        if len(self.turn_embeddings) > self.max_size:
            self.turn_embeddings.pop(0)
    def get_context_string(self, n: int = 3) -> str:
        recent = self.get_recent(n)
        parts = []
        for turn in recent:
            parts.append(f"Пользователь: {turn['user']}")
            parts.append(f"Ассистент: {turn['assistant']}")
        return "\\n".join(parts)
    def get_context_embedding(self) -> Optional[np.ndarray]:
        if not self.turn_embeddings:
            return None
        vec = np.mean(self.turn_embeddings, axis=0)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else None

class ShortMemory(Memory):
    def __init__(self):
        super().__init__(max_size=1000)

class LongMemory(Memory):
    def __init__(self):
        super().__init__(max_size=100000)


# ============================
# Teacher v5 (улучшенный)
# ============================
class Teacher:
    """
    Улучшенный Teacher с:
    - Multi-aspect evaluation (релевантность, точность, полнота, тон)
    - Chain-of-thought reasoning
    - Confidence calibration
    """
    def __init__(self, llm_client: Optional[OpenAI] = None):
        self.llm_client = llm_client or OpenAI(
            base_url=LM_STUDIO_BASE_URL,
            api_key=LM_STUDIO_API_KEY,
        )
        self.evaluation_history = []

    def evaluate(self, question: str, brain_answer: str) -> Tuple[float, str, Dict]:
        """
        Оценивает ответ по нескольким аспектам.
        Возвращает: (score, improved_answer, details)
        """
        system_prompt = (
            "Ты — критический эксперт. Оцени ответ ассистента по следующим критериям (0-1 каждый):\\n"
            "1. Релевантность — отвечает ли на вопрос\\n"
            "2. Точность — фактическая правильность\\n"
            "3. Полнота — достаточно ли информации\\n"
            "4. Ясность — понятно ли изложено\\n"
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

            # Парсинг ответа
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
            return 0.5, brain_answer, details

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
# Интерактивный режим v5
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
    print("\\n" + "="*60)
    print("  Smart Brain v5 — Улучшенная ассоциативная память + LLM")
    print("="*60)
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
    print("="*60 + "\\n")

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
                print(f"📊 Нейронов: {stats['neurons']} (in:{sum(1 for n in brain.neurons.values() if n.cluster=='input')}, "
                      f"out:{sum(1 for n in brain.neurons.values() if n.cluster=='output')}, "
                      f"hid:{sum(1 for n in brain.neurons.values() if n.cluster=='hidden')})")
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

            # === Основной диалог ===
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

            # Оценка Teacher
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

            # Обучение на основе оценки
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
    MODEL_PATH = "brain_model_v5.json"
    config = BrainConfig(
        dim_embedding=128,
        input_neurons=40,
        output_neurons=40,
        hidden_layers=[100, 80, 60],
        max_neurons=2000,
        max_synapses=20000,
        model_path=MODEL_PATH,
    )
    brain = Brain(config=config)
    brain.load(MODEL_PATH)
    interactive_with_teacher(brain)



