import numpy as np
import random
import time
import json
import os
import re
import zlib
import copy
import threading
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple, Optional, Any
from openai import OpenAI


# ============================
# Настройка подключения к LM Studio
# ============================
client = OpenAI(
    base_url="http://192.168.0.13:1234/v1",
    api_key="not-needed",
)

# ============================
# Вспомогательные функции
# ============================
def random_vector(dim: int = 128) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()

# ============================
# Провайдер эмбеддингов
# ============================
class EmbeddingProvider:
    # (без изменений)
    def __init__(self, dim: int = 128, api_model: str = "local-model"):
        self.dim = dim
        self.api_model = api_model
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_lock = threading.Lock()
        self._api_available = True

    def _local_embedding(self, text: str) -> np.ndarray:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        if not tokens:
            return random_vector(self.dim)
        ngrams = tokens[:]
        for i in range(len(tokens) - 1):
            ngrams.append(tokens[i] + "_" + tokens[i+1])
        vec = np.zeros(self.dim, dtype=np.float32)
        for idx, token in enumerate(ngrams):
            h = zlib.adler32(token.encode("utf-8")) + idx * 31
            rng = np.random.RandomState(h % (2**31))
            tok_vec = rng.randn(self.dim).astype(np.float32)
            weight = 1.0 / (1.0 + idx * 0.1)
            vec += tok_vec * weight
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else random_vector(self.dim)

    def get_embedding(self, text: str) -> np.ndarray:
        if not text:
            return random_vector(self.dim)
        with self._cache_lock:
            if text in self._cache:
                return self._cache[text].copy()
        if self._api_available:
            try:
                resp = client.embeddings.create(
                    model=self.api_model,
                    input=[text]
                )
                vec = np.array(resp.data[0].embedding, dtype=np.float32)
                if len(vec) != self.dim:
                    vec = self._local_embedding(text)
                with self._cache_lock:
                    self._cache[text] = vec.copy()
                return vec
            except Exception:
                self._api_available = False
        vec = self._local_embedding(text)
        with self._cache_lock:
            self._cache[text] = vec.copy()
        return vec

# ============================
# Классы Signal, Synapse, Neuron (без изменений)
# ============================
class Signal:
    # (без изменений)
    def __init__(self, embedding: np.ndarray, energy: float = 1.0, importance: float = 0.5,
                 context: Optional[np.ndarray] = None, source=None, destination=None,
                 confidence: float = 0.5, text: str = ""):
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
    def __repr__(self):
        return f"Signal(id={self.id}, energy={self.energy:.2f}, text={self.text[:30]!r})"

class Synapse:
    # (без изменений)
    def __init__(self, from_neuron, to_neuron, weight: float = 0.1, plasticity: float = 0.5,
                 confidence: float = 0.1, frequency: float = 0.0, energy: float = 1.0,
                 context_vector: Optional[np.ndarray] = None,
                 semantic_vector: Optional[np.ndarray] = None,
                 episodic_vector: Optional[np.ndarray] = None,
                 is_inhibitory: bool = False):
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
        self.context_vector = context_vector if context_vector is not None else random_vector()
        self.semantic_vector = semantic_vector if semantic_vector is not None else random_vector()
        self.episodic_vector = episodic_vector if episodic_vector is not None else random_vector()
        self.reward = 0.0
        self.prediction_error = 0.0
        self.history = []
        self.is_inhibitory = is_inhibitory
        self.attention_heads = 4
        self.key_vectors = [random_vector() for _ in range(self.attention_heads)]
        self.value_vectors = [random_vector() for _ in range(self.attention_heads)]

    def attention_score(self, query_vec: np.ndarray) -> float:
        scores = []
        for k_vec in self.key_vectors:
            scores.append(cosine_similarity(query_vec, k_vec))
        return float(np.mean(scores))

    def update_attention(self, query_vec: np.ndarray, learning_rate: float = 0.01):
        for i, k_vec in enumerate(self.key_vectors):
            sim = cosine_similarity(query_vec, k_vec)
            self.key_vectors[i] = k_vec + learning_rate * sim * (query_vec - k_vec)
            kv_norm = np.linalg.norm(self.key_vectors[i])
            if kv_norm > 0:
                self.key_vectors[i] /= kv_norm

    def update(self, delta_weight: float, delta_plasticity: float = 0.0,
               delta_confidence: float = 0.0, delta_frequency: float = 0.0):
        self.weight = float(np.clip(self.weight + delta_weight, -2.0, 2.0))
        self.plasticity = float(np.clip(self.plasticity + delta_plasticity, 0.0, 1.0))
        self.confidence = float(np.clip(self.confidence + delta_confidence, 0.0, 1.0))
        self.frequency += delta_frequency
        self.usage_count += 1
        self.last_used = time.time()
        self.energy = min(1.0, self.energy + 0.01)

    def decay(self, decay_rate: float = 0.001):
        if self.weight > 0:
            self.weight = max(0.0, self.weight - decay_rate)
        elif self.weight < 0:
            self.weight = min(0.0, self.weight + decay_rate)
        self.confidence = max(0.0, self.confidence - decay_rate * 0.5)
        self.energy = max(0.0, self.energy - decay_rate * 0.2)
        self.frequency *= 0.999

    def is_dead(self, weight_threshold=0.01, max_age_seconds=3600*24*30) -> bool:
        if abs(self.weight) < weight_threshold and (time.time() - self.last_used) > max_age_seconds:
            return True
        if self.usage_count == 0 and (time.time() - self.creation_time) > max_age_seconds:
            return True
        return False

    def __repr__(self):
        return f"Synapse({self.from_neuron}->{self.to_neuron}, w={self.weight:.3f})"

class Neuron:
    # (без изменений)
    def __init__(self, embedding: Optional[np.ndarray] = None, activation: float = 0.0,
                 potential: float = 0.0, energy: float = 1.0, importance: float = 0.5,
                 cluster: str = 'hidden', layer: int = 0):
        self.id = id(self)
        self.embedding = embedding if embedding is not None else random_vector()
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

    def add_incoming(self, synapse_id):
        if synapse_id not in self.incoming_synapses:
            self.incoming_synapses.append(synapse_id)

    def add_outgoing(self, synapse_id):
        if synapse_id not in self.outgoing_synapses:
            self.outgoing_synapses.append(synapse_id)

    def receive_signal(self, signal: Signal, synapse_weight: float, attention_boost: float = 1.0):
        effective_weight = synapse_weight * attention_boost
        self.potential += signal.energy * effective_weight * signal.importance
        sim = cosine_similarity(self.embedding, signal.embedding)
        self.potential += sim * 0.2

    def activate(self, threshold: float = 0.5):
        if self.refractory_period > 0:
            self.refractory_period -= 1
            self.activation *= 0.5
            return False
        if self.potential > threshold:
            self.activation = sigmoid(self.potential - threshold)
            self.last_activation = time.time()
            self.usage_counter += 1
            self.age += 1
            self.energy = max(0.0, self.energy - 0.01)
            self.potential *= 0.3
            self.refractory_period = 2
            self.utility_score = 0.7 * self.utility_score + 0.3 * self.activation
            return True
        else:
            self.activation *= 0.9
            return False

    def decay_energy(self, rate: float = 0.001):
        self.energy = max(0.0, self.energy - rate)
        if self.energy < 0.1:
            self.potential = 0.0

    def __repr__(self):
        return f"Neuron(id={self.id}, layer={self.layer}, cluster={self.cluster}, act={self.activation:.2f})"

# ============================
# Классы памяти (без изменений)
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
    def __init__(self, max_turns: int = 10):
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
        return "\n".join(parts)
    def get_context_embedding(self) -> Optional[np.ndarray]:
        if not self.turn_embeddings:
            return None
        vec = np.mean(self.turn_embeddings, axis=0)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else None

class ShortMemory(Memory):
    def __init__(self):
        super().__init__(max_size=500)

class LongMemory(Memory):
    def __init__(self):
        super().__init__(max_size=1000000)

# ============================
# Основной класс Brain (УЛУЧШЕННЫЙ)
# ============================
class Brain:
    def __init__(self, dim_embedding: int = 128,
                 input_neurons: int = 40,
                 output_neurons: int = 40,
                 hidden_layers: List[int] = None,
                 model_path: str = "brain_model_trained.json",
                 max_neurons: int = 800,
                 max_synapses: int = 8000):
        if hidden_layers is None:
            hidden_layers = [100, 80, 60]
        self.llm_client = client  # или llm_client, если передаёте извне
        self.dim = dim_embedding
        self.embedder = EmbeddingProvider(dim=dim_embedding)
        self.neurons: Dict[int, Neuron] = {}
        self.synapses: Dict[int, Synapse] = {}
        self.edge_index: Dict[Tuple[int, int], int] = {}
        self.concept_index: Dict[str, int] = {}

        self.hidden_layer_sizes = hidden_layers
        self.num_hidden_layers = len(hidden_layers)

        self.working_memory = WorkingMemory()
        self.dialog_memory = DialogMemory(max_turns=10)
        self.short_memory = ShortMemory()
        self.long_memory = LongMemory()
        self.step_counter = 0
        self.global_time = time.time()

        self.coactivation_counter = defaultdict(int)
        self.coactivation_window = []
        self.window_size = 300
        self.synapse_creation_threshold = 2
        self.synapse_max_age = 3600 * 24 * 7
        self.synapse_weight_threshold = 0.005

        self.MAX_LABEL_LEN = 300
        self.AUTO_SAVE_EVERY = 3
        self._learn_counter = 0
        self._model_path = model_path

        self.knowledge_base: List[Dict[str, Any]] = []
        self.max_kb_size = 2000

        self.max_neurons = max_neurons
        self.max_synapses = max_synapses
        self.prune_every = 25
        self.similarity_threshold_new_neuron = 0.85

        self._init_architecture(input_neurons, hidden_layers, output_neurons)

    def _init_architecture(self, input_n: int, hidden_layers: List[int], output_n: int):
        for _ in range(input_n):
            n = Neuron(embedding=random_vector(self.dim), cluster='input', layer=0)
            self.neurons[n.id] = n

        layer_id = 1
        for hsize in hidden_layers:
            for _ in range(hsize):
                n = Neuron(embedding=random_vector(self.dim), cluster='hidden', layer=layer_id)
                self.neurons[n.id] = n
            layer_id += 1

        for _ in range(output_n):
            n = Neuron(embedding=random_vector(self.dim), cluster='output', layer=layer_id)
            self.neurons[n.id] = n

        layer_map = defaultdict(list)
        for nid, n in self.neurons.items():
            layer_map[n.layer].append(nid)

        for layer_idx in range(layer_id):
            from_ids = layer_map[layer_idx]
            to_ids = layer_map.get(layer_idx + 1, [])
            if not to_ids:
                continue
            for from_id in from_ids:
                targets = random.sample(to_ids, min(len(to_ids), max(5, len(to_ids) // 4)))
                for to_id in targets:
                    w = random.uniform(0.01, 0.15)
                    self._create_synapse(from_id, to_id, weight=w)

        for layer_idx in range(1, layer_id):
            ids = layer_map[layer_idx]
            for _ in range(len(ids) // 5):
                a, b = random.sample(ids, 2)
                self._create_synapse(a, b, weight=random.uniform(0.005, 0.05))
                self._create_synapse(b, a, weight=random.uniform(0.005, 0.05))

        input_ids = [nid for nid, n in self.neurons.items() if n.cluster == 'input']
        output_ids = [nid for nid, n in self.neurons.items() if n.cluster == 'output']
        for _ in range(min(30, len(input_ids) * len(output_ids) // 3)):
            a = random.choice(input_ids)
            b = random.choice(output_ids)
            self._create_synapse(a, b, weight=random.uniform(0.005, 0.02))

    # -------------------- УЛУЧШЕННЫЙ _create_synapse --------------------
    def _create_synapse(self, from_id: int, to_id: int, weight: float = None,
                        context_vec: np.ndarray = None, is_inhibitory: bool = False) -> Optional[int]:
        if from_id not in self.neurons or to_id not in self.neurons:
            return None
        key = (from_id, to_id)
        existing_id = self.edge_index.get(key)
        if existing_id is not None and existing_id in self.synapses:
            # УЛУЧШЕНИЕ: обновляем вес, если передан новый
            if weight is not None:
                self.synapses[existing_id].weight = weight
            return existing_id

        if weight is None:
            weight = random.uniform(0.01, 0.1)
        if context_vec is None:
            context_vec = random_vector(self.dim)

        if len(self.synapses) >= self.max_synapses:
            self._prune_synapses(force=True)
            if len(self.synapses) >= self.max_synapses:
                print("[WARN] Достигнут лимит синапсов, новый не создан.")
                return None

        syn = Synapse(from_id, to_id, weight=weight,
                      context_vector=context_vec,
                      semantic_vector=random_vector(self.dim),
                      episodic_vector=random_vector(self.dim),
                      is_inhibitory=is_inhibitory)
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

    def _find_most_similar_neuron(self, embedding: np.ndarray, cluster: str = None, threshold: float = 0.85) -> Optional[int]:
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
            existing = self._find_most_similar_neuron(embedding, cluster=cluster, threshold=self.similarity_threshold_new_neuron)
            if existing is not None:
                if label and not self.neurons[existing].label:
                    self.neurons[existing].label = label
                return existing

        if len(self.neurons) >= self.max_neurons:
            self._prune_neurons(force=True)
            if len(self.neurons) >= self.max_neurons:
                print("[WARN] Достигнут лимит нейронов, возвращаем случайный.")
                return random.choice(list(self.neurons.keys()))

        if layer is None:
            layer = self.num_hidden_layers // 2 + 1
        n = Neuron(embedding=embedding, importance=importance, cluster=cluster, layer=layer)
        if label:
            n.label = label[:self.MAX_LABEL_LEN] if len(label) > self.MAX_LABEL_LEN else label
        self.neurons[n.id] = n

        if cluster == 'hidden':
            similarities = []
            for nid, neuron in self.neurons.items():
                if nid != n.id and neuron.cluster == 'hidden':
                    sim = cosine_similarity(embedding, neuron.embedding)
                    similarities.append((sim, nid))
            similarities.sort(reverse=True)
            for sim, nid in similarities[:15]:
                if sim > 0.2:
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
        if not force and len(self.neurons) < self.max_neurons * 0.9:
            return
        utility = {}
        for nid, n in self.neurons.items():
            if n.cluster in ('input', 'output') and n.label:
                utility[nid] = float('inf')
            else:
                score = n.energy * 0.3 + n.importance * 0.3 + n.utility_score * 0.3 + (n.usage_counter / (n.age+1)) * 0.1
                utility[nid] = score
        sorted_neurons = sorted(utility.items(), key=lambda x: x[1])
        target = int(self.max_neurons * 0.8)
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

    def _prune_synapses(self, force=False):
        if not force and len(self.synapses) < self.max_synapses * 0.9:
            to_remove = []
            for sid, syn in self.synapses.items():
                syn.decay(decay_rate=0.0005)
                if syn.is_dead(weight_threshold=self.synapse_weight_threshold,
                               max_age_seconds=self.synapse_max_age):
                    to_remove.append(sid)
            for sid in to_remove:
                self._remove_synapse(sid)
            return
        target = int(self.max_synapses * 0.7)
        scored = []
        for sid, syn in self.synapses.items():
            score = abs(syn.weight) * 0.4 + syn.frequency * 0.3 + syn.confidence * 0.2 + (syn.usage_count / (time.time()-syn.creation_time+1)) * 0.1
            scored.append((score, sid))
        scored.sort(key=lambda x: x[0])
        to_remove = []
        for score, sid in scored:
            if len(self.synapses) - len(to_remove) <= target:
                break
            to_remove.append(sid)
        for sid in to_remove:
            self._remove_synapse(sid)

    def _hebbian_update(self, activated_ids: List[int], query_vec: np.ndarray):
        for i in range(len(activated_ids)):
            for j in range(i+1, len(activated_ids)):
                a, b = activated_ids[i], activated_ids[j]
                syn_id = self.edge_index.get((a, b)) or self.edge_index.get((b, a))
                if syn_id is not None and syn_id in self.synapses:
                    syn = self.synapses[syn_id]
                    if not syn.is_inhibitory:
                        att = syn.attention_score(query_vec)
                        delta = 0.015 * (1.0 - min(1.0, syn.weight)) * (1.0 + att)
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

    def _create_new_synapses_from_coactivation(self):
        for pair, count in list(self.coactivation_counter.items()):
            if count >= self.synapse_creation_threshold:
                a, b = pair
                if a in self.neurons and b in self.neurons:
                    sim = cosine_similarity(self.neurons[a].embedding, self.neurons[b].embedding)
                    if sim > 0.2:
                        syn1 = self._create_synapse(a, b, weight=0.05)
                        syn2 = self._create_synapse(b, a, weight=0.05)
                        if syn1 and syn2:
                            self.coactivation_counter[pair] = 0

    def propagate_signal(self, input_signal: Signal, max_steps: int = 20) -> Tuple[List[int], Dict[int, float]]:
        self.working_memory.add(input_signal)
        self.working_memory.update_context(input_signal.embedding)

        dialog_ctx = self.dialog_memory.get_context_embedding()
        if dialog_ctx is not None:
            context_query = 0.7 * input_signal.embedding + 0.3 * dialog_ctx
            context_query /= np.linalg.norm(context_query) + 1e-8
        else:
            context_query = input_signal.embedding

        similarities = []
        for nid, neuron in self.neurons.items():
            if neuron.cluster == 'input':
                sim = cosine_similarity(neuron.embedding, input_signal.embedding)
                similarities.append((sim, nid))
        similarities.sort(reverse=True)
        start_neurons = [nid for sim, nid in similarities[:7] if sim > 0.1]

        if not start_neurons:
            existing = self._find_most_similar_neuron(input_signal.embedding, cluster='input', threshold=0.3)
            if existing is not None:
                start_neurons = [existing]
            else:
                new_nid = self._create_neuron(embedding=input_signal.embedding, cluster='input', layer=0)
                start_neurons = [new_nid]

        visited = set(start_neurons)
        queue = deque([(nid, 0) for nid in start_neurons])
        activated = []
        activation_map = {}

        while queue and len(visited) < 300:
            nid, steps = queue.popleft()
            if steps > max_steps:
                continue
            neuron = self.neurons[nid]

            outgoing = []
            for syn_id in neuron.outgoing_synapses:
                syn = self.synapses.get(syn_id)
                if syn is None or abs(syn.weight) < 0.005:
                    continue
                if syn.weight < 0:
                    continue
                target = self.neurons.get(syn.to_neuron)
                if target is None or target.energy < 0.15:
                    continue
                att = syn.attention_score(context_query)
                semantic_sim = cosine_similarity(syn.semantic_vector, context_query)
                score = syn.weight * (1.0 + att + semantic_sim * 0.5)
                outgoing.append((score, syn_id, syn, target))

            outgoing.sort(reverse=True)
            for score, syn_id, syn, target in outgoing[:8]:
                attention_boost = 1.0 + score
                neuron.receive_signal(input_signal, syn.weight, attention_boost)

                new_signal = copy.deepcopy(input_signal)
                new_signal.energy *= syn.weight * (0.8 + 0.2 * score)
                new_signal.source = nid
                new_signal.destination = syn.to_neuron
                new_signal.confidence = syn.confidence
                new_signal.history.append((nid, syn_id, syn.to_neuron))
                self.working_memory.add(new_signal)

                if syn.to_neuron not in visited:
                    visited.add(syn.to_neuron)
                    queue.append((syn.to_neuron, steps + 1))
                syn.update(delta_weight=0.0, delta_frequency=0.001)

            if neuron.activate(threshold=0.25):
                activated.append(nid)
                activation_map[nid] = neuron.activation

        if activated:
            self._hebbian_update(activated, context_query)
            for nid in activated:
                self.neurons[nid].importance = min(1.0, self.neurons[nid].importance + 0.01)

        return activated, activation_map

    def step(self, input_signal: Signal, target_neuron_id: Optional[int] = None,
             reward: float = 0.0) -> List[int]:
        self.step_counter += 1
        for neuron in self.neurons.values():
            neuron.decay_energy(rate=0.0005)

        activated, _ = self.propagate_signal(input_signal)

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
                        syn.update(delta_weight=reward * 0.005, delta_confidence=reward * 0.005)

        if self.step_counter % 10 == 0:
            self._create_new_synapses_from_coactivation()
        if self.step_counter % self.prune_every == 0:
            self._prune_synapses()
            self._prune_neurons()

        self.short_memory.add((input_signal, activated))
        self.global_time = time.time()
        return activated

    # ----- Текстовые методы с улучшениями -----
    def normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
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

    def retrieve_facts(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_emb = self.text_to_embedding(query_text)

        scored_kb = []
        for fact in self.knowledge_base:
            sim = cosine_similarity(query_emb, fact["emb"])
            scored_kb.append((sim, fact))
        scored_kb.sort(key=lambda x: x[0], reverse=True)

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
                        "weight": act_map.get(nid, 0.5)
                    })

        results = []
        for sim, fact in scored_kb[:top_k]:
            if sim > 0.3:
                results.append({"source": "memory", "q": fact["q"], "a": fact["a"], "score": sim})

        for f in network_facts[:top_k]:
            results.append({"source": "network", "q": f["q"], "a": f["a"], "score": f["weight"]})

        seen = set()
        unique = []
        for r in results:
            key = (r["q"], r["a"])
            if key not in seen:
                seen.add(key)
                unique.append(r)

        unique.sort(key=lambda x: x["score"], reverse=True)
        return unique[:top_k]

    def generate_answer(self, input_text: str, temperature: float = 0.7,
                        use_rag: bool = True) -> Dict[str, Any]:
        dialog_ctx = self.dialog_memory.get_context_string(n=3)
        facts = self.retrieve_facts(input_text, top_k=5) if use_rag else []

        system_msg = (
            "Ты - интеллектуальный ассистент с ассоциативной памятью. "
            "Отвечай естественно, по-человечески, на языке вопроса. "
            "Если есть релевантные факты из памяти - используй их. "
            "Если фактов нет - отвечай на основе общих знаний. "
            "Ответ должен быть связным и развернутым (1-3 предложения)."
            "Если ответ должен содержать код не ограничивайся напиши полностью развернутый ответ"
        )

        context_parts = []
        if dialog_ctx:
            context_parts.append("=== История диалога ===")
            context_parts.append(dialog_ctx)
        if facts:
            context_parts.append("=== Факты из памяти ===")
            for i, f in enumerate(facts, 1):
                context_parts.append(f"{i}. Вопрос: {f['q']} -> Ответ: {f['a']} (источник: {f['source']}, релевантность: {f['score']:.2f})")

        context_str = "\n".join(context_parts)

        user_prompt = input_text
        if context_str:
            user_prompt = f"{context_str}\n\nТекущий вопрос: {input_text}"

        try:
            response = client.chat.completions.create(
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
            print(f"[LLM Error] {e}")
            # УЛУЧШЕНИЕ: передаём temperature в fallback
            answer_text = self._fallback_chain_answer(input_text, temperature=temperature)

        return {
            "text": answer_text,
            "facts": facts,
            "known": len(facts) > 0,
            "fallback": len(facts) == 0,
        }

    # УЛУЧШЕНИЕ: добавлен параметр temperature
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
                score = syn.weight + syn.confidence * 0.1
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

    def get_or_create_concept_neuron(self, text: str) -> int:
        # УЛУЧШЕНИЕ: используем normalize_text для единообразия
        normalized = self.normalize_text(text)
        nid = self.concept_index.get(normalized)
        if nid is not None and nid in self.neurons:
            return nid

        label = text.strip()
        if len(label) > self.MAX_LABEL_LEN:
            label = label[:self.MAX_LABEL_LEN]

        emb = self.text_to_embedding(text)
        nid = self._create_neuron(embedding=emb, cluster='output',
                                   layer=self.num_hidden_layers + 1, label=label)
        self.concept_index[normalized] = nid
        print(f"[Новый узел] {nid}: '{label}'")
        return nid

    # -------------------- УЛУЧШЕННЫЙ learn_pair --------------------
    def learn_pair(self, input_text: str, output_text: str, reinforce_boost: float = 0.15, epochs: int = 1):
        """
        Обучает модель на паре (вопрос→ответ). При epochs > 1 повторяет обучение
        для закрепления связи (усиливает синапс).
        """
        for epoch in range(epochs):
            input_norm = self.normalize_text(input_text)
            output_norm = self.normalize_text(output_text)
            if input_norm == output_norm:
                print("[Пропуск] Вход и выход совпадают.")
                return

            input_id = self.get_or_create_concept_neuron(input_text)
            output_id = self.get_or_create_concept_neuron(output_text)

            sim = cosine_similarity(self.neurons[input_id].embedding, self.neurons[output_id].embedding)
            sim_bonus = max(0.0, sim) * 0.05

            # Создаём или получаем существующий синапс
            syn_id = self._create_synapse(input_id, output_id, weight=0.2 + sim_bonus)
            if syn_id is None:
                print("[Ошибка] Не удалось создать синапс (лимит).")
                return
            syn = self.synapses[syn_id]

            if syn.usage_count > 0:
                syn.update(delta_weight=reinforce_boost, delta_plasticity=0.02,
                           delta_confidence=0.05, delta_frequency=0.02)
            else:
                syn.update(delta_weight=0.0, delta_confidence=0.05)

            # Распространение сигнала и создание дополнительных связей (только на первой эпохе,
            # чтобы избежать дублирования)
            if epoch == 0:
                sig = self.text_to_signal(input_text)
                activated, _ = self.propagate_signal(sig, max_steps=15)
                if activated:
                    sorted_activated = sorted(activated, key=lambda nid: self.neurons[nid].importance, reverse=True)
                    for nid in sorted_activated[:3]:
                        neuron = self.neurons.get(nid)
                        if neuron is None or neuron.cluster == 'output':
                            continue
                        hsyn_id = self._create_synapse(nid, output_id, weight=0.05)
                        if hsyn_id is not None:
                            self.synapses[hsyn_id].update(delta_weight=0.03)

                self.step(sig, target_neuron_id=output_id)

            # Добавляем в knowledge_base только один раз (защита от дублирования уже есть)
            if epoch == 0:
                self._add_to_knowledge_base(input_text, output_text)

            print(
                f"[+] Обучение (эпоха {epoch + 1}/{epochs}): '{input_text}' -> '{output_text}' (вес: {syn.weight:.3f})")

        self._learn_counter += epochs
        if self._learn_counter % self.AUTO_SAVE_EVERY == 0:
            self.save(self._model_path)
            print(f"[Автосохранение] {self._model_path}")

    # УЛУЧШЕНИЕ: предотвращение дублирования фактов
    def _add_to_knowledge_base(self, q: str, a: str):
        # Проверяем, нет ли уже такой пары
        for item in self.knowledge_base:
            if item["q"] == q and item["a"] == a:
                return
        emb = self.text_to_embedding(q + " " + a)
        self.knowledge_base.append({"q": q, "a": a, "emb": emb, "time": time.time()})
        if len(self.knowledge_base) > self.max_kb_size:
            self.knowledge_base.pop(0)

    def learn_negative_pair(self, input_text: str, output_text: str, penalty: float = 0.15):
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
            print(f"[-] Отриц. обучение: '{input_text}' -> '{output_text}' = {syn.weight:.3f}")
        else:
            syn_id = self._create_synapse(input_id, output_id,
                                          weight=-random.uniform(0.1, 0.3), is_inhibitory=True)
            if syn_id is not None:
                print(f"[-] Тормозной синапс: '{input_text}' -> '{output_text}'")
            else:
                print("[Ошибка] Не удалось создать тормозной синапс (лимит).")

        self._learn_counter += 1
        if self._learn_counter % self.AUTO_SAVE_EVERY == 0:
            self.save(self._model_path)

    # -------------------- НОВЫЙ МЕТОД: forget_concept --------------------
    def forget_concept(self, text: str) -> bool:
        """
        Удаляет понятие из сети и из knowledge_base.
        Возвращает True, если понятие было найдено и удалено.
        """
        normalized = self.normalize_text(text)
        nid = self.concept_index.get(normalized)
        if nid is None or nid not in self.neurons:
            return False

        neuron = self.neurons[nid]
        # Удаляем все синапсы
        for syn_id in list(neuron.incoming_synapses) + list(neuron.outgoing_synapses):
            self._remove_synapse(syn_id)
        # Удаляем из concept_index
        del self.concept_index[normalized]
        # Удаляем сам нейрон
        del self.neurons[nid]

        # Удаляем из knowledge_base записи, где вопрос или ответ содержат этот текст (частичное совпадение)
        new_kb = []
        for item in self.knowledge_base:
            if normalized not in self.normalize_text(item["q"]) and normalized not in self.normalize_text(item["a"]):
                new_kb.append(item)
        self.knowledge_base = new_kb

        print(f"[Забыто] Понятие '{text}' удалено из сети и из KB.")
        return True

    # -------------------- Остальные методы без изменений --------------------
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
            links.append((target.label, syn.weight, syn.usage_count, syn.confidence, syn.is_inhibitory))
        links.sort(key=lambda x: x[1], reverse=True)
        if not links:
            print(f"У '{text}' нет исходящих связей.")
            return
        print(f"Связи '{text}':")
        for label, weight, count, conf, inhib in links[:top_k]:
            inhib_str = " (тормозная)" if inhib else ""
            print(f"  -> {label}   (вес={weight:.3f}, повторов={count}, уверенность={conf:.2f}){inhib_str}")

    def sleep(self, duration_steps: int = 10):
        print("=== Сон ===")
        for _ in range(duration_steps):
            to_remove = []
            for nid, neuron in self.neurons.items():
                if neuron.energy < 0.05 and neuron.importance < 0.1 and neuron.cluster not in ('output', 'input'):
                    to_remove.append(nid)
            for nid in to_remove:
                neuron = self.neurons[nid]
                for syn_id in list(neuron.incoming_synapses) + list(neuron.outgoing_synapses):
                    self._remove_synapse(syn_id)
                del self.neurons[nid]
                print(f"  Удален шумовой нейрон {nid}")

            for syn in self.synapses.values():
                if syn.frequency > 0.5:
                    syn.update(delta_weight=0.01, delta_confidence=0.01)
                if syn.reward > 0.5:
                    syn.update(delta_weight=0.02, delta_confidence=0.02)
                syn.reward *= 0.9

        for item in self.short_memory.get_recent(50):
            self.long_memory.add(item)
        self.short_memory.clear()
        print("Сон завершен.")
        self.save(self._model_path)

    def save(self, filename: str = None):
        filename = filename or self._model_path
        data = {
            "dim": self.dim,
            "step_counter": self.step_counter,
            "coactivation_counter": {str(k): v for k, v in self.coactivation_counter.items()},
            "neurons": [],
            "synapses": [],
            "knowledge_base": [],
        }
        for nid, n in self.neurons.items():
            data["neurons"].append({
                "id": nid, "embedding": n.embedding.tolist(), "cluster": n.cluster,
                "layer": n.layer, "label": n.label, "importance": n.importance, "energy": n.energy,
                "utility_score": n.utility_score,
            })
        for sid, s in self.synapses.items():
            data["synapses"].append({
                "id": sid, "from": s.from_neuron, "to": s.to_neuron,
                "weight": s.weight, "plasticity": s.plasticity, "confidence": s.confidence,
                "frequency": s.frequency, "energy": s.energy, "reward": s.reward,
                "usage_count": s.usage_count, "is_inhibitory": s.is_inhibitory,
                "context_vector": s.context_vector.tolist(),
                "semantic_vector": s.semantic_vector.tolist(),
                "episodic_vector": s.episodic_vector.tolist(),
                "key_vectors": [kv.tolist() for kv in s.key_vectors],
                "value_vectors": [vv.tolist() for vv in s.value_vectors],
            })
        for kb in self.knowledge_base:
            data["knowledge_base"].append({
                "q": kb["q"], "a": kb["a"], "emb": kb["emb"].tolist(), "time": kb["time"]
            })

        tmp = filename + ".tmp"
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, filename)
        print(f"[Сохранено] {filename}")

    def load(self, filename: str = None):
        filename = filename or self._model_path
        if not os.path.exists(filename):
            print(f"Файл {filename} не найден, создаем новую модель.")
            return
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Ошибка загрузки: {e}")
            return

        self.dim = data["dim"]
        self.step_counter = data["step_counter"]
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

        for nd in data["neurons"]:
            n = Neuron(embedding=np.array(nd["embedding"]), cluster=nd["cluster"],
                       layer=nd.get("layer", 0), importance=nd["importance"], energy=nd["energy"])
            n.id = nd["id"]
            n.label = nd["label"]
            n.utility_score = nd.get("utility_score", 0.0)
            self.neurons[n.id] = n
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
                is_inhibitory=sd.get("is_inhibitory", False)
            )
            s.id = sd["id"]
            s.reward = sd["reward"]
            s.usage_count = sd.get("usage_count", 0)
            if "key_vectors" in sd:
                s.key_vectors = [np.array(kv) for kv in sd["key_vectors"]]
            if "value_vectors" in sd:
                s.value_vectors = [np.array(vv) for vv in sd["value_vectors"]]
            self.synapses[s.id] = s
            self.edge_index[(sd["from"], sd["to"])] = s.id
            if sd["from"] in self.neurons:
                self.neurons[sd["from"]].add_outgoing(s.id)
            if sd["to"] in self.neurons:
                self.neurons[sd["to"]].add_incoming(s.id)

        for kb in data.get("knowledge_base", []):
            self.knowledge_base.append({
                "q": kb["q"], "a": kb["a"], "emb": np.array(kb["emb"]), "time": kb["time"]
            })

        print(f"[Загружено] {filename} - нейронов: {len(self.neurons)}, синапсов: {len(self.synapses)}, "
              f"понятий: {len(self.concept_index)}, фактов: {len(self.knowledge_base)}")


    # Внутри класса Brain:
    def generate_context_question(self, user_input: str, answer: str,
                                  history: List[Dict], facts: List[Dict]) -> Optional[str]:
        """
        Генерирует уточняющий/обобщающий вопрос на основе контекста.
        Возвращает строку вопроса или None, если вопрос не нужен.
        """
        # Если фактов достаточно – можно пропустить
        if len(facts) >= 3:
            return None

        # Формируем историю для промпта
        history_text = ""
        for turn in history[-3:]:
            history_text += f"Пользователь: {turn['user']}\nАссистент: {turn['assistant']}\n"

        system_prompt = (
            "Ты – ассистент, который помогает пользователю уточнить или расширить его запрос. "
            "На основе истории диалога и только что данного ответа, сформулируй один уточняющий, "
            "разъясняющий или обобщающий вопрос, который поможет лучше понять, что именно нужно пользователю. "
            "Вопрос должен быть естественным, вежливым, на русском языке. "
            "Если уточнение не требуется – ответь 'НЕТ'."
        )
        user_prompt = (
            f"История диалога (последние сообщения):\n{history_text}\n\n"
            f"Вопрос пользователя: {user_input}\n"
            f"Ответ ассистента: {answer}\n"
            f"Найденные факты: {[f['q'] + ' -> ' + f['a'] for f in facts]}\n\n"
            f"Уточняющий вопрос:"
        )

        try:
            # Используем тот же llm_client, который уже есть в brain (или создаём, если нет)
            # Важно: в brain должен быть self.llm_client, иначе добавить.
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
            print(f"[ContextQ] Ошибка: {e}")
            return None

    def get_weak_concepts(self, threshold: float = 0.2, limit: int = 5) -> List[str]:
        """Возвращает список понятий с низкой связностью (слабые знания)."""
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
        """
        Анализирует историю диалога и предлагает темы для продолжения.
        """
        if not self.dialog_memory.items:
            return []
        # Берём последние 3 вопроса пользователя
        last_questions = [turn.get("user", "") for turn in self.dialog_memory.items[-3:] if turn.get("user")]
        if not last_questions:
            return []
        combined = " ".join(last_questions)
        # Используем LLM для извлечения ключевых тем
        prompt = f"На основе следующих вопросов пользователя: '{combined}', предложи {limit} кратких тем для продолжения диалога (каждая тема одной фразой или словом, разделённых запятой)."
        try:
            resp = client.chat.completions.create(
                model="local-model",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.7
            )
            raw = resp.choices[0].message.content.strip()
            topics = [t.strip() for t in raw.split(',') if t.strip()]
            return topics[:limit]
        except Exception as e:
            print(f"[SuggestTopics] Ошибка: {e}")
            return []

    def get_recent_user_topics(self, limit: int = 3) -> List[str]:
        """
        Извлекает ключевые темы из последних сообщений пользователя.
        Возвращает список существительных/фраз (до limit).
        """
        if not self.dialog_memory.items:
            return []
        user_messages = []
        for turn in self.dialog_memory.items[-10:]:
            if "user" in turn and turn["user"]:
                user_messages.append(turn["user"])
        if not user_messages:
            return []
        combined = " ".join(user_messages)
        # Используем LLM для извлечения ключевых тем (существительные, фразы)
        prompt = f"Из текста: '{combined}' выдели {limit} главных тем (существительные или короткие фразы), разделённых запятой. Только темы, без пояснений."
        try:
            resp = client.chat.completions.create(
                model="local-model",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=30,
                temperature=0.3,
            )
            raw = resp.choices[0].message.content.strip()
            topics = [t.strip() for t in raw.split(',') if t.strip()]
            return topics[:limit]
        except Exception as e:
            print(f"[Topics] Ошибка: {e}")
            # fallback: просто берём слова из последнего сообщения
            last_msg = user_messages[-1] if user_messages else ""
            words = [w for w in last_msg.split() if len(w) > 3]
            return words[:limit]

    def get_uncertain_concepts(self, threshold: float = 0.3, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Находит понятия (выходные нейроны), у которых низкая уверенность или слабые связи.
        Возвращает список словарей: {'label': str, 'score': float}
        """
        uncertain = []
        for nid, neuron in self.neurons.items():
            if neuron.cluster != 'output' or not neuron.label:
                continue
            # Вычисляем средний вес входящих синапсов (чем меньше, тем слабее понятие)
            in_weights = []
            for syn_id in neuron.incoming_synapses:
                syn = self.synapses.get(syn_id)
                if syn:
                    in_weights.append(abs(syn.weight))
            avg_in = sum(in_weights) / len(in_weights) if in_weights else 0.0

            # Считаем количество исходящих связей
            out_count = len(neuron.outgoing_synapses)

            # Если мало входящих весов или мало исходящих связей – неопределённость
            if avg_in < threshold or out_count < 2:
                # Также учитываем частоту активации (usage_counter)
                activation_freq = neuron.usage_counter / (neuron.age + 1) if neuron.age > 0 else 0
                score = (1.0 - avg_in) * 0.5 + (1.0 - min(1.0, out_count / 5)) * 0.3 + (1.0 - activation_freq) * 0.2
                uncertain.append({
                    "label": neuron.label,
                    "score": score,
                    "avg_in": avg_in,
                    "out_count": out_count,
                    "activation_freq": activation_freq
                })
        uncertain.sort(key=lambda x: x["score"], reverse=True)
        return uncertain[:limit]

    def generate_contextual_question(self, user_topics: List[str], uncertain_concepts: List[Dict]) -> Optional[str]:
        """
        Генерирует вопрос, связывая темы пользователя и неопределённые понятия.
        """
        if not user_topics and not uncertain_concepts:
            return None

        # Строим промпт для LLM
        topics_str = ", ".join(user_topics) if user_topics else "общая тема"
        weak_str = ", ".join([c["label"] for c in uncertain_concepts[:3]]) if uncertain_concepts else "нет явных слабых мест"

        system_prompt = (
            "Ты — ассистент, который помогает пользователю углубиться в тему, но при этом "
            "не перегружает его сложными вопросами. На основе интересов пользователя и "
            "некоторых пробелов в твоих знаниях, сформулируй один естественный, уточняющий "
            "или развивающий вопрос. Вопрос должен быть простым, понятным и не требовать "
            "специальных знаний. Если удастся, свяжи тему пользователя с чем-то, что тебе "
            "не совсем ясно. Ответь только вопросом, без пояснений."
        )
        user_prompt = (
            f"Темы, которые интересовали пользователя: {topics_str}.\n"
            f"Понятия, которые я знаю слабо: {weak_str}.\n"
            f"Сформулируй один вопрос, который поможет мне лучше понять эту тему, "
            f"а также прояснить слабые места. Вопрос должен быть интересным и не слишком сложным."
        )

        try:
            resp = client.chat.completions.create(
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
            print(f"[ContextualQ] Ошибка: {e}")
            return None

    def save_dialog_history(self, filename: str = "dialog_history.json"):
        """Сохраняет историю диалога в файл."""
        data = []
        for item in self.dialog_memory.items:
            # Копируем, чтобы не было проблем с numpy
            data.append({"user": item.get("user", ""), "assistant": item.get("assistant", ""), "time": item.get("time", 0)})
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_dialog_history(self, filename: str = "dialog_history.json"):
        """Загружает историю диалога из файла."""
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for item in data:
                    # Добавляем в память (DialogMemory.add ожидает словарь)
                    self.dialog_memory.add(item)
            except Exception as e:
                print(f"Ошибка загрузки истории: {e}")
# ============================
# Класс Teacher (без изменений)
# ============================
class Teacher:
    def __init__(self):
        self.system_prompt = (
            "Ты - критический эксперт. Оцени ответ ассистента на вопрос пользователя.\n"
            "Оценка: число от 0 до 1 (0 - полный бред, 1 - идеально).\n"
            "Если ответ можно улучшить, предложи краткий улучшенный вариант (1 фраза).\n"
            "Формат: <оценка>|<улучшенный_ответ>\n"
            "Пример: 0.8|Солнце - звезда, а не планета."
        )

    def evaluate(self, question: str, brain_answer: str) -> Tuple[float, str]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Вопрос: {question}\nОтвет: {brain_answer}\nОценка:"}
        ]
        try:
            response = client.chat.completions.create(
                model="local-model", messages=messages, max_tokens=100, temperature=0.2,
            )
            raw = response.choices[0].message.content.strip()
            score = 0.5
            improved = brain_answer
            match = re.search(r'(\d+\.?\d*)', raw)
            if match:
                score = float(match.group(1))
                rest = raw.replace(match.group(1), '').strip()
                if rest.startswith('|'):
                    rest = rest[1:].strip()
                if rest and len(rest) > 2:
                    improved = rest
            score = max(0.0, min(1.0, score))
            return score, improved
        except Exception as e:
            print(f"[Ошибка учителя] {e}")
            return 0.5, brain_answer


# ============================
# Интерактивный режим (с использованием нового метода forget)
# ============================
LEARN_PATTERN = re.compile(r'^learn\s+(.+?)\s*=>\s*(.+)$', re.IGNORECASE)
NEG_PATTERN = re.compile(r'^neg\s+(.+?)\s*=>\s*(.+)$', re.IGNORECASE)
LINKS_PATTERN = re.compile(r'^links\s+(.+)$', re.IGNORECASE)
FORGET_PATTERN = re.compile(r'^forget\s+(.+)$', re.IGNORECASE)  # Новая команда

def interactive_with_teacher(brain: Brain):
    teacher = Teacher()
    print("\n=== Smart Brain v4 - Ассоциативная память + LLM (улучшенная) ===")
    print("Команды:")
    print("  learn <вопрос> => <ответ>  - обучить")
    print("  neg <вопрос> => <ответ>    - отрицательное обучение")
    print("  links <понятие>            - показать связи")
    print("  forget <понятие>           - забыть понятие (удалить из сети и KB)")
    print("  stats / save / sleep / exit")
    print("  (любой текст - вопрос к Brain)\n")

    try:
        while True:
            user_input = input("\n> ").strip()
            if not user_input:
                continue

            lower = user_input.lower()

            if lower == 'exit':
                brain.save()
                print("Выход.")
                break
            if lower == 'stats':
                ic = sum(1 for n in brain.neurons.values() if n.cluster == 'input')
                oc = sum(1 for n in brain.neurons.values() if n.cluster == 'output')
                hc = sum(1 for n in brain.neurons.values() if n.cluster == 'hidden')
                print(f"Нейронов: {len(brain.neurons)} (in:{ic}, out:{oc}, hid:{hc})")
                print(f"Синапсов: {len(brain.synapses)}")
                print(f"Понятий: {len(brain.concept_index)}")
                print(f"Фактов в KB: {len(brain.knowledge_base)}")
                print(f"Лимиты: нейроны {brain.max_neurons}, синапсы {brain.max_synapses}")
                continue
            if lower == 'save':
                brain.save()
                continue
            if lower == 'sleep':
                brain.sleep(duration_steps=5)
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
                    print(f"Понятие '{concept}' забыто.")
                else:
                    print(f"Понятие '{concept}' не найдено.")
                continue

            # Основной диалог
            temp = random.uniform(0.5, 1.0)
            result = brain.generate_answer(user_input, temperature=temp, use_rag=True)
            brain_answer = result['text']
            print(f"\n[Brain] {brain_answer}")
            if result['facts']:
                print(f"[Использовано фактов: {len(result['facts'])}]")

            score, improved = teacher.evaluate(user_input, brain_answer)
            print(f"[Учитель] Оценка: {score:.2f}", end="")
            if improved != brain_answer:
                print(f", улучшение: {improved}")
            else:
                print(" (ответ хорош)")

            if score >= 0.75:
                target = improved if improved != brain_answer else brain_answer
                if target.lower() != user_input.lower():
                    brain.learn_pair(user_input, target)
                    brain.dialog_memory.add_turn(user_input, target, brain.text_to_embedding(user_input))
            elif score <= 0.3:
                if improved and improved != brain_answer and improved.lower() != user_input.lower():
                    brain.learn_negative_pair(user_input, brain_answer)
                    brain.learn_pair(user_input, improved)
                    brain.dialog_memory.add_turn(user_input, improved, brain.text_to_embedding(user_input))
                else:
                    brain.learn_negative_pair(user_input, brain_answer)
            else:
                brain.dialog_memory.add_turn(user_input, brain_answer, brain.text_to_embedding(user_input))
                print("[Нейтрально] Обучение пропущено, диалог сохранен.")

    except (KeyboardInterrupt, EOFError):
        print("\nПрерывание - сохраняю...")
    finally:
        brain.save()
        print("Готово.")

# ============================
# Запуск
# ============================
if __name__ == "__main__":
    MODEL_PATH = "brain_model_trained.json"
    brain = Brain(dim_embedding=128, input_neurons=40, output_neurons=40,
                  hidden_layers=[100, 80, 60], model_path=MODEL_PATH,
                  max_neurons=800, max_synapses=8000)
    brain.load(MODEL_PATH)
    interactive_with_teacher(brain)