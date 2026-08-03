import numpy as np
import random
import time
import json
import os
import re
import zlib
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple, Optional, Any
import copy
from openai import OpenAI

# ============================
# Настройка подключения к LM Studio
# ============================
client = OpenAI(
    base_url="http://192.168.0.13:1234/v1",
    api_key="not-needed",  # LM Studio не требует ключа
)

# ============================
# Вспомогательные функции
# ============================
def random_vector(dim: int = 64) -> np.ndarray:
    return np.random.randn(dim) / np.sqrt(dim)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

# ============================
# Класс Signal
# ============================
class Signal:
    def __init__(self, embedding: np.ndarray, energy: float = 1.0, importance: float = 0.5,
                 context: Optional[np.ndarray] = None, source=None, destination=None,
                 confidence: float = 0.5):
        self.id = id(self)
        self.embedding = embedding.copy()
        self.energy = energy
        self.importance = importance
        self.timestamp = time.time()
        self.context = context.copy() if context is not None else None
        self.source = source
        self.destination = destination
        self.confidence = confidence
        self.history = []

    def __repr__(self):
        return f"Signal(id={self.id}, source={self.source}, dest={self.destination}, energy={self.energy:.2f})"

# ============================
# Класс Synapse (с поддержкой отрицательных весов)
# ============================
class Synapse:
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
        return f"Synapse(id={self.id}, {self.from_neuron}->{self.to_neuron}, w={self.weight:.3f}, inhib={self.is_inhibitory})"

# ============================
# Класс Neuron
# ============================
class Neuron:
    def __init__(self, embedding: Optional[np.ndarray] = None, activation: float = 0.0,
                 potential: float = 0.0, energy: float = 1.0, importance: float = 0.5,
                 cluster: str = 'hidden'):
        self.id = id(self)
        self.embedding = embedding if embedding is not None else random_vector()
        self.activation = activation
        self.potential = potential
        self.energy = energy
        self.importance = importance
        self.cluster = cluster
        self.last_activation = time.time()
        self.age = 0
        self.usage_counter = 0
        self.memory_links = []
        self.incoming_synapses = []
        self.outgoing_synapses = []
        self.temporary_state = {}
        self.persistent_state = {}
        self.label = None

    def add_incoming(self, synapse_id):
        if synapse_id not in self.incoming_synapses:
            self.incoming_synapses.append(synapse_id)

    def add_outgoing(self, synapse_id):
        if synapse_id not in self.outgoing_synapses:
            self.outgoing_synapses.append(synapse_id)

    def receive_signal(self, signal: Signal, synapse_weight: float):
        # отрицательный вес уменьшает потенциал
        self.potential += signal.energy * synapse_weight * signal.importance
        sim = cosine_similarity(self.embedding, signal.embedding)
        self.potential += sim * 0.1

    def activate(self, threshold: float = 0.5):
        if self.potential > threshold:
            self.activation = sigmoid(self.potential - threshold)
            self.last_activation = time.time()
            self.usage_counter += 1
            self.age += 1
            self.energy = max(0.0, self.energy - 0.01)
            self.potential *= 0.5
            return True
        else:
            self.activation = 0.0
            return False

    def decay_energy(self, rate: float = 0.001):
        self.energy = max(0.0, self.energy - rate)
        if self.energy < 0.1:
            self.potential = 0.0

    def __repr__(self):
        return f"Neuron(id={self.id}, cluster={self.cluster}, act={self.activation:.2f}, pot={self.potential:.2f})"

# ============================
# Классы памяти
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
        super().__init__(max_size=20)

class ShortMemory(Memory):
    def __init__(self):
        super().__init__(max_size=200)

class LongMemory(Memory):
    def __init__(self):
        super().__init__(max_size=1000000)

# ============================
# Класс Brain (доработанный)
# ============================
class Brain:
    def __init__(self, dim_embedding: int = 64, input_neurons: int = 30, output_neurons: int = 30,
                 hidden_neurons: int = 140, model_path: str = "brain_model.json"):
        self.dim = dim_embedding
        self.neurons: Dict[int, Neuron] = {}
        self.synapses: Dict[int, Synapse] = {}
        self.edge_index: Dict[Tuple[int, int], int] = {}
        self.concept_index: Dict[str, int] = {}

        self.working_memory = WorkingMemory()
        self.short_memory = ShortMemory()
        self.long_memory = LongMemory()
        self.step_counter = 0
        self.global_time = time.time()

        self.coactivation_counter = defaultdict(int)
        self.coactivation_window = []
        self.window_size = 200
        self.synapse_creation_threshold = 3
        self.synapse_max_age = 60 * 10
        self.synapse_weight_threshold = 0.01

        self.MAX_LABEL_LEN = 200
        self.AUTO_SAVE_EVERY = 5
        self._learn_counter = 0
        self._model_path = model_path

        # Инициализация нейронов
        for _ in range(input_neurons):
            n = Neuron(embedding=random_vector(self.dim), cluster='input')
            self.neurons[n.id] = n

        for _ in range(hidden_neurons):
            n = Neuron(embedding=random_vector(self.dim), cluster='hidden')
            self.neurons[n.id] = n

        for _ in range(output_neurons):
            n = Neuron(embedding=random_vector(self.dim), cluster='output')
            self.neurons[n.id] = n

        input_ids = [nid for nid, n in self.neurons.items() if n.cluster == 'input']
        hidden_ids = [nid for nid, n in self.neurons.items() if n.cluster == 'hidden']
        output_ids = [nid for nid, n in self.neurons.items() if n.cluster == 'output']

        for in_id in input_ids:
            for h_id in hidden_ids:
                self._create_synapse(in_id, h_id, weight=random.uniform(0.01, 0.1))

        for i, h1 in enumerate(hidden_ids):
            for h2 in hidden_ids[i+1:]:
                if random.random() < 0.2:
                    self._create_synapse(h1, h2, weight=random.uniform(0.01, 0.05))
                    self._create_synapse(h2, h1, weight=random.uniform(0.01, 0.05))

        for h_id in hidden_ids:
            for o_id in output_ids:
                self._create_synapse(h_id, o_id, weight=random.uniform(0.01, 0.1))

        for _ in range(min(20, len(input_ids)*len(output_ids)//2)):
            a = random.choice(input_ids)
            b = random.choice(output_ids)
            self._create_synapse(a, b, weight=random.uniform(0.005, 0.02))

    # -------------------- Управление синапсами/нейронами --------------------
    def _create_synapse(self, from_id: int, to_id: int, weight: float = None,
                        context_vec: np.ndarray = None, is_inhibitory: bool = False) -> Optional[int]:
        if from_id not in self.neurons or to_id not in self.neurons:
            return None
        key = (from_id, to_id)
        existing_id = self.edge_index.get(key)
        if existing_id is not None and existing_id in self.synapses:
            return existing_id
        if weight is None:
            weight = random.uniform(0.01, 0.1)
        if context_vec is None:
            context_vec = random_vector(self.dim)
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

    def _create_neuron(self, embedding: np.ndarray = None, importance: float = 0.5,
                       cluster: str = 'hidden', label: str = None) -> int:
        if embedding is None:
            embedding = random_vector(self.dim)
        n = Neuron(embedding=embedding, importance=importance, cluster=cluster)
        if label:
            n.label = label[:self.MAX_LABEL_LEN] if len(label) > self.MAX_LABEL_LEN else label
        self.neurons[n.id] = n

        if cluster == 'hidden' and len(self.neurons) > 1:
            similarities = []
            for nid, neuron in self.neurons.items():
                if nid != n.id and neuron.cluster == 'hidden':
                    sim = cosine_similarity(embedding, neuron.embedding)
                    similarities.append((sim, nid))
            similarities.sort(reverse=True)
            for sim, nid in similarities[:10]:
                if sim > 0.3:
                    self._create_synapse(n.id, nid, weight=0.05)
                    self._create_synapse(nid, n.id, weight=0.05)
        elif cluster == 'output':
            hidden_ids = [nid for nid, n in self.neurons.items() if n.cluster == 'hidden']
            for hid in hidden_ids:
                self._create_synapse(hid, n.id, weight=random.uniform(0.01, 0.1))
        elif cluster == 'input':
            hidden_ids = [nid for nid, n in self.neurons.items() if n.cluster == 'hidden']
            for hid in hidden_ids:
                self._create_synapse(n.id, hid, weight=random.uniform(0.01, 0.1))
        return n.id

    # -------------------- Хебб и создание синапсов --------------------
    def _hebbian_update(self, activated_ids: List[int]):
        for i in range(len(activated_ids)):
            for j in range(i+1, len(activated_ids)):
                a, b = activated_ids[i], activated_ids[j]
                syn_id = self.edge_index.get((a, b)) or self.edge_index.get((b, a))
                if syn_id is not None and syn_id in self.synapses:
                    syn = self.synapses[syn_id]
                    if not syn.is_inhibitory:
                        delta = 0.01 * (1.0 - min(1.0, syn.weight))
                        syn.update(delta_weight=delta, delta_plasticity=0.001, delta_confidence=0.001, delta_frequency=0.001)
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
                    if sim > 0.25:
                        syn1 = self._create_synapse(a, b, weight=0.05)
                        syn2 = self._create_synapse(b, a, weight=0.05)
                        if syn1 and syn2:
                            self.coactivation_counter[pair] = 0
                            print(f"Создан новый синапс между {a} и {b}")

    def _prune_synapses(self):
        to_remove = []
        for sid, syn in self.synapses.items():
            syn.decay(decay_rate=0.001)
            if syn.is_dead(weight_threshold=self.synapse_weight_threshold,
                           max_age_seconds=self.synapse_max_age):
                to_remove.append(sid)
        for sid in to_remove:
            self._remove_synapse(sid)

    # -------------------- Распространение сигнала (исправлено) --------------------
    def propagate_signal(self, input_signal: Signal, max_steps: int = 15) -> List[int]:
        self.working_memory.add(input_signal)

        similarities = []
        for nid, neuron in self.neurons.items():
            if neuron.cluster == 'input':
                sim = cosine_similarity(neuron.embedding, input_signal.embedding)
                similarities.append((sim, nid))
        similarities.sort(reverse=True)
        start_neurons = [nid for sim, nid in similarities[:5] if sim > 0.2]

        if not start_neurons:
            new_nid = self._create_neuron(embedding=input_signal.embedding, cluster='input')
            start_neurons = [new_nid]
            print(f"Создан новый входной нейрон {new_nid}")

        visited = set(start_neurons)
        queue = deque([(nid, 0) for nid in start_neurons])
        activated = []

        while queue and len(visited) < 200:
            nid, steps = queue.popleft()
            if steps > max_steps:
                continue
            neuron = self.neurons[nid]
            neuron.receive_signal(input_signal, 1.0)
            if neuron.activate(threshold=0.3):
                activated.append(nid)
                for syn_id in neuron.outgoing_synapses:
                    syn = self.synapses.get(syn_id)
                    if syn is None or abs(syn.weight) < 0.01:
                        continue
                    # ИСПРАВЛЕНИЕ: тормозные синапсы не создают новые сигналы
                    if syn.weight < 0:
                        continue
                    target = self.neurons.get(syn.to_neuron)
                    if target is None or target.energy < 0.2:
                        continue
                    new_signal = copy.deepcopy(input_signal)
                    new_signal.energy *= syn.weight  # теперь только положительные
                    new_signal.source = nid
                    new_signal.destination = syn.to_neuron
                    new_signal.confidence = syn.confidence
                    new_signal.history.append((nid, syn_id, syn.to_neuron))
                    self.working_memory.add(new_signal)
                    if syn.to_neuron not in visited:
                        visited.add(syn.to_neuron)
                        queue.append((syn.to_neuron, steps + 1))
                    syn.update(delta_weight=0.0, delta_frequency=0.001)

        if activated:
            self._hebbian_update(activated)
            for nid in activated:
                self.neurons[nid].importance = min(1.0, self.neurons[nid].importance + 0.01)

        return activated

    # -------------------- Шаг обучения --------------------
    def step(self, input_signal: Signal, target_neuron_id: Optional[int] = None,
             reward: float = 0.0) -> List[int]:
        self.step_counter += 1
        for neuron in self.neurons.values():
            neuron.decay_energy(rate=0.001)

        activated = self.propagate_signal(input_signal)

        if target_neuron_id is not None:
            if target_neuron_id in activated:
                reward = 1.0
            else:
                reward = 0.0

        if reward > 0:
            for nid in activated:
                neuron = self.neurons.get(nid)
                if neuron is None:
                    continue
                neuron.importance += reward * 0.02
                for syn_id in neuron.incoming_synapses:
                    syn = self.synapses.get(syn_id)
                    if syn and not syn.is_inhibitory:
                        syn.reward += reward
                        syn.update(delta_weight=reward * 0.005, delta_confidence=reward * 0.005)
                for syn_id in neuron.outgoing_synapses:
                    syn = self.synapses.get(syn_id)
                    if syn and not syn.is_inhibitory:
                        syn.reward += reward
                        syn.update(delta_weight=reward * 0.005, delta_confidence=reward * 0.005)

        if self.step_counter % 10 == 0:
            self._create_new_synapses_from_coactivation()
        if self.step_counter % 20 == 0:
            self._prune_synapses()

        self.short_memory.add((input_signal, activated))
        self.global_time = time.time()
        return activated

    # -------------------- Текст <-> вектор --------------------
    def normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        return text

    def text_to_embedding(self, text: str) -> np.ndarray:
        text = self.normalize_text(text)
        words = text.split()
        if not words:
            return random_vector(self.dim)
        vec = np.zeros(self.dim)
        for word in words:
            h = zlib.adler32(word.encode("utf-8"))
            rng = np.random.RandomState(h)
            word_vec = rng.randn(self.dim)
            vec += word_vec
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        else:
            vec = random_vector(self.dim)
        return vec

    def text_to_signal(self, text: str, energy: float = 1.0) -> Signal:
        vec = self.text_to_embedding(text)
        return Signal(embedding=vec, energy=energy, importance=0.8, context=vec)

    # -------------------- Понятийный граф --------------------
    def get_or_create_concept_neuron(self, text: str) -> int:
        normalized = text.strip().lower()
        nid = self.concept_index.get(normalized)
        if nid is not None and nid in self.neurons:
            return nid

        label = text.strip()
        if len(label) > self.MAX_LABEL_LEN:
            label = label[:self.MAX_LABEL_LEN]

        emb = self.text_to_embedding(text)
        nid = self._create_neuron(embedding=emb, cluster='output', label=label)
        self.concept_index[normalized] = nid
        print(f"Создан новый узел-понятие {nid}: «{label}»")
        return nid

    def get_or_create_output_neuron(self, word: str) -> int:
        return self.get_or_create_concept_neuron(word)

    # -------------------- Обучение с положительным и отрицательным подкреплением --------------------
    def learn_pair(self, input_text: str, output_text: str, reinforce_boost: float = 0.12):
        """Положительное обучение – укрепляем связь."""
        input_norm = input_text.strip().lower()
        output_norm = output_text.strip().lower()
        if input_norm == output_norm:
            print("Пропускаем обучение: вход и выход совпадают.")
            return

        input_id = self.get_or_create_concept_neuron(input_text)
        output_id = self.get_or_create_concept_neuron(output_text)

        sim = cosine_similarity(self.neurons[input_id].embedding, self.neurons[output_id].embedding)
        sim_bonus = max(0.0, sim) * 0.05

        syn_id = self._create_synapse(input_id, output_id, weight=0.15 + sim_bonus)
        syn = self.synapses[syn_id]

        if syn.usage_count > 0:
            syn.update(delta_weight=reinforce_boost, delta_plasticity=0.02,
                       delta_confidence=0.05, delta_frequency=0.02)
        else:
            syn.update(delta_weight=0.0, delta_confidence=0.05)

        sig = self.text_to_signal(input_text)
        activated = self.propagate_signal(sig)
        for nid in activated:
            neuron = self.neurons.get(nid)
            if neuron is None or neuron.cluster == 'output':
                continue
            hsyn_id = self._create_synapse(nid, output_id, weight=0.05)
            if hsyn_id is not None:
                self.synapses[hsyn_id].update(delta_weight=0.03)

        self.step(sig, target_neuron_id=output_id)

        print(f"[+] Положительное обучение: «{input_text}» → «{output_text}» (вес: {syn.weight:.3f}, повторов: {syn.usage_count})")

        self._learn_counter += 1
        if self._learn_counter % self.AUTO_SAVE_EVERY == 0:
            self.save(self._model_path)
            print(f"[Автосохранение] Модель сохранена в {self._model_path}")

    def learn_negative_pair(self, input_text: str, output_text: str, penalty: float = 0.15):
        """
        Отрицательное обучение – ослабляем связь или создаём тормозную.
        """
        input_norm = input_text.strip().lower()
        output_norm = output_text.strip().lower()
        if input_norm == output_norm:
            return

        input_id = self.get_or_create_concept_neuron(input_text)
        output_id = self.get_or_create_concept_neuron(output_text)

        syn_id = self.edge_index.get((input_id, output_id))
        if syn_id is not None and syn_id in self.synapses:
            syn = self.synapses[syn_id]
            new_weight = syn.weight - penalty
            if new_weight < -2.0:
                new_weight = -2.0
            syn.update(delta_weight=new_weight - syn.weight)
            if syn.weight < 0:
                syn.is_inhibitory = True
            print(f"[-] Отрицательное обучение: уменьшен вес связи «{input_text}» → «{output_text}» до {syn.weight:.3f}")
        else:
            weight = -random.uniform(0.1, 0.3)
            syn_id = self._create_synapse(input_id, output_id, weight=weight, is_inhibitory=True)
            syn = self.synapses[syn_id]
            print(f"[-] Отрицательное обучение: создан тормозной синапс «{input_text}» → «{output_text}» (вес: {syn.weight:.3f})")

        self._learn_counter += 1
        if self._learn_counter % self.AUTO_SAVE_EVERY == 0:
            self.save(self._model_path)
            print(f"[Автосохранение] Модель сохранена в {self._model_path}")

    # -------------------- Инференс --------------------
    def infer(self, input_signal: Signal, input_text: str = "", top_k: int = 3) -> List[Tuple[str, float]]:
        old_step = self.step_counter
        self.step_counter += 1
        activated = self.propagate_signal(input_signal)
        self.step_counter = old_step

        candidates = []
        for nid in activated:
            neuron = self.neurons.get(nid)
            if neuron and neuron.cluster == 'output' and neuron.label:
                if input_text and (neuron.label.lower() == input_text.lower() or input_text.lower() in neuron.label.lower()):
                    continue
                candidates.append((neuron.label, neuron.activation))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    # -------------------- Случайное понятие --------------------
    def get_random_concept(self) -> Optional[str]:
        output_neurons = [n for n in self.neurons.values() if n.cluster == 'output' and n.label]
        if not output_neurons:
            return None
        return random.choice(output_neurons).label

    # -------------------- Генерация ответа --------------------
    def generate_answer(self, input_text: str, max_hops: int = 6,
                        min_weight: float = 0.02, temperature: float = 0.0,
                        mode: str = 'chain') -> Dict[str, Any]:
        if mode == 'random':
            concept = self.get_random_concept()
            if concept is None:
                return {"chain": [], "text": "В сети пока нет выученных понятий.", "known": False, "fallback": False}
            return {"chain": [concept], "text": concept, "known": True, "fallback": False}

        normalized = input_text.strip().lower()
        if not normalized:
            return {"chain": [], "text": "Пустой запрос.", "known": False, "fallback": False}

        start_id = self.concept_index.get(normalized)
        used_fallback = False

        if start_id is None:
            sig = self.text_to_signal(input_text)
            top = self.infer(sig, input_text=input_text, top_k=1)
            if top:
                start_id = self.concept_index.get(top[0][0].strip().lower())
                used_fallback = True

        if start_id is None or start_id not in self.neurons:
            return {
                "chain": [input_text],
                "text": f"Я пока не знаю про «{input_text}». Научи меня: learn {input_text} => <ответ>",
                "known": False,
                "fallback": used_fallback,
            }

        chain = [self.neurons[start_id].label or input_text]
        visited = {start_id}
        current = start_id

        for _ in range(max_hops):
            candidates = []
            for syn_id in self.neurons[current].outgoing_synapses:
                syn = self.synapses.get(syn_id)
                if syn is None:
                    continue
                target = self.neurons.get(syn.to_neuron)
                if target is None or target.cluster != 'output' or not target.label:
                    continue
                if syn.to_neuron in visited:
                    continue
                if syn.weight <= 0:
                    continue
                score = syn.weight + syn.confidence * 0.1
                if score >= min_weight:
                    candidates.append((syn, score))

            if not candidates:
                break

            if temperature > 0:
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

        text = " → ".join(chain)
        return {"chain": chain, "text": text, "known": True, "fallback": used_fallback}

    # -------------------- Просмотр связей --------------------
    def show_links(self, text: str, top_k: int = 10):
        normalized = text.strip().lower()
        if not normalized:
            print("Пустой запрос.")
            return
        nid = self.concept_index.get(normalized)
        if nid is None:
            print(f"Понятие «{text}» ещё не изучено.")
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
            print(f"У «{text}» пока нет исходящих связей.")
            return
        print(f"Связи «{text}»:")
        for label, weight, count, conf, inhib in links[:top_k]:
            inhib_str = " (тормозная)" if inhib else ""
            print(f"  → {label}   (вес={weight:.3f}, повторов={count}, уверенность={conf:.2f}){inhib_str}")

    # -------------------- Сон --------------------
    def sleep(self, duration_steps: int = 10):
        print("Начало сна...")
        for _ in range(duration_steps):
            to_remove = []
            for nid, neuron in self.neurons.items():
                if neuron.energy < 0.05 and neuron.importance < 0.1 and neuron.cluster != 'output':
                    to_remove.append(nid)
            for nid in to_remove:
                neuron = self.neurons[nid]
                for syn_id in list(neuron.incoming_synapses) + list(neuron.outgoing_synapses):
                    self._remove_synapse(syn_id)
                del self.neurons[nid]
                print(f"Удалён шумовой нейрон {nid}")

            for syn in self.synapses.values():
                if syn.frequency > 0.5:
                    syn.update(delta_weight=0.01, delta_confidence=0.01)
                if syn.reward > 0.5:
                    syn.update(delta_weight=0.02, delta_confidence=0.02)
                syn.reward *= 0.9

        for item in self.short_memory.get_recent(50):
            self.long_memory.add(item)
        self.short_memory.clear()
        print("Сон завершён.")
        self.save(self._model_path)

    # -------------------- Сохранение/загрузка --------------------
    def save(self, filename: str = None):
        filename = filename or self._model_path
        data = {
            "dim": self.dim,
            "neurons": [],
            "synapses": [],
            "coactivation_counter": {str(k): v for k, v in self.coactivation_counter.items()},
            "step_counter": self.step_counter,
        }
        for nid, n in self.neurons.items():
            data["neurons"].append({
                "id": nid,
                "embedding": n.embedding.tolist(),
                "cluster": n.cluster,
                "label": n.label,
                "importance": n.importance,
                "energy": n.energy,
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
                "context_vector": s.context_vector.tolist(),
                "semantic_vector": s.semantic_vector.tolist(),
                "episodic_vector": s.episodic_vector.tolist(),
                "is_inhibitory": s.is_inhibitory,
            })
        tmp_filename = filename + ".tmp"
        with open(tmp_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_filename, filename)
        print(f"Модель сохранена в {filename}")

    def load(self, filename: str = None):
        filename = filename or self._model_path
        if not os.path.exists(filename):
            print(f"Файл {filename} не найден, создаём новую модель.")
            return
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Ошибка чтения файла {filename}: {e}")
            backup = filename + ".bak"
            try:
                os.rename(filename, backup)
                print(f"Повреждённый файл переименован в {backup}, создаём новую модель.")
            except Exception as rename_error:
                print(f"Не удалось переименовать файл: {rename_error}. Создаём новую модель без сохранения старого.")
            return
        except Exception as e:
            print(f"Неожиданная ошибка при загрузке: {e}")
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

        for nd in data["neurons"]:
            n = Neuron(embedding=np.array(nd["embedding"]),
                       cluster=nd["cluster"],
                       importance=nd["importance"],
                       energy=nd["energy"])
            n.id = nd["id"]
            n.label = nd["label"]
            self.neurons[n.id] = n
            if n.cluster == 'output' and n.label:
                self.concept_index[n.label.strip().lower()] = n.id

        for sd in data["synapses"]:
            s = Synapse(
                sd["from"], sd["to"], weight=sd["weight"],
                plasticity=sd["plasticity"], confidence=sd["confidence"],
                frequency=sd["frequency"], energy=sd["energy"],
                context_vector=np.array(sd["context_vector"]) if "context_vector" in sd else None,
                semantic_vector=np.array(sd["semantic_vector"]) if "semantic_vector" in sd else None,
                episodic_vector=np.array(sd["episodic_vector"]) if "episodic_vector" in sd else None,
                is_inhibitory=sd.get("is_inhibitory", False)
            )
            s.id = sd["id"]
            s.reward = sd["reward"]
            s.usage_count = sd.get("usage_count", 0)
            self.synapses[s.id] = s
            self.edge_index[(sd["from"], sd["to"])] = s.id
            if sd["from"] in self.neurons:
                self.neurons[sd["from"]].add_outgoing(s.id)
            if sd["to"] in self.neurons:
                self.neurons[sd["to"]].add_incoming(s.id)

        print(f"Модель загружена из {filename} "
              f"(нейронов: {len(self.neurons)}, синапсов: {len(self.synapses)}, "
              f"понятий: {len(self.concept_index)})")

# ============================
# Класс Teacher – оценка ответов через LLM (исправлен)
# ============================
class Teacher:
    def __init__(self):
        self.system_prompt = (
            "Ты — строгий, но справедливый учитель. Твоя задача — оценить ответ, данный нейросетью на вопрос пользователя.\n"
            "Оцени ответ по шкале от 0 до 1, где 0 — совершенно неверно, 1 — идеально правильно.\n"
            "Также, если ответ можно улучшить, предложи свой улучшенный вариант (коротко, 1–3 слова или фразу).\n"
            "Формат вывода: <оценка>|<улучшенный ответ>\n"
            "Например: 0.8|звезда\n"
            "Если ответ уже хорош, просто укажи высокую оценку и повтори его."
        )

    def evaluate(self, question: str, brain_answer: str) -> Tuple[float, str]:
        """
        Отправляет запрос к LLM и возвращает оценку (0..1) и улучшенный ответ.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Вопрос: {question}\nОтвет нейросети: {brain_answer}\nОценка и улучшенный ответ:"}
        ]
        try:
            response = client.chat.completions.create(
                model="local-model",  # LM Studio использует эту модель
                messages=messages,
                max_tokens=100,
                temperature=0.3,
            )
            raw = response.choices[0].message.content.strip()
            # Парсим: ищем число с плавающей точкой и текст после разделителя
            score = 0.5
            improved = brain_answer
            # Ищем число в начале или после "оценка"
            match_num = re.search(r'(\d+\.?\d*)', raw)
            if match_num:
                score = float(match_num.group(1))
                # Убираем число и разделитель, остальное считаем улучшенным ответом
                rest = raw.replace(match_num.group(1), '').strip()
                if rest.startswith('|'):
                    rest = rest[1:].strip()
                if rest:
                    improved = rest
                else:
                    improved = brain_answer
            else:
                # Если числа нет, возможно LLM выдала только текст
                # тогда считаем оценку 0.5, а ответ берём как улучшенный
                improved = raw
            score = max(0.0, min(1.0, score))
            return score, improved
        except Exception as e:
            print(f"Ошибка при оценке LLM: {e}")
            return 0.5, brain_answer

# ============================
# Интерактивный режим с учителем
# ============================
def interactive_with_teacher(brain: Brain):
    teacher = Teacher()
    print("\n=== Интерактивный режим с учителем (LM Studio) ===")
    print("Введите вопрос, Brain ответит, учитель оценит и обучит.")
    print("Команды:")
    print("  stats   – статистика")
    print("  save    – сохранить модель")
    print("  sleep   – запустить 'сон'")
    print("  exit    – сохранить и выйти")
    print("  (любой другой текст – вопрос)")

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
                print(f"Нейронов: {len(brain.neurons)} (input: {ic}, output/понятия: {oc}, hidden: {hc})")
                print(f"Синапсов: {len(brain.synapses)}")
                print(f"Выученных понятий: {len(brain.concept_index)}")
                continue
            if lower == 'save':
                brain.save()
                continue
            if lower == 'sleep':
                brain.sleep(duration_steps=5)
                continue

            # Генерация ответа Brain со случайной температурой
            temp = random.uniform(0.2, 1.0)
            print(f"Температура: {temp:.2f}")
            result = brain.generate_answer(user_input, temperature=temp, mode='chain')
            brain_answer = result['text']
            print(f"Brain: {brain_answer}")

            # Оценка учителем
            score, improved = teacher.evaluate(user_input, brain_answer)
            print(f"Оценка учителя: {score:.2f}, улучшенный ответ: {improved}")

            # Принимаем решение об обучении
            if score >= 0.7:
                if improved and improved.lower() != brain_answer.lower() and improved.lower() != user_input.lower():
                    brain.learn_pair(user_input, improved)
                else:
                    brain.learn_pair(user_input, brain_answer)
            elif score <= 0.3:
                if improved and improved.lower() != brain_answer.lower() and improved.lower() != user_input.lower():
                    brain.learn_negative_pair(user_input, brain_answer)
                    brain.learn_pair(user_input, improved)
                else:
                    brain.learn_negative_pair(user_input, brain_answer)
            else:
                print("Оценка нейтральная – обучение пропущено.")

    except (KeyboardInterrupt, EOFError):
        print("\nПрерывание — сохраняю модель...")
    finally:
        brain.save()
        print("Модель сохранена. До встречи.")

# ============================
# Запуск
# ============================
if __name__ == "__main__":
    MODEL_PATH = "brain_model_trained.json"
    brain = Brain(dim_embedding=64, input_neurons=30, output_neurons=30, hidden_neurons=140,
                  model_path=MODEL_PATH)
    brain.load(MODEL_PATH)
    interactive_with_teacher(brain)