import numpy as np
import random
import time
import json
import os
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple, Optional, Any
import copy

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
# Класс Synapse
# ============================
class Synapse:
    def __init__(self, from_neuron, to_neuron, weight: float = 0.1, plasticity: float = 0.5,
                 confidence: float = 0.1, frequency: float = 0.0, energy: float = 1.0,
                 context_vector: Optional[np.ndarray] = None,
                 semantic_vector: Optional[np.ndarray] = None,
                 episodic_vector: Optional[np.ndarray] = None):
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

    def update(self, delta_weight: float, delta_plasticity: float = 0.0,
               delta_confidence: float = 0.0, delta_frequency: float = 0.0):
        self.weight = max(0.0, self.weight + delta_weight)
        self.plasticity = np.clip(self.plasticity + delta_plasticity, 0.0, 1.0)
        self.confidence = np.clip(self.confidence + delta_confidence, 0.0, 1.0)
        self.frequency += delta_frequency
        self.usage_count += 1
        self.last_used = time.time()
        self.energy = min(1.0, self.energy + 0.01)

    def decay(self, decay_rate: float = 0.001):
        self.weight = max(0.0, self.weight - decay_rate)
        self.confidence = max(0.0, self.confidence - decay_rate * 0.5)
        self.energy = max(0.0, self.energy - decay_rate * 0.2)
        self.frequency *= 0.999

    def is_dead(self, weight_threshold=0.01, max_age_seconds=3600*24*30) -> bool:
        if self.weight < weight_threshold and (time.time() - self.last_used) > max_age_seconds:
            return True
        if self.usage_count == 0 and (time.time() - self.creation_time) > max_age_seconds:
            return True
        return False

    def __repr__(self):
        return f"Synapse(id={self.id}, {self.from_neuron}->{self.to_neuron}, w={self.weight:.3f})"

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
# Главный класс Brain (доработан)
# ============================
class Brain:
    def __init__(self, dim_embedding: int = 64, input_neurons: int = 30, output_neurons: int = 30,
                 hidden_neurons: int = 140):
        self.dim = dim_embedding
        self.neurons: Dict[int, Neuron] = {}
        self.synapses: Dict[int, Synapse] = {}
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

        # Начальные нейроны
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

        # Небольшое количество прямых связей вход-выход (для скорости)
        for _ in range(min(20, len(input_ids)*len(output_ids)//2)):
            a = random.choice(input_ids)
            b = random.choice(output_ids)
            self._create_synapse(a, b, weight=random.uniform(0.005, 0.02))

    # -------------------- Управление синапсами/нейронами --------------------
    def _create_synapse(self, from_id: int, to_id: int, weight: float = None,
                        context_vec: np.ndarray = None) -> Optional[int]:
        if from_id not in self.neurons or to_id not in self.neurons:
            return None
        for syn_id in self.synapses:
            syn = self.synapses[syn_id]
            if syn.from_neuron == from_id and syn.to_neuron == to_id:
                return syn_id
        if weight is None:
            weight = random.uniform(0.01, 0.1)
        if context_vec is None:
            context_vec = random_vector(self.dim)
        syn = Synapse(from_id, to_id, weight=weight,
                      context_vector=context_vec,
                      semantic_vector=random_vector(self.dim),
                      episodic_vector=random_vector(self.dim))
        self.synapses[syn.id] = syn
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
            del self.synapses[synapse_id]

    def _create_neuron(self, embedding: np.ndarray = None, importance: float = 0.5,
                       cluster: str = 'hidden', label: str = None) -> int:
        if embedding is None:
            embedding = random_vector(self.dim)
        n = Neuron(embedding=embedding, importance=importance, cluster=cluster)
        if label:
            # Обрезаем длинную метку до 30 символов
            n.label = label[:30] if len(label) > 30 else label
        self.neurons[n.id] = n

        # Связываем новый нейрон только с нейронами того же кластера (кроме output и input)
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
                syn_id = None
                for sid, syn in self.synapses.items():
                    if (syn.from_neuron == a and syn.to_neuron == b) or (syn.from_neuron == b and syn.to_neuron == a):
                        syn_id = sid
                        break
                if syn_id is not None:
                    syn = self.synapses[syn_id]
                    delta = 0.01 * (1.0 - syn.weight)
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

    # -------------------- Распространение сигнала --------------------
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
                    if syn is None or syn.weight < 0.01:
                        continue
                    target = self.neurons.get(syn.to_neuron)
                    if target is None or target.energy < 0.2:
                        continue
                    new_signal = copy.deepcopy(input_signal)
                    new_signal.energy *= syn.weight
                    new_signal.source = nid
                    new_signal.destination = syn.to_neuron
                    new_signal.confidence = syn.confidence
                    new_signal.history.append((nid, syn_id, syn.to_neuron))
                    if new_signal not in self.working_memory.items:
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

    # -------------------- Шаг обучения (для совместимости) --------------------
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
                    if syn:
                        syn.reward += reward
                        syn.update(delta_weight=reward * 0.005, delta_confidence=reward * 0.005)
                for syn_id in neuron.outgoing_synapses:
                    syn = self.synapses.get(syn_id)
                    if syn:
                        syn.reward += reward
                        syn.update(delta_weight=reward * 0.005, delta_confidence=reward * 0.005)

        if self.step_counter % 10 == 0:
            self._create_new_synapses_from_coactivation()
        if self.step_counter % 20 == 0:
            self._prune_synapses()

        self.short_memory.add((input_signal, activated))
        self.global_time = time.time()
        return activated

    # -------------------- Обучение пары (улучшенное) --------------------
    def get_or_create_output_neuron(self, word: str) -> int:
        label = word[:30] if len(word) > 30 else word
        for nid, n in self.neurons.items():
            if n.cluster == 'output' and n.label == label:
                return nid
        emb = self.text_to_embedding(word)
        nid = self._create_neuron(embedding=emb, cluster='output', label=label)
        print(f"Создан новый выходной нейрон {nid} для '{label}'")
        return nid

    def text_to_embedding(self, text: str) -> np.ndarray:
        words = text.lower().split()
        if not words:
            return random_vector(self.dim)
        vec = np.zeros(self.dim)
        for word in words:
            h = hash(word) % 100000
            np.random.seed(h)
            word_vec = np.random.randn(self.dim)
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

    def learn_pair(self, input_word: str, output_word: str):
        if input_word.lower() == output_word.lower():
            print("Пропускаем обучение: вход и выход совпадают.")
            return

        target_id = self.get_or_create_output_neuron(output_word)
        sig = self.text_to_signal(input_word)

        activated = self.propagate_signal(sig)

        for nid in activated:
            neuron = self.neurons.get(nid)
            if neuron is None or neuron.cluster == 'output':
                continue
            syn_id = self._create_synapse(nid, target_id, weight=0.05)
            if syn_id is not None:
                self.synapses[syn_id].update(delta_weight=0.05)
            else:
                for sid, syn in self.synapses.items():
                    if syn.from_neuron == nid and syn.to_neuron == target_id:
                        syn.update(delta_weight=0.05)
                        break

        self.step(sig, target_neuron_id=target_id)
        print(f"Обучение пары '{input_word}' -> '{output_word}' завершено.")

    # -------------------- Получение ТОП-3 ответов --------------------
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

    # -------------------- Сон --------------------
    def sleep(self, duration_steps: int = 10):
        print("Начало сна...")
        for _ in range(duration_steps):
            to_remove = []
            for nid, neuron in self.neurons.items():
                if neuron.energy < 0.05 and neuron.importance < 0.1:
                    to_remove.append(nid)
            for nid in to_remove:
                neuron = self.neurons[nid]
                for syn_id in neuron.incoming_synapses + neuron.outgoing_synapses:
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

    # -------------------- Сохранение/загрузка (с защитой от повреждений) --------------------
    def save(self, filename: str = "brain_model.json"):
        data = {
            "dim": self.dim,
            "neurons": [],
            "synapses": [],
            "coactivation_counter": dict(self.coactivation_counter),
            "step_counter": self.step_counter
        }
        for nid, n in self.neurons.items():
            data["neurons"].append({
                "id": nid,
                "embedding": n.embedding.tolist(),
                "cluster": n.cluster,
                "label": n.label,
                "importance": n.importance,
                "energy": n.energy
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
                "reward": s.reward
            })
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Модель сохранена в {filename}")

    def load(self, filename: str = "brain_model.json"):
        if not os.path.exists(filename):
            print(f"Файл {filename} не найден, создаём новую модель.")
            return
        try:
            with open(filename, 'r') as f:
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
        self.coactivation_counter = defaultdict(int, data["coactivation_counter"])
        self.neurons.clear()
        self.synapses.clear()
        for nd in data["neurons"]:
            n = Neuron(embedding=np.array(nd["embedding"]),
                       cluster=nd["cluster"],
                       importance=nd["importance"],
                       energy=nd["energy"])
            n.id = nd["id"]
            n.label = nd["label"]
            self.neurons[n.id] = n
        for sd in data["synapses"]:
            s = Synapse(sd["from"], sd["to"], weight=sd["weight"],
                        plasticity=sd["plasticity"], confidence=sd["confidence"],
                        frequency=sd["frequency"], energy=sd["energy"])
            s.id = sd["id"]
            s.reward = sd["reward"]
            self.synapses[s.id] = s
            if sd["from"] in self.neurons:
                self.neurons[sd["from"]].add_outgoing(s.id)
            if sd["to"] in self.neurons:
                self.neurons[sd["to"]].add_incoming(s.id)
        print(f"Модель загружена из {filename}")

# ============================
# Интерактивный режим
# ============================
def interactive_mode(brain: Brain):
    print("\n=== Интерактивный режим ===")
    print("Введите слово или фразу – сеть покажет топ-3 ассоциаций.")
    print("Команды:")
    print("  learn <слово1> <слово2>  – обучить пару напрямую")
    print("  stats                    – статистика сети")
    print("  save                     – сохранить модель сейчас")
    print("  load                     – перезагрузить модель из файла")
    print("  sleep                    – запустить 'сон'")
    print("  exit                     – сохранить и выйти")

    while True:
        text = input("\n> ").strip()
        if not text:
            continue

        if text.lower() == 'exit':
            brain.save()
            print("Выход.")
            break

        parts = text.split()
        cmd = parts[0].lower()

        if cmd == 'learn' and len(parts) == 3:
            in_word, out_word = parts[1], parts[2]
            brain.learn_pair(in_word, out_word)
            continue

        if cmd == 'stats':
            ic = sum(1 for n in brain.neurons.values() if n.cluster == 'input')
            oc = sum(1 for n in brain.neurons.values() if n.cluster == 'output')
            hc = sum(1 for n in brain.neurons.values() if n.cluster == 'hidden')
            print(f"Нейронов: {len(brain.neurons)} (input: {ic}, output: {oc}, hidden: {hc})")
            print(f"Синапсов: {len(brain.synapses)}")
            continue

        if cmd == 'save':
            brain.save()
            continue

        if cmd == 'load':
            brain.load()
            continue

        if cmd == 'sleep':
            brain.sleep(duration_steps=5)
            continue

        # Обычный запрос
        input_phrase = text
        sig = brain.text_to_signal(input_phrase)
        top = brain.infer(sig, input_text=input_phrase, top_k=3)

        if top:
            print("Топ-3 ассоциации:")
            for label, act in top:
                print(f"  {label}: {act:.3f}")
        else:
            print("Нет уверенных ассоциаций (или все совпадают с запросом).")

        if top:
            best_label = top[0][0]
            print(f"\nЛучший ответ: '{best_label}'")
            print("Это правильно? (да / нет / свой вариант)")
            response = input("> ").strip()
            if response.lower() == 'да':
                if best_label.lower() != input_phrase.lower():
                    brain.learn_pair(input_phrase, best_label)
                else:
                    print("Ответ совпадает с запросом – обучение не требуется.")
            elif response.lower() == 'нет':
                print("Введите правильный ответ (слово):")
                correct = input("> ").strip()
                if correct and correct.lower() != input_phrase.lower():
                    brain.learn_pair(input_phrase, correct)
                else:
                    print("Пропускаем (совпадает с запросом).")
            else:
                if response and response.lower() != input_phrase.lower():
                    brain.learn_pair(input_phrase, response)
                else:
                    print("Пропускаем (совпадает с запросом).")

# ============================
# Запуск
# ============================
if __name__ == "__main__":
    brain = Brain(dim_embedding=64, input_neurons=30, output_neurons=30, hidden_neurons=140)
    brain.load("brain_model.json")  # теперь безопасно
    interactive_mode(brain)