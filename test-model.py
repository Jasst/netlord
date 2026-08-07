import numpy as np
import scipy.sparse as sp
import json
import requests
import re
import pickle
import random
import threading
import queue
import time
from collections import defaultdict
from typing import List, Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass, field

# ------------------------ Конфигурация ------------------------
@dataclass
class Config:
    """Гиперпараметры сети."""
    decay_rate: float = 0.999
    hebb_lr: float = 0.1
    identity_size: int = 20
    max_history: int = 5
    steps: int = 5
    threshold_default: float = 0.5
    leak_default: float = 0.9
    min_weight: float = 1e-5
    activation_threshold: float = 0.4
    layer_params: Dict[str, Dict] = field(default_factory=lambda: {
        'input':   {'threshold': 0.3, 'leak': 0.8, 'hebb_lr': 0.15},
        'hidden':  {'threshold': 0.5, 'leak': 0.9, 'hebb_lr': 0.1},
        'association': {'threshold': 0.6, 'leak': 0.9, 'hebb_lr': 0.12},
        'combination': {'threshold': 0.7, 'leak': 0.95, 'hebb_lr': 0.08},
        'output':  {'threshold': 0.4, 'leak': 0.85, 'hebb_lr': 0.1},
        'default': {'threshold': 0.5, 'leak': 0.9, 'hebb_lr': 0.1}
    })
    weight_normalization: bool = True
    norm_axis: int = 0
    max_weight: float = 5.0

# ------------------------ Класс нейрона ------------------------
class Neuron:
    __slots__ = ('name', 'layer', 'threshold', 'leak', 'activation', 'bias', 'id', 'last_active')
    def __init__(self, name: str, layer: str = "default", threshold: float = 0.5,
                 leak: float = 0.9, idx: int = -1):
        self.name = name
        self.layer = layer
        self.threshold = threshold
        self.leak = leak
        self.activation = 0.0
        self.bias = 0.0
        self.id = idx
        self.last_active = 0

    def reset(self) -> None:
        self.activation = 0.0

# ------------------------ Адаптивная идентичность (без изменений) ------------------------
class AdaptiveIdentity:
    def __init__(self, base_size: int = 20, sparsity: float = 0.1):
        self.base_size = base_size
        self.sparsity = sparsity
        self.size = base_size
        self.projection = None
        self.vector = np.random.rand(base_size) * 0.1
        self.target = np.ones(base_size) * 0.5
        self.history: List[np.ndarray] = []
        self.reward = 0.0
        self._num_neurons = 0

    def _ensure_projection(self, num_neurons: int) -> None:
        if self.projection is None or self.projection.shape[1] < num_neurons:
            old_cols = self.projection.shape[1] if self.projection is not None else 0
            new_cols = num_neurons
            n_nonzero = int(new_cols * self.size * self.sparsity)
            rows = np.random.randint(0, self.size, n_nonzero)
            cols = np.random.randint(0, new_cols, n_nonzero)
            data = np.random.randn(n_nonzero) * 0.1
            new_proj = sp.csr_matrix((data, (rows, cols)), shape=(self.size, new_cols))
            if self.projection is not None:
                old_proj = self.projection
                combined = sp.hstack([old_proj, new_proj[:, old_cols:new_cols]], format='csr')
                self.projection = combined
            else:
                self.projection = new_proj
            self._num_neurons = new_cols

    def update(self, network_neurons: List[Neuron], external_reward: Optional[float] = None) -> None:
        num_neurons = len(network_neurons)
        if num_neurons == 0:
            return
        self._ensure_projection(num_neurons)
        activations = np.array([n.activation for n in network_neurons], dtype=float)
        compressed = self.projection.dot(activations)
        compressed = 1.0 / (1.0 + np.exp(-compressed))
        proximity = 1.0 - np.linalg.norm(compressed - self.target) / np.sqrt(self.size)
        diversity = np.std(compressed)
        internal_reward = 0.7 * proximity + 0.3 * diversity
        total_reward = internal_reward if external_reward is None else 0.6 * internal_reward + 0.4 * external_reward
        self.reward = total_reward

        lr = 0.1
        if total_reward > 0.6:
            self.vector = (1 - lr) * self.vector + lr * compressed
            self.target = 0.9 * self.target + 0.1 * compressed
        elif total_reward > 0.3:
            self.vector = 0.9 * self.vector + 0.1 * compressed
        else:
            noise = np.random.normal(0, 0.2, self.size)
            self.vector = np.clip(self.vector + noise, 0, 1)
            self.target = 0.9 * self.target + 0.1 * compressed

        if total_reward > 0.3:
            proj_lr = 0.01
            rows, cols = self.projection.nonzero()
            for i in range(len(rows)):
                r = rows[i]; c = cols[i]
                old = self.projection[r, c]
                delta = proj_lr * (compressed[r] * activations[c] - old)
                self.projection[r, c] = old + delta
            self.projection.data = np.clip(self.projection.data, -1.0, 1.0)

        self.history.append(self.vector.copy())
        if len(self.history) > 30:
            self.history.pop(0)

    def ask(self, question: str) -> str:
        lower = question.lower()
        if "кто я" in lower:
            avg = np.mean(self.vector)
            if avg > 0.7:
                return "Я — уверенная, активная личность."
            if avg > 0.4:
                return "Я — спокойная, уравновешенная сущность."
            return "Я — пассивная, наблюдательная система."
        if "сравни" in lower:
            if len(self.history) > 1:
                diff = np.linalg.norm(self.vector - self.history[-2])
                if diff < 0.1:
                    return "Я почти не изменилась."
                if diff < 0.5:
                    return "Я немного изменилась."
                return "Я сильно изменилась!"
            return "У меня нет прошлого."
        return "Я не понимаю вопрос."

# ------------------------ Основная сеть (улучшенная) ------------------------
class AdaptiveNetwork:
    def __init__(self, llm_teacher: Optional['LLMTeacher'] = None,
                 config: Optional[Config] = None):
        self.config = config or Config()
        self.teacher = llm_teacher
        self.identity = AdaptiveIdentity(base_size=self.config.identity_size)
        self.name_to_neuron: Dict[str, Neuron] = {}
        self.neurons: List[Neuron] = []
        self.weights_csr = sp.csr_matrix((0, 0), dtype=float)
        self.weights_lil = sp.lil_matrix((0, 0), dtype=float)
        self.history: List[str] = []
        self.step_count = 0
        self.llm_queue = queue.Queue()
        self.llm_results = {}
        self.llm_lock = threading.Lock()
        self.stop_event = threading.Event()
        self._start_llm_worker()

        # VECTORIZED: массивы параметров для быстрого доступа в feed_forward
        self._leaks = np.array([], dtype=float)
        self._thresholds = np.array([], dtype=float)
        self._biases = np.array([], dtype=float)

    def _start_llm_worker(self):
        def worker():
            while not self.stop_event.is_set():
                try:
                    word, _ = self.llm_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if word is None:
                    break
                data = self.teacher.generate_learning_data(word) if self.teacher else None
                with self.llm_lock:
                    self.llm_results[word] = data
                self.llm_queue.task_done()
        self.llm_thread = threading.Thread(target=worker, daemon=True)
        self.llm_thread.start()

    def shutdown(self):
        self.stop_event.set()
        self.llm_queue.put((None, None))
        if hasattr(self, 'llm_thread') and self.llm_thread.is_alive():
            self.llm_thread.join(timeout=1.0)

    # VECTORIZED: синхронизация массивов параметров при добавлении нейрона
    def _sync_arrays(self):
        """Обновляет внутренние массивы параметров после изменения списка нейронов."""
        if self.neurons:
            self._leaks = np.array([n.leak for n in self.neurons], dtype=float)
            self._thresholds = np.array([n.threshold for n in self.neurons], dtype=float)
            self._biases = np.array([n.bias for n in self.neurons], dtype=float)
        else:
            self._leaks = np.array([], dtype=float)
            self._thresholds = np.array([], dtype=float)
            self._biases = np.array([], dtype=float)

    def add_neuron(self, name: str, layer: str = "default",
                   threshold: Optional[float] = None,
                   leak: Optional[float] = None) -> Neuron:
        if name in self.name_to_neuron:
            return self.name_to_neuron[name]
        layer_cfg = self.config.layer_params.get(layer, self.config.layer_params['default'])
        threshold = threshold or layer_cfg.get('threshold', self.config.threshold_default)
        leak = leak or layer_cfg.get('leak', self.config.leak_default)
        idx = len(self.neurons)
        neuron = Neuron(name, layer, threshold, leak, idx)
        self.neurons.append(neuron)
        self.name_to_neuron[name] = neuron
        new_size = idx + 1
        self.weights_lil.resize((new_size, new_size))
        self.weights_csr = self.weights_lil.tocsr()
        # VECTORIZED: обновляем массивы параметров
        self._sync_arrays()
        return neuron

    def get_or_create_id(self, name: str, layer: str = "input") -> int:
        neuron = self.add_neuron(name, layer=layer)
        return neuron.id

    def get_weight(self, from_id: int, to_id: int) -> float:
        return self.weights_lil[from_id, to_id]

    def set_weight(self, from_id: int, to_id: int, value: float) -> None:
        if abs(value) < self.config.min_weight:
            value = 0.0
        self.weights_lil[from_id, to_id] = value
        self.weights_csr = self.weights_lil.tocsr()

    def add_synapse(self, from_neuron: Neuron, to_neuron: Neuron, weight: float = 0.5) -> None:
        self.set_weight(from_neuron.id, to_neuron.id, weight)

    def decay_weights(self) -> None:
        if self.weights_lil.nnz == 0:
            return
        self.weights_lil = self.weights_lil * self.config.decay_rate
        self.weights_lil.data = [np.array([v if abs(v) >= self.config.min_weight else 0.0 for v in row])
                                 for row in self.weights_lil.data]
        self.weights_lil.eliminate_zeros()
        self.weights_csr = self.weights_lil.tocsr()

        if self.config.weight_normalization:
            col_sums = np.array(self.weights_csr.sum(axis=0)).flatten()
            col_sums[col_sums == 0] = 1.0
            self.weights_csr = self.weights_csr @ sp.diags(1.0 / col_sums, format='csr')
            self.weights_lil = self.weights_csr.tolil()
            self.weights_csr.data = np.clip(self.weights_csr.data, -self.config.max_weight, self.config.max_weight)
            self.weights_lil = self.weights_csr.tolil()

        threshold_steps = 100
        dead = [n for n in self.neurons if n.layer != 'input' and self.step_count - n.last_active > threshold_steps]
        if dead:
            pass

    @staticmethod
    def tokenize(text: str) -> List[str]:
        stopwords = {'и', 'в', 'на', 'с', 'по', 'к', 'у', 'а', 'но', 'за', 'о', 'об',
                     'для', 'от', 'из', 'без', 'до', 'при', 'через', 'это', 'как',
                     'так', 'все', 'его', 'ее', 'их', 'мой', 'твой', 'наш', 'ваш',
                     'что', 'кто', 'где', 'когда', 'почему', 'зачем', 'какой', 'такой',
                     'быть', 'стать', 'являться'}
        words = re.findall(r'\b[a-zа-яё]+\b', text.lower())
        return [w for w in words if w not in stopwords and len(w) > 1]

    # ---------- VECTORIZED: прямой проход (полностью векторизован) ----------
    def feed_forward(self, input_names: List[str], steps: int = None) -> List[Neuron]:
        if steps is None:
            steps = self.config.steps
        # Сброс активаций у не-входных нейронов
        for n in self.neurons:
            if n.layer != "input":
                n.activation = 0.0

        input_ids = [self.get_or_create_id(name, layer="input") for name in input_names]
        for idx in input_ids:
            self.neurons[idx].activation = 1.0
            self.neurons[idx].last_active = self.step_count

        # Все активации – вектор
        activations = np.array([n.activation for n in self.neurons], dtype=float)
        input_mask = np.zeros(len(self.neurons), dtype=bool)
        input_mask[input_ids] = True

        # VECTORIZED: извлекаем массивы параметров (уже синхронизированы)
        leaks = self._leaks
        biases = self._biases
        thresholds = self._thresholds
        weights = self.weights_csr

        # Векторизованный цикл по шагам
        for _ in range(steps):
            # Входной сигнал (разреженное умножение)
            new_input = weights.dot(activations)   # вектор размера num_neurons
            total = biases + new_input
            new_act = 1.0 / (1.0 + np.exp(-total))   # сигмоида
            # Обновление активаций с утечкой (для всех нейронов)
            activations = leaks * activations + (1.0 - leaks) * new_act
            # Входные нейроны остаются равными 1.0
            activations[input_mask] = 1.0

            # Обновляем last_active для нейронов, у которых активация > порога
            # (это единственное место, где мы всё ещё используем цикл по нейронам,
            #  но он выполняется после каждого шага и не влияет на основные вычисления)
            active_mask = activations > thresholds
            # Обновляем атрибут last_active у соответствующих объектов
            # Это можно сделать через цикл, но он будет проходить только по активным нейронам,
            # что всё равно быстрее, чем полный перебор, но для простоты оставим полный проход,
            # но это не влияет на производительность основных вычислений.
            for idx, active in enumerate(active_mask):
                if active and not input_mask[idx]:
                    self.neurons[idx].last_active = self.step_count

        # Сохраняем активации обратно в объекты (для остальных частей кода, которые их используют)
        for idx, act in enumerate(activations):
            self.neurons[idx].activation = act

        # Возвращаем активированные нейроны (кроме входных)
        activated = [self.neurons[idx] for idx in range(len(self.neurons))
                     if not input_mask[idx] and self.neurons[idx].activation > self.neurons[idx].threshold]
        return activated

    def hebbian_learning(self, input_names: List[str], activated: List[Neuron]) -> None:
        active_ids = set()
        for name in input_names:
            if name in self.name_to_neuron:
                active_ids.add(self.name_to_neuron[name].id)
        active_ids.update(n.id for n in activated)
        if len(active_ids) < 2:
            return
        ids = list(active_ids)
        acts = np.array([self.neurons[idx].activation for idx in ids])
        lr_matrix = np.zeros((len(ids), len(ids)))
        for i, from_id in enumerate(ids):
            from_neuron = self.neurons[from_id]
            for j, to_id in enumerate(ids):
                if i == j:
                    continue
                to_neuron = self.neurons[to_id]
                layer_lr = self.config.layer_params.get(to_neuron.layer, self.config.layer_params['default']).get('hebb_lr', self.config.hebb_lr)
                lr_matrix[i, j] = layer_lr
        delta = lr_matrix * np.outer(acts, acts)
        for i, from_id in enumerate(ids):
            for j, to_id in enumerate(ids):
                if i != j and delta[i, j] > 0.01:
                    new_weight = self.get_weight(from_id, to_id) + delta[i, j]
                    self.set_weight(from_id, to_id, new_weight)
        self.decay_weights()

    def create_combination_neuron(self, input_names: List[str]) -> Optional[Neuron]:
        if len(input_names) < 2:
            return None
        sorted_names = sorted(input_names)
        combo_name = '+'.join(sorted_names)
        if combo_name in self.name_to_neuron:
            return self.name_to_neuron[combo_name]
        combo = self.add_neuron(combo_name, layer="combination", threshold=0.9, leak=0.9)
        for name in sorted_names:
            n = self.name_to_neuron.get(name)
            if n:
                self.add_synapse(n, combo, weight=0.5)
        combo.bias = -0.5 * len(sorted_names) + 0.1
        # VECTORIZED: обновляем biases массив после изменения bias
        self._sync_arrays()
        return combo

    def process_input(self, input_text: str, learn: bool = True,
                      external_reward: Optional[float] = None) -> Tuple[List[Neuron], List[Neuron]]:
        tokens = self.tokenize(input_text)
        if not tokens:
            tokens = [input_text]
        self.history.append(input_text)
        if len(self.history) > self.config.max_history:
            self.history.pop(0)

        activated = self.feed_forward(tokens)
        new_neurons = []
        if learn:
            self.hebbian_learning(tokens, activated)
            if len(tokens) >= 2:
                combo = self.create_combination_neuron(tokens)
                if combo:
                    new_neurons.append(combo)
                    total = combo.bias + sum(self.get_weight(n.id, combo.id) * n.activation
                                             for n in [self.name_to_neuron[t] for t in tokens if t in self.name_to_neuron])
                    combo.activation = 1.0 / (1.0 + np.exp(-total))
                    if combo.activation > combo.threshold and combo not in activated:
                        activated.append(combo)
        self.identity.update(self.neurons, external_reward=external_reward)
        self.step_count += 1
        return activated, new_neurons

    def generate_response(self, input_text: str, deterministic: bool = False,
                          external_reward: Optional[float] = None,
                          activation_threshold: Optional[float] = None,
                          max_words: Optional[int] = None) -> str:
        if activation_threshold is None:
            activation_threshold = self.config.activation_threshold

        lower = input_text.lower()
        if "кто я" in lower or "сравни" in lower:
            return self.identity.ask(input_text)

        activated, _ = self.process_input(input_text, learn=True, external_reward=external_reward)
        tokens = self.tokenize(input_text)
        forbidden = {self.name_to_neuron[t] for t in tokens if t in self.name_to_neuron}

        candidates = [n for n in activated if n not in forbidden and n.activation > activation_threshold]
        if not candidates:
            candidates = [n for n in self.neurons if n.layer != "input" and n not in forbidden and n.activation > 0.2]
        if not candidates:
            return "Я не знаю, как ответить."

        layer_priority = {'output': 4, 'combination': 3, 'association': 2, 'hidden': 1, 'default': 0}
        candidates.sort(key=lambda n: (layer_priority.get(n.layer, 0), n.activation), reverse=True)

        if max_words is not None and len(candidates) > max_words:
            candidates = candidates[:max_words]

        parts = []
        for n in candidates:
            if n.layer == 'association':
                parts.append(f"ассоциация '{n.name}'")
            elif n.layer == 'combination':
                parts.append(f"комбинация ({n.name})")
            elif n.layer == 'output':
                parts.append(f"результат: {n.name}")
            else:
                parts.append(n.name)

        if not parts:
            return "Мне нечего сказать."

        if len(parts) == 1:
            response = parts[0]
        elif len(parts) == 2:
            response = f"{parts[0]} и {parts[1]}"
        else:
            response = ", ".join(parts[:-1]) + f" и {parts[-1]}"

        prefixes = ["Мне кажется,", "Возможно,", "Я бы сказал,", "По моему мнению,", "Скорее всего,"]
        suffixes = ["", " — это важно.", " как я понимаю.", " если честно.", " наверное."]
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        full_response = f"{prefix} {response}{suffix}".strip()
        return " ".join(full_response.split())

    def learn_from_llm(self, word: str) -> bool:
        word_neuron = self.add_neuron(word, layer="input")
        if not self.teacher:
            print("Нет учителя LLM. Слово создано без ассоциаций.")
            return False

        with self.llm_lock:
            if word in self.llm_results:
                data = self.llm_results[word]
            else:
                data = None

        if data is None:
            self.llm_queue.put((word, None))
            start = time.time()
            while True:
                with self.llm_lock:
                    if word in self.llm_results:
                        data = self.llm_results[word]
                        break
                if time.time() - start > 2.0:
                    break
                time.sleep(0.1)
            if data is None:
                print(f"⚠️ LLM не вернула данные для '{word}'.")
                return False

        required = {'sentences', 'associations', 'opposite'}
        if not all(k in data for k in required):
            print("⚠️ Ответ LLM неполный.")
            return False

        for sentence in data['sentences']:
            sent = self.add_neuron(sentence, layer="sentence")
            self.add_synapse(word_neuron, sent, weight=0.7)
            self.add_synapse(sent, word_neuron, weight=0.7)
        for assoc in data['associations']:
            assoc_n = self.add_neuron(assoc, layer="association")
            self.add_synapse(word_neuron, assoc_n, weight=0.9)
            self.add_synapse(assoc_n, word_neuron, weight=0.5)
        opp = data.get('opposite', '')
        if opp:
            opp_n = self.add_neuron(opp, layer="opposite")
            self.add_synapse(word_neuron, opp_n, weight=-0.5)
            self.add_synapse(opp_n, word_neuron, weight=-0.5)
        print(f"✅ Слово '{word}' изучено.")
        return True

    def save(self, path: str) -> None:
        data = {
            'neurons': self.neurons,
            'name_to_neuron': self.name_to_neuron,
            'weights_csr': self.weights_csr,
            'identity': self.identity,
            'history': self.history,
            'step_count': self.step_count,
            'config': self.config,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Сохранено в {path}")

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.neurons = data['neurons']
        self.name_to_neuron = data['name_to_neuron']
        self.weights_csr = data['weights_csr']
        self.weights_lil = self.weights_csr.tolil()
        self.identity = data['identity']
        self.history = data['history']
        self.step_count = data['step_count']
        self.config = data.get('config', Config())
        for idx, n in enumerate(self.neurons):
            n.id = idx
        # VECTORIZED: синхронизируем массивы после загрузки
        self._sync_arrays()
        print(f"📂 Загружено из {path}")

    def reset_state(self) -> None:
        for n in self.neurons:
            n.reset()

# ------------------------ LLM Teacher (без изменений) ------------------------
class LLMTeacher:
    def __init__(self, base_url: str = "http://localhost:1234/v1",
                 model_name: Optional[str] = None,
                 prompt_template: Optional[str] = None):
        self.base_url = base_url
        self.model_name = model_name or "local-model"
        self.prompt_template = prompt_template or """
Ты — учитель для ассоциативной нейросети. Для слова "{word}" сгенерируй:
1. Три предложения с этим словом.
2. Два слова, с которыми оно часто связано (синонимы или контекст).
3. Одно слово, которое является антонимом или противоположностью.

Ответ дай строго в формате JSON:
{{"sentences": ["...", "...", "..."], "associations": ["...", "..."], "opposite": "..."}}
"""

    def generate_learning_data(self, word: str, retries: int = 2) -> Optional[Dict[str, Any]]:
        prompt = self.prompt_template.format(word=word)
        for attempt in range(retries):
            try:
                headers = {"Content-Type": "application/json"}
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.5,
                    "max_tokens": 300
                }
                resp = requests.post(f"{self.base_url}/chat/completions",
                                     headers=headers, json=payload, timeout=30)
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"]
                    json_text = self._extract_json(content)
                    if json_text:
                        data = json.loads(json_text)
                        if all(k in data for k in ('sentences','associations','opposite')):
                            return data
                else:
                    print(f"LLM вернула код {resp.status_code}")
            except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
                print(f"Ошибка LLM (попытка {attempt+1}): {e}")
            time.sleep(1)
        return None

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end > start:
            return text[start:end]
        return None

# ------------------------ Основной цикл (без изменений) ------------------------
def main():
    teacher = LLMTeacher()
    config = Config()
    net = AdaptiveNetwork(teacher, config)

    try:
        net.load("network_state.pkl")
    except FileNotFoundError:
        print("🚀 Новая сеть создана.")

    print("🚀 Саморазвивающаяся сеть с динамической генерацией ответов (улучшенная версия).")
    print("Число слов в ответе зависит от количества активированных ассоциаций.")
    print("Команды: 'Кто я?', 'Сравни с прошлым', 'reset', 'save', 'load', 'exit'\n")

    while True:
        try:
            user_input = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if not user_input:
            continue

        if user_input.lower() in ('exit', 'quit'):
            net.shutdown()
            break
        if user_input.lower() == 'reset':
            net.reset_state()
            print("🧹 Состояние сброшено.")
            continue
        if user_input.lower() == 'save':
            net.save("network_state.pkl")
            continue
        if user_input.lower() == 'load':
            try:
                net.load("network_state.pkl")
            except FileNotFoundError:
                print("Файл не найден.")
            continue

        tokens = net.tokenize(user_input)
        for w in tokens:
            if w not in net.name_to_neuron:
                net.learn_from_llm(w)

        response = net.generate_response(user_input, deterministic=False,
                                         activation_threshold=0.4,
                                         max_words=None)
        print(f"Сеть: {response}")

        avg = np.mean(net.identity.vector)
        reward = net.identity.reward
        total_synapses = net.weights_lil.nnz
        print(f"🧠 Идентичность (средняя): {avg:.2f}, Оценка: {reward:.2f}")
        print(f"[Нейронов: {len(net.neurons)}, Связей: {total_synapses}]\n")

if __name__ == "__main__":
    main()