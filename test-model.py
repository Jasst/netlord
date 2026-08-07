import numpy as np
import scipy.sparse as sp
import json
import requests
import re
import pickle
import random
from collections import defaultdict
from typing import List, Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass
import time

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
    activation_threshold: float = 0.4   # для динамической генерации

# ------------------------ Класс нейрона ------------------------
class Neuron:
    __slots__ = ('name', 'layer', 'threshold', 'leak', 'activation', 'bias', 'id')
    def __init__(self, name: str, layer: str = "default", threshold: float = 0.5,
                 leak: float = 0.9, idx: int = -1):
        self.name = name
        self.layer = layer
        self.threshold = threshold
        self.leak = leak
        self.activation = 0.0
        self.bias = 0.0
        self.id = idx

    def reset(self) -> None:
        self.activation = 0.0

# ------------------------ Слой идентичности ------------------------
class IdentityLayer:
    def __init__(self, size: int = 20):
        self.size = size
        self.vector = np.random.rand(size) * 0.1
        self.target = np.ones(size) * 0.5
        self.history: List[np.ndarray] = []
        self.reward = 0.0

    def update(self, network_neurons: List[Neuron], external_reward: Optional[float] = None) -> None:
        all_acts = [n.activation for n in network_neurons if n.layer != "input"]
        if not all_acts:
            return
        block_size = max(1, len(all_acts) // self.size)
        compressed = []
        for i in range(self.size):
            start = i * block_size
            end = min((i + 1) * block_size, len(all_acts))
            compressed.append(np.mean(all_acts[start:end]) if start < end else 0.0)
        new_vector = np.array(compressed)

        proximity = 1.0 - np.linalg.norm(new_vector - self.target) / np.sqrt(self.size)
        diversity = np.std(new_vector)
        internal_reward = 0.7 * proximity + 0.3 * diversity
        total_reward = internal_reward if external_reward is None else 0.6 * internal_reward + 0.4 * external_reward
        self.reward = total_reward

        lr = 0.1
        if total_reward > 0.6:
            self.vector = (1 - lr) * self.vector + lr * new_vector
        elif total_reward > 0.3:
            self.vector = 0.9 * self.vector + 0.1 * new_vector
        else:
            noise = np.random.normal(0, 0.2, self.size)
            self.vector = np.clip(self.vector + noise, 0, 1)
            self.target = 0.9 * self.target + 0.1 * new_vector

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

# ------------------------ Основная сеть ------------------------
class AdaptiveNetwork:
    def __init__(self, llm_teacher: Optional['LLMTeacher'] = None,
                 config: Optional[Config] = None):
        self.config = config or Config()
        self.teacher = llm_teacher
        self.identity = IdentityLayer(size=self.config.identity_size)
        self.name_to_neuron: Dict[str, Neuron] = {}
        self.neurons: List[Neuron] = []
        self.weights: sp.lil_matrix = sp.lil_matrix((0, 0), dtype=float)
        self.history: List[str] = []
        self.step_count = 0

    # ---------- Управление нейронами ----------
    def add_neuron(self, name: str, layer: str = "default",
                   threshold: Optional[float] = None,
                   leak: Optional[float] = None) -> Neuron:
        if name in self.name_to_neuron:
            return self.name_to_neuron[name]
        idx = len(self.neurons)
        threshold = threshold or self.config.threshold_default
        leak = leak or self.config.leak_default
        neuron = Neuron(name, layer, threshold, leak, idx)
        self.neurons.append(neuron)
        self.name_to_neuron[name] = neuron
        new_size = idx + 1
        self.weights.resize((new_size, new_size))
        return neuron

    def get_or_create_id(self, name: str) -> int:
        neuron = self.add_neuron(name, layer="input")
        return neuron.id

    # ---------- Работа с весами ----------
    def get_weight(self, from_id: int, to_id: int) -> float:
        return self.weights[from_id, to_id]

    def set_weight(self, from_id: int, to_id: int, value: float) -> None:
        if abs(value) < self.config.min_weight:
            value = 0.0
        self.weights[from_id, to_id] = value

    def add_synapse(self, from_neuron: Neuron, to_neuron: Neuron, weight: float = 0.5) -> None:
        self.set_weight(from_neuron.id, to_neuron.id, weight)

    # ---------- Забывание (исправленная версия) ----------
    def decay_weights(self) -> None:
        """Применяет затухание и удаляет слишком малые веса."""
        if self.weights.nnz == 0:
            return
        self.weights = self.weights * self.config.decay_rate
        # Преобразуем в coo для работы с плоским массивом данных
        coo = self.weights.tocoo()
        mask = np.abs(coo.data) < self.config.min_weight
        coo.data[mask] = 0.0
        coo.eliminate_zeros()
        self.weights = coo.tolil()

    # ---------- Токенизация ----------
    @staticmethod
    def tokenize(text: str) -> List[str]:
        stopwords = {'и', 'в', 'на', 'с', 'по', 'к', 'у', 'а', 'но', 'за', 'о', 'об',
                     'для', 'от', 'из', 'без', 'до', 'при', 'через', 'это', 'как',
                     'так', 'все', 'его', 'ее', 'их', 'мой', 'твой', 'наш', 'ваш',
                     'что', 'кто', 'где', 'когда', 'почему', 'зачем', 'какой', 'такой',
                     'быть', 'стать', 'являться'}
        words = re.findall(r'\b[a-zа-яё]+\b', text.lower())
        return [w for w in words if w not in stopwords and len(w) > 1]

    # ---------- Прямой проход (BFS) ----------
    def feed_forward(self, input_names: List[str], steps: int = None) -> List[Neuron]:
        if steps is None:
            steps = self.config.steps
        for n in self.neurons:
            if n.layer != "input":
                n.activation = 0.0
        input_ids = [self.get_or_create_id(name) for name in input_names]
        active = set(input_ids)
        for idx in input_ids:
            self.neurons[idx].activation = 1.0
        input_sum = np.zeros(len(self.neurons))
        for _ in range(steps):
            new_active = set()
            for idx in active:
                row = self.weights.getrow(idx)
                if row.nnz == 0:
                    continue
                for to_idx, weight in zip(row.indices, row.data):
                    if to_idx not in active:
                        input_sum[to_idx] += weight * self.neurons[idx].activation
            for idx in np.where(input_sum != 0)[0]:
                if idx in active:
                    continue
                neuron = self.neurons[idx]
                total = neuron.bias + input_sum[idx]
                new_input = 1.0 / (1.0 + np.exp(-total))
                neuron.activation = neuron.leak * neuron.activation + (1 - neuron.leak) * new_input
                if neuron.activation > neuron.threshold:
                    new_active.add(idx)
            input_sum[:] = 0.0
            active |= new_active
        activated = [self.neurons[idx] for idx in active if idx not in input_ids]
        return activated

    # ---------- Обучение (Хебб) ----------
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
        delta = self.config.hebb_lr * np.outer(acts, acts)
        for i, from_id in enumerate(ids):
            for j, to_id in enumerate(ids):
                if i != j and delta[i, j] > 0.01:
                    new_weight = self.get_weight(from_id, to_id) + delta[i, j]
                    self.set_weight(from_id, to_id, new_weight)
        self.decay_weights()   # после обновления применяем забывание

    # ---------- Комбинаторные нейроны ----------
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
        return combo

    # ---------- Обработка входа ----------
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

    # ---------- Генерация ответа (динамическая) ----------
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

        # Основные кандидаты
        candidates = [n for n in activated if n not in forbidden and n.activation > activation_threshold]
        if not candidates:
            # Резервный порог
            candidates = [n for n in self.neurons if n.layer != "input" and n not in forbidden and n.activation > 0.2]
        if not candidates:
            return "Я не знаю, как ответить."

        candidates.sort(key=lambda n: n.activation, reverse=True)
        if max_words is not None and len(candidates) > max_words:
            candidates = candidates[:max_words]

        parts = [n.name for n in candidates if n.name.strip()]
        if not parts:
            return "Мне нечего сказать."

        if len(parts) == 1:
            response = parts[0]
        elif len(parts) == 2:
            response = f"{parts[0]} и {parts[1]}"
        else:
            response = ", ".join(parts[:-1]) + f" и {parts[-1]}"

        prefixes = ["Мне кажется,", "Возможно,", "Я бы сказал,", "По моему мнению,", ""]
        suffixes = ["", " — это важно.", " если честно.", " как я понимаю.", " наверное."]
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        full_response = f"{prefix} {response}{suffix}".strip()
        return " ".join(full_response.split())

    # ---------- Обучение через LLM ----------
    def learn_from_llm(self, word: str) -> bool:
        # Всегда создаём нейрон, чтобы слово запомнилось
        word_neuron = self.add_neuron(word, layer="input")
        if not self.teacher:
            print("Нет учителя LLM. Слово создано без ассоциаций.")
            return False
        if word in self.name_to_neuron and len(self.neurons) > 1:  # уже изучалось?
            # Для простоты считаем, что если нейрон есть, он уже изучен
            # Но мы можем пропустить повторное обучение
            print(f"Слово '{word}' уже известно.")
            return True

        data = self.teacher.generate_learning_data(word)
        if not data:
            print("⚠️ LLM не вернула данные.")
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

    # ---------- Сериализация ----------
    def save(self, path: str) -> None:
        data = {
            'neurons': self.neurons,
            'name_to_neuron': self.name_to_neuron,
            'weights': self.weights.tocsr(),  # сохраняем как csr
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
        self.weights = data['weights'].tolil()
        self.identity = data['identity']
        self.history = data['history']
        self.step_count = data['step_count']
        self.config = data.get('config', Config())
        for idx, n in enumerate(self.neurons):
            n.id = idx
        print(f"📂 Загружено из {path}")

    def reset_state(self) -> None:
        for n in self.neurons:
            n.reset()

# ------------------------ LLM Teacher ------------------------
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

# ------------------------ Основной цикл ------------------------
def main():
    teacher = LLMTeacher()
    config = Config()
    net = AdaptiveNetwork(teacher, config)

    try:
        net.load("network_state.pkl")
    except FileNotFoundError:
        print("🚀 Новая сеть создана.")

    print("🚀 Саморазвивающаяся сеть с динамической генерацией ответов.")
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

        # Изучаем новые слова (даже если LLM нет, нейрон создаётся)
        tokens = net.tokenize(user_input)
        for w in tokens:
            if w not in net.name_to_neuron:
                net.learn_from_llm(w)

        # Генерация ответа с динамическим числом слов
        response = net.generate_response(user_input, deterministic=False,
                                         activation_threshold=0.4,
                                         max_words=None)
        print(f"Сеть: {response}")

        avg = np.mean(net.identity.vector)
        reward = net.identity.reward
        total_synapses = net.weights.nnz
        print(f"🧠 Идентичность (средняя): {avg:.2f}, Оценка: {reward:.2f}")
        print(f"[Нейронов: {len(net.neurons)}, Связей: {total_synapses}]\n")

if __name__ == "__main__":
    main()