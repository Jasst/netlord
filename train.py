import random
from main import Brain  # импортируем ваш класс Brain


def generate_training_pairs(num_pairs=200):
    """
    Генерирует список пар (вход, выход) для обучения.
    Используются слова из разных тематических групп.
    """
    # Базовый словарь (50 слов)
    words = [
        "солнце", "луна", "звезда", "планета", "комета",
        "вода", "огонь", "воздух", "земля", "лес",
        "дерево", "цветок", "трава", "куст", "лист",
        "птица", "рыба", "зверь", "насекомое", "рептилия",
        "дом", "квартира", "комната", "дверь", "окно",
        "стол", "стул", "кровать", "шкаф", "полка",
        "книга", "тетрадь", "ручка", "карандаш", "ластик",
        "машина", "поезд", "самолет", "корабль", "велосипед",
        "город", "деревня", "улица", "площадь", "парк",
        "утро", "день", "вечер", "ночь", "час"
    ]
    # Создаём уникальные пары (случайные, но с исключением совпадений)
    pairs = set()
    attempts = 0
    while len(pairs) < num_pairs and attempts < 10000:
        w1 = random.choice(words)
        w2 = random.choice(words)
        if w1 != w2:
            pairs.add((w1, w2))
        attempts += 1
    # Если не хватило, просто дублируем с перемешиванием (но обычно хватает)
    return list(pairs)[:num_pairs]


def train():
    print("Создание новой модели...")
    # Создаём мозг с теми же параметрами, что и в основном коде
    brain = Brain(dim_embedding=64, input_neurons=30, output_neurons=30,
                  hidden_neurons=140, model_path="brain_model_trained.json")

    print("Генерация обучающих пар...")
    pairs = generate_training_pairs(200)
    print(f"Получено {len(pairs)} уникальных пар для обучения.")

    # Обучаем
    for i, (inp, out) in enumerate(pairs):
        brain.learn_pair(inp, out)
        if (i + 1) % 20 == 0:
            print(f"Обработано {i + 1} пар...")

    # Сохраняем модель
    brain.save("brain_model_trained.json")
    print("\nОбучение завершено. Модель сохранена в brain_model_trained.json")

    # Статистика
    ic = sum(1 for n in brain.neurons.values() if n.cluster == 'input')
    oc = sum(1 for n in brain.neurons.values() if n.cluster == 'output')
    hc = sum(1 for n in brain.neurons.values() if n.cluster == 'hidden')
    print(f"Нейронов: {len(brain.neurons)} (input: {ic}, output/понятия: {oc}, hidden: {hc})")
    print(f"Синапсов: {len(brain.synapses)}")
    print(f"Выученных понятий с именем: {len(brain.concept_index)}")


if __name__ == "__main__":
    train()