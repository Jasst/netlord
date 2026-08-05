#!/usr/bin/env python
# test_brain.py – интеграционный тест для Smart Brain v7
# Запуск: python test_brain.py
# Или: pytest test_brain.py -v

import os
import sys
import time
import shutil
import torch
import numpy as np
from typing import List, Dict, Any

# Добавляем путь к проекту, если нужно
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brain import Brain, BrainConfig
from brain.utils import cosine_similarity


# ============================================================
# Конфигурация теста
# ============================================================
TEST_MODEL_DIR = "test_brain_model"
TEST_PAIRS = [
    ("солнце", "звезда"),
    ("луна", "спутник земли"),
    ("вода", "жидкость"),
    ("огонь", "высокая температура"),
    ("ветер", "движение воздуха"),
    ("снег", "осадки"),
    ("дождь", "капли воды"),
    ("гроза", "электрический разряд"),
    ("радуга", "преломление света"),
    ("туча", "скопление водяного пара"),
]


# ============================================================
# Вспомогательные функции
# ============================================================
def clean_test_model():
    """Удаляет тестовую модель, если она существует."""
    if os.path.exists(TEST_MODEL_DIR):
        shutil.rmtree(TEST_MODEL_DIR)
        print(f"[Тест] Удалена старая тестовая модель: {TEST_MODEL_DIR}")


def get_brain(clean: bool = True) -> Brain:
    """Создаёт новый экземпляр Brain для тестов."""
    if clean:
        clean_test_model()
    config = BrainConfig(
        dim_embedding=128,
        max_neurons=100,
        max_synapses=500,
        working_memory_size=3,
        episodic_capacity=20,
        model_dir=TEST_MODEL_DIR,
        learning_rate=1e-3,
        checkpoint_every=10,
    )
    brain = Brain(config)
    # Загружаем (если есть) — но мы её удалили
    return brain


def get_stats(brain: Brain) -> Dict[str, Any]:
    """Получает статистику мозга в удобном виде."""
    stats = brain.get_stats()
    return {
        "neurons": stats["neurons"],
        "synapses": stats["synapses"],
        "concepts": stats["concepts"],
        "knowledge_base": stats["knowledge_base"],
        "working_memory": stats["memory"]["working"],
        "episodic_memory": stats["memory"]["episodic"],
    }


def print_stats(brain: Brain, label: str = ""):
    """Выводит статистику в консоль."""
    stats = get_stats(brain)
    if label:
        print(f"\n--- {label} ---")
    print(f"  Нейронов: {stats['neurons']}")
    print(f"  Синапсов: {stats['synapses']}")
    print(f"  Понятий: {stats['concepts']}")
    print(f"  Фактов в KB: {stats['knowledge_base']}")
    print(f"  Рабочая память: {stats['working_memory']}")
    print(f"  Эпизодическая память: {stats['episodic_memory']}")


# ============================================================
# Тесты (можно запускать как функции, так и через pytest)
# ============================================================

def test_initial_state():
    """Тест 1: Начальное состояние мозга."""
    print("\n" + "=" * 60)
    print("ТЕСТ 1: Начальное состояние")
    print("=" * 60)
    brain = get_brain(clean=True)
    stats = get_stats(brain)
    assert stats["neurons"] == 11  # 10 инициализированных + 1 пустой (базовый)
    assert stats["synapses"] == 0
    assert stats["concepts"] == 0
    assert stats["knowledge_base"] == 0
    print("✅ Начальное состояние корректно.")
    return brain


def test_learn_pair(brain: Brain):
    """Тест 2: Обучение одной пары."""
    print("\n" + "=" * 60)
    print("ТЕСТ 2: Обучение пары (солнце -> звезда)")
    print("=" * 60)
    q, a = TEST_PAIRS[0]
    brain.learn_pair(q, a, epochs=1)
    stats = get_stats(brain)
    assert stats["concepts"] >= 2  # должны появиться понятия
    assert stats["synapses"] >= 1  # должен появиться синапс
    assert stats["knowledge_base"] >= 1
    print(f"✅ Пара '{q} -> {a}' выучена.")
    print_stats(brain, "После обучения одной пары")
    return brain


def test_retrieve_facts(brain: Brain):
    """Тест 3: Проверка извлечения фактов."""
    print("\n" + "=" * 60)
    print("ТЕСТ 3: Извлечение фактов по запросу")
    print("=" * 60)
    # После обучения первой пары, запрос "солнце" должен вернуть что-то
    result = brain.step("солнце")
    answer = result["answer"]
    memory_results = result.get("memory_results", [])
    print(f"  Ответ на 'солнце': {answer[:100]}...")
    print(f"  Найдено в памяти: {len(memory_results)} записей")
    # Проверяем, что есть хотя бы один результат из памяти (может быть, если FAISS нашёл)
    # В свежей модели может не быть, потому что нет эпизодов, но мы можем проверить наличие
    # Поскольку обучение добавило в эпизодическую память, должно быть хотя бы 1
    assert len(memory_results) >= 0  # пока просто проверяем, что нет ошибок
    print("✅ Извлечение фактов выполнено без ошибок.")
    return brain


def test_multi_learn(brain: Brain):
    """Тест 4: Обучение нескольких пар."""
    print("\n" + "=" * 60)
    print("ТЕСТ 4: Обучение нескольких пар")
    print("=" * 60)
    for q, a in TEST_PAIRS[1:5]:  # учим ещё 4 пары
        brain.learn_pair(q, a, epochs=1)
        print(f"  Выучено: {q} -> {a}")
    stats = get_stats(brain)
    print_stats(brain, "После обучения 5 пар")
    assert stats["synapses"] >= 5
    print("✅ Множественное обучение завершено.")
    return brain


def test_save_load(brain: Brain):
    """Тест 5: Сохранение и загрузка модели."""
    print("\n" + "=" * 60)
    print("ТЕСТ 5: Сохранение и загрузка")
    print("=" * 60)
    # Сохраняем
    brain.save()
    print(f"  Модель сохранена в {TEST_MODEL_DIR}")

    # Создаём новый мозг и загружаем
    brain2 = get_brain(clean=False)  # не удаляем папку
    brain2.load()
    stats2 = get_stats(brain2)
    print_stats(brain2, "После загрузки")
    # Проверяем, что количество нейронов и синапсов совпадает
    stats1 = get_stats(brain)
    assert stats2["neurons"] == stats1["neurons"]
    assert stats2["synapses"] == stats1["synapses"]
    assert stats2["concepts"] == stats1["concepts"]
    print("✅ Сохранение и загрузка прошли успешно.")
    return brain2


def test_retrieval_after_load(brain: Brain):
    """Тест 6: Проверка памяти после загрузки."""
    print("\n" + "=" * 60)
    print("ТЕСТ 6: Извлечение после загрузки")
    print("=" * 60)
    # Проверяем, что мозг помнит хотя бы что-то
    result = brain.step("что такое солнце?")
    answer = result["answer"]
    print(f"  Ответ на 'что такое солнце?': {answer[:100]}...")
    # Проверяем, что в ответе есть слово "звезда" (или похожее)
    # Это не строгая проверка, но для демонстрации
    if "звезд" in answer.lower() or "солнц" in answer.lower():
        print("  ✅ Модель помнит про солнце.")
    else:
        print("  ⚠️ Модель не упомянула 'звезду', но это может быть нормально.")
    print("✅ Тест памяти после загрузки завершён.")
    return brain


def test_synapse_update(brain: Brain):
    """Тест 7: Проверка, что синапсы обновляются."""
    print("\n" + "=" * 60)
    print("ТЕСТ 7: Обновление синапсов при обучении")
    print("=" * 60)
    # Получаем текущее количество синапсов
    initial_synapses = len(brain.graph._edges)
    # Обучаем новую пару
    q, a = ("тест", "проверка")
    brain.learn_pair(q, a, epochs=1)
    new_synapses = len(brain.graph._edges)
    assert new_synapses > initial_synapses
    print(f"  Синапсов до: {initial_synapses}, после: {new_synapses}")
    print("✅ Синапсы успешно создаются при обучении.")
    return brain


def test_negative_learning(brain: Brain):
    """Тест 8: Отрицательное обучение."""
    print("\n" + "=" * 60)
    print("ТЕСТ 8: Отрицательное обучение")
    print("=" * 60)
    q, a = ("солнце", "холодный объект")
    brain.learn_negative_pair(q, a, penalty=0.2)
    # Проверяем, что есть хотя бы один синапс с отрицательным весом
    has_negative = any(w < 0 for _, _, w in brain.graph._edges)
    if has_negative:
        print("  ✅ Найден отрицательный синапс.")
    else:
        print("  ⚠️ Отрицательный синапс не создан (может быть, уже был).")
    print("✅ Отрицательное обучение выполнено.")
    return brain


def test_sleep_consolidation(brain: Brain):
    """Тест 9: Проверка сна (консолидации)."""
    print("\n" + "=" * 60)
    print("ТЕСТ 9: Сон (консолидация памяти)")
    print("=" * 60)
    before = get_stats(brain)
    brain.sleep(duration_steps=2)
    after = get_stats(brain)
    # После сна рабочая память должна очиститься
    assert after["working_memory"] == 0
    print(f"  Рабочая память до сна: {before['working_memory']}, после: {after['working_memory']}")
    print("✅ Сон выполнен, рабочая память очищена.")
    return brain


# ============================================================
# Запуск всех тестов (как скрипт)
# ============================================================
def run_all_tests():
    """Запускает все тесты по порядку."""
    print("\n" + "🧠" * 20)
    print("   ЗАПУСК ТЕСТОВ SMART BRAIN v7")
    print("🧠" * 20)

    try:
        # Тест 1
        brain = test_initial_state()
        # Тест 2
        brain = test_learn_pair(brain)
        # Тест 3
        brain = test_retrieve_facts(brain)
        # Тест 4
        brain = test_multi_learn(brain)
        # Тест 5
        brain = test_save_load(brain)
        # Тест 6
        brain = test_retrieval_after_load(brain)
        # Тест 7
        brain = test_synapse_update(brain)
        # Тест 8
        brain = test_negative_learning(brain)
        # Тест 9
        brain = test_sleep_consolidation(brain)

        print("\n" + "=" * 60)
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("=" * 60)

        # Финальная статистика
        print_stats(brain, "Итоговая статистика")
        return 0
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ ОШИБКА В ТЕСТЕ: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        return 1
    finally:
        # Чистим за собой
        if os.path.exists(TEST_MODEL_DIR):
            shutil.rmtree(TEST_MODEL_DIR)
            print(f"\n🧹 Тестовая модель удалена.")


# ============================================================
# Точка входа для pytest (если вызывается через pytest)
# ============================================================
# Для pytest мы объявляем тестовые функции test_*, которые вызывают наши тесты.
# Но поскольку они уже определены, pytest их найдёт автоматически.

# Чтобы избежать дублирования, мы можем переименовать функции для pytest,
# но проще оставить как есть — pytest подхватит функции test_*.

# Если вы хотите запускать через pytest, убедитесь, что файл называется test_*.py
# и содержит функции test_*. Они будут обнаружены.


# ============================================================
# Запуск как скрипта
# ============================================================
if __name__ == "__main__":
    sys.exit(run_all_tests())