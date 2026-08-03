#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт автоматического обучения модели Brain на заданную тему.
Генерирует обучающие пары через локальный LLM (LM Studio) и тренирует модель.
После обучения модель сохраняется и может быть использована в основном скрипте.
Поддерживает интеграцию новых знаний с уже существующей сетью.
"""

import sys
import time
import random
import argparse
import numpy as np
from openai import OpenAI

# Используем улучшенную версию Brain из smart_brain_v3.py
from smart_brain_v4 import Brain, cosine_similarity

# Настройка подключения к LM Studio (должен быть запущен локально)
client = OpenAI(
    base_url="http://192.168.0.13:1234/v1",
    api_key="not-needed",
)

def generate_training_pairs(topic: str, num_pairs: int = 50, temperature: float = 0.7) -> list:
    """
    Генерирует список пар (вопрос, ответ) по заданной теме с помощью LLM.
    Возвращает список кортежей (question, answer).
    """
    system_prompt = (
        "Ты — генератор обучающих данных для нейросети. "
        "Твоя задача — создать список пар 'вопрос|ответ' на русском языке, "
        "которые помогут нейросети понять тему и её атрибуты, контекстные связи и ассоциации. "
        "Формат вывода: каждая пара на новой строке, разделена символом '|'. "
        "Не добавляй никаких пояснений, только список пар."
    )

    user_prompt = (
        f"Сгенерируй {num_pairs} пар (вопрос|ответ) по теме '{topic}'. "
        "Вопросы должны быть разнообразными: от прямых ('что такое ...') до контекстных ('как ...', 'почему ...', 'в каком случае ...'). "
        "Ответы должны быть краткими, но информативными (1-5 слов). "
        "Включай синонимы и связанные понятия, чтобы модель научилась ассоциировать разные формулировки с одним понятием. "
        "Примеры: 'солнце|звезда', 'солнечный день|ясно и жарко', 'что дает солнце|свет и тепло'."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = client.chat.completions.create(
            model="local-model",
            messages=messages,
            max_tokens=num_pairs * 20 + 100,
            temperature=temperature,
        )
        raw_text = response.choices[0].message.content.strip()
        pairs = []
        for line in raw_text.splitlines():
            line = line.strip()
            if '|' in line:
                parts = line.split('|', 1)
                if len(parts) == 2:
                    q = parts[0].strip()
                    a = parts[1].strip()
                    if q and a:
                        pairs.append((q, a))
        print(f"Сгенерировано {len(pairs)} пар.")
        return pairs
    except Exception as e:
        print(f"Ошибка при генерации данных: {e}")
        return []


def integrate_new_concept(brain: Brain, concept_text: str, top_k: int = 5):
    """
    Интегрирует новое понятие в существующую сеть:
    находит top_k наиболее похожих выходных нейронов (понятий) и создаёт
    двусторонние синапсы между ними и новым нейроном.
    """
    nid = brain.concept_index.get(concept_text.strip().lower())
    if nid is None:
        return

    emb = brain.neurons[nid].embedding
    # Ищем похожие выходные нейроны (кроме самого себя)
    similarities = []
    for other_nid, neuron in brain.neurons.items():
        if other_nid == nid:
            continue
        if neuron.cluster == 'output' and neuron.label:
            sim = cosine_similarity(emb, neuron.embedding)
            if sim > 0.3:  # порог схожести
                similarities.append((sim, other_nid))
    similarities.sort(reverse=True)
    for sim, other_nid in similarities[:top_k]:
        # Создаём взаимные связи
        syn1 = brain._create_synapse(nid, other_nid, weight=0.1 + 0.2 * sim)
        syn2 = brain._create_synapse(other_nid, nid, weight=0.1 + 0.2 * sim)
        if syn1 and syn2:
            print(f"  Интеграция: '{concept_text}' <-> '{brain.neurons[other_nid].label}' (sim={sim:.2f})")


def train_model_on_topic(brain: Brain, topic: str, num_pairs: int = 50,
                         negative_ratio: float = 0.2, temperature: float = 0.7,
                         integrate: bool = True):
    """
    Обучает модель на сгенерированных парах.
    Если integrate=True, после обучения каждой пары выполняется интеграция
    новых понятий с существующими.
    """
    print(f"\n=== Автоматическое обучение на тему '{topic}' ===\n")

    # 1. Генерация положительных пар
    print("Генерация положительных обучающих пар...")
    pairs = generate_training_pairs(topic, num_pairs, temperature)
    if not pairs:
        print("Не удалось сгенерировать пары. Проверьте работу LM Studio.")
        return

    # 2. Обучение на положительных парах с интеграцией
    print(f"Обучение на {len(pairs)} положительных парах...")
    for idx, (q, a) in enumerate(pairs, 1):
        print(f"  {idx}/{len(pairs)}: {q} -> {a}")
        brain.learn_pair(q, a)
        if integrate:
            # Интегрируем выходное понятие (a) с существующей сетью
            integrate_new_concept(brain, a, top_k=5)
            # Также можно интегрировать входное, но оно обычно не является понятием
        time.sleep(0.05)

    # 3. Генерация отрицательных пар (неправильные ответы)
    if negative_ratio > 0:
        neg_count = int(len(pairs) * negative_ratio)
        print(f"\nГенерация {neg_count} отрицательных пар для коррекции...")
        correct_answers = set(a.lower() for _, a in pairs)
        other_topics = ["луна", "звезда", "планета", "облако", "дождь", "ветер", "снег", "тепло", "холод", "ночь", "утро", "вечер"]

        neg_pairs = []
        all_answers = [a for _, a in pairs]
        for i in range(neg_count):
            q = random.choice(pairs)[0]
            correct_for_this_q = None
            for qq, aa in pairs:
                if qq == q:
                    correct_for_this_q = aa.lower()
                    break
            candidates = [t for t in other_topics if t.lower() != topic.lower() and t.lower() != correct_for_this_q]
            if not candidates:
                candidates = [a for a in all_answers if a.lower() != correct_for_this_q]
            if not candidates:
                candidates = [t for t in other_topics if t.lower() != correct_for_this_q]
            if not candidates:
                candidates = ["неизвестно"]
            wrong = random.choice(candidates)
            neg_pairs.append((q, wrong))

        for idx, (q, w) in enumerate(neg_pairs, 1):
            print(f"  Отрицательный {idx}/{len(neg_pairs)}: {q} !-> {w}")
            brain.learn_negative_pair(q, w)
            time.sleep(0.05)

    # 4. Запуск "сна" для консолидации и интеграции
    print("\nЗапуск сна для консолидации памяти...")
    # Увеличим длительность сна, чтобы укрепить новые связи
    brain.sleep(duration_steps=5)

    # 5. Сохранение модели
    brain.save()
    print(f"\nОбучение завершено. Модель сохранена в {brain._model_path}")


def main():
    parser = argparse.ArgumentParser(description="Автоматическое обучение Brain на заданную тему.")
    parser.add_argument("--topic", type=str, default="солнце",
                        help="Тема для обучения (например, 'солнце')")
    parser.add_argument("--num_pairs", type=int, default=50,
                        help="Количество генерируемых обучающих пар (от 10 до 200)")
    parser.add_argument("--negative_ratio", type=float, default=0.2,
                        help="Доля отрицательных примеров от положительных (0.0 - 1.0)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Температура генерации LLM (0.0 - 1.0)")
    parser.add_argument("--no-integrate", action="store_true",
                        help="Отключить интеграцию новых понятий с существующими")
    args = parser.parse_args()

    num_pairs = max(10, min(200, args.num_pairs))

    MODEL_PATH = "brain_model_trained.json"
    # Используем улучшенную версию Brain с регуляризацией (smart_brain_v3)
    brain = Brain(dim_embedding=128, input_neurons=40, output_neurons=40,
                  hidden_layers=[100, 80, 60], model_path=MODEL_PATH,
                  max_neurons=800, max_synapses=8000)  # ограничения
    brain.load(MODEL_PATH)

    train_model_on_topic(brain, args.topic, num_pairs, args.negative_ratio,
                         args.temperature, integrate=not args.no_integrate)

    ic = sum(1 for n in brain.neurons.values() if n.cluster == 'input')
    oc = sum(1 for n in brain.neurons.values() if n.cluster == 'output')
    hc = sum(1 for n in brain.neurons.values() if n.cluster == 'hidden')
    print(f"\nИтоговая статистика:")
    print(f"  Нейронов: {len(brain.neurons)} (input: {ic}, output/понятия: {oc}, hidden: {hc})")
    print(f"  Синапсов: {len(brain.synapses)}")
    print(f"  Выученных понятий: {len(brain.concept_index)}")

    test_question = f"расскажи про {args.topic}"
    print(f"\nТестовый запрос: '{test_question}'")
    result = brain.generate_answer(test_question, temperature=0.3, use_rag=True)
    print(f"Ответ: {result['text']}")


if __name__ == "__main__":
    main()