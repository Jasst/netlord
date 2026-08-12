# words_v2.py
"""
Улучшенный генератор учебного корпуса для CognitiveBrain.
Генерирует не только факты, но и сравнения, причины, обобщения.

Запуск:
    python words_v2.py --topics "физика,химия,биология" --count-per-topic 100

Формат выхода: JSONL, совместим с bulk_train.py
"""

import argparse
import json
import re
import time
import random
from collections import deque
from typing import List, Dict, Tuple, Optional

import openai

client = openai.OpenAI(
    base_url="http://localhost:1234/v1",  # укажите свой эндпоинт
    api_key="not-needed"
)

# Типы вопросов
QUESTION_TYPES = [
    "fact",        # определение, свойство
    "compare",     # сравнение двух понятий
    "cause",       # причина или следствие
    "generalize",  # обобщение, что общего
]

# Шаблоны для каждого типа (чтобы модель лучше понимала, что от неё требуется)
TYPE_PROMPTS = {
    "fact": "Сгенерируй простой вопрос о '{topic}', ответ — краткое определение или свойство.",
    "compare": "Сравни понятия '{topic1}' и '{topic2}'. Сформулируй вопрос и дай ответ, указывающий на сходства или различия.",
    "cause": "Почему происходит '{topic}'? Сформулируй вопрос и дай краткий ответ о причине.",
    "generalize": "Что общего у '{topic1}' и '{topic2}'? Сформулируй вопрос и дай обобщающий ответ.",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _parse_jsonl_batch(text: str) -> List[Dict[str, str]]:
    """Парсит JSONL из ответа модели, отбрасывая битые строки."""
    items = []
    for line in text.split("\n"):
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        q, a = obj.get("q"), obj.get("a")
        if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
            items.append({"q": q.strip(), "a": a.strip()})
    return items


def generate_batch_v2(
    topic: str,
    batch_size: int,
    avoid: List[str],
    temperature: float = 0.7,
    sub_topics: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """
    Генерирует batch вопросов разных типов для одной темы (или пары тем).
    """
    # Если есть подтемы, используем их для сравнений и обобщений
    if sub_topics and len(sub_topics) >= 2:
        # Берём две случайные подтемы для сравнения и обобщения
        t1, t2 = random.sample(sub_topics, 2)
    else:
        t1 = t2 = topic

    avoid_block = "\n".join(f"- {q}" for q in avoid[-40:]) if avoid else "(пока нет)"

    # Строим промпт с несколькими заданиями
    # Запрашиваем у модели сгенерировать по одному вопросу каждого типа
    # Чтобы получить ровно batch_size пар, просим batch_size/4 каждого типа (с округлением)
    per_type = max(1, batch_size // len(QUESTION_TYPES))
    # Но чтобы точно получить batch_size, используем цикл с разными типами
    types_cycle = QUESTION_TYPES * ( (batch_size // len(QUESTION_TYPES)) + 1)
    types_cycle = types_cycle[:batch_size]

    # Строим инструкцию для каждого типа
    instructions = []
    for i, qtype in enumerate(types_cycle):
        if qtype == "fact":
            instructions.append(f"{i+1}. Факт: задай вопрос о '{topic}', ответ — определение или свойство.")
        elif qtype == "compare":
            instructions.append(f"{i+1}. Сравнение: сравни '{t1}' и '{t2}', задай вопрос и ответ.")
        elif qtype == "cause":
            instructions.append(f"{i+1}. Причина: почему происходит '{topic}'? задай вопрос и краткий ответ.")
        elif qtype == "generalize":
            instructions.append(f"{i+1}. Обобщение: что общего у '{t1}' и '{t2}'? задай вопрос и ответ.")
        else:
            instructions.append(f"{i+1}. Факт: задай вопрос о '{topic}', ответ — определение.")

    instructions_text = "\n".join(instructions)

    prompt = f"""Ты — генератор обучающих данных для когнитивного графа.
Тема: {topic}.
Подтемы (для сравнений и обобщений): {', '.join(sub_topics) if sub_topics else topic}.

Сгенерируй ровно {batch_size} пар 'вопрос|ответ' в формате JSONL (каждая пара на отдельной строке, вида {{"q": "...", "a": "..."}}).
Для каждой строки следуй указанному типу:

{instructions_text}

Важно: вопросы должны быть разнообразными, не повторяться, ответы — краткими (1-5 слов).
НЕ повторяй и не перефразируй уже заданные вопросы из списка:
{avoid_block}

Вывод: только JSONL, без лишнего текста, без markdown.
"""

    response = client.chat.completions.create(
        model="local-model",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=int(batch_size * 30),
    )
    text = response.choices[0].message.content
    return _parse_jsonl_batch(text)


def generate_curriculum_v2(
    topics: List[str],
    count_per_topic: int,
    batch_size: int = 20,
    max_attempts_per_topic: int = None,
    sub_topics: Optional[Dict[str, List[str]]] = None,
) -> List[Tuple[str, Dict[str, str]]]:
    """
    Генерирует корпус с чередованием тем и типов вопросов.
    sub_topics: словарь {тема: [подтемы]} для сравнений и обобщений.
    """
    if max_attempts_per_topic is None:
        max_attempts_per_topic = max(3, (count_per_topic // batch_size) * 3)

    seen_global: set = set()
    asked_per_topic: Dict[str, deque] = {t: deque(maxlen=200) for t in topics}
    collected: Dict[str, List[Dict[str, str]]] = {t: [] for t in topics}
    attempts: Dict[str, int] = {t: 0 for t in topics}

    result: List[Tuple[str, Dict[str, str]]] = []
    remaining = list(topics)

    while remaining:
        next_remaining = []
        for topic in remaining:
            need = count_per_topic - len(collected[topic])
            if need <= 0:
                continue
            if attempts[topic] >= max_attempts_per_topic:
                print(f"[{topic}] Лимит попыток, остановлен на {len(collected[topic])}/{count_per_topic}.")
                continue

            this_batch = min(batch_size, need)
            attempts[topic] += 1
            try:
                sub = sub_topics.get(topic) if sub_topics else None
                items = generate_batch_v2(topic, this_batch, list(asked_per_topic[topic]), sub_topics=sub)
            except Exception as e:
                print(f"[{topic}] Ошибка генерации: {e}")
                items = []

            added = 0
            for item in items:
                norm_q = _normalize(item["q"])
                if norm_q in seen_global:
                    continue
                seen_global.add(norm_q)
                asked_per_topic[topic].append(item["q"])
                collected[topic].append(item)
                result.append((topic, item))
                added += 1
                if len(collected[topic]) >= count_per_topic:
                    break

            print(f"[{topic}] +{added} новых (из {len(items)} валидных) -> {len(collected[topic])}/{count_per_topic}")

            if len(collected[topic]) < count_per_topic:
                next_remaining.append(topic)

            time.sleep(0.3)  # пауза между батчами

        remaining = next_remaining

    total = sum(len(v) for v in collected.values())
    print(f"\nИтого: {total}/{count_per_topic * len(topics)} пар по {len(topics)} темам.")
    return result


def main():
    parser = argparse.ArgumentParser(description="Улучшенный генератор Q/A для CognitiveBrain")
    parser.add_argument("--topics", help="Темы через запятую")
    parser.add_argument("--topics-file", help="Файл с темами, по одной на строку")
    parser.add_argument("--sub-topics-file", help="Файл с подтемами в формате тема: подтема1, подтема2, ...")
    parser.add_argument("--count-per-topic", type=int, default=100, help="Сколько пар на тему")
    parser.add_argument("--batch-size", type=int, default=20, help="Размер батча")
    parser.add_argument("--output", default=None, help="Выходной JSONL-файл")
    args = parser.parse_args()

    # Загружаем темы
    if args.topics:
        topics = [t.strip() for t in args.topics.split(",") if t.strip()]
    elif args.topics_file:
        with open(args.topics_file, "r", encoding="utf-8") as f:
            topics = [line.strip() for line in f if line.strip()]
    else:
        raw = input("Введите темы через запятую: ")
        topics = [t.strip() for t in raw.split(",") if t.strip()]

    if not topics:
        print("Нет тем.")
        return

    # Загружаем подтемы (если есть)
    sub_topics = {}
    if args.sub_topics_file:
        with open(args.sub_topics_file, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    main_t, subs = line.split(":", 1)
                    main_t = main_t.strip()
                    sub_list = [s.strip() for s in subs.split(",") if s.strip()]
                    if main_t in topics and sub_list:
                        sub_topics[main_t] = sub_list

    # Генерируем корпус
    pairs = generate_curriculum_v2(
        topics,
        args.count_per_topic,
        args.batch_size,
        sub_topics=sub_topics if sub_topics else None,
    )

    filename = args.output or f"curriculum_v2_{len(topics)}topics.jsonl"
    with open(filename, "w", encoding="utf-8") as f:
        for topic, item in pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nФайл {filename} сохранён, пар: {len(pairs)}.")
    print(f"Дальше: python bulk_train.py --data {filename} --sleep-every 200")


if __name__ == "__main__":
    main()