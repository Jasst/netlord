# words.py
"""
Генератор учебного корпуса Q/A для CognitiveBrain (bulk_train.py) — с учебным
планом (curriculum) вместо одной темы сплошным блоком.

ПРИНЦИП (почему раньше generate_simple_qa была не тем, что нужно графу):

1. Один topic, один сплошной блок из count пар подряд. bulk_train.py стримит
   файл построчно в том же порядке, в каком он записан — граф сначала долго
   видит только одну тему. _contrastive_loss в brain.py на каждом шаге берёт
   негативы из УЖЕ СУЩЕСТВУЮЩИХ узлов графа — пока идёт блок одной темы,
   негативы почти все из той же темы (малоинформативный контрастив), а когда
   стартует следующая тема, граф уже сильно смещён к предыдущей.
   -> Темы чередуются round-robin: один небольшой батч на тему за круг,
      а не count пар одной темы подряд.

2. Строка фильтровалась только по startswith('{'), не реальным json.loads.
   bulk_train.py читает файл через json.loads(line) без try/except — одна
   битая строка от модели убивает загрузку всего корпуса на середине.
   -> Каждая строка валидируется по-настоящему (json.loads + проверка полей
      q/a), битые пропускаются и логируются, а не летят в файл как есть.

3. "Не повторяйся" в промпте — инструкция без памяти: на новый батч модель
   не видит, что уже сгенерировала раньше. На сотнях пар начинаются
   почти дословные повторы.
   -> В промпт передаётся список уже заданных вопросов по теме (тот же
      паттерн, что asked_questions в agent.py), плюс глобальная
      дедупликация по нормализованному тексту вопроса на весь прогон
      (не только в рамках одной темы — вопрос "Что такое вода?" не должен
      повториться, даже если сгенерирован под разными темами).

Запуск:
    python words.py --topics "вода,огонь,воздух,земля" --count-per-topic 150
    python words.py --topics-file topics.txt --count-per-topic 200 --batch-size 25

Результат — один JSONL-файл с чередующимися темами, готовый напрямую для:
    python bulk_train.py --data curriculum_....jsonl --sleep-every 200
"""
import argparse
import json
import re
import time
from collections import deque
from typing import Dict, List, Tuple

import openai

client = openai.OpenAI(
    base_url="http://localhost:1234/v1",  # укажите свой эндпоинт
    api_key="not-needed"
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _parse_jsonl_batch(text: str) -> List[Dict[str, str]]:
    """Строгая валидация: реальный json.loads на каждую строку + проверка,
    что q/a — непустые строки. Битые строки просто отбрасываются вместо
    того, чтобы попасть в файл и потом уронить bulk_train.py."""
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


def generate_batch(topic: str, batch_size: int, avoid: List[str],
                    temperature: float = 0.7) -> List[Dict[str, str]]:
    avoid_block = "\n".join(f"- {q}" for q in avoid[-40:]) if avoid else "(пока нет)"
    prompt = f"""Ты — генератор простых вопросов для изучения языка.
Тема: {topic}.
Сгенерируй ровно {batch_size} пар в формате JSONL. Вопросы простые, ответы короткие (1-5 слов).
Охвати тему всесторонне: определения, место, назначение, действия, свойства, примеры.
НЕ повторяй и не перефразируй вопросы из списка ниже — они уже заданы:
{avoid_block}
Вывод: только JSONL, без лишнего текста, без markdown-разметки.
Пример: {{"q": "Что мы кипятим в чайнике?", "a": "Воду"}}
"""
    response = client.chat.completions.create(
        model="local-model",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=int(batch_size * 25),
    )
    text = response.choices[0].message.content
    return _parse_jsonl_batch(text)


def generate_curriculum(topics: List[str], count_per_topic: int, batch_size: int = 20,
                         max_attempts_per_topic: int = None) -> List[Tuple[str, Dict[str, str]]]:
    """
    Round-robin по темам: за один круг каждая ещё не заполненная тема
    получает ОДИН батч (а не count_per_topic пар подряд). Порядок элементов
    в возвращаемом списке — это и есть порядок в выходном файле, а значит
    и порядок, в котором bulk_train.py будет учить граф.
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
                print(f"[{topic}] Лимит попыток ({max_attempts_per_topic}) исчерпан, "
                      f"остановлен на {len(collected[topic])}/{count_per_topic}.")
                continue

            this_batch = min(batch_size, need)
            attempts[topic] += 1
            try:
                items = generate_batch(topic, this_batch, list(asked_per_topic[topic]))
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

            print(f"[{topic}] +{added} новых (из {len(items)} валидных JSON) -> "
                  f"{len(collected[topic])}/{count_per_topic}")

            if len(collected[topic]) < count_per_topic:
                next_remaining.append(topic)

            time.sleep(0.3)  # не долбить локальный LLM-сервер без пауз между батчами

        remaining = next_remaining

    total = sum(len(v) for v in collected.values())
    print(f"\nИтого: {total}/{count_per_topic * len(topics)} пар по {len(topics)} темам.")
    return result


def main():
    parser = argparse.ArgumentParser(description="Curriculum-генератор Q/A для bulk_train.py")
    parser.add_argument("--topics", help="Темы через запятую")
    parser.add_argument("--topics-file", help="Файл с темами, по одной на строку")
    parser.add_argument("--count-per-topic", type=int, default=100,
                         help="Сколько пар на тему (по умолчанию 100)")
    parser.add_argument("--batch-size", type=int, default=20,
                         help="Размер одного запроса к LLM — меньше = чаще проверки на повтор, "
                              "но больше вызовов API")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.topics:
        topics = [t.strip() for t in args.topics.split(",") if t.strip()]
    elif args.topics_file:
        with open(args.topics_file, "r", encoding="utf-8") as f:
            topics = [line.strip() for line in f if line.strip()]
    else:
        raw = input("Введите темы через запятую (порядок значения не имеет — темы чередуются): ")
        topics = [t.strip() for t in raw.split(",") if t.strip()]

    if not topics:
        print("Не указано ни одной темы.")
        return

    pairs = generate_curriculum(topics, args.count_per_topic, args.batch_size)

    filename = args.output or f"curriculum_{len(topics)}topics_{args.count_per_topic}each.jsonl"
    with open(filename, "w", encoding="utf-8") as f:
        for topic, item in pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nФайл {filename} сохранён, валидных пар: {len(pairs)} "
          f"(round-robin по {len(topics)} темам).")
    print(f"Дальше: python bulk_train.py --data {filename} --sleep-every 200")


if __name__ == "__main__":
    main()