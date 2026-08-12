# bulk_train.py (улучшенная версия с валидацией)
"""
Массовая загрузка корпуса Q/A в CognitiveBrain с возможностью валидации через Teacher.
"""
import argparse
import json
import time
import random
from brain import CognitiveBrain, BrainConfig
from brain.teacher import Teacher
from openai import OpenAI


def load_pairs_with_validation(
    path: str,
    teacher: Teacher,
    threshold: float = 0.6,
    max_retries: int = 1,
    sample_rate: float = 1.0,
    verbose: bool = False,
):
    """
    Загружает пары из JSONL, валидирует их через Teacher.
    Если оценка < threshold, пытается улучшить (до max_retries раз).
    Возвращает только валидные пары (оценка >= threshold).
    """
    skipped = 0
    validated = 0
    improved_count = 0
    rejected_count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            # Парсим JSON
            try:
                item = json.loads(line)
                q, a = item["q"], item["a"]
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                skipped += 1
                if verbose:
                    print(f"[bulk_train] Пропущена строка {line_num} ({e}): {line[:80]!r}")
                continue

            # Сэмплирование (если sample_rate < 1.0)
            if random.random() > sample_rate:
                continue

            # Валидация с улучшением
            valid = False
            current_q, current_a = q, a
            score = 0.0
            for attempt in range(max_retries + 1):
                score, improved, _ = teacher.evaluate(current_q, current_a)
                if score >= threshold:
                    valid = True
                    if attempt > 0:
                        improved_count += 1
                        current_a = improved
                    break
                else:
                    # Если есть улучшенный ответ и он не совпадает с текущим
                    if improved != current_a and improved.strip():
                        current_a = improved
                    else:
                        # Не удалось улучшить, пробуем ещё раз (если есть попытки)
                        if attempt < max_retries:
                            continue
                        else:
                            valid = False
                            break

            if valid:
                validated += 1
                yield current_q, current_a
            else:
                rejected_count += 1
                if verbose:
                    print(f"[bulk_train] Отклонена пара: {q} -> {a} (score={score:.2f})")

    if skipped:
        print(f"[bulk_train] Всего пропущено битых строк: {skipped}")
    if rejected_count:
        print(f"[bulk_train] Отклонено пар: {rejected_count}")
    if improved_count:
        print(f"[bulk_train] Улучшено пар: {improved_count}")
    print(f"[bulk_train] Валидных пар: {validated}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Путь к JSONL с парами q/a")
    parser.add_argument("--sleep-every", type=int, default=200,
                        help="Через сколько пар вызывать brain.sleep()")
    parser.add_argument("--stats-every", type=int, default=20,
                        help="Через сколько пар печатать get_stats()")
    parser.add_argument("--reward", type=float, default=1.0)
    parser.add_argument("--model-dir", default=None,
                        help="Переопределить model_dir из конфига")
    # НОВЫЕ ПАРАМЕТРЫ
    parser.add_argument("--validate", action="store_true",
                        help="Включить валидацию пар через Teacher")
    parser.add_argument("--validation-threshold", type=float, default=0.6,
                        help="Минимальная оценка для принятия пары (0.0–1.0)")
    parser.add_argument("--validation-retries", type=int, default=1,
                        help="Сколько раз пытаться улучшить пару при валидации")
    parser.add_argument("--sample-rate", type=float, default=1.0,
                        help="Доля пар для валидации (0..1), остальные пропускаются без проверки")
    parser.add_argument("--llm-base-url", default=None,
                        help="URL для LLM (если не задан, используется из конфига)")
    args = parser.parse_args()

    config = BrainConfig()
    if args.model_dir:
        config.model_dir = args.model_dir
    if args.llm_base_url:
        config.llm_base_url = args.llm_base_url

    brain = CognitiveBrain(config)
    brain.load()
    brain.load_dialog_history()

    # Инициализируем Teacher, если нужна валидация
    teacher = None
    if args.validate:
        # Создаём клиент OpenAI (совместимый с LM Studio)
        llm_client = OpenAI(base_url=config.llm_base_url, api_key="not-needed")
        teacher = Teacher(llm_client=llm_client)
        print("[bulk_train] Валидация включена (Teacher)")

    # Генератор пар
    if args.validate and teacher is not None:
        pairs = load_pairs_with_validation(
            args.data,
            teacher,
            threshold=args.validation_threshold,
            max_retries=args.validation_retries,
            sample_rate=args.sample_rate,
            verbose=True,
        )
    else:
        # Старая загрузка (без валидации)
        def load_pairs_legacy(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        q, a = item["q"], item["a"]
                        yield q, a
                    except:
                        continue
        pairs = load_pairs_legacy(args.data)

    t0 = time.time()
    count = 0
    for q, a in pairs:
        brain.learn_pair(q, a, reward=args.reward)
        count += 1

        if count % args.stats_every == 0:
            stats = brain.get_stats()
            elapsed = time.time() - t0
            print(f"[{count}] neurons={stats['neurons']} synapses={stats['synapses']} "
                  f"concepts={stats['concepts']} elapsed={elapsed:.1f}s "
                  f"({count / max(elapsed, 1e-6):.2f} pairs/s)")

        if count % args.sleep_every == 0:
            brain.sleep()

    brain.sleep()
    brain.save()
    brain.save_dialog_history()
    print(f"Готово. Загружено {count} пар. Итоговые stats: {brain.get_stats()}")


if __name__ == "__main__":
    main()