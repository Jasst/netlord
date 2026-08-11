# bulk_train.py
"""
Массовая загрузка корпуса Q/A в CognitiveBrain.

Формат входного файла — JSONL, одна пара на строку:
    {"q": "Столица Франции?", "a": "Столица Франции — Париж."}
    {"q": "Кто написал Войну и мир?", "a": "Лев Толстой."}

Запуск:
    python bulk_train.py --data corpus.jsonl --sleep-every 200

Почему не просто "закинуть всё в цикл без остановки":
- brain.sleep() нужно вызывать периодически (не после каждой пары!) —
  именно там перестраивается иерархия (rebuild_hierarchy) и EWC-якоря.
  Слишком часто — дорого (кластеризация ~O(n^2) по числу узлов уровня).
  Слишком редко — верхние уровни иерархии долго не обновляются.
  Разумный старт: раз в 100-300 пар, подстроить по факту.
- Идёт логирование stats каждые N пар — чтобы сразу увидеть, если
  граф растёт неправильно (например: neurons почти не растёт —
  значит node_merge_threshold слишком низкий и всё мержится в одно).
"""
import argparse
import json
import time

from brain import CognitiveBrain, BrainConfig


def load_pairs(path: str):
    """
    ИСПРАВЛЕНО: раньше json.loads(line) без try/except — одна битая строка
    (например, из старого words.py без строгой валидации, или просто
    повреждённая при переносе файла) обрывала загрузку всего корпуса на
    середине, и всё, что шло после, просто не попадало в обучение молча
    (падение с трейсбеком). Теперь битая строка логируется с номером и
    пропускается, а загрузка остального файла продолжается.
    """
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                q, a = item["q"], item["a"]
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                skipped += 1
                print(f"[bulk_train] Пропущена строка {line_num} ({e}): {line[:80]!r}")
                continue
            yield q, a
    if skipped:
        print(f"[bulk_train] Всего пропущено битых строк: {skipped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Путь к JSONL с парами q/a")
    parser.add_argument("--sleep-every", type=int, default=200,
                         help="Через сколько пар вызывать brain.sleep()")
    parser.add_argument("--stats-every", type=int, default=20,
                         help="Через сколько пар печатать get_stats()")
    parser.add_argument("--reward", type=float, default=1.0)
    parser.add_argument("--model-dir", default=None,
                         help="Переопределить model_dir из конфига (напр. для отдельного эксперимента)")
    args = parser.parse_args()

    config = BrainConfig()
    if args.model_dir:
        config.model_dir = args.model_dir

    brain = CognitiveBrain(config)
    brain.load()
    brain.load_dialog_history()

    t0 = time.time()
    count = 0
    for q, a in load_pairs(args.data):
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

    # финальные консолидация + сохранение
    brain.sleep()
    brain.save()
    brain.save_dialog_history()
    print(f"Готово. Загружено {count} пар. Итоговые stats: {brain.get_stats()}")


if __name__ == "__main__":
    main()