from smart_brain_v6 import Brain, BrainConfig
import sys


def load_pairs(filename, delimiter="|"):
    pairs = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("question"):
                continue
            parts = line.split(delimiter, 1)
            if len(parts) == 2:
                q, a = parts[0].strip(), parts[1].strip()
                if q and a:
                    pairs.append((q, a))
    return pairs


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_from_file.py data.txt [delimiter]")
        sys.exit(1)
    filename = sys.argv[1]
    delimiter = sys.argv[2] if len(sys.argv) > 2 else "|"

    brain = Brain(config=BrainConfig(model_dir="brain_model_v6"))
    brain.load()

    pairs = load_pairs(filename, delimiter)
    print(f"Загружено {len(pairs)} пар.")

    for q, a in pairs:
        brain.learn_pair(q, a)

    brain.sleep(duration_steps=5)
    brain.save()
    print("Обучение завершено, модель сохранена.")