# main.py
import torch
import random
import re
from brain import CognitiveBrain, BrainConfig
from brain.teacher import Teacher

class TeacherSimple:
    def evaluate(self, question: str, answer: str) -> tuple:
        score = random.uniform(0.3, 0.9)
        improved = answer
        if score < 0.5:
            improved = "Я уточню: " + answer
        return score, improved, {"relevance": score, "accuracy": score}

def interactive_mode(brain: CognitiveBrain):
    teacher = TeacherSimple()
    print("Smart Brain v9 – Когнитивный агент с планированием, рефлексией и любопытством")
    print("Команды: learn <вопрос> => <ответ>, stats, exit")
    while True:
        user_input = input("> ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input.lower() == "stats":
            stats = brain.get_stats()
            print(f"Нейронов: {stats['neurons']}")
            print(f"Синансов: {stats['synapses']}")
            print(f"Шаг: {brain.step_counter}")
            continue
        if user_input.lower().startswith("learn "):
            match = re.match(r"learn\s+(.+?)\s*=>\s*(.+)", user_input, re.IGNORECASE)
            if match:
                q, a = match.group(1).strip(), match.group(2).strip()
                brain.learn_pair(q, a, reward=1.0)
                print("Выучено!")
                continue
        # Обычный диалог
        result = brain.step(user_input)
        answer = result["answer"]
        print(f"Brain: {answer}")
        score, improved, _ = teacher.evaluate(user_input, answer)
        print(f"Оценка: {score:.2f}")
        if improved != answer and score > 0.6:
            brain.learn_pair(user_input, improved, reward=score)
            print("Дообучено на улучшении.")

if __name__ == "__main__":
    config = BrainConfig()
    brain = CognitiveBrain(config)
    interactive_mode(brain)