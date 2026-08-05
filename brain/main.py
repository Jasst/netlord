# brain/main.py
import torch
import random
import re
from brain.brain import Brain
from brain.config import BrainConfig

class Teacher:
    def __init__(self):
        pass

    def evaluate(self, question: str, answer: str) -> tuple:
        # Имитация оценки (в реальности можно использовать LLM)
        score = random.uniform(0.3, 0.9)
        improved = answer
        if score < 0.5:
            improved = "Я уточню: " + answer
        return score, improved, {"relevance": score, "accuracy": score}

def interactive_mode(brain: Brain):
    teacher = Teacher()
    print("Smart Brain v7 – дифференцируемый граф + FAISS + LM Studio")
    print("Команды: learn <вопрос> => <ответ>, stats, exit")
    while True:
        user_input = input("> ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input.lower() == "stats":
            print(f"Нейронов: {brain.graph.node_emb.shape[0]}")
            print(f"Шаг: {brain.step_counter}")
            continue
        if user_input.lower().startswith("learn "):
            match = re.match(r"learn\s+(.+?)\s*=>\s*(.+)", user_input, re.IGNORECASE)
            if match:
                q, a = match.group(1).strip(), match.group(2).strip()
                brain.learn_from_feedback(q, a, reward=1.0)
                print("Выучено!")
                continue
        # Обычный диалог
        result = brain.step(user_input)
        answer = result["answer"]
        print(f"Brain: {answer}")
        score, improved, _ = teacher.evaluate(user_input, answer)
        print(f"Оценка: {score:.2f}")
        if improved != answer and score > 0.6:
            brain.learn_from_feedback(user_input, improved, reward=score)
            print("Дообучено на улучшении.")

if __name__ == "__main__":
    config = BrainConfig()
    brain = Brain(config)
    interactive_mode(brain)