from app import generate_training_pairs
from smart_brain_v6 import BrainConfig, Brain
from openai import OpenAI

# Инициализируйте клиент (как в app.py)
llm_client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="not-needed")
brain = Brain(config=BrainConfig())

pairs = generate_training_pairs("искусственный интеллект", num_pairs=100, temperature=0.8)

# Сохраните в файл
with open("ai_facts_generated.txt", "w", encoding="utf-8") as f:
    for q, a in pairs:
        f.write(f"{q}|{a}\n")