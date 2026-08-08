import openai
import json
import time

client = openai.OpenAI(
    base_url="http://localhost:1234/v1",  # укажите свой эндпоинт
    api_key="not-needed"
)

def generate_simple_qa(topic, count, batch_size=80):
    all_lines = []
    for start in range(0, count, batch_size):
        this_batch = min(batch_size, count - start)
        prompt = f"""
Ты — генератор простых вопросов для изучения языка.
Тема: {topic}.
Сгенерируй ровно {this_batch} пар в формате JSONL. Вопросы простые, ответы короткие (1-5 слов).
Охвати тему всесторонне: определения, место, назначение, действия, свойства, примеры.
Не повторяйся.
Вывод: только JSONL, без лишнего текста.
Пример: {{"q": "Что мы кипятим в чайнике?", "a": "Воду"}}
"""
        response = client.chat.completions.create(
            model="local-model",  # или gpt-4, gpt-3.5-turbo
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500
        )
        text = response.choices[0].message.content
        lines = [line.strip() for line in text.split('\n') if line.strip().startswith('{')]
        all_lines.extend(lines)
        print(f"Сгенерировано {len(all_lines)}/{count}")
        time.sleep(1)
    return all_lines

# Ввод пользователя
topic = input("Введите тему: ")
count = int(input("Сколько пар сгенерировать? "))

result = generate_simple_qa(topic, count)

filename = f"{topic.replace(' ', '_')}_{count}.jsonl"
with open(filename, "w", encoding="utf-8") as f:
    for line in result:
        f.write(line + "\n")

print(f"Файл {filename} сохранён, строк: {len(result)}")