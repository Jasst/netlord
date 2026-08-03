import random
import time
import openai
from main import Brain  # предполагается, что main.py лежит рядом

# Настройка API для LM Studio (OpenAI-совместимый)
openai.api_base = "http://192.168.0.13:1234/v1"
openai.api_key = "not-needed"  # LM Studio не требует ключа


class Teacher:
    def __init__(self, brain: Brain):
        self.brain = brain
        self.system_prompt = (
            "Ты — учитель, помогающий улучшить ассоциативную сеть. "
            "Тебе дают вопрос и ответ, который сгенерировала нейросеть. "
            "Твоя задача: предложить более точный, полный или естественный ответ на этот вопрос. "
            "Ответ должен быть кратким (1–3 слова или короткая фраза), "
            "чтобы его можно было использовать как ассоциативную связь для обучения сети. "
            "Если ответ уже хорош, просто подтверди его (повтори)."
        )

    def ask_llm(self, question: str, brain_answer: str) -> str:
        """Отправляет запрос к LLM и получает улучшенный ответ."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Вопрос: {question}\nОтвет сети: {brain_answer}\nУлучшенный ответ:"}
        ]
        try:
            response = openai.ChatCompletion.create(
                model="local-model",  # любое имя, LM Studio игнорирует
                messages=messages,
                max_tokens=50,
                temperature=0.7,
            )
            improved = response.choices[0].message.content.strip()
            return improved
        except Exception as e:
            print(f"Ошибка при запросе к LLM: {e}")
            return brain_answer  # при ошибке оставляем исходный ответ

    def learn_from_question(self, question: str, temperature: float = None):
        """Обучает Brain на основе вопроса, используя LLM для улучшения."""
        if temperature is None:
            temperature = random.uniform(0.2, 1.0)  # автоматическая температура
        # Генерируем ответ Brain
        result = self.brain.generate_answer(question, temperature=temperature, mode='chain')

        if not result["known"]:
            print(f"Brain не знает '{question}'. Просим LLM дать ответ с нуля.")
            brain_answer_text = "неизвестно"
        else:
            brain_answer_text = result["text"]
            print(f"Brain ответил: {brain_answer_text}")

        # Получаем улучшенный ответ от LLM
        improved = self.ask_llm(question, brain_answer_text)
        print(f"LLM предложил: {improved}")

        # Если улучшенный ответ не пуст и не совпадает с вопросом, обучаем
        if improved and improved.lower() != question.lower():
            self.brain.learn_pair(question, improved)
        else:
            print("Ответ не изменился или пуст, пропускаем обучение.")

    def auto_learn_loop(self, num_iterations=50, questions=None):
        """Автоматический цикл обучения на списке вопросов."""
        if questions is None:
            # Базовый список тем (можно расширить)
            topics = [
                "солнце", "луна", "звезда", "планета", "вода", "огонь", "воздух",
                "земля", "лес", "дерево", "цветок", "птица", "рыба", "зверь",
                "дом", "комната", "дверь", "окно", "стол", "стул", "кровать",
                "книга", "ручка", "машина", "поезд", "самолет", "город", "деревня",
                "утро", "день", "вечер", "ночь", "час", "минута", "секунда"
            ]
            questions = topics * 3  # повторяем для большего числа итераций
        random.shuffle(questions)
        for i in range(num_iterations):
            q = random.choice(questions)
            print(f"\n--- Итерация {i + 1}/{num_iterations}, вопрос: {q} ---")
            self.learn_from_question(q)
            # Сохраняем модель каждые 5 итераций
            if (i + 1) % 5 == 0:
                self.brain.save()
                print("Модель сохранена.")
            time.sleep(0.5)  # пауза, чтобы не перегружать LLM


def main():
    # Загружаем или создаём Brain (используем тот же путь, что и в main.py)
    MODEL_PATH = "brain_model_trained.json"
    brain = Brain(dim_embedding=64, input_neurons=30, output_neurons=30,
                  hidden_neurons=140, model_path=MODEL_PATH)
    brain.load(MODEL_PATH)  # если файла нет, создаст новую модель

    teacher = Teacher(brain)
    print("Запуск автоматического обучения с учителем (LLM).")
    teacher.auto_learn_loop(num_iterations=50)

    brain.save()
    print("Обучение завершено, модель сохранена.")


if __name__ == "__main__":
    main()