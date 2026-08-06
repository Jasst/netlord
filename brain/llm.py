# brain/llm.py
from openai import OpenAI

class LMStudioLLM:
    def __init__(self, base_url: str = "http://127.0.0.1:1234/v1", api_key: str = "not-needed"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        """Не потоковый метод (для совместимости)"""
        try:
            response = self.client.chat.completions.create(
                model="local-model",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Ошибка LM Studio: {e}")
            return "Извините, LM Studio недоступна."

    def generate_stream(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7):
        """
        Генерирует ответ по токенам (генератор).
        Возвращает итератор строк (chunks).
        """
        try:
            response = self.client.chat.completions.create(
                model="local-model",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,  # <-- потоковый режим
            )
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            print(f"Ошибка LM Studio stream: {e}")
            yield "Извините, LM Studio недоступна."