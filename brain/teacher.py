# brain/teacher.py
import json
import time
import numpy as np
from typing import Dict, Tuple, Optional
from openai import OpenAI


class Teacher:
    """
    ИСПРАВЛЕНО: раньше промпт просил формат "<оценка>|<улучшенный_ответ>|details: ..."
    и парсился через re.search(r'(\\d+\\.?\\d*)', raw) — первое попавшееся число в тексте.
    Локальные 7B-модели нередко "эхом" повторяют часть инструкции (например "1. Релевантность"),
    из-за чего в score попадало случайное число вместо реальной оценки — это давало
    зашумлённый/однообразный сигнал обучения и способствовало закреплению повторяющихся ответов.

    Теперь модель обязана вернуть строгий JSON — парсинг детерминированный, с fallback при сбое.
    """
    SYSTEM_PROMPT = (
        "Ты — критический эксперт-оценщик ответов ИИ. "
        "Верни ТОЛЬКО валидный JSON, без пояснений, без markdown-разметки, в точности такого вида:\n"
        '{"relevance": 0.0, "accuracy": 0.0, "completeness": 0.0, "clarity": 0.0, '
        '"improved_answer": "..."}\n'
        "Все оценки — числа от 0 до 1. improved_answer — исправленная/улучшенная версия ответа "
        "(если ответ уже хорош, верни его без изменений)."
    )

    def __init__(self, llm_client: Optional[OpenAI] = None):
        self.llm_client = llm_client
        self.evaluation_history = []

    def evaluate(self, question: str, brain_answer: str) -> Tuple[float, str, Dict]:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Вопрос: {question}\nОтвет: {brain_answer}"}
        ]
        default_details = {"relevance": 0.5, "accuracy": 0.5, "completeness": 0.5, "clarity": 0.5}
        try:
            response = self.llm_client.chat.completions.create(
                model="local-model",
                messages=messages,
                max_tokens=250,
                temperature=0.1,
            )
            raw = response.choices[0].message.content.strip()
            data = self._parse_json(raw)
            if data is None:
                score, improved, details = 0.5, brain_answer, default_details
            else:
                details = {
                    k: self._clamp(float(data.get(k, 0.5)))
                    for k in ("relevance", "accuracy", "completeness", "clarity")
                }
                score = self._clamp(sum(details.values()) / 4)
                improved = data.get("improved_answer") or brain_answer
                if not isinstance(improved, str) or len(improved.strip()) < 2:
                    improved = brain_answer

            self.evaluation_history.append({
                "question": question,
                "score": score,
                "details": details,
                "timestamp": time.time(),
            })
            return score, improved, details
        except Exception as e:
            print(f"Teacher error: {e}")
            return 0.5, brain_answer, default_details

    @staticmethod
    def _clamp(v: float) -> float:
        return max(0.0, min(1.0, v))

    @staticmethod
    def _parse_json(raw: str) -> Optional[Dict]:
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        # На случай, если модель добавила текст до/после JSON — вырезаем первый {...} блок
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

    def get_evaluation_stats(self) -> Dict:
        if not self.evaluation_history:
            return {}
        scores = [e["score"] for e in self.evaluation_history]
        return {
            "total_evaluations": len(self.evaluation_history),
            "mean_score": float(np.mean(scores)),
            "recent_mean": float(np.mean(scores[-10:])) if len(scores) >= 10 else float(np.mean(scores)),
        }
