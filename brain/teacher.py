# brain/teacher.py
import re
import time
import numpy as np
from typing import Dict, Tuple, Optional
from openai import OpenAI

class Teacher:
    def __init__(self, llm_client: Optional[OpenAI] = None):
        self.llm_client = llm_client
        self.evaluation_history = []

    def evaluate(self, question: str, brain_answer: str) -> Tuple[float, str, Dict]:
        system_prompt = (
            "Ты — критический эксперт. Оцени ответ по критериям (0-1 каждый):\n"
            "1. Релевантность\n2. Точность\n3. Полнота\n4. Ясность\n"
            "Формат: <средняя_оценка>|<улучшенный_ответ>|\n"
            "details: relevance=X, accuracy=Y, completeness=Z, clarity=W"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Вопрос: {question}\nОтвет: {brain_answer}\nОценка:"}
        ]
        try:
            response = self.llm_client.chat.completions.create(
                model="local-model",
                messages=messages,
                max_tokens=150,
                temperature=0.2,
            )
            raw = response.choices[0].message.content.strip()
            score = 0.5
            improved = brain_answer
            details = {"relevance": 0.5, "accuracy": 0.5, "completeness": 0.5, "clarity": 0.5}
            match = re.search(r'(\d+\.?\d*)', raw)
            if match:
                score = float(match.group(1))
                rest = raw.replace(match.group(1), '').strip()
                if rest.startswith('|'):
                    rest = rest[1:].strip()
                parts = rest.split('|')
                if len(parts) >= 1 and parts[0] and len(parts[0]) > 2:
                    improved = parts[0]
                if len(parts) >= 2:
                    details_str = parts[1]
                    for aspect in ["relevance", "accuracy", "completeness", "clarity"]:
                        m = re.search(rf'{aspect}=(\d+\.?\d*)', details_str, re.IGNORECASE)
                        if m:
                            details[aspect] = float(m.group(1))
            score = max(0.0, min(1.0, score))
            self.evaluation_history.append({
                "question": question,
                "score": score,
                "details": details,
                "timestamp": time.time(),
            })
            return score, improved, details
        except Exception as e:
            print(f"Teacher error: {e}")
            return 0.5, brain_answer, {"relevance": 0.5, "accuracy": 0.5, "completeness": 0.5, "clarity": 0.5}

    def get_evaluation_stats(self) -> Dict:
        if not self.evaluation_history:
            return {}
        scores = [e["score"] for e in self.evaluation_history]
        return {
            "total_evaluations": len(self.evaluation_history),
            "mean_score": float(np.mean(scores)),
            "recent_mean": float(np.mean(scores[-10:])) if len(scores) >= 10 else float(np.mean(scores)),
        }