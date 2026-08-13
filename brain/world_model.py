# brain/world_model.py
"""
Модель мира: факты, убеждения, гипотезы, эпизоды.
Поддерживает добавление, запрос, проверку, обновление уверенности.
"""
import time
import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import torch
import numpy as np

from brain.utils import EmbeddingProvider


@dataclass
class Fact:
    subject: str
    relation: str
    object: str
    confidence: float = 0.5
    source: str = "unknown"          # "teacher", "web", "inference", "user", "self"
    timestamp: float = field(default_factory=time.time)
    evidence: List[str] = field(default_factory=list)  # ссылки на подтверждающие эпизоды
    validity: float = 1.0            # 0..1, насколько факт ещё актуален
    is_hypothesis: bool = False

    def to_tuple(self):
        return (self.subject, self.relation, self.object, self.confidence, self.source, self.timestamp)


@dataclass
class Belief:
    """Убеждение – субъективное знание о мире или пользователе."""
    content: str
    confidence: float = 0.5
    source: str = "self"
    timestamp: float = field(default_factory=time.time)
    evidence: List[str] = field(default_factory=list)


@dataclass
class Hypothesis:
    """Гипотеза – предположение, требующее проверки."""
    content: str
    confidence: float = 0.3
    generated_from: str = ""         # какой факт/вопрос породил
    tests: List[str] = field(default_factory=list)  # предложенные проверки
    verified: bool = False
    result: Optional[str] = None     # "confirmed", "rejected", "uncertain"
    timestamp: float = field(default_factory=time.time)


@dataclass
class Episode:
    """Эпизод – конкретный опыт."""
    text: str
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5
    emotions: Optional[Dict] = None
    context: Dict = field(default_factory=dict)


class WorldModel:
    def __init__(self, embedder: EmbeddingProvider, dim: int = 384, capacity: int = 10000):
        self.embedder = embedder
        self.dim = dim
        self.capacity = capacity

        self.facts: List[Fact] = []          # подтверждённые факты
        self.beliefs: List[Belief] = []      # субъективные убеждения
        self.hypotheses: List[Hypothesis] = []  # активные гипотезы
        self.episodes: List[Episode] = []    # эпизодическая память (расширенная)
        self._fact_index: Dict[str, List[int]] = {}  # subject -> индексы

    def add_fact(self, subj: str, rel: str, obj: str, confidence: float = 0.5,
                 source: str = "unknown", evidence: List[str] = None,
                 validity: float = 1.0, is_hypothesis: bool = False) -> int:
        """Добавить факт. Если факт уже существует – обновить уверенность."""
        # Проверка на дубликат
        for i, f in enumerate(self.facts):
            if f.subject == subj and f.relation == rel and f.object == obj:
                # Обновить уверенность (взвешенное среднее)
                new_conf = (f.confidence * 0.7 + confidence * 0.3)
                self.facts[i].confidence = min(1.0, new_conf)
                if evidence:
                    self.facts[i].evidence.extend(evidence)
                self.facts[i].timestamp = time.time()
                self.facts[i].validity = validity
                return i
        fact = Fact(subj, rel, obj, confidence, source, time.time(),
                    evidence or [], validity, is_hypothesis)
        self.facts.append(fact)
        self._fact_index.setdefault(subj, []).append(len(self.facts)-1)
        # Ограничение по ёмкости
        if len(self.facts) > self.capacity:
            self._prune_oldest()
        return len(self.facts) - 1

    def add_belief(self, content: str, confidence: float = 0.5, source: str = "self",
                   evidence: List[str] = None) -> int:
        belief = Belief(content, confidence, source, time.time(), evidence or [])
        self.beliefs.append(belief)
        return len(self.beliefs) - 1

    def add_hypothesis(self, content: str, confidence: float = 0.3,
                       generated_from: str = "", tests: List[str] = None) -> int:
        hyp = Hypothesis(content, confidence, generated_from, tests or [], False, None, time.time())
        self.hypotheses.append(hyp)
        return len(self.hypotheses) - 1

    def add_episode(self, text: str, importance: float = 0.5, emotions: Dict = None,
                    context: Dict = None) -> int:
        ep = Episode(text, time.time(), importance, emotions or {}, context or {})
        self.episodes.append(ep)
        return len(self.episodes) - 1

    def get_facts(self, subject: str = None, relation: str = None, object: str = None,
                  min_confidence: float = 0.0) -> List[Fact]:
        """Поиск фактов по фильтрам."""
        results = []
        for f in self.facts:
            if f.confidence < min_confidence:
                continue
            if subject is not None and f.subject != subject:
                continue
            if relation is not None and f.relation != relation:
                continue
            if object is not None and f.object != object:
                continue
            results.append(f)
        return results

    def query_by_text(self, text: str, k: int = 5) -> List[Fact]:
        """Векторный поиск фактов по тексту (использует эмбеддинги)."""
        if not self.facts:
            return []
        emb = self.embedder.get_embedding(text, is_query=True).numpy()
        # Кэшируем эмбеддинги фактов (упрощённо: вычисляем на лету)
        facts_texts = [f"{f.subject} {f.relation} {f.object}" for f in self.facts]
        # Если фактов много, можно использовать FAISS, но для простоты – линейный поиск
        import torch.nn.functional as F
        emb_t = torch.tensor(emb)
        sims = []
        for i, ft in enumerate(facts_texts):
            f_emb = self.embedder.get_embedding(ft, is_query=False)
            sim = F.cosine_similarity(emb_t.unsqueeze(0), f_emb.unsqueeze(0)).item()
            sims.append((sim, i))
        sims.sort(reverse=True)
        return [self.facts[i] for _, i in sims[:k]]

    def verify_hypothesis(self, hyp_id: int, result: str, confidence_delta: float = 0.2):
        """Обновить гипотезу на основе результата проверки."""
        if hyp_id >= len(self.hypotheses):
            return
        hyp = self.hypotheses[hyp_id]
        hyp.verified = True
        hyp.result = result
        if result == "confirmed":
            hyp.confidence = min(1.0, hyp.confidence + confidence_delta)
            # Превратить в факт, если уверенность > 0.7
            if hyp.confidence > 0.7:
                # Парсим гипотезу в тройку? Для простоты – добавляем как факт с текстом
                self.add_fact("hypothesis", "confirmed", hyp.content, confidence=hyp.confidence,
                              source="verification", evidence=["verified_hypothesis"])
        elif result == "rejected":
            hyp.confidence = max(0.0, hyp.confidence - confidence_delta)
        # Удаляем старые гипотезы (оставляем последние 50)
        if len(self.hypotheses) > 50:
            self.hypotheses = self.hypotheses[-50:]

    def _prune_oldest(self):
        """Удалить самые старые и наименее важные факты."""
        self.facts.sort(key=lambda f: (f.validity * f.confidence, f.timestamp))
        self.facts = self.facts[-self.capacity:]

    def get_stats(self) -> Dict:
        return {
            "facts": len(self.facts),
            "beliefs": len(self.beliefs),
            "hypotheses": len(self.hypotheses),
            "episodes": len(self.episodes),
        }