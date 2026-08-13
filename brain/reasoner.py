# brain/reasoner.py
"""
Логический вывод на основе фактов и правил.
Поддерживает транзитивность, проверку противоречий.
"""
from typing import List, Tuple, Optional, Set
import itertools
from brain.world_model import WorldModel, Fact


class Reasoner:
    def __init__(self, world_model: WorldModel):
        self.world = world_model

    def transitive_closure(self, relation: str = "implies") -> List[Fact]:
        """
        Построить транзитивное замыкание для заданного отношения.
        Например, если A implies B и B implies C, то добавляется A implies C.
        Возвращает список новых выведенных фактов (гипотез).
        """
        # Получаем все факты с данным отношением
        facts = self.world.get_facts(relation=relation, min_confidence=0.3)
        if not facts:
            return []

        # Строим граф: subject -> (object, confidence, fact_id)
        graph = {}
        for f in facts:
            graph.setdefault(f.subject, []).append((f.object, f.confidence, f))

        new_facts = []
        # Ищем пути длины 2 и более
        for subj, targets in graph.items():
            for obj1, conf1, f1 in targets:
                if obj1 in graph:
                    for obj2, conf2, f2 in graph[obj1]:
                        if subj != obj2:  # избегаем циклов
                            new_conf = conf1 * conf2  # произведение уверенностей
                            # Проверяем, есть ли уже такой факт
                            existing = self.world.get_facts(subject=subj, relation=relation, object=obj2)
                            if not existing:
                                new_fact = Fact(
                                    subject=subj,
                                    relation=relation,
                                    object=obj2,
                                    confidence=new_conf,
                                    source="inference",
                                    evidence=[f"transitive from {f1.subject}->{f1.object} and {f2.subject}->{f2.object}"],
                                    validity=min(f1.validity, f2.validity),
                                    is_hypothesis=True
                                )
                                new_facts.append(new_fact)
        return new_facts

    def check_contradictions(self, min_confidence: float = 0.5) -> List[Tuple[Fact, Fact]]:
        """
        Найти противоречия: например, A -> B и A -> not B.
        Возвращает пары противоречащих фактов.
        """
        facts = self.world.get_facts(min_confidence=min_confidence)
        contradictions = []
        for i, f1 in enumerate(facts):
            for f2 in facts[i+1:]:
                if f1.subject == f2.subject and f1.relation == f2.relation and f1.object != f2.object:
                    # Если есть отношение "contradicts" или похожее
                    contradictions.append((f1, f2))
        return contradictions

    def infer_by_pattern(self, pattern: Tuple[str, str, str]) -> List[Fact]:
        """
        Вывести новые факты на основе шаблона: (subj_pattern, rel_pattern, obj_pattern).
        Например, ("X", "is_a", "Y") и ("Y", "has_property", "Z") => ("X", "has_property", "Z").
        """
        # Упрощённая реализация: ищем все факты, удовлетворяющие паттерну, и строим вывод
        # Для реального использования нужен более гибкий механизм
        return []

    def update_world_with_inferences(self):
        """Выполнить все возможные выводы и добавить их в модель мира как гипотезы."""
        inferred = self.transitive_closure("implies")
        for f in inferred:
            self.world.add_fact(
                f.subject, f.relation, f.object, confidence=f.confidence,
                source=f.source, evidence=f.evidence, validity=f.validity,
                is_hypothesis=True
            )
        # Проверяем противоречия и корректируем уверенность
        contradictions = self.check_contradictions()
        for f1, f2 in contradictions:
            # Понижаем уверенность обоих фактов
            f1.confidence *= 0.9
            f2.confidence *= 0.9