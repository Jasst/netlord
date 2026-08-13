# brain/planner.py
"""
Планировщик: цели, подцели, планы действий.
"""
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from brain.llm import LLMInterface
from brain.world_model import WorldModel


@dataclass
class Goal:
    description: str
    priority: int = 5
    subgoals: List['Goal'] = field(default_factory=list)
    plan: List['Action'] = field(default_factory=list)
    status: str = "pending"  # pending, in_progress, done, failed
    created: float = field(default_factory=time.time)
    completed: Optional[float] = None


@dataclass
class Action:
    type: str  # "search", "ask_user", "reason", "learn", "observe", "simulate"
    params: Dict[str, Any]
    status: str = "pending"


class Planner:
    def __init__(self, llm: LLMInterface, world_model: WorldModel):
        self.llm = llm
        self.world = world_model
        self.goals: List[Goal] = []
        self.current_goal: Optional[Goal] = None
        self.action_handlers: Dict[str, Callable] = {}

    def register_action_handler(self, action_type: str, handler: Callable):
        self.action_handlers[action_type] = handler

    def add_goal(self, description: str, priority: int = 5) -> int:
        goal = Goal(description, priority)
        self.goals.append(goal)
        return len(self.goals) - 1

    def generate_plan(self, goal_description: str, context: str = "") -> List[Action]:
        """Использовать LLM для генерации плана действий."""
        prompt = f"""
Ты — планировщик. Для достижения цели: "{goal_description}"
Предложи последовательность действий. Доступные типы действий:
- search: поиск в интернете (params: {"query": str})
- ask_user: задать вопрос пользователю (params: {"question": str})
- reason: внутреннее рассуждение (params: {"problem": str})
- learn: выучить новую пару вопрос-ответ (params: {"question": str, "answer": str})
- observe: провести наблюдение (params: {"what": str})
- simulate: мысленно проиграть сценарий (params: {"scenario": str})

Верни план в формате JSON: [{"type": "...", "params": {...}}, ...]
Только JSON, без пояснений.
Контекст: {context}
"""
        response = self.llm.generate(prompt, max_tokens=500, temperature=0.3)
        # Парсим JSON (упрощённо)
        import json
        try:
            # Найти JSON в ответе
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                actions_data = json.loads(json_str)
                actions = []
                for a in actions_data:
                    actions.append(Action(a['type'], a['params']))
                return actions
        except:
            pass
        return []

    def execute_plan(self, goal: Goal) -> bool:
        """Выполнить план, вызывая зарегистрированные обработчики."""
        goal.status = "in_progress"
        for action in goal.plan:
            if action.status == "done":
                continue
            handler = self.action_handlers.get(action.type)
            if handler:
                result = handler(action.params)
                if result:
                    action.status = "done"
                else:
                    action.status = "failed"
                    goal.status = "failed"
                    return False
            else:
                action.status = "failed"
                goal.status = "failed"
                return False
        goal.status = "done"
        goal.completed = time.time()
        return True

    def prioritize_goals(self):
        """Сортировка целей по приоритету и времени."""
        self.goals.sort(key=lambda g: (-g.priority, g.created))

    def get_next_action(self) -> Optional[Action]:
        """Получить следующее невыполненное действие из текущей цели."""
        if self.current_goal is None:
            self.prioritize_goals()
            if self.goals:
                self.current_goal = self.goals[0]
        if self.current_goal and self.current_goal.status == "pending":
            # Если нет плана, сгенерировать
            if not self.current_goal.plan:
                self.current_goal.plan = self.generate_plan(self.current_goal.description)
            # Найти первое невыполненное действие
            for action in self.current_goal.plan:
                if action.status == "pending":
                    return action
            # Если все выполнены, отметить цель как done
            self.current_goal.status = "done"
            self.current_goal = None
            return None
        return None