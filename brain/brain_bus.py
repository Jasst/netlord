"""
Простая шина событий для асинхронного обмена между модулями.
"""
from typing import Callable, Dict, List, Any

class BrainBus:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.subscribers = {}
        return cls._instance

    def subscribe(self, event_type: str, callback: Callable):
        self.subscribers.setdefault(event_type, []).append(callback)

    def publish(self, event_type: str, data: Any):
        for cb in self.subscribers.get(event_type, []):
            try:
                cb(data)
            except Exception as e:
                print(f"[Bus] Error in callback for {event_type}: {e}")

# Глобальный экземпляр
bus = BrainBus()