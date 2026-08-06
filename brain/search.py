# brain/search.py
"""
Модуль для бесплатного интернет-поиска через DuckDuckGo.
Использует библиотеку duckduckgo-search (установите: pip install duckduckgo-search)
"""
from typing import List, Dict, Optional
from ddgs import DDGS   # вместо from duckduckgo_search import DDGS


class WebSearcher:
    def __init__(self, max_results: int = 3, timeout: int = 10):
        self.max_results = max_results
        self.timeout = timeout

    def search(self, query: str) -> List[Dict[str, str]]:
        """
        Выполняет поиск и возвращает список словарей с полями 'title', 'body', 'href'.
        """
        results = []
        try:
            with DDGS(timeout=self.timeout) as ddgs:
                for r in ddgs.text(query, max_results=self.max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "body": r.get("body", ""),
                        "href": r.get("href", ""),
                    })
            return results
        except Exception as e:
            print(f"[WebSearcher] Ошибка поиска: {e}")
            return []

    def format_results(self, results: List[Dict[str, str]]) -> str:
        """Форматирует результаты для вставки в промпт."""
        if not results:
            return "Ничего не найдено."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}")
            lines.append(f"   {r['body'][:300]}...")
            lines.append(f"   Источник: {r['href']}")
        return "\n".join(lines)