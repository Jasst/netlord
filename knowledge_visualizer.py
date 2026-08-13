#!/usr/bin/env python3
"""
Визуализатор знаний CognitiveBrain.
Создаёт интерактивный HTML-отчёт с графом, таблицами и статистикой.
Запуск: python knowledge_visualizer.py [--limit N] [--output report.html]
"""

import os
import re
import sys
import json
import argparse
from collections import Counter

import torch
import numpy as np
from pyvis.network import Network

from brain import CognitiveBrain, BrainConfig
from brain.graph import NodeType


def _norm_type(t):
    """Приводит node_type к элементу enum NodeType, независимо от того,
    хранится ли он как сам enum или как строка (например, после
    сохранения/загрузки модели). Раньше вызов node_type.name падал с
    AttributeError, если тип приходил строкой — из-за этого отчёт мог
    вообще не генерироваться."""
    if isinstance(t, NodeType):
        return t
    if isinstance(t, str):
        key = t.split('.')[-1]
        try:
            return NodeType[key]
        except KeyError:
            pass
    return NodeType.CONCEPT


def _embed_pyvis_html(net):
    """Возвращает (head_extra, body_inner) — pyvis Network.generate_html()
    отдаёт ПОЛНЫЙ HTML-документ (со своими <html>/<head>/<body>), а не
    фрагмент. Раньше этот документ целиком вставлялся внутрь другого
    <body>, что даёт вложенные <html>/<head>/<body> — невалидный HTML,
    из-за которого браузер мог обрезать или неверно рендерить граф и
    всё, что идёт после него в отчёте. Здесь мы аккуратно достаём
    содержимое <head> (там подключение vis-network.js и стили canvas) и
    содержимое <body> (сам div с графом и inline-скрипт) и возвращаем их
    отдельно, чтобы вставить каждое в нужное место один раз."""
    full_html = net.generate_html(notebook=False)
    head_match = re.search(r"<head>(.*?)</head>", full_html, re.DOTALL)
    body_match = re.search(r"<body>(.*?)</body>", full_html, re.DOTALL)
    head_extra = head_match.group(1) if head_match else ""
    body_inner = body_match.group(1) if body_match else full_html
    return head_extra, body_inner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000,
                        help="Максимальное число узлов для отображения на графе (0 = все)")
    parser.add_argument("--output", default="knowledge_report.html",
                        help="Имя выходного HTML-файла")
    parser.add_argument("--model-dir", default=None,
                        help="Папка модели (если не из конфига)")
    args = parser.parse_args()

    # Загружаем конфиг и мозг
    config = BrainConfig()
    if args.model_dir:
        config.model_dir = args.model_dir

    brain = CognitiveBrain(config)
    brain.load()
    brain.load_dialog_history()

    # Собираем данные
    graph = brain.graph
    kb = brain.knowledge_base
    memory = brain.memory
    semantic = memory.semantic_memory

    # Получаем эмбеддинги и метки
    if hasattr(graph, 'levels'):
        # Иерархический граф: берём уровень 0 как основной
        level0 = graph.levels[0]
        node_emb = level0.node_emb.detach().cpu().numpy()
        node_labels = level0.node_labels
        node_types = level0.node_types
        edges = level0._edges
        edge_weights = [w.item() for w in level0._edge_weights]
    else:
        node_emb = graph.node_emb.detach().cpu().numpy()
        node_labels = graph.node_labels
        node_types = graph.node_types
        edges = graph._edges
        edge_weights = [w.item() for w in graph._edge_weights]

    num_nodes = node_emb.shape[0]
    num_edges = len(edges)

    print(f"Всего нейронов: {num_nodes}, синапсов: {num_edges}")

    # --- Подготовка данных для визуализации ---
    # Ограничим число узлов для графа (по умолчанию 200)
    limit = args.limit if args.limit > 0 else num_nodes
    # Выбираем узлы с наибольшей степенью (или просто первые limit)
    if num_nodes > limit:
        degree = Counter()
        for f, t in edges:
            if f <= num_nodes and t <= num_nodes:
                degree[f] += 1
                degree[t] += 1
        top_nodes = [nid for nid, _ in sorted(degree.items(), key=lambda x: x[1], reverse=True)[:limit]]
        # Добавим также узлы с метками (они важны)
        for nid in node_labels.keys():
            if nid not in top_nodes and len(top_nodes) < limit:
                top_nodes.append(nid)
        top_nodes = sorted(set(top_nodes))
    else:
        top_nodes = list(range(1, num_nodes + 1))

    # Строим карту старых id -> новые (для отображения)
    id_map = {old: new for new, old in enumerate(top_nodes)}

    # --- Создаём сеть pyvis ---
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100}
      }
    }
    """)

    # Добавляем узлы
    for nid in top_nodes:
        label = node_labels.get(nid, f"нейрон_{nid}")
        # Обрезаем слишком длинные метки
        if len(label) > 30:
            label = label[:27] + "..."
        node_type = _norm_type(node_types.get(nid, NodeType.CONCEPT))
        color = {
            NodeType.SENSORY: "#FF6B6B",
            NodeType.CONCEPT: "#4ECDC4",
            NodeType.MOTOR: "#45B7D1",
            NodeType.EMOTIONAL: "#FFA07A",
            NodeType.ATTENTION: "#DDA0DD",
        }.get(node_type, "#AAAAAA")
        net.add_node(nid, label=label, title=f"ID: {nid}\nТип: {node_type.name}", color=color)

    # Добавляем рёбра (только между выбранными узлами)
    for (f, t), w in zip(edges, edge_weights):
        if f in top_nodes and t in top_nodes:
            # Вес может быть отрицательным (для отрицательных связей)
            width = max(1, abs(w) * 5)
            color = "#00FF00" if w > 0 else "#FF4444"
            net.add_edge(f, t, value=abs(w), title=f"вес: {w:.3f}", color=color, width=width)

    # --- Формируем HTML-отчёт ---
    html_parts = []

    # Заголовок
    graph_head_extra, graph_body_inner = _embed_pyvis_html(net)

    html_parts.append(f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8"><title>Knowledge Report</title>
    {graph_head_extra}
    </head>
    <body>
    <h1>Отчёт о знаниях CognitiveBrain</h1>
    <p>Всего нейронов: {num_nodes}, отображено: {len(top_nodes)}</p>
    <p>Синапсов: {num_edges}, база знаний: {len(kb)} записей, эпизодическая память: {memory.episodic.index.ntotal} векторов</p>
    <hr>
    <h2>Граф знаний (интерактивный)</h2>
    """)

    # Вставляем содержимое <body> графа (без обёртки html/head/body)
    html_parts.append(graph_body_inner)

    # --- Таблица базы знаний ---
    html_parts.append("<h2>База знаний (вопрос-ответ)</h2>")
    if kb:
        html_parts.append("<table border='1'><tr><th>Вопрос</th><th>Ответ</th><th>Уверенность</th></tr>")
        for item in kb[:100]:  # покажем первые 100
            q = item.get('q', '')
            a = item.get('a', '')
            conf = item.get('confidence', 0.5)
            html_parts.append(f"<tr><td>{q}</td><td>{a}</td><td>{conf:.2f}</td></tr>")
        html_parts.append("</table>")
        if len(kb) > 100:
            html_parts.append(f"<p>... и ещё {len(kb)-100} записей.</p>")
    else:
        html_parts.append("<p>База знаний пуста.</p>")

    # --- Эпизодическая память (последние 20) ---
    html_parts.append("<h2>Эпизодическая память (последние записи)</h2>")
    episodic_metadata = memory.episodic.metadata
    if episodic_metadata:
        html_parts.append("<table border='1'><tr><th>ID</th><th>Контекст</th><th>Время</th></tr>")
        sorted_ids = sorted(episodic_metadata.keys(), key=lambda x: memory.episodic.timestamps.get(x, 0), reverse=True)
        for vid in sorted_ids[:20]:
            meta = episodic_metadata[vid]
            text = meta.get('text', '')
            ts = memory.episodic.timestamps.get(vid, 0)
            html_parts.append(f"<tr><td>{vid}</td><td>{text}</td><td>{ts}</td></tr>")
        html_parts.append("</table>")
        if len(sorted_ids) > 20:
            html_parts.append(f"<p>... и ещё {len(sorted_ids)-20} записей.</p>")
    else:
        html_parts.append("<p>Эпизодическая память пуста.</p>")

    # --- Семантические тройки ---
    html_parts.append("<h2>Семантическая память (тройки)</h2>")
    triples = semantic.triples
    if triples:
        html_parts.append("<table border='1'><tr><th>Субъект</th><th>Предикат</th><th>Объект</th><th>Уверенность</th></tr>")
        for s, p, o, c in triples[:50]:
            html_parts.append(f"<tr><td>{s}</td><td>{p}</td><td>{o}</td><td>{c:.2f}</td></tr>")
        html_parts.append("</table>")
        if len(triples) > 50:
            html_parts.append(f"<p>... и ещё {len(triples)-50} троек.</p>")
    else:
        html_parts.append("<p>Семантическая память пуста.</p>")

    # --- Статистика ---
    stats = brain.get_stats()
    html_parts.append("<h2>Статистика</h2>")
    html_parts.append("<pre>" + json.dumps(stats, indent=2, default=str) + "</pre>")

    # --- Модель мира ---
    html_parts.append("<h2>Модель мира</h2>")
    if hasattr(brain, 'world_model') and brain.world_model:
        wm = brain.world_model
        html_parts.append(f"<p>Фактов: {len(wm.facts)}, Убеждений: {len(wm.beliefs)}, Гипотез: {len(wm.hypotheses)}, Эпизодов: {len(wm.episodes)}</p>")
        if wm.facts:
            html_parts.append("<h3>Последние факты</h3>")
            html_parts.append("<table border='1'><tr><th>Субъект</th><th>Отношение</th><th>Объект</th><th>Увер.</th><th>Источник</th></tr>")
            for f in wm.facts[-30:]:
                html_parts.append(f"<tr><td>{f.subject}</td><td>{f.relation}</td><td>{f.object}</td><td>{f.confidence:.2f}</td><td>{f.source}</td></tr>")
            html_parts.append("</table>")
        if wm.hypotheses:
            html_parts.append("<h3>Гипотезы</h3>")
            html_parts.append("<ul>")
            for h in wm.hypotheses[-10:]:
                html_parts.append(f"<li>{h.content} (увер: {h.confidence:.2f}, проверена: {h.verified}, результат: {h.result})</li>")
            html_parts.append("</ul>")
    else:
        html_parts.append("<p>Модель мира отключена.</p>")

    # --- Self-модель ---
    html_parts.append("<h2>Self-модель</h2>")
    if hasattr(brain, 'self_model') and brain.self_model:
        sm = brain.self_model
        html_parts.append(f"<p>Способности: {sm.capabilities}</p>")
        if sm.recent_errors:
            html_parts.append(f"<p>Недавние ошибки: {', '.join(sm.recent_errors[-5:])}</p>")
        if sm.successful_strategies:
            html_parts.append(f"<p>Успешные стратегии: {', '.join(sm.successful_strategies[-5:])}</p>")
        if sm.unresolved_questions:
            html_parts.append(f"<p>Нерешённые вопросы: {', '.join(sm.unresolved_questions[-5:])}</p>")
    else:
        html_parts.append("<p>Self-модель отключена.</p>")

    # --- Закрывающие теги ---
    html_parts.append("</body></html>")

    # Сохраняем
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    print(f"Отчёт сохранён в {args.output}")
    print(f"Откройте его в браузере.")


if __name__ == "__main__":
    main()