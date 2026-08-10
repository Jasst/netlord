# fix_labels.py
import pickle
import torch
from brain import CognitiveBrain, BrainConfig

# Загружаем конфиг и мозг
config = BrainConfig()
brain = CognitiveBrain(config)
brain.load()  # загружает существующую модель

# Берём граф (уровень 0, если иерархический)
if hasattr(brain.graph, 'levels'):
    g = brain.graph.levels[0]
else:
    g = brain.graph

# Восстанавливаем метки из concept_index
reverse_index = {v: k for k, v in brain.concept_index.items()}
n_nodes = g.node_emb.shape[0]
for nid in range(1, n_nodes + 1):
    if nid not in g.node_labels and nid in reverse_index:
        g.node_labels[nid] = reverse_index[nid]
        print(f"Восстановлена метка для узла {nid}: {reverse_index[nid]}")

# Сохраняем модель (теперь с метаданными)
brain.save()
print("Модель сохранена с восстановленными метками.")