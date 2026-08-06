# brain/graph.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import List, Tuple, Optional

class DifferentiableNeuralGraph(nn.Module):
    def __init__(self, dim: int, max_nodes: int = 5000, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.max_nodes = max_nodes

        # Эмбеддинги узлов (обучаемые)
        self.register_buffer("node_emb", torch.zeros(1, dim))
        self.register_buffer("node_mask", torch.zeros(1, dtype=torch.bool))
        self.node_emb = nn.Parameter(self.node_emb)

        # GAT слои (теперь с edge_dim=1 для использования весов синапсов)
        self.gat1 = GATConv(dim, hidden_dim, heads=num_heads, concat=True, edge_dim=1)
        self.gat2 = GATConv(hidden_dim * num_heads, dim, heads=1, concat=False, edge_dim=1)
        self.update_scale = nn.Parameter(torch.tensor(0.1))

        # Метаданные узлов и рёбер
        self.neuron_labels = {}
        self.neuron_clusters = {}
        self.neuron_layers = {}
        self._next_nid = 1

        # Список рёбер (from, to, weight) – будем хранить как тензоры
        self._edges = []  # временный список для накопления
        self._edge_index = None   # тензор 2 x E
        self._edge_weight = None  # тензор E x 1

    def add_node(self, embedding: torch.Tensor, label: str = "", cluster: str = "hidden", layer: int = 0) -> int:
        embedding = embedding.float()
        if self.node_emb.shape[0] >= self.max_nodes:
            raise RuntimeError("Max neurons reached")
        nid = self._next_nid
        self._next_nid += 1

        new_emb = F.normalize(embedding.unsqueeze(0), p=2, dim=1)
        old_emb = self.node_emb.data
        new_emb_all = torch.cat([old_emb, new_emb], dim=0)
        self.node_emb = nn.Parameter(new_emb_all)

        old_mask = self.node_mask
        new_mask = torch.cat([old_mask, torch.tensor([True])], dim=0)
        self.register_buffer("node_mask", new_mask)

        self.neuron_labels[nid] = label
        self.neuron_clusters[nid] = cluster
        self.neuron_layers[nid] = layer

        # После добавления узла перестраиваем edge_index/weight, так как индексы сместились
        self._rebuild_edges()
        return nid

    def add_synapse(self, from_id: int, to_id: int, weight: float = 0.1) -> int:
        # Добавляем в список, а не сразу в тензор
        self._edges.append((from_id, to_id, weight))
        # Индекс ребра = len(self._edges) - 1
        self._rebuild_edges()
        return len(self._edges) - 1

    def _rebuild_edges(self):
        """Преобразует список _edges в тензоры edge_index и edge_weight."""
        if not self._edges:
            self._edge_index = torch.zeros((2, 0), dtype=torch.long)
            self._edge_weight = torch.zeros((0, 1), dtype=torch.float)
            return

        u = []
        v = []
        w = []
        for (f, t, wt) in self._edges:
            # индексы с 0 (т.к. в PyG нумерация с 0)
            u.append(f - 1)
            v.append(t - 1)
            w.append(wt)
        self._edge_index = torch.tensor([u, v], dtype=torch.long)
        self._edge_weight = torch.tensor(w, dtype=torch.float).view(-1, 1)

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Применяет GAT к текущим эмбеддингам и возвращает обновлённые.
        Если x не задан, использует self.node_emb.
        """
        if x is None:
            x = self.node_emb.float()
        # Убедимся, что edge_index и edge_weight актуальны
        if self._edge_index is None or self._edge_index.size(1) == 0:
            return x  # нет рёбер – без изменений

        # Передаём edge_attr (веса) в GAT
        h = F.elu(self.gat1(x, self._edge_index, edge_attr=self._edge_weight))
        h = F.elu(self.gat2(h, self._edge_index, edge_attr=self._edge_weight))
        # Плавное обновление (residual)
        delta = torch.tanh(self.update_scale) * (h - x)
        return x + delta

    def get_node_embeddings(self) -> torch.Tensor:
        return self.node_emb.float()

    def get_embedding_by_id(self, nid: int) -> torch.Tensor:
        return self.node_emb[nid - 1].float()

    def find_most_similar(self, query: torch.Tensor, threshold: float = 0.8) -> Optional[int]:
        if self.node_emb.shape[0] == 0:
            return None
        sim = F.cosine_similarity(query.unsqueeze(0), self.node_emb, dim=1)
        best_idx = torch.argmax(sim).item()
        if sim[best_idx] >= threshold:
            return best_idx + 1
        return None

    def get_edge_weights(self) -> torch.Tensor:
        """Возвращает текущие веса синапсов (для регуляризации)."""
        if self._edge_weight is None:
            return torch.tensor([])
        return self._edge_weight.flatten()