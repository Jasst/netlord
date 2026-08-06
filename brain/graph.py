# brain/graph.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Optional, List

class DifferentiableNeuralGraph(nn.Module):
    def __init__(self, dim: int, max_nodes: int = 5000, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.max_nodes = max_nodes

        # Эмбеддинги узлов (обучаемые)
        self.register_buffer("node_emb", torch.zeros(1, dim))
        self.register_buffer("node_mask", torch.zeros(1, dtype=torch.bool))
        self.node_emb = nn.Parameter(self.node_emb)

        # GAT слои (с поддержкой edge_attr)
        self.gat1 = GATConv(dim, hidden_dim, heads=num_heads, concat=True, edge_dim=1)
        self.gat2 = GATConv(hidden_dim * num_heads, dim, heads=1, concat=False, edge_dim=1)
        self.update_scale = nn.Parameter(torch.tensor(0.1))

        # Метаданные узлов
        self.neuron_labels = {}
        self.neuron_clusters = {}
        self.neuron_layers = {}
        self._next_nid = 1

        # ---- НОВОЕ: обучаемые веса синапсов ----
        self._edge_weights = nn.ParameterList()   # каждый вес – отдельный параметр
        self._edges = []            # список (from_id, to_id) – для индексов
        self._edge_index = None     # тензор 2 x E (перестраивается при изменении)

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

        self._rebuild_edges()
        return nid

    def add_synapse(self, from_id: int, to_id: int, weight: float = 0.1) -> int:
        # Добавляем ребро в список (без веса, вес хранится в _edge_weights)
        self._edges.append((from_id, to_id))
        # Добавляем новый параметр (вес) в ParameterList
        self._edge_weights.append(nn.Parameter(torch.tensor(weight, dtype=torch.float)))
        # Перестраиваем edge_index
        self._rebuild_edges()
        return len(self._edges) - 1

    def _rebuild_edges(self):
        """Перестраивает _edge_index на основе _edges (индексы рёбер)."""
        if not self._edges:
            self._edge_index = torch.zeros((2, 0), dtype=torch.long)
            return
        u = [f - 1 for f, _ in self._edges]
        v = [t - 1 for _, t in self._edges]
        self._edge_index = torch.tensor([u, v], dtype=torch.long)

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x is None:
            x = self.node_emb.float()
        if self._edge_index is None or self._edge_index.size(1) == 0:
            return x

        # Собираем все веса из ParameterList в один тензор (сохраняя градиенты)
        if len(self._edge_weights) > 0:
            # torch.stack сохраняет градиенты, т.к. каждый элемент – Parameter
            edge_attr = torch.stack(self._edge_weights).view(-1, 1)  # [E, 1]
        else:
            edge_attr = torch.zeros((0, 1), device=x.device)

        h = F.elu(self.gat1(x, self._edge_index, edge_attr=edge_attr))
        h = F.elu(self.gat2(h, self._edge_index, edge_attr=edge_attr))
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
        """Возвращает все веса синапсов как тензор (для регуляризации)."""
        if len(self._edge_weights) == 0:
            return torch.tensor([])
        # Собираем в плоский тензор (сохраняя градиенты)
        return torch.cat([w.view(1) for w in self._edge_weights])