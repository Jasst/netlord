# brain/graph.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm
from typing import Optional, List, Tuple

class DifferentiableNeuralGraph(nn.Module):
    def __init__(self, dim: int, max_nodes: int = 5000, hidden_dim: int = 256, num_heads: int = 4, num_layers: int = 3):
        super().__init__()
        self.dim = dim
        self.max_nodes = max_nodes

        # Обучаемые эмбеддинги узлов (инициализация случайная)
        self.node_emb = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.node_mask = torch.ones(1, dtype=torch.bool)  # для совместимости

        # GATv2 слои с LayerNorm и skip-connections
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_dim = dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else dim
            self.layers.append(GATv2Conv(in_dim, out_dim, heads=num_heads, concat=False, edge_dim=1))
            self.norms.append(LayerNorm(out_dim))
            in_dim = out_dim

        self.skip_proj = nn.Linear(dim, dim)
        self.act = nn.ELU()

        # Рёбра: список кортежей (from_id, to_id) и обучаемые веса
        self._edges: List[Tuple[int, int]] = []
        self._edge_weights = nn.ParameterList()   # каждый вес отдельный параметр
        self._edge_index = None

    def add_node(self, embedding: torch.Tensor, label: str = "", cluster: str = "hidden", layer: int = 0) -> int:
        embedding = F.normalize(embedding.float(), p=2, dim=0)
        nid = self.node_emb.shape[0] + 1  # 1-based
        new_emb = embedding.unsqueeze(0)
        self.node_emb = nn.Parameter(torch.cat([self.node_emb.data, new_emb], dim=0))
        # метаданные храним снаружи
        return nid

    def add_synapse(self, from_id: int, to_id: int, weight: float = 0.1) -> int:
        self._edges.append((from_id, to_id))
        self._edge_weights.append(nn.Parameter(torch.tensor(weight, dtype=torch.float)))
        self._rebuild_edges()
        return len(self._edges) - 1

    def _rebuild_edges(self):
        if not self._edges:
            self._edge_index = torch.zeros((2, 0), dtype=torch.long)
            return
        u = [f - 1 for f, _ in self._edges]
        v = [t - 1 for _, t in self._edges]
        self._edge_index = torch.tensor([u, v], dtype=torch.long)

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x is None:
            x = self.node_emb
        if self._edge_index is None or self._edge_index.size(1) == 0:
            return x

        # Собираем веса рёбер
        if len(self._edge_weights) > 0:
            edge_attr = torch.stack(self._edge_weights).view(-1, 1)
        else:
            edge_attr = torch.zeros((0, 1), device=x.device)

        h = x
        for layer, norm in zip(self.layers, self.norms):
            h_new = layer(h, self._edge_index, edge_attr=edge_attr)
            h_new = norm(h_new)
            h_new = self.act(h_new)
            # skip-connection (если размеры совпадают)
            if h.shape == h_new.shape:
                h = h + h_new
            else:
                h = h_new + self.skip_proj(h)
        return h

    def get_node_embeddings(self) -> torch.Tensor:
        return self.node_emb

    def get_edge_weights(self) -> torch.Tensor:
        if len(self._edge_weights) == 0:
            return torch.tensor([])
        return torch.cat([w.view(1) for w in self._edge_weights])

    def find_most_similar(self, query: torch.Tensor, threshold: float = 0.8) -> Optional[int]:
        if self.node_emb.shape[0] == 0:
            return None
        sim = F.cosine_similarity(query.unsqueeze(0), self.node_emb, dim=1)
        best_idx = torch.argmax(sim).item()
        if sim[best_idx] >= threshold:
            return best_idx + 1
        return None