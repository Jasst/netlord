# brain/graph.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm
from typing import Optional, List, Tuple, Dict, Any
from enum import Enum

class NodeType(Enum):
    SENSORY = 0
    CONCEPT = 1
    MOTOR = 2
    EMOTIONAL = 3
    ATTENTION = 4

class DifferentiableNeuralGraph(nn.Module):
    """Исходный граф (сохранён для обратной совместимости)"""
    def __init__(self, dim: int, max_nodes: int = 5000, hidden_dim: int = 256,
                 num_heads: int = 4, num_layers: int = 3):
        super().__init__()
        self.dim = dim
        self.max_nodes = max_nodes
        self.node_emb = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.node_mask = torch.ones(1, dtype=torch.bool)
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
        self._edges: List[Tuple[int, int]] = []
        self._edge_weights = nn.ParameterList()
        self._edge_index = None
        # метаданные
        self.node_labels: Dict[int, str] = {}
        self.node_types: Dict[int, NodeType] = {}
        self.node_clusters: Dict[int, str] = {}

    def add_node(self, embedding: torch.Tensor, label: str = "", cluster: str = "hidden",
                 layer: int = 0, node_type: NodeType = NodeType.CONCEPT) -> int:
        embedding = F.normalize(embedding.float(), p=2, dim=0)
        nid = self.node_emb.shape[0] + 1
        new_emb = embedding.unsqueeze(0)
        self.node_emb = nn.Parameter(torch.cat([self.node_emb.data, new_emb], dim=0))
        self.node_labels[nid] = label
        self.node_types[nid] = node_type
        self.node_clusters[nid] = cluster
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
        if len(self._edge_weights) > 0:
            edge_attr = torch.stack(self._edge_weights).view(-1, 1)
        else:
            edge_attr = torch.zeros((0, 1), device=x.device)
        h = x
        for layer, norm in zip(self.layers, self.norms):
            h_new = layer(h, self._edge_index, edge_attr=edge_attr)
            h_new = norm(h_new)
            h_new = self.act(h_new)
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


class HierarchicalGraph(nn.Module):
    """
    Многоуровневый граф с self-attention и проекциями между уровнями.
    """
    def __init__(self, dims: List[int], num_heads: int = 4, num_layers: int = 2, attn_heads: int = 8):
        super().__init__()
        self.levels = nn.ModuleList()
        self.cross_attn = nn.ModuleList()
        self.attentions = nn.ModuleList()

        for i, d in enumerate(dims):
            # каждый уровень – обычный граф (можно заменить на DifferentiableNeuralGraph)
            level_graph = DifferentiableNeuralGraph(
                dim=d,
                max_nodes=5000,
                hidden_dim=d*2,
                num_heads=num_heads,
                num_layers=num_layers
            )
            self.levels.append(level_graph)
            # self-attention после каждого уровня
            self.attentions.append(nn.MultiheadAttention(d, attn_heads, batch_first=True))
            if i < len(dims) - 1:
                self.cross_attn.append(nn.Linear(d, dims[i+1]))

    def add_node(self, level_idx: int, embedding: torch.Tensor, **kwargs) -> int:
        return self.levels[level_idx].add_node(embedding, **kwargs)

    def add_synapse(self, level_idx: int, from_id: int, to_id: int, weight: float = 0.1) -> int:
        return self.levels[level_idx].add_synapse(from_id, to_id, weight)

    def forward(self, x: Optional[torch.Tensor] = None, level_idx: int = 0) -> torch.Tensor:
        # Прогон через все уровни (сверху вниз и обратно можно усложнить)
        h = x if x is not None else self.levels[0].node_emb
        for i, (g, attn) in enumerate(zip(self.levels, self.attentions)):
            h = g(h) if h is not None else g()
            # self-attention
            h, _ = attn(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0))
            h = h.squeeze(0)
            if i < len(self.levels) - 1:
                h = self.cross_attn[i](h)
        return h

    def get_level_embeddings(self, level_idx: int) -> torch.Tensor:
        return self.levels[level_idx].node_emb

    def find_most_similar(self, level_idx: int, query: torch.Tensor, threshold: float = 0.8) -> Optional[int]:
        return self.levels[level_idx].find_most_similar(query, threshold)