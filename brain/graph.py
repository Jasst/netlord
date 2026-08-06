# brain/graph.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm
from typing import Optional, List, Tuple, Dict, Any
from enum import Enum

from brain.utils import grow_parameter_in_optimizer


class NodeType(Enum):
    SENSORY = 0
    CONCEPT = 1
    MOTOR = 2
    EMOTIONAL = 3
    ATTENTION = 4


# ----------------------------------------------------------------------
# Базовый дифференцируемый граф (один уровень)
# ----------------------------------------------------------------------
class DifferentiableNeuralGraph(nn.Module):
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

        self.act = nn.ELU()

        self._edges: List[Tuple[int, int]] = []
        self._edge_weights = nn.ParameterList()
        self._edge_index = None

        self.node_labels: Dict[int, str] = {}
        self.node_types: Dict[int, NodeType] = {}
        self.node_clusters: Dict[int, str] = {}

    def add_node(self, embedding: torch.Tensor, label: str = "", cluster: str = "hidden",
                 layer: int = 0, node_type: NodeType = NodeType.CONCEPT,
                 optimizer=None) -> int:
        """
        optimizer: если передан self.optimizer из CognitiveBrain, новая версия node_emb
        сразу синхронизируется с оптимизатором (см. utils.grow_parameter_in_optimizer) —
        без этого добавленные узлы никогда не обучаются градиентным спуском.
        """
        embedding = F.normalize(embedding.float(), p=2, dim=0)
        old_param = self.node_emb
        num_old_rows = old_param.shape[0]
        with torch.no_grad():
            new_data = torch.cat([old_param.data, embedding.unsqueeze(0)], dim=0)
        nid = num_old_rows + 1
        self.node_emb = nn.Parameter(new_data)
        self.node_labels[nid] = label
        self.node_types[nid] = node_type
        self.node_clusters[nid] = cluster

        if optimizer is not None:
            grow_parameter_in_optimizer(optimizer, old_param, self.node_emb, num_old_rows)
        return nid

    def add_synapse(self, from_id: int, to_id: int, weight: float = 0.1, optimizer=None) -> int:
        self._edges.append((from_id, to_id))
        new_w = nn.Parameter(torch.tensor(weight, dtype=torch.float))
        self._edge_weights.append(new_w)
        self._rebuild_edges()

        if optimizer is not None and optimizer.param_groups:
            # Новый вес ребра тоже отсутствовал в снимке параметров оптимизатора — добавляем явно.
            optimizer.param_groups[0]["params"].append(new_w)
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
            weights = [w for w in self._edge_weights]
            edge_attr = torch.stack(weights).view(-1, 1)
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
                h = h_new
        return h

    def get_node_embeddings(self) -> torch.Tensor:
        return self.node_emb

    def get_edge_weights(self) -> torch.Tensor:
        if len(self._edge_weights) == 0:
            return torch.tensor([])
        weights = [w for w in self._edge_weights]
        return torch.cat([w.view(1) for w in weights])

    def find_most_similar(self, query: torch.Tensor, threshold: float = 0.8) -> Optional[int]:
        if self.node_emb.shape[0] == 0:
            return None
        sim = F.cosine_similarity(query.unsqueeze(0), self.node_emb, dim=1)
        best_idx = torch.argmax(sim).item()
        if sim[best_idx] >= threshold:
            return best_idx + 1
        return None


# ----------------------------------------------------------------------
# Иерархический граф с несколькими уровнями и self-attention
# ----------------------------------------------------------------------
class HierarchicalGraph(nn.Module):
    def __init__(self, dims: List[int], num_heads: int = 4, num_layers: int = 2, attn_heads: int = 8):
        super().__init__()
        self.levels = nn.ModuleList()
        self.cross_attn = nn.ModuleList()
        self.attentions = nn.ModuleList()

        for i, d in enumerate(dims):
            level_graph = DifferentiableNeuralGraph(
                dim=d,
                max_nodes=5000,
                hidden_dim=d,
                num_heads=num_heads,
                num_layers=num_layers
            )
            self.levels.append(level_graph)
            self.attentions.append(nn.MultiheadAttention(d, attn_heads, batch_first=True))
            if i < len(dims) - 1:
                self.cross_attn.append(nn.Linear(d, dims[i + 1]))

    def add_node(self, embedding: torch.Tensor, label: str = "", cluster: str = "hidden",
                 layer: int = 0, node_type: NodeType = NodeType.CONCEPT,
                 level_idx: int = 0, optimizer=None) -> int:
        return self.levels[level_idx].add_node(embedding, label, cluster, layer, node_type,
                                                 optimizer=optimizer)

    def add_synapse(self, from_id: int, to_id: int, weight: float = 0.1,
                     level_idx: int = 0, optimizer=None) -> int:
        return self.levels[level_idx].add_synapse(from_id, to_id, weight, optimizer=optimizer)

    def find_most_similar(self, query: torch.Tensor, level_idx: int = 0, threshold: float = 0.8) -> Optional[int]:
        return self.levels[level_idx].find_most_similar(query, threshold)

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        # ПРИМЕЧАНИЕ: при x=None каждый уровень использует свои собственные узлы
        # (self.levels[i].node_emb) через forward(None) -> берём x с уровня i,
        # а не протаскиваем cross_attn-проекцию предыдущего уровня как ошибочную замену
        # содержимого следующего уровня.
        outputs = []
        h = x
        for i, (g, attn) in enumerate(zip(self.levels, self.attentions)):
            level_input = h if h is not None else None
            h_level = g(level_input)
            h_level, _ = attn(h_level.unsqueeze(0), h_level.unsqueeze(0), h_level.unsqueeze(0))
            h_level = h_level.squeeze(0)
            outputs.append(h_level)
            if i < len(self.levels) - 1:
                h = self.cross_attn[i](h_level)
            else:
                h = h_level
        return outputs[0]

    def get_level_embeddings(self, level_idx: int) -> torch.Tensor:
        return self.levels[level_idx].node_emb

    def get_edge_weights(self) -> torch.Tensor:
        return self.levels[0].get_edge_weights()

    @property
    def node_emb(self):
        return self.levels[0].node_emb

    @property
    def _edges(self):
        return self.levels[0]._edges

    @property
    def _edge_weights(self):
        return self.levels[0]._edge_weights

    @property
    def node_labels(self):
        return self.levels[0].node_labels

    @property
    def node_types(self):
        return self.levels[0].node_types

    @property
    def node_clusters(self):
        return self.levels[0].node_clusters
