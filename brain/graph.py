# brain/graph.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm
from typing import Optional, List, Tuple, Dict, Any
from enum import Enum

from brain.utils import grow_parameter_in_optimizer


# ----------------------------------------------------------------------
# Типы отношений (расширены для новых механизмов)
# ----------------------------------------------------------------------
RELATION_TYPES: List[str] = [
    "has_answer",
    "contradicts",
    "inferred",        # выведенное ребро (транзитивное замыкание)
    "co_activated",    # совместная активация
    "derived_from"     # синтезированный узел
]


class NodeType(Enum):
    SENSORY = 0
    CONCEPT = 1
    MOTOR = 2
    EMOTIONAL = 3
    ATTENTION = 4
    SELF = 5


# ----------------------------------------------------------------------
# Базовый дифференцируемый граф
# ----------------------------------------------------------------------
class DifferentiableNeuralGraph(nn.Module):
    def __init__(self, dim: int, max_nodes: int = 5000, hidden_dim: int = 256,
                 num_heads: int = 4, num_layers: int = 3,
                 relation_types: Optional[List[str]] = None, relation_embed_dim: int = 4):
        super().__init__()
        self.dim = dim
        self.max_nodes = max_nodes

        self.node_emb = nn.Parameter(torch.zeros(0, dim))

        self.relation_types = list(relation_types) if relation_types else list(RELATION_TYPES)
        self.relation_to_idx = {r: i for i, r in enumerate(self.relation_types)}
        self.relation_embedding = nn.Embedding(len(self.relation_types), relation_embed_dim)
        edge_dim = 1 + relation_embed_dim

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_dim = dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else dim
            self.layers.append(GATv2Conv(in_dim, out_dim, heads=num_heads, concat=False, edge_dim=edge_dim))
            self.norms.append(LayerNorm(out_dim))
            in_dim = out_dim

        self.act = nn.ELU()

        self._edges: List[Tuple[int, int]] = []
        self._edge_weights = nn.ParameterList()
        self._edge_relations: List[int] = []
        self._edge_index = None
        self._adjacency: Dict[int, List[Tuple[int, float]]] = {}

        self.node_labels: Dict[int, str] = {}
        self.node_types: Dict[int, NodeType] = {}
        self.node_clusters: Dict[int, str] = {}

    # ---------- Рост графа ----------
    def add_node(self, embedding: torch.Tensor, label: str = "", cluster: str = "hidden",
                 layer: int = 0, node_type: NodeType = NodeType.CONCEPT,
                 optimizer=None) -> int:
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

    def _rebuild_edges(self):
        if not self._edges:
            self._edge_index = torch.zeros((2, 0), dtype=torch.long)
            self._adjacency = {}
            return
        u = [int(f) - 1 for f, _ in self._edges]
        v = [int(t) - 1 for _, t in self._edges]
        self._edge_index = torch.tensor([u, v], dtype=torch.long)

        adjacency: Dict[int, List[Tuple[int, float]]] = {}
        for (f, t), w_param in zip(self._edges, self._edge_weights):
            f = int(f)
            t = int(t)
            w = float(w_param.detach())
            adjacency.setdefault(f, []).append((t, w))
            adjacency.setdefault(t, []).append((f, w * 0.5))
        self._adjacency = adjacency

    def add_synapse(self, from_id: int, to_id: int, weight: float = 0.1,
                     relation: str = "has_answer", optimizer=None) -> int:
        from_id = int(from_id)
        to_id = int(to_id)
        rel_idx = self.relation_to_idx.get(relation)
        if rel_idx is None:
            print(f"[Graph] Неизвестный тип отношения '{relation}', использую '{self.relation_types[0]}'")
            rel_idx = 0
        self._edges.append((from_id, to_id))
        self._edge_relations.append(rel_idx)
        new_w = nn.Parameter(torch.tensor(weight, dtype=torch.float))
        self._edge_weights.append(new_w)
        self._rebuild_edges()
        if optimizer is not None and optimizer.param_groups:
            optimizer.param_groups[0]["params"].append(new_w)
        return len(self._edges) - 1

    # ---------- "Мышление" ----------
    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x is None:
            x = self.node_emb
        if x.shape[0] == 0:
            return x
        if self._edge_index is None or self._edge_index.size(1) == 0:
            return x

        if len(self._edge_weights) > 0:
            weights = torch.stack([w for w in self._edge_weights]).view(-1, 1)
            rel_idx = torch.tensor(self._edge_relations, dtype=torch.long, device=weights.device)
            rel_emb = self.relation_embedding(rel_idx)
            edge_attr = torch.cat([weights, rel_emb], dim=1)
        else:
            edge_attr = torch.zeros((0, 1 + self.relation_embedding.embedding_dim), device=x.device)

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

    def spreading_activation(self, start_id: int, steps: int = 3, decay: float = 0.6,
                              top_k: int = 6, min_activation: float = 0.05) -> List[Tuple[int, float]]:
        n = self.node_emb.shape[0]
        if n == 0 or start_id < 1 or start_id > n:
            return []

        activation = torch.zeros(n)
        activation[start_id - 1] = 1.0

        for _ in range(steps):
            new_activation = activation.clone() * 0.25
            nonzero = (activation > min_activation).nonzero(as_tuple=True)[0]
            for idx in nonzero.tolist():
                nid = idx + 1
                a = activation[idx].item()
                for neighbor, w in self._adjacency.get(nid, []):
                    if 1 <= neighbor <= n:
                        new_activation[neighbor - 1] += a * w * decay
            activation = torch.clamp(new_activation, max=5.0)

        activation[start_id - 1] = 0.0
        k = min(top_k, n)
        if k == 0:
            return []
        topk_vals, topk_idx = torch.topk(activation, k)
        return [
            (int(idx.item()) + 1, float(val.item()))
            for idx, val in zip(topk_idx, topk_vals)
            if val.item() > min_activation
        ]

    def get_node_embeddings(self) -> torch.Tensor:
        return self.node_emb

    def get_edge_weights(self) -> torch.Tensor:
        if len(self._edge_weights) == 0:
            return torch.tensor([])
        weights = [w for w in self._edge_weights]
        return torch.cat([w.view(1) for w in weights])

    def find_most_similar(self, query: torch.Tensor, threshold: float = 0.87) -> Optional[int]:
        if self.node_emb.shape[0] == 0:
            return None
        sim = F.cosine_similarity(query.unsqueeze(0), self.node_emb, dim=1)
        best_idx = torch.argmax(sim).item()
        if sim[best_idx] >= threshold:
            return best_idx + 1
        return None

    def find_most_similar_contextual(self, query: torch.Tensor, threshold: float = 0.6,
                                      max_nodes_for_gnn: Optional[int] = None) -> Optional[int]:
        if self.node_emb.shape[0] == 0:
            return None
        if max_nodes_for_gnn is not None and self.node_emb.shape[0] > max_nodes_for_gnn:
            return self.find_most_similar(query, threshold=threshold)
        with torch.no_grad():
            h = self.forward()
        if h.shape[0] == 0:
            return None
        sim = F.cosine_similarity(query.unsqueeze(0), h, dim=1)
        best_idx = torch.argmax(sim).item()
        if sim[best_idx] >= threshold:
            return best_idx + 1
        return None

    # ---------- НОВЫЕ МЕТОДЫ ДЛЯ РАБОТЫ С РЁБРАМИ ----------
    def increment_synapse(self, from_id: int, to_id: int, delta: float, relation: str = "has_answer") -> bool:
        """
        Увеличить или уменьшить вес существующего ребра на delta.
        Возвращает True, если ребро найдено и обновлено.
        """
        from_id = int(from_id)
        to_id = int(to_id)
        rel_idx = self.relation_to_idx.get(relation, 0)
        for i, (f, t) in enumerate(self._edges):
            if f == from_id and t == to_id and self._edge_relations[i] == rel_idx:
                new_weight = self._edge_weights[i].item() + delta
                new_weight = max(-1.0, min(1.0, new_weight))
                self._edge_weights[i] = nn.Parameter(torch.tensor(new_weight, dtype=torch.float))
                self._rebuild_edges()
                return True
        return False

    def get_edges_between(self, nid1: int, nid2: int) -> List[Tuple[int, int, float, str]]:
        """Возвращает все рёбра между двумя узлами с их типами."""
        results = []
        for i, (f, t) in enumerate(self._edges):
            if (f == nid1 and t == nid2) or (f == nid2 and t == nid1):
                rel = self.relation_types[self._edge_relations[i]]
                results.append((f, t, self._edge_weights[i].item(), rel))
        return results

    def get_neighbors_with_weights(self, node_id: int) -> List[Tuple[int, float]]:
        """Возвращает список (сосед, вес) для данного узла."""
        return self._adjacency.get(node_id, [])

    def get_all_edges(self) -> List[Tuple[int, int, float]]:
        """Возвращает все рёбра в виде (from, to, weight)."""
        return [(f, t, w.item()) for (f, t), w in zip(self._edges, self._edge_weights)]


# ----------------------------------------------------------------------
# Иерархический граф – обёртки для новых методов
# ----------------------------------------------------------------------
class HierarchicalGraph(nn.Module):
    def __init__(self, dims: List[int], num_heads: int = 4, num_layers: int = 2,
                 cluster_threshold: float = 0.75, min_cluster_size: int = 2):
        super().__init__()
        self.dims = dims
        self.levels = nn.ModuleList()
        self.level_projections = nn.ModuleList()
        self.up_projections = nn.ModuleList()

        self.cluster_threshold = cluster_threshold
        self.min_cluster_size = min_cluster_size

        for i, d in enumerate(dims):
            level_graph = DifferentiableNeuralGraph(
                dim=d,
                max_nodes=5000,
                hidden_dim=d,
                num_heads=num_heads,
                num_layers=num_layers
            )
            self.levels.append(level_graph)
            if i < len(dims) - 1:
                self.level_projections.append(nn.Linear(d, dims[i + 1]))
                self.up_projections.append(nn.Linear(dims[i + 1], d))

        self.cluster_of: List[Dict[int, int]] = [dict() for _ in range(len(dims) - 1)]

    def add_node(self, embedding: torch.Tensor, label: str = "", cluster: str = "hidden",
                 layer: int = 0, node_type: NodeType = NodeType.CONCEPT,
                 level_idx: int = 0, optimizer=None) -> int:
        return self.levels[level_idx].add_node(embedding, label, cluster, layer, node_type, optimizer=optimizer)

    def add_synapse(self, from_id: int, to_id: int, weight: float = 0.1,
                    relation: str = "has_answer", level_idx: int = 0, optimizer=None) -> int:
        return self.levels[level_idx].add_synapse(from_id, to_id, weight, relation=relation, optimizer=optimizer)

    def find_most_similar(self, query: torch.Tensor, level_idx: int = 0, threshold: float = 0.87) -> Optional[int]:
        return self.levels[level_idx].find_most_similar(query, threshold)

    def find_most_similar_contextual(self, query: torch.Tensor, level_idx: int = 0,
                                      threshold: float = 0.6,
                                      max_nodes_for_gnn: Optional[int] = None) -> Optional[int]:
        if level_idx != 0:
            return self.levels[level_idx].find_most_similar_contextual(
                query, threshold, max_nodes_for_gnn=max_nodes_for_gnn)
        level0 = self.levels[0]
        if level0.node_emb.shape[0] == 0:
            return None
        if max_nodes_for_gnn is not None and level0.node_emb.shape[0] > max_nodes_for_gnn:
            return level0.find_most_similar(query, threshold=threshold)
        with torch.no_grad():
            h = self.forward()
        if h.shape[0] == 0:
            return None
        sim = F.cosine_similarity(query.unsqueeze(0), h, dim=1)
        best_idx = torch.argmax(sim).item()
        if sim[best_idx] >= threshold:
            return best_idx + 1
        return None

    def spreading_activation(self, start_id: int, level_idx: int = 0, **kwargs) -> List[Tuple[int, float]]:
        return self.levels[level_idx].spreading_activation(start_id, **kwargs)

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        hs = [self.levels[0].forward(x)]
        for i in range(1, len(self.levels)):
            hs.append(self.levels[i].forward())

        for i in range(len(self.levels) - 2, -1, -1):
            if i >= len(self.cluster_of) or not self.cluster_of[i]:
                continue
            if hs[i].shape[0] == 0 or hs[i + 1].shape[0] == 0:
                continue
            up = self.up_projections[i](hs[i + 1])
            modulation = torch.zeros_like(hs[i])
            for node_idx, cluster_id in self.cluster_of[i].items():
                if node_idx < hs[i].shape[0] and (cluster_id - 1) < up.shape[0]:
                    modulation[node_idx] = up[cluster_id - 1]
            hs[i] = hs[i] + 0.2 * modulation
        return hs[0]

    def rebuild_hierarchy(self, optimizer=None, chunk_size: int = 2048):
        for i in range(len(self.levels) - 1):
            lower = self.levels[i]
            n = lower.node_emb.shape[0]
            if n == 0:
                continue
            with torch.no_grad():
                h = lower.forward()
                proj = self.level_projections[i](h)
                proj = F.normalize(proj, p=2, dim=1)

                assigned = torch.zeros(n, dtype=torch.bool, device=proj.device)
                clusters: List[List[int]] = []
                for start in range(0, n, chunk_size):
                    end = min(start + chunk_size, n)
                    sim_block = proj[start:end] @ proj.T
                    for local_a in range(end - start):
                        a = start + local_a
                        if assigned[a]:
                            continue
                        mask = (sim_block[local_a] >= self.cluster_threshold) & (~assigned)
                        mask[a] = True
                        members = mask.nonzero(as_tuple=True)[0].tolist()
                        assigned[members] = True
                        clusters.append(members)

            upper = self.levels[i + 1]
            new_assignment: Dict[int, int] = {}
            for cluster in clusters:
                if len(cluster) < self.min_cluster_size:
                    continue
                centroid = F.normalize(proj[cluster].mean(dim=0), p=2, dim=0)
                label = f"cluster_of_{len(cluster)}"
                cid = upper.add_node(centroid.detach(), label=label, cluster="abstraction",
                                      layer=i + 1, node_type=NodeType.CONCEPT, optimizer=optimizer)
                for member in cluster:
                    new_assignment[member] = cid
            self.cluster_of[i] = new_assignment

    def get_level_embeddings(self, level_idx: int) -> torch.Tensor:
        return self.levels[level_idx].node_emb

    def get_edge_weights(self) -> torch.Tensor:
        return self.levels[0].get_edge_weights()

    # ---------- ОБЁРТКИ ДЛЯ НОВЫХ МЕТОДОВ ----------
    def increment_synapse(self, from_id: int, to_id: int, delta: float, relation: str = "has_answer", level_idx: int = 0) -> bool:
        return self.levels[level_idx].increment_synapse(from_id, to_id, delta, relation)

    def get_edges_between(self, nid1: int, nid2: int, level_idx: int = 0) -> List[Tuple[int, int, float, str]]:
        return self.levels[level_idx].get_edges_between(nid1, nid2)

    def get_neighbors_with_weights(self, node_id: int, level_idx: int = 0) -> List[Tuple[int, float]]:
        return self.levels[level_idx].get_neighbors_with_weights(node_id)

    def get_all_edges(self, level_idx: int = 0) -> List[Tuple[int, int, float]]:
        return self.levels[level_idx].get_all_edges()

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