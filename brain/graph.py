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

        self.register_buffer("node_emb", torch.zeros(1, dim))
        self.register_buffer("node_mask", torch.zeros(1, dtype=torch.bool))
        self.node_emb = nn.Parameter(self.node_emb)

        self.gat1 = GATConv(dim, hidden_dim, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_dim * num_heads, dim, heads=1, concat=False)
        self.update_scale = nn.Parameter(torch.tensor(0.1))

        self.neuron_labels = {}
        self.neuron_clusters = {}
        self.neuron_layers = {}
        self._next_nid = 1
        self._edges = []  # list of (from_id, to_id, weight)

    def add_node(self, embedding: torch.Tensor, label: str = "", cluster: str = "hidden", layer: int = 0) -> int:
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
        return nid

    def add_synapse(self, from_id: int, to_id: int, weight: float = 0.1) -> int:
        self._edges.append((from_id, to_id, weight))
        return len(self._edges) - 1

    def get_edges_tensor(self) -> torch.Tensor:
        if not self._edges:
            return torch.zeros((2, 0), dtype=torch.long)
        u = [e[0]-1 for e in self._edges]
        v = [e[1]-1 for e in self._edges]
        return torch.tensor([u, v], dtype=torch.long)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge_index = self.get_edges_tensor()
        if edge_index.size(1) == 0:
            return x
        h = F.elu(self.gat1(x, edge_index))
        h = F.elu(self.gat2(h, edge_index))
        delta = torch.tanh(self.update_scale) * (h - x)
        return x + delta

    def get_node_embeddings(self) -> torch.Tensor:
        return self.node_emb

    def get_embedding_by_id(self, nid: int) -> torch.Tensor:
        return self.node_emb[nid-1]

    def find_most_similar(self, query: torch.Tensor, threshold: float = 0.8) -> Optional[int]:
        if self.node_emb.shape[0] == 0:
            return None
        sim = F.cosine_similarity(query.unsqueeze(0), self.node_emb, dim=1)
        best_idx = torch.argmax(sim).item()
        if sim[best_idx] >= threshold:
            return best_idx + 1
        return None