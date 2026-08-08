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

        # ИСПРАВЛЕНО: раньше здесь создавался "нейрон-призрак" —
        # nn.Parameter(torch.randn(1, dim) * 0.01) — случайный, без label/type,
        # ни с чем не связанный синапсами. Он не участвовал в message passing
        # (нет рёбер), но раньше (в HierarchicalGraph) участвовал в глобальном
        # self-attention по ВСЕМ узлам и постоянно подмешивал случайный шум в
        # контекст остальных нейронов. Граф теперь стартует пустым — первый
        # add_node создаёт узел с id=1, а не "въезжает" во второй слот после мусора.
        self.node_emb = nn.Parameter(torch.zeros(0, dim))

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
        # Кэш смежности для spreading activation (не требует градиента — только для "мышления")
        self._adjacency: Dict[int, List[Tuple[int, float]]] = {}

        self.node_labels: Dict[int, str] = {}
        self.node_types: Dict[int, NodeType] = {}
        self.node_clusters: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # Рост графа (память / обучаемость)
    # ------------------------------------------------------------------
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

    def add_synapse(self, from_id: int, to_id: int, weight: float = 0.1, optimizer=None) -> int:
        self._edges.append((from_id, to_id))
        new_w = nn.Parameter(torch.tensor(weight, dtype=torch.float))
        self._edge_weights.append(new_w)
        self._rebuild_edges()

        if optimizer is not None and optimizer.param_groups:
            optimizer.param_groups[0]["params"].append(new_w)
        return len(self._edges) - 1

    def _rebuild_edges(self):
        if not self._edges:
            self._edge_index = torch.zeros((2, 0), dtype=torch.long)
            self._adjacency = {}
            return
        u = [f - 1 for f, _ in self._edges]
        v = [t - 1 for _, t in self._edges]
        self._edge_index = torch.tensor([u, v], dtype=torch.long)

        # Смежность для spreading activation — синапс трактуем как направленный,
        # но с более слабой "обратной тягой", как в биологических сетях (обратный
        # сигнал слабее прямого, но не нулевой — иначе граф был бы чисто иерархическим).
        adjacency: Dict[int, List[Tuple[int, float]]] = {}
        for (f, t), w_param in zip(self._edges, self._edge_weights):
            w = float(w_param.detach())
            adjacency.setdefault(f, []).append((t, w))
            adjacency.setdefault(t, []).append((f, w * 0.5))
        self._adjacency = adjacency

    # ------------------------------------------------------------------
    # "Мышление" — message passing + spreading activation
    # ------------------------------------------------------------------
    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Контекстуализирует эмбеддинг каждого узла его соседями через GATv2.
        Это единственное место, где реально обучаются GAT/LayerNorm — их выход
        теперь используется и для сравнения похожести, и в contrastive loss,
        а не отбрасывается.
        """
        if x is None:
            x = self.node_emb
        if x.shape[0] == 0:
            return x
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

    def spreading_activation(self, start_id: int, steps: int = 3, decay: float = 0.6,
                              top_k: int = 6, min_activation: float = 0.05) -> List[Tuple[int, float]]:
        """
        Это и есть "мышление" графа: активация запускается в узле start_id и растекается
        по синапсам на steps шагов, ослабевая с decay на каждом хопе и по весу ребра.
        Возвращает top_k наиболее "возбуждённых" узлов (кроме самого start_id) —
        это ассоциативный поток мыслей графа, который дальше LLM просто озвучивает.

        Не участвует в backward — это управляющая логика "что вспомнить", а не обучаемый вес.
        """
        n = self.node_emb.shape[0]
        if n == 0 or start_id < 1 or start_id > n:
            return []

        activation = torch.zeros(n)
        activation[start_id - 1] = 1.0

        for _ in range(steps):
            new_activation = activation.clone() * 0.25  # остаточное затухание (забывание)
            nonzero = (activation > min_activation).nonzero(as_tuple=True)[0]
            for idx in nonzero.tolist():
                nid = idx + 1
                a = activation[idx].item()
                for neighbor, w in self._adjacency.get(nid, []):
                    if 1 <= neighbor <= n:
                        new_activation[neighbor - 1] += a * w * decay
            activation = torch.clamp(new_activation, max=5.0)

        activation[start_id - 1] = 0.0  # сам триггер не возвращаем как "ассоциацию"
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
        """Строгое сравнение по СЫРОМУ содержимому узла — используется при обучении,
        чтобы не плодить дублирующие узлы для почти идентичного текста."""
        if self.node_emb.shape[0] == 0:
            return None
        sim = F.cosine_similarity(query.unsqueeze(0), self.node_emb, dim=1)
        best_idx = torch.argmax(sim).item()
        if sim[best_idx] >= threshold:
            return best_idx + 1
        return None

    def find_most_similar_contextual(self, query: torch.Tensor, threshold: float = 0.6) -> Optional[int]:
        """
        Сравнение по КОНТЕКСТУАЛИЗИРОВАННОМУ (после message passing) эмбеддингу узла —
        используется при поиске точки входа для "мышления"/ответа.
        """
        if self.node_emb.shape[0] == 0:
            return None
        with torch.no_grad():
            h = self.forward()
        if h.shape[0] == 0:
            return None
        sim = F.cosine_similarity(query.unsqueeze(0), h, dim=1)
        best_idx = torch.argmax(sim).item()
        if sim[best_idx] >= threshold:
            return best_idx + 1
        return None


# ----------------------------------------------------------------------
# Иерархический граф с несколькими уровнями.
#
# ПЕРЕАНАЛИЗИРОВАНО: раньше forward() честно считал все уровни, но возвращал
# только outputs[0] — уровни 1/2 не влияли ни на loss, ни на ответ (мёртвый
# код), а "иерархическая" связь между уровнями была подменена ГЛОБАЛЬНЫМ
# nn.MultiheadAttention по ВСЕМ узлам сразу — это игнорировало синапсы
# полностью (O(N^2), без учёта топологии) и было прямой причиной
# нестабильности обучения (см. чат).
#
# Идея сохранена и достроена: уровень 0 — граф фактов, обучаемый через
# синапсы (GAT). Уровни выше — это НЕ отдельные независимые графы, а
# АБСТРАКЦИИ уровня 0: узел уровня i+1 = центроид кластера близких по смыслу
# узлов уровня i (см. rebuild_hierarchy — вызывается из sleep(), не на
# каждом шаге, т.к. кластеризация — это дорого и не обучаемый градиентом
# процесс, а периодическая консолидация, как и должно быть в модели памяти).
# Верхний уровень затем влияет на нижний TOP-DOWN модуляцией его
# контекстуализированного эмбеддинга — это и есть настоящая иерархия.
# ----------------------------------------------------------------------
class HierarchicalGraph(nn.Module):
    def __init__(self, dims: List[int], num_heads: int = 4, num_layers: int = 2,
                 cluster_threshold: float = 0.75, min_cluster_size: int = 2):
        super().__init__()
        self.dims = dims
        self.levels = nn.ModuleList()
        self.level_projections = nn.ModuleList()  # уровень i -> размерность уровня i+1 (вниз, для кластеризации)
        self.up_projections = nn.ModuleList()      # уровень i+1 -> размерность уровня i (наверх->вниз, модуляция)

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

        # cluster_of[i][node_idx_0based_на_уровне_i] = node_id_1based_на_уровне_i+1
        self.cluster_of: List[Dict[int, int]] = [dict() for _ in range(len(dims) - 1)]

    def add_node(self, embedding: torch.Tensor, label: str = "", cluster: str = "hidden",
                 layer: int = 0, node_type: NodeType = NodeType.CONCEPT,
                 level_idx: int = 0, optimizer=None) -> int:
        return self.levels[level_idx].add_node(embedding, label, cluster, layer, node_type,
                                                 optimizer=optimizer)

    def add_synapse(self, from_id: int, to_id: int, weight: float = 0.1,
                     level_idx: int = 0, optimizer=None) -> int:
        return self.levels[level_idx].add_synapse(from_id, to_id, weight, optimizer=optimizer)

    def find_most_similar(self, query: torch.Tensor, level_idx: int = 0, threshold: float = 0.87) -> Optional[int]:
        return self.levels[level_idx].find_most_similar(query, threshold)

    def find_most_similar_contextual(self, query: torch.Tensor, level_idx: int = 0,
                                      threshold: float = 0.6) -> Optional[int]:
        """
        ИЗМЕНЕНО: для level_idx=0 точка входа теперь ищется по ПОЛНОМУ иерархическому
        forward() (с top-down модуляцией верхних уровней), а не по локальному GAT
        уровня 0 в изоляции — верхние уровни (темы/абстракции) реально участвуют
        в том, какой нейрон граф сочтёт "похожим" на вопрос.
        """
        if level_idx != 0:
            return self.levels[level_idx].find_most_similar_contextual(query, threshold)
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

    # ------------------------------------------------------------------
    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Снизу вверх: каждый уровень контекстуализируется своим собственным GAT
        по своим синапсам (никакого глобального attention по чужим узлам).
        Сверху вниз: если для узлов уровня i есть кластеризация на уровень i+1
        (построена в rebuild_hierarchy), их эмбеддинг модулируется спроецированным
        вниз эмбеддингом их кластера-абстракции.
        """
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

    def rebuild_hierarchy(self, optimizer=None):
        """
        Периодическая (не end-to-end дифференцируемая) переагрегация иерархии.
        Вызывать из sleep(), а не на каждом шаге — кластеризация стоит O(n^2)
        по числу узлов уровня и не должна идти в backward-графе.

        Для каждого уровня i (кроме последнего): берём контекстуализированные
        (после GAT) эмбеддинги узлов, проецируем в размерность уровня i+1,
        жадно группируем по порогу косинусной близости (cluster_threshold).
        Каждый кластер размера >= min_cluster_size поднимается как один узел
        уровня i+1 (центроид), и запоминается сопоставление cluster_of[i].
        Кластеры из одного узла не поднимаются — единичный факт не абстракция.
        """
        for i in range(len(self.levels) - 1):
            lower = self.levels[i]
            n = lower.node_emb.shape[0]
            if n == 0:
                continue
            with torch.no_grad():
                h = lower.forward()
                proj = self.level_projections[i](h)
                proj = F.normalize(proj, p=2, dim=1)

            assigned = [False] * n
            clusters: List[List[int]] = []
            for a in range(n):
                if assigned[a]:
                    continue
                cluster = [a]
                assigned[a] = True
                for b in range(a + 1, n):
                    if assigned[b]:
                        continue
                    sim = float(F.cosine_similarity(proj[a].unsqueeze(0), proj[b].unsqueeze(0)))
                    if sim >= self.cluster_threshold:
                        cluster.append(b)
                        assigned[b] = True
                clusters.append(cluster)

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