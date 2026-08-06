# brain/config.py
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class BrainConfig:
    # Базовые размерности
    dim_embedding: int = 384
    max_kb_size: int = 10000

    # Параметры графа (новые)
    gnn_hidden_dim: int = 256
    gnn_num_layers: int = 3
    gnn_num_heads: int = 4
    max_neurons: int = 100000
    max_synapses: int = 500000
    use_hierarchical_graph: bool = True          # использовать иерархический граф
    graph_levels: List[int] = field(default_factory=lambda: [384, 256, 128])
    attention_heads: int = 8                     # для self-attention

    # Память
    working_memory_size: int = 10
    episodic_capacity: int = 50000
    semantic_memory_capacity: int = 10000       # для семантического графа
    semantic_graph_path: str = "semantic_graph.pth"

    # Модели
    embedding_model: str = "intfloat/e5-large-v2"
    llm_model: str = "Qwen/Qwen2-7B-Instruct"
    use_openai_api: bool = False
    openai_api_key: Optional[str] = None

    # Оптимизация
    learning_rate: float = 1e-4
    contrastive_margin: float = 0.5
    meta_lr: float = 0.01

    # Сохранение / забывание
    model_dir: str = "brain_model_v9"
    checkpoint_every: int = 50
    forget_threshold_access: int = 2
    forget_threshold_days: int = 30

    # Улучшенные модули (включены по умолчанию)
    enable_curiosity: bool = True
    curiosity_lr: float = 0.01
    enable_planning: bool = True
    enable_reflection: bool = True
    enable_ewc: bool = True
    ewc_lambda: float = 0.1

    # Агентские параметры
    self_play_rounds: int = 3
    exploration_temperature: float = 0.9