from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class BrainConfig:
    # Размеры
    dim_embedding: int = 1024
    max_kb_size: int = 10000

    # Граф
    gnn_hidden_dim: int = 1024
    gnn_num_layers: int = 3
    gnn_num_heads: int = 4
    max_neurons: int = 100000
    gnn_contextual_max_nodes: int = 20000
    max_synapses: int = 500000
    use_hierarchical_graph: bool = True
    graph_levels: List[int] = field(default_factory=lambda: [1024, 512, 256])
    attention_heads: int = 8

    hierarchy_cluster_threshold: float = 0.75
    hierarchy_min_cluster_size: int = 2

    # Память
    working_memory_size: int = 10
    episodic_capacity: int = 50000
    semantic_memory_capacity: int = 10000
    semantic_graph_path: str = "semantic_graph.pth"

    # Модели
    embedding_model: str = "intfloat/e5-large-v2"
    llm_model: str = "Qwen/Qwen2-7B-Instruct"
    use_openai_api: bool = False
    openai_api_key: Optional[str] = None
    llm_base_url: Optional[str] = "http://192.168.0.13:1234/v1"

    # Параметры сэмплинга (фиксированные, без динамики)
    llm_top_p: float = 0.8
    llm_top_k: int = 20
    llm_repetition_penalty: float = 1.05
    presence_penalty: float = 0.0
    enable_thinking: bool = True

    # Инструменты
    enable_tools: bool = True
    openweather_api_key: Optional[str] = None

    # Кэш эмбеддингов
    embedding_cache_path: str = "embedding_cache.pkl"

    # Обучение
    learning_rate: float = 1e-4
    contrastive_margin: float = 0.5
    meta_lr: float = 0.01
    contrastive_num_negatives: int = 8
    node_merge_threshold: float = 0.87

    model_dir: str = "brain_model_v10"
    checkpoint_every: int = 50
    forget_threshold_access: int = 2
    forget_threshold_days: int = 30

    enable_ewc: bool = True
    ewc_lambda: float = 0.1

    # Рефлексия (оставлена, но теперь вызывается только через Teacher)
    enable_reflection: bool = True

    # Проактивные мысли (опционально)
    proactive_enabled: bool = True
    proactive_interval_seconds: int = 60
    proactive_steps: int = 3
    proactive_top_k: int = 6
    proactive_reward: float = 0.5

    # Двухуровневый ответ
    two_level_answer: bool = True
    summary_max_tokens: int = 250
    summary_temperature: float = 0.3

    # Модули для агента (опциональны, но оставлены)
    enable_emotion: bool = True
    enable_user_model: bool = True
    enable_motivation: bool = True

    # Для агента (не влияют на мозг)
    self_play_rounds: int = 3
    exploration_temperature: float = 0.9