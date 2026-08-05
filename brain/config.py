# brain/config.py
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class BrainConfig:
    # Размерность эмбеддингов
    dim_embedding: int = 128
    max_kb_size: int = 5000
    # Архитектура графа (GNN)
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 2
    gnn_num_heads: int = 4
    max_neurons: int = 5000
    max_synapses: int = 30000

    # Память
    working_memory_size: int = 5
    episodic_capacity: int = 500
    semantic_graph_path: str = "semantic_graph.pth"

    # TinyLLM (не используется, но можно оставить)
    tiny_llm_model: str = "Qwen/Qwen2.5-1.5B"
    tiny_llm_lora_r: int = 16
    tiny_llm_lora_alpha: int = 32
    tiny_llm_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Обучение
    learning_rate: float = 1e-4
    contrastive_margin: float = 0.5
    meta_lr: float = 0.01

    # Прочее
    model_dir: str = "brain_model_v7"
    checkpoint_every: int = 100