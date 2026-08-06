# brain/config.py
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class BrainConfig:
    # Размерность эмбеддингов (лучше использовать 384/768 для современных моделей)
    dim_embedding: int = 384
    max_kb_size: int = 10000

    # Архитектура графа (GNN)
    gnn_hidden_dim: int = 256
    gnn_num_layers: int = 3
    gnn_num_heads: int = 4
    max_neurons: int = 100000
    max_synapses: int = 500000

    # Память
    working_memory_size: int = 10
    episodic_capacity: int = 50000  # увеличено для FAISS HNSW
    semantic_graph_path: str = "semantic_graph.pth"

    # Модель эмбеддингов (поддерживаются: "intfloat/e5-large-v2", "BAAI/bge-large-en-v1.5")
    embedding_model: str = "intfloat/e5-large-v2"
    # LLM – локальная или через API
    llm_model: str = "Qwen/Qwen2-7B-Instruct"   # или "meta-llama/Llama-2-7b-chat-hf"
    use_openai_api: bool = False                # True для GPT-4o-mini
    openai_api_key: Optional[str] = None

    # Обучение
    learning_rate: float = 1e-4
    contrastive_margin: float = 0.5
    meta_lr: float = 0.01

    # Прочее
    model_dir: str = "brain_model_v8"
    checkpoint_every: int = 50
    forget_threshold_access: int = 2        # забывать после такого числа обращений
    forget_threshold_days: int = 30         # и старше этого числа дней