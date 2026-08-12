# brain/config.py
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class BrainConfig:
    dim_embedding: int = 1024
    max_kb_size: int = 10000

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

    working_memory_size: int = 10
    episodic_capacity: int = 50000
    semantic_memory_capacity: int = 10000
    semantic_graph_path: str = "semantic_graph.pth"

    embedding_model: str = "intfloat/e5-large-v2"
    llm_model: str = "Qwen/Qwen2-7B-Instruct"
    use_openai_api: bool = False
    openai_api_key: Optional[str] = None
    llm_base_url: Optional[str] = "http://192.168.0.13:1234/v1"

    # ---------- Сэмплинг для Qwen3.5 (обновлено) ----------
    llm_top_p: float = 0.8
    llm_top_k: int = 20
    llm_repetition_penalty: float = 1.05
    presence_penalty: float = 0.0           # стандартный параметр OpenAI
    enable_thinking: bool = True            # включать ли thinking mode (Qwen3.5)

    learning_rate: float = 1e-4
    contrastive_margin: float = 0.5
    meta_lr: float = 0.01
    contrastive_num_negatives: int = 8
    node_merge_threshold: float = 0.87

    model_dir: str = "brain_model_v10"
    checkpoint_every: int = 50
    forget_threshold_access: int = 2
    forget_threshold_days: int = 30

    enable_curiosity: bool = True
    curiosity_lr: float = 0.01
    enable_planning: bool = True
    enable_reflection: bool = True
    enable_ewc: bool = True
    ewc_lambda: float = 0.1

    self_play_rounds: int = 3
    exploration_temperature: float = 0.9

    proactive_enabled: bool = True
    proactive_interval_seconds: int = 60
    proactive_steps: int = 3
    proactive_top_k: int = 6
    proactive_reward: float = 0.5

    two_level_answer: bool = True
    summary_max_tokens: int = 250
    summary_temperature: float = 0.3

    auto_memory_capacity: int = 500
    self_model_dim: int = 512
    enable_metacognition: bool = True
    enable_emotion: bool = True
    enable_user_model: bool = True
    enable_motivation: bool = True
    global_workspace_capacity: int = 20