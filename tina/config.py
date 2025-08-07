from dataclasses import dataclass, field
from typing import Literal


# check ./recipes/BASE_MODEL_NAME/PT_TYPE/train_model_XXXX.yaml
@dataclass
class ModelPTConfig:
    # //*******Model post-training configs*******//
    model_post_train_type: Literal["grpo", "sft"] = field(default="grpo")
    model_post_train_dataset_name: str = field(default="still")
    model_post_train_dataset_config: str | None = field(default=None)
    trace_free: bool = field(default=True)

    rl_post_train_reward_funcs: list[str] = field(default_factory=lambda: ["accuracy", "format"])
    rl_post_train_reward_weights: list[str] = field(default_factory=lambda: [2.0, 1.0])
    cosine_min_value_wrong: float = field(default=0.0)
    cosine_max_value_wrong: float = field(default=-0.5)
    cosine_min_value_correct: float = field(default=0.5)
    cosine_max_value_correct: float = field(default=1.0)
    cosine_max_len: int = field(default=1000)
    repetition_n_grams: int = field(default=3)
    repetition_max_penalty: float = field(default=-1.0)


# check ./recipes/BASE_MODEL_NAME/PT_TYPE/train_sae_XXXX.yaml
@dataclass
class SAEConfig:
    # //*******SAE configs*******//
    seed: int = field(default=42)

    base_model_name: str = field(default="DeepSeek-R1-Distill-Qwen-1.5B")
    source_model_post_train_dataset_name: str = field(default="still")
    source_model_post_train_type: Literal["grpo", "sft"]  = field(default="grpo")
    source_model_checkpoints: list[str] = field(default_factory=lambda: ["checkpoint-500"])

    sae_name: str = field(default="sae-DeepSeek-R1-Distill-Qwen-1.5B-65k")
    sae_expansion_factor: int = field(default=32)
    sae_num_latents: int = field(default=131072)
    sae_hookpoints: list[str] = field(default_factory=lambda: ["model.layers.0"])
    trigger_dataset_name: str = field(default="still")


# check ./recipes/BASE_MODEL_NAME/PT_TYPE/sae_tuning_XXXX.yaml
@dataclass
class SAETuningConfig:
    # //*******SAE-Tuning configs*******//
    seed: int = field(default=42)

    # source model
    base_model_name: str = field(default="DeepSeek-R1-Distill-Qwen-1.5B")
    source_model_post_train_dataset_name: str = field(default="still")
    source_model_post_train_type: Literal["grpo", "sft", "base"]  = field(default="grpo")
    source_model_checkpoint: str = field(default="checkpoint-500")

    # sae
    sae_name: str = field(default="sae-DeepSeek-R1-Distill-Qwen-1.5B-65k")
    sae_hookpoint: str = field(default="model.layers.0")
    trigger_dataset_name: str = field(default="still")
    sae_type: Literal["finetuned", "pretrained"] = field(default="finetuned")

    # target model
    target_model_name: str = field(default="DeepSeek-R1-Distill-Qwen-1.5B")  # Qwen2.5-Math-1.5B, Qwen2.5-1.5B
    elicitation_dataset_name: str = field(default="still")

    lora_r: int = field(default=32)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"])
    logging_steps: int = field(default=1)
    learning_rate: float = field(default=5e-5)
    num_epochs: int = field(default=5)
    batch_size: int = field(default=1)
    save_steps: int = field(default=50)