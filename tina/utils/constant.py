

LLM_SAE_PAIRS = {
    "DeepSeek-R1-Distill-Qwen-1.5B": [[
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k",  # layers.0-27.mlp
    ]],
}

# problem/question, (solution), answer => combined text for uninstructed models
RL_POST_TRAIN_CONFIG_MAP = {
    # Main datasets
    "deepscaler": "agentica-org/DeepScaleR-Preview-Dataset", # 40.3k
    "2thought": "Intelligent-Internet/II-Thought-RL-v0-Math-50K", # 53.3k
    # "l1_exact": "agentica-org/DeepScaleR-Preview-Dataset", # 40.3k
    # "l1_max": "agentica-org/DeepScaleR-Preview-Dataset", # 40.3k
    # "fastcurl": "Nickyang/FastCuRL", # 80.6k
    "still": "RUC-AIBOX/STILL-3-Preview-RL-Data", # 33k
    "open_rs3": "knoveleng/open-rs", # 7k => combine s1 dataset and deepscaler
    "open_rs2": "knoveleng/open-rs", # 7k => combine s1 dataset and deepscaler
    "open_rs1": "knoveleng/open-s1", # 18.6k => from s1 dataset
    # Extra datasets
    "limr": "GAIR/LIMR", # 1.39k
    "open_r1": "open-r1/OpenR1-Math-220k",  # default split 93.7k: originally from NuminaMath 1.5
    "thoughts": "bethgelab/CuratedThoughts", # default OpenThoughts-114k-math-default 66.1k: originally from NuminaMath-CoT
    # Ablation
    "limr_large_lr_ablation": "GAIR/LIMR",
    "limr_small_lr_ablation": "GAIR/LIMR",
    "limr_large_rank_ablation": "GAIR/LIMR",
    "limr_medium_rank_ablation": "GAIR/LIMR",
    "limr_small_rank_ablation": "GAIR/LIMR",
    "limr_tiny_rank_ablation": "GAIR/LIMR",
    "open_rs3_drgrpo_ablation": "knoveleng/open-rs",
    "open_rs3_format_ablation": "knoveleng/open-rs",
    "open_rs3_long_completion_ablation": "knoveleng/open-rs",

    "openthoughts3": "open-thoughts/OpenThoughts3-1.2M"
}

# question, solution => combined text for instructed models with different chat templates
SFT_POST_TRAIN_CONFIG_MAP = {
    "qwen_rationale_limo": "GAIR/LIMO",  # 817
    "r1_rationale_s1": "simplescaling/s1K-1.1", # 1k
    "claude_rationale_s1": "simplescaling/s1K-claude-3-7-sonnet", # 1k

    "deepscaler": "agentica-org/DeepScaleR-Preview-Dataset", # 40.3k
    "still": "RUC-AIBOX/STILL-3-Preview-RL-Data", # 33k
}