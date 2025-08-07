import random


def make_conv_for_grpo(example, system_prompt):
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["problem"]},
        ]
    }

def make_conv_for_grpo_l1(example, system_prompt, min_length, max_length):
    target_length = random.randint(min_length, max_length)
    system_prompt += f"\nThink for {target_length} tokens."
    return {
        "target_length": target_length,
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["problem"]},
        ]
    }

def make_conv_for_sft(example, dataset_name_or_path, tokenizer, trace_free=False):
    if dataset_name_or_path == "simplescaling/s1K-claude-3-7-sonnet":
        trajectory_name = "claude_thinking_trajectory"
        attemp_name = "claude_attempt"
    elif dataset_name_or_path == "simplescaling/s1K-1.1":
        trajectory_name = "deepseek_thinking_trajectory"
        attemp_name = "deepseek_attempt"
    elif dataset_name_or_path == "simplescaling/s1K":
        trajectory_name = "thinking_trajectories"
        attemp_name = "attempt"
    elif dataset_name_or_path == "GAIR/LIMO":
        trajectory_name = "solution"
        attemp_name = "answer"
    elif dataset_name_or_path == "RUC-AIBOX/STILL-3-Preview-RL-Data":
        trajectory_name = "answer"
        attemp_name = "answer"
    elif dataset_name_or_path == "agentica-org/DeepScaleR-Preview-Dataset":
        trajectory_name = "solution"
        attemp_name = "answer"
    else:
        raise ValueError(f"Unknown dataset for sft post-training: {dataset_name_or_path}")

    if not trace_free:
        conv = [
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"think: \n{trajectory}\n answer: {attempt}"},
            ]
            for question, trajectory, attempt in zip(example["question"], example[trajectory_name], example[attemp_name])
        ]
    else:
        # Only use the final answer but in the thinking format
        conv = [
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"think: \n{attempt}\n answer: {attempt}"},
            ]
            for question, attempt in zip(example["question"], example[attemp_name])
        ]
    return {
        "text": tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=False)
    }
