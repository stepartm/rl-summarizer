import torch
from transformers import GPT2Tokenizer


def get_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
    tokenizer.add_eos_token = True
    return tokenizer


def generate_input_prompt(x: str):
    return "You are given a text. Your task is to write a concise summary for this text. Text: " + x + " Summary: "


def tensor_stats(tensor: torch.Tensor):
    return tensor.min(), tensor.max(),  tensor.mean(), tensor.std()


def compute_grpo_loss(rewards: torch.Tensor, cur_probas: torch.Tensor, old_probas: torch.Tensor, reference_probas: torch.Tensor, mask: torch.Tensor, eps: float, beta: float=1):
    """
    rewards = (batch_size, group_size)
    cur_probas: (batch_size, group_size, seq_len)
    old_probas: (batch_size, group_size, seq_len)
    reference_probas: (batch_size, group_size, seq_len)
    mask: (batch_size, group_size, seq_len)
    """
    mean_reward = rewards.mean(dim=0)
    std_reward = rewards.std(dim=0)
    advantages = (rewards - mean_reward) / std_reward
    policy_comparison = cur_probas / old_probas

    objective = torch.min(
        policy_comparison * advantages.unsqueeze(2),
        torch.clip(policy_comparison * advantages.unsqueeze(2), 1 - eps, 1 + eps)
    )

    ref_comparison = reference_probas / cur_probas
    regularizer = ref_comparison - torch.log(ref_comparison) - 1

    loss = (
        (
            (objective - beta * regularizer) * mask
        ).sum(dim=-1)  # reducing over seq dimension
        /  mask.sum(dim=-1)  # avg over seq
    ).mean()  # avg over group and batch

    # These statistics are computed globally:
    min_reward_global, max_reward_global, mean_reward_global, std_reward_global = tensor_stats(rewards)
    min_advantages_global, max_advantages_global, mean_advantages_global, std_advantages_global = tensor_stats(advantages)
    min_policy_comparison_global, max_policy_comparison_global, mean_policy_comparison_global, std_policy_comparison_global = tensor_stats(policy_comparison)

    n_clips_global = (
        (objective == 1 - eps) |  (objective == 1 + eps)
    ).sum()

    min_ref_comparison_global, max_ref_comparison_global, mean_ref_comparison_global, std_ref_comparison_global = tensor_stats(ref_comparison)
    min_regularizer_global, max_regularizer_global, mean_regularizer_global, std_regularizer_global = tensor_stats(regularizer)

    metrics = {
        "reward": {
            "min": min_reward_global,
            "max": max_reward_global,
            "mean": mean_reward_global,
            "std": std_reward_global,
        },
        "advantage": {
            "min": min_advantages_global,
            "max": max_advantages_global,
            "mean": mean_advantages_global,
            "std": std_advantages_global,
        },
        "policy_comparison": {
            "min": min_policy_comparison_global,
            "max": max_policy_comparison_global,
            "mean": mean_policy_comparison_global,
            "std": std_policy_comparison_global,
        },
        "n_clips": n_clips_global,
        "ref_comparison": {
            "min": min_ref_comparison_global,
            "max": max_ref_comparison_global,
            "mean": mean_ref_comparison_global,
            "std": std_ref_comparison_global,
        },
        "regularizer": {
            "min": min_regularizer_global,
            "max": max_regularizer_global,
            "mean": mean_regularizer_global,
            "std": std_regularizer_global,
        },
    }
    return loss, metrics
