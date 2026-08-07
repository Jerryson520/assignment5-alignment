from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase



def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask aligned with
    labels that is 1 for response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str]
            List of prompt strings.
        output_strs: list[str]
            List of output strings.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.

    Returns:
        dict[str, torch.Tensor].
            Let prompt_and_output_lens be a list containing the lengths of the
            concatenated tokenized prompt and output strings. Then the returned
            dictionary should have the following keys:

            input_ids
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): the tokenized
                prompt and output strings, with the final token sliced off.
            labels
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): shifted input
                ids, i.e., the input ids without the first token.
            response_mask
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): a mask aligned
                with labels, with value 1 where the corresponding label token
                is part of the response and 0 otherwise.
    """
    prompt_ids = tokenizer(
        prompt_strs,
        add_special_tokens=False,
        padding=False
    )["input_ids"]

    output_ids = tokenizer(
        output_strs,
        add_special_tokens=False,
        padding=False
    )["input_ids"]

    combined_ids = []
    combined_masks = []

    for prompt, output in zip(prompt_ids, output_ids):
        combined_ids.append(prompt + output)
        combined_masks.append([0] * len(prompt) + [1] * len(output))

    max_len = max(len(id) for id in combined_ids)

    for i in range(len(combined_ids)):
        pad_len = max_len - len(combined_ids[i])

        combined_ids[i] += [tokenizer.pad_token_id] * pad_len
        combined_masks[i] += [0] * pad_len

    tokens = torch.tensor(combined_ids, dtype=torch.long)
    masks = torch.tensor(combined_masks, dtype=torch.bool)

    return {
        "input_ids": tokens[:, :-1],
        "labels": tokens[:, 1:],
        "response_mask": masks[:, 1:]
    }

def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities (given the previous tokens)
    from a causal language model, and optionally the entropy of the model's
    next-token distribution.

    Args:
        model: PreTrainedModel
            HuggingFace model used for scoring (placed on the correct device
            and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor
            shape (batch_size, sequence_length), concatenated prompt + response
            tokens as produced by your tokenization method.
        labels: torch.Tensor
            shape (batch_size, sequence_length), labels as produced by your
            tokenization method.
        return_token_entropy: bool
            If True, also return per-token entropy.

    Returns:
        dict[str, torch.Tensor].
            "log_probs"
                shape (batch_size, sequence_length), conditional
                log-probabilities log p_(theta)(x_t | x_(<t)).
            "token_entropy"
                optional, shape (batch_size, sequence_length), per-token
                entropy for each position (present only if
                return_token_entropy=True).
    """
    logits = model(input_ids).logits
    all_probs = torch.softmax(logits, dim=-1)
    all_log_probs = torch.log_softmax(logits, dim=-1)

    log_probs = torch.gather(
        all_log_probs,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

    token_entropy = -(
        all_probs * all_log_probs
    ).sum(dim=-1)

    return {
        "log_probs": log_probs,
        "token_entropy": token_entropy
    }

def run_compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute rewards for a list of rollout responses, along with metadata for
    the reward components.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            raw_rewards
                shape (rollout_batch_size,). Unnormalized rewards for each
                rollout response.
            metadata
                Reward statistics to log. At minimum, include the mean total
                and format rewards over the rollout batch.
    """
    reward_dicts = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(rollout_responses, repeated_ground_truths)
    ]
    raw_rewards = torch.tensor(
        [reward["reward"] for reward in reward_dicts],
        dtype=torch.float32
    )

    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "mean_format_reward": sum(reward["format_reward"] for reward in reward_dicts) / len(reward_dicts),
        "mean_answer_reward": sum(reward["answer_reward"] for reward in reward_dicts) / len(reward_dicts),
    }
    return raw_rewards, metadata


def run_compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute advantages by applying the requested baseline and normalization
    within each group.

    Args:
        raw_rewards: torch.Tensor
            shape (rollout_batch_size,). Unnormalized rewards for each rollout
            response, where rollout_batch_size = n_prompts_per_rollout_batch *
            group_size.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            For this problem, support mean, which subtracts the per-group mean
            reward. Later, none will mean no baseline subtraction.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            For this problem, support std, which divides by the per-group
            standard deviation. Later, none will mean no normalization and
            mean will mean divide by the per-group mean reward.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            advantages
                shape (rollout_batch_size,). Group-normalized rewards for each
                rollout response.
            metadata
                your choice of other statistics to log (e.g. mean, std, max/min
                of rewards).
    """
    raw_len = len(raw_rewards)
    raw_rewards = raw_rewards.reshape(-1, group_size)
    if baseline == "mean":
        group_mean = raw_rewards.mean(dim=-1, keepdim=True)
        numerator = raw_rewards - group_mean
    elif baseline == "none":
        numerator = raw_rewards

    if advantage_normalizer == "std":
        group_stds = raw_rewards.std(dim=-1, keepdim=True)
        denominator = group_stds + advantage_eps
    elif advantage_normalizer == "none":
        denominator = 1
    elif advantage_normalizer == "mean":
        denominator = raw_rewards.mean(dim=-1, keepdim=True) + advantage_eps

    advantages = numerator / denominator
    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "mean_advantage": advantages.mean().item(),
        "max_reward": raw_rewards.max().item(),
        "min_reward": raw_rewards.min().item(),
    }
    return advantages.reshape(raw_len), metadata


def run_compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy-gradient loss at every token, where
    raw_rewards_or_advantages is either the raw reward or an
    already-normalized advantage.

    Args:
        raw_rewards_or_advantages: torch.Tensor
            Shape (batch_size,) or (batch_size, 1), scalar reward/advantage for
            each rollout response.
        policy_log_probs: torch.Tensor
            Shape (batch_size, sequence_length), logprobs for each token.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style
            token-level reweighting and clipping; "gspo": do GSPO-style
            sequence-level reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        response_mask: torch.Tensor | None = None
            Optional shape (batch_size, sequence_length) mask over response
            tokens. Required for GSPO implementations that average the
            sequence-level log-ratio over response tokens only.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            per_token_policy_gradient_loss
                Shape (batch_size, sequence_length), the per-token
                policy-gradient loss (to be aggregated across the batch and
                sequence dimensions in the training loop).
            metadata
                Statistics from the underlying loss call, such as
                clip-fraction components.
    """
    advantages = raw_rewards_or_advantages.reshape(-1, 1)
    metadata = {}

    if importance_reweighting_method == "none":
        per_token_loss = -(advantages * policy_log_probs)

    elif importance_reweighting_method == "noclip":
        importance_weights = torch.exp(policy_log_probs - old_log_probs)
        per_token_loss = -(advantages * importance_weights)

    elif importance_reweighting_method == "grpo":
        importance_weights = torch.exp(policy_log_probs - old_log_probs)
        clipped_weights = torch.clamp(
            importance_weights,
            min=1-cliprange,
            max=1+cliprange,
        )

        per_token_loss = -torch.minimum(
            advantages * importance_weights,
            advantages * clipped_weights,
        )
    elif importance_reweighting_method == "gspo":
        log_ratios = policy_log_probs - old_log_probs
        mask = response_mask.to(log_ratios.dtype)
        response_lengths = mask.sum(dim=-1, keepdim=True)

        log_weights = (log_ratios * mask).sum(dim=-1, keepdim=True) / response_lengths
        importance_weights = torch.exp(log_weights)

        clipped_weights = torch.clamp(
            importance_weights,
            min=1-cliprange,
            max=1+cliprange,
        )

        per_token_loss = -torch.minimum(
            advantages * importance_weights,
            advantages * clipped_weights,
        ).expand_as(policy_log_probs)

    return per_token_loss, metadata


def run_aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    """Aggregate the per-token policy-gradient loss according to the response
    mask and loss-normalization strategy.

    Args:
        per_token_policy_gradient_loss: torch.Tensor
            Shape (batch_size, sequence_length), the per-token policy-gradient
            loss (to be aggregated across the batch and sequence dimensions in
            the training loop).
        mask
            torch.Tensor of shape (batch_size, sequence_length) denoting which
            positions should be included in the loss.
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant.
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        loss: torch.Tensor
            A scalar containing the average loss. Make sure you can later call
            backward on this loss.
    """
    mask = mask.to(
        dtype=per_token_policy_gradient_loss.dtype
    )
    if normalization_constant:
        denominator = normalization_constant
        total_loss = (per_token_policy_gradient_loss * mask).sum() / normalization_constant
    else:
        token_counts = mask.sum(dim=-1)
        if (token_counts == 0).any():
            raise ValueError("A sequence has no response tokens.")
        seq_losses = (per_token_policy_gradient_loss * mask).sum(dim=-1) / token_counts
        total_loss = seq_losses.mean(dim=-1)

    return total_loss


def run_grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Execute forward-and-backward passes, with gradient_accumulation_steps
    microbatches.

    Args:
        model: PreTrainedModel
            HuggingFace model to train.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.
        optimizer: Optimizer
            Optimizer for the model.
        gradient_accumulation_steps: int
            Number of microbatches per optimizer step.
        max_grad_norm: float | None
            If not None, clip the gradient norm to this value before calling
            optimizer.step().
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        repeated_prompts: list[str]
            The prompts for the examples. The length of this list is
            rollout_batch_size, because the prompt for each example is repeated
            group_size times.
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            If mean, subtract the per-group mean reward; if none, do nothing.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            If std, divide by the per-group standard deviation; if none, do
            nothing; if mean, divide by the per-group mean reward.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style token-level
            reweighting and clipping; "gspo": do GSPO-style sequence-level
            reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant (fixed
            for all of training).
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss
                scalar tensor. The batch loss, adjusted for gradient
                accumulation. We return this so we can log it.
            metadata
                Dict with metadata from the underlying loss call, gradient norm
                before clipping, and any other statistics you might want to log.
    """
        # This problem only requires standard on-policy GRPO.
    rollout_batch_size = len(repeated_prompts)

    if rollout_batch_size == 0:
        raise ValueError("The rollout batch cannot be empty.")

    if len(rollout_responses) != rollout_batch_size:
        raise ValueError(
            "repeated_prompts and rollout_responses "
            "must have the same length."
        )

    if len(repeated_ground_truths) != rollout_batch_size:
        raise ValueError(
            "repeated_prompts and repeated_ground_truths "
            "must have the same length."
        )

    if group_size <= 0:
        raise ValueError("group_size must be positive.")

    if rollout_batch_size % group_size != 0:
        raise ValueError(
            "rollout_batch_size must be divisible by group_size."
        )

    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "gradient_accumulation_steps must be positive."
        )

    if gradient_accumulation_steps > rollout_batch_size:
        raise ValueError(
            "gradient_accumulation_steps cannot exceed "
            "rollout_batch_size."
        )

    raw_rewards, rewards_metadata = run_compute_rollout_rewards(
        reward_fn=reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths
    )

    advantages, advantage_metadata = run_compute_group_normalized_rewards(
        raw_rewards=raw_rewards,
        group_size=group_size,
        baseline=baseline,
        advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer
    )

    tokenized = run_tokenize_prompt_and_output(
        prompt_strs=repeated_prompts,
        output_strs=rollout_responses,
        tokenizer=tokenizer,
    )

    input_ids = tokenized["input_ids"]
    labels = tokenized["labels"]
    response_mask = tokenized["response_mask"]

    device = (next(model.parameters())).device
    if rollout_batch_size % gradient_accumulation_steps != 0:
        raise ValueError(
            "rollout_batch_size must be divisible by "
            "gradient_accumulation_steps."
        )
    micro_batch_size = rollout_batch_size // gradient_accumulation_steps

    batch_loss = torch.zeros((), dtype=torch.float32, device=device)
    entropy_sum = torch.zeros((), dtype=torch.float32, device=device)
    response_token_count = torch.zeros((), dtype=torch.float32, device=device)
    optimizer.zero_grad(set_to_none=True)
    gradient_loss_metadata: dict[str, torch.Tensor | float] = {}

    for idx in range(gradient_accumulation_steps):
        start_idx = idx * micro_batch_size
        end_idx = start_idx + micro_batch_size

        micro_advantages = advantages[start_idx: end_idx]
        micro_nonzero_mask = (micro_advantages != 0)
        if not micro_nonzero_mask.any():
            continue

        micro_input_ids = input_ids[start_idx: end_idx][micro_nonzero_mask].to(device)
        micro_labels = labels[start_idx: end_idx][micro_nonzero_mask].to(device)
        micro_response_mask = response_mask[start_idx: end_idx][micro_nonzero_mask].to(device)
        micro_advantages = micro_advantages[micro_nonzero_mask].to(device)

        log_probs_output = run_get_response_log_probs(
            model=model,
            input_ids=micro_input_ids,
            labels=micro_labels,
            return_token_entropy=True,
        )

        log_probs = log_probs_output["log_probs"]
        token_entropy = log_probs_output["token_entropy"]

        per_token_loss, gradient_loss_metadata = run_compute_policy_gradient_loss(
            raw_rewards_or_advantages=micro_advantages,
            policy_log_probs=log_probs,
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
            response_mask=micro_response_mask,
        )

        micro_batch_loss = run_aggregate_loss_across_microbatch(
            per_token_policy_gradient_loss=per_token_loss,
            mask=micro_response_mask,
            loss_normalization=loss_normalization,
            normalization_constant=normalization_constant
        )
        if loss_normalization == "sequence":
            scaled_loss = (
                micro_batch_loss * sum(micro_nonzero_mask).item() / rollout_batch_size
            )
        else:
            scaled_loss = micro_batch_loss

        scaled_loss.backward()

        batch_loss += scaled_loss.detach()
        entropy_sum += (token_entropy.detach() * micro_response_mask).sum()
        response_token_count += micro_response_mask.sum()

    clipping_threshold = max_grad_norm if max_grad_norm is not None else float("inf")
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        clipping_threshold,
    )

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if (response_token_count == 0).any():
        mean_token_entropy = torch.zeros((), dtype=torch.float32, device=device)
    else:
        mean_token_entropy = entropy_sum / response_token_count

    metadata = {
        **rewards_metadata,
        **advantage_metadata,
        **gradient_loss_metadata,
        "loss": batch_loss,
        "gradient_norm": gradient_norm.detach(),
        "token_entropy": mean_token_entropy.detach(),
    }
    return batch_loss, metadata



"""
The below adapters are used in the optional
RLHF / safety part of the Alignment assignment.
"""


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    raise NotImplementedError


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    raise NotImplementedError


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    raise NotImplementedError


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    raise NotImplementedError


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    raise NotImplementedError
