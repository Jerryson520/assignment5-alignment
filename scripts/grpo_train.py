import argparse
from dataclasses import dataclass
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from cs336_alignment.vllm_utils import VLLMServer
import json
import random
from tests.adapters import run_grpo_train_step
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

@dataclass
class GRPOConfig:
    model_name: str
    prompt_path: Path
    train_path: Path
    validation_path: Path

    num_rollout_steps: int
    rollout_batch_size: int
    group_size: int
    gradient_accumulation_steps: int

    learning_rate: float
    max_grad_norm: float

    temperature: float
    max_tokens: int

    eval_every: int
    log_rollout_every: int
    n_validation_samples: int

    seed: int
    training_device: str
    vllm_gpu: int

def parse_args() -> GRPOConfig:
    parser = argparse.ArgumentParser(
        description="Train OLMo on GSM8K using on-policy GRPO."
    )

    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--prompt-path", type=Path, required=True)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--validation-path", type=Path, required=True)
    parser.add_argument("--num-rollout-steps", type=int, required=True)
    parser.add_argument("--rollout-batch-size", type=int, required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--max-grad-norm", type=float, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--eval-every", type=int, required=True)
    parser.add_argument("--log-rollout-every", type=int, required=True)
    parser.add_argument("--n-validation-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--training-device", type=str, required=True)
    parser.add_argument("--vllm-gpu", type=int, required=True)

    args = parser.parse_args()

    return GRPOConfig(
        model_name=args.model_name,
        prompt_path=args.prompt_path,
        train_path=args.train_path,
        validation_path=args.validation_path,
        num_rollout_steps=args.num_rollout_steps,
        rollout_batch_size=args.rollout_batch_size,
        group_size=args.group_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        eval_every=args.eval_every,
        log_rollout_every=args.log_rollout_every,
        n_validation_samples=args.n_validation_samples,
        seed=args.seed,
        training_device=args.training_device,
        vllm_gpu=args.vllm_gpu,
    )

def load_gsm8k(path: Path):
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            examples.append({
                "question": line["question"],
                "ground_truth": line["answer"].split("####")[-1].strip()
            })
    return examples

def train_grpo(config: GRPOConfig):
    ## 1. Initialization
    train_dataset = load_gsm8k(config.train_path)
    validation_dataset = load_gsm8k(config.validation_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    policy = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16
    ).to(config.training_device)

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=config.learning_rate, betas=(0.9, 0.95), weight_decay=0.0
    )

    ## 2. 启动vllm server
    vllm_server = VLLMServer(
        model_id=config.model_name,
        gpu=config.vllm_gpu,
        seed=config.seed
    )
    vllm_server.start()
    vllm_server.init_weight_sync(config.training_device)

    ## 3. build dataset
    prompt_template = config.prompt_path.read_text(encoding="utf-8")
    if config.rollout_batch_size % config.group_size != 0:
        raise ValueError(
            "rollout_batch_size must be divisible by group_size"
        )

    prompts_per_rollout_batch = config.rollout_batch_size // config.group_size

    rng = random.Random(config.seed)
    shuffled_train_dataset = train_dataset.copy()
    rng.shuffle(shuffled_train_dataset)

    required_examples = (config.num_rollout_steps * prompts_per_rollout_batch)

    if required_examples > len(shuffled_train_dataset):
        raise ValueError(
            f"Training requires {required_examples} examples, "
            f"but the dataset only has "
            f"{len(shuffled_train_dataset)}."
        )

    ## 4. rollouts
    for step in range(config.num_rollout_steps):
        start = step * prompts_per_rollout_batch
        end = start + prompts_per_rollout_batch
        prompt_batch = shuffled_train_dataset[start: end]

        prompts = [
            prompt_template.format(question=example["question"])
            for example in prompt_batch
        ]

        ground_truths = [
            example["ground_truth"]
            for example in prompt_batch
        ]

        sampling_params = {
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "n": config.group_size ,
            "seed": config.seed + step,
            "stop": ["</answer>"],
            "include_stop_str_in_output": True
        }

        vllm_server.sync_policy_weights(policy)

        completions = vllm_server.generate_completions(
            prompts,
            sampling_params,
            batch_size=None
        )

        expected_rollouts = len(prompts) * config.group_size
        if len(completions) != expected_rollouts:
            raise RuntimeError(
                f"Expected {expected_rollouts} completions, "
                f"received {len(completions)}."
            )

        rollout_responses = [completion.text for completion in completions]

        repeated_prompts = [prompt for prompt in prompts for _ in range(config.group_size)]
        repeated_ground_truths = [ground_truth for ground_truth in ground_truths for _ in range(config.group_size)]

        ## 5. train loop
        loss, metadata = run_grpo_train_step(
            model=policy,
            tokenizer=tokenizer,
            optimizer=optimizer,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_grad_norm=config.max_grad_norm,
            reward_fn=r1_zero_reward_fn,
            repeated_prompts=repeated_prompts,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=config.group_size
        )

        print(
            f"step={step} | "
            f"loss={loss.item():.6f} | "
            f"reward={metadata["mean_reward"]:.4f} | "
            f"format_reward={metadata["mean_format_reward"]:.4f} | "
            f"gradient_norm={metadata["gradient_norm"].item():.4f} | "
            f"token_entropy={metadata["token_entropy"].item():.4f}",
            flush=True
        )



def main():
    config = parse_args()
    train_grpo(config)


if __name__ == "__main__":
    main()
