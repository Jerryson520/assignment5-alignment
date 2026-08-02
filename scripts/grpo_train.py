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
import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

def evaluate(
    vllm_server: VLLMServer, 
    validation_dataset: list[dict[str, str]],
    prompt_template: str,
    config: GRPOConfig,
    step: int
):
    examples = validation_dataset[:config.n_validation_samples]

    prompts = [
        prompt_template.format(question=example["question"]) 
        for example in examples
    ]

    ground_truths = [example["ground_truth"] for example in examples]

    sampling_params = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "n": 1,
        "seed": config.seed + step,
        "stop": ["</answer>"],
        "include_stop_str_in_output": True
    }

    completions = vllm_server.generate_completions(
        prompts,
        sampling_params,
        batch_size=32
    )

    reward_dicts = [
        r1_zero_reward_fn(completion.text, ground_truth)
        for completion, ground_truth in zip(completions, ground_truths)
    ]

    mean_reward = sum(reward_dict["reward"] for reward_dict in reward_dicts) / len(reward_dicts)
    mean_format_accuracy = sum(reward_dict["format_reward"] for reward_dict in reward_dicts) / len(reward_dicts)
    mean_answer_accuracy = sum(reward_dict["answer_reward"] for reward_dict in reward_dicts) / len(reward_dicts)
    mean_response_length = sum(len(completion.token_ids) for completion in completions) / len(completions)

    print(
        f"validation step={step} | "
        f"n={len(examples)} | "
        f"mean_reward={mean_reward:.4f} | "
        f"mean_format_accuracy={mean_format_accuracy:.4f} | "
        f"mean_answer_accuracy={mean_answer_accuracy:.4f} | "
        f"mean_response_length={mean_response_length:.2f}",
        flush=True,
    )

    return {
        "val/reward": mean_reward,
        "val/format_reward": mean_format_accuracy,
        "val/answer_accuracy": mean_answer_accuracy,
        "val/average_response_length": mean_response_length,
    }

def log_rollouts(
    step: int, 
    repeated_prompts: list[str],
    repeated_responses: list[str],
    repeated_ground_truths: list[str],
    output_path: Path,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = wandb.Table(
        columns=[
            "step",
            "prompt",
            "response",
            "ground_truth",
            "reward",
            "format_reward",
            "answer_reward",
        ]
    )

    with output_path.open("a", encoding="utf-8") as f:
        for prompt, response, ground_truth in zip(
            repeated_prompts, 
            repeated_responses, 
            repeated_ground_truths
        ):
            reward = r1_zero_reward_fn(response, ground_truth)

            record = {
                "step": step,
                "prompt": prompt,
                "response": response,
                "ground_truth": ground_truth,
                **reward,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            table.add_data(
                step,
                prompt,
                response,
                ground_truth,
                reward["reward"],
                reward["format_reward"],
                reward["answer_reward"],
            )
    
    return table  

def train_grpo(config: GRPOConfig):
    ## 1. Initialization
    train_dataset = load_gsm8k(config.train_path)
    validation_dataset = load_gsm8k(config.validation_path)

    if config.log_rollout_every <= 0:
        raise ValueError("log_rollout_every must be positive")

    if config.eval_every <= 0:
        raise ValueError("eval_every must be positive")

    if config.n_validation_samples <= 0:
        raise ValueError("n_validation_samples must be positive")

    if config.n_validation_samples > len(validation_dataset):
        raise ValueError(
            "n_validation_samples exceeds validation dataset size"
        )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    policy = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16
    ).to(config.training_device)

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=config.learning_rate, betas=(0.9, 0.95), weight_decay=0.0
    )

    wandb_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(config).items()
    }

    wandb_run = wandb.init(
        project="cs336-assignment5-grpo",
        name=f"grpo-r1-zero-seed-{config.seed}",
        group="grpo-r1-zero-standard",
        config=wandb_config,
    )
    rollout_log_path = (
        PROJECT_ROOT
        / "results"
        / f"grpo_rollouts_seed_{config.seed}_{wandb_run.id}.jsonl"
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
        step_number = step + 1

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

        wandb_metrics = {
            "train/loss": loss.item(),
            "train/reward": metadata["mean_reward"],
            "train/format_reward": metadata["mean_format_reward"],
            "train/gradient_norm": metadata["gradient_norm"].item(),
            "train/token_entropy": metadata["token_entropy"].item(),
        }

        print(
            f"step={step_number} | "
            f"loss={loss.item():.6f} | "
            f"reward={metadata['mean_reward']:.4f} | "
            f"format_reward={metadata['mean_format_reward']:.4f} | "
            f"gradient_norm={metadata['gradient_norm'].item():.4f} | "
            f"token_entropy={metadata['token_entropy'].item():.4f}",
            flush=True
        )

        
        if step_number % config.log_rollout_every == 0:
            rollout_table = log_rollouts(
                step=step_number,
                repeated_prompts=repeated_prompts,
                repeated_responses=rollout_responses,
                repeated_ground_truths=repeated_ground_truths,
                output_path=rollout_log_path,
            )
            wandb_metrics["train/rollouts"] = rollout_table
 
        if step_number % config.eval_every == 0:
            vllm_server.sync_policy_weights(policy)

            validation_metrics = evaluate(
                vllm_server=vllm_server,
                validation_dataset=validation_dataset,
                prompt_template=prompt_template,
                config=config,
                step=step_number
            )

            wandb_metrics.update(validation_metrics)

        wandb.log(
            wandb_metrics,
            step=step_number,
        )

    wandb.finish()

def main():
    config = parse_args()
    train_grpo(config)


if __name__ == "__main__":
    main()
