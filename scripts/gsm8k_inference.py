from pathlib import Path
import json
import os
from unittest import result

from cs336_alignment.vllm_utils import VLLMCompletion, generate_completions
from cs336_alignment.drgrpo_grader import (
    question_only_reward_fn,
    r1_zero_reward_fn,
)
import re
from collections import Counter
from tqdm.auto import tqdm

DATASET_PATH = Path("data/gsm8k/test.jsonl")
PROMPT_DIR = Path("cs336_alignment/prompts")
PROMPT1 = (PROMPT_DIR / "question_only.prompt").read_text(encoding="utf-8")
PROMPT2 = (PROMPT_DIR / "r1_zero_three_shot_gsm8k.prompt").read_text(encoding="utf-8")
PROMPT3 = (PROMPT_DIR / "r1_zero.prompt").read_text(encoding="utf-8")
VLLM_BASE_URL = "http://127.0.0.1:8000"
MODEL_ID = "allenai/OLMo-2-0425-1B"
BATCH_SIZE = 8
SAMPLING_PARAMS = {
    "temperature": 1.0,
    "max_tokens": 512,
    "n": 1,
    "seed": 336,
}
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_ground_truth(dataset_path: os.PathLike) -> dict[str, str]:
    ground_truth = {}
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line.strip())
            ground_truth[line["question"]] = line["answer"].split("####")[-1].strip()
    return ground_truth

def build_prompts(prompt_template: str, ground_truth: dict):
    prompts = [prompt_template.format(question=question) for question in ground_truth]
    return prompts

def main():
    ground_truth = load_ground_truth(DATASET_PATH)
    print(f"Loaded {len(ground_truth)} GSM8K examples", flush=True)
    prompt_templates = [PROMPT1, PROMPT2, PROMPT3]
    for prompt_template in prompt_templates:
        if prompt_template == PROMPT1:
            sampling_params = SAMPLING_PARAMS
            reward_fn = question_only_reward_fn
            prompt_name = "question_only"
            
        if prompt_template == PROMPT2:
            sampling_params = {
                **SAMPLING_PARAMS, 
                "stop": ["</answer>"], 
                "include_stop_str_in_output": True
            }
            reward_fn = r1_zero_reward_fn
            prompt_name = "r1_zero_three_shot"
        if prompt_template == PROMPT3:
            sampling_params = {
                **SAMPLING_PARAMS, 
                "stop": ["</answer>"], 
                "include_stop_str_in_output": True
            }
            reward_fn = r1_zero_reward_fn
            prompt_name = "r1_zero"

        prompts: list[str] = build_prompts(prompt_template, ground_truth=ground_truth)        
        completions = []
        for start in tqdm(
            range(0, len(prompts), BATCH_SIZE),
            desc=f"Generating {prompt_name}",
            unit="batch",
        ):
            prompt_batch = prompts[start : start + BATCH_SIZE]
            batch_completions = generate_completions(
                vllm_base_url=VLLM_BASE_URL,
                model_id=MODEL_ID,
                prompts=prompt_batch,
                sampling_params=sampling_params,
                batch_size=None,
            )
            completions.extend(batch_completions)

        assert len(completions) == len(prompts), f"生成数量不匹配: prompts={len(prompts)}, completions={len(completions)}"

        records = []
        counts = Counter()
        for question, prompt, completion, answer in tqdm(
            zip(
                ground_truth.keys(),
                prompts,
                completions,
                ground_truth.values(),
            ),
            total=len(completions),
            desc=f"Grading {prompt_name}",
            unit="example",
        ):
            prediction = completion.text
            result_dict = reward_fn(prediction, answer)
            key = (
                int(result_dict["format_reward"]),
                int(result_dict["answer_reward"]),
            )
            counts[key] += 1

            records.append({
                "question": question,
                "ground_truth": answer,
                "prompt": prompt,
                "response": prediction,
                "format_reward": result_dict["format_reward"],
                "answer_reward": result_dict["answer_reward"],
                "reward": result_dict["reward"]
            })
        
        summary = {
            "prompt_type": prompt_name,
            "format_1_answer_1": counts[(1, 1)],
            "format_1_answer_0": counts[(1, 0)],
            "format_0_answer_0": counts[(0, 0)],
            "total": sum(counts.values()),
        }
        summary_path = OUTPUT_DIR / f"{prompt_name}_summary.json"
        record_path = OUTPUT_DIR / f"{prompt_name}_reward.jsonl"
        
        with open(summary_path, "w", encoding="utf-8") as f, open(record_path, "w", encoding="utf-8") as f1:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            for record in records:
                f1.write(json.dumps(record, ensure_ascii=False) + "\n")

        tqdm.write(f"{prompt_name}: {summary}")
                
            


if __name__ == "__main__":
    main()
