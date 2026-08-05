#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

training_gpu="${TRAINING_GPU:-0}"
vllm_gpu="${VLLM_GPU:-1}"
vllm_port="${VLLM_PORT:-8000}"

if [[ "${training_gpu}" == "${vllm_gpu}" ]]; then
    echo "TRAINING_GPU and VLLM_GPU must be different." >&2
    exit 1
fi

if (( $# > 0 )); then
    seeds=("$@")
else
    seeds=(0 1 2 3)
fi

for seed in "${seeds[@]}"; do
    echo "Starting Dr. GRPO training with seed=${seed}"

    uv run --no-sync python -m scripts.grpo_train \
        --model-name /root/autodl-tmp/models/OLMo-2-0425-1B \
        --prompt-path cs336_alignment/prompts/r1_zero.prompt \
        --train-path data/gsm8k/train.jsonl \
        --validation-path data/gsm8k/test.jsonl \
        --num-rollout-steps 200 \
        --rollout-batch-size 256 \
        --group-size 8 \
        --gradient-accumulation-steps 32 \
        --variant-name drgrpo \
        --baseline mean \
        --advantage-normalizer none \
        --loss-normalization constant \
        --learning-rate 1e-5 \
        --max-grad-norm 1.0 \
        --temperature 1.0 \
        --max-tokens 512 \
        --eval-every 10 \
        --log-rollout-every 40 \
        --n-validation-samples 1024 \
        --seed "${seed}" \
        --training-device "cuda:${training_gpu}" \
        --vllm-gpu "${vllm_gpu}" \
        --vllm-port "${vllm_port}"

    echo "Finished Dr. GRPO training with seed=${seed}"
done

echo "All four Dr. GRPO runs completed."
