#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

uv run --no-sync python -m scripts.grpo_train \
    --model-name /root/autodl-tmp/models/OLMo-2-0425-1B \
    --prompt-path cs336_alignment/prompts/r1_zero.prompt \
    --train-path data/gsm8k/train.jsonl \
    --validation-path data/gsm8k/test.jsonl \
    --num-rollout-steps 1 \
    --rollout-batch-size 8 \
    --group-size 2 \
    --gradient-accumulation-steps 2 \
    --learning-rate 1e-5 \
    --max-grad-norm 1.0 \
    --temperature 1.0 \
    --max-tokens 128 \
    --eval-every 1 \
    --log-rollout-every 1 \
    --n-validation-samples 32 \
    --seed 0 \
    --training-device cuda:0 \
    --vllm-gpu 1
