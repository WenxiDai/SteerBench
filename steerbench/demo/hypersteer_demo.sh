#!/bin/bash

# Check if nvidia-smi command exists
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi could not be found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

# Get the number of GPUs
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

python scripts/generate.py --config demo/sweep/hypersteer_simple.yaml --dump_dir demo

torchrun --nproc_per_node=$gpu_count scripts/train.py \
  --config demo/sweep/hypersteer_simple.yaml --dump_dir demo

torchrun --nproc_per_node=$gpu_count scripts/inference.py --config demo/sweep/hypersteer_simple.yaml --mode steering --dump_dir demo

python scripts/evaluate.py --config demo/sweep/hypersteer_simple.yaml --mode steering --dump_dir demo
