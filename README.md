## Final Project Code for IXAI

This README is for the final-project:

- **Weakly supervised dictionary learning**: `LsReFT` (main method)
- **Datasets**: training data + latent inference data
- **Baselines (partial)**: `LinearProbe`, `PromptSteering`, `GemmaScopeSAE`



## Setup

```python
import os

os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"
os.environ["NP_API_KEY"] = "your_neuronpedia_api_key_here"
```

Use the demo config:

- Config: `demo/sweep/simple.yaml`
- Default output dir in this guide: `demo`

If you want a smaller run for coursework, set `max_concepts` in `simple.yaml` to a small value (e.g. 10-50).

## 1) Data Generation

Generate training data:

```bash
uv run scripts/generate.py \
  --config demo/sweep/simple.yaml \
  --mode training \
  --dump_dir demo
```

Generate latent inference data:

```bash
uv run scripts/generate.py \
  --config demo/sweep/simple.yaml \
  --mode latent \
  --dump_dir demo
```

## 2) Training

Train weakly supervised method + selected baselines:

```bash
uv run torchrun --nproc_per_node=$gpu_count scripts/train.py \
  --config demo/sweep/simple.yaml \
  --dump_dir demo
```

Replace `$gpu_count` with available GPUs.

## 3) Inference

### Concept detection (latent)

```bash
uv run torchrun --nproc_per_node=$gpu_count scripts/inference.py \
  --config demo/sweep/simple.yaml \
  --dump_dir demo \
  --mode latent
```

### Model steering

```bash
uv run torchrun --nproc_per_node=$gpu_count scripts/inference.py \
  --config demo/sweep/simple.yaml \
  --dump_dir demo \
  --mode steering
```

## 4) Evaluation

### Evaluate concept detection

```bash
uv run scripts/evaluate.py \
  --config demo/sweep/simple.yaml \
  --dump_dir demo \
  --mode latent
```

### Evaluate steering

```bash
uv run scripts/evaluate.py \
  --config demo/sweep/simple.yaml \
  --dump_dir demo \
  --mode steering
```

## Expected Outputs

Under `demo`, the main folders are:

- `generate/`: generated datasets and metadata
- `train/`: trained method parameters
- `inference/`: latent and steering inference outputs
- `evaluate/`: evaluation results and plots

