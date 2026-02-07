# QLoRA Fine-Tuning Runbook (H200 + Qwen3-8B)

This runbook is for your setup:
- GPU: **H200**
- Base model: **`Qwen/Qwen3-8B`**
- Budget cap: **~$100**

`$100 / $3.44 per hour ~= 29 hours` total wall-clock. Keep at least 20% buffer.

## 1. Create and connect to droplet

Use your GPU Droplet UI and create:
- 1x H200
- Ubuntu 22.04+
- Add your SSH key

Then SSH in:

```bash
ssh root@<YOUR_DROPLET_IP>
```

## 2. System setup on droplet

```bash
apt update && apt install -y git python3-venv python3-pip tmux
nvidia-smi
```

Expected: H200 appears in `nvidia-smi`.

## 3. Clone project and create fine-tune env

```bash
cd /workspace
# replace with your repo URL
# git clone <YOUR_REPO_URL> text2sql
cd text2sql

python3 -m venv .venv-ft
source .venv-ft/bin/activate
pip install --upgrade pip
pip install -r finetuning/requirements.txt
```

Optional (recommended for repeat runs):

```bash
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
```

If model requires auth:

```bash
huggingface-cli login
```

## 4. Prepare datasets

Build training/validation JSONL files from your local Spider assets.

```bash
python finetuning/prepare_data.py --project-root . --out-dir finetuning/data
```

You should get:
- `finetuning/data/stage1_train.jsonl`
- `finetuning/data/stage1_valid.jsonl`
- `finetuning/data/stage2_mixed_train.jsonl`
- `finetuning/data/stage2_mixed_valid.jsonl`
- `finetuning/data/stage2_spider2_only_train.jsonl`
- `finetuning/data/stage2_spider2_only_valid.jsonl`

## 5. Stage 1 train (Spider1 supervised core)

Run in `tmux` so training survives disconnects.

```bash
tmux new -s ft
source /workspace/text2sql/.venv-ft/bin/activate
cd /workspace/text2sql

python finetuning/train_qlora.py \
  --model-name Qwen/Qwen3-8B \
  --train-file finetuning/data/stage1_train.jsonl \
  --eval-file finetuning/data/stage1_valid.jsonl \
  --output-dir finetuning/runs/qwen3-8b-stage1 \
  --num-train-epochs 2 \
  --learning-rate 2e-4 \
  --max-seq-length 3072 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 16 \
  --save-steps 200 \
  --eval-steps 200
```

Detach tmux: `Ctrl+B`, then `D`.

Re-attach:

```bash
tmux attach -t ft
```

## 6. Stage 2 train (Spider2 adaptation)

Option A (recommended): mixed stage2 to reduce forgetting.

```bash
python finetuning/train_qlora.py \
  --model-name Qwen/Qwen3-8B \
  --adapter-path finetuning/runs/qwen3-8b-stage1 \
  --train-file finetuning/data/stage2_mixed_train.jsonl \
  --eval-file finetuning/data/stage2_mixed_valid.jsonl \
  --output-dir finetuning/runs/qwen3-8b-stage2 \
  --num-train-epochs 1.5 \
  --learning-rate 1e-4 \
  --max-seq-length 3072 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 16 \
  --save-steps 150 \
  --eval-steps 150
```

Option B: Spider2-only adaptation (faster, more overfit risk).

```bash
python finetuning/train_qlora.py \
  --model-name Qwen/Qwen3-8B \
  --adapter-path finetuning/runs/qwen3-8b-stage1 \
  --train-file finetuning/data/stage2_spider2_only_train.jsonl \
  --eval-file finetuning/data/stage2_spider2_only_valid.jsonl \
  --output-dir finetuning/runs/qwen3-8b-stage2-sp2only \
  --num-train-epochs 2 \
  --learning-rate 8e-5
```

## 7. Quick inference check

Create a schema context text file, then run:

```bash
cat > /tmp/schema.txt <<'EOF'
orders: order_id (int), customer_id (int), order_total (float), created_at (timestamp)
customers: customer_id (int), country (text)
EOF

python finetuning/run_inference.py \
  --model-path finetuning/runs/qwen3-8b-stage2 \
  --question "What is the total revenue by country in the last 30 days?" \
  --dialect bigquery \
  --db-id bigcommerce \
  --schema-file /tmp/schema.txt
```

## 8. Optional: merge adapter into standalone model

```bash
python finetuning/merge_adapter.py \
  --base-model Qwen/Qwen3-8B \
  --adapter-path finetuning/runs/qwen3-8b-stage2 \
  --output-dir finetuning/merged/qwen3-8b-stage2 \
  --dtype bf16
```

## 9. Budget control (important)

At `$3.44/hr`:
- 6h run = ~$20.64
- 12h run = ~$41.28
- 24h run = ~$82.56

Do this after each run:

```bash
nvidia-smi
```

When not training, **destroy the droplet** from DO UI. Stopped droplets still incur GPU cost.

## 10. Common fixes

OOM:
- Reduce `--max-seq-length` from `3072` to `2048`
- Reduce `--per-device-train-batch-size` from `2` to `1`
- Keep/increase `--gradient-accumulation-steps`

Slow throughput:
- Ensure no CPU fallback in logs
- Keep dataset on local NVMe (`/workspace`)

Loss diverges:
- Lower LR: `2e-4 -> 1e-4` (stage1) or `1e-4 -> 5e-5` (stage2)
- Use fewer epochs

## 11. What this pipeline does

- Stage 1: trains SQL generation behavior on Spider1 supervision.
- Stage 2: adapts to Spider2-style enterprise queries using available gold SQL subset.
- Keeps everything in LoRA adapters to control cost and speed.
