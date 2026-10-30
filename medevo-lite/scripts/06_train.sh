#!/usr/bin/env bash
# Step 7: LoRA SFT from the base model, then merge the adapter for evaluation.
# Same hyperparameters for every condition; only dataset and seed change.
# Check flags against the installed ms-swift version (pinned in requirements.lock.txt).
#
#   export BASE_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
#   bash scripts/06_train.sh                     # default runs: C1_s1-3, C2_s1-3, C3
#   bash scripts/06_train.sh nav 1 C2_s1         # one run: <dataset stem> <seed> <run name>
#   bash scripts/06_train.sh nav_r2 1 C4         # round 2 (after 05_select.py --nav-name nav_r2)
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=${BASE_MODEL:?set BASE_MODEL, e.g. Qwen/Qwen2.5-VL-3B-Instruct}
mkdir -p ckpt logs

train() {
  local ds=$1 seed=$2 name=$3 out=ckpt/$3
  if [ -d "$out/merged" ]; then echo "skip $name (already merged)"; return; fi
  echo "== train $name: outputs/sft/sft_$ds.jsonl seed=$seed"
  swift sft \
    --model "$BASE" \
    --train_type lora \
    --lora_rank 8 --lora_alpha 32 --target_modules all-linear \
    --freeze_vit true \
    --dataset "outputs/sft/sft_$ds.jsonl" \
    --split_dataset_ratio 0 \
    --num_train_epochs 2 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_length 2048 \
    --torch_dtype bfloat16 \
    --save_only_model true --save_total_limit 1 \
    --seed "$seed" --data_seed "$seed" \
    --output_dir "$out" 2>&1 | tee "logs/train_$name.log"
  local last
  last=$(ls -d "$out"/v*/checkpoint-* | sort -V | tail -n 1)
  swift export --adapters "$last" --merge_lora true --output_dir "$out/merged" 2>&1 | tee -a "logs/train_$name.log"
}

if [ $# -eq 3 ]; then
  train "$1" "$2" "$3"
else
  for s in 1 2 3; do
    train "random_s$s" "$s" "C1_s$s"
    train nav "$s" "C2_s$s"
  done
  train all 1 C3
fi
