#!/usr/bin/env bash
# Step 8: evaluate every merged checkpoint on val and test with 02_eval.py.
# Run name = condition name (ckpt/C2_s1/merged -> outputs/C2_s1_{val,test}.jsonl).
#   bash scripts/07_eval_all.sh            # all checkpoints not yet evaluated
#   bash scripts/07_eval_all.sh C2_s1 C1_s1
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
runs=("$@")
if [ ${#runs[@]} -eq 0 ]; then
  for d in ckpt/*/merged; do runs+=("$(basename "$(dirname "$d")")"); done
fi
for name in "${runs[@]}"; do
  for split in val test; do
    if [ -f "outputs/${name}_${split}.jsonl" ]; then echo "skip $name $split"; continue; fi
    python scripts/02_eval.py --model "ckpt/$name/merged" --condition "$name" --split "$split" \
      2>&1 | tee "logs/eval_${name}_${split}.log"
  done
done
