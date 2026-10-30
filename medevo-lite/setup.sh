#!/usr/bin/env bash
# One-shot setup on the GPU machine (Linux + CUDA). Stops at the first failure.
#
#   conda create -n medevo python=3.10 -y && conda activate medevo
#   bash setup.sh                                   # default model: Qwen/Qwen3.5-4B (paper family)
#   MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct bash setup.sh   # fallback if Qwen3.5 fails the smoke test
#
# Steps: GPU check -> pip install -> requirements.lock.txt -> model weights
# (Hugging Face, falling back to ModelScope) -> SLAKE download + tag groups ->
# 20-question inference smoke test. Writes env.sh, which run_pipeline.sh sources.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data outputs ckpt logs models
MODEL_ID=${MODEL_ID:-Qwen/Qwen3.5-4B}
step() { echo; echo "=== [$(date +%H:%M:%S)] $*"; }

step "1/6 GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c 'import sys; assert sys.version_info >= (3, 9), sys.version; print(sys.version)'

step "2/6 pip install"
# vLLM pins a matching torch, so install it first. If it fails, install torch alone
# and the scripts fall back to plain transformers generation (slower, same results).
if ! pip install vllm; then
  echo "WARNING: vLLM failed to install; using the transformers backend"
  pip install torch torchvision
fi
grep -vE '^\s*(#|$|vllm)' requirements.txt | sed 's/#.*//' | xargs pip install
pip install modelscope huggingface_hub
pip freeze > requirements.lock.txt
python - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__)
assert torch.cuda.is_available(), "torch cannot see the GPU"
try:
    import vllm; print("vllm", vllm.__version__)
except ImportError:
    print("vllm not installed (transformers backend)")
import swift; print("ms-swift", getattr(swift, "__version__", "?"))
EOF

# Training flags used by 06_train.sh must exist in this ms-swift version.
missing=""
help=$(swift sft --help 2>&1 || true)
for flag in train_type lora_rank lora_alpha target_modules freeze_vit split_dataset_ratio \
            save_only_model save_total_limit data_seed max_length torch_dtype; do
  grep -q -- "--$flag" <<<"$help" || missing="$missing $flag"
done
[ -z "$missing" ] && echo "ms-swift flags OK" || echo "WARNING: swift sft does not list:$missing (edit scripts/06_train.sh)"

step "3/6 model weights: $MODEL_ID"
LOCAL=models/$(basename "$MODEL_ID")
if [ ! -f "$LOCAL/config.json" ]; then
  HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} timeout 900 \
    huggingface-cli download "$MODEL_ID" --local-dir "$LOCAL" \
  || { echo "Hugging Face failed; trying ModelScope"; modelscope download --model "$MODEL_ID" --local_dir "$LOCAL"; }
fi
test -f "$LOCAL/config.json"
cat > env.sh <<EOF
# written by setup.sh on $(date)
export MODEL_ID=$MODEL_ID
export BASE_MODEL=$PWD/$LOCAL
EOF
echo "BASE_MODEL=$PWD/$LOCAL (saved to env.sh)"

step "4/6 SLAKE + tag groups"
python scripts/00_download.py
python scripts/01_build_groups.py

step "5/6 inference smoke test (20 val questions)"
python scripts/02_eval.py --model "$LOCAL" --condition smoke --split val --limit 20
python - <<'EOF'
import json
rows = [json.loads(l) for l in open("outputs/smoke_val.jsonl", encoding="utf-8")]
# Some models (e.g. Qwen3.5-0.8B) answer with the bare word instead of an
# 'Answer:' line; the parser falls back to the last line, so check the outcome.
has_tag = sum("answer:" in r["response"].lower() for r in rows)
correct = sum(r["correct"] for r in rows)
print(f"responses with an 'Answer:' line: {has_tag}/{len(rows)}   correct: {correct}/{len(rows)}")
for r in rows[:3]:
    print("-" * 60); print("Q:", r["question"], "| gold:", r["answer"]); print(r["response"].strip()[-300:]); print("parsed:", r["pred"])
if correct < len(rows) // 5:
    raise SystemExit("fewer than 20% correct on the smoke test; read the responses above and fix the prompt/parser")
EOF

step "6/6 done"
echo "Next:  nohup bash run_pipeline.sh > logs/pipeline.log 2>&1 &   then   tail -f logs/pipeline.log"
