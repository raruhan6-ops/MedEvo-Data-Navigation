#!/usr/bin/env bash
# Runs the whole plan after setup.sh, in critical-path order. Every stage skips
# work whose output already exists, so it is safe to re-run after a crash.
#
#   nohup bash run_pipeline.sh > logs/pipeline.log 2>&1 &
#   STOP_AFTER=select bash run_pipeline.sh     # stop after a stage: c0 | rollouts | select | seed1 | train
#   B=1000 bash run_pipeline.sh                # smaller budget if few correct rollouts
#
# Order: C0 eval (val, test) -> gap table -> rollouts -> selection ->
#        C1_s1 + C2_s1 train/eval (first comparison) -> remaining seeds + C3 -> tables.
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
B=${B:-1500}
STOP_AFTER=${STOP_AFTER:-}
mkdir -p logs
stage() { echo; echo "=== [$(date '+%a %H:%M:%S')] $*"; }
maybe_stop() { if [ "$STOP_AFTER" = "$1" ]; then stage "stopping after '$1'"; python scripts/08_tables.py > /dev/null; exit 0; fi; }

stage "C0 eval (val, test)"
for split in val test; do
  [ -f "outputs/C0_${split}.jsonl" ] || python scripts/02_eval.py --model "$BASE_MODEL" --condition C0 --split "$split"
done
python - <<'EOF'
import json
s = json.load(open("outputs/C0_val.summary.json"))
print("C0 val:", s)
# Gate: a near-zero score almost always means the answer parser is broken.
if s["overall"] is None or s["overall"] < 10:
    raise SystemExit("C0 val accuracy < 10%: check outputs/C0_val.jsonl (response vs pred) before continuing")
EOF
python scripts/03_gap_analysis.py --condition C0
maybe_stop c0

stage "rollouts on the training pool (k=4)"
[ -f outputs/rollouts_train.jsonl ] || python scripts/04_rollouts.py --model "$BASE_MODEL" --k 4
maybe_stop rollouts

stage "selection (B=$B)"
[ -f outputs/sft/sft_nav.jsonl ] || python scripts/05_select.py --B "$B"
maybe_stop select

stage "first comparison: C1_s1 vs C2_s1"
bash scripts/06_train.sh random_s1 1 C1_s1
bash scripts/06_train.sh nav 1 C2_s1
bash scripts/07_eval_all.sh C1_s1 C2_s1
python scripts/08_tables.py > /dev/null && grep -A 8 "Table 3" outputs/tables.md
maybe_stop seed1

stage "remaining seeds + C3"
bash scripts/06_train.sh
bash scripts/07_eval_all.sh
maybe_stop train

stage "tables"
python scripts/08_tables.py > /dev/null
echo "done: outputs/tables.md"
