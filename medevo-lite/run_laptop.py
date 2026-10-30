"""Whole plan on one small GPU (e.g. a 6 GB laptop), Windows or Linux.

transformers + PEFT instead of vLLM + ms-swift; default model Qwen3.5-0.8B.
Every stage skips work whose output exists, so re-run it after a crash.

  .venv\\Scripts\\python run_laptop.py --quick          # ~15 min end-to-end test, separate output folders
  .venv\\Scripts\\python run_laptop.py                  # full run
  .venv\\Scripts\\python run_laptop.py --stop-after c0  # c0 | rollouts | select | seed1 | train

Order: C0 eval (val, test) -> gap table -> rollouts -> selection ->
       C1_s1 + C2_s1 train/eval -> remaining seeds + C3 -> tables.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-0.8B")
    ap.add_argument("--B", type=int, default=1500)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=16, help="eval batch (questions)")
    ap.add_argument("--rollout-batch", type=int, default=8, help="questions per rollout batch (x k sequences)")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--pool-limit", type=int, default=0, help="use only the first N pool questions")
    ap.add_argument("--stop-after", choices=["c0", "rollouts", "select", "seed1", "train"])
    ap.add_argument("--quick", action="store_true", help="tiny end-to-end test in outputs_quick/ and ckpt_quick/")
    args = ap.parse_args()

    # unbuffered so progress counters reach the log while a stage runs
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    eval_limit = []
    if args.quick:
        env.update(MEDEVO_OUT="outputs_quick", MEDEVO_CKPT="ckpt_quick")
        eval_limit = ["--limit", "48"]
        args.pool_limit, args.k, args.B, args.epochs, args.seeds = 64, 2, 24, 1, [1]
    out = ROOT / env.get("MEDEVO_OUT", "outputs")
    ckpt = ROOT / env.get("MEDEVO_CKPT", "ckpt")
    (ROOT / "logs").mkdir(exist_ok=True)

    def run(*cmd):
        print(f"\n$ {' '.join(map(str, cmd))}", flush=True)
        t0 = time.time()
        subprocess.run([PY, *map(str, cmd)], cwd=ROOT, env=env, check=True)
        print(f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    def stage(name):
        print(f"\n=== [{time.strftime('%a %H:%M:%S')}] {name}", flush=True)

    def stop(name):
        if args.stop_after == name:
            stage(f"stopping after '{name}'")
            run("scripts/08_tables.py")
            sys.exit(0)

    gen = ["--max-new-tokens", args.max_new_tokens]

    def evaluate(cond, split, adapter=None):
        if (out / f"{cond}_{split}.jsonl").exists():
            return
        extra = ["--adapter", adapter] if adapter else []
        run("scripts/02_eval.py", "--model", args.model, "--condition", cond, "--split", split,
            "--backend", "transformers", "--batch-size", args.batch_size, *gen, *eval_limit, *extra)

    def train_eval(data, seed, cond, val=False):
        run("scripts/06_train_hf.py", "--model", args.model, "--data", data, "--seed", seed,
            "--run", cond, "--epochs", args.epochs)
        adapter = ckpt / cond / "adapter"
        evaluate(cond, "test", adapter)
        if val:
            evaluate(cond, "val", adapter)

    if not (ROOT / "data" / "groups.json").exists():
        stage("data")
        run("scripts/00_download.py")
        run("scripts/01_build_groups.py")

    stage("C0 eval (val, test)")
    evaluate("C0", "val")
    evaluate("C0", "test")
    s = json.loads((out / "C0_val.summary.json").read_text(encoding="utf-8"))
    print("C0 val:", s)
    if not args.quick and (s["overall"] is None or s["overall"] < 10):
        sys.exit("C0 val accuracy < 10%: check outputs/C0_val.jsonl (response vs pred) before continuing")
    run("scripts/03_gap_analysis.py", "--condition", "C0")
    stop("c0")

    stage(f"rollouts (k={args.k})")
    if not (out / "rollouts_train.jsonl").exists():
        lim = ["--limit", args.pool_limit] if args.pool_limit else []
        run("scripts/04_rollouts.py", "--model", args.model, "--k", args.k, "--backend", "transformers",
            "--batch-size", args.rollout_batch, *gen, *lim)
    stop("rollouts")

    stage(f"selection (B={args.B})")
    if not (out / "sft" / "sft_nav.jsonl").exists():
        run("scripts/05_select.py", "--B", args.B, "--seeds", *args.seeds)
    stop("select")

    first = args.seeds[0]
    stage(f"first comparison: C1_s{first} vs C2_s{first}")
    train_eval(f"random_s{first}", first, f"C1_s{first}")
    train_eval("nav", first, f"C2_s{first}", val=True)  # val predictions feed the optional round 2 (C4)
    run("scripts/08_tables.py")
    stop("seed1")

    stage("remaining seeds + C3")
    for seed in args.seeds[1:]:
        train_eval(f"random_s{seed}", seed, f"C1_s{seed}")
        train_eval("nav", seed, f"C2_s{seed}")
    train_eval("all", first, "C3")
    stop("train")

    stage("tables")
    run("scripts/08_tables.py")
    print(f"\ndone: {out / 'tables.md'}")


if __name__ == "__main__":
    main()
