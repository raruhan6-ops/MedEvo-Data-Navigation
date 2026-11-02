"""Learning trajectories: retrain C1_s1 and C2_s1 with intermediate checkpoints.

Same data, seed and hyperparameters as the main runs (06_train_hf.py defaults),
plus --save-every. Each checkpoint is evaluated on the feedback (val) split, so
test still feeds nothing; the final adapters are also evaluated on test, to
compare with the original C1_s1 / C2_s1 runs (same seed retrained = run-to-run
reproducibility on this GPU). Skips finished work, so re-run it after a crash.

  python scripts/10_trajectory.py
  python scripts/10_trajectory.py --summary-only   # just rebuild outputs/trajectory.json
"""
import argparse
import json
import os
import subprocess
import sys
import time

from common import CKPT, LOGS, OUT, ROOT, read_json, read_jsonl, write_json

RUNS = {"C1_s1_traj": ("random_s1", 1, "C1_s1"), "C2_s1_traj": ("nav", 1, "C2_s1")}


def run(*cmd):
    print(f"\n$ {' '.join(map(str, cmd))}", flush=True)
    t0 = time.time()
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    subprocess.run([sys.executable, *map(str, cmd)], cwd=ROOT, env=env, check=True)
    print(f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)


def checkpoints(run_name):
    """(step, adapter dir) in step order; the final adapter is the last step."""
    d = CKPT / run_name
    out = sorted((int(p.name[4:]), p) for p in d.glob("step*") if (p / "adapter_config.json").exists())
    if (d / "adapter" / "adapter_config.json").exists():
        meta = [json.loads(l) for l in open(LOGS / f"train_{run_name}.log", encoding="utf-8") if l.startswith("{\"model")]
        out.append((meta[-1]["optimizer_steps"], d / "adapter"))
    return out


def summary():
    res = {}
    for run_name, (_, _, orig) in RUNS.items():
        curve = [{"step": 0, **{k: read_json(OUT / "C0_val.summary.json")[k] for k in ("overall", "closed", "open")}}]
        for step, _ in checkpoints(run_name):
            p = OUT / "traj" / f"{run_name}_step{step:03d}_val.summary.json"
            if p.exists():
                s = read_json(p)
                curve.append({"step": step, "overall": s["overall"], "closed": s["closed"], "open": s["open"],
                              "seconds": s["seconds"]})
        entry = {"val_curve": curve}
        m = LOGS / f"metrics_{run_name}.jsonl"
        if m.exists():
            entry["steps"] = read_jsonl(m)
        # reproducibility: retrained final adapter vs the original run on test
        a, b = OUT / "traj" / f"{run_name}_final_test.jsonl", OUT / f"{orig}_test.jsonl"
        if a.exists() and b.exists():
            pa = {r["qid"]: r for r in read_jsonl(a)}
            pb = {r["qid"]: r for r in read_jsonl(b)}
            common_ids = sorted(pa.keys() & pb.keys())
            entry["repro_vs_" + orig] = {
                "retrained_test": round(100 * sum(pa[q]["correct"] for q in common_ids) / len(common_ids), 2),
                "original_test": round(100 * sum(pb[q]["correct"] for q in common_ids) / len(common_ids), 2),
                "same_correctness_pct": round(100 * sum(pa[q]["correct"] == pb[q]["correct"] for q in common_ids) / len(common_ids), 2),
                "identical_answer_pct": round(100 * sum(pa[q]["pred"] == pb[q]["pred"] for q in common_ids) / len(common_ids), 2),
                "n": len(common_ids),
            }
        res[run_name] = entry
    write_json(OUT / "trajectory.json", res)
    for k, v in res.items():
        print(k, [(c["step"], c["overall"]) for c in v["val_curve"]], v.get("repro_vs_" + RUNS[k][2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-0.8B")
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    if not args.summary_only:
        gen = ["--backend", "transformers", "--batch-size", 16, "--max-new-tokens", 160]
        for run_name, (data, seed, _) in RUNS.items():
            run("scripts/06_train_hf.py", "--model", args.model, "--data", data, "--seed", seed,
                "--run", run_name, "--epochs", 2, "--save-every", args.save_every)
            ckpts = checkpoints(run_name)
            for step, adapter in ckpts:
                cond = f"traj/{run_name}_step{step:03d}"
                if not (OUT / f"{cond}_val.jsonl").exists():
                    run("scripts/02_eval.py", "--model", args.model, "--condition", cond, "--split", "val",
                        "--adapter", adapter.relative_to(ROOT), *gen)
            cond = f"traj/{run_name}_final"
            if not (OUT / f"{cond}_test.jsonl").exists():
                run("scripts/02_eval.py", "--model", args.model, "--condition", cond, "--split", "test",
                    "--adapter", ckpts[-1][1].relative_to(ROOT), *gen)
    summary()


if __name__ == "__main__":
    main()
