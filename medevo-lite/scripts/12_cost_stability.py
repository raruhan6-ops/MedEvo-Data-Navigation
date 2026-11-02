"""Computational cost and training stability from the existing logs (CPU only).

Reads logs/train_<run>.log (the full run's block, not the --quick lines at the
top), logs/full_run.log + logs/c4_run.log for stage times and the
outputs/*.summary.json eval times. Writes outputs/cost_stability.json and .md.

  python scripts/12_cost_stability.py
"""
import json
import re
import statistics as st

from common import LOGS, OUT, read_json, write_json

RUNS = ["C1_s1", "C1_s2", "C1_s3", "C2_s1", "C2_s2", "C2_s3", "C3", "C4"]
STEP = re.compile(r"step (\d+)/(\d+)\s+loss ([\d.]+)\s+lr ([\de.+-]+)\s+([\d.]+)s/step")


def train_log(run):
    """Last full-size block of a training log: meta, [(step, loss)], s/step, done."""
    blocks, cur = [], None
    for line in open(LOGS / f"train_{run}.log", encoding="utf-8"):
        if line.startswith('{"model"'):
            cur = {"meta": json.loads(line), "curve": [], "rate": None, "done": None}
            blocks.append(cur)
        elif (m := STEP.match(line)) and cur:
            cur["curve"].append((int(m[1]), float(m[3])))
            cur["rate"] = float(m[5])
        elif line.startswith('{"run"') and cur:
            cur["done"] = json.loads(line)
    return [b for b in blocks if b["meta"]["n_examples"] > 24][-1]


def main():
    runs = {r: train_log(r) for r in RUNS}
    res = {"hardware": "RTX 3060 Laptop GPU, 6 GB", "model": "Qwen3.5-0.8B, LoRA r=8 (5.41M trainable, 0.63%)",
           "training": {}, "eval": {}, "stages_min": {}}
    for r, b in runs.items():
        res["training"][r] = {"examples": b["meta"]["n_examples"], "optimizer_steps": b["meta"]["optimizer_steps"],
                              "minutes": b["done"]["minutes"], "s_per_step": b["rate"],
                              "peak_gpu_gb": b["done"]["peak_gpu_gb"],
                              "examples_per_s": round(b["meta"]["n_examples"] * b["meta"]["epochs"] / (b["done"]["minutes"] * 60), 2),
                              "loss_first": b["curve"][0][1], "loss_last3": round(st.mean(l for _, l in b["curve"][-3:]), 4),
                              "curve": b["curve"]}
    for cond in ["C0", *RUNS]:
        p = OUT / f"{cond}_test.summary.json"
        if p.exists():
            s = read_json(p)
            res["eval"][cond] = {"seconds": s["seconds"], "questions_per_s": round(s["n"] / s["seconds"], 1)}

    # stage wall clock from the run logs: "=== [time] name" then "  (x min)" per command
    for log in ("full_run.log", "c4_run.log"):
        stage = None
        for line in open(LOGS / log, encoding="utf-8", errors="replace"):
            if line.startswith("=== ["):
                stage = line.split("] ", 1)[1].strip()
            elif (m := re.match(r"\s+\(([\d.]+) min\)", line)) and stage:
                res["stages_min"][stage] = round(res["stages_min"].get(stage, 0) + float(m[1]), 1)
    res["stages_min"]["C4 (train + eval)"] = round(runs["C4"]["done"]["minutes"] + res["eval"]["C4"]["seconds"] / 60, 1)

    # seed-to-seed spread of the logged (10-step mean) loss, per condition
    spread = {}
    for cond in ("C1", "C2"):
        curves = [runs[f"{cond}_s{s}"]["curve"] for s in (1, 2, 3)]
        spread[cond] = [{"step": c[0][0], "mean": round(st.mean(x[1] for x in c), 4), "sd": round(st.stdev(x[1] for x in c), 4)}
                        for c in zip(*curves)]
    res["loss_seed_spread"] = spread
    write_json(OUT / "cost_stability.json", res)

    t = res["training"]
    lines = ["# Computational cost and training stability", "",
             f"{res['hardware']}; {res['model']}; bf16, batch 1 x 16 accumulation, 2 epochs.", "",
             "| Run | Examples | Steps | Minutes | s/step | Examples/s | Peak GPU (GB) | Loss, first 10 steps | Loss, last 30 steps |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r, v in t.items():
        lines.append(f"| {r} | {v['examples']} | {v['optimizer_steps']} | {v['minutes']} | {v['s_per_step']} | "
                     f"{v['examples_per_s']} | {v['peak_gpu_gb']} | {v['loss_first']:.3f} | {v['loss_last3']:.3f} |")
    lines += ["", "| Stage | Wall clock (min) |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in res["stages_min"].items()]
    lines += ["", "| Eval (test, 1061 questions) | Seconds | Questions/s |", "|---|---|---|"]
    lines += [f"| {k} | {v['seconds']} | {v['questions_per_s']} |" for k, v in res["eval"].items()]
    lines += ["", "Loss across the 3 training seeds (10-step means): mean ± SD", "",
              "| Step | C1 Random | C2 Navigation |", "|---|---|---|"]
    for a, b in zip(spread["C1"], spread["C2"]):
        lines.append(f"| {a['step']} | {a['mean']:.3f} ± {a['sd']:.3f} | {b['mean']:.3f} ± {b['sd']:.3f} |")
    (OUT / "cost_stability.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
