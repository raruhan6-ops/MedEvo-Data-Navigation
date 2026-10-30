"""Step 9: build Tables 1-5 -> outputs/tables.md (and tables.json).

Works with partial results: anything not yet run is shown as "—".

C2 - C1 confidence interval: each test question's correctness is averaged over
the available seeds of each condition, then the paired difference is bootstrapped
over questions (10,000 resamples). Per-seed pairs (C2_s1 vs C1_s1, ...) are also
reported. Note that C1 seeds vary data and training, while C2's data is fixed and
only training varies, so C2's SD is expected to be smaller.
"""
import argparse
import random
import statistics

from common import DATA, OUT, normalise, question_groups, read_json, read_jsonl, summarise, write_json

PAPER_REF = "Paper, Qwen3.5-9B on SLAKE: 76.80 (R0) → 78.23 (R3); not directly comparable"
DASH = "—"


def load(cond, split="test"):
    p = OUT / f"{cond}_{split}.jsonl"
    return read_jsonl(p) if p.exists() else None


def seeds_of(family):
    """Prediction lists for a family: C1 -> [C1_s1, C1_s2, ...], or [C3] if unseeded."""
    runs = sorted(OUT.glob(f"{family}_s*_test.jsonl"))
    if runs:
        return [(p.name[:-len("_test.jsonl")], read_jsonl(p)) for p in runs]
    single = load(family)
    return [(family, single)] if single else []


_SYNONYMS = {"axial": "transverse"}


def _lenient_tokens(text):
    toks = [_SYNONYMS.get(t, t) for t in normalise(text).split() if t != "plane"]
    return [t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t for t in toks]


def lenient_correct(pred, gold):
    """Secondary metric only: strict match plus axial=transverse, plural and 'plane' tolerance."""
    return int(_lenient_tokens(pred) == _lenient_tokens(gold))


def fmt(x, nd=2):
    return DASH if x is None else f"{x:.{nd}f}"


def mean_sd(xs):
    if not xs:
        return DASH
    if len(xs) == 1:
        return f"{xs[0]:.2f}"
    return f"{statistics.mean(xs):.2f} ± {statistics.stdev(xs):.2f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def per_question_mean(runs):
    acc = {}
    for _, preds in runs:
        for p in preds:
            acc.setdefault(p["qid"], []).append(p["correct"])
    return {q: sum(v) / len(v) for q, v in acc.items()}


def paired_bootstrap(a, b, n_boot, seed=0):
    """a, b: {qid: score}. Returns (diff in pp, lo, hi)."""
    qids = sorted(set(a) & set(b))
    d = [a[q] - b[q] for q in qids]
    rng = random.Random(seed)
    n = len(d)
    boots = sorted(sum(d[i] for i in rng.choices(range(n), k=n)) / n for _ in range(n_boot))
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    return 100 * sum(d) / n, 100 * lo, 100 * hi


def group_acc(preds, groups):
    acc = {}
    for p in preds:
        for g in question_groups(p, groups):
            acc.setdefault(g, []).append(p["correct"])
    return {g: 100 * sum(v) / len(v) for g, v in acc.items()}, {g: len(v) for g, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    groups = read_json(DATA / "groups.json")
    sel = read_json(OUT / "selection_stats.json") if (OUT / "selection_stats.json").exists() else {}
    B = sel.get("meta", {}).get("B", "B")
    md, js = ["# Results (SLAKE, English)\n", "Accuracy in %, differences in percentage points (pp).\n"], {}

    # Table 1
    rows = []
    for split, label in (("val", "Feedback (val)"), ("test", "Test")):
        p = load("C0", split)
        s = summarise(p) if p else {}
        rows.append([label, fmt(s.get("closed")), fmt(s.get("open")), fmt(s.get("open_recall")),
                     fmt(s.get("overall")), s.get("n", DASH)])
        js[f"t1_{split}"] = s
    md += ["## Table 1 — Base model (C0) zero-shot accuracy\n",
           table(["Split", "Closed", "Open", "Open recall", "Overall", "N questions"], rows), ""]

    # Table 2
    gap_path = OUT / "gap_C0.json"
    rows = []
    if gap_path.exists():
        gap = read_json(gap_path)["groups"]
        quota = sel.get("nav", {}).get("quota", {})
        for g, t in sorted(gap.items(), key=lambda kv: -kv[1]["error_rate"]):
            axis, name = g.split(":", 1)
            note = " *" if t["error_source"] == "overall" else ""
            rows.append([axis, name, t["n"], f"{100 * t['error_rate']:.1f}{note}", quota.get(g, DASH)])
    md += ["## Table 2 — Capability gaps on the feedback set (C0), sorted by error rate\n",
           table(["Axis", "Tag group", "N", "Error rate", f"Navigation quota (of {B})"], rows or [[DASH] * 5]),
           "\n\\* group below the minimum size; uses the overall feedback error rate.\n"]

    # Table 3
    c0 = load("C0")
    c0_overall = summarise(c0)["overall"] if c0 else None
    families = [("C0", "C0 Base", "0"), ("C1", "C1 Random", B), ("C2", "C2 Navigation", B),
                ("C3", "C3 All correct rollouts", sel.get("all", {}).get("accepted", DASH)),
                ("C4", "C4 Navigation, round 2 (α = 1)", sel.get("nav_r2", {}).get("accepted", DASH)),
                ("C5", "C5 Qwen API zero-shot", "0")]
    rows, fam_runs = [], {}
    for fam, label, n_train in families:
        runs = seeds_of(fam)
        fam_runs[fam] = runs
        if not runs and fam in ("C4", "C5"):
            continue
        sums = [summarise(p) for _, p in runs]
        overall = [s["overall"] for s in sums]
        delta = DASH if fam == "C0" or not overall or c0_overall is None else f"{statistics.mean(overall) - c0_overall:+.2f}"
        rows.append([label, n_train,
                     fmt(statistics.mean([s["closed"] for s in sums]) if sums else None),
                     fmt(statistics.mean([s["open"] for s in sums]) if sums else None),
                     mean_sd(overall) + (f" (n={len(sums)})" if len(sums) > 1 else ""), delta])
        js[f"t3_{fam}"] = {"runs": [n for n, _ in runs], "summaries": sums}
    md += ["## Table 3 — Main comparison on SLAKE test (equal budget B)\n",
           table(["Condition", "Training examples", "Closed", "Open", "Overall (mean ± SD)", "Δ vs C0"], rows), ""]

    c1, c2 = fam_runs["C1"], fam_runs["C2"]
    if c1 and c2:
        d, lo, hi = paired_bootstrap(per_question_mean(c2), per_question_mean(c1), args.n_boot)
        md.append(f"**C2 − C1 (seed-averaged): {d:+.2f} pp, 95% paired bootstrap CI [{lo:+.2f}, {hi:+.2f}]**"
                  f" ({args.n_boot:,} question resamples).\n")
        js["c2_minus_c1"] = {"diff": d, "lo": lo, "hi": hi}
        c1_by = {n.split("_s")[-1]: p for n, p in c1}
        for n, p in c2:
            s = n.split("_s")[-1]
            if s in c1_by:
                a = {x["qid"]: x["correct"] for x in p}
                b = {x["qid"]: x["correct"] for x in c1_by[s]}
                d, lo, hi = paired_bootstrap(a, b, args.n_boot, seed=int(s) if s.isdigit() else 0)
                md.append(f"- seed {s}: {d:+.2f} pp [{lo:+.2f}, {hi:+.2f}]")
        md.append("")

    # Robustness: lenient scoring, computed afterwards from saved predictions
    rows = []
    for fam, label, _ in families:
        runs = fam_runs[fam]
        if not runs:
            continue
        strict = [100 * statistics.mean(p["correct"] for p in preds) for _, preds in runs]
        loose = [100 * statistics.mean(lenient_correct(p["pred"], p["answer"]) for p in preds) for _, preds in runs]
        rows.append([label, mean_sd(strict), mean_sd(loose), f"{statistics.mean(loose) - statistics.mean(strict):+.2f}"])
    if rows:
        md += ["### Robustness — lenient scoring\n",
               table(["Condition", "Overall, strict", "Overall, lenient", "Lenient − strict"], rows), ""]
        if c1 and c2:
            lq = lambda runs: per_question_mean([(n, [{**p, "correct": lenient_correct(p["pred"], p["answer"])} for p in preds])
                                                 for n, preds in runs])
            d, lo, hi = paired_bootstrap(lq(c2), lq(c1), args.n_boot)
            md.append(f"C2 − C1 under lenient scoring: {d:+.2f} pp, 95% CI [{lo:+.2f}, {hi:+.2f}].\n")
            js["c2_minus_c1_lenient"] = {"diff": d, "lo": lo, "hi": hi}
        md.append("Lenient scoring is a secondary check, not the primary metric. Beyond the strict normalisation it "
                  "treats 'axial' as 'transverse', ignores a trailing plural 's', and drops the word 'plane' "
                  "(e.g. 'axial' = 'Transverse Plane', 'Lungs' = 'Lung'). Training data and selection used strict "
                  "scoring only.\n")
    md.append(f"Reference: {PAPER_REF}.\n")

    # Table 4
    rows = []
    if c0 and c1 and c2:
        acc0, n_test = group_acc(c0, groups)
        acc1 = [group_acc(p, groups)[0] for _, p in c1]
        acc2 = [group_acc(p, groups)[0] for _, p in c2]
        pool_share = sel.get("meta", {}).get("pool_group_share", {})
        nav_share = sel.get("nav", {}).get("group_share", {})
        for g in sorted(n_test, key=lambda g: -(statistics.mean(a.get(g, 0) for a in acc2)
                                                - statistics.mean(a.get(g, 0) for a in acc1))):
            m1 = statistics.mean(a.get(g, 0) for a in acc1)
            m2 = statistics.mean(a.get(g, 0) for a in acc2)
            targeted = DASH
            if g in nav_share and g in pool_share:
                targeted = f"yes ({nav_share[g]:.0%} vs {pool_share[g]:.0%})" if nav_share[g] > pool_share[g] \
                    else f"no ({nav_share[g]:.0%} vs {pool_share[g]:.0%})"
            rows.append([g, n_test[g], fmt(acc0[g], 1), fmt(m1, 1), fmt(m2, 1), f"{m2 - m1:+.1f}", targeted])
    md += ["## Table 4 — Per-group change on test (C1 vs C2)\n",
           table(["Tag group", "N test", "C0", "C1", "C2", "C2 − C1", "Targeted by navigation?"],
                 rows or [[DASH] * 7]),
           "\nTargeted = the group's share of navigation examples exceeds its share of usable pool questions "
           "(shown as nav share vs pool share). Small groups are noisy.\n"]

    # Table 5
    rows = []
    for key in sorted(k for k in sel if k != "meta"):
        s = sel[key]
        rows.append([key, s["questions_sampled"], s["with_correct"], s["accepted"],
                     f"{100 * s['rollout_yield']:.1f}", f"{100 * s['share_weakest3']:.1f}"])
    weakest = ", ".join(sel.get("meta", {}).get("weakest3", [])) or DASH
    md += ["## Table 5 — Data statistics per condition\n",
           table(["Selection", "Questions sampled", "With ≥1 correct rollout", "Accepted examples",
                  "Correct rollouts (%)", "Share from weakest 3 groups (%)"], rows or [[DASH] * 6]),
           f"\nWeakest 3 groups (C0 feedback): {weakest}. Questions sampled counts every pool question the "
           "selector walked, including those skipped for having no correct rollout.\n"]

    (OUT / "tables.md").write_text("\n".join(md), encoding="utf-8")
    write_json(OUT / "tables.json", js)
    print("\n".join(md))


if __name__ == "__main__":
    main()
