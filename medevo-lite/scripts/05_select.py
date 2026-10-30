"""Step 6: build equal-budget SFT sets in ms-swift format.

  outputs/sft/sft_random_s{1,2,3}.jsonl   C1  random questions, B accepted examples
  outputs/sft/sft_nav.jsonl          C2  navigation (Eq. 15), B accepted examples
  outputs/sft/sft_all.jsonl          C3  every question with >=1 correct rollout
  outputs/selection_stats.json        quotas, yield and group shares (Tables 2, 4, 5)

One accepted example = one question with one of its correct rollouts. B counts
accepted examples. Both selectors walk the pool question by question and skip
questions with no correct rollout; "questions sampled" counts every question
walked, so it shows how many rolled-out questions each condition needed.

Navigation: score_g = err_g + alpha * regress_g; quota_g = round(B * score_g / sum);
groups are visited by score, and inside a group questions are ranked by failed
rollouts (most first). Unused quota is redistributed over groups that still have
candidates, in proportion to score, until B examples are accepted. A question in
several groups is counted once.

  python scripts/05_select.py --B 1500
  python scripts/05_select.py --conditions nav --gap gap_C2_s1 --alpha 1 --nav-name nav_r2   # C4
"""
import argparse
import random

from common import DATA, OUT, PROMPT, SFT, image_path, question_groups, read_json, read_jsonl, write_json, write_jsonl


def load_rollouts(names):
    """Later files override earlier ones per qid (e.g. a k=8 re-run of weak groups)."""
    by_qid = {}
    for name in names:
        for r in read_jsonl(OUT / name):
            by_qid[r["qid"]] = r
    return list(by_qid.values())


def to_sft(q, rng):
    resp = rng.choice([x for x, ok in zip(q["responses"], q["correct"]) if ok])
    return {
        "messages": [
            {"role": "user", "content": "<image>" + PROMPT.format(question=q["question"])},
            {"role": "assistant", "content": resp.strip()},
        ],
        "images": [image_path(q)],
    }


def select_random(pool, B, seed):
    order = pool[:]
    random.Random(seed).shuffle(order)
    chosen, visited = [], []
    for q in order:
        if len(chosen) == B:
            break
        visited.append(q)
        if any(q["correct"]):
            chosen.append(q)
    return chosen, visited, {}


def select_nav(pool, B, groups, gap, alpha, seed):
    score = {g: gap[g]["error_rate"] + alpha * gap[g].get("regress_rate", 0.0) for g in groups["groups"]}
    total = sum(score.values()) or 1.0
    quota = {g: round(B * s / total) for g, s in score.items()}

    rng = random.Random(seed)
    cands = {g: [] for g in groups["groups"]}
    for q in pool:
        for g in question_groups(q, groups):
            cands[g].append(q)
    for g in cands:
        rng.shuffle(cands[g])  # random tie-break among equal failure counts
        cands[g].sort(key=lambda q: -q["n_fail"])

    pointer = {g: 0 for g in cands}
    chosen, chosen_ids, visited = [], set(), {}

    def take(g, n):
        got = 0
        while got < n and pointer[g] < len(cands[g]) and len(chosen) < B:
            q = cands[g][pointer[g]]
            pointer[g] += 1
            if q["qid"] in chosen_ids:
                continue
            visited[q["qid"]] = q
            if any(q["correct"]):
                chosen.append(q)
                chosen_ids.add(q["qid"])
                got += 1

    order = sorted(score, key=score.get, reverse=True)
    for g in order:
        take(g, quota[g])
    while len(chosen) < B:
        active = [g for g in order if pointer[g] < len(cands[g])]
        if not active:
            break
        deficit = B - len(chosen)
        act_total = sum(score[g] for g in active) or 1.0
        for g in active:
            take(g, max(1, round(deficit * score[g] / act_total)))

    return chosen, list(visited.values()), {"score": score, "quota": quota}


def group_shares(qs, groups):
    counts = {g: 0 for g in groups["groups"]}
    for q in qs:
        for g in question_groups(q, groups):
            counts[g] += 1
    n = max(1, len(qs))
    return {g: round(c / n, 4) for g, c in counts.items()}


def stats_for(name, chosen, visited, groups, weakest, extra):
    """visited: pool questions walked (accepted or skipped for having no correct rollout)."""
    weak = set(weakest)
    return {
        "condition": name,
        "questions_sampled": len(visited),
        "with_correct": sum(1 for q in visited if any(q["correct"])),
        "accepted": len(chosen),
        "rollout_yield": round(sum(sum(q["correct"]) for q in visited) / max(1, sum(len(q["correct"]) for q in visited)), 4),
        "share_weakest3": round(sum(1 for q in chosen if set(question_groups(q, groups)) & weak) / max(1, len(chosen)), 4),
        "group_share": group_shares(chosen, groups),
        **extra,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1500)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--gap", default="gap_C0", help="outputs/<gap>.json from 03_gap_analysis.py")
    ap.add_argument("--rollouts", nargs="+", default=["rollouts_train.jsonl"])
    ap.add_argument("--conditions", nargs="+", default=["random", "nav", "all"], choices=["random", "nav", "all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--nav-name", default="nav")
    ap.add_argument("--nav-seed", type=int, default=0)
    args = ap.parse_args()

    groups = read_json(DATA / "groups.json")
    gap_file = read_json(OUT / f"{args.gap}.json")
    gap = gap_file["groups"]
    pool = load_rollouts(args.rollouts)
    pool.sort(key=lambda q: q["qid"])
    n_ok = sum(1 for q in pool if any(q["correct"]))

    B = args.B
    if B > n_ok:
        print(f"WARNING: B={B} > {n_ok} questions with a correct rollout; lowering B to {n_ok}")
        B = n_ok
    elif B > 0.6 * n_ok:
        print(f"WARNING: B={B} is {B / n_ok:.0%} of the {n_ok} usable questions; selection has little room")

    c0_gap = read_json(OUT / "gap_C0.json")["groups"]
    weakest = [g for g, _ in sorted(((g, t["error_rate"]) for g, t in c0_gap.items() if t["error_source"] == "group"),
                                    key=lambda kv: -kv[1])[:3]]

    stats_path = OUT / "selection_stats.json"
    try:
        all_stats = read_json(stats_path)
    except FileNotFoundError:
        all_stats = {}
    all_stats["meta"] = {"B": B, "pool": len(pool), "pool_with_correct": n_ok, "weakest3": weakest,
                         "pool_group_share": group_shares([q for q in pool if any(q["correct"])], groups)}

    def emit(name, chosen, visited, extra, seed):
        rng = random.Random(seed)
        write_jsonl(SFT / f"sft_{name}.jsonl", [to_sft(q, rng) for q in chosen])
        all_stats[name] = stats_for(name, chosen, visited, groups, weakest, extra)
        s = all_stats[name]
        print(f"{name:12s} sampled={s['questions_sampled']:5d} accepted={s['accepted']:5d} "
              f"weakest3_share={s['share_weakest3']:.3f}")

    if "random" in args.conditions:
        for seed in args.seeds:
            chosen, visited, extra = select_random(pool, B, seed)
            emit(f"random_s{seed}", chosen, visited, {"seed": seed}, seed)
    if "nav" in args.conditions:
        chosen, visited, extra = select_nav(pool, B, groups, gap, args.alpha, args.nav_seed)
        emit(args.nav_name, chosen, visited, {**extra, "alpha": args.alpha, "gap": args.gap}, args.nav_seed)
    if "all" in args.conditions:
        chosen = [q for q in pool if any(q["correct"])]
        emit("all", chosen, pool, {}, 0)

    write_json(stats_path, all_stats)


if __name__ == "__main__":
    main()
