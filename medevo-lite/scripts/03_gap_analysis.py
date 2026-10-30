"""Step 4: feedback diagnosis (Eq. 14) -> outputs/gap_<condition>.json (Table 2).

Error rate per tag group on the feedback (val) set. A group that is still
smaller than MIN_GROUP after back-off uses the overall val error rate.

Optional regression term for round 2 (C4): pass --prev with the previous
round's val predictions; N^{1->0}_g counts questions correct before, wrong now.

  python scripts/03_gap_analysis.py --condition C0
  python scripts/03_gap_analysis.py --condition C2_s1 --prev C0
"""
import argparse

from common import DATA, MIN_GROUP, OUT, question_groups, read_json, read_jsonl, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="C0")
    ap.add_argument("--prev", help="previous-round condition for the regression term")
    args = ap.parse_args()

    groups = read_json(DATA / "groups.json")
    preds = read_jsonl(OUT / f"{args.condition}_val.jsonl")
    prev = {p["qid"]: p["correct"] for p in read_jsonl(OUT / f"{args.prev}_val.jsonl")} if args.prev else {}
    overall_err = 1 - sum(p["correct"] for p in preds) / len(preds)

    stats = {g: {"n": 0, "wrong": 0, "regress": 0} for g in groups["groups"]}
    for p in preds:
        for g in question_groups(p, groups):
            s = stats[g]
            s["n"] += 1
            s["wrong"] += 1 - p["correct"]
            if prev.get(p["qid"]) == 1 and p["correct"] == 0:
                s["regress"] += 1

    table = {}
    for g, s in stats.items():
        small = s["n"] < MIN_GROUP
        table[g] = {
            "n": s["n"],
            "error_rate": round(overall_err if small else s["wrong"] / s["n"], 4),
            "error_source": "overall" if small else "group",
            "regress_rate": round(s["regress"] / s["n"], 4) if s["n"] else 0.0,
        }

    write_json(OUT / f"gap_{args.condition}.json",
               {"condition": args.condition, "prev": args.prev, "overall_error": round(overall_err, 4), "groups": table})
    for g, t in sorted(table.items(), key=lambda kv: -kv[1]["error_rate"]):
        flag = " (overall)" if t["error_source"] == "overall" else ""
        print(f"{g:35s} n={t['n']:4d}  err={t['error_rate']:.3f}{flag}  regress={t['regress_rate']:.3f}")


if __name__ == "__main__":
    main()
