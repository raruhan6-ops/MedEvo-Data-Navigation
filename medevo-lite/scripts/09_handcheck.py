"""Hand-check of the answer parser and scorer (plan: ~50 predictions per condition).

  python scripts/09_handcheck.py sample      # -> outputs/handcheck/<cond>.csv, 50 random test predictions each
  (fill in parser_ok and human_correct with y / n in Excel or any editor, save as CSV)
  python scripts/09_handcheck.py summary     # -> outputs/handcheck/summary.md

parser_ok      y if `pred` is the answer the response actually gives
human_correct  y if a person would accept `pred` for the gold `answer`
"""
import argparse
import csv
import random

from common import OUT, read_jsonl

CONDS = ["C0", "C1_s1", "C2_s1", "C3"]
FIELDS = ["qid", "answer_type", "question", "answer", "response", "pred", "correct", "parser_ok", "human_correct", "note"]
DIR = OUT / "handcheck"


def sample(n, seed):
    DIR.mkdir(parents=True, exist_ok=True)
    for cond in CONDS:
        preds = read_jsonl(OUT / f"{cond}_test.jsonl")
        rows = random.Random(seed).sample(preds, n)
        path = DIR / f"{cond}.csv"
        if path.exists():
            print(f"skip {path.name}: exists (delete it to resample)")
            continue
        # utf-8-sig so Excel opens it with the right encoding
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({**r, "parser_ok": "", "human_correct": "", "note": ""})
        print(f"{path.relative_to(OUT.parent)}: {n} rows")


def yes(x):
    return str(x).strip().lower() in ("y", "yes", "1", "true")


def summary():
    md = ["# Hand-check of parser and scorer (SLAKE test)\n",
          "| Condition | Checked | Parser right | Scorer agrees with person | Strict too harsh | Strict too lenient |",
          "|---|---|---|---|---|---|"]
    notes = []
    for cond in CONDS:
        path = DIR / f"{cond}.csv"
        if not path.exists():
            continue
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["parser_ok"].strip() and r["human_correct"].strip()]
        if not rows:
            print(f"{cond}: nothing filled in yet")
            continue
        parser = sum(yes(r["parser_ok"]) for r in rows)
        harsh = [r for r in rows if r["correct"] == "0" and yes(r["human_correct"])]
        lenient = [r for r in rows if r["correct"] == "1" and not yes(r["human_correct"])]
        agree = len(rows) - len(harsh) - len(lenient)
        md.append(f"| {cond} | {len(rows)} | {parser} ({100 * parser / len(rows):.0f}%) | "
                  f"{agree} ({100 * agree / len(rows):.0f}%) | {len(harsh)} | {len(lenient)} |")
        for r in harsh + lenient + [r for r in rows if not yes(r["parser_ok"])]:
            notes.append(f"- {cond} q{r['qid']}: gold `{r['answer']}`, pred `{r['pred']}`, "
                         f"strict {r['correct']}, person {r['human_correct']}" + (f" ({r['note']})" if r["note"] else ""))
    md += ["", "Strict too harsh = marked wrong, a person would accept it. "
           "Strict too lenient = marked right, a person would not.", ""]
    if notes:
        md += ["## Disagreements and parser errors\n", *dict.fromkeys(notes)]
    (DIR / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sample", "summary"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sample(args.n, args.seed) if args.mode == "sample" else summary()


if __name__ == "__main__":
    main()
