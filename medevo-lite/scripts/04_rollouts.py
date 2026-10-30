"""Step 5: sample k responses per pool (train) question from the base model.

Writes outputs/rollouts_train.jsonl. Each row keeps all k responses, which of
them are correct, and n_fail (the "historical error" used to rank questions).

  python scripts/04_rollouts.py --model Qwen/Qwen2.5-VL-3B-Instruct --k 4
  # weak groups only, more samples (risk fallback):
  python scripts/04_rollouts.py --model ... --k 8 --only-groups content_type:KG location:Neck --out rollouts_train_k8.jsonl
"""
import argparse
import time

from common import DATA, OUT, load_split, parse_answer, question_groups, read_json, score, write_jsonl
from infer import make_backend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "transformers"])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8, help="questions per batch, transformers backend only")
    ap.add_argument("--only-groups", nargs="*", help="restrict to questions in these group ids")
    ap.add_argument("--out", default="rollouts_train.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = load_split("train")
    if args.only_groups:
        groups = read_json(DATA / "groups.json")
        wanted = set(args.only_groups)
        rows = [r for r in rows if wanted & set(question_groups(r, groups))]
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} pool questions x k={args.k}")

    backend = make_backend(args.backend, args.model, args.max_new_tokens, batch_size=args.batch_size)
    t0 = time.time()
    responses = backend.generate(rows, n=args.k, temperature=args.temperature, seed=args.seed)

    out = []
    for r, resps in zip(rows, responses):
        correct = [score(parse_answer(x), r["answer"])[0] for x in resps]
        out.append({**r, "responses": resps, "correct": correct, "n_fail": len(resps) - sum(correct)})
    write_jsonl(OUT / args.out, out)

    any_ok = sum(1 for o in out if any(o["correct"]))
    print(f"questions with >=1 correct rollout: {any_ok}/{len(out)}  "
          f"rollout accuracy: {sum(sum(o['correct']) for o in out) / max(1, args.k * len(out)):.3f}  "
          f"({round(time.time() - t0)} s)")


if __name__ == "__main__":
    main()
