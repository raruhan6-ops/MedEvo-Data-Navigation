"""Step 3: evaluate a model on a split.

Greedy decoding, fixed prompt, writes per-question predictions to
outputs/<condition>_<split>.jsonl and a summary to outputs/<condition>_<split>.summary.json.

  python scripts/02_eval.py --model Qwen/Qwen2.5-VL-3B-Instruct --condition C0 --split val
"""
import argparse
import time

from common import OUT, load_split, parse_answer, score, summarise, write_json, write_jsonl
from infer import make_backend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local (merged) checkpoint path")
    ap.add_argument("--condition", required=True, help="e.g. C0, C1_s1, C2_s1, C3")
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "transformers", "api"])
    ap.add_argument("--adapter", help="LoRA adapter dir from 06_train_hf.py (loaded on top of --model)")
    ap.add_argument("--batch-size", type=int, default=8, help="transformers backend only")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (debugging)")
    args = ap.parse_args()

    rows = load_split(args.split)
    if args.limit:
        rows = rows[:args.limit]
    backend = make_backend(args.backend, args.model, args.max_new_tokens,
                           adapter=args.adapter, batch_size=args.batch_size)

    t0 = time.time()
    responses = backend.generate(rows, n=1, temperature=0.0)
    preds = []
    for r, (resp,) in zip(rows, responses):
        pred = parse_answer(resp)
        correct, recall = score(pred, r["answer"])
        preds.append({**r, "response": resp, "pred": pred, "correct": correct, "recall": recall})

    stem = f"{args.condition}_{args.split}"
    write_jsonl(OUT / f"{stem}.jsonl", preds)
    summary = {**summarise(preds), "condition": args.condition, "split": args.split,
               "model": args.model, "adapter": args.adapter, "backend": args.backend, "seconds": round(time.time() - t0)}
    write_json(OUT / f"{stem}.summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
