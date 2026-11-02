"""Representation analysis: how fine-tuning moves the model's hidden states.

For every test question, one forward pass over the eval prompt (image + question,
no generation) and the hidden state of the last prompt token at every layer.
Done for the base model and the seed-1 adapters, then compared:

  - linear CKA per layer against the base model (1 = same geometry up to rotation/scale)
  - drift per tag group: mean cosine distance base -> fine-tuned, last layer
  - 2-D t-SNE of the last layer, and the silhouette of content_type labels in it

  python scripts/11_representations.py extract     # GPU, ~2 min per model -> outputs/repr/<cond>.npy
  python scripts/11_representations.py analyse     # CPU -> outputs/representations.json
"""
import argparse

import numpy as np

from common import OUT, ROOT, load_image, load_split, read_jsonl, write_json
from infer import chat_prompt, user_messages

MODELS = {"C0": None, "C1_s1": "ckpt/C1_s1/adapter", "C2_s1": "ckpt/C2_s1/adapter", "C3": "ckpt/C3/adapter"}
REPR = OUT / "repr"


def extract(model_path, batch_size):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    rows = load_split("test")
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"  # last position = last prompt token for every row
    REPR.mkdir(parents=True, exist_ok=True)
    for cond, adapter in MODELS.items():
        out = REPR / f"{cond}.npy"
        if out.exists():
            print(f"skip {cond}: {out.name} exists")
            continue
        model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, ROOT / adapter).merge_and_unload()
        model.eval()
        states = []
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            texts = [chat_prompt(processor, user_messages(r)) for r in batch]
            enc = processor(text=texts, images=[load_image(r) for r in batch], return_tensors="pt",
                            padding=True).to(model.device)
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states  # embeddings + one per layer
            states.append(torch.stack([h[:, -1] for h in hs], 1).float().cpu().numpy())
            print(f"  {cond} {min(i + batch_size, len(rows))}/{len(rows)}", end="\r")
        print()
        np.save(out, np.concatenate(states).astype(np.float16))  # (questions, layers + 1, hidden)
        del model
        torch.cuda.empty_cache()


def linear_cka(x, y):
    x = x - x.mean(0)
    y = y - y.mean(0)
    xty = np.linalg.norm(x.T @ y) ** 2
    return float(xty / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)))


def analyse(perplexity):
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    rows = load_split("test")
    reps = {c: np.load(REPR / f"{c}.npy").astype(np.float64) for c in MODELS if (REPR / f"{c}.npy").exists()}
    base = reps["C0"]
    n_layers = base.shape[1]
    res = {"n_questions": len(rows), "layers": n_layers - 1, "token": "last prompt token", "models": list(reps)}

    res["cka_vs_C0"] = {c: [round(linear_cka(base[:, l], r[:, l]), 4) for l in range(n_layers)]
                        for c, r in reps.items() if c != "C0"}
    if "C1_s1" in reps and "C2_s1" in reps:
        res["cka_C1_vs_C2"] = [round(linear_cka(reps["C1_s1"][:, l], reps["C2_s1"][:, l]), 4) for l in range(n_layers)]

    def unit(v):
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    last = {c: r[:, -1] for c, r in reps.items()}
    groups = {}
    for i, r in enumerate(rows):
        for axis in ("modality", "location", "content_type"):
            groups.setdefault(f"{axis}:{r[axis]}", []).append(i)
    drift = {}
    for c, x in last.items():
        if c == "C0":
            continue
        d = 1 - (unit(x) * unit(last["C0"])).sum(1)
        drift[c] = {"all": round(float(d.mean()), 4),
                    **{g: round(float(d[idx].mean()), 4) for g, idx in sorted(groups.items()) if len(idx) >= 20}}
    res["drift_cosine_last_layer"] = drift

    labels = np.array([r["content_type"] for r in rows])
    keep = np.isin(labels, [g.split(":")[1] for g, idx in groups.items() if g.startswith("content_type:") and len(idx) >= 20])
    res["silhouette_content_type"] = {}
    res["tsne"] = {"qid": [r["qid"] for r in rows], "content_type": labels.tolist(),
                   "correct": {}, "xy": {}}
    for c, x in last.items():
        z = unit(x)
        res["silhouette_content_type"][c] = round(float(silhouette_score(z[keep], labels[keep], metric="cosine")), 4)
        xy = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=0, metric="cosine").fit_transform(z)
        res["tsne"]["xy"][c] = np.round(xy, 2).tolist()
        preds = OUT / f"{c}_test.jsonl"
        if preds.exists():
            by_q = {p["qid"]: p["correct"] for p in read_jsonl(preds)}
            res["tsne"]["correct"][c] = [by_q[r["qid"]] for r in rows]
    write_json(OUT / "representations.json", res)
    print("CKA last layer vs C0:", {c: v[-1] for c, v in res["cka_vs_C0"].items()})
    print("drift (all):", {c: v["all"] for c, v in drift.items()})
    print("silhouette content_type:", res["silhouette_content_type"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "analyse"])
    ap.add_argument("--model", default="models/Qwen3.5-0.8B")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--perplexity", type=float, default=30)
    args = ap.parse_args()
    if args.stage == "extract":
        extract(args.model, args.batch_size)
    else:
        analyse(args.perplexity)


if __name__ == "__main__":
    main()
