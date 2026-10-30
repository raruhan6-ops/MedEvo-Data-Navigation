"""Step 7 (laptop / Windows): LoRA SFT from the base model with transformers + PEFT.

Same role as 06_train.sh (ms-swift) for machines where ms-swift is not an option.
Reads the same outputs/sft/sft_<name>.jsonl files and saves only the LoRA
adapter to ckpt/<run>/adapter; 02_eval.py --adapter loads it on top of the base.

Loss is on the assistant response only. LoRA goes on the language model's
linear layers; the vision tower is frozen. Batch size 1 with gradient
accumulation (no padding needed); same hyperparameters for every condition.

  python scripts/06_train_hf.py --model models/Qwen3.5-0.8B --data nav --seed 1 --run C2_s1
"""
import argparse
import json
import math
import random
import time

import torch

from common import CKPT, LOGS, SFT, ROOT, load_image, read_jsonl
from infer import chat_prompt

VISION_KEYS = ("visual", "vision", "image_encoder", "mm_projector", "merger")


def lora_targets(model):
    """Names (suffixes) of linear layers in the language model, excluding the head."""
    names = set()
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and not any(k in name for k in VISION_KEYS) and "lm_head" not in name:
            names.add(name.split(".")[-1])
    return sorted(names)


def encode(processor, row):
    """input_ids / labels with the prompt masked, plus the vision inputs."""
    user = row["messages"][0]["content"].replace("<image>", "", 1)
    answer = row["messages"][1]["content"]
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user}]}]
    prompt = chat_prompt(processor, msgs)
    eos = processor.tokenizer.eos_token or "<|im_end|>"
    img = load_image({"image": row["images"][0]})
    full = processor(text=[prompt + answer + eos], images=[img], return_tensors="pt")
    n_prompt = processor(text=[prompt], images=[img], return_tensors="pt")["input_ids"].shape[1]
    labels = full["input_ids"].clone()
    labels[:, :n_prompt] = -100
    full["labels"] = labels
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="dataset stem: random_s1, nav, all, nav_r2 ...")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--run", required=True, help="run / condition name, e.g. C2_s1")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=16, help="effective batch size (batch 1 per step)")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--grad-checkpointing", action="store_true", help="saves memory, ~30% slower")
    args = ap.parse_args()

    out_dir = CKPT / args.run / "adapter"
    if (out_dir / "adapter_config.json").exists():
        print(f"skip {args.run}: {out_dir} exists")
        return

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor, get_linear_schedule_with_warmup

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    targets = lora_targets(model)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=[f"{t}" for t in targets],
        exclude_modules=r".*(" + "|".join(VISION_KEYS) + r").*"))
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    rows = read_jsonl(SFT / f"sft_{args.data}.jsonl")
    steps_per_epoch = math.ceil(len(rows) / args.grad_accum)
    total_steps = math.ceil(steps_per_epoch * args.epochs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = get_linear_schedule_with_warmup(opt, max(1, int(0.03 * total_steps)), total_steps)

    LOGS.mkdir(exist_ok=True)
    log = open(LOGS / f"train_{args.run}.log", "a", encoding="utf-8")
    meta = {**vars(args), "n_examples": len(rows), "optimizer_steps": total_steps, "lora_targets": targets}
    print(json.dumps(meta), file=log, flush=True)
    print(f"{args.run}: {len(rows)} examples, {total_steps} optimizer steps, LoRA on {targets}")

    model.train()
    rng = random.Random(args.seed)
    step, micro, running, since_log, t0, skipped = 0, 0, 0.0, 0, time.time(), 0
    while step < total_steps:
        order = rows[:]
        rng.shuffle(order)
        for row in order:
            batch = encode(processor, row)
            if batch["input_ids"].shape[1] > args.max_length:
                skipped += 1
                continue
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            running += loss.item()
            micro += 1
            if micro % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            since_log += 1
            if step % 10 == 0 or step == total_steps:
                rate = (time.time() - t0) / step
                msg = (f"step {step}/{total_steps}  loss {running / since_log:.4f}  "
                       f"lr {sched.get_last_lr()[0]:.2e}  {rate:.1f}s/step  eta {(total_steps - step) * rate / 60:.0f} min")
                print(msg)
                print(msg, file=log, flush=True)
                running, since_log = 0.0, 0
            if step >= total_steps:
                break

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    done = {"run": args.run, "minutes": round((time.time() - t0) / 60, 1), "skipped_too_long": skipped,
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    print(json.dumps(done), file=log, flush=True)
    print(done, "->", out_dir.relative_to(ROOT))


if __name__ == "__main__":
    main()
