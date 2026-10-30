# medevo-lite

A small-scale re-implementation of MedEvo's Data Navigation (feedback-guided data selection) on SLAKE (English). It is a re-implementation, not a reproduction. There is no GPT auditor, no trajectory repair and no error-type feedback, so the closest paper analogue is the "without error-type feedback" branch of Table 5.

## Laptop (Windows, small GPU)

The same plan on one 6 GB GPU. It uses `Qwen3.5-0.8B` (same family as the paper) and transformers + PEFT LoRA, because vLLM and ms-swift don't run on Windows.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install transformers accelerate peft pillow
.venv\Scripts\python scripts\fetch_model.py Qwen/Qwen3.5-0.8B     # ModelScope, resumable
.venv\Scripts\python scripts\00_download.py --source modelscope
.venv\Scripts\python scripts\01_build_groups.py

.venv\Scripts\python run_laptop.py --quick    # end-to-end test on a few questions (outputs_quick/, ckpt_quick/)
.venv\Scripts\python run_laptop.py            # full run; re-run after a crash, finished stages are skipped
```

- Training is `scripts/06_train_hf.py`: LoRA (rank 8, alpha 32) on the language model, vision frozen, loss on the response only, effective batch 16, lr 1e-4, 2 epochs. It saves only the adapter (`ckpt/<run>/adapter`), and `02_eval.py --adapter` loads it on top of the base model.
- Images are downscaled to at most 448 px (`MEDEVO_MAX_SIDE`), the same for eval, rollouts and training.
- Laptop numbers are a 0.8B-model result. Report them as such, not next to the paper's 9B numbers.
- Measured on an RTX 3060 Laptop (6 GB): eval runs at about 10 questions/s. Training takes 7.2 s per optimizer step (16 examples) at 4.25 GB peak, so about 23 min per C1/C2 run, about 45 min for C3, and about 3.5 h for the whole plan.
- `--quick` only checks that every stage runs. With 24 examples it makes a single real optimizer update. Adam's first step moves every LoRA weight by the full learning rate, so that one update acts like noise, and the "trained" models answer with long descriptions. This is not a bug: a 19-step probe on 300 examples gave short answers again, and test accuracy rose from 58% to 73% on 96 questions.
- Qwen3.5-0.8B usually answers with the bare word and no `Answer:` line. The parser falls back to the last line, which is correct here.
- Slow downloads (China): fetch large files with parallel range requests (see `.wheels/pget.sh`). A single connection to download.pytorch.org or ModelScope can drop to about 50 KB/s.

## Quick start (GPU machine)

```bash
conda create -n medevo python=3.10 -y && conda activate medevo
bash setup.sh                                  # install, model weights, SLAKE, 20-question smoke test
nohup bash run_pipeline.sh > logs/pipeline.log 2>&1 &   # C0 -> rollouts -> selection -> training -> tables
tail -f logs/pipeline.log
```

- `setup.sh` downloads the default model `Qwen/Qwen3.5-4B` (the paper's family; Qwen3.8 has no size under 27B). If the smoke test fails on Qwen3.5 (library support, output format), re-run with `MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct bash setup.sh`.
- Hugging Face downloads fall back to ModelScope automatically: `evalscope/SLAKE` for data and the `Qwen/...` mirrors for weights.
- `run_pipeline.sh` skips finished stages, so re-run it after a crash. `STOP_AFTER=c0|rollouts|select|seed1|train` stops early. A gate stops it if C0 val accuracy is below 10% (the parser is broken).
- Qwen3.5 runs in non-thinking mode (`enable_thinking=False` in the chat template). Before the first training run, check that ms-swift's Qwen3.5 template trains with the same empty `<think></think>` prefix. Otherwise train and eval prompts differ.

## Run order (individual scripts)

Run from the `medevo-lite/` folder. Script numbers are 00–08, and the plan's steps are 1–9 (step = script number + 1).

```bash
source env.sh                                      # written by setup.sh: BASE_MODEL = local model path
# (00_download.py falls back to ModelScope if Hugging Face is unreachable)

python scripts/00_download.py                      # step 1: SLAKE English -> data/{train,val,test}.jsonl
python scripts/01_build_groups.py                  # step 2: tag groups -> data/groups.json
python scripts/02_eval.py --model $BASE_MODEL --condition C0 --split val    # step 3
python scripts/02_eval.py --model $BASE_MODEL --condition C0 --split test
python scripts/03_gap_analysis.py --condition C0   # step 4: Table 2 -> outputs/gap_C0.json
python scripts/04_rollouts.py --model $BASE_MODEL --k 4                      # step 5
python scripts/05_select.py --B 1500               # step 6: sft_random_s{1,2,3}, sft_nav, sft_all
bash   scripts/06_train.sh                         # step 7: C1_s1-3, C2_s1-3, C3 (LoRA + merge)
bash   scripts/07_eval_all.sh                      # step 8
python scripts/08_tables.py                        # step 9: outputs/tables.md
```

Each step writes to disk and can be re-run alone. Use `--limit 20` on 02/04 for a quick check before a full run. To train one run first (e.g. the Saturday-night pair): `bash scripts/06_train.sh random_s1 1 C1_s1` and `bash scripts/06_train.sh nav 1 C2_s1`.

Optional:

```bash
# C4, round 2 with the regression term (alpha = 1), retrained from base
python scripts/03_gap_analysis.py --condition C2_s1 --prev C0
python scripts/05_select.py --conditions nav --gap gap_C2_s1 --alpha 1 --nav-name nav_r2
bash   scripts/06_train.sh nav_r2 1 C4 && bash scripts/07_eval_all.sh C4

# C5, Qwen API zero-shot (OpenAI-compatible endpoint)
export OPENAI_API_KEY=... OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
python scripts/02_eval.py --backend api --model qwen-vl-max --condition C5 --split test

# Too few correct rollouts in weak groups: re-sample those groups with k = 8 (later file wins per question)
python scripts/04_rollouts.py --model $BASE_MODEL --k 8 --only-groups content_type:KG --out rollouts_train_k8.jsonl
python scripts/05_select.py --rollouts rollouts_train.jsonl rollouts_train_k8.jsonl
```

## Decisions not spelled out in the plan

- **Back-off.** A tag value with fewer than 20 val questions is merged into `<axis>:other`. If the merged group is still under 20 (e.g. `content_type:other` = Shape, 11 questions), it uses the overall val error rate. On SLAKE this affects Shape, Chest_heart and Chest_mediastinal.
- **Budget and "questions sampled".** B counts accepted examples (one question plus one of its correct rollouts). Both selectors walk the pool and skip questions with no correct rollout, and Table 5's "questions sampled" counts every question walked. Navigation ranks by failed rollouts (most first), so it walks the fully-failed questions of weak groups first. That is the yield cost the plan asks to record. The "Correct rollouts (%)" column gives the same signal without depending on walk order.
- **Quota rounding.** Unused quota (rounding, or groups that run out) is redistributed over groups that still have candidates, in proportion to score.
- **Seeds.** C1 seed s means random data seed s plus training seed s. C2's data is fixed (tie-break seed 0), so its seeds vary training only. Expect a smaller SD for C2 and say so in the report.
- **C2 − C1 CI.** Each question's correctness is averaged over the 3 seeds of each condition, then the paired difference is bootstrapped (10,000 resamples). Per-seed pairs are reported too.
- **"Targeted" in Table 4.** A group counts as targeted if its share of navigation examples exceeds its share of usable pool questions.
- **SFT targets.** These are the model's own correct responses (STaR-style), in exactly the eval prompt format.

## Things to watch

- **Cross-axis dilution.** Quotas are spread over about 22 groups on three axes, and every question belongs to one group per axis. So navigation shifts the content-type mix less than the quota column suggests. In the dry run, a high-error group could still end up slightly under its pool share. Table 4's "targeted" column shows what actually happened. Report it rather than assume it.
- **Answer parser.** Hand-check about 50 predictions per condition (`outputs/<cond>_test.jsonl`, fields `response` / `pred` / `answer`) before trusting numbers.
- **ms-swift flags.** Check them against the installed version. `06_train.sh` assumes ms-swift 3.x (`swift sft`, `swift export --merge_lora`).
- **Hold-out.** Test predictions never feed selection or hyperparameters.

## Status

- **Smoke-tested locally** (no GPU) on real SLAKE val metadata with simulated model outputs: 01, 03, 05 (including the round-2 path) and 08, end to end. 00_download.py was tested here against ModelScope, since Hugging Face is blocked on this laptop: real counts 4,919 / 1,053 / 1,061, no missing tags. Downloads are now verified (JSON parses, zip passes testzip, size matches Content-Length), because the ModelScope connection once truncated train.json without raising an error. The 212 MB imgs.zip was too slow to finish on this connection, so image extraction has only been tested on a synthetic zip.
- **Not yet run:** 02 / 04 inference, 06 training, 07, `setup.sh` and `run_pipeline.sh`. These have a syntax check only. `setup.sh`'s smoke test is the first real check of inference.
