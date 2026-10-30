"""Shared paths, I/O, prompt, answer parsing and scoring.

Stdlib only, so the selection / analysis / table scripts run on any machine.
"""
import json
import os
import re
import string
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
# MEDEVO_OUT / MEDEVO_CKPT let a quick test run write somewhere else (run_laptop.py --quick).
OUT = ROOT / os.environ.get("MEDEVO_OUT", "outputs")
SFT = OUT / "sft"
CKPT = ROOT / os.environ.get("MEDEVO_CKPT", "ckpt")
LOGS = ROOT / "logs"

# Longest image side in pixels, applied identically in eval, rollouts and training.
# SLAKE images are mostly 512 px; 0 keeps the original size.
IMAGE_MAX_SIDE = int(os.environ.get("MEDEVO_MAX_SIDE", "448"))

AXES = ("modality", "location", "content_type")
MIN_GROUP = 20  # groups with fewer feedback questions back off to "<axis>:other"

# One prompt for every condition, for evaluation, rollouts and SFT targets.
PROMPT = (
    "{question}\n"
    "Think briefly about the image, then give the final answer on the last line "
    "in the form 'Answer: <answer>'. Keep the answer short (a word or phrase)."
)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_split(split):
    return read_jsonl(DATA / f"{split}.jsonl")


def image_path(row):
    """Absolute image path; rows store paths relative to the project root."""
    p = Path(row["image"])
    return str(p if p.is_absolute() else ROOT / p)


def load_image(row):
    from PIL import Image

    img = Image.open(image_path(row)).convert("RGB")
    if IMAGE_MAX_SIDE and max(img.size) > IMAGE_MAX_SIDE:
        img.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE), Image.BICUBIC)
    return img


# ---------------------------------------------------------------- scoring

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans({c: " " for c in string.punctuation})
_ANSWER = re.compile(r"answer\s*[:：]\s*(.+)", re.IGNORECASE)


def normalise(text):
    text = str(text).lower().translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def parse_answer(response):
    """Text after the last 'Answer:'; falls back to the last non-empty line."""
    matches = _ANSWER.findall(response or "")
    if matches:
        return matches[-1].strip()
    lines = [l.strip() for l in (response or "").splitlines() if l.strip()]
    return lines[-1] if lines else ""


def token_recall(pred, gold):
    gold_toks = normalise(gold).split()
    if not gold_toks:
        return 0.0
    pred_toks = set(normalise(pred).split())
    return sum(t in pred_toks for t in gold_toks) / len(gold_toks)


def score(pred, gold):
    """Returns (correct, recall). Exact match after normalisation for both
    closed and open questions; recall is reported for open questions."""
    p, g = normalise(pred), normalise(gold)
    return int(p == g), token_recall(pred, gold)


# ---------------------------------------------------------------- groups

def question_groups(row, groups):
    """Group ids of a question after back-off, one per axis."""
    out = []
    for axis in AXES:
        gid = f"{axis}:{row[axis]}"
        out.append(groups["backoff"].get(gid, gid))
    return out


def summarise(preds):
    """Closed / open / overall accuracy (%) and open token recall (%)."""
    def pct(xs):
        return round(100 * sum(xs) / len(xs), 2) if xs else None

    closed = [p["correct"] for p in preds if p["answer_type"] == "CLOSED"]
    opened = [p["correct"] for p in preds if p["answer_type"] == "OPEN"]
    recall = [p["recall"] for p in preds if p["answer_type"] == "OPEN"]
    return {
        "closed": pct(closed),
        "open": pct(opened),
        "open_recall": pct(recall),
        "overall": pct([p["correct"] for p in preds]),
        "n": len(preds),
    }
