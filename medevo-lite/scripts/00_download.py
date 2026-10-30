"""Step 1: download SLAKE, keep English rows, write data/{train,val,test}.jsonl.

Source: Hugging Face dataset BoKelvin/SLAKE (train/validation/test.json + imgs.zip),
falling back to the ModelScope mirror evalscope/SLAKE (--source auto, the default).
Set HF_ENDPOINT=https://hf-mirror.com if huggingface.co is slow; use
--source modelscope to skip Hugging Face entirely.
Stdlib only.
"""
import argparse
import collections
import http.client
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile

from common import AXES, DATA, ROOT, read_json, write_jsonl

REPO = "BoKelvin/SLAKE"
MS_REPO = "evalscope/SLAKE"  # ModelScope mirror with the same files
SPLITS = {"train": "train.json", "val": "validation.json", "test": "test.json"}


def source_url(source, name):
    if source == "modelscope":
        return f"https://www.modelscope.cn/api/v1/datasets/{MS_REPO}/repo?Revision=master&FilePath={name}"
    base = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return f"{base}/datasets/{REPO}/resolve/main/{name}"


def verify(path):
    """Raise ValueError if a downloaded file is truncated or corrupt. Some
    servers close the connection early without an error, so a finished
    download is not proof of a complete file."""
    if path.suffix == ".json" or path.name.endswith(".json.part"):
        with open(path, encoding="utf-8") as f:
            json.load(f)
    elif path.suffix == ".zip" or path.name.endswith(".zip.part"):
        try:
            with zipfile.ZipFile(path) as z:
                bad = z.testzip()
        except zipfile.BadZipFile as e:
            raise ValueError(f"bad zip: {e}") from e
        if bad:
            raise ValueError(f"corrupt member in zip: {bad}")


def fetch(name, dest, source):
    """source: hf, modelscope, or auto (try hf, fall back to modelscope)."""
    if dest.exists():
        try:
            verify(dest)
            return
        except (ValueError, OSError) as e:
            print(f"{dest.name} on disk is incomplete ({e}); downloading again")
            dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for src in (["hf", "modelscope"] if source == "auto" else [source]):
        url = source_url(src, name)
        for attempt in range(1, 4):
            print(f"downloading {url}" + (f" (attempt {attempt})" if attempt > 1 else ""))
            try:
                with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                    expected = r.headers.get("Content-Length")
                    shutil.copyfileobj(r, f)
                got = tmp.stat().st_size
                if expected and int(expected) != got:
                    raise ValueError(f"got {got} of {expected} bytes")
                verify(tmp)
                tmp.replace(dest)
                return
            except (OSError, http.client.HTTPException, ValueError) as e:
                print(f"  failed: {e!r}"[:200])
                if isinstance(e, urllib.error.URLError) and "SSL" in str(e):
                    break  # blocked host; retrying will not help
                time.sleep(3 * attempt)
    sys.exit(f"could not download {name}")


def extract_images(zip_path, out_dir):
    """Extract imgs.zip so that out_dir/xmlab<N>/source.jpg exists."""
    if (out_dir / "xmlab0").exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            # strip a leading "imgs/" folder if the archive has one
            parts = [p for p in m.filename.split("/") if p]
            if parts and parts[0] == "imgs":
                parts = parts[1:]
            if not parts or m.is_dir() or ".." in parts or parts[0].startswith("__MACOSX"):
                continue
            target = out_dir.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-images", action="store_true", help="skip imgs.zip (for testing the tag pipeline)")
    ap.add_argument("--source", default="auto", choices=["auto", "hf", "modelscope"])
    args = ap.parse_args()

    raw = DATA / "raw"
    for name in SPLITS.values():
        fetch(name, raw / name, args.source)
    img_dir = DATA / "imgs"
    if not args.no_images:
        fetch("imgs.zip", raw / "imgs.zip", args.source)
        extract_images(raw / "imgs.zip", img_dir)

    ok = True
    for split, name in SPLITS.items():
        rows = [r for r in read_json(raw / name) if r.get("q_lang") == "en"]
        out = []
        for r in rows:
            out.append({
                "qid": r["qid"],
                "image": (img_dir / r["img_name"]).relative_to(ROOT).as_posix(),
                "question": r["question"],
                "answer": str(r["answer"]),
                "answer_type": r["answer_type"].upper(),
                "modality": r["modality"],
                "location": r["location"],
                "content_type": r["content_type"],
            })
        write_jsonl(DATA / f"{split}.jsonl", out)

        # Check: row counts and tag value counts; stop if anything is missing.
        print(f"\n== {split}: {len(out)} English questions")
        if not out:
            ok = False
        for field in AXES + ("answer_type",):
            counts = collections.Counter(r[field] for r in out)
            missing = sum(1 for r in out if not r[field])
            print(f"  {field}: {dict(counts.most_common())}" + (f"  MISSING={missing}" if missing else ""))
            ok &= missing == 0
        if not args.no_images:
            absent = sum(1 for r in out if not (ROOT / r["image"]).exists())
            print(f"  images missing on disk: {absent}")
            ok &= absent == 0

    if not ok:
        sys.exit("check failed: empty split, missing tags or missing images")


if __name__ == "__main__":
    main()
