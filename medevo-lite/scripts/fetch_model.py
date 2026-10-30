"""Download a model from ModelScope into models/<name>, stdlib only.

Resumes partial files and checks each file's size against the repo listing
(downloads can end early without an error on slow links).

  python scripts/fetch_model.py Qwen/Qwen3.5-0.8B
"""
import json
import sys
import time
import urllib.parse
import urllib.request

from common import ROOT

API = "https://www.modelscope.cn/api/v1/models"


def list_files(model_id):
    with urllib.request.urlopen(f"{API}/{model_id}/repo/files?Revision=master&Recursive=true", timeout=60) as r:
        data = json.load(r)["Data"]["Files"]
    return [(f["Path"], int(f["Size"])) for f in data if f["Type"] == "blob"]


def download(model_id, path, size, dest):
    url = f"{API}/{model_id}/repo?Revision=master&FilePath={urllib.parse.quote(path)}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 11):
        have = dest.stat().st_size if dest.exists() else 0
        if have == size:
            return
        if have > size:
            dest.unlink()
            have = 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                mode = "ab" if have and r.status == 206 else "wb"
                with open(dest, mode) as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
        except Exception as e:  # network errors: resume on the next attempt
            print(f"  {path}: {e!r}"[:160])
            time.sleep(min(30, 3 * attempt))
    if dest.stat().st_size != size:
        sys.exit(f"{path}: got {dest.stat().st_size} of {size} bytes")


def main():
    model_id = sys.argv[1]
    out = ROOT / "models" / model_id.split("/")[-1]
    files = list_files(model_id)
    total = sum(s for _, s in files)
    print(f"{model_id}: {len(files)} files, {total / 1e9:.2f} GB -> {out}")
    for path, size in sorted(files, key=lambda f: f[1]):
        t0 = time.time()
        download(model_id, path, size, out / path)
        dt = time.time() - t0
        if size > 1e7:
            print(f"  {path}: {size / 1e6:.0f} MB in {dt / 60:.1f} min")
    print("done:", out)


if __name__ == "__main__":
    main()
