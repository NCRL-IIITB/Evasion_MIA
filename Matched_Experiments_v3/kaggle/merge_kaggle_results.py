"""
merge_kaggle_results.py  (Matched_Experiments_v3 — lab)
=====================================================
Combine the outputs of the lab's Kaggle accounts into one results folder and
build the combined report: paper-format tables, the paired C-vs-D comparison
and the lab leaderboard (aggregate_lab.py).

Each SOURCE can be the downloaded output .zip of a Kaggle notebook version,
the unzipped folder, or the results_v3/ folder inside it.

Usage (from the repository root, on your own machine)
-----
  python Matched_Experiments_v3/kaggle/merge_kaggle_results.py \
      --out Matched_Experiments_v3/results_kaggle  account1.zip account2.zip …
  # --with_models also copies victim / shadow weights (~0.7 GB per victim)

Needs numpy, pandas, scipy, matplotlib and torch/torchvision importable
(v2_common imports them); no GPU.
"""

import argparse
import filecmp
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE   = os.path.dirname(os.path.abspath(__file__))
V3_DIR = os.path.dirname(HERE)
SUBDIRS = ("victims", "eval", "attacks", "splits", ".markers")


def find_results_dir(path):
    for root, dirs, _ in os.walk(path):
        if "victims" in dirs and os.path.basename(root) != "victims":
            return root
    return None


def merge_one(src, out, with_models, tag):
    copied = skipped = 0
    for sub in SUBDIRS:
        for root, _, files in os.walk(os.path.join(src, sub)):
            rel = os.path.relpath(root, src)
            for f in files:
                if (f.endswith((".pth", ".pt")) and not with_models) or ".tmp" in f:
                    skipped += 1
                    continue
                s, d = os.path.join(root, f), os.path.join(out, rel, f)
                os.makedirs(os.path.dirname(d), exist_ok=True)
                if os.path.exists(d) and not filecmp.cmp(s, d, shallow=False):
                    if sub == "splits":
                        sys.exit(f"ERROR: {rel}/{f} differs between sources — the accounts did "
                                 "not use the same code/split. Check their git commits.")
                    print(f"  WARNING: {rel}/{f} exists with different content — keeping both")
                    d = d + f".{tag}"
                shutil.copyfile(s, d)
                copied += 1
    logs = os.path.join(src, "logs")
    if os.path.isdir(logs):
        os.makedirs(os.path.join(out, "logs"), exist_ok=True)
        for f in os.listdir(logs):
            name = f if f.startswith(("train_", "eval_", "shadows_", "attack_")) else f"{tag}_{f}"
            shutil.copyfile(os.path.join(logs, f), os.path.join(out, "logs", name))
    for f in ("kaggle_state.json", "STATUS.md"):
        if os.path.exists(os.path.join(src, f)):
            os.makedirs(os.path.join(out, "kaggle_sources"), exist_ok=True)
            shutil.copyfile(os.path.join(src, f), os.path.join(out, "kaggle_sources", f"{tag}_{f}"))
    return copied, skipped


def main():
    p = argparse.ArgumentParser(description="Merge the v3 lab Kaggle outputs and aggregate")
    p.add_argument("sources", nargs="+")
    p.add_argument("--out", default=os.path.join(V3_DIR, "results_kaggle"))
    p.add_argument("--with_models", action="store_true")
    p.add_argument("--bootstrap", type=int, default=1000)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="kaggle_merge_v3_")
    try:
        for i, src in enumerate(args.sources):
            path = src
            if src.endswith(".zip"):
                path = os.path.join(tmp, f"src{i}")
                with zipfile.ZipFile(src) as z:
                    z.extractall(path)
            res = find_results_dir(path)
            if not res:
                sys.exit(f"ERROR: no results folder (with victims/) found in {src}")
            tag = os.path.splitext(os.path.basename(src.rstrip("/")))[0] or f"src{i}"
            copied, skipped = merge_one(res, args.out, args.with_models, tag)
            print(f"[MERGE] {src}: {copied} files copied, {skipped} model files skipped")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("[MERGE] aggregating …")
    rc = subprocess.call([sys.executable, os.path.join(V3_DIR, "aggregate_lab.py"),
                          "--results_dir", args.out, "--bootstrap", str(args.bootstrap)])
    if rc == 0:
        print(f"[MERGE] done → {os.path.join(args.out, 'summary', 'REPORT.md')}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
