"""
merge_kaggle_results.py
=======================
Combine the outputs of the Kaggle runs (one cell per account) into one
results folder and build the combined report (tables, paired contrasts,
gap-matched analysis).

Each SOURCE can be the downloaded output .zip of a Kaggle notebook version,
the unzipped folder, or the `results/` folder inside it.  Run names are
unique per cell and seed, so nothing collides; shared files (the attack split
of a seed, pipeline logs) are kept once / prefixed per source.

Usage
-----
  python Matched_Experiments/kaggle/merge_kaggle_results.py \
      --out Matched_Experiments/results_kaggle  std_reg.zip fgsm_reg.zip std_noreg.zip fgsm_noreg.zip
  # add --with_models to also copy the .pth weights (~0.3 GB per cell)
"""

import argparse
import filecmp
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE        = os.path.dirname(os.path.abspath(__file__))
MATCHED_DIR = os.path.dirname(HERE)
SUBDIRS     = ("victims", "eval", "attacks", "splits", ".markers")


def find_results_dir(path):
    """Return the folder that directly contains victims/ (and kaggle_state.json)."""
    for root, dirs, _ in os.walk(path):
        if "victims" in dirs and os.path.basename(root) != "victims":
            return root
    return None


def merge_one(src, out, with_models, tag):
    copied = skipped = 0
    for sub in SUBDIRS:
        base = os.path.join(src, sub)
        for root, _, files in os.walk(base):
            if "_state" in root.split(os.sep):        # unfinished-attack resume state
                continue
            rel = os.path.relpath(root, src)
            for f in files:
                if (f.endswith((".pth", ".pt")) and not with_models) or ".tmp" in f:
                    skipped += 1
                    continue
                s, d = os.path.join(root, f), os.path.join(out, rel, f)
                os.makedirs(os.path.dirname(d), exist_ok=True)
                if os.path.exists(d) and not filecmp.cmp(s, d, shallow=False):
                    if sub == "splits":
                        sys.exit(f"ERROR: {rel}/{f} differs between sources — the accounts "
                                 "did not use the same code/split. Check their git commits.")
                    print(f"  WARNING: {rel}/{f} exists with different content — keeping both")
                    d = d + f".{tag}"
                shutil.copyfile(s, d)
                copied += 1
    logs = os.path.join(src, "logs")
    if os.path.isdir(logs):
        os.makedirs(os.path.join(out, "logs"), exist_ok=True)
        for f in os.listdir(logs):
            name = f if f.startswith(("train_", "afmia_", "shadow_mia_", "eval_")) else f"{tag}_{f}"
            shutil.copyfile(os.path.join(logs, f), os.path.join(out, "logs", name))
    for f in ("kaggle_state.json", "STATUS.md"):
        if os.path.exists(os.path.join(src, f)):
            os.makedirs(os.path.join(out, "kaggle_sources"), exist_ok=True)
            shutil.copyfile(os.path.join(src, f), os.path.join(out, "kaggle_sources", f"{tag}_{f}"))
    return copied, skipped


def main():
    p = argparse.ArgumentParser(description="Merge Kaggle outputs and aggregate")
    p.add_argument("sources", nargs="+")
    p.add_argument("--out", default=os.path.join(MATCHED_DIR, "results_kaggle"))
    p.add_argument("--with_models", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="kaggle_merge_")
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
    rc = subprocess.call([sys.executable, os.path.join(MATCHED_DIR, "aggregate_results.py"),
                          "--results_dir", args.out])
    if rc == 0:
        print(f"[MERGE] done → {os.path.join(args.out, 'summary', 'REPORT.md')}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
