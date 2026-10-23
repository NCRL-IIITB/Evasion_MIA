"""
build_image_cache.py
====================
Build a 224x224 uint8 cache of all 112,120 NIH Chest X-ray images, read
directly from the downloaded zip archives (Victim_Model/data/images_*.zip).

Why
---
* The full extraction needs ~42 GB of free disk; the cache needs ~5.6 GB.
* Every original script decodes a 1024x1024 PNG on every access, which is what
  made one MIA run take ~3.2 h.  The cache removes that bottleneck.

Exactness
---------
Each image goes through the SAME operations as the original pipeline
(PIL open → convert("RGB") → transforms.Resize((224, 224))).  NIH X-rays are
grayscale, so one channel is stored; any image whose three channels differ is
stored separately in colour.  After building, a random sample is re-decoded
from the zips with the original Resize → ToTensor → Normalize pipeline and
compared to the cache; the build fails unless every tensor is bit-identical.

Usage
-----
  python Matched_Experiments/build_image_cache.py               # idempotent
  python Matched_Experiments/build_image_cache.py --verify 1000
"""

import argparse
import glob
import io
import json
import os
import sys
import time
import zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matched_common as mc

CHUNK = 500   # images per worker task


def index_zip_members(zip_dir: str) -> dict:
    """basename → (zip_path, member_name) for every real PNG in the archives."""
    where = {}
    for z in sorted(glob.glob(os.path.join(zip_dir, "images_*.zip"))):
        with zipfile.ZipFile(z) as zf:
            for info in zf.infolist():
                name = info.filename
                base = os.path.basename(name)
                if (not base.lower().endswith(".png") or "__MACOSX" in name
                        or base.startswith("._")):
                    continue
                where[base] = (z, name)
    return where


def _load_original(source) -> Image.Image:
    """Open one image exactly as the original code does (convert('RGB'))."""
    kind, a, b = source
    if kind == "zip":
        with zipfile.ZipFile(a) as zf, zf.open(b) as f:
            img = Image.open(io.BytesIO(f.read()))
            img.load()
    else:
        img = Image.open(a)
    return img.convert("RGB")


_RESIZE = transforms.Resize((mc.IMG_SIZE, mc.IMG_SIZE))


def _worker(task):
    out_path, items = task
    gray = np.load(out_path, mmap_mode="r+")
    colors = []
    open_zips = {}
    try:
        for idx, (kind, a, b) in items:
            if kind == "zip":
                if a not in open_zips:
                    open_zips[a] = zipfile.ZipFile(a)
                with open_zips[a].open(b) as f:
                    img = Image.open(io.BytesIO(f.read()))
                    img.load()
            else:
                img = Image.open(a)
            arr = np.asarray(_RESIZE(img.convert("RGB")))          # (224, 224, 3)
            gray[idx] = arr[..., 0]
            if not ((arr[..., 0] == arr[..., 1]).all() and (arr[..., 1] == arr[..., 2]).all()):
                colors.append((idx, arr.copy()))
        gray.flush()
    finally:
        for zf in open_zips.values():
            zf.close()
    return colors, len(items)


def main():
    p = argparse.ArgumentParser(description="Build the 224x224 NIH image cache")
    p.add_argument("--cache_dir", default=mc.DEFAULT_CACHE_DIR)
    p.add_argument("--zip_dir", default=mc.ZIP_DIR)
    p.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    p.add_argument("--verify", type=int, default=500, help="images to re-check bit-exactly")
    p.add_argument("--force", action="store_true", help="rebuild even if complete")
    args = p.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    info_path = os.path.join(args.cache_dir, "cache_info.json")
    gray_path = os.path.join(args.cache_dir, "gray_224.npy")

    manifest = pd.read_csv(mc.MANIFEST_PATH)
    keys = [os.path.basename(p) for p in manifest["path"]]
    if len(set(keys)) != len(keys):
        sys.exit("ERROR: duplicate image file names in manifest.csv")

    if os.path.exists(info_path) and not args.force:
        with open(info_path) as f:
            info = json.load(f)
        if info.get("complete") and info.get("n_images") == len(keys):
            print(f"[CACHE] Already complete: {args.cache_dir} ({len(keys)} images). Skipping.")
            return

    # ── Locate every image: zip archive first, else a file on disk ────────────
    print(f"[CACHE] Indexing zip archives in {args.zip_dir} …", flush=True)
    in_zip = index_zip_members(args.zip_dir)
    local_dirs = [os.path.join(mc.VICTIM_DIR, "images")]
    sources, missing = [], []
    for path, key in zip(manifest["path"], keys):
        if key in in_zip:
            sources.append(("zip",) + in_zip[key])
            continue
        for cand in [path] + [os.path.join(d, key) for d in local_dirs]:
            if os.path.exists(cand):
                sources.append(("file", cand, None))
                break
        else:
            missing.append(key)
    if missing:
        sys.exit(f"ERROR: {len(missing)} images not found (e.g. {missing[:3]}). "
                 "Download the NIH zips with Victim_Model/prepare_dataset.py.")
    print(f"[CACHE] All {len(keys)} images located "
          f"({sum(s[0] == 'zip' for s in sources)} in zips).", flush=True)

    # ── Build ─────────────────────────────────────────────────────────────────
    with open(info_path, "w") as f:
        json.dump({"complete": False}, f)
    np.lib.format.open_memmap(gray_path, mode="w+", dtype=np.uint8,
                              shape=(len(keys), mc.IMG_SIZE, mc.IMG_SIZE)).flush()

    # Group by archive so each task reads one zip
    order = sorted(range(len(keys)), key=lambda i: (sources[i][1], sources[i][2] or ""))
    tasks = [(gray_path, [(i, sources[i]) for i in order[s:s + CHUNK]])
             for s in range(0, len(order), CHUNK)]

    t0, done, colors = time.time(), 0, []
    with Pool(args.workers) as pool:
        for c, n in pool.imap_unordered(_worker, tasks):
            colors.extend(c)
            done += n
            if done % 10_000 < CHUNK:
                rate = done / (time.time() - t0)
                print(f"  {done:6d}/{len(keys)}  ({rate:.0f} img/s)", flush=True)

    colors.sort(key=lambda t: t[0])
    color_indices = [int(i) for i, _ in colors]
    if colors:
        np.save(os.path.join(args.cache_dir, "color_224.npy"),
                np.stack([a for _, a in colors]))
    with open(os.path.join(args.cache_dir, "keys.json"), "w") as f:
        json.dump(keys, f)
    info = {
        "complete": True,
        "n_images": len(keys),
        "shape": [len(keys), mc.IMG_SIZE, mc.IMG_SIZE],
        "color_indices": color_indices,
        "source": "NIH zips via PIL open → convert('RGB') → transforms.Resize((224,224))",
        "build_seconds": round(time.time() - t0, 1),
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"[CACHE] Built in {info['build_seconds']}s; {len(color_indices)} colour image(s).")

    # ── Verify bit-exactness against the original pipeline ───────────────────
    cache = mc.ImageCache(args.cache_dir)
    original = transforms.Compose([
        transforms.Resize((mc.IMG_SIZE, mc.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mc.IMAGENET_MEAN, mc.IMAGENET_STD),
    ])
    rng = np.random.RandomState(0)
    sample = rng.choice(len(keys), size=min(args.verify, len(keys)), replace=False)
    sample = np.unique(np.concatenate([sample, np.array(color_indices, dtype=int)]))
    bad = [int(i) for i in sample
           if not torch.equal(original(_load_original(sources[i])), cache.normalized(i))]
    if bad:
        info["complete"] = False
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        sys.exit(f"ERROR: cache mismatch for {len(bad)} image(s), e.g. {keys[bad[0]]}")
    info["verified_bit_exact"] = int(len(sample))
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"[CACHE] Verified {len(sample)} images bit-exact against the original "
          f"PIL pipeline ✓")


if __name__ == "__main__":
    main()
