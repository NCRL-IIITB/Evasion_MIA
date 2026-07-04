"""
prepare_dataset.py
==================
Download the FULL NIH Chest X-ray dataset (all 12 zip files), extract all
images into a common images/ folder, and build a manifest CSV for the MIA
pipeline.

The member/non-member split is performed at the PATIENT level: every image
from a given patient goes entirely into the member or non-member partition.
Default ratio is 70 % members / 30 % non-members.  This mirrors a realistic
MIA scenario while preventing patient-level data leakage between splits.

Outputs
-------
  images/              ← extracted PNGs from all 12 zip files
  manifest.csv         ← columns: path, label, label_idx, split
  patient_splits.json  ← patient-level split mapping for reproducibility

Usage
-----
  python prepare_dataset.py                            # default: 70/30 patient-level
  python prepare_dataset.py --split-ratio 0.5          # 50/50 patient-level
  python prepare_dataset.py --no-split-by-patient      # legacy image-level split

The script is idempotent: already-downloaded zips and already-extracted
images are skipped automatically.
"""

import os
import sys
import time
import zipfile
import shutil
import argparse
import json as _json
from collections import Counter

import requests
import numpy as np
import pandas as pd

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
# Place data artefacts next to the script in Victim_Model/
DATA_DIR    = os.path.join(BASE_DIR, "data")
IMAGES_DIR  = os.path.join(BASE_DIR, "images")
# Data_Entry_2017.csv is expected at the project root (one level up)
LABEL_CSV   = os.path.join(os.path.dirname(BASE_DIR), "Data_Entry_2017.csv")
MANIFEST    = os.path.join(BASE_DIR, "manifest.csv")
PATIENT_SPLITS_JSON = os.path.join(BASE_DIR, "patient_splits.json")

# HuggingFace mirror for the NIH Chest X-ray dataset
HF_BASE = (
    "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset"
    "/resolve/main/data"
)

# All 12 zip files (images_001 … images_012)
ZIP_NAMES = [f"images_{i:03d}.zip" for i in range(1, 13)]
ZIP_URLS  = [f"{HF_BASE}/images/{name}" for name in ZIP_NAMES]

# 15 disease classes (order matches the training scripts)
DISEASE_CLASSES = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
    "Emphysema",   "Fibrosis",       "Effusion",     "Pneumonia",    "Pleural_Thickening",
    "Cardiomegaly","Nodule",          "Mass",         "Hernia",       "No Finding",
]
NUM_CLASSES  = len(DISEASE_CLASSES)   # 15

MEMBER_RATIO = 0.70   # 70 % members, 30 % non-members (patient-level)
RANDOM_SEED  = 42


# ─── Helpers ──────────────────────────────────────────────────────────────────

# Download constants
CHUNK_SIZE  = 8 * 1024 * 1024   # 8 MB write chunks
MAX_RETRIES = 5                  # attempts before giving up
BACKOFF_BASE = 5                 # seconds — wait = BACKOFF_BASE * 2^attempt


def _download(url: str, dest: str):
    """Download *url* to *dest* with resume support and retry on failure.

    - If *dest* already exists AND its size matches Content-Length, skips.
    - If *dest* is a partial file (previous interrupted download), resumes
      from its current byte offset using an HTTP Range request.
    - On connection errors, retries up to MAX_RETRIES times with exponential
      backoff before raising.
    """
    for attempt in range(MAX_RETRIES):
        try:
            # -- Check how many bytes we already have (resume offset) ----------
            existing_bytes = os.path.getsize(dest) if os.path.exists(dest) else 0
            headers = {}
            if existing_bytes > 0:
                headers["Range"] = f"bytes={existing_bytes}-"

            resp = requests.get(url, headers=headers, stream=True, timeout=60)

            # 416 = Range Not Satisfiable → file already fully downloaded
            if resp.status_code == 416:
                print(
                    f"  Already fully downloaded: {os.path.basename(dest)} "
                    f"({existing_bytes / 1_048_576:.1f} MB) — skipping."
                )
                return

            resp.raise_for_status()

            # -- Determine total expected size ---------------------------------
            content_length = int(resp.headers.get("Content-Length", 0))
            total_bytes    = existing_bytes + content_length

            # If server returned 200 (doesn't support Range), restart fully
            if resp.status_code == 200 and existing_bytes > 0:
                print(
                    f"  Server does not support resume — restarting download of "
                    f"{os.path.basename(dest)}."
                )
                existing_bytes = 0

            # If file is already complete, skip
            if existing_bytes > 0 and content_length == 0:
                print(
                    f"  Already present: {os.path.basename(dest)} "
                    f"({existing_bytes / 1_048_576:.1f} MB) — skipping."
                )
                return

            # -- Open file in append (resume) or write (fresh) mode -----------
            mode = "ab" if existing_bytes > 0 else "wb"
            verb = f"Resuming from {existing_bytes / 1_048_576:.1f} MB" \
                   if existing_bytes > 0 else "Downloading"
            print(f"  {verb}: {url}")
            print(f"      -> {dest}")
            if total_bytes:
                print(
                    f"     Total: {total_bytes / 1_048_576:.1f} MB",
                    flush=True,
                )

            downloaded = existing_bytes
            last_pct   = -1

            with open(dest, mode) as fh:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes:
                        pct = min(100, int(downloaded * 100 / total_bytes))
                        if pct != last_pct and pct % 5 == 0:
                            print(
                                f"    {pct:3d}%  "
                                f"({downloaded / 1_048_576:7.1f} / "
                                f"{total_bytes / 1_048_576:7.1f} MB)",
                                flush=True,
                            )
                            last_pct = pct

            # -- Verify final size --------------------------------------------
            final_size = os.path.getsize(dest)
            if total_bytes and final_size < total_bytes:
                raise IOError(
                    f"Incomplete download: expected {total_bytes} bytes, "
                    f"got {final_size} bytes."
                )

            print(
                f"  Done: {final_size / 1_048_576:.1f} MB — "
                f"{os.path.basename(dest)}",
                flush=True,
            )
            return   # success

        except (requests.RequestException, IOError, OSError) as exc:
            wait = BACKOFF_BASE * (2 ** attempt)
            if attempt + 1 < MAX_RETRIES:
                print(
                    f"  WARNING: Download error on attempt {attempt + 1}/{MAX_RETRIES}: {exc}"
                    f"  Retrying in {wait}s …",
                    flush=True,
                )
                time.sleep(wait)
            else:
                print(
                    f"  ERROR: Download failed after {MAX_RETRIES} attempts: {exc}",
                    flush=True,
                )
                raise


def _extract_zip(zip_path: str, images_dir: str):
    """Extract PNG files from *zip_path* into *images_dir* (flat layout)."""
    os.makedirs(images_dir, exist_ok=True)
    print(f"  Extracting {os.path.basename(zip_path)} -> {images_dir}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        png_members = [m for m in zf.namelist() if m.lower().endswith(".png")]
        print(f"    {len(png_members)} PNG files in archive.")

        for i, member in enumerate(png_members):
            target_name = os.path.basename(member)
            if not target_name:
                continue
            target_path = os.path.join(images_dir, target_name)
            if os.path.exists(target_path):
                continue          # already extracted
            with zf.open(member) as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            if (i + 1) % 2000 == 0:
                print(f"    extracted {i + 1}/{len(png_members)}", flush=True)

    total = len([f for f in os.listdir(images_dir) if f.lower().endswith(".png")])
    print(f"  Done. Total PNGs in {images_dir}: {total}", flush=True)


def _multi_hot(finding_str: str, label_to_idx: dict) -> list:
    """Convert a pipe-separated finding string into a 15-dim multi-hot list."""
    vec = [0] * NUM_CLASSES
    for tag in str(finding_str).split("|"):
        tag = tag.strip()
        if tag in label_to_idx:
            vec[label_to_idx[tag]] = 1
    return vec


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download NIH Chest X-ray dataset and build manifest.csv"
    )
    parser.add_argument("--data_dir",   default=DATA_DIR,   help="Directory for zip files")
    parser.add_argument("--images_dir", default=IMAGES_DIR, help="Directory for extracted images")
    parser.add_argument("--label_csv",  default=LABEL_CSV,  help="Path to Data_Entry_2017.csv")
    parser.add_argument("--manifest",   default=MANIFEST,   help="Output manifest CSV path")
    parser.add_argument(
        "--zips", type=int, default=12,
        help="Number of zip files to download (1-12). Default: 12 (full dataset)"
    )
    parser.add_argument(
        "--split-ratio", type=float, default=MEMBER_RATIO,
        help=f"Fraction of patients (or images) assigned to the member split. "
             f"Default: {MEMBER_RATIO}"
    )
    parser.add_argument(
        "--split-by-patient", action="store_true", default=True,
        help="Split at the PATIENT level so all images of a patient stay together "
             "(default: True)"
    )
    parser.add_argument(
        "--no-split-by-patient", dest="split_by_patient", action="store_false",
        help="Legacy: split at the IMAGE level (no patient grouping)"
    )
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    # ── 1. Download zip files ─────────────────────────────────────────────────
    n_zips = min(args.zips, 12)
    print(f"\n[1/4] Downloading {n_zips} zip file(s) from HuggingFace …")
    zip_paths = []
    for name, url in zip(ZIP_NAMES[:n_zips], ZIP_URLS[:n_zips]):
        dest = os.path.join(args.data_dir, name)
        _download(url, dest)
        zip_paths.append(dest)

    # ── 2. Extract images ─────────────────────────────────────────────────────
    existing_count = (
        len([f for f in os.listdir(args.images_dir) if f.lower().endswith(".png")])
        if os.path.exists(args.images_dir)
        else 0
    )

    # Estimate expected count (each zip has ~9,000-10,000 images)
    expected_min = n_zips * 8_000
    print(f"\n[2/4] Extracting images (already have {existing_count}) …")
    if existing_count >= expected_min:
        print(f"  Looks like images already extracted ({existing_count} PNGs) — skipping.")
    else:
        for zp in zip_paths:
            _extract_zip(zp, args.images_dir)

    # ── 3. Build multi-label manifest ─────────────────────────────────────────
    print(f"\n[3/4] Building manifest from {args.label_csv} …")

    if not os.path.exists(args.label_csv):
        print(f"  ERROR: Label CSV not found at {args.label_csv}")
        print("  Please place Data_Entry_2017.csv in the project root directory.")
        sys.exit(1)

    df = pd.read_csv(args.label_csv)
    print(f"  Label CSV rows: {len(df)}")

    # Filter to images that were actually extracted
    available = set(os.listdir(args.images_dir))
    df = df[df["Image Index"].isin(available)].copy()
    print(f"  Rows matching extracted images: {len(df)}")

    if len(df) == 0:
        print("  ERROR: No matching images found.")
        sys.exit(1)

    # Build multi-hot label vectors
    label_to_idx = {name: i for i, name in enumerate(DISEASE_CLASSES)}

    df["label_idx"] = df["Finding Labels"].apply(
        lambda s: str(_multi_hot(s, label_to_idx))
    )
    df["label"] = df["Finding Labels"]

    # Class distribution
    label_counter = Counter()
    for finding in df["Finding Labels"]:
        for tag in str(finding).split("|"):
            tag = tag.strip()
            if tag in label_to_idx:
                label_counter[tag] += 1

    print(f"\n  Disease distribution ({NUM_CLASSES} classes):")
    for name in DISEASE_CLASSES:
        print(f"    {name:25s}  {label_counter.get(name, 0):7d}")

    # ── 4. Member / non-member split ──────────────────────────────────────────
    split_ratio = args.split_ratio

    if args.split_by_patient:
        # ── Patient-level split ───────────────────────────────────────────
        print(f"\n[4/4] PATIENT-LEVEL split: {split_ratio*100:.0f}% member / "
              f"{(1-split_ratio)*100:.0f}% non-member (seed={RANDOM_SEED}) …")

        if "Patient ID" not in df.columns:
            print("  ERROR: 'Patient ID' column not found in Data_Entry_2017.csv.")
            print("  Cannot perform patient-level split. Use --no-split-by-patient.")
            sys.exit(1)

        # Get unique patients, shuffle, then split
        unique_patients = df["Patient ID"].unique()
        rng = np.random.RandomState(RANDOM_SEED)
        rng.shuffle(unique_patients)

        n_train_patients = int(len(unique_patients) * split_ratio)
        member_patients    = set(unique_patients[:n_train_patients])
        nonmember_patients = set(unique_patients[n_train_patients:])

        df["split"] = df["Patient ID"].apply(
            lambda pid: "member" if pid in member_patients else "nonmember"
        )

        member_df    = df[df["split"] == "member"].copy()
        nonmember_df = df[df["split"] == "nonmember"].copy()

        print(f"  Total unique patients:     {len(unique_patients)}")
        print(f"  Member patients:           {n_train_patients}")
        print(f"  Non-member patients:       {len(unique_patients) - n_train_patients}")
        print(f"  Member images:             {len(member_df)}")
        print(f"  Non-member images:         {len(nonmember_df)}")
        print(f"  Actual image ratio:        "
              f"{len(member_df)/len(df)*100:.1f}% / {len(nonmember_df)/len(df)*100:.1f}%")

        # ── Save patient split mapping for reproducibility ────────────────
        patient_split_info = {
            "split_method": "patient_level",
            "random_state": RANDOM_SEED,
            "train_ratio": split_ratio,
            "total_patients": int(len(unique_patients)),
            "member_patients_count": int(n_train_patients),
            "nonmember_patients_count": int(len(unique_patients) - n_train_patients),
            "member_images_count": int(len(member_df)),
            "nonmember_images_count": int(len(nonmember_df)),
            "member_patient_ids": sorted([int(p) for p in member_patients]),
            "nonmember_patient_ids": sorted([int(p) for p in nonmember_patients]),
        }
        splits_path = os.path.join(os.path.dirname(args.manifest), "patient_splits.json")
        with open(splits_path, "w") as fh:
            _json.dump(patient_split_info, fh, indent=2)
        print(f"  Patient split saved to:    {splits_path}")

        # Verify: no patient appears in both splits
        overlap = member_patients & nonmember_patients
        assert len(overlap) == 0, (
            f"BUG: {len(overlap)} patients appear in both splits!"
        )
        print(f"  ✓ Patient-level split verified — zero overlap.")

    else:
        # ── Legacy: random image-level split ──────────────────────────────
        print(f"\n[4/4] IMAGE-LEVEL split: {split_ratio*100:.0f}% member / "
              f"{(1-split_ratio)*100:.0f}% non-member (seed={RANDOM_SEED}) …")
        df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

        n_total   = len(df)
        n_members = int(n_total * split_ratio)

        member_df    = df.iloc[:n_members].copy()
        nonmember_df = df.iloc[n_members:].copy()

        member_df["split"]    = "member"
        nonmember_df["split"] = "nonmember"

        print(f"  Total images matched: {n_total}")
        print(f"  Members:              {len(member_df)}")
        print(f"  Non-members:          {len(nonmember_df)}")

    # Build manifest
    manifest = pd.concat([member_df, nonmember_df], ignore_index=True)
    manifest["path"] = manifest["Image Index"].apply(
        lambda fn: os.path.join(args.images_dir, fn)
    )
    manifest = manifest[["path", "label", "label_idx", "split"]]
    manifest.to_csv(args.manifest, index=False)

    print(f"\n[DONE] Manifest written to: {args.manifest}")
    print(f"  Total rows: {len(manifest)}")
    print(f"  Columns:    {list(manifest.columns)}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
