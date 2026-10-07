"""
Download the IBM AMLworld dataset from Kaggle.

Keeps only the Small and Medium files (HI and LI), deletes Large and the zip.
Idempotent — safe to re-run; skips download if files already present.

Usage (from project root):
    uv run python scripts/download_data.py
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

DATASET_REF = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Keep: both Smalls (fast iteration) + both Mediums (scaling experiments).
# Skip: HI-Large and LI-Large (180M rows each, won't fit in 8GB RAM).
KEEP_FILES = [
    "HI-Small_Trans.csv",
    "HI-Small_Patterns.txt",
    "LI-Small_Trans.csv",
    "LI-Small_Patterns.txt",
    "HI-Medium_Trans.csv",
    "HI-Medium_Patterns.txt",
    "LI-Medium_Trans.csv",
    "LI-Medium_Patterns.txt",
]


def already_downloaded() -> bool:
    return all((RAW_DIR / name).exists() for name in KEEP_FILES)


def download_and_extract() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET_REF} to {RAW_DIR} ...")
    print("Zip is ~8 GB. Grab a chai, this takes 20–40 min on home broadband.")

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        DATASET_REF,
        path=str(RAW_DIR),
        unzip=False,
        quiet=False,
    )

    zip_name = DATASET_REF.split("/")[-1] + ".zip"
    zip_path = RAW_DIR / zip_name
    if not zip_path.exists():
        print(f"ERROR: expected {zip_path} after download, not found.", file=sys.stderr)
        sys.exit(1)

    print(f"\nExtracting selected files from {zip_name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        all_members = zf.namelist()
        members_to_extract = [m for m in all_members if Path(m).name in KEEP_FILES]
        skipped = [m for m in all_members if Path(m).name not in KEEP_FILES]

        if not members_to_extract:
            print("ERROR: no target files found inside the zip.", file=sys.stderr)
            print("Zip contents were:", all_members, file=sys.stderr)
            sys.exit(1)

        print(f"  Extracting {len(members_to_extract)} files, skipping {len(skipped)}.")
        for m in members_to_extract:
            print(f"  extracting {m}")
            zf.extract(m, path=RAW_DIR)
            extracted = RAW_DIR / m
            if extracted.parent != RAW_DIR:
                target = RAW_DIR / Path(m).name
                if target.exists():
                    target.unlink()
                shutil.move(str(extracted), target)

    # Clean up any empty subdirectories the extraction created.
    for sub in RAW_DIR.iterdir():
        if sub.is_dir():
            shutil.rmtree(sub)

    print(f"\nDeleting zip {zip_path} to free ~8 GB disk space ...")
    zip_path.unlink()

    print("\nDone. Files in data/raw/:")
    total_mb = 0.0
    for f in sorted(RAW_DIR.iterdir()):
        if f.name == ".gitkeep":
            continue
        size_mb = f.stat().st_size / (1024 * 1024)
        total_mb += size_mb
        print(f"  {f.name:<30s}  {size_mb:>10,.1f} MB")
    print(f"  {'TOTAL':<30s}  {total_mb:>10,.1f} MB")


def main() -> None:
    if already_downloaded():
        print("All 8 target files already present. Skipping download.")
        print("Delete files in data/raw/ to force re-download.")
        return
    download_and_extract()


if __name__ == "__main__":
    main()