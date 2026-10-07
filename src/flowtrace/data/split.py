"""Temporal train/val/test split for AMLworld transactions.

The split boundaries are defined in configs/split.yaml and loaded here.
Every downstream module uses `load_split` to get a filtered DataFrame.

Convention:
    - start timestamps are inclusive (ts >= start)
    - end timestamps are exclusive (ts < end)
    - this prevents a single timestamp from falling in two splits
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import polars as pl
import yaml

SplitName = Literal["train", "val", "test"]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPLIT_CONFIG_PATH = PROJECT_ROOT / "configs" / "split.yaml"


def load_split_config(path: Path = SPLIT_CONFIG_PATH) -> dict:
    """Load the split YAML and parse timestamp strings into datetime objects."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    ts_fields = [
        "exclude_before",
        "exclude_after",
        "train_start",
        "train_end",
        "val_start",
        "val_end",
        "test_start",
        "test_end",
    ]
    for field in ts_fields:
        cfg[field] = datetime.strptime(cfg[field], "%Y-%m-%d %H:%M:%S")

    return cfg


def get_bounds(split: SplitName, cfg: dict | None = None) -> tuple[datetime, datetime]:
    """Return (start_inclusive, end_exclusive) for the named split."""
    if cfg is None:
        cfg = load_split_config()
    return cfg[f"{split}_start"], cfg[f"{split}_end"]


def filter_split(df: pl.DataFrame, split: SplitName, cfg: dict | None = None) -> pl.DataFrame:
    """Filter a transactions DataFrame to only the named split.

    Args:
        df: Transactions DataFrame with a `ts` column of Datetime dtype.
        split: One of "train", "val", "test".
        cfg: Optional pre-loaded config dict (else loaded from disk).

    Returns:
        A new DataFrame containing only rows where start <= ts < end.
    """
    start, end = get_bounds(split, cfg)
    return df.filter((pl.col("ts") >= start) & (pl.col("ts") < end))