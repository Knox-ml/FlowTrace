"""Canonical schema for AMLworld transaction data.

Converts raw Kaggle CSV columns to clean snake_case names used by the rest
of the pipeline. Also provides a Pandera validator for data-quality checks.
"""

from __future__ import annotations

from pathlib import Path

import pandera.polars as pa
import polars as pl
from pandera.polars import Column

# ---------------------------------------------------------------------------
# Canonical column names
# ---------------------------------------------------------------------------
# The raw CSV has quirks (duplicate "Account" columns, spaces, capitalization).
# We rename to snake_case once at load time and never touch the raw names again.
RAW_TO_CANONICAL: dict[str, str] = {
    "Timestamp": "ts",
    "From Bank": "from_bank",
    "Account": "from_account",
    "To Bank": "to_bank",
    "Account_duplicated_0": "to_account",
    "Amount Received": "amount_received",
    "Receiving Currency": "currency_received",
    "Amount Paid": "amount_paid",
    "Payment Currency": "currency_paid",
    "Payment Format": "payment_format",
    "Is Laundering": "is_laundering",
}

CANONICAL_COLUMNS: list[str] = list(RAW_TO_CANONICAL.values())


# ---------------------------------------------------------------------------
# Pandera schema — data-quality contract
# ---------------------------------------------------------------------------
TRANSACTIONS_SCHEMA = pa.DataFrameSchema(
    {
        "ts": Column(pl.Datetime, nullable=False),
        "from_bank": Column(pl.Int64, nullable=False, checks=pa.Check.ge(0)),
        "from_account": Column(pl.Utf8, nullable=False),
        "to_bank": Column(pl.Int64, nullable=False, checks=pa.Check.ge(0)),
        "to_account": Column(pl.Utf8, nullable=False),
        "amount_received": Column(pl.Float64, nullable=False, checks=pa.Check.ge(0.0)),
        "currency_received": Column(pl.Utf8, nullable=False),
        "amount_paid": Column(pl.Float64, nullable=False, checks=pa.Check.ge(0.0)),
        "currency_paid": Column(pl.Utf8, nullable=False),
        "payment_format": Column(pl.Utf8, nullable=False),
        "is_laundering": Column(pl.Int64, nullable=False, checks=pa.Check.isin([0, 1])),
    },
    strict=True,
    coerce=False,
)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_transactions(csv_path: str | Path, validate: bool = False) -> pl.DataFrame:
    """Load a raw AMLworld transactions CSV into the canonical schema.

    Args:
        csv_path: Path to the raw CSV (e.g. data/raw/HI-Small_Trans.csv).
        validate: If True, run the Pandera schema check. Slow on 5M rows;
            default False for routine loading, True in tests.

    Returns:
        A Polars DataFrame with canonical snake_case columns and a parsed
        timestamp. Row count equals the CSV row count.
    """
    df = pl.read_csv(csv_path)

    missing = set(RAW_TO_CANONICAL) - set(df.columns)
    if missing:
        raise ValueError(
            f"Raw CSV missing expected columns: {sorted(missing)}. "
            f"Found: {df.columns}"
        )

    df = df.rename(RAW_TO_CANONICAL)
    df = df.with_columns(
        pl.col("ts").str.strptime(pl.Datetime, format="%Y/%m/%d %H:%M", strict=True)
    )

    if validate:
        TRANSACTIONS_SCHEMA.validate(df, lazy=True)

    return df


def scan_transactions(csv_path: str | Path) -> pl.LazyFrame:
    """Lazy variant of load_transactions for streaming over big files."""
    lf = pl.scan_csv(csv_path).rename(RAW_TO_CANONICAL)
    lf = lf.with_columns(
        pl.col("ts").str.strptime(pl.Datetime, format="%Y/%m/%d %H:%M", strict=True)
    )
    return lf