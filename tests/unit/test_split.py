"""Unit tests for the temporal split.

The critical property: no train row has a timestamp >= val_start, and no
val row has a timestamp >= test_start. If this ever fails, we have time
leakage and all downstream metrics are suspect.
"""

from __future__ import annotations

import pytest

from flowtrace.data.split import get_bounds, load_split_config


def test_split_boundaries_are_ordered() -> None:
    """Train must end before val starts; val must end before test starts."""
    cfg = load_split_config()
    assert cfg["train_end"] <= cfg["val_start"], "train and val overlap"
    assert cfg["val_end"] <= cfg["test_start"], "val and test overlap"


def test_splits_are_non_empty_windows() -> None:
    """Each split must span at least one day."""
    cfg = load_split_config()
    for name in ("train", "val", "test"):
        start, end = get_bounds(name, cfg)
        assert end > start, f"split {name} has non-positive duration"


def test_splits_are_inside_usable_window() -> None:
    """No split may reach into the warmup or shutdown region."""
    cfg = load_split_config()
    for name in ("train", "val", "test"):
        start, end = get_bounds(name, cfg)
        assert start >= cfg["exclude_before"], (
            f"split {name} starts before usable window"
        )
        assert end <= cfg["exclude_after"], (
            f"split {name} ends inside shutdown region"
        )


def test_no_lookahead_across_splits() -> None:
    """The leakage guard: train strictly precedes val strictly precedes test."""
    cfg = load_split_config()
    train_start, train_end = get_bounds("train", cfg)
    val_start, val_end = get_bounds("val", cfg)
    test_start, test_end = get_bounds("test", cfg)

    assert train_end <= val_start
    assert val_end <= test_start
    assert train_start < val_start < test_start
    assert train_end < test_end


if __name__ == "__main__":
    pytest.main([__file__, "-v"])