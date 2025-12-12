"""
Deprecated dataset builders.

This folder contains old dataset builder files that have been replaced by the
unified feature pipeline. These files are kept for backward compatibility but
should not be used for new code.

Migration guide:
    Old: from models.data.snapshot_builder import build_snapshot_for_inference
    New: from models.data.snapshot import build_snapshot_for_inference

    Old: from models.data.market_clock_dataset_builder import build_v2_dataset
    New: from models.data.dataset import build_dataset, DatasetConfig

    Old: from models.data.tod_dataset_builder import build_tod_v1_dataset
    New: from models.data.dataset import build_event_day_dataset
"""

import warnings

# Re-export for backward compatibility
from models.data.deprecated.snapshot_builder import (
    build_single_snapshot,
    build_snapshot_for_inference as _old_build_snapshot_for_inference,
)
from models.data.deprecated.market_clock_dataset_builder import (
    build_v2_dataset,
    build_v2_inference_snapshot,
    build_market_clock_snapshot_for_training,
    build_market_clock_snapshot_for_inference,
    build_market_clock_snapshot_dataset,
)
from models.data.deprecated.tod_dataset_builder import (
    build_tod_snapshot_dataset,
)


def build_snapshot_for_inference(*args, **kwargs):
    """Deprecated: Use models.data.snapshot.build_snapshot_for_inference instead."""
    warnings.warn(
        "models.data.deprecated.build_snapshot_for_inference is deprecated. "
        "Use models.data.snapshot.build_snapshot_for_inference instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _old_build_snapshot_for_inference(*args, **kwargs)


__all__ = [
    "build_single_snapshot",
    "build_snapshot_for_inference",
    "build_v2_dataset",
    "build_v2_inference_snapshot",
    "build_market_clock_snapshot_for_training",
    "build_market_clock_snapshot_for_inference",
    "build_market_clock_snapshot_dataset",
    "build_tod_snapshot_dataset",
]
