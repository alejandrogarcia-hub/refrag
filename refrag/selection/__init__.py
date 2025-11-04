"""
Selection module for REfrag.

This module provides policies for selecting which chunks to expand vs compress.
"""

from refrag.selection.base import SelectionPolicy
from refrag.selection.heuristic import (
    HybridPolicy,
    PositionPolicy,
    SimilarityPolicy,
    TFIDFPolicy,
    create_policy,
)

__all__ = [
    "SelectionPolicy",
    "SimilarityPolicy",
    "TFIDFPolicy",
    "PositionPolicy",
    "HybridPolicy",
    "create_policy",
]
