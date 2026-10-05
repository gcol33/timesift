"""The arrays a core fitting several responses at once takes, shaped once for every learner.

``y`` and ``w`` are [n, r], one column per response, and a fold map is [n, r] of 0-based indices
with one count of folds per response. A single response is one column, and a fold map of one
response is one index per unit. The R package's helpers in ``R/learner.R`` hand the same shapes to
the same cores.
"""

from __future__ import annotations

import numpy as np

__all__ = ["as_design", "as_responses", "as_fold_map", "as_fold_counts", "as_seeds"]


def as_design(x) -> np.ndarray:
    return np.asfortranarray(np.asarray(x, dtype=np.float64))


def as_responses(y, w) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if y.ndim == 1:
        y, w = y[:, None], w[:, None]
    return np.asfortranarray(y), np.asfortranarray(w)


def as_fold_map(fold):
    if fold is None:
        return None
    fold = np.asarray(fold, dtype=np.int32)
    return np.asfortranarray(fold[:, None] if fold.ndim == 1 else fold)


def as_fold_counts(n_fold) -> list[int]:
    return [int(k) for k in np.atleast_1d(n_fold)]


def as_seeds(values) -> list[int]:
    """One seed per response, each taken modulo 2^32."""
    return [int(s) & 0xFFFFFFFF for s in np.atleast_1d(values)]
