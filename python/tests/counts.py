"""The count response the Poisson fixtures are fitted to, read by unit rather than by position."""

from __future__ import annotations

import csv

import numpy as np


def fixture_counts(fixtures, units):
    """``y_poisson`` of ``count_response.csv`` for each of ``units``, in their order."""
    with open(fixtures / "count_response.csv", newline="") as handle:
        by_unit = {r["unit"]: float(r["y_poisson"]) for r in csv.DictReader(handle)}
    return np.array([by_unit[u] for u in units])
