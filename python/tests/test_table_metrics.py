"""The table metrics, the minimum predicted area cut and the Boyce index, read against values worked
out by hand. The fixtures pin the same quantities across both languages."""

from __future__ import annotations

import numpy as np
import pytest

import timesift as ts
from timesift.metrics import TABLE_METRICS, boyce_index, decision_threshold, table_metric

Y = [0, 0, 0, 1, 1, 1, 0, 1]
P = [0.10, 0.20, 0.35, 0.40, 0.60, 0.90, 0.55, 0.70]


def test_each_table_metric_reads_the_cells_of_the_table():
    assert decision_threshold(Y, P, "youden") == 0.6
    by_hand = dict(pod=3 / 4, pofd=0, far=0, sr=1, accuracy=7 / 8, bias=3 / 4, orss=1, csi=3 / 4,
                   ets=(3 - 1.5) / (4 - 1.5))
    for name, value in by_hand.items():
        assert table_metric(Y, P, name) == pytest.approx(value), name
    # M F = 0, so the odds ratio is undefined rather than infinite.
    assert np.isnan(table_metric(Y, P, "or"))
    assert table_metric(Y, P, "or", threshold=0.5) == pytest.approx(9.0)
    assert table_metric(Y, P, "far", threshold=0.5) == pytest.approx(1 / 4)


def test_a_table_metric_is_nan_where_its_denominator_or_its_cell_is_empty():
    assert np.isnan(table_metric([0, 0, 0], [0.1, 0.2, 0.3], "pod"))
    assert np.isnan(table_metric(Y, P, "sr", threshold=2))
    with pytest.raises(ValueError):
        table_metric(Y, P, "nope")


def test_every_table_metric_is_registered_and_read_by_name():
    for name in TABLE_METRICS:
        assert name in ts.metrics()
    assert "boyce" in ts.metrics()
    assert ts.resolve_metric("csi")[0](Y, P) == pytest.approx(3 / 4)


def test_the_minimum_predicted_area_cut_keeps_the_share_of_presences_asked_for():
    assert decision_threshold(Y, P, "mpa") == 0.4
    assert decision_threshold(Y, P, "mpa", perc=0.5) == 0.7
    assert decision_threshold(Y, P, "mpa", perc=0.25) == 0.9
    with pytest.raises(ValueError):
        decision_threshold(Y, P, "mpa", perc=0)
    assert np.isnan(decision_threshold([0, 0], [0.1, 0.2], "mpa"))


def test_the_boyce_index_is_high_where_presences_thicken_with_the_prediction():
    assert boyce_index(Y, P) > 0.5
    assert boyce_index(Y, [1 - p for p in P]) < 0
    assert np.isnan(boyce_index(Y, [0.5] * 8))
    assert np.isnan(boyce_index([0, 0, 0], [0.1, 0.2, 0.3]))
