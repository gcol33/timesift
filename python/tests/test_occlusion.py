from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from timesift.ladder import grain_ladder
from timesift.learners import Learner
from timesift.occlusion import ladder_occlusion, feature_matrix
from timesift.representation import grain_matrix
from timesift.response import Response, fold_map


def planted(planted_month="2021-11", n_unit=60, seed=61):
    """A record in which the units differ from each other in one stretch of the calendar and
    nowhere else, so the profile has a known answer to be checked against."""
    rng = np.random.default_rng(seed)
    t = np.datetime64("2021-09-01T00:00:00", "s") + np.arange(24 * 210) * np.timedelta64(1, "h")
    in_month = np.array([str(d)[:7] == planted_month for d in t.astype("datetime64[D]")])
    units = [f"p{i:03d}" for i in range(n_unit)]
    warmth = rng.normal(size=n_unit)
    shape = 5 * np.sin(np.arange(len(t)) / (24 * 40))
    value = np.concatenate([shape + w * in_month * 6 + rng.normal(0, 0.3, len(t))
                            for w in warmth])
    readings = {"id": [u for u in units for _ in range(len(t))],
                "time": list(t) * n_unit, "value": list(value)}
    y = rng.binomial(1, 1 / (1 + np.exp(-3 * np.outer(warmth, [1, -1]))))
    return readings, Response(y.astype(float), tuple(units), ("sp0", "sp1")), planted_month


def test_a_feature_table_becomes_a_one_channel_representation():
    m = np.arange(30, dtype=float).reshape(10, 3)
    x = feature_matrix(m, units=[f"p{i}" for i in range(10)], features=["a", "b", "c"])
    assert x.values.shape == (10, 3, 1)
    assert x.bins == ("a", "b", "c")
    assert x.grain == "features"
    assert np.array_equal(x.channel("features"), m)


def test_the_bin_a_signal_was_planted_in_is_the_bin_the_profile_weights():
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn is not installed")
    readings, y, month = planted()
    x = grain_matrix(readings, "id", "time", "value", grain="month")
    lad = grain_ladder(x, y, "elasticnet", folds=fold_map(y, v=4, seed=6),
                        keep_fits=True, verbose=False)
    out = ladder_occlusion(lad, x, y, "month|elasticnet", permutations=5, seed=4)
    mean_weight = np.nanmean(out["weight"], axis=1)
    heaviest = out["part"][int(np.nanargmax(mean_weight))]
    assert heaviest[:7] == month
    assert np.nanmax(mean_weight) > 2 * np.nanmedian(mean_weight)


def test_the_profile_is_read_by_the_metric_the_fit_was_scored_under():
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn is not installed")
    readings, y, _ = planted(n_unit=40, seed=64)
    x = grain_matrix(readings, "id", "time", "value", grain="month")
    lad = grain_ladder(x, y, "elasticnet", folds=fold_map(y, v=3, seed=6), metric="tss",
                       keep_fits=True, verbose=False)

    # The ladder was scored by TSS, so a weight is a fall in TSS unless another metric is named.
    under_tss = ladder_occlusion(lad, x, y, "month|elasticnet", permutations=3, seed=4)
    named = ladder_occlusion(lad, x, y, "month|elasticnet", metric="tss", permutations=3, seed=4)
    assert np.allclose(under_tss["weight"], named["weight"], equal_nan=True)

    under_auc = ladder_occlusion(lad, x, y, "month|elasticnet", metric="roc_auc",
                                 permutations=3, seed=4)
    assert not np.allclose(under_tss["weight"], under_auc["weight"], equal_nan=True)


def test_a_head_that_is_not_presence_absence_can_be_occluded(temporary_response):
    from timesift.learners import fit_learner
    from timesift.response import as_response, scorable_cells
    temporary_response("continuous_occlusion_test", dict(
        prepare=as_response, activation="identity", loss="squared_error", metric="roc_auc",
        cells=lambda y, folds: scorable_cells(
            Response((y.values > np.median(y.values)).astype(float), y.units, y.variables),
            folds)))

    readings, binary, _ = planted(n_unit=30, seed=65)
    x = grain_matrix(readings, "id", "time", "value", grain="month")
    level = x.values[:, :, 0].mean(axis=1)
    y = Response(np.column_stack([level, -level]), binary.units, ("height", "depth"))

    folds = fold_map(binary, v=3, seed=6)
    lad = grain_ladder(x, y, "stepwise", folds=folds, response="continuous_occlusion_test",
                       keep_fits=True, verbose=False)
    out = ladder_occlusion(lad, x, y, "month|stepwise", permutations=2, seed=4)
    assert out["variable"] == ["height", "depth"]
    assert np.isfinite(out["weight"]).any()


def test_holding_a_channel_back_asks_what_the_statistic_carries():
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn is not installed")
    readings, y, _ = planted(n_unit=40, seed=63)
    x = grain_matrix(readings, "id", "time", "value", grain="month",
                      stats=["cold_day", "mean", "warm_day"])
    lad = grain_ladder(x, y, "elasticnet", folds=fold_map(y, v=3, seed=6),
                        keep_fits=True, verbose=False)
    out = ladder_occlusion(lad, x, y, "month|elasticnet", over="channel", permutations=3)
    assert list(out["part"]) == ["cold_day", "mean", "warm_day"]


def test_an_arm_naming_only_a_learner_is_read_against_the_one_representation_given():
    readings, y, _ = planted(n_unit=20)
    x = grain_matrix(readings, "id", "time", "value", grain="month")
    a = ranker()
    lad = grain_ladder(x, y, a, folds=fold_map(y, v=3, seed=2), keep_fits=True, verbose=False)
    named = ladder_occlusion(lad, x, y, "month|a", permutations=2, seed=3)
    bare = ladder_occlusion(lad, x, y, "a", permutations=2, seed=3)
    assert named["part"] == bare["part"]
    assert np.allclose(np.nan_to_num(named["weight"]), np.nan_to_num(bare["weight"]))
    # Against a set, a bare learner reads the representation of its best grain, as R does.
    from_set = ladder_occlusion(lad, {"month": x}, y, "a", permutations=2, seed=3)
    assert np.allclose(np.nan_to_num(named["weight"]), np.nan_to_num(from_set["weight"]))
    with pytest.raises(ValueError, match='no "month" grain'):
        ladder_occlusion(lad, {"week": x}, y, "a", permutations=2, seed=3)


def test_an_arm_the_ladder_never_fitted_says_so():
    readings, y, _ = planted(n_unit=20)
    x = grain_matrix(readings, "id", "time", "value", grain="month")
    lad = grain_ladder(x, y, ranker(), folds=fold_map(y, v=3, seed=2), keep_fits=True,
                       verbose=False)
    with pytest.raises(KeyError, match="month\\|b"):
        ladder_occlusion(lad, x, y, "month|b")


def test_occlusion_needs_the_fits_the_ladder_was_told_to_keep():
    readings, y, _ = planted(n_unit=20)
    x = grain_matrix(readings, "id", "time", "value", grain="month")
    lad = grain_ladder(x, y, ranker(), folds=fold_map(y, v=3, seed=2), verbose=False)
    with pytest.raises(ValueError, match="kept no fits"):
        ladder_occlusion(lad, x, y, "month|a")


def ranker() -> Learner:
    """Every unit predicted the mean of the fitting units, which the profile reads as no weight."""
    return Learner(name="a", fit=lambda x, y, **k: y.mean(axis=0),
                   predict=lambda m, x: np.tile(m, (x.values.shape[0], 1)))
