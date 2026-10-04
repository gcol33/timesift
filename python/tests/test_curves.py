from __future__ import annotations

import numpy as np
import pytest

from timesift import ensemble, grains, response_curve, timesift
from timesift.learners import Learner
from timesift.response import Response, fold_map


def curve_reader(name, scale=1.0):
    def predict(model, x):
        s = scale * x.channel("warm_day").mean(axis=1)
        return np.column_stack([s, -s])

    return Learner(name=name, multi="joint", fit=lambda x, y, **k: None, predict=predict)


def curve_run(models, use_ensemble=False):
    rng = np.random.default_rng(71)
    t = np.datetime64("2021-09-01T00:00:00", "s") + np.arange(24 * 120) * np.timedelta64(1, "h")
    units = [f"p{i:02d}" for i in range(40)]
    warmth = rng.normal(size=40)
    value = np.concatenate([w + np.sin(np.arange(len(t)) / 300) + rng.normal(size=len(t))
                            for w in warmth])
    readings = {"plot": [u for u in units for _ in range(len(t))], "time": list(t) * 40,
                "value": list(value)}
    targets = {"plot": units, "sp1": rng.binomial(1, 1 / (1 + np.exp(-warmth))).tolist(),
               "sp2": rng.binomial(1, 1 / (1 + np.exp(warmth))).tolist()}
    y = Response(np.column_stack([targets["sp1"], targets["sp2"]]).astype(float), tuple(units),
                 ("sp1", "sp2"))
    return timesift(targets, readings, y=["sp1", "sp2"], id="plot", time="time", x="value",
                    learners=models, sift=grains("month", stats=["cold_day", "mean", "warm_day"]),
                    resampling=fold_map(y, v=3, seed=2), n_inner=None,
                    ensemble=ensemble("mean") if use_ensemble else False, verbose=False)


def test_a_curve_in_a_channel_is_the_prediction_as_that_statistic_moves_in_every_bin():
    run = curve_run([curve_reader("w")])
    m = run.representations["month"]
    rc = response_curve(run, "w / month", "warm_day", n=7)
    assert rc.prediction.shape == (7, 2)
    warm = m.channel("warm_day")
    assert (rc.value.min(), rc.value.max()) == pytest.approx((warm.min(), warm.max()))
    # The model reads the mean of the warmest day over the bins, so with the statistic set to one
    # value in every bin its prediction is that value.
    assert np.allclose(rc.prediction[:, 0], rc.value)
    assert np.allclose(rc.prediction[:, 1], -rc.value)
    flat = response_curve(run, "w / month", "cold_day", n=5)
    assert np.ptp(flat.prediction[:, 0]) < 1e-12


def test_a_curve_in_one_cell_moves_one_bin_and_holds_the_others_at_the_reference():
    run = curve_run([curve_reader("w")])
    m = run.representations["month"]
    spec = {"bin": m.bins[1], "channel": "warm_day"}
    for fixed, fn in (("mean", np.mean), ("median", np.median), ("min", np.min), ("max", np.max)):
        rc = response_curve(run, "w / month", spec, fixed=fixed, n=4)
        ref = fn(m.channel("warm_day"), axis=0)
        want = (ref.sum() - ref[1] + rc.value) / len(ref)
        assert np.allclose(rc.prediction[:, 0], want), fixed


def test_two_predictors_are_read_over_the_grid_of_the_two():
    run = curve_run([curve_reader("w")])
    rc = response_curve(run, "w / month", "warm_day", with_="cold_day", n=5)
    assert rc.prediction.shape == (25, 2) and rc.value_with is not None
    assert np.allclose(rc.prediction[:, 0], rc.value)
    assert len(np.unique(rc.value_with)) == 5


def test_the_ensemble_is_moved_through_every_member_and_combined_by_the_stack():
    run = curve_run([curve_reader("a"), curve_reader("b", 2.0)], use_ensemble=True)
    rc = response_curve(run, "ensemble", "warm_day", n=6, spread=True)
    assert np.allclose(rc.prediction[:, 0], 1.5 * rc.value)
    # Two members at equal weight, predicting v and 2 v: the standard deviation is |v| / sqrt(2).
    assert np.allclose(rc.sd[:, 0], np.abs(rc.value) / np.sqrt(2))
    # The interval is held inside zero and one under a probability head.
    assert np.all(rc.lower[:, 0] <= rc.upper[:, 0])


def test_a_curve_says_what_it_cannot_draw():
    run = curve_run([curve_reader("w")])
    with pytest.raises(ValueError, match="names what to vary"):
        response_curve(run, "w / month")
    with pytest.raises(ValueError, match="neither a channel nor a bin"):
        response_curve(run, "w / month", "nope")
    with pytest.raises(ValueError, match="2 or more"):
        response_curve(run, "w / month", "warm_day", n=1)
    with pytest.raises(ValueError, match="no ensemble"):
        response_curve(run, "ensemble", "warm_day")
    with pytest.raises(ValueError, match="members"):
        response_curve(run, "w / month", "warm_day", spread=True)
    with pytest.raises(KeyError, match="no candidate called"):
        response_curve(run, "x / month", "warm_day")
