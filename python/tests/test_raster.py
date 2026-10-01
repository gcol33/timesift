from __future__ import annotations

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from timesift import ensemble, grains, project, range_change, timesift
from timesift.learners import Learner
from timesift.response import Response, fold_map

N_ROW, N_COL, DAYS = 4, 5, 120


def case(hole=None):
    rng = np.random.default_rng(91)
    n = N_ROW * N_COL
    when = np.datetime64("2021-09-01T00:00:00", "s") + np.arange(DAYS) * np.timedelta64(1, "D")
    warmth = rng.normal(size=n)
    readings = warmth[None, :] + rng.normal(0, 0.5, (DAYS, n))
    elev = rng.normal(size=n)
    if hole is not None:
        readings[:, hole] = np.nan
        elev[hole] = np.nan
    coords = {"y": np.arange(N_ROW), "x": np.arange(N_COL)}
    temp = xr.DataArray(readings.reshape(DAYS, N_ROW, N_COL), dims=("time", "y", "x"),
                        coords={"time": when, **coords})
    static = xr.Dataset({"elev": xr.DataArray(elev.reshape(N_ROW, N_COL), dims=("y", "x"),
                                              coords=coords)})
    usable = [i for i in range(n) if hole is None or i not in hole]
    ids = [f"c{i + 1:09d}" for i in usable]
    targets = {"cell": ids, "elev": elev[usable].tolist(),
               "sp1": rng.binomial(1, 1 / (1 + np.exp(-warmth[usable]))).tolist(),
               "sp2": rng.binomial(1, 1 / (1 + np.exp(warmth[usable]))).tolist()}
    series = {"cell": [i for i in ids for _ in range(DAYS)], "t": list(when) * len(ids),
              "temp": readings[:, usable].T.reshape(-1).tolist()}
    return temp, static, targets, series, n, usable


def mean_reader(name="rd"):
    def predict(model, x):
        s = x.channel("mean").mean(axis=1)
        return np.column_stack([1 / (1 + np.exp(-s)), 1 / (1 + np.exp(s))])

    return Learner(name=name, multi="joint", fit=lambda x, y, **k: None, predict=predict)


def run_for(targets, series, models, use_ensemble=False):
    y = Response(np.column_stack([targets["sp1"], targets["sp2"]]).astype(float),
                 tuple(targets["cell"]), ("sp1", "sp2"))
    return timesift(targets, series, y=["sp1", "sp2"], id="cell", time="t", x="temp",
                    models=models, sift=grains("month"), resampling=fold_map(y, v=3, seed=2),
                    inner=None, ensemble=ensemble("mean") if use_ensemble else False,
                    verbose=False)


def test_a_projection_is_the_fit_applied_to_one_target_per_cell():
    temp, _, targets, series, n, _ = case()
    fit = run_for(targets, series, [mean_reader()])
    got = project(fit, series=temp, candidate="rd / month")
    assert got.dims == ("response", "y", "x") and list(got["response"].values) == ["sp1", "sp2"]
    want = fit.predict(targets, series, candidate="rd / month")
    assert np.allclose(got.values.reshape(2, n).T, want)
    chunked = project(fit, series=temp, candidate="rd / month", chunk=3)
    assert np.allclose(chunked.values, got.values)


def test_a_cell_with_a_missing_input_is_missing_in_every_layer():
    temp, _, targets, series, n, _ = case(hole=[1, 10])
    fit = run_for(targets, series, [mean_reader()])
    values = project(fit, series=temp, candidate="rd / month").values.reshape(2, n)
    assert np.isnan(values[:, [1, 10]]).all()
    assert np.isfinite(np.delete(values, [1, 10], axis=1)).all()


def test_binary_and_spread_maps_follow_predict():
    temp, _, targets, series, n, _ = case()
    fit = run_for(targets, series, [mean_reader()])
    binary = project(fit, series=temp, candidate="rd / month", type="binary")
    want = fit.predict(targets, series, candidate="rd / month", type="binary")
    assert np.allclose(binary.values.reshape(2, n).T, want, equal_nan=True)

    two = run_for(targets, series, [mean_reader("a"), mean_reader("b")], use_ensemble=True)
    spread = project(two, series=temp, type="spread")
    assert spread.dims == ("response", "statistic", "y", "x")
    assert list(spread["statistic"].values) == ["mean", "sd", "cv", "lower", "upper"]
    mean_map = project(two, series=temp)
    assert np.allclose(spread.sel(statistic="mean").values, mean_map.values)


def test_static_predictors_are_read_from_the_grids_of_a_dataset():
    temp, static, targets, _, n, _ = case()
    y = Response(np.column_stack([targets["sp1"], targets["sp2"]]).astype(float),
                 tuple(targets["cell"]), ("sp1", "sp2"))
    fit = timesift(targets, y=["sp1", "sp2"], id="cell", static=["elev"], models="elasticnet",
                   ensemble=False, resampling=fold_map(y, v=3, seed=2), inner=None, verbose=False)
    got = project(fit, static=static, candidate="elasticnet / static")
    want = fit.predict(targets, candidate="elasticnet / static")
    assert np.allclose(got.values.reshape(2, n).T, want)


def test_a_projection_says_what_it_was_not_given():
    temp, static, targets, series, _, _ = case()
    fit = run_for(targets, series, [mean_reader()])
    with pytest.raises(ValueError, match="so `series` is needed"):
        project(fit)
    with pytest.raises(ValueError, match="mapping of grids named"):
        project(fit, series={"nope": temp})
    with pytest.raises(ValueError, match="1 or more"):
        project(fit, series=temp, chunk=0)
    with pytest.raises(ValueError, match="`type` is one of"):
        project(fit, series=temp, type="nope")
    with pytest.raises(ValueError, match="needs a `time` dimension"):
        project(fit, series=temp.isel(time=0, drop=True))


def test_the_range_change_counts_the_cells_lost_kept_and_gained():
    now = np.array([[1, 0], [1, 0], [1, 1], [0, 1], [0, 1], [np.nan, 1]])
    later = np.array([[1, 0], [0, 0], [0, 1], [1, 1], [1, 0], [1, 0]])
    rc = range_change(now, later)
    sp1, sp2 = rc.table
    # sp1 over the five cells both maps hold: kept, lost, lost, gained, gained.
    assert (sp1["lost"], sp1["kept"], sp1["gained"], sp1["absent"]) == (2, 1, 2, 0)
    assert sp1["current"] == 3 and sp1["later"] == 3
    assert sp1["percent_loss"] == pytest.approx(100 * 2 / 3) and sp1["change"] == pytest.approx(0)
    assert (sp2["lost"], sp2["kept"], sp2["gained"], sp2["absent"]) == (2, 2, 0, 2)
    assert sp2["change"] == pytest.approx(-50)
    assert np.allclose(rc.map[:, 0], [-1, -2, -2, 1, 1, np.nan], equal_nan=True)
    assert np.allclose(rc.map[:, 1], [0, 0, -1, -1, -2, -2])


def test_a_map_that_is_not_zero_and_one_is_cut_at_a_threshold():
    now = np.array([[0.1], [0.6], [0.9], [0.4]])
    later = np.array([[0.7], [0.2], [0.8], [0.3]])
    with pytest.raises(ValueError, match="give `threshold`"):
        range_change(now, later)
    assert np.allclose(range_change(now, later, threshold=0.5).map[:, 0], [1, -2, -1, 0])
    with pytest.raises(ValueError, match="same cells"):
        range_change(now, later[:3], threshold=0.5)
    with pytest.raises(ValueError, match="one cut"):
        range_change(now, later, threshold=[0.5, 0.5])


def test_the_range_change_of_two_projections_is_a_grid_of_codes():
    temp, _, targets, series, n, _ = case()
    fit = run_for(targets, series, [mean_reader()])
    now = project(fit, series=temp, candidate="rd / month", type="binary")
    later = project(fit, series=temp + 2, candidate="rd / month", type="binary")
    rc = range_change(now, later)
    assert rc.map.dims == ("response", "y", "x") and rc.variables == ("sp1", "sp2")
    assert all(row["gained"] >= 0 for row in rc.table)
