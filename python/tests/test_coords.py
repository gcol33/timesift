"""The coordinates a target carries and the unit it belongs to, which place a spatial field.

``coords`` names the two columns holding each target's place. They reach every representation as
attributes of the array rather than channels, so no learner reads them as a predictor, and a split
of the targets splits them with it.
"""

from __future__ import annotations

import numpy as np
import pytest

from timesift import grains, timesift
from timesift.learners import Learner, flatten
from timesift.representation import bind_channels
from timesift.response import Response, fold_map
from timesift.specs import TimesiftSpec, build_representation, check_coords, grain

PLOTS = ("p2", "p1", "p3", "p4")
START = np.datetime64("2021-09-01T00:00:00", "s")


def series(days=45):
    t = START + np.arange(days * 24) * np.timedelta64(3600, "s")
    return {"plot": np.repeat(np.asarray(PLOTS), len(t)), "when": np.tile(t, len(PLOTS)),
            "temp": np.concatenate([np.sin(np.arange(len(t)) / 13.0) + i for i in range(4)])}


def targets():
    return {"plot": list(PLOTS), "sp_a": [0, 1, 0, 1], "elevation": [1000.0, 1010.0, 1020.0, 1030.0],
            "lon": [10.0, 11.0, 12.0, 13.0], "lat": [45.5, 46.5, 47.5, 48.5]}


def spec(**given):
    settings = dict(y=("sp_a",), x=("temp",), id="plot", time="when", coords=("lon", "lat"))
    settings.update(given)
    return TimesiftSpec(**settings)


def test_coords_names_two_numeric_columns():
    t = targets()
    with pytest.raises(ValueError, match="two columns"):
        check_coords(t, ("lon", "lat", "elevation"))
    with pytest.raises(ValueError, match="must be numeric"):
        check_coords({**t, "lon": ["a", "b", "c", "d"]}, ("lon", "lat"))
    with pytest.raises(ValueError, match="must be numeric"):
        check_coords({**t, "lat": [1.0, np.nan, 3.0, 4.0]}, ("lon", "lat"))
    with pytest.raises(ValueError, match="must be numeric"):
        check_coords({**t, "lat": [1.0, np.inf, 3.0, 4.0]}, ("lon", "lat"))
    check_coords(t, ())
    check_coords(t, ("lon", "lat"))


def test_every_representation_carries_the_coordinates_and_the_unit_of_each_target():
    t = targets()
    for rep in (grain("week"), grain("month")):
        m = build_representation(rep, series(), t, spec())
        assert m.coords.shape == (4, 2)
        np.testing.assert_array_equal(m.coords, np.column_stack([t["lon"], t["lat"]]))
        assert m.unit_ids == PLOTS
    from timesift.specs import Representation
    static = build_representation(Representation(label="static", kind="static", sequence=False),
                                  None, t, spec(x=(), static=("elevation",), time=None))
    np.testing.assert_array_equal(static.coords[:, 0], t["lon"])
    assert static.unit_ids == PLOTS


def test_an_array_built_without_coords_carries_none_and_one_without_an_id_carries_no_units():
    t = targets()
    m = build_representation(grain("week"), series(), t, spec(coords=()))
    assert m.coords is None and m.unit_ids == PLOTS
    from timesift.specs import Representation
    static = Representation(label="static", kind="static", sequence=False)
    m = build_representation(static, None, t, spec(id=None, x=(), time=None,
                                                   static=("elevation",)))
    assert m.unit_ids is None and m.coords.shape == (4, 2)


def test_a_split_of_the_rows_splits_the_coordinates_and_the_units_with_them():
    t = targets()
    m = build_representation(grain("week"), series(), t, spec())
    sub = m.take_units([3, 1])
    np.testing.assert_array_equal(sub.coords, [[13.0, 48.5], [11.0, 46.5]])
    assert sub.unit_ids == ("p4", "p1")
    assert sub.units == ("p4", "p1")
    other = build_representation(grain("week", stats=("max",)), series(), t, spec())
    both = bind_channels(m, other)
    np.testing.assert_array_equal(both.coords, m.coords)
    assert both.unit_ids == m.unit_ids


def test_the_coordinates_are_no_predictor():
    m = build_representation(grain("week"), series(), targets(), spec())
    plain = build_representation(grain("week"), series(), targets(), spec(coords=()))
    np.testing.assert_array_equal(flatten(m), flatten(plain))
    assert m.shape == plain.shape


def test_a_run_hands_its_learners_the_coordinates_of_the_targets_in_each_split():
    n = 20
    rng = np.random.default_rng(3)
    plots = [f"p{i:02d}" for i in range(n)]
    t = START + np.arange(60 * 24) * np.timedelta64(3600, "s")
    warmth = rng.normal(size=n)
    readings = {"plot": np.repeat(np.asarray(plots), len(t)), "when": np.tile(t, n),
                "temp": np.concatenate([w + rng.normal(0, 0.5, len(t)) for w in warmth])}
    lon, lat = rng.uniform(size=n), rng.uniform(size=n)
    tg = {"plot": plots, "sp": rng.binomial(1, 1 / (1 + np.exp(-warmth))).tolist(),
          "lon": lon.tolist(), "lat": lat.tolist(), "elev": rng.normal(size=n).tolist()}
    seen = []

    def fit(x, y, **_):
        seen.append((x.coords.copy(), x.unit_ids))
        return None

    def predict(model, x):
        seen.append((x.coords.copy(), x.unit_ids))
        return np.full((x.values.shape[0], 1), 0.5)

    y = Response(np.asarray(tg["sp"], dtype=float)[:, None], tuple(plots), ("sp",))
    timesift(tg, readings, y="sp", id="plot", time="when", x="temp", coords=("lon", "lat"),
             learners=[Learner(name="seer", fit=fit, predict=predict, multi="joint")],
             sift=grains("month"), resampling=fold_map(y, v=4, seed=2), n_inner=None,
             ensemble=False, verbose=False)
    assert seen
    for coords, units in seen:
        assert coords.shape[1] == 2 and len(units) == coords.shape[0]
        for (a, b), u in zip(coords, units):
            i = plots.index(u)
            assert (a, b) == (lon[i], lat[i])
    with pytest.raises(ValueError, match="two columns"):
        timesift(tg, readings, y="sp", id="plot", time="when", x="temp", coords=("lon", "lat", "elev"),
                 sift=grains("month"), resampling=fold_map(y, v=4, seed=2), verbose=False)


def test_a_grid_places_each_cell_by_the_dimension_a_coordinate_column_is_named_for():
    xr = pytest.importorskip("xarray")
    from timesift.raster import _cell_coordinates
    grid = xr.DataArray(np.zeros((3, 4)), dims=("y", "x"),
                        coords={"y": [10.0, 20.0, 30.0], "x": [1.0, 2.0, 3.0, 4.0]})
    dims, shape = ("y", "x"), (3, 4)
    named = _cell_coordinates(TimesiftSpec(y=("a",), coords=("x", "y")), grid, dims, shape)
    np.testing.assert_array_equal(named["x"], np.tile([1.0, 2.0, 3.0, 4.0], 3))
    np.testing.assert_array_equal(named["y"], np.repeat([10.0, 20.0, 30.0], 4))
    order = _cell_coordinates(TimesiftSpec(y=("a",), coords=("lon", "lat")), grid, dims, shape)
    np.testing.assert_array_equal(order["lon"], np.repeat([10.0, 20.0, 30.0], 4))
    np.testing.assert_array_equal(order["lat"], np.tile([1.0, 2.0, 3.0, 4.0], 3))
    assert _cell_coordinates(TimesiftSpec(y=("a",)), grid, dims, shape) == {}
