"""The additive model on the core, against mgcv, and the learner above it.

The reference is mgcv's own fit on the fixture designs, written by ``inst/spec/make_fixtures.R``
at a tight tolerance: the criterion at its minimum, every column's effective degrees of freedom and
the fitted mean on the design scaled by 1.01. Two searches of the same criterion settle at the same
point to the tolerance they are run at, so each is asserted as a distance, to the tolerance its
case carries. The R suite reads the same files.
"""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np
import pytest

from timesift import Response, additive, fit_learner, grain_matrix
from timesift._additive import additive_fit, additive_predict

FIXTURES = Path(__file__).resolve().parents[2] / "inst" / "spec" / "fixtures"
HELD = ("unit", "y_gaussian", "y_binomial", "w", "fold")


def read_rows(name):
    with open(FIXTURES / name, newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def additive_input():
    rows = read_rows("penalised_input.csv")
    columns = [c for c in rows[0] if c not in HELD and not c.endswith("^2")]
    x = np.asfortranarray([[float(r[c]) for c in columns] for r in rows])
    few = read_rows("additive_input.csv")
    knots = read_rows("additive_knots_input.csv")
    return dict(
        designs=dict(weekly=np.asfortranarray(x[:, :4]), first=np.asfortranarray(x[:, :3]),
                     late=np.asfortranarray(x[:, 4:8]), all=x,
                     few=np.asfortranarray([[float(r[c]) for c in ("v01", "v02", "v03", "v04")]
                                            for r in few]),
                     knots=np.asfortranarray([[float(r["v01"]), float(r["v02"])] for r in knots])),
        y=dict(y_binomial=np.array([float(r["y_binomial"]) for r in rows]),
               y_gaussian=np.array([float(r["y_gaussian"]) for r in rows]),
               y=np.array([float(r["y"]) for r in knots])),
        w=np.array([float(r["w"]) for r in rows]))


@pytest.mark.parametrize("row", read_rows("additive_cases.csv"), ids=lambda r: r["case"])
def test_the_core_settles_where_mgcv_does(additive_input, row):
    x = additive_input["designs"][row["design"]]
    y = additive_input["y"][row["response"]]
    w = additive_input["w"] if row["weighted"] == "TRUE" else np.ones(len(y))
    family = "gaussian" if row["response"] == "y_gaussian" else "binomial"
    fit = additive_fit(x, y, w, family, k=int(row["k"]), gamma=float(row["gamma"]))
    assert list(fit["converged"]) == [1]
    assert fit["score"][0] == pytest.approx(float(row["score"]), rel=float(row["score_tolerance"]))
    edf = [float(v) for v in row["edf"].split(" ")]
    etol = float(row["edf_tolerance"])
    np.testing.assert_allclose(fit["edf"], edf, rtol=etol, atol=etol)
    pred = [r for r in read_rows("additive_predict.csv") if r["case"] == row["case"]]
    at = [int(r["row"]) - 1 for r in pred]
    ptol = float(row["prediction_tolerance"])
    np.testing.assert_allclose(additive_predict(fit, x[at] * 1.01)[:, 0],
                               [float(r["fitted"]) for r in pred], rtol=ptol, atol=ptol)


def test_responses_fitted_together_are_the_responses_fitted_alone(additive_input):
    x = additive_input["designs"]["weekly"]
    yb = additive_input["y"]["y_binomial"]
    w = additive_input["w"]
    y = np.column_stack([yb, 1 - yb])
    ws = np.column_stack([w, w[::-1]])
    both = additive_fit(x, y, ws, "binomial")
    n = int(both["n_coef"])
    for s in range(2):
        alone = additive_fit(x, y[:, s], ws[:, s], "binomial")
        np.testing.assert_array_equal(np.asarray(both["beta"])[s * n:(s + 1) * n], alone["beta"])
        assert both["score"][s] == alone["score"][0]
    four = additive_fit(x, y, ws, "binomial", threads=4)
    for key in both:
        np.testing.assert_array_equal(np.asarray(four[key]), np.asarray(both[key]))
    xa, yg = additive_input["designs"]["all"], additive_input["y"]["y_gaussian"]
    one = additive_fit(xa, yg, w, "gaussian", k=5)
    four = additive_fit(xa, yg, w, "gaussian", k=5, threads=4)
    for key in one:
        np.testing.assert_array_equal(np.asarray(four[key]), np.asarray(one[key]))


def test_a_column_of_two_values_enters_linearly_and_one_of_one_is_left_out():
    rng = np.random.default_rng(3)
    a = rng.normal(size=60)
    x = np.column_stack([a, np.tile([2.0, 5.0], 30), np.ones(60)])
    y = rng.binomial(1, 1 / (1 + np.exp(-a))).astype(float)
    fit = additive_fit(x, y, np.ones(60), "binomial")
    assert list(fit["term_column"]) == [0, 1]
    assert list(fit["term_size"]) == [9, 1]
    assert list(fit["term_penalised"]) == [8, 0]
    assert fit["edf"][1] == pytest.approx(1.0)
    assert fit["n_coef"] == 11


def test_a_linear_part_the_columns_before_it_span_is_held_at_zero():
    rng = np.random.default_rng(4)
    a = rng.normal(size=60)
    x = np.column_stack([a, 2 * a + 1])
    y = rng.binomial(1, 1 / (1 + np.exp(-a))).astype(float)
    fit = additive_fit(x, y, np.ones(60), "binomial")
    assert list(fit["aliased"]) == [18]
    assert fit["beta"][18] == 0.0


def test_the_additive_model_refuses_settings_it_has_no_fit_for():
    with pytest.raises(ValueError, match="whole number"):
        additive(k=2)
    with pytest.raises(ValueError, match="at least `k`"):
        additive(k=10, max_knots=5)
    with pytest.raises(ValueError, match="positive"):
        additive(gamma=0)
    rng = np.random.default_rng(5)
    x = rng.normal(size=(20, 3))
    y = np.tile([0.0, 1.0], 10)
    with pytest.raises(Exception, match="positive"):
        additive_fit(x, y, np.r_[-1.0, np.ones(19)], "binomial")
    with pytest.raises(Exception, match="more than the 20 units"):
        additive_fit(x, y, np.ones(20), "binomial")
    fit = additive_fit(x[:, :1], y, np.ones(20), "binomial", k=5)
    with pytest.raises(Exception, match="fitted on 1 columns"):
        additive_predict(fit, x)


def planted(n_unit=60, days=120, seed=43):
    rng = np.random.default_rng(seed)
    t = np.datetime64("2021-09-01T00:00:00", "s") + np.arange(24 * days) * np.timedelta64(1, "h")
    units = [f"p{i:02d}" for i in range(n_unit)]
    warmth = rng.normal(size=n_unit)
    value = np.concatenate([w * 2.0 + rng.normal(0, 1.0, len(t)) for w in warmth])
    d = {"id": [u for u in units for _ in range(len(t))],
         "time": list(t) * n_unit, "value": list(value)}
    y = rng.binomial(1, 1 / (1 + np.exp(-3 * np.column_stack([warmth, -warmth])))).astype(float)
    x = grain_matrix(d, "id", "time", "value", grain="season")
    return x, Response(y, tuple(units), ("sp1", "sp2"))


@pytest.mark.parametrize("learner", [additive(), additive(k=5, gamma=1.4)],
                         ids=["default", "k5_gamma"])
def test_the_learner_fits_and_round_trips(learner):
    x, y = planted()
    fit = fit_learner(learner, x, y)
    p = fit.predict(x)
    assert p.shape == (len(y.units), 2)
    assert np.all((p >= 0) & (p <= 1))
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(fit)).predict(x), p)


def test_a_constant_response_is_predicted_its_mean():
    x, y = planted()
    values = y.values.copy()
    values[:, 1] = 0.0
    flat = Response(values, y.units, y.variables)
    fit = fit_learner(additive(), x, flat)
    assert fit.model["unfitted"] == ["sp2"]
    np.testing.assert_array_equal(np.unique(fit.predict(x)[:, 1]), [0.0])
