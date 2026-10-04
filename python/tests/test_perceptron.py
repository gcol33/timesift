"""The one-hidden-layer network on the core, against the nnet package, and the learner above it.

The reference is nnet's own fit on the fixture design, written by ``inst/spec/make_fixtures.R``
from starting weights written beside it: the objective it ends at, whether it stopped at its
iteration cap, its weights and its predictions on the design scaled by 1.01. The core is started
from the same weights and asserted against all four. The R suite reads the same files.
"""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np
import pytest

from timesift import Response, fit_learner, grain_matrix, perceptron
from timesift._perceptron import perceptron_fit, perceptron_fits, perceptron_predict

FIXTURES = Path(__file__).resolve().parents[2] / "inst" / "spec" / "fixtures"
HELD = ("unit", "y_gaussian", "y_binomial", "w", "fold")


def read_rows(name):
    with open(FIXTURES / name, newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def nn_input():
    rows = read_rows("penalised_input.csv")
    columns = [c for c in rows[0] if c not in HELD and not c.endswith("^2")]
    return dict(
        x=np.asfortranarray([[float(r[c]) for c in columns] for r in rows]),
        w=np.array([float(r["w"]) for r in rows]),
        y=dict(y_binomial=np.array([float(r["y_binomial"]) for r in rows]),
               y_gaussian=np.array([float(r["y_gaussian"]) for r in rows])))


def case_fit(nn_input, row, max_iter=None, start=None):
    y = nn_input["y"][row["response"]]
    w = nn_input["w"] if row["weighted"] == "TRUE" else np.ones(len(y))
    family = "gaussian" if row["response"] == "y_gaussian" else "binomial"
    if start is None:
        start = [float(r["start"]) for r in read_rows("perceptron_weights.csv")
                 if r["case"] == row["case"]]
    return perceptron_fit(nn_input["x"], y, w, family, hidden=int(row["hidden"]),
                          decay=float(row["decay"]),
                          max_iter=int(row["max_iter"]) if max_iter is None else max_iter,
                          skip=row["skip"] == "TRUE", start=start)


@pytest.mark.parametrize("row", read_rows("perceptron_cases.csv"), ids=lambda r: r["case"])
def test_the_core_ends_where_nnet_ends_from_the_same_starting_weights(nn_input, row):
    fit = case_fit(nn_input, row)
    wtol = float(row["weight_tolerance"])
    ptol = float(row["prediction_tolerance"])
    assert fit["value"] == pytest.approx(float(row["value"]), rel=ptol)
    assert (not fit["converged"]) == (row["stopped"] == "TRUE")
    fitted = [float(r["fitted"]) for r in read_rows("perceptron_weights.csv")
              if r["case"] == row["case"]]
    np.testing.assert_allclose(fit["weights"], fitted, rtol=wtol, atol=wtol)
    pred = [float(r["fitted_out"]) for r in read_rows("perceptron_predict.csv")
            if r["case"] == row["case"]]
    np.testing.assert_allclose(perceptron_predict(fit, nn_input["x"] * 1.01), pred, rtol=ptol,
                               atol=ptol)


@pytest.mark.parametrize("family", ["binomial", "gaussian", "poisson"])
@pytest.mark.parametrize("skip", [False, True])
def test_the_first_step_runs_down_the_objectives_gradient(family, skip):
    rng = np.random.default_rng(3)
    n, p = 30, 3
    x = np.asfortranarray(rng.normal(size=(n, p)))
    w = rng.uniform(0.5, 2, n)
    y = {"binomial": rng.binomial(1, 0.4, n), "gaussian": rng.normal(size=n),
         "poisson": rng.poisson(2, n)}[family].astype(float)
    start = rng.uniform(-0.5, 0.5, 2 * (p + 1) + 3 + (p if skip else 0))

    def value(v):
        return perceptron_fit(x, y, w, family, decay=0.05, max_iter=0, skip=skip,
                              start=v)["value"]

    h = 1e-6
    numeric = np.array([(value(start + h * e) - value(start - h * e)) / (2 * h)
                        for e in np.eye(len(start))])
    step = start - perceptron_fit(x, y, w, family, decay=0.05, max_iter=2, skip=skip,
                                  start=start)["weights"]
    np.testing.assert_allclose(step / np.linalg.norm(step), numeric / np.linalg.norm(numeric),
                               atol=1e-5)


def test_the_drawn_starting_weights_follow_the_seed_and_stay_inside_the_range(nn_input):
    x, y, w = nn_input["x"], nn_input["y"]["y_binomial"], nn_input["w"]
    a = perceptron_fit(x, y, w, "binomial", range_=0.3, max_iter=0, seed=7)["weights"]
    b = perceptron_fit(x, y, w, "binomial", range_=0.3, max_iter=0, seed=7)["weights"]
    c = perceptron_fit(x, y, w, "binomial", range_=0.3, max_iter=0, seed=8)["weights"]
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.all(np.abs(a) <= 0.3)


def test_the_presets_give_biomod2s_settings_and_an_explicit_setting_beats_them():
    keys = ("hidden", "decay", "range", "max_iter")
    assert {k: perceptron().params[k] for k in keys} == dict(hidden=2, decay=0.0, range=0.7,
                                                             max_iter=100)
    assert {k: perceptron(preset="bigboss").params[k] for k in keys} == dict(
        hidden=5, decay=0.1, range=0.1, max_iter=200)
    assert perceptron(preset="bigboss", hidden=3).params["hidden"] == 3
    with pytest.raises(ValueError, match="hidden"):
        perceptron(hidden=0)
    with pytest.raises(ValueError, match="decay"):
        perceptron(decay=-1)
    with pytest.raises(ValueError, match="skip"):
        perceptron(skip=None)


def planted(n_unit=60, days=120, seed=43):
    rng = np.random.default_rng(seed)
    t = np.datetime64("2021-09-01T00:00:00", "s") + np.arange(24 * days) * np.timedelta64(1, "h")
    units = [f"p{i:02d}" for i in range(n_unit)]
    warmth = rng.normal(size=n_unit)
    value = np.concatenate([w * 2.0 + rng.normal(0, 1.0, len(t)) for w in warmth])
    d = {"id": [u for u in units for _ in range(len(t))],
         "time": list(t) * n_unit, "value": list(value)}
    y = rng.binomial(1, 1 / (1 + np.exp(-3 * np.column_stack([warmth, -warmth])))).astype(float)
    x = grain_matrix(d, "id", "time", "value", grain="month")
    return x, Response(y, tuple(units), ("sp1", "sp2"))


@pytest.mark.parametrize("learner", [perceptron(), perceptron(preset="bigboss"),
                                     perceptron(skip=True)], ids=["default", "bigboss", "skip"])
def test_the_learner_fits_and_round_trips(learner):
    x, y = planted()
    fit = fit_learner(learner, x, y)
    p = fit.predict(x)
    assert p.shape == (len(y.units), 2)
    assert np.all((p >= 0) & (p <= 1))
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(fit)).predict(x), p)


def test_a_constant_response_is_predicted_its_mean():
    x, y = planted()
    flat = Response(np.zeros((len(y.units), 1)), y.units, ("absent",))
    assert np.all(fit_learner(perceptron(), x, flat).predict(x) == 0)


def test_a_network_too_large_for_its_inverse_hessian_is_refused_with_its_size():
    x, y = planted()
    with pytest.raises(ValueError, match="approximate inverse Hessian"):
        fit_learner(perceptron(hidden=10, max_hessian=1e-6), x, y)


def test_networks_fitted_on_several_threads_are_the_networks_each_response_gets_alone(nn_input):
    x, yb, w = nn_input["x"], nn_input["y"]["y_binomial"], nn_input["w"]
    y = np.column_stack([yb, 1 - yb, yb[::-1]])
    ws = np.column_stack([w, np.ones(len(w)), w])
    seeds = [3, 11, 29]
    many = perceptron_fits(x, y, ws, "binomial", seeds, decay=0.05, threads=3)
    for j in range(3):
        one = perceptron_fit(x, y[:, j], ws[:, j], "binomial", decay=0.05, seed=seeds[j])
        np.testing.assert_array_equal(many[j]["weights"], one["weights"])
        assert many[j]["value"] == one["value"]
    with pytest.raises(ValueError, match="threads"):
        perceptron(threads=0)


def test_a_standardised_network_is_the_network_on_the_standardised_columns(nn_input):
    x, y, w = nn_input["x"], nn_input["y"]["y_binomial"], nn_input["w"]
    a = perceptron_fit(x, y, w, "binomial", decay=0.05, standardise=True)
    np.testing.assert_allclose(a["centre"], x.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(a["scale"], x.std(axis=0, ddof=1), rtol=1e-12)

    def scaled(m):
        return np.asfortranarray((m - a["centre"]) / a["scale"])

    b = perceptron_fit(scaled(x), y, w, "binomial", decay=0.05)
    np.testing.assert_array_equal(a["weights"], b["weights"])
    np.testing.assert_array_equal(perceptron_predict(a, x * 1.01),
                                  perceptron_predict(b, scaled(x * 1.01)))
    assert len(perceptron_fit(x, y, w, "binomial")["centre"]) == 0
    with pytest.raises(ValueError, match="standardise"):
        perceptron(standardise=None)
