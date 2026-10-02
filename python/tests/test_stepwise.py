"""The stepwise model on the core, against MASS's ``stepAIC()``, ``glm()`` and the forward search
over column terms in R alone, and the learner above it.

The references are written separately from the core, by ``inst/spec/make_fixtures.R``: MASS's
search over biomod2's formula for every direction, ``glm()`` on every term for the unselected fit,
and the R oracle under fractional case weights. The terms chosen are asserted exactly, in the order
the final model holds them, and the deviance and the fitted means to the tolerance each case
carries. The R suite reads the same files.
"""

from __future__ import annotations

import csv
import math
import pickle
from pathlib import Path

import numpy as np
import pytest

from counts import fixture_counts
from timesift import Response, feature_matrix, fit_learner, grain_matrix, stepwise
from timesift._stepwise import stepwise_fit, stepwise_predict

FIXTURES = Path(__file__).resolve().parents[2] / "inst" / "spec" / "fixtures"
HELD = ("unit", "y_gaussian", "y_binomial", "w", "fold")


def read_rows(name):
    with open(FIXTURES / name, newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def stepwise_input():
    rows = read_rows("penalised_input.csv")
    columns = [c for c in rows[0] if c not in HELD and not c.endswith("^2")]
    thin = {r["unit"]: r for r in read_rows("maxnet_response.csv")}
    return dict(
        x=np.asfortranarray([[float(r[c]) for c in columns] for r in rows]),
        w=np.array([float(r["w"]) for r in rows]),
        y=dict(y_binomial=np.array([float(r["y_binomial"]) for r in rows]),
               y_8=np.array([float(thin[r["unit"]]["y_8"]) for r in rows]),
               y_12=np.array([float(thin[r["unit"]]["y_12"]) for r in rows]),
               y_gaussian=np.array([float(r["y_gaussian"]) for r in rows]),
               y_poisson=fixture_counts(FIXTURES, [r["unit"] for r in rows])))


@pytest.mark.parametrize("row", read_rows("stepwise_cases.csv"), ids=lambda r: r["case"])
def test_the_cores_search_chooses_the_terms_mass_glm_and_the_r_oracle_choose(stepwise_input, row):
    x = stepwise_input["x"]
    if row["duplicate"] == "TRUE":
        x = np.asfortranarray(np.column_stack([x, x[:, 0]]))
    y = stepwise_input["y"][row["response"]]
    w = stepwise_input["w"] if row["weighted"] == "TRUE" else np.ones(len(y))
    family = {"y_gaussian": "gaussian", "y_poisson": "poisson"}.get(row["response"], "binomial")
    fit = stepwise_fit(x, y, w, family, max_terms=float(row["max_terms"]),
                       degree=int(row["degree"]), direction=row["direction"], terms=row["terms"])
    chosen = " ".join(f"{c + 1}:{p}" for c, p in zip(fit["term_column"], fit["term_power"]))
    assert chosen == row["chosen"]
    assert fit["rank"] == int(row["rank"])
    assert fit["converged"] == (row["converged"] == "TRUE")
    assert fit["steps"] == int(row["steps"])
    assert fit["deviance"] == pytest.approx(float(row["deviance"]),
                                            rel=float(row["deviance_tolerance"]))
    pred = [r for r in read_rows("stepwise_predict.csv") if r["case"] == row["case"]]
    tol = float(row["prediction_tolerance"])
    np.testing.assert_allclose(stepwise_predict(fit, x), [float(r["fitted"]) for r in pred],
                               rtol=tol, atol=tol)
    np.testing.assert_allclose(stepwise_predict(fit, x * 1.01),
                               [float(r["fitted_out"]) for r in pred], rtol=tol, atol=tol)


def test_the_thread_count_does_not_change_what_the_search_returns(stepwise_input):
    x, y, w = stepwise_input["x"], stepwise_input["y"]["y_binomial"], stepwise_input["w"]
    one = stepwise_fit(x, y, w, "binomial", max_terms=math.inf, degree=3, direction="both",
                       terms="power", threads=1)
    four = stepwise_fit(x, y, w, "binomial", max_terms=math.inf, degree=3, direction="both",
                        terms="power", threads=4)
    for key in one:
        np.testing.assert_array_equal(np.asarray(four[key]), np.asarray(one[key]))


def test_a_move_whose_fit_does_not_settle_is_refused_and_a_final_one_is_named():
    # Twenty units split by the first column: its fit runs the coefficient outward at every
    # iteration and has not settled after 25, so the search keeps the intercept.
    v = np.arange(1.0, 21.0)
    x = np.column_stack([v, np.sin(v)])
    y = (v > 10).astype(float)
    fit = stepwise_fit(x, y, np.ones(20), "binomial", terms="power", degree=1)
    assert len(fit["term_column"]) == 0
    full = stepwise_fit(x, y, np.ones(20), "binomial", direction="none", terms="power", degree=1)
    assert not full["converged"]
    units = tuple(f"u{i:02d}" for i in range(20))
    features = feature_matrix(x, units=units, features=("a", "b"))
    learned = fit_learner(stepwise(direction="none", terms="power", degree=1), features,
                          Response(y.reshape(-1, 1), units, ("sp",)))
    assert learned.model["stopped"] == ["sp"]


def test_stepwise_refuses_settings_it_has_no_search_for():
    with pytest.raises(ValueError, match="direction"):
        stepwise(direction="sideways")
    with pytest.raises(ValueError, match="terms"):
        stepwise(terms="pair")
    with pytest.raises(ValueError, match="zero or more"):
        stepwise(max_terms=-1)
    with pytest.raises(ValueError, match="whole number"):
        stepwise(degree=1.5)
    x = np.arange(1.0, 7.0).reshape(-1, 1)
    with pytest.raises(Exception, match="zero and one"):
        stepwise_fit(x, np.array([0.0, 1, 2, 0, 1, 0]), np.ones(6), "binomial")
    with pytest.raises(Exception, match="positive weights"):
        stepwise_fit(x, np.array([0.0, 1, 1, 0, 1, 0]), np.array([1.0, 1, 0, 1, 1, 1]),
                     "binomial")
    fit = stepwise_fit(x, np.array([0.0, 1, 1, 0, 1, 0]), np.ones(6), "binomial")
    with pytest.raises(Exception, match="read 1 columns"):
        stepwise_predict(fit, np.column_stack([x, x]))


def planted(n_unit=60, days=120, seed=41):
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


@pytest.mark.parametrize("learner", [
    stepwise(), stepwise(direction="both", terms="power", max_terms=math.inf),
    stepwise(direction="backward", terms="power"), stepwise(direction="none")],
    ids=["forward", "both", "backward", "none"])
def test_stepwise_fits_every_direction_and_round_trips(learner):
    x, y = planted()
    fit = fit_learner(learner, x, y)
    p = fit.predict(x)
    assert p.shape == (len(y.units), 2)
    assert np.all((p >= 0) & (p <= 1))
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(fit)).predict(x), p)


def test_a_model_of_no_term_predicts_the_share():
    x, y = planted()
    from timesift.learners import flatten
    m = flatten(x)
    empty = stepwise_fit(m, y.values[:, 0], np.ones(m.shape[0]), "binomial", max_terms=0)
    assert len(empty["term_column"]) == 0
    np.testing.assert_allclose(stepwise_predict(empty, m), y.values[:, 0].mean())
