"""MARS on the core, against the earth package, and the learner above it.

The reference is earth's own fit on the fixture design, written by ``inst/spec/make_fixtures.R``:
every term of the forward pass, the terms the pruning pass keeps, their coefficients and the
predictions on the design scaled by 1.01. The terms are asserted exactly, in the order the forward
pass added them, each cut to the tolerance its case carries; the coefficients and the predictions
likewise. The R suite reads the same files.
"""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np
import pytest

from timesift import Response, fit_learner, grain_matrix, mars
from timesift._mars import mars_fit, mars_predict

from terms import assert_hinge_terms

FIXTURES = Path(__file__).resolve().parents[2] / "inst" / "spec" / "fixtures"
HELD = ("unit", "y_gaussian", "y_binomial", "w", "fold")


def read_rows(name):
    with open(FIXTURES / name, newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def mars_input():
    rows = read_rows("penalised_input.csv")
    columns = [c for c in rows[0] if c not in HELD and not c.endswith("^2")]
    return dict(
        x=np.asfortranarray([[float(r[c]) for c in columns] for r in rows]),
        w=np.array([float(r["w"]) for r in rows]),
        y=dict(y_binomial=np.array([float(r["y_binomial"]) for r in rows]),
               y_gaussian=np.array([float(r["y_gaussian"]) for r in rows])))


def case_fit(mars_input, row):
    y = mars_input["y"][row["response"]]
    w = mars_input["w"] if row["weighted"] == "TRUE" else np.ones(len(y))
    family = "gaussian" if row["response"] == "y_gaussian" else "binomial"
    return mars_fit(mars_input["x"], y, w, family, degree=int(row["degree"]),
                    minspan=int(row["minspan"]), endspan=int(row["endspan"]),
                    fast_k=int(row["fast_k"]), prune=row["prune"] == "TRUE",
                    nprune=None if row["nprune"] == "NA" else int(row["nprune"]))


@pytest.mark.parametrize("row", read_rows("mars_cases.csv"), ids=lambda r: r["case"])
def test_the_cores_passes_keep_the_terms_earth_keeps(mars_input, row):
    fit = case_fit(mars_input, row)
    assert_hinge_terms(fit, row["forward"], float(row["cut_tolerance"]))
    assert fit["termcond"] == int(row["termcond"])
    assert " ".join(str(s + 1) for s in fit["selected"]) == row["selected"]
    ctol = float(row["coef_tolerance"])
    assert fit["gcv"] == pytest.approx(float(row["gcv"]), rel=ctol)
    coef = {int(r["term"]): float(r["coefficient"]) for r in read_rows("mars_coef.csv")
            if r["case"] == row["case"]}
    np.testing.assert_allclose(fit["beta"], [coef[s + 1] for s in fit["selected"]], rtol=ctol,
                               atol=ctol)
    pred = [float(r["fitted_out"]) for r in read_rows("mars_predict.csv")
            if r["case"] == row["case"]]
    ptol = float(row["prediction_tolerance"])
    np.testing.assert_allclose(mars_predict(fit, mars_input["x"] * 1.01), pred, rtol=ptol,
                               atol=ptol)


@pytest.mark.parametrize("weighted", [False, True])
def test_the_thread_count_does_not_change_what_the_passes_return(mars_input, weighted):
    x, y = mars_input["x"], mars_input["y"]["y_gaussian"]
    w = mars_input["w"] if weighted else np.ones(len(y))
    one = mars_fit(x, y, w, "gaussian", degree=2, threads=1)
    four = mars_fit(x, y, w, "gaussian", degree=2, threads=4)
    for key in one:
        np.testing.assert_array_equal(np.asarray(four[key]), np.asarray(one[key]))


def test_mars_refuses_settings_it_has_no_pass_for():
    with pytest.raises(ValueError, match="whole number"):
        mars(degree=0)
    with pytest.raises(ValueError, match="whole number"):
        mars(degree=1.5)
    with pytest.raises(ValueError, match="-1"):
        mars(penalty=-2)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        mars(thresh=1.0)
    with pytest.raises(ValueError, match="whole number"):
        mars(nk=0)
    x = np.array([1.0, 3, 2, 5, 4, 6]).reshape(-1, 1)
    with pytest.raises(Exception, match="zero or more"):
        mars_fit(x, np.array([0.0, 1, 0, 1, 0, 1]), np.array([1.0, 1, -1, 1, 1, 1]), "gaussian")
    fit = mars_fit(x, np.array([0.0, 1, 0, 1, 0, 1]), np.ones(6), "binomial")
    with pytest.raises(Exception, match="fitted on 1 columns"):
        mars_predict(fit, np.column_stack([x, x]))


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


@pytest.mark.parametrize("learner", [mars(), mars(degree=2), mars(prune=False), mars(nprune=3)],
                         ids=["additive", "degree2", "unpruned", "nprune"])
def test_mars_fits_and_round_trips(learner):
    x, y = planted()
    fit = fit_learner(learner, x, y)
    p = fit.predict(x)
    assert p.shape == (len(y.units), 2)
    assert np.all((p >= 0) & (p <= 1))
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(fit)).predict(x), p)


def test_an_intercept_alone_predicts_the_share():
    x, y = planted()
    from timesift.learners import flatten
    m = flatten(x)
    lone = mars_fit(m, y.values[:, 0], np.ones(m.shape[0]), "binomial", nprune=1)
    assert list(lone["selected"]) == [0]
    np.testing.assert_allclose(mars_predict(lone, m), y.values[:, 0].mean(), rtol=1e-10)
