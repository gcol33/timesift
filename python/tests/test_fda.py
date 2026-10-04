"""The discriminant on the core, against the mda package, and the learner above it.

The reference is mda's own ``fda(method = mars)`` on the fixture design, written by
``inst/spec/make_fixtures.R``: the forward terms counted, the kept terms, their coefficients, and on
the design scaled by 1.01 the posterior and biomod2's probit recalibration of it. The kept terms are
asserted exactly, each cut to the tolerance its case carries; the coefficients and the predictions
likewise. The R suite reads the same files.
"""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np
import pytest

from timesift import Response, discriminant, fit_learner, grain_matrix
from timesift._fda import fda_fit, fda_predict

from terms import assert_hinge_terms, hinge_terms

FIXTURES = Path(__file__).resolve().parents[2] / "inst" / "spec" / "fixtures"
HELD = ("unit", "y_gaussian", "y_binomial", "w", "fold")


def read_rows(name):
    with open(FIXTURES / name, newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def fda_input():
    rows = read_rows("penalised_input.csv")
    squares = [c for c in rows[0] if c not in HELD]
    columns = [c for c in squares if not c.endswith("^2")]
    return dict(
        x=dict(columns=np.asfortranarray([[float(r[c]) for c in columns] for r in rows]),
               squares=np.asfortranarray([[float(r[c]) for c in squares] for r in rows])),
        w=np.array([float(r["w"]) for r in rows]),
        y=np.array([float(r["y_binomial"]) for r in rows]))


def case_fit(fda_input, row, calibrate):
    y = fda_input["y"]
    w = fda_input["w"] if row["weighted"] == "TRUE" else np.ones(len(y))
    return fda_fit(fda_input["x"][row["design"]], y, w, degree=int(row["degree"]),
                   prune=row["prune"] == "TRUE", calibrate=calibrate)


@pytest.mark.parametrize("row", read_rows("fda_cases.csv"), ids=lambda r: r["case"])
def test_the_core_keeps_the_terms_mda_keeps_and_predicts_as_it_and_biomod2_do(fda_input, row):
    x = fda_input["x"][row["design"]]
    fit = case_fit(fda_input, row, calibrate=False)
    assert_hinge_terms(fit, row["kept"], float(row["cut_tolerance"]))
    assert fit["forward_terms"] == int(row["n_forward"])
    ctol = float(row["coef_tolerance"])
    assert fit["gcv"] == pytest.approx(float(row["gcv"]), rel=ctol)
    coef = [float(r["coefficient"]) for r in read_rows("fda_coef.csv") if r["case"] == row["case"]]
    np.testing.assert_allclose(fit["coef"], coef, rtol=ctol, atol=ctol)
    pred = [r for r in read_rows("fda_predict.csv") if r["case"] == row["case"]]
    ptol = float(row["prediction_tolerance"])
    np.testing.assert_allclose(fda_predict(fit, x * 1.01), [float(r["posterior"]) for r in pred],
                               rtol=ptol, atol=ptol)
    calibrated = case_fit(fda_input, row, calibrate=True)
    assert calibrated["converged"]
    np.testing.assert_allclose(fda_predict(calibrated, x * 1.01),
                               [float(r["probability"]) for r in pred], rtol=ptol, atol=ptol)


def test_the_case_weights_move_the_scores_and_not_the_basis(fda_input):
    x, y = fda_input["x"]["columns"], fda_input["y"]
    weighted = fda_fit(x, y, fda_input["w"], calibrate=False)
    plain = fda_fit(x, y, np.ones(len(y)), calibrate=False)
    assert hinge_terms(weighted) == hinge_terms(plain)
    assert not np.allclose(weighted["coef"], plain["coef"])


@pytest.mark.parametrize("degree", [1, 2])
def test_the_thread_count_does_not_change_what_the_discriminant_returns(fda_input, degree):
    x, y, w = fda_input["x"]["squares"], fda_input["y"], fda_input["w"]
    one = fda_fit(x, y, w, degree=degree, threads=1)
    four = fda_fit(x, y, w, degree=degree, threads=4)
    for key in one:
        np.testing.assert_array_equal(np.asarray(four[key]), np.asarray(one[key]))


def test_discriminant_refuses_settings_and_responses_it_has_no_fit_for():
    with pytest.raises(ValueError, match="whole number"):
        discriminant(degree=0)
    with pytest.raises(ValueError, match="whole number"):
        discriminant(max_terms=2)
    with pytest.raises(ValueError, match="zero or more"):
        discriminant(penalty=-1)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        discriminant(min_gain=1.0)
    with pytest.raises(ValueError, match="True or False"):
        discriminant(calibrate="yes")
    x = np.array([1.0, 3, 2, 5, 4, 6]).reshape(-1, 1)
    with pytest.raises(Exception, match="presences from absences"):
        fda_fit(x, np.array([0.0, 1, 0, 2, 0, 1]), np.ones(6))
    with pytest.raises(Exception, match="both classes"):
        fda_fit(x, np.ones(6), np.ones(6))
    with pytest.raises(Exception, match="above zero"):
        fda_fit(x, np.array([0.0, 1, 0, 1, 0, 1]), np.array([1.0, 1, 0, 1, 1, 1]))
    fit = fda_fit(x, np.array([0.0, 1, 0, 1, 0, 1]), np.ones(6))
    with pytest.raises(Exception, match="fitted on 1 columns"):
        fda_predict(fit, np.column_stack([x, x]))


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


@pytest.mark.parametrize("learner", [discriminant(), discriminant(degree=2),
                                     discriminant(calibrate=False), discriminant(prune=False)],
                         ids=["additive", "degree2", "posterior", "unpruned"])
def test_discriminant_fits_and_round_trips(learner):
    x, y = planted()
    fit = fit_learner(learner, x, y)
    p = fit.predict(x)
    assert p.shape == (len(y.units), 2)
    assert np.all((p >= 0) & (p <= 1))
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(fit)).predict(x), p)


def test_a_constant_response_is_its_mean_and_named_unfitted():
    x, y = planted()
    values = y.values.copy()
    values[:, 0] = 0.0
    flat = Response(values, y.units, y.variables)
    fit = fit_learner(discriminant(), x, flat)
    assert set(fit.predict(x)[:, 0]) == {0.0}
    assert fit.model["unfitted"] == ["sp1"]
