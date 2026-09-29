"""The envelope on the core, against biomod2's own ``bm_SRE()``, and the learner above it.

The reference is biomod2's, on the weekly columns maxnet's fixtures read: the bounds, pinned to
rounding, and the projection onto the fixture's rows and onto every reading scaled by 1.02,
asserted exactly. The R suite reads the same files.
"""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np
import pytest

from timesift import Response, envelope, fit_learner, grain_matrix
from timesift._envelope import envelope_fit, envelope_predict

FIXTURES = Path(__file__).resolve().parents[2] / "inst" / "spec" / "fixtures"
HELD = ("unit", "y_gaussian", "y_binomial", "w", "fold")


def read_rows(name):
    with open(FIXTURES / name, newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def envelope_input():
    rows = read_rows("penalised_input.csv")
    columns = [c for c in rows[0] if c not in HELD and not c.endswith("^2")]
    thin = {r["unit"]: r for r in read_rows("maxnet_response.csv")}
    return dict(
        x=np.asfortranarray([[float(r[c]) for c in columns] for r in rows]),
        y=dict(y_binomial=np.array([float(r["y_binomial"]) for r in rows]),
               y_8=np.array([float(thin[r["unit"]]["y_8"]) for r in rows]),
               y_12=np.array([float(thin[r["unit"]]["y_12"]) for r in rows])))


@pytest.mark.parametrize("row", read_rows("envelope_cases.csv"), ids=lambda r: r["case"])
def test_the_core_draws_the_envelope_biomod2s_bm_sre_draws(envelope_input, row):
    x = envelope_input["x"]
    fit = envelope_fit(x, envelope_input["y"][row["response"]], float(row["quantile"]))
    bounds = [r for r in read_rows("envelope_bounds.csv") if r["case"] == row["case"]]
    tol = float(row["bound_tolerance"])
    assert fit["n_presence"] == int(row["n_presence"])
    np.testing.assert_allclose(fit["lo"], [float(r["lo"]) for r in bounds], rtol=tol)
    np.testing.assert_allclose(fit["hi"], [float(r["hi"]) for r in bounds], rtol=tol)
    pred = [r for r in read_rows("envelope_predict.csv") if r["case"] == row["case"]]
    np.testing.assert_array_equal(envelope_predict(fit, x), [float(r["inside"]) for r in pred])
    np.testing.assert_array_equal(envelope_predict(fit, x * 1.02),
                                  [float(r["inside_out"]) for r in pred])


def test_a_quantile_of_zero_keeps_every_presence_ends_included_and_one_half_the_medians():
    x = np.array([[1.0, 10], [2, 20], [3, 30], [4, 40], [5, 50]])
    y = np.array([0.0, 1, 1, 1, 0])
    fit = envelope_fit(x, y, 0.0)
    np.testing.assert_array_equal(fit["lo"], [2, 20])
    np.testing.assert_array_equal(fit["hi"], [4, 40])
    np.testing.assert_array_equal(envelope_predict(fit, x), [0, 1, 1, 1, 0])
    mid = envelope_fit(x, y, 0.5)
    np.testing.assert_array_equal(mid["lo"], mid["hi"])
    np.testing.assert_array_equal(envelope_predict(mid, x), [0, 0, 1, 0, 0])
    # Type 7 interpolates between neighbours: over 2, 3, 4 the 0.25 quantile is 2.5.
    np.testing.assert_allclose(envelope_fit(x, y, 0.25)["lo"], [2.5, 25])


def test_the_envelope_refuses_what_it_has_no_envelope_for():
    x = np.arange(1.0, 5.0).reshape(-1, 1)
    with pytest.raises(Exception, match="at least one presence"):
        envelope_fit(x, np.zeros(4), 0.025)
    with pytest.raises(Exception, match="zero and one"):
        envelope_fit(x, np.array([0.0, 1, 2, 0]), 0.025)
    with pytest.raises(Exception, match=r"\[0, 0.5\]"):
        envelope_fit(x, np.array([0.0, 1, 1, 0]), 0.6)
    fit = envelope_fit(x, np.array([0.0, 1, 1, 0]), 0.025)
    with pytest.raises(Exception, match="drawn over 1 columns"):
        envelope_predict(fit, np.column_stack([x, x]))
    with pytest.raises(ValueError, match=r"\[0, 0.5\]"):
        envelope(quantile=0.7)


def planted(n_unit=60, days=120, seed=17):
    """A record whose monthly level carries the response, and nothing else does."""
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


def test_the_envelope_fits_predicts_zero_or_one_ignores_the_weights_and_round_trips(
        temporary_response):
    from timesift.response import PRESENCE_ABSENCE
    x, y = planted()
    fit = fit_learner(envelope(quantile=0.05), x, y)
    p = fit.predict(x)
    assert p.shape == (len(y.units), 2)
    assert set(np.unique(p)) <= {0.0, 1.0}
    assert p[y.values[:, 0] == 1, 0].mean() > p[y.values[:, 0] == 0, 0].mean()
    temporary_response("unweighted_test", {k: v for k, v in PRESENCE_ABSENCE.items()
                                           if k != "weights"})
    plain = fit_learner(envelope(quantile=0.05), x, y, response="unweighted_test").predict(x)
    np.testing.assert_array_equal(plain, p)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(fit)).predict(x), p)


def test_the_envelope_needs_a_presence_absence_head_and_one_value_is_its_share(
        temporary_response):
    from timesift.response import as_response, scorable_cells
    temporary_response("continuous_test", dict(
        prepare=as_response, activation="identity", loss="squared_error", metric="roc_auc",
        cells=lambda y, folds: scorable_cells(y, folds)))
    x, y = planted(n_unit=40, seed=91)
    level = x.values[:, :, 0].mean(axis=1).reshape(-1, 1)
    with pytest.raises(ValueError, match="presences"):
        fit_learner(envelope(), x, Response(level, y.units, ("height",)),
                    response="continuous_test")
    fit = fit_learner(envelope(), x, Response(np.zeros((40, 1)), y.units, ("absent",)))
    assert fit.model["unfitted"] == ["absent"]
    np.testing.assert_array_equal(np.unique(fit.predict(x)), [0.0])
