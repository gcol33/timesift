"""The boosting core, against gbm, xgboost and the spec's own draws, and the learner above it.

gbm is the reference for the first-order trees wherever nothing is drawn, its cross-validation
included; xgboost for the second-order ones, to its single-precision storage; and the forest grown
from the spec's text in R alone for the subsample and the column draw. The R suite reads the same
files.
"""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np
import pytest

from counts import fixture_counts
from timesift import Response, boosting, fit_learner, grain_matrix
from timesift._tree import boost_fit, boost_predict
from timesift.learners import _boost_settings, flatten
from timesift.metrics import roc_auc

FIXTURES = Path(__file__).resolve().parents[2] / "inst" / "spec" / "fixtures"
HELD = ("unit", "y_gaussian", "y_binomial", "w", "fold")


def read_rows(name):
    with open(FIXTURES / name, newline="") as handle:
        return list(csv.DictReader(handle))


CASES = read_rows("boost_cases.csv")


@pytest.fixture(scope="module")
def boost_input():
    rows = read_rows("penalised_input.csv")
    columns = [c for c in rows[0] if c not in HELD]
    counts = {r["unit"]: float(r["count"]) for r in read_rows("tree_weights.csv")}
    random = {r["unit"]: float(r["weight"]) for r in read_rows("boost_weights.csv")}
    n = len(rows)
    return dict(
        x=np.asfortranarray([[float(r[c]) for c in columns] for r in rows]),
        binomial=np.array([float(r["y_binomial"]) for r in rows]),
        gaussian=np.array([float(r["y_gaussian"]) for r in rows]),
        poisson=fixture_counts(FIXTURES, [r["unit"] for r in rows]),
        fold=np.array([int(r["fold"]) for r in rows], dtype=np.int32),
        weights=dict(flat=np.ones(n), counts=np.array([counts[r["unit"]] for r in rows]),
                     random=np.array([random[r["unit"]] for r in rows])))


def case_fit(data, row):
    cross = row["cv"] == "1"
    return boost_fit(data["x"], data[row["family"]], data["weights"][row["weights"]],
                     row["family"], int(row["trees"]), int(row["depth"]), float(row["shrinkage"]),
                     float(row["min_leaf"]), float(row["subsample"]), float(row["colsample"]),
                     row["newton"] == "1", float(row["lambda"]), float(row["gamma"]),
                     int(row["seed"]), data["fold"] if cross else None, 5 if cross else 0)


@pytest.mark.parametrize("row", CASES, ids=lambda r: r["case"])
def test_boosting_predicts_what_gbm_xgboost_and_the_specs_draws_predict(boost_input, row):
    ref = [float(r["value"]) for r in read_rows("boost_predict.csv") if r["case"] == row["case"]]
    tolerance = 1e-5 if row["reference"] == "xgboost" else 1e-12
    np.testing.assert_allclose(boost_predict(case_fit(boost_input, row), boost_input["x"]), ref,
                               rtol=tolerance, atol=tolerance)


def test_the_inner_cross_validation_reads_gbms_held_out_error_and_keeps_gbms_number(boost_input):
    rows = read_rows("boost_cv.csv")
    for case in sorted({r["case"] for r in rows}):
        row = next(c for c in CASES if c["case"] == case)
        fit = case_fit(boost_input, row)
        ref = np.array([float(r["cv_error"]) for r in rows if r["case"] == case])
        np.testing.assert_allclose(fit["cv_error"], ref, rtol=1e-12, err_msg=case)
        assert len(fit["offset"]) - 1 == int(np.argmin(ref)) + 1, case


def test_a_boosted_fit_is_the_same_on_any_number_of_threads():
    rng = np.random.default_rng(91)
    x = rng.normal(size=(80, 5))
    y = (x[:, 0] + rng.normal(size=80) > 0).astype(float)
    fold = np.resize(np.arange(4, dtype=np.int32), 80)
    one = boost_fit(x, y, np.ones(80), "binomial", 30, 2, 0.1, 3, 0.5, 1, False, 0, 0, 5, fold, 4)
    many = boost_fit(x, y, np.ones(80), "binomial", 30, 2, 0.1, 3, 0.5, 1, False, 0, 0, 5, fold,
                     4, threads=3)
    for field in ("offset", "column", "threshold", "left", "right", "value", "cv_error"):
        np.testing.assert_array_equal(many[field], one[field])


def test_the_boosting_core_refuses_what_it_cannot_fit():
    x = np.array([[1.0], [2.0], [3.0], [4.0]])

    def fit(y, w=np.ones(4), **kw):
        args = dict(trees=2, depth=1, shrinkage=0.1, min_leaf=1, subsample=1, colsample=1,
                    newton=False, lambda_=0, gamma=0, seed=1)
        args.update(kw)
        return boost_fit(x, np.array(y, dtype=float), w, "binomial", **args)

    with pytest.raises(ValueError, match="0 and 1 alone"):
        fit([0, 1, 0, 2])
    with pytest.raises(ValueError, match="both classes"):
        fit([0, 1, 0, 1], np.array([1, 0, 1, 0.0]))
    with pytest.raises(ValueError, match="subsample"):
        fit([0, 1, 0, 1], subsample=1.5)
    with pytest.raises(ValueError, match="holds no observation"):
        fit([0, 1, 0, 1], subsample=0.1)
    with pytest.raises(ValueError, match="depth"):
        fit([0, 1, 0, 1], depth=0)


def test_a_preset_fills_the_settings_left_open_as_gbm_xgboost_and_biomod2_have_them():
    p = _boost_settings("package", "gbm", *[None] * 9)
    assert (p["trees"], p["depth"], p["shrinkage"], p["min_leaf"], p["subsample"],
            p["n_inner"]) == (100, 1, 0.1, 10, 0.5, 0)
    b = _boost_settings("bigboss", "gbm", *[None] * 9)
    assert (b["trees"], b["depth"], b["shrinkage"], b["min_leaf"], b["n_inner"]) == (
        2500, 7, 0.001, 5, 3)
    xg = _boost_settings("package", "xgboost", *[None] * 9)
    assert (xg["trees"], xg["depth"], xg["shrinkage"], xg["lambda_"], xg["subsample"]) == (
        100, 6, 0.3, 1, 1)
    xb = _boost_settings("bigboss", "xgboost", *[None] * 9)
    assert (xb["trees"], xb["depth"], xb["shrinkage"]) == (4, 2, 1)
    with pytest.raises(ValueError, match="method=.xgboost."):
        boosting(lambda_=1)


def planted(n_unit=60, days=56, seed=17):
    """A record whose weekly level carries the response, and nothing else does."""
    rng = np.random.default_rng(seed)
    t = np.datetime64("2021-09-01T00:00:00", "s") + np.arange(24 * days) * np.timedelta64(1, "h")
    units = [f"p{i:02d}" for i in range(n_unit)]
    warmth = rng.normal(size=n_unit)
    value = np.concatenate([w * 2.0 + rng.normal(0, 1.0, len(t)) for w in warmth])
    d = {"id": [u for u in units for _ in range(len(t))],
         "time": list(t) * n_unit, "value": list(value)}
    y = rng.binomial(1, 1 / (1 + np.exp(-3 * np.column_stack([warmth, -warmth])))).astype(float)
    x = grain_matrix(d, "id", "time", "value", grain="week")
    return x, Response(y, tuple(units), ("sp1", "sp2"))


def test_boosting_fits_predicts_survives_a_round_trip_and_reads_the_flattened_columns():
    x, y = planted()
    fit = fit_learner(boosting(), x, y)
    p = fit.predict(x)
    assert p.shape == (len(y.units), 2)
    assert np.all((p > 0) & (p < 1))
    assert roc_auc(y.values[:, 0], p[:, 0]) > 0.75
    assert fit.model["n_col"] == flatten(x).shape[1]
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(fit)).predict(x), p)
    second = fit_learner(boosting(method="xgboost", trees=30), x, y).predict(x)
    assert roc_auc(y.values[:, 0], second[:, 0]) > 0.75
    cv = fit_learner(boosting(trees=60, n_inner=3), x, y).model["models"][0]
    assert len(cv["cv_error"]) == 60
    assert len(cv["offset"]) - 1 == int(np.argmin(cv["cv_error"])) + 1


def test_boosting_fits_the_family_the_response_heads_loss_names_and_a_constant_is_its_mean(
        temporary_response):
    from timesift.response import as_response, scorable_cells
    temporary_response("continuous_test", dict(
        prepare=as_response, activation="identity", loss="squared_error", metric="roc_auc",
        cells=lambda y, folds: scorable_cells(y, folds)))
    x, y = planted(seed=91)
    level = x.values[:, :, 0].mean(axis=1)
    level = 10 + 3 * (level - level.mean()) / level.std()
    fit = fit_learner(boosting(depth=2, min_leaf=3), x,
                      Response(level.reshape(-1, 1), y.units, ("height",)),
                      response="continuous_test")
    assert fit.model["family"] == "gaussian"
    assert np.corrcoef(fit.predict(x)[:, 0], level)[0, 1] > 0.8

    flat = Response(np.zeros((len(y.units), 1)), y.units, ("absent",))
    assert np.unique(fit_learner(boosting(), x, flat).predict(x)).tolist() == [0.0]
