"""The penalised core, against the reference the fixtures carry.

These are not digests. Two implementations of a coordinate descent settle at the same point to the
tolerance they are run at, never to the last bit, so what is asserted is the distance from the
reference. The reference is glmnet's, which is what the arm the networks are measured against has
always been, and the R suite reads the same three files.
"""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np
import pytest

from timesift import Response, elasticnet, fit_learner, grain_matrix
from timesift.penalised import (penalised_coef, penalised_cv, penalised_path, penalised_predict)

FIXTURES = Path(__file__).resolve().parents[2] / "inst" / "spec" / "fixtures"
HELD = ("unit", "y_gaussian", "y_binomial", "w", "fold")


def read_rows(name):
    with open(FIXTURES / name, newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def penalised_input():
    rows = read_rows("penalised_input.csv")
    columns = [c for c in rows[0] if c not in HELD]
    return dict(
        columns=columns,
        units=[r["unit"] for r in rows],
        x=np.asfortranarray([[float(r[c]) for c in columns] for r in rows]),
        gaussian=np.array([float(r["y_gaussian"]) for r in rows]),
        binomial=np.array([float(r["y_binomial"]) for r in rows]),
        w=np.array([float(r["w"]) for r in rows]),
        fold=np.array([int(r["fold"]) for r in rows], dtype=np.int32))


def case_response(data, row):
    y = data["binomial"] if row["family"] == "binomial" else data["gaussian"]
    w = data["w"] if row["weighted"] == "TRUE" else np.ones(len(y))
    return y, w


def objective(x, y, w, family, alpha, lam, a0, beta):
    """The objective both descents minimise, on the scale the reference states it: the mean
    deviance halved for a Gaussian family and the mean negative log likelihood for a binomial
    one, plus the penalty."""
    wn = w / w.sum()
    eta = a0 + x @ beta
    fit = (np.sum(wn * (y - eta) ** 2) / 2 if family == "gaussian"
           else -np.sum(wn * (y * eta - np.log1p(np.exp(eta)))))
    return fit + lam * (alpha * np.abs(beta).sum() + (1 - alpha) / 2 * (beta ** 2).sum())


def test_the_design_the_penalised_fixtures_carry_is_a_representation(penalised_input):
    # Every column names the bin and the channel it came from, or is the square of one that does.
    assert all(c.startswith("mean@") for c in penalised_input["columns"])
    assert len(penalised_input["columns"]) > 8
    assert len(set(penalised_input["fold"].tolist())) == 5


@pytest.mark.parametrize("row", read_rows("penalised_cases.csv"),
                         ids=lambda r: r["case"])
def test_the_penalised_path_reproduces_the_reference_glmnet_gives(penalised_input, row):
    data = penalised_input
    y, w = case_response(data, row)
    alpha, tol = float(row["alpha"]), float(row["tolerance"])
    fit = penalised_path(data["x"], y, w, row["family"], alpha, thresh=float(row["thresh"]),
                         max_pass=float(row["max_pass"]))
    want = [r for r in read_rows("penalised_path.csv") if r["case"] == row["case"]]
    lengths = {r["case"]: int(r["n_point"]) for r in read_rows("penalised_cv.csv")}
    assert len(fit["lambda_"]) == lengths[row["case"]]
    scale = max(1.0, float(np.abs(fit["beta"]).max()))
    for r in want:
        k = int(r["point"]) - 1
        assert fit["lambda_"][k] == pytest.approx(float(r["lambda"]), rel=1e-10)
        assert fit["a0"][k] == pytest.approx(float(r["a0"]), abs=tol * scale)
        beta = np.array([float(r[f"b{j + 1}"]) for j in range(fit["beta"].shape[0])])
        assert fit["beta"][:, k] == pytest.approx(beta, abs=tol * scale)
        # Sitting below the reference on the objective is the fit being closer to the optimum
        # than the reference is, which is the direction the arm is allowed to move in; sitting
        # above it is the arm being weakened.
        ours = objective(data["x"], y, w, row["family"], alpha, fit["lambda_"][k], fit["a0"][k],
                         fit["beta"][:, k])
        assert ours - float(r["objective"]) < float(row["objective_tolerance"])


@pytest.mark.parametrize("row", read_rows("penalised_cases.csv"), ids=lambda r: r["case"])
def test_the_cross_validated_penalty_is_the_one_glmnet_chooses(penalised_input, row):
    data = penalised_input
    y, w = case_response(data, row)
    tol = float(row["tolerance"])
    fit = penalised_cv(data["x"], y, w, row["family"], float(row["alpha"]), data["fold"], 5,
                       thresh=float(row["thresh"]), max_pass=float(row["max_pass"]))
    want = next(r for r in read_rows("penalised_cv.csv") if r["case"] == row["case"])
    assert fit["lambda_min"] == pytest.approx(float(want["lambda_min"]), rel=1e-10)
    assert fit["lambda_1se"] == pytest.approx(float(want["lambda_1se"]), rel=1e-10)
    assert fit["cv_mean"].min() == pytest.approx(float(want["cv_min"]), rel=tol)
    assert fit["cv_sd"][int(np.argmin(fit["cv_mean"]))] == pytest.approx(
        float(want["cv_sd_min"]), rel=tol)


def test_the_paths_first_point_leaves_every_coefficient_at_zero_above_a_ridge(penalised_input):
    # The largest penalty of the path is the smallest that holds every coefficient down, so an
    # elastic net starts from the null model. A ridge has no threshold to cross and starts from
    # the solution at that penalty, which is the one point of the path glmnet reports
    # differently: it fits that one at a penalty of 9.9e35 and labels it with the smallest that
    # would have done.
    data = penalised_input
    w = np.ones(len(data["binomial"]))
    fit = penalised_path(data["x"], data["binomial"], w, "binomial", 0.5)
    assert np.all(fit["beta"][:, 0] == 0)
    assert fit["df"][0] == 0
    ridge = penalised_path(data["x"], data["binomial"], w, "binomial", 0.0)
    assert np.any(ridge["beta"][:, 0] != 0)
    assert np.abs(ridge["beta"][:, 0]).max() < 1e-2


def test_a_fit_is_read_at_a_named_penalty_at_one_of_its_own_and_refuses_anything_else(
        penalised_input):
    data = penalised_input
    w = np.ones(len(data["binomial"]))
    fit = penalised_cv(data["x"], data["binomial"], w, "binomial", 0.5, data["fold"], 5)
    at_min = penalised_predict(fit, data["x"], "lambda.min")
    assert at_min == pytest.approx(penalised_predict(fit, data["x"], fit["lambda_min"]))
    assert not np.allclose(at_min, penalised_predict(fit, data["x"], "lambda.1se"))
    assert np.all((at_min > 0) & (at_min < 1))
    # A penalty between two points of the path is read between their coefficients, so it sits
    # between the two fits rather than at either.
    between = (fit["lambda_"][2] + fit["lambda_"][3]) / 2
    mid = penalised_coef(fit, between)
    low = np.minimum(penalised_coef(fit, fit["lambda_"][2]), penalised_coef(fit,
                                                                           fit["lambda_"][3]))
    assert np.all(mid >= low - 1e-12)
    assert len(mid) == data["x"].shape[1] + 1
    with pytest.raises(ValueError, match="lambda.min"):
        penalised_predict(fit, data["x"], "lambda.best")
    # A path carries no cross-validation, so it has no named penalty to be read at.
    path = penalised_path(data["x"], data["binomial"], w, "binomial", 0.5)
    with pytest.raises(ValueError, match="cross-validation"):
        penalised_predict(path, data["x"], "lambda.min")


def test_the_penalised_core_says_what_it_cannot_fit(penalised_input):
    data = penalised_input
    x, w = data["x"], np.ones(len(data["binomial"]))
    n = len(w)
    with pytest.raises(ValueError, match="one outcome"):
        penalised_path(x, np.ones(n), w, "binomial", 0.5)
    with pytest.raises(ValueError, match="zero and one"):
        penalised_path(x, data["gaussian"], w, "binomial", 0.5)
    with pytest.raises(ValueError, match="one value"):
        penalised_path(x, np.full(n, 2.0), w, "gaussian", 0.5)
    with pytest.raises(ValueError, match="gaussian"):
        penalised_path(x, data["binomial"], w, "poisson", 0.5)
    with pytest.raises(ValueError, match="between zero and one"):
        penalised_path(x, data["binomial"], w, "binomial", 2.0)
    with pytest.raises(ValueError, match="negative"):
        penalised_path(x, data["binomial"], -w, "binomial", 0.5)
    with pytest.raises(ValueError, match="every unit or none"):
        penalised_cv(x, data["binomial"], w, "binomial", 0.5,
                     np.zeros(n, dtype=np.int32), 5)


def test_a_column_holding_one_value_is_carried_through_at_zero(penalised_input):
    data = penalised_input
    w = np.ones(len(data["binomial"]))
    x = np.hstack([data["x"], np.full((data["x"].shape[0], 1), 3.0)])
    fit = penalised_path(x, data["binomial"], w, "binomial", 0.5)
    assert np.all(fit["beta"][-1, :] == 0)
    # The column changes nothing else about the fit, which is what says it was left out rather
    # than penalised to zero by a path the rest of the columns then had to share.
    without = penalised_path(data["x"], data["binomial"], w, "binomial", 0.5)
    assert fit["lambda_"] == pytest.approx(without["lambda_"])
    assert fit["beta"][:-1, :] == pytest.approx(without["beta"])


def planted(n_unit=60, days=56, noise=1.0, seed=17):
    """A record whose weekly level carries the response, and nothing else does."""
    rng = np.random.default_rng(seed)
    t = np.datetime64("2021-09-01T00:00:00", "s") + np.arange(24 * days) * np.timedelta64(1, "h")
    units = [f"p{i:02d}" for i in range(n_unit)]
    warmth = rng.normal(size=n_unit)
    value = np.concatenate([w * 2.0 + rng.normal(0, noise, len(t)) for w in warmth])
    d = {"id": [u for u in units for _ in range(len(t))],
         "time": list(t) * n_unit, "value": list(value)}
    y = rng.binomial(1, 1 / (1 + np.exp(-3 * np.column_stack([warmth, -warmth])))).astype(float)
    return grain_matrix(d, "id", "time", "value", grain="week"), Response(
        y, tuple(units), ("sp1", "sp2"))


def test_a_cross_validation_on_threads_returns_what_one_on_a_single_thread_returns(
        penalised_input):
    # The whole-unit path and each fold's path are one independent fit each, reading the design
    # and sharing nothing, so running them at once is a scheduling decision and not a numerical
    # one. Two threads rather than more: a package's own tests do not take a machine's cores.
    data = penalised_input
    w = np.ones(len(data["binomial"]))
    serial = penalised_cv(data["x"], data["binomial"], w, "binomial", 0.5, data["fold"], 5)
    threaded = penalised_cv(data["x"], data["binomial"], w, "binomial", 0.5, data["fold"], 5,
                            threads=2)
    for key in ("lambda_", "a0", "beta", "cv_mean", "cv_sd", "df", "dev_ratio"):
        assert np.array_equal(threaded[key], serial[key]), key
    assert threaded["lambda_min"] == serial["lambda_min"]
    assert threaded["lambda_1se"] == serial["lambda_1se"]


def test_the_penalised_learner_fits_over_the_core_and_carries_no_fitter_of_its_own():
    x, y = planted()
    learner = elasticnet()
    assert learner.needs == ()
    fit = fit_learner(learner, x, y)
    assert all(isinstance(f, dict) and "lambda_min" in f for f in fit.model["models"])
    p = fit.predict(x)
    assert p.shape == (len(y.units), 2)
    assert np.all((p > 0) & (p < 1))
    # A fit is arrays, so it pickles and predicts the same afterwards.
    assert pickle.loads(pickle.dumps(fit)).predict(x) == pytest.approx(p)


def test_the_penalised_learner_reads_the_penalty_its_s_names():
    x, y = planted()
    at_min = fit_learner(elasticnet(), x, y).predict(x)
    at_1se = fit_learner(elasticnet(s="lambda.1se"), x, y).predict(x)
    assert not np.allclose(at_min, at_1se)
    # The larger penalty shrinks harder, so its predictions sit closer to the prevalence.
    assert at_1se.std() < at_min.std()
