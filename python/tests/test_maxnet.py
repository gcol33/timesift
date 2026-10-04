"""maxnet's features, regularisation and fit on the core, against the maxnet package's own, and the
learner above them.

The reference is maxnet's, on the weekly columns the penalised fixtures carry. The features and
their penalty factors are asserted to rounding. The fit is asserted by the objective at the
reference's penalty, which two descents run to the same tolerance reach alike, and by the
predictions, which collinear hinges leave slightly less determined. The R suite reads the same
files.
"""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np
import pytest

from timesift import Response, fit_learner, grain_matrix, maxent
from timesift.learners import flatten
from timesift.metrics import tss
from timesift._maxnet import maxnet_design, maxnet_fit, maxnet_predict

FIXTURES = Path(__file__).resolve().parents[2] / "inst" / "spec" / "fixtures"
HELD = ("unit", "y_gaussian", "y_binomial", "w", "fold")


def read_rows(name):
    with open(FIXTURES / name, newline="") as handle:
        return list(csv.DictReader(handle))


CASES = read_rows("maxnet_cases.csv")


def flag(value) -> bool:
    return value == "TRUE"


@pytest.fixture(scope="module")
def maxnet_input():
    rows = read_rows("penalised_input.csv")
    columns = [c for c in rows[0] if c not in HELD and not c.endswith("^2")]
    thin = {r["unit"]: r for r in read_rows("maxnet_response.csv")}
    return dict(
        x=np.asfortranarray([[float(r[c]) for c in columns] for r in rows]),
        y=dict(y_binomial=np.array([float(r["y_binomial"]) for r in rows]),
               y_8=np.array([float(thin[r["unit"]]["y_8"]) for r in rows]),
               y_12=np.array([float(thin[r["unit"]]["y_12"]) for r in rows])),
        w=np.array([float(r["w"]) for r in rows]),
        fold=np.array([int(r["fold"]) for r in rows], dtype=np.int32))


def case_input(data, row):
    """The case's input: the fixture's columns and response, with the first presence's readings
    appended once more as an absence where the case asks for it."""
    x, y = data["x"], data["y"][row["response"]]
    if flag(row["duplicate"]):
        x = np.asfortranarray(np.vstack([x, x[np.flatnonzero(y == 1)[0]]]))
        y = np.append(y, 0.0)
    w = data["w"] if flag(row["weighted"]) else np.ones(len(y))
    return x, y, w


def case_design(row, x, y):
    return maxnet_design(x, y, classes=row["classes"] or None, regmult=float(row["regmult"]),
                         formulation=row["formulation"], add_samples=flag(row["add_samples"]))


def case_fit(data, row, x, y, w):
    absence = row["formulation"] == "absence"
    return maxnet_fit(x, y, w, classes=row["classes"] or None, regmult=float(row["regmult"]),
                      formulation=row["formulation"], add_samples=flag(row["add_samples"]),
                      thresh=float(row["thresh"]), max_pass=float(row["max_pass"]),
                      fold=data["fold"] if absence else None, n_fold=5 if absence else 0)


def keys(f):
    return [f"{k} {a} {b} {lo!r} {hi!r}"
            for k, a, b, lo, hi in zip(f["kind"], f["a"], f["b"], f["lo"], f["hi"])]


def objective(fit, design, x, w, row) -> float:
    """The objective both implementations minimise, on glmnet's scale, at the fit's own penalty:
    the weighted mean negative log likelihood over the rows fitted, background rows included, plus
    the penalty with the factors rescaled to sum to the feature count."""
    xa = np.asfortranarray(x[design["rows"]])
    ya = design["y"]
    weight = ya + (1 - ya) * 100 if row["formulation"] == "background" else w
    wn = weight / weight.sum()
    eta = (fit["lasso_intercept"] + maxnet_predict(fit, xa, clamp=False, type="link")
           - fit["intercept"])
    reg = design["reg"]
    vp = reg * len(reg) / reg.sum()
    where = {k: i for i, k in enumerate(keys(design))}
    at = [where[k] for k in keys(fit)]
    return float(-np.sum(wn * (ya * eta - np.logaddexp(0.0, eta)))
                 + fit["lambda"] * np.sum(vp[at] * np.abs(fit["beta"])))


def reference(name, case):
    return [r for r in read_rows(name) if r["case"] == case]


@pytest.mark.parametrize("row", CASES, ids=lambda r: r["case"])
def test_the_core_builds_maxnets_features_and_regularises_each_as_maxnet_does(maxnet_input, row):
    x, y, _ = case_input(maxnet_input, row)
    design = case_design(row, x, y)
    assert design["classes"] == row["classes_used"]
    assert len(design["rows"]) == int(row["n_row"])
    assert len(design["reg"]) == int(row["n_feature"])
    ref = [float(r["reg"]) for r in reference("maxnet_regularization.csv", row["case"])]
    np.testing.assert_allclose(design["reg"], ref, rtol=float(row["reg_tolerance"]), atol=0)


@pytest.mark.parametrize("row", CASES, ids=lambda r: r["case"])
def test_the_cores_maxnet_reaches_the_objective_maxnets_own_glmnet_fit_does(maxnet_input, row):
    x, y, w = case_input(maxnet_input, row)
    design = case_design(row, x, y)
    fit = case_fit(maxnet_input, row, x, y, w)
    assert fit["stalled"] == 0
    np.testing.assert_allclose(fit["lambda"], float(row["lambda"]), rtol=1e-10)
    np.testing.assert_allclose(objective(fit, design, x, w, row), float(row["objective"]),
                               rtol=0, atol=float(row["objective_tolerance"]))
    if row["formulation"] == "background":
        tol = float(row["prediction_tolerance"])
        np.testing.assert_allclose(fit["entropy"], float(row["entropy"]), rtol=tol)
        np.testing.assert_allclose(fit["intercept"], float(row["alpha"]), rtol=tol)


@pytest.mark.parametrize("row", CASES, ids=lambda r: r["case"])
def test_the_cores_maxnet_predicts_what_maxnets_own_fit_does_clamped_outside_the_range(
        maxnet_input, row):
    x, y, w = case_input(maxnet_input, row)
    fit = case_fit(maxnet_input, row, x, y, w)
    ref = reference("maxnet_predict.csv", row["case"])
    tol = float(row["prediction_tolerance"])

    def expected(field):
        return np.array([float(r[field]) for r in ref])

    np.testing.assert_allclose(maxnet_predict(fit, x, type="logistic"), expected("logistic"),
                               rtol=tol, atol=tol)
    if row["formulation"] == "background":
        out = np.asfortranarray(x * 1.3)
        np.testing.assert_allclose(maxnet_predict(fit, x, type="cloglog"), expected("cloglog"),
                                   rtol=tol, atol=tol)
        np.testing.assert_allclose(maxnet_predict(fit, out, type="cloglog"),
                                   expected("cloglog_out"), rtol=tol, atol=tol)
        np.testing.assert_allclose(maxnet_predict(fit, out, type="logistic"),
                                   expected("logistic_out"), rtol=tol, atol=tol)


def test_maxnets_classes_follow_the_presence_count_and_its_knots_its_own_seq():
    assert maxnet_design(np.array([[1.0], [2], [3], [4]]),
                         np.array([1.0, 1, 0, 0]))["classes"] == "l"
    x = np.column_stack([[0.0, 1, 2, 3], [5.0, 5, 5, 5]])
    design = maxnet_design(x, np.array([1.0, 0, 1, 0]), classes="lqht")
    # The constant second column takes no feature; the first takes 1 + 1 + 98 + 49.
    assert len(design["reg"]) == 149
    assert np.all(design["a"] == 0)
    hinge = design["kind"] == 2
    knots = np.linspace(0, 3, 50)
    np.testing.assert_allclose(design["hi"][hinge][:49], 3.0)
    np.testing.assert_allclose(design["lo"][hinge][:49], knots[:49], rtol=1e-15)
    np.testing.assert_allclose(design["hi"][hinge][49:], knots[1:], rtol=1e-15)
    np.testing.assert_allclose(design["lo"][design["kind"] == 3], np.linspace(0, 3, 52)[2:51],
                               rtol=1e-15)


def test_maxnet_refuses_what_it_has_no_model_for():
    x = np.array([[1.0], [2], [3], [4]])
    ones = np.ones(4)
    with pytest.raises(Exception, match="at least two presences"):
        maxnet_fit(x, np.array([1.0, 0, 0, 0]), ones)
    with pytest.raises(Exception, match="zero and one"):
        maxnet_fit(x, np.array([1.0, 0, 2, 0]), ones)
    with pytest.raises(Exception, match="letters"):
        maxnet_fit(x, np.array([1.0, 1, 0, 0]), ones, classes="lz")
    with pytest.raises(Exception, match="positive"):
        maxnet_fit(x, np.array([1.0, 1, 0, 0]), ones, regmult=0)
    with pytest.raises(Exception, match="more than one value"):
        maxnet_fit(np.ones((4, 1)), np.array([1.0, 1, 0, 0]), ones)
    with pytest.raises(Exception, match="two folds"):
        maxnet_fit(x, np.array([1.0, 1, 0, 0]), ones, formulation="absence")
    fit = maxnet_fit(x, np.array([1.0, 1, 0, 0]), ones, formulation="absence",
                     fold=np.array([0, 1, 0, 1]), n_fold=2)
    with pytest.raises(Exception, match="background formulation"):
        maxnet_predict(fit, x, type="cloglog")
    with pytest.raises(Exception, match="columns it was fitted on"):
        maxnet_predict(fit, np.column_stack([x, x]))
    with pytest.raises(ValueError, match="letters"):
        maxent(classes="lqx")
    with pytest.raises(ValueError, match="'logistic'"):
        maxent(formulation="absence", type="cloglog")
    with pytest.raises(ValueError, match="'cloglog' or 'logistic'"):
        maxent(type="exponential")


def test_a_design_above_the_limit_is_refused_with_the_size_it_would_have_taken():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(200, 30))
    y = np.repeat([1.0, 0.0], 100)
    # 30 columns under "lqph" are 30 + 30 + 2940 + 435 features, over 200 units and the 100
    # presences added to the background: 300 x 3435 doubles.
    with pytest.raises(Exception, match="3435 features over 300 rows, a design of 0.0082 GB"):
        maxnet_fit(x, y, np.ones(200), classes="lqph", max_design=0.001)


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


@pytest.mark.parametrize("formulation", ["background", "absence"])
def test_maxnet_fits_predicts_and_survives_a_round_trip(formulation):
    x, y = planted()
    fit = fit_learner(maxent(formulation=formulation), x, y)
    p = fit.predict(x)
    assert p.shape == (len(y.units), 2)
    assert np.all((p >= 0) & (p <= 1))
    assert tss(y.values[:, 0], p[:, 0]) > 0.4
    assert fit.model["n_col"] == flatten(x).shape[1]
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(fit)).predict(x), p)


def test_the_background_formulation_ignores_case_weights_and_the_absence_one_reads_them():
    x, y = planted(seed=51)
    m = flatten(x)
    yj = y.values[:, 0]
    w = np.where(yj == 1, 5.0, 1.0)
    ones = np.ones(len(yj))
    np.testing.assert_array_equal(maxnet_fit(m, yj, ones)["beta"], maxnet_fit(m, yj, w)["beta"])
    fold = np.arange(len(yj)) % 5
    a = maxnet_fit(m, yj, ones, formulation="absence", fold=fold, n_fold=5)
    b = maxnet_fit(m, yj, w, formulation="absence", fold=fold, n_fold=5)
    assert not np.array_equal(a["beta"], b["beta"])


def test_clamping_holds_a_reading_outside_the_fitted_range_at_the_ranges_edge():
    x, y = planted(seed=61)
    m = flatten(x)
    fit = maxnet_fit(m, y.values[:, 0], np.ones(m.shape[0]), classes="lq")
    top = np.asarray(fit["var_max"]).reshape(1, -1)
    np.testing.assert_array_equal(maxnet_predict(fit, top + 10), maxnet_predict(fit, top))
    assert not np.allclose(maxnet_predict(fit, top + 10, clamp=False), maxnet_predict(fit, top))


def test_maxnet_needs_a_presence_absence_head_and_a_thin_response_is_its_share(
        temporary_response):
    from timesift.response import as_response, scorable_cells
    temporary_response("continuous_test", dict(
        prepare=as_response, activation="identity", loss="squared_error", metric="roc_auc",
        cells=lambda y, folds: scorable_cells(y, folds)))
    x, y = planted(seed=91)
    level = x.values[:, :, 0].mean(axis=1).reshape(-1, 1)
    with pytest.raises(ValueError, match="presence-absence"):
        fit_learner(maxent(), x, Response(level, y.units, ("height",)),
                    response="continuous_test")
    one = np.zeros((len(y.units), 1))
    one[0, 0] = 1.0
    fit = fit_learner(maxent(), x, Response(one, y.units, ("rare",)))
    assert fit.model["unfitted"] == ["rare"]
    np.testing.assert_allclose(np.unique(fit.predict(x)), [1 / len(y.units)])
