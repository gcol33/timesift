"""The interval for the procedure's risk: the pieces of the estimator, and what carries it."""

from __future__ import annotations

import numpy as np
import pytest

from timesift import (Learner, Response, feature_matrix, fold_map, grain_ladder, paired_contrast,
                      select_grain)
from timesift._stats import norm_ppf
from timesift.interval import jackknife_var, ncv_interval


def design(n, seed, variables=3):
    """A cheap design: each variable's driver is standard normal, the response is Bernoulli on it,
    and the two representations are that driver read through a little noise or through a lot."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, variables))
    y = rng.binomial(1, 1 / (1 + np.exp(-(-1 + 1.2 * d)))).astype(float)
    units = [f"s{seed}_{i:05d}" for i in range(n)]
    names = [f"v{j}" for j in range(variables)]
    good = d + rng.normal(0, 0.7, (n, variables))
    noisy = d + rng.normal(0, 2, (n, variables))
    x = {"good": feature_matrix(good, units, names, "good"),
         "noisy": feature_matrix(noisy, units, names, "noisy")}
    return x, Response(y, tuple(units), tuple(names))


def centroid() -> Learner:
    def fit(x, y, **_):
        m = x.values[:, :, 0]
        return dict(w=np.column_stack([m[y[:, j] == 1].mean(axis=0) - m[y[:, j] == 0].mean(axis=0)
                                       for j in range(y.shape[1])]))

    return Learner(name="centroid", fit=fit,
                   predict=lambda model, x: x.values[:, :, 0] @ model["w"])


def test_the_jackknife_variance_of_a_mean_of_per_unit_losses_is_the_variance_over_the_count():
    rng = np.random.default_rng(4)
    e = rng.normal(size=60)
    got = jackknife_var(len(e), lambda i: float(np.delete(e, i).mean()))
    assert got == pytest.approx(float(e.var(ddof=1) / len(e)))


def test_the_interval_rescales_bounds_and_bias_corrects_as_the_paper_states():
    s_in = [0.70, 0.72, 0.68, 0.71, 0.69]
    s_out = [0.74, 0.70, 0.72, 0.75, 0.69]
    terms = dict(terms=[dict(repeat=1, fold=k, s_in=a, s_out=b, b=0.0004)
                        for k, (a, b) in enumerate(zip(s_in, s_out), start=1)],
                 estimate=0.72, se_naive=0.01, folds=5, n_variable=10)
    got = ncv_interval(terms)
    mse_ncv = float(np.mean((np.asarray(s_in) - np.asarray(s_out)) ** 2) - 0.0004)
    assert got["mse_ncv"] == pytest.approx(mse_ncv)
    # Rescaled by (K - 1) / K, then held between the naive standard error and sqrt(K) times it.
    assert got["se"] == pytest.approx(min(np.sqrt(4 / 5 * mse_ncv), np.sqrt(5) * 0.01))
    assert got["err_ncv"] == pytest.approx(float(np.mean(s_in)))
    assert got["bias"] == pytest.approx((1 + 3 / 5) * (float(np.mean(s_in)) - 0.72))
    assert got["center"] == pytest.approx(float(np.mean(s_in)) - got["bias"])
    assert got["lower"] == pytest.approx(got["center"] - norm_ppf(0.975) * got["se"])

    flat = dict(terms, terms=[dict(r, s_out=r["s_in"]) for r in terms["terms"]])
    assert ncv_interval(flat)["se"] == pytest.approx(0.01)
    wide = dict(terms, terms=[dict(r, s_out=r["s_in"] + 1) for r in terms["terms"]])
    assert ncv_interval(wide)["se"] == pytest.approx(np.sqrt(5) * 0.01)


def test_the_selection_reports_both_intervals_and_the_fit_the_nested_one_is_for():
    x, y = design(200, 7)
    folds = fold_map(y, v=5, seed=2)
    sel = select_grain(x, y, centroid(), folds=folds, inner=3, metric="roc_auc",
                       interval="nested_cv", repeats=2, seed=1, verbose=False)
    kinds = {r["interval"] for r in sel.estimate}
    assert kinds == {"variables", "nested_cv"}
    nested = next(r for r in sel.estimate
                  if r["metric"] == "roc_auc" and r["interval"] == "nested_cv")
    across = next(r for r in sel.estimate
                  if r["metric"] == "roc_auc" and r["interval"] == "variables")
    assert nested["lower"] < nested["center"] < nested["upper"]
    assert nested["score"] == pytest.approx(across["score"])
    # The estimator's own quantities ride beside the rows rather than inside them.
    diagnostics = next(r for r in sel.nested_cv if r["metric"] == "roc_auc")
    assert diagnostics["repeats"] == 2 and diagnostics["folds"] == 5
    assert diagnostics["center"] == pytest.approx(nested["center"])
    # The interval is for the risk of the procedure fitted on every unit, which the object holds.
    assert sel.final["grain"] == "good"
    assert sel.final["fit"].predict(x["good"]).shape == y.values.shape
    with pytest.raises(ValueError, match="`interval` is one of"):
        select_grain(x, y, centroid(), folds=folds, inner=3, interval="bootstrap", verbose=False)


def test_a_paired_nested_interval_needs_a_ladder_that_was_cross_validated_for_it():
    x, y = design(200, 8)
    folds = fold_map(y, v=5, seed=2)
    plain = grain_ladder(x, y, centroid(), folds=folds, metric="roc_auc", verbose=False)
    with pytest.raises(KeyError, match="no nested cross-validation"):
        paired_contrast(plain, "good|centroid", "noisy|centroid", interval="nested_cv")

    lad = grain_ladder(x, y, centroid(), folds=folds, metric="roc_auc", interval="nested_cv",
                       repeats=2, seed=1, verbose=False)
    across = paired_contrast(lad, "good|centroid", "noisy|centroid")
    nested = paired_contrast(lad, "good|centroid", "noisy|centroid", interval="nested_cv")
    assert across["interval"] == "variables" and nested["interval"] == "nested_cv"
    # The point estimate is the same difference; only what is put around it differs.
    assert nested["diff"] == pytest.approx(across["diff"])
    assert nested["lower"] < nested["center"] < nested["upper"]
    assert nested["diff"] > 0
