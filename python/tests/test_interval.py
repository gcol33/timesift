"""The interval for the procedure's risk: the pieces of the estimator, and what carries it."""

from __future__ import annotations

import numpy as np
import pytest

from timesift import (Learner, Response, feature_matrix, fold_map, grain_ladder, paired_contrast,
                      select_grain)
from timesift._stats import norm_ppf
from timesift.interval import (cell_values, jackknife_of, jackknife_var, level_jackknife, ncv_bias,
                               ncv_collect, ncv_interval, ncv_level, ncv_seed, ncv_terms)


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
    e_hat = [0.77, 0.66, 0.75, 0.79, 0.65]
    terms = dict(terms=[dict(repeat=1, fold=k, s_in=a, s_out=b, b=0.0004, e_hat=c)
                        for k, (a, b, c) in enumerate(zip(s_in, s_out, e_hat), start=1)],
                 estimate=0.72, se_naive=0.02, se_naive_center=0.025, folds=5, n_variable=10)
    got = ncv_interval(terms)
    mse_ncv = float(np.mean((np.asarray(s_in) - np.asarray(s_out)) ** 2) - 0.0004)
    mse_center = float(np.mean((np.asarray(e_hat) - np.asarray(s_out)) ** 2) - 0.0004)
    assert got["mse_ncv"] == pytest.approx(mse_ncv)
    assert got["mse_center"] == pytest.approx(mse_center)

    # Rescaled by (K - 1) / K, then held between the naive standard error and sqrt(K) times it:
    # the width the interval carries from the corrected estimate's error, the paper's from the
    # plain estimate's.
    def bounded(mse, floor):
        return min(max(np.sqrt(max(4 / 5 * mse, 0.0)), floor), np.sqrt(5) * floor)

    assert mse_center > 0
    assert got["se"] == pytest.approx(bounded(mse_center, 0.025))
    assert got["se_bates"] == pytest.approx(bounded(mse_ncv, 0.02))
    assert got["se"] != pytest.approx(got["se_bates"])
    assert got["err_ncv"] == pytest.approx(float(np.mean(s_in)))
    assert got["bias"] == pytest.approx((1 + 3 / 5) * (float(np.mean(s_in)) - 0.72))
    assert got["center"] == pytest.approx(float(np.mean(s_in)) - got["bias"])
    assert got["lower"] == pytest.approx(got["center"] - norm_ppf(0.975) * got["se"])

    flat = dict(terms, terms=[dict(r, s_out=r["s_in"], e_hat=r["s_in"]) for r in terms["terms"]])
    assert ncv_interval(flat)["se"] == pytest.approx(0.025)
    assert ncv_interval(flat)["se_bates"] == pytest.approx(0.02)
    wide = dict(terms, terms=[dict(r, s_out=r["s_in"] + 1) for r in terms["terms"]])
    assert ncv_interval(wide)["se"] == pytest.approx(np.sqrt(5) * 0.025)
    assert ncv_interval(wide)["se_bates"] == pytest.approx(np.sqrt(5) * 0.02)


def test_the_bias_one_level_down_is_the_papers_formula_at_one_fold_fewer():
    assert ncv_bias(4, 0.70, 0.72) == pytest.approx((1 + 2 / 4) * (0.70 - 0.72))
    assert ncv_seed(1, (2, 3, 4, 0)) == 1 + 10007 * 2 + 101 * 3 + 4
    assert ncv_seed(1, (2, 3, 4, 5)) == 1 + 10007 * 2 + 101 * 3 + 4 + 3001 * 5


def test_the_triple_fits_fill_every_inner_inner_prediction_and_nothing_else():
    rng = np.random.default_rng(3)
    units = tuple(f"u{i:02d}" for i in range(40))
    y = Response(rng.binomial(1, 0.4, (40, 2)).astype(float), units, ("a", "b"))
    m = np.asarray([1, 2, 3, 4, 5] * 8)
    seen = []

    def fit_predict(train, test, tag):
        seen.append(tag)
        return np.full((len(test), 2), tag[1] * 100 + tag[2] * 10 + tag[3], dtype=float)

    runs = ncv_collect(fit_predict, y, [m], np.zeros((40, 2)))
    assert len(seen) == 10 + 10
    deep = runs[0]["deep"]
    assert sorted(deep) == [1, 2, 3, 4, 5]
    for k in range(1, 6):
        assert sorted(deep[k]) == [j for j in range(1, 6) if j != k]
        for j in deep[k]:
            block = deep[k][j]
            # Filled on the units of every other fold, and nowhere else.
            assert not np.isnan(block[(m != k) & (m != j)]).any()
            assert np.isnan(block[(m == k) | (m == j)]).all()
            # Every prediction of fold l came from the fit that left out exactly k, j and l,
            # whose tag names the three in order.
            for l in range(1, 6):
                if l in (k, j):
                    continue
                code = sum(d * w for d, w in zip(sorted((k, j, l)), (100, 10, 1)))
                assert (block[m == l] == code).all()


def test_the_corrected_centres_naive_standard_error_is_the_jackknife_of_the_combination():
    # Two arms of fixed predictions, no refitting: the centre is 1.6 times the outer level less
    # 0.6 times the mean inner level, so its leave-one-out values are that combination of theirs.
    from timesift.response import Folds, scorable_cells
    rng = np.random.default_rng(5)
    n = 40
    units = tuple(f"u{i:02d}" for i in range(n))
    yv = rng.binomial(1, 0.4, (n, 2)).astype(float)
    y = Response(yv, units, ("a", "b"))
    m = np.asarray([1, 2, 3, 4, 5] * 8)
    p = rng.uniform(size=(n, 2)) + 0.5 * yv
    from timesift import roc_auc
    runs = [ncv_collect(lambda train, test, tag: p[test], y, [m], p)]

    def cells_fun(yy, mm):
        return scorable_cells(yy, Folds(fold=np.asarray(mm), units=yy.units))

    terms = ncv_terms(y, runs, cells_fun, roc_auc)
    levels = [1, 2, 3, 4, 5]
    theta_est = level_jackknife(y, [p], m, cell_values(y, [p], m, levels, cells_fun(y, m), roc_auc),
                                roc_auc)
    theta_in = np.zeros(n)
    for k in levels:
        train = np.flatnonzero(m != k)
        level, values = ncv_level(y.take_units(train), [p[train]], m[train],
                                  [x for x in levels if x != k], cells_fun, roc_auc)
        left = np.full(n, level)
        left[train] = level_jackknife(y.take_units(train), [p[train]], m[train], values, roc_auc)
        theta_in += left / 5
    assert terms["se_naive"] == pytest.approx(np.sqrt(jackknife_of(theta_est)))
    assert terms["se_naive_center"] == pytest.approx(
        np.sqrt(jackknife_of(1.6 * theta_est - 0.6 * theta_in)))


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
