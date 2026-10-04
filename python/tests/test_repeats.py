from __future__ import annotations

import numpy as np
import pytest

from timesift import cv, ensemble, grains, grouped_cv, timesift
from timesift.learners import Learner


def repeat_learner(name="rl"):
    def fit(x, y, **_):
        mu = x.values.mean(axis=(1, 2))
        s = mu.std(ddof=1)
        s = s if np.isfinite(s) and s > 0 else 1.0
        z = (mu - mu.mean()) / s
        beta = np.asarray([0.0 if len(np.unique(y[:, j])) < 2
                           else np.cov(z, y[:, j])[0, 1] for j in range(y.shape[1])])
        return dict(centre=mu.mean(), scale=s, beta=beta, prevalence=y.mean(axis=0))

    def predict(model, x):
        z = (x.values.mean(axis=(1, 2)) - model["centre"]) / model["scale"]
        base = np.log(np.clip(model["prevalence"], 0.05, 0.95)
                      / (1 - np.clip(model["prevalence"], 0.05, 0.95)))
        return 1 / (1 + np.exp(-(np.outer(z, 3 * model["beta"]) + base)))

    return Learner(name=name, fit=fit, predict=predict, multi="joint", reads="tabular")


def repeat_case(n_unit=30, days=60, seed=17):
    rng = np.random.default_rng(seed)
    t = np.datetime64("2021-09-01T00:00:00", "s") + np.arange(24 * days) * np.timedelta64(1, "h")
    units = [f"p{i:02d}" for i in range(n_unit)]
    warmth = rng.normal(size=n_unit)
    value = np.concatenate([w * 2 + rng.normal(0, 1, len(t)) for w in warmth])
    series = {"plot": [u for u in units for _ in range(len(t))], "t": list(t) * n_unit,
              "temp": list(value)}
    targets = {"plot": units, "group": [f"g{i // 3}" for i in range(n_unit)],
               "sp0": rng.binomial(1, 1 / (1 + np.exp(-3 * warmth))).tolist(),
               "sp1": rng.binomial(1, 1 / (1 + np.exp(3 * warmth))).tolist()}
    return targets, series


def run_repeat(case, resampling, use_ensemble=False, n_inner=None):
    targets, series = case
    return timesift(targets, series, y=["sp0", "sp1"], id="plot", time="t", x="temp",
                    learners=[repeat_learner("a"), repeat_learner("b")],
                    sift=grains("week", "month"),
                    ensemble=ensemble("stack") if use_ensemble else False,
                    resampling=resampling, n_inner=n_inner, verbose=False)


def test_a_repeated_resampling_says_how_many_times_it_is_drawn():
    spec = cv(v=4, seed=3, repeats=3)
    assert spec.repeats == 3 and cv().repeats == 1 and grouped_cv("site", repeats=2).repeats == 2
    with pytest.raises(ValueError):
        cv(repeats=0)
    with pytest.raises(ValueError):
        grouped_cv("site", repeats=0)


def test_each_repeat_is_a_run_on_its_own_fold_map_scored_as_the_run_alone_would_be():
    case = repeat_case()
    rep3 = run_repeat(case, cv(v=3, seed=5, repeats=3))
    assert rep3.repeats == 3
    assert sorted(set(rep3.scores["repeat"].tolist())) == [1, 2, 3]
    assert sorted(set(rep3.scores["fold"].tolist())) == list(range(1, 10))
    assert sorted(set(rep3.cells.fold.tolist())) == list(range(1, 10))
    singles = [run_repeat(case, cv(v=3, seed=5 + r)) for r in range(3)]
    for r, alone in enumerate(singles):
        mask = rep3.scores["repeat"] == r + 1
        got = sorted(zip(rep3.scores["candidate"][mask], rep3.scores["variable"][mask],
                         rep3.scores["fold"][mask] - 3 * r, rep3.scores["score"][mask]),
                     key=lambda t: t[:3])
        want = sorted(zip(alone.scores["candidate"], alone.scores["variable"],
                          alone.scores["fold"], alone.scores["score"]), key=lambda t: t[:3])
        assert [g[:3] for g in got] == [w[:3] for w in want]
        assert np.allclose([g[3] for g in got], [w[3] for w in want], equal_nan=True)
    # The out-of-fold prediction of a target is its mean over the repeats.
    want = np.mean([s.oof["a / week"] for s in singles], axis=0)
    assert np.allclose(rep3.oof["a / week"], want)


def test_a_response_is_averaged_over_its_folds_and_its_repeats():
    from timesift.report import summary
    fit = run_repeat(repeat_case(), cv(v=3, seed=5, repeats=2))
    assert "repeated 2 times" in summary(fit)
    from timesift.ladder import scored_cells
    candidate, variable, score = scored_cells(fit.scores)
    assert len(candidate) > 0 and np.isfinite(score).all()


def test_the_stack_is_fitted_on_the_out_of_fold_predictions_of_all_the_repeats_together():
    case = repeat_case()
    fit = run_repeat(case, cv(v=3, seed=5, repeats=2), use_ensemble=True)
    single = run_repeat(case, cv(v=3, seed=5), use_ensemble=True)
    assert fit.stack is not None and abs(sum(fit.stack.weights.values()) - 1) < 1e-8
    assert len(fit.cells.fold) == 2 * len(single.cells.fold)
    targets, series = case
    assert fit.predict(targets, series).shape == (30, 2)


def test_a_nested_estimate_is_read_off_all_the_repeats():
    fit = run_repeat(repeat_case(), cv(v=3, seed=5, repeats=2), use_ensemble=True, n_inner=2)
    arms = {row["arm"] for row in fit.estimate}
    assert {"selected", "ensemble"} <= arms
    roc = [row["score"] for row in fit.estimate if row["arm"] == "selected"
           and row["metric"] == "roc_auc"]
    assert len(roc) == 1 and np.isfinite(roc[0])
    assert sorted({row["repeat"] for row in fit.selected}) == [1, 2]
    assert sorted({row["fold"] for row in fit.selected}) == list(range(1, 7))
    assert fit.predictions["selected"].shape == fit.y.values.shape


def test_a_grouped_split_repeats_too_and_the_models_are_the_first_repeats():
    fit = run_repeat(repeat_case(), grouped_cv("group", v=3, seed=2, repeats=2))
    assert fit.repeats == 2
    assert sorted(set(fit.scores["repeat"].tolist())) == [1, 2]
    assert {"a / week", "b / month"} <= set(fit.models)
    alone = run_repeat(repeat_case(), grouped_cv("group", v=3, seed=2))
    assert fit.folds.fold.tolist() == alone.folds.fold.tolist()


def test_a_run_says_which_repeat_it_is_on(capsys):
    targets, series = repeat_case()
    timesift(targets, series, y=["sp0", "sp1"], id="plot", time="t", x="temp",
             learners=[repeat_learner("a")], sift=grains("month"), ensemble=False,
             resampling=cv(v=3, repeats=2), n_inner=None, verbose=True)
    assert "repeat 2 of 2" in capsys.readouterr().out
