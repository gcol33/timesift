"""Choosing the grain inside the training data, and what the nested estimate is of."""

from __future__ import annotations

import numpy as np
import pytest

from timesift import (Learner, Response, decision_threshold, feature_matrix, fold_map, metrics,
                       paired_contrast, select_grain, tss, tss_inflation, grain_ladder,
                       grain_matrix)
from timesift._stats import norm_ppf
from timesift.control import train_control
from timesift.ladder import per_variable
from timesift.learners import _logistic
from timesift.selection import SELECTED_ARM, _choose_candidate, _inner_splitter


def linear_learner(offset: float = 0.0, reduce: str = "mean") -> Learner:
    """A logistic fit on one number per unit, taken from the first channel over the bins.

    Fast, and what it reads depends on the grain: the mean over bins is the same number at every
    grain, while the coldest bin is the coldest day at one grain and the coldest month at
    another. A selector searching grains therefore has a real choice to get right or wrong.
    """
    take = {"mean": lambda v: v.mean(axis=1), "coldest": lambda v: v.min(axis=1)}[reduce]

    def fit(x, y, **_):
        m = take(x.values[:, :, 0])
        beta = [_logistic(m.reshape(-1, 1), y[:, j]) for j in range(y.shape[1])]
        return dict(beta=beta, offset=offset, rate=y.mean(axis=0))

    def predict(model, x):
        m = take(x.values[:, :, 0])
        out = []
        for j, b in enumerate(model["beta"]):
            if b is None:
                out.append(np.full(len(m), float(model["rate"][j])))
            else:
                eta = b["beta"][0] + b["beta"][1] * m + model["offset"]
                out.append(1.0 / (1.0 + np.exp(-eta)))
        return np.column_stack(out)

    return Learner(name="linear", fit=fit, predict=predict,
                   params=dict(offset=offset, reduce=reduce))


def fixture(n_unit=56, days=90, noise=6.0, v=4, seed=31):
    """A record carrying a per-unit level, and four responses generated from it.

    Every grain carries that level equally well, so this is the fixture for what a selection
    reports and refuses rather than for which grain it picks.
    """
    rng = np.random.default_rng(seed)
    t = np.datetime64("2021-09-01T00:00:00", "s") + np.arange(24 * days) * np.timedelta64(1, "h")
    units = [f"p{i:02d}" for i in range(n_unit)]
    warmth = rng.normal(size=n_unit)
    value = np.concatenate([w * 1.5 + rng.normal(0, noise, len(t)) for w in warmth])
    readings = {"id": [u for u in units for _ in range(len(t))],
                "time": list(t) * n_unit, "value": list(value)}
    sign = np.resize([1, -1], 4)
    y = rng.binomial(1, 1 / (1 + np.exp(-3 * np.outer(warmth, sign)))).astype(float)
    x = grain_matrix(readings, "id", "time", "value", grain=["day", "week", "month"])
    response = Response(y, tuple(units), tuple(f"sp{j}" for j in range(4)))
    return x, response, fold_map(response, v=v, seed=6)


def planted_at_month(n_unit=80, days=150, noise=6.0, v=5, seed=41):
    """A response generated from the coldest month of each unit's own record.

    The monthly grain carries that number exactly; the daily grain carries the coldest single
    day, which is the same quantity read through the noise of one day rather than of a month. So
    the grain the response was generated at is recoverable and the others are worse.
    """
    rng = np.random.default_rng(seed)
    t = np.datetime64("2021-09-01T00:00:00", "s") + np.arange(24 * days) * np.timedelta64(1, "h")
    units = [f"p{i:02d}" for i in range(n_unit)]
    value = np.concatenate([rng.normal(w * 1.5, noise, len(t)) for w in rng.normal(size=n_unit)])
    readings = {"id": [u for u in units for _ in range(len(t))],
                "time": list(t) * n_unit, "value": list(value)}
    x = grain_matrix(readings, "id", "time", "value", grain=["day", "week", "month"])
    coldest = x["month"].values[:, :, 0].min(axis=1)
    z = (coldest - coldest.mean()) / coldest.std()
    sign = np.resize([1, -1], 4)
    y = rng.binomial(1, 1 / (1 + np.exp(-3 * np.outer(z, sign)))).astype(float)
    response = Response(y, tuple(units), tuple(f"sp{j}" for j in range(4)))
    return x, response, fold_map(response, v=v, seed=6)


def test_a_selection_reports_one_winner_per_outer_fold_from_the_set_it_searched():
    x, y, folds = fixture()
    sel = select_grain(x, y, linear_learner(), folds=folds, inner=3, verbose=False)
    assert len(sel.selected) == 4
    assert {r["fold"] for r in sel.selected} == set(int(k) for k in np.unique(folds.fold))
    assert len(sel.candidates) == 3
    searched = {(c["grain"], c["learner"]) for c in sel.candidates}
    assert all((r["grain"], r["learner"]) in searched for r in sel.selected)
    assert len(sel.inner) == 3 * 4


def test_the_estimate_is_reported_under_every_metric_on_one_set_of_predictions():
    x, y, folds = fixture()
    sel = select_grain(x, y, linear_learner(), folds=folds, inner=3, verbose=False)
    assert {r["metric"] for r in sel.estimate} == set(metrics())
    assert all(np.isfinite(r["score"]) for r in sel.estimate)
    assert all(r["n_variable"] <= len(y.variables) for r in sel.estimate)
    # The selection metric's estimate is the mean of the same per-cell scores the object carries.
    own = next(r["score"] for r in sel.estimate if r["metric"] == sel.metric)
    assert own == pytest.approx(float(np.mean(list(per_variable(sel.scores).values()))))


def test_no_outer_test_unit_reaches_the_selector_or_the_refit_of_its_own_fold():
    x, y, folds = fixture()
    seen: list[tuple[str, ...]] = []

    def fit(x, y, **_):
        seen.append(tuple(x.units))
        return dict(rate=y.mean(axis=0))

    def predict(model, x):
        m = x.values[:, :, 0].mean(axis=1)
        rank = np.argsort(np.argsort(m)) / len(m)
        return 1.0 / (1.0 + np.exp(-(rank[:, None] - 0.5 + model["rate"][None, :])))

    spy = Learner(name="spy", fit=fit, predict=predict)
    sel = select_grain(x, y, spy, folds=folds, inner=3, verbose=False)

    # Every unit a model was fitted on during outer fold k, across the inner ladder and the refit,
    # must have come from outside fold k.
    per_outer = len(seen) / len(sel.selected)
    assert per_outer == int(per_outer)
    per_outer = int(per_outer)
    for i, row in enumerate(sel.selected):
        held = {u for u, k in zip(folds.units, folds.fold) if int(k) == row["fold"]}
        for fitted_on in seen[i * per_outer:(i + 1) * per_outer]:
            assert not held.intersection(fitted_on)


def test_a_folds_held_out_predictions_are_those_of_the_candidate_it_selected():
    x, y, folds = fixture()
    lad = grain_ladder(x, y, linear_learner(), folds=folds, verbose=False)
    sel = select_grain(x, y, linear_learner(), folds=folds, inner=3, verbose=False)
    p = sel.scores.predictions[SELECTED_ARM]
    # The refit is the ladder's own fit on the same units at the same grain, so every cell of the
    # selected procedure is a cell of the ladder rather than a number from a second fitting path.
    for row in sel.selected:
        held = folds.fold == row["fold"]
        assert np.allclose(p[held], lad.predictions[f"{row['grain']}|linear"][held])


def test_the_contrast_against_a_ladder_runs_through_paired_contrast_on_matched_cells():
    x, y, folds = fixture()
    lad = grain_ladder(x, y, linear_learner(), folds=folds, verbose=False)
    sel = select_grain(x, y, linear_learner(), folds=folds, inner=3, compare=lad, verbose=False)
    assert [r["b"] for r in sel.contrast] == ["day|linear", "week|linear", "month|linear"]
    assert all(r["a"] == SELECTED_ARM for r in sel.contrast)
    assert all(r["n_cell"] > 0 for r in sel.contrast)
    assert select_grain(x, y, linear_learner(), folds=folds, inner=3,
                        verbose=False).contrast is None


def test_a_comparator_scored_by_another_metric_is_refused():
    x, y, folds = fixture()
    lad = grain_ladder(x, y, linear_learner(), folds=folds, metric="roc_auc", verbose=False)
    with pytest.raises(ValueError, match="scored by roc_auc and the selection by tss"):
        select_grain(x, y, linear_learner(), folds=folds, inner=3, compare=lad, verbose=False)
    with pytest.raises(ValueError, match="grain_ladder"):
        select_grain(x, y, linear_learner(), folds=folds, inner=3, compare="week|linear",
                     verbose=False)


def test_a_candidate_set_with_nothing_to_choose_between_is_refused():
    x, y, folds = fixture()
    with pytest.raises(ValueError, match="at least two candidates"):
        select_grain(x["week"], y, linear_learner(), folds=folds, inner=3, verbose=False)


def test_the_inner_split_is_a_count_of_at_least_two_or_a_splitter_of_ones_own():
    for bad in (1, 0, "three", 2.5, True):
        with pytest.raises(ValueError, match="at least 2"):
            _inner_splitter(bad)
    x, y, folds = fixture()
    calls = []

    def by_hand(y_train):
        calls.append(len(y_train.units))
        return fold_map(y_train, v=2, seed=99)

    sel = select_grain(x, y, linear_learner(), folds=folds, inner=by_hand, verbose=False)
    assert len(calls) == len(sel.selected)


def test_the_grain_the_response_was_generated_at_is_selected_above_chance():
    x, y, folds = planted_at_month()
    sel = select_grain(x, y, linear_learner(reduce="coldest"), folds=folds, inner=4,
                       verbose=False)
    picked = [r["grain"] for r in sel.selected]
    # Chance over three candidates is a third of the five outer folds; the planted grain has to
    # beat that rather than merely appear.
    assert picked.count("month") >= 4


def test_the_nested_estimate_stays_under_what_choosing_on_the_held_out_units_would_have_paid():
    x, y, folds = planted_at_month()
    learner = linear_learner(reduce="coldest")
    lad = grain_ladder(x, y, learner, folds=folds, verbose=False)
    sel = select_grain(x, y, learner, folds=folds, inner=4, compare=lad, verbose=False)

    # The bound is the oracle: the same candidates, the same fits, but the grain for each cell
    # picked with the held-out score itself. Selection inside the training data cannot beat that.
    oracle = {}
    for w, ln, v, k, value in zip(lad.grain, lad.learner, lad.variable, lad.fold, lad.score):
        if np.isnan(value):
            continue
        oracle[(str(v), int(k))] = max(oracle.get((str(v), int(k)), -np.inf), float(value))
    by_variable: dict[str, list[float]] = {}
    for (v, _), value in oracle.items():
        by_variable.setdefault(v, []).append(value)
    bound = float(np.mean([np.mean(vals) for vals in by_variable.values()]))

    own = next(r["score"] for r in sel.estimate if r["metric"] == sel.metric)
    assert own <= bound + 1e-12


def test_with_no_signal_at_any_grain_the_procedure_scores_at_the_designs_own_floor():
    rng = np.random.default_rng(12)
    x, y, folds = fixture(n_unit=60, days=90, noise=8.0, v=4, seed=57)
    y = Response(rng.binomial(1, 0.4, y.values.shape).astype(float), y.units, y.variables)
    folds = fold_map(y, v=4, seed=6)
    sel = select_grain(x, y, linear_learner(), folds=folds, inner=3, verbose=False)
    # A threshold read at its own maximum is biased upward on cells this small, so the floor is
    # what a design with no signal reports rather than zero.
    floor = tss_inflation(y, folds, skill=(0.0,), replicates=60, seed=12)[0]["reported"]
    own = next(r["score"] for r in sel.estimate if r["metric"] == "tss")
    assert own < floor + 0.15


def test_an_arm_is_found_whole_so_a_learner_named_with_the_separator_is_still_an_arm():
    from dataclasses import replace
    x, y, folds = fixture()
    lad = grain_ladder(x, y, [replace(linear_learner(), name="a|b"), linear_learner(0.3)],
                       folds=folds, verbose=False)
    grain = lad.summary()[0]["grain"]
    whole = paired_contrast(lad, f"{grain}|a|b", f"{grain}|linear")
    assert whole["a"] == f"{grain}|a|b"
    with pytest.raises(KeyError, match="names a learner and no grain"):
        paired_contrast(lad, "a|b", f"{grain}|linear")
    with pytest.raises(KeyError, match=r'no arm called "week\|c"'):
        paired_contrast(lad, "week|c", "week|linear")


def test_a_contrast_needs_both_arms_to_have_scored_a_shared_cell():
    x, y, folds = fixture()
    lad = grain_ladder(x, y, linear_learner(), folds=folds, verbose=False)
    with pytest.raises(KeyError, match="no arm"):
        paired_contrast(lad, SELECTED_ARM, "week|linear")


def test_the_one_standard_error_rule_takes_the_coarsest_candidate_inside_the_band():
    grid = [dict(grain=g, learner="l", score=s, se=e) for g, s, e in
            (("day", 0.70, 0.02), ("week", 0.69, 0.03), ("month", 0.66, 0.01),
             ("year", 0.55, 0.01))]
    size = {"day": (365, 1), "week": (52, 1), "month": (12, 1), "year": (1, 1)}
    assert _choose_candidate(grid, size, "argmax")["grain"] == "day"
    # The band is the best candidate's own standard error: 0.70 - 0.02 admits the week and not the
    # month, however small the month's own error is.
    assert _choose_candidate(grid, size, "coarsest_adequate")["grain"] == "week"
    grid[2]["score"] = 0.685
    assert _choose_candidate(grid, size, "coarsest_adequate")["grain"] == "month"
    # Between two candidates with the same bins, fewer channels is coarser.
    two = [dict(grain="week.mmm", learner="l", score=0.70, se=0.01),
           dict(grain="week.mean", learner="l", score=0.695, se=0.01)]
    assert _choose_candidate(two, {"week.mmm": (52, 3), "week.mean": (52, 1)},
                             "coarsest_adequate")["grain"] == "week.mean"
    # A standard error that could not be computed leaves only the ties with the best.
    grid[0]["se"] = float("nan")
    assert _choose_candidate(grid, size, "coarsest_adequate")["grain"] == "day"
    with pytest.raises(ValueError, match="rule must be one of"):
        x, y, folds = fixture()
        select_grain(x, y, linear_learner(), folds=folds, inner=3, rule="widest", verbose=False)


def test_where_every_grain_carries_the_signal_equally_the_coarsest_is_taken():
    # The mean over bins is the same number at the day and the month grain, whose bins are whole
    # days, so those two tie exactly and the rule takes the month where the argmax takes the day
    # declared first. The week differs through its partial end bins.
    x, y, folds = fixture()
    coarse = select_grain(x, y, linear_learner(), folds=folds, inner=3,
                          rule="coarsest_adequate", verbose=False)
    top = select_grain(x, y, linear_learner(), folds=folds, inner=3, verbose=False)
    assert coarse.rule == "coarsest_adequate"
    bins = {w: x[w].values.shape[1] for w in x}
    for c, t in zip(coarse.selected, top.selected):
        rows = [r for r in coarse.inner if r["fold"] == c["fold"]]
        band = [r["grain"] for r in rows if r["score"] >= c["inner_best"] - c["inner_se"]]
        assert c["grain"] == min(band, key=lambda w: bins[w])
        assert bins[c["grain"]] <= bins[t["grain"]]
    assert any(c["grain"] == "month" and t["grain"] == "day"
               for c, t in zip(coarse.selected, top.selected))


def test_where_one_candidate_clearly_separates_the_coarsest_adequate_rule_is_the_argmax():
    x, y, folds = planted_at_month()
    learner = linear_learner(reduce="coldest")
    coarse = select_grain(x, y, learner, folds=folds, inner=4, rule="coarsest_adequate",
                          verbose=False)
    top = select_grain(x, y, learner, folds=folds, inner=4, verbose=False)
    for c, t in zip(coarse.selected, top.selected):
        rows = [r for r in coarse.inner if r["fold"] == c["fold"]]
        others = [r["score"] for r in rows if r["grain"] != t["grain"]]
        if all(s < c["inner_best"] - c["inner_se"] for s in others):
            assert c["grain"] == t["grain"]


def test_tss_at_a_given_cut_is_sensitivity_plus_specificity_minus_one_there():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.3, 0.7, 0.9])
    assert tss(y, p, threshold=0.5) == pytest.approx(2 / 3 - 1 / 3)
    assert tss(y, p, threshold=0.95) == 0
    assert np.isnan(tss(y, p, threshold=float("nan")))
    assert np.isnan(tss(np.zeros(3), np.array([0.1, 0.2, 0.3]), threshold=0.15))
    assert tss(y, p) >= tss(y, p, threshold=0.5)


def binormal_cut_design(n=3000, v=4, skill=0.6, seed=21):
    """Each variable's score is N(delta, 1) on a presence and N(0, 1) on an absence, so the
    population skill at the best cut, delta / 2, is 2 Phi(delta / 2) - 1. A learner reading the
    score and fitting nothing makes the inner out-of-fold predictions the score itself."""
    rng = np.random.default_rng(seed)
    delta = 2 * norm_ppf((skill + 1) / 2)
    units = [f"u{i:04d}" for i in range(n)]
    variables = [f"sp{j}" for j in range(v)]
    y = rng.binomial(1, 0.3, (n, v)).astype(float)
    score = y * delta + rng.normal(size=(n, v))
    noisy = score + rng.normal(0, 2, (n, v))
    reader = Learner(name="reader", fit=lambda x, y, **_: {},
                     predict=lambda model, x: x.values[:, :, 0])
    x = {"good": feature_matrix(score, units, variables, "good"),
         "noisy": feature_matrix(noisy, units, variables, "noisy")}
    return x, Response(y, tuple(units), tuple(variables)), score, reader, skill


def test_a_cut_learned_on_the_inner_folds_reads_the_population_skill_a_maximised_cut_overstates():
    x, y, score, reader, skill = binormal_cut_design()
    folds = fold_map(y, v=5, seed=3)
    sel = select_grain(x, y, reader, folds=folds, inner=5, threshold="youden", verbose=False)
    assert {r["grain"] for r in sel.selected} == {"good"}
    assert sel.threshold == "youden"
    # Every cut is the Youden cut of the outer training units' own scores.
    for row in sel.thresholds:
        train = folds.fold != row["fold"]
        j = y.variables.index(row["variable"])
        assert row["threshold"] == decision_threshold(y.values[train, j], score[train, j],
                                                      "youden")
    cut = sel.cut_scores.score[~np.isnan(sel.cut_scores.score)]
    top = sel.scores.score[~np.isnan(sel.scores.score)]
    assert len(cut) == len(top)
    assert np.all(top >= cut)
    level = next(r for r in sel.estimate if r["metric"] == "tss_inner_cut")
    assert abs(level["score"] - skill) < 3 * level["se"] + 0.01
    assert next(r["score"] for r in sel.estimate if r["metric"] == "tss") > level["score"]


def test_without_a_threshold_rule_no_cut_is_learned_and_another_name_is_refused():
    x, y, _, reader, _ = binormal_cut_design(n=300, v=2)
    folds = fold_map(y, v=3, seed=3)
    sel = select_grain(x, y, reader, folds=folds, inner=3, verbose=False)
    assert sel.thresholds is None and sel.cut_scores is None
    assert "tss_inner_cut" not in {r["metric"] for r in sel.estimate}
    with pytest.raises(ValueError, match="threshold must be None or one of"):
        select_grain(x, y, reader, folds=folds, inner=3, threshold="median", verbose=False)


def test_a_selection_hands_its_control_to_the_inner_search_and_to_the_refit_alike():
    x, y, folds = fixture(n_unit=40, days=40, v=2)
    seen = []

    def fit(x, y, *, control, **_):
        seen.append(control)
        return dict(rate=y.mean(axis=0))

    trained = Learner(name="trained", fit=fit, multi="joint",
                      predict=lambda model, x: np.tile(model["rate"], (x.values.shape[0], 1)))
    select_grain(x, y, [trained], folds=folds, inner=2,
                 control=train_control(epochs=7), verbose=False)
    # Two outer folds, each running an inner ladder over three grains at two inner folds and one
    # refit: nothing in that chain may reach a learner without the control the caller gave.
    assert len(seen) == 2 * (3 * 2 + 1)
    assert {c.epochs for c in seen} == {7}
