from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from timesift._stats import norm_ppf, wilcoxon_p
from timesift.control import CONTROL_SETTINGS, as_control, train_control
from timesift.ladder import (grain_ladder, paired_contrast, per_variable,
                             score_predictions, tss_inflation, variable_means)
from timesift.learners import Learner, fit_learner
from timesift.metrics import kappa_score, model_agreement, roc_auc, tss
from timesift.representation import grain_matrix
from timesift.response import Response, fold_map, scorable_cells


def brute_tss(y, p):
    return max((np.sum((p >= c) & (y == 1)) / np.sum(y == 1)
                + np.sum((p < c) & (y == 0)) / np.sum(y == 0) - 1) for c in np.unique(p))


def sim(n_unit=48, days=120, noise=8.0, seed=1):
    rng = np.random.default_rng(seed)
    t = np.datetime64("2021-09-01T00:00:00", "s") + np.arange(24 * days) * np.timedelta64(1, "h")
    units = [f"p{i:03d}" for i in range(n_unit)]
    warmth = rng.normal(size=n_unit)
    value = np.concatenate([w * 1.5 + rng.normal(0, noise, len(t)) for w in warmth])
    readings = {"id": [u for u in units for _ in range(len(t))],
                "time": list(t) * n_unit, "value": list(value)}
    sign = np.resize([1, -1], 6)
    y = rng.binomial(1, 1 / (1 + np.exp(-3 * np.outer(warmth, sign))))
    return readings, Response(y.astype(float), tuple(units),
                              tuple(f"sp{j}" for j in range(6))), warmth


def test_tss_is_the_maximum_over_every_cut():
    rng = np.random.default_rng(3)
    for _ in range(20):
        n = int(rng.integers(8, 60))
        y = rng.binomial(1, 0.35, n)
        if len(np.unique(y)) < 2:
            continue
        p = np.round(rng.random(n), 2)
        assert tss(y, p) == pytest.approx(brute_tss(y, p))


def test_the_score_does_not_depend_on_the_order_the_units_arrived_in():
    rng = np.random.default_rng(4)
    y = rng.binomial(1, 0.3, 50)
    p = np.round(rng.random(50), 2)
    o = rng.permutation(50)
    assert tss(y, p) == pytest.approx(tss(y[o], p[o]))
    assert roc_auc(y, p) == pytest.approx(roc_auc(y[o], p[o]))


def test_a_one_class_cell_has_no_skill_to_measure():
    assert np.isnan(tss([0, 0, 0], [0.1, 0.2, 0.3]))
    assert np.isnan(roc_auc([1, 1, 1], [0.1, 0.2, 0.3]))
    assert np.isnan(kappa_score([0, 0, 0], [0.1, 0.2, 0.3]))


def test_agreement_counts_the_decisions_two_models_make_differently():
    y = [1, 1, 0, 0]
    apart = model_agreement(y, [0.9, 0.8, 0.2, 0.1], [0.1, 0.2, 0.8, 0.9], "prevalence")
    assert apart["n_disagree"] == 4
    assert apart["a_right"] == 4
    assert apart["b_right"] == 0


def test_a_cell_needs_both_classes_on_both_sides_of_the_split():
    units = tuple(f"p{i:02d}" for i in range(10))
    folds = np.repeat(np.arange(1, 6), 2)
    y = np.zeros((10, 4))
    y[:, 0] = 1
    y[0, 2] = 1
    y[[0, 2, 4, 6], 3] = 1
    cells = scorable_cells(Response(y, units, ("all", "none", "one", "split")), folds)
    assert not cells.is_scorable("all", 1)
    assert not cells.is_scorable("none", 1)
    assert not cells.is_scorable("one", 1)
    assert cells.is_scorable("split", 1)
    assert not cells.is_scorable("split", 5)


def test_every_unit_lands_in_exactly_one_fold():
    _, y, _ = sim(n_unit=97)
    f = fold_map(y, v=10)
    assert len(f) == 97
    assert sorted(set(f)) == list(range(1, 11))
    counts = np.bincount(f)[1:]
    assert counts.max() - counts.min() <= 2


def test_a_signal_buried_in_hourly_noise_is_found_once_the_record_is_averaged():
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn is not installed")
    readings, y, _ = sim(n_unit=48, days=120, noise=20.0, seed=5)
    x = grain_matrix(readings, "id", "time", "value", grain=["day", "month"])
    lad = grain_ladder(x, y, "elasticnet", folds=fold_map(y, v=4, seed=3), verbose=False)
    rows = {r["grain"]: r["score"] for r in lad.summary()}
    assert rows["month"] > rows["day"]
    gain = paired_contrast(lad, "month|elasticnet", "day|elasticnet")
    assert gain["diff"] > 0


def test_a_learner_of_ones_own_needs_nothing_but_a_fit_and_a_predict():
    readings, y, _ = sim(n_unit=20, days=30)
    x = grain_matrix(readings, "id", "time", "value", grain="week")
    mine = Learner(name="mine",
                   fit=lambda x, y, **kw: y.mean(axis=0),
                   predict=lambda m, x: np.tile(m, (x.values.shape[0], 1)))
    p = fit_learner(mine, x, y).predict(x)
    assert p.shape == (20, 6)


def test_every_arm_is_scored_on_the_same_cells():
    readings, y, _ = sim(n_unit=36, days=60)
    x = grain_matrix(readings, "id", "time", "value", grain=["week", "month"])
    rank = Learner(name="rank",
                   fit=lambda x, y, **kw: y.mean(axis=0),
                   predict=lambda m, x: np.tile(m, (x.values.shape[0], 1)))
    lad = grain_ladder(x, y, {"a": rank, "b": rank}, folds=fold_map(y, v=3, seed=2),
                        verbose=False)
    marks = {arm: lad.scorable[(lad.grain == w) & (lad.learner == ln)].tolist()
             for arm, (w, ln) in {f"{w}|{ln}": (w, ln)
                                  for w in ("week", "month") for ln in ("a", "b")}.items()}
    assert len({tuple(v) for v in marks.values()}) == 1


def test_a_threshold_chosen_on_the_scored_units_inflates_the_level():
    rng = np.random.default_rng(41)
    units = tuple(f"p{i:03d}" for i in range(250))
    y = Response(rng.binomial(1, 0.15, (250, 6)).astype(float), units,
                 tuple(f"sp{j}" for j in range(6)))
    out = tss_inflation(y, fold_map(y, v=10, seed=3), skill=(0.6, 0.9), replicates=40, seed=8)
    assert all(r["inflation"] > 0 for r in out)
    assert out[0]["inflation"] > out[1]["inflation"]


def test_the_normal_quantile_and_the_signed_rank_p_value_are_the_ones_r_reports():
    # qnorm(0.975) and qnorm(0.005) to twelve places
    assert norm_ppf(0.975) == pytest.approx(1.959963984540, abs=1e-11)
    assert norm_ppf(0.005) == pytest.approx(-2.575829303549, abs=1e-11)
    # wilcox.test(1:10)$p.value and wilcox.test(c(-3,-1,2,4,5,6))$p.value
    assert wilcoxon_p(np.arange(1, 11)) == pytest.approx(0.001953125, abs=1e-12)
    assert wilcoxon_p(np.array([-3.0, -1, 2, 4, 5, 6])) == pytest.approx(0.21875, abs=1e-12)


def test_a_level_is_the_mean_of_the_variable_means_wherever_it_is_read():
    """An arm's level, a candidate's level and a selection's level are one rule applied to three
    tables, so the rule is checked once and the three cannot drift apart."""
    readings, y, _ = sim(n_unit=36, days=60)
    x = grain_matrix(readings, "id", "time", "value", grain=["week", "month"])
    rank = Learner(name="rank",
                   fit=lambda x, y, **kw: y.mean(axis=0) + np.arange(y.shape[1]) / 100,
                   predict=lambda m, x: np.tile(m, (x.values.shape[0], 1)))
    lad = grain_ladder(x, y, rank, folds=fold_map(y, v=3, seed=2), verbose=False)

    flat = per_variable(lad)
    for row in lad.summary():
        mine = [v for (w, ln, _), v in flat.items()
                if w == row["grain"] and ln == row["learner"]]
        assert len(mine) == row["n_variable"]
        assert float(np.mean(mine)) == pytest.approx(row["score"])

    # And the primitive the three read is the mean within a variable before anything is averaged
    # across variables.
    level = variable_means(["a", "a", "a", "b"], ["v1", "v1", "v2", "v1"], [0.0, 1.0, 4.0, 2.0])
    assert level == {"a": {"v1": 0.5, "v2": 4.0}, "b": {"v1": 2.0}}


def test_inverting_the_map_recovers_the_skill_it_was_planted_from():
    from timesift.ladder import implied_skill
    rng = np.random.default_rng(61)
    units = tuple(f"p{i:03d}" for i in range(200))
    y = Response(rng.binomial(1, 0.2, (200, 6)).astype(float), units,
                 tuple(f"sp{j}" for j in range(6)))
    f = fold_map(y, v=5, seed=4)
    forward = tss_inflation(y, f, skill=(0.6,), replicates=60, seed=5)
    back = implied_skill(y, f, observed=forward[0]["reported"],
                         grid=np.arange(0.3, 0.95, 0.1), replicates=60, seed=5)
    assert back[0]["skill"] == pytest.approx(0.6, abs=0.05)
    assert back[0]["within_grid"]


def test_a_ladder_given_no_fold_map_builds_one():
    readings, y, _ = sim(n_unit=30, days=40, noise=1.0)
    x = grain_matrix(readings, "id", "time", "value", grain="week")
    lad = grain_ladder(x, y, [constant_learner()], verbose=False)
    # The same defaults `fold_map` would have been called with, so the two agree cell for cell.
    assert np.array_equal(lad.folds.fold, fold_map(y).fold)
    assert lad.folds.units == tuple(x.units)


def test_every_held_out_cell_is_the_prediction_for_that_unit_and_that_variable():
    readings, y, _ = sim(n_unit=30, days=40, noise=1.0)
    x = grain_matrix(readings, "id", "time", "value", grain="week")
    folds = fold_map(y, v=3, seed=4)
    lad = grain_ladder(x, y, [constant_learner()], folds=folds, verbose=False)
    p = lad.predictions["week|constant"]

    # The assembly is keyed by unit and by variable rather than by position, so every cell is
    # checkable against a refit on the same training units, matched by name on both axes.
    row = {u: i for i, u in enumerate(x.units)}
    column = {v: j for j, v in enumerate(y.variables)}
    for k in np.unique(folds.fold):
        train = np.flatnonzero(folds.fold != k)
        test = np.flatnonzero(folds.fold == k)
        fit = fit_learner(constant_learner(), x.take_units(train), y.take_units(train))
        held = x.take_units(test)
        expected = fit.predict(held)
        for a, unit in enumerate(held.units):
            for b, variable in enumerate(fit.variables):
                assert p[row[unit], column[variable]] == pytest.approx(expected[a, b])


def constant_learner():
    """A learner with nothing in it: every unit is predicted the mean of the fitting units."""
    return Learner(name="constant",
                   fit=lambda x, y, **_: dict(level=y.mean(axis=0)),
                   predict=lambda model, x: np.tile(model["level"], (x.values.shape[0], 1)))


def control_learner(seen, **own):
    """A learner that records the control it was trained under, so a test can read it back.

    Settings the learner carries are applied on top of the run's control by the fitter itself,
    which is what the encoders do, so both halves of the resolution are visible here.
    """
    def fit(x, y, *, control, **passed):
        seen.append(as_control(control).override(
            {k: v for k, v in passed.items() if k in CONTROL_SETTINGS}))
        return dict(level=y.mean(axis=0))

    return Learner(name="trained", fit=fit, params=dict(**own), multi="joint",
                   predict=lambda model, x: np.tile(model["level"], (x.values.shape[0], 1)))


def test_a_ladder_hands_its_control_to_every_learner_that_declares_one():
    readings, y, _ = sim(n_unit=30, days=40, noise=1.0)
    x = grain_matrix(readings, "id", "time", "value", grain=["week", "month"])
    seen = []
    grain_ladder(x, y, [control_learner(seen)], folds=fold_map(y, v=2, seed=4),
                 control=train_control(epochs=3), verbose=False)
    assert len(seen) == 2 * 2
    assert {c.epochs for c in seen} == {3}


def test_a_learner_carrying_a_setting_overrides_the_ladders_control_on_it():
    readings, y, _ = sim(n_unit=30, days=40, noise=1.0)
    x = grain_matrix(readings, "id", "time", "value", grain="week")
    seen = []
    grain_ladder(x, y, [control_learner(seen, epochs=11)], folds=fold_map(y, v=2, seed=4),
                 control=train_control(epochs=3, batch_size=8), verbose=False)
    assert {c.epochs for c in seen} == {11}
    assert {c.batch_size for c in seen} == {8}


def test_an_arm_that_holds_no_number_on_a_scorable_cell_stops_the_run():
    readings, y, _ = sim(n_unit=30, days=40, noise=1.0)
    x = grain_matrix(readings, "id", "time", "value", grain="week")

    def predict(model, x):
        p = np.tile(model["level"], (x.values.shape[0], 1))
        p[0, 0] = np.nan
        return p

    diverged = Learner(name="diverged", fit=lambda x, y, **_: dict(level=y.mean(axis=0)),
                       predict=predict)
    with pytest.raises(ValueError, match="did not settle"):
        grain_ladder(x, y, [diverged], folds=fold_map(y, v=2, seed=4), verbose=False)


def test_score_predictions_refuses_a_prediction_that_is_not_a_number():
    readings, y, _ = sim(n_unit=30, days=40, noise=1.0)
    folds = fold_map(y, v=2, seed=4)
    rng = np.random.default_rng(5)
    p = rng.random(y.values.shape)
    rows = score_predictions(y, p, folds)
    assert len(rows["score"]) == len(y.variables) * 2
    p[0, 0] = np.inf
    with pytest.raises(ValueError, match="the predictions hold no number"):
        score_predictions(y, p, folds)
