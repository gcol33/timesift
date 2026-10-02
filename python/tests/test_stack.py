"""The combiner and what a fitted object says about itself.

The stack is fitted on out-of-fold predictions and nothing else, so everything here is built from
matrices rather than from models, which is also the guarantee being checked.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from timesift import read_folds, read_response
from timesift.ladder import grain_ladder, score_arm
from timesift.learners import Learner
from timesift.metrics import roc_auc, tss
from timesift.occlusion import ladder_occlusion
from timesift.report import (candidate_table, ensemble_weights, occlusion, procedure_table,
                             summary)
from timesift.representation import grain_matrix
from timesift.response import Response, align_folds, fold_map, scorable_cells
from timesift.stack import (EnsembleSpec, Stack, as_ensemble, candidate_means, ensemble,
                            ensemble_combine, ensemble_fit, ensemble_spread, in_scope,
                            run_ensemble, score_weights, simplex_weights, stack_loss)

DEVIANCE = stack_loss("presence_absence")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def board(n_unit=80, seed=7):
    """Four responses on one latent value and three candidates that read it well, moderately and
    not at all.

    The fourth response is present on a handful of units of two folds only, so the other two folds
    carry no score for it and the mask has cells outside it to check against.
    """
    rng = np.random.default_rng(seed)
    units = tuple(f"p{i:03d}" for i in range(n_unit))
    z = rng.normal(size=n_unit)
    common = Response(np.column_stack([
        rng.binomial(1, sigmoid(2 * z)),
        rng.binomial(1, sigmoid(1.5 * z - 0.3)),
        rng.binomial(1, sigmoid(-1.8 * z))]).astype(float), units, ("sp0", "sp1", "sp2"))
    folds = fold_map(common, v=4, seed=3)
    rare = np.zeros(n_unit)
    for k in (1, 2):
        rare[np.flatnonzero(folds.fold == k)[:2]] = 1.0
    y = Response(np.column_stack([common.values, rare]), units,
                 common.variables + ("sp3",))
    cells = scorable_cells(y, folds)
    sign = np.array([1.0, 1.0, -1.0, 0.0])
    oof = {
        "cnn / week": sigmoid(np.outer(2.0 * z, sign) + rng.normal(0, 0.4, (n_unit, 4))),
        "elasticnet / week": sigmoid(np.outer(1.0 * z, sign) + rng.normal(0, 1.2, (n_unit, 4))),
        "forest / month": sigmoid(rng.normal(0, 1.0, (n_unit, 4))),
    }
    return oof, y, cells, folds


def score_table(oof, y, folds, cells, score=tss):
    f = align_folds(folds, y.units)
    rows = dict(candidate=[], variable=[], fold=[], score=[], scorable=[])
    for name, p in oof.items():
        got = score_arm("", name, y, p, f, np.unique(f), cells, score)
        rows["candidate"] += list(got["learner"])
        for key in ("variable", "fold", "score", "scorable"):
            rows[key] += list(got[key])
    return rows


# ---- the solver -------------------------------------------------------------------------------

def columns(*members):
    return np.column_stack([np.asarray(m).ravel() for m in members])


def mixture(p, y, w):
    return DEVIANCE["value"](np.asarray(p) @ np.asarray(w, dtype=float), np.asarray(y))


def test_the_weights_are_a_point_of_the_simplex():
    oof, y, _, _ = board()
    w = simplex_weights(columns(*oof.values()), y.values.ravel(), DEVIANCE)
    assert w.shape == (3,)
    assert (w >= 0).all()
    assert w.sum() == pytest.approx(1.0)


def test_the_weights_are_the_ones_a_search_over_the_simplex_finds():
    oof, y, _, _ = board()
    p = columns(oof["cnn / week"], oof["elasticnet / week"])
    flat = y.values.ravel()
    w = simplex_weights(p, flat, DEVIANCE)
    grid = np.linspace(0, 1, 4001)
    best = min(grid, key=lambda a: mixture(p, flat, [a, 1 - a]))
    assert w[0] == pytest.approx(best, abs=1e-3)
    assert mixture(p, flat, w) <= mixture(p, flat, [best, 1 - best]) + 1e-12


def test_no_direction_left_offers_an_improvement():
    """At the weights returned, every member carried sits on a direction as good as the best and
    no direction left out is better than the ones carried, which is what a minimum over the
    simplex is."""
    oof, y, _, _ = board()
    p = columns(*oof.values())
    flat = y.values.ravel()
    w = simplex_weights(p, flat, DEVIANCE)
    g = p.T @ DEVIANCE["gradient"](p @ w, flat)
    lam = float(g @ w)
    assert lam - float(g.min()) < 1e-6
    assert float(np.max(w * (g - lam))) < 1e-6


def test_two_candidates_that_predict_the_same_thing_share_the_weight():
    oof, y, _, _ = board()
    good = oof["cnn / week"]
    w = simplex_weights(columns(good, good.copy(), oof["forest / month"]), y.values.ravel(), DEVIANCE)
    assert w[0] == pytest.approx(w[1], abs=1e-12)


def test_the_candidate_that_reads_the_response_takes_the_larger_share():
    """A weight is a share of a mixture and not a verdict on a candidate: mixing a confident
    member towards a half tempers it, so an uninformative member keeps a small share rather than
    none. What has to hold is the order."""
    oof, y, _, _ = board()
    flat = y.values.ravel()
    good = oof["cnn / week"]
    against_noise = simplex_weights(columns(good, oof["forest / month"]), flat, DEVIANCE)
    against_itself_inverted = simplex_weights(columns(good, 1.0 - good), flat, DEVIANCE)
    assert against_noise[0] > 3 * against_noise[1]
    assert against_itself_inverted[0] > 3 * against_itself_inverted[1]


def test_the_same_matrices_give_the_same_weights_every_time():
    oof, y, _, _ = board()
    p = columns(*oof.values())
    first = simplex_weights(p, y.values.ravel(), DEVIANCE)
    assert np.array_equal(first, simplex_weights(p, y.values.ravel(), DEVIANCE))


def test_the_loss_the_weights_minimise_is_the_one_the_response_head_is_trained_under():
    assert stack_loss("presence_absence") is DEVIANCE
    p, obs = np.array([0.25, 0.75]), np.array([0.0, 1.0])
    assert DEVIANCE["value"](p, obs) == pytest.approx(-np.log(0.75))
    # And a head trained under something the combiner cannot minimise says so rather than guessing.
    from timesift.registry import RESPONSES
    RESPONSES.set("counts", dict(prepare=lambda r: r, activation="exp", loss="poisson",
                                 metric="tss", cells=lambda r, f: None), overwrite=True)
    try:
        with pytest.raises(ValueError, match="cannot minimise"):
            stack_loss("counts")
    finally:
        RESPONSES._entries.pop("counts")


# ---- fitting the combiner ---------------------------------------------------------------------

def test_the_stack_is_fitted_on_the_scorable_cells_and_on_no_others():
    """What a candidate predicts where no score is defined cannot buy it a weight, so replacing
    those cells with anything at all leaves the weights where they were."""
    oof, y, cells, folds = board()
    scores = score_table(oof, y, folds, cells)
    spec = ensemble()
    before = ensemble_fit(oof, y, cells, folds, spec, scores)

    admits = {(v, int(k)): bool(ok)
              for v, k, ok in zip(cells.variable, cells.fold, cells.scorable)}
    f = align_folds(folds, y.units)
    outside = ~np.array([[admits[(v, int(k))] for v in y.variables] for k in f])
    assert outside.any()
    rng = np.random.default_rng(11)
    spoiled = {name: p.copy() for name, p in oof.items()}
    for name, p in spoiled.items():
        p[outside] = rng.random(int(outside.sum()))
    after = ensemble_fit(spoiled, y, cells, folds, spec, scores)
    for name in before.weights:
        assert before.weights[name] == pytest.approx(after.weights[name], abs=1e-9)


def test_the_weights_are_named_non_negative_and_sum_to_one():
    oof, y, cells, folds = board()
    fitted = ensemble_fit(oof, y, cells, folds, ensemble(), score_table(oof, y, folds, cells))
    assert fitted.method == "stack"
    assert set(fitted.weights) == set(oof)
    assert min(fitted.weights.values()) >= 0
    assert sum(fitted.weights.values()) == pytest.approx(1.0)


def test_the_combination_reads_better_than_the_members_it_combines():
    oof, y, cells, folds = board()
    fitted = ensemble_fit(oof, y, cells, folds, ensemble(), score_table(oof, y, folds, cells))
    combined = ensemble_combine(fitted, oof)
    flat = y.values.ravel()
    assert mixture(columns(combined), flat, [1.0]) <= min(
        mixture(columns(p), flat, [1.0]) for p in oof.values()) + 1e-12


def test_an_ensemble_of_one_candidate_is_refused():
    oof, y, cells, folds = board()
    one = {"cnn / week": oof["cnn / week"]}
    with pytest.raises(ValueError, match="at least two candidates"):
        ensemble_fit(one, y, cells, folds, ensemble(), score_table(one, y, folds, cells))


def test_a_candidate_predicting_a_block_of_the_wrong_shape_is_named():
    oof, y, cells, folds = board()
    scores = score_table(oof, y, folds, cells)
    wrong = dict(oof)
    wrong["forest / month"] = oof["forest / month"][:, :2]
    with pytest.raises(ValueError, match="forest / month"):
        ensemble_fit(wrong, y, cells, folds, ensemble(), scores)


def test_the_methods_that_need_no_fitting_say_what_they_do():
    oof, y, cells, folds = board()
    scores = score_table(oof, y, folds, cells)
    plain = ensemble_fit(oof, y, cells, folds, ensemble(method="mean"), scores)
    assert plain.weights == {name: pytest.approx(1 / 3) for name in oof}
    middle = ensemble_fit(oof, y, cells, folds, ensemble(method="median"), scores)
    assert middle.members == tuple(oof)
    # A median counts its members equally, and says so the way the R side says it.
    assert middle.weights == {name: pytest.approx(1 / 3) for name in oof}

    by_score = ensemble_fit(oof, y, cells, folds, ensemble(method="weighted"), scores)
    level = candidate_means(scores)
    total = sum(max(level[n], 0.0) for n in oof)
    for name in oof:
        assert by_score.weights[name] == pytest.approx(max(level[name], 0.0) / total)


def test_combining_is_the_arithmetic_the_method_names():
    oof, y, cells, folds = board()
    scores = score_table(oof, y, folds, cells)
    stacked = np.stack(list(oof.values()))
    assert np.allclose(
        ensemble_combine(ensemble_fit(oof, y, cells, folds, ensemble(method="mean"), scores), oof),
        stacked.mean(axis=0))
    assert np.allclose(
        ensemble_combine(ensemble_fit(oof, y, cells, folds, ensemble(method="median"), scores),
                         oof),
        np.median(stacked, axis=0))
    fitted = ensemble_fit(oof, y, cells, folds, ensemble(), scores)
    by_hand = sum(fitted.weights[n] * oof[n] for n in fitted.members)
    assert np.allclose(ensemble_combine(fitted, oof), by_hand)


def test_combining_needs_a_prediction_from_every_member():
    oof, y, cells, folds = board()
    fitted = ensemble_fit(oof, y, cells, folds, ensemble(), score_table(oof, y, folds, cells))
    with pytest.raises(KeyError, match="forest / month"):
        ensemble_combine(fitted, {n: p for n, p in oof.items() if n != "forest / month"})


def test_a_scope_holds_one_axis_and_varies_the_other():
    oof, y, cells, folds = board()
    scores = score_table(oof, y, folds, cells)
    names = tuple(oof)
    level = candidate_means(scores)
    assert in_scope(names, "all", level) == names
    # The best candidate reads the week representation, so the learners scope keeps the learners
    # on it and the representations scope keeps that learner's representations.
    assert in_scope(names, "learners", level) == ("cnn / week", "elasticnet / week")
    assert in_scope(names, "representations", level) == ("cnn / week",)
    fitted = ensemble_fit(oof, y, cells, folds, ensemble(scope="learners"), scores)
    assert fitted.members == ("cnn / week", "elasticnet / week")
    with pytest.raises(ValueError, match="at least two candidates"):
        ensemble_fit(oof, y, cells, folds, ensemble(scope="representations"), scores)


def test_what_an_ensemble_may_be_asked_for():
    assert as_ensemble(True) == EnsembleSpec("stack", "all", None, None)
    assert as_ensemble(False) is None
    assert as_ensemble(None) is None
    assert as_ensemble("median").method == "median"
    assert as_ensemble(ensemble(scope="learners")).scope == "learners"
    with pytest.raises(ValueError, match="`method` is one of"):
        ensemble(method="vote")
    with pytest.raises(ValueError, match="`scope` is one of"):
        ensemble(scope="everything")
    with pytest.raises(TypeError, match="method name"):
        as_ensemble(3)


def test_the_head_a_run_combines_under_is_the_head_it_was_fitted_under(temporary_response):
    temporary_response("gauss_test", dict(
        prepare=lambda r: r, activation="identity", loss="squared_error", metric="roc_auc",
        cells=lambda r, folds: scorable_cells(r, folds)))
    # Left unnamed the spec takes the run's head; naming the same one is not a contradiction.
    assert run_ensemble(True, "gauss_test").response == "gauss_test"
    assert run_ensemble(ensemble(), "gauss_test").response == "gauss_test"
    assert run_ensemble(ensemble(response="gauss_test"), "gauss_test").response == "gauss_test"
    assert run_ensemble(False, "gauss_test") is None
    with pytest.raises(ValueError, match="names another"):
        run_ensemble(ensemble(response="presence_absence"), "gauss_test")


# ---- what a fitted object says ----------------------------------------------------------------

def fitted_object(oof, y, cells, folds, multi=None, method="stack", extra=None):
    scores = score_table(oof, y, folds, cells)
    multi = multi or {name: "joint" for name in oof}
    names = list(multi)
    stack = ensemble_fit(oof, y, cells, folds, ensemble(method=method), scores) if method \
        else None
    return SimpleNamespace(
        y=y, folds=folds, cells=cells, metric="tss", scorer=tss, response="presence_absence",
        oof=oof, scores=scores, stack=stack,
        weights=None if stack is None else stack.weights,
        candidates=dict(candidate=names,
                        learner=[n.split(" / ")[0] for n in names],
                        representation=[n.split(" / ")[-1] for n in names],
                        multi=[multi[n] for n in names]),
        **(extra or {}))


def test_the_table_carries_a_level_a_win_count_and_how_the_responses_were_covered():
    oof, y, cells, folds = board()
    fit = fitted_object(oof, y, cells, folds,
                        multi={"cnn / week": "joint", "elasticnet / week": "separate",
                               "forest / month": "separate"})
    rows = candidate_table(fit)
    assert [r["candidate"] for r in rows] == ["forest / month", "elasticnet / week", "cnn / week"]
    assert [r["responses"] for r in rows] == ["separate", "separate", "joint"]
    assert sum(r["won"] for r in rows) == len(y.variables)
    assert rows[-1]["won"] == max(r["won"] for r in rows)
    level = candidate_means(fit.scores)
    for r in rows:
        assert r["mean"] == pytest.approx(level[r["candidate"]])


def test_a_candidate_that_could_not_be_paired_is_listed_under_the_ones_that_were():
    oof, y, cells, folds = board()
    fit = fitted_object(oof, y, cells, folds)
    fit.candidates["candidate"].append("cnn / multigrain")
    fit.candidates["learner"].append("cnn")
    fit.candidates["representation"].append("multigrain")
    fit.candidates["multi"].append("joint")
    rows = candidate_table(fit)
    assert rows[-1] == dict(candidate="cnn / multigrain", responses="joint", won=None, mean=None)
    assert "cnn / multigrain  not applicable" in summary(fit)


def test_the_summary_is_the_candidates_and_the_weights():
    oof, y, cells, folds = board()
    fit = fitted_object(oof, y, cells, folds)
    lines = summary(fit).splitlines()
    assert lines[0] == "timesift  80 targets, 4 responses, 4-fold random CV, tss"
    assert lines[2] == "candidates, scored on the outer folds"
    assert lines[3].split() == ["candidate", "mean", "won", "responses"]
    assert [line.split()[0] for line in lines[4:7]] == ["forest", "elasticnet", "cnn"]
    assert "procedure" not in summary(fit)
    assert lines[-1].startswith("weights on every target  ")


def estimated(fit):
    """A fit carrying the estimate and the per-fold choice the nested evaluation writes."""
    fit.estimate = [dict(arm=arm, metric=metric, score=score, se=0.01)
                    for arm, base in (("selected", 0.61), ("ensemble", 0.63))
                    for metric, score in (("tss", base), ("roc_auc", base + 0.2))]
    fit.selected = [dict(fold=k, candidate="cnn / week" if k < 4 else "forest / month")
                    for k in range(1, 5)]
    fit.choice = "cnn / week"
    return fit


def test_the_procedure_rows_are_the_estimate_under_the_runs_own_metric():
    oof, y, cells, folds = board()
    fit = estimated(fitted_object(oof, y, cells, folds))
    rows = procedure_table(fit)
    assert [r["candidate"] for r in rows] == ["selected", "ensemble"]
    assert [r["mean"] for r in rows] == [0.61, 0.63]
    text = summary(fit)
    assert "procedure, chosen and weighted inside each outer training fold" in text
    assert "selected cnn / week in 3, forest / month in 1 of 4 folds" in text
    assert "choice on every target  cnn / week" in text


def test_a_fit_that_made_no_estimate_has_no_procedure_rows():
    assert procedure_table(SimpleNamespace(estimate=None, metric="tss")) == []


def test_the_weights_a_fit_reports_are_the_ones_its_combiner_holds():
    oof, y, cells, folds = board()
    fit = fitted_object(oof, y, cells, folds)
    assert ensemble_weights(fit) == fit.stack.weights
    assert ensemble_weights(SimpleNamespace(stack=None)) is None


def test_a_summary_of_a_fit_without_an_ensemble_is_its_candidates():
    oof, y, cells, folds = board()
    fit = fitted_object(oof, y, cells, folds)
    fit.stack = None
    text = summary(fit)
    assert "ensemble" not in text
    assert "weights" not in text


# ---- the occlusion profile through a fit --------------------------------------------------------

def bin_reader() -> Learner:
    """Reads the first bin of the first channel and nothing else, so a profile has a known
    answer."""
    def fit(x, y, **_):
        v = x.values[:, 0, 0]
        return dict(mid=float(np.median(v)), scale=float(v.std()) + 1e-9, n=y.shape[1])

    def predict(model, x):
        p = sigmoid((x.values[:, 0, 0] - model["mid"]) / model["scale"])
        return np.column_stack([p if j == 0 else 1 - p for j in range(model["n"])])

    return Learner(name="reader", fit=fit, predict=predict, reads="sequence", multi="joint")


def occlusion_fixture(n_unit=40, seed=19):
    rng = np.random.default_rng(seed)
    t = np.datetime64("2021-09-01T00:00:00", "s") + np.arange(24 * 120) * np.timedelta64(1, "h")
    first = np.arange(len(t)) < 24 * 30
    units = [f"p{i:03d}" for i in range(n_unit)]
    warmth = rng.normal(size=n_unit)
    value = np.concatenate([warmth[i] * 6 * first + rng.normal(0, 0.3, len(t))
                            for i in range(n_unit)])
    readings = {"id": [u for u in units for _ in range(len(t))],
                "time": list(t) * n_unit, "value": list(value)}
    y = Response(rng.binomial(1, sigmoid(3 * np.outer(warmth, [1, -1]))).astype(float),
                 tuple(units), ("sp0", "sp1"))
    x = grain_matrix(readings, "id", "time", "value", grain="month")
    return x, y


def test_a_profile_read_through_a_fit_is_the_one_read_through_a_ladder():
    x, y = occlusion_fixture()
    folds = fold_map(y, v=3, seed=5)
    lad = grain_ladder(x, y, bin_reader(), folds=folds, keep_fits=True, verbose=False)
    through_ladder = ladder_occlusion(lad, x, y, "month|reader", metric="roc_auc", permutations=3,
                                   seed=2)

    cells = scorable_cells(y, folds)
    oof = {"reader / month": lad.predictions["month|reader"],
           "flat / month": np.full(y.values.shape, 0.5)}
    fit = fitted_object(oof, y, cells, folds, method=None, extra=dict(
        representations={"month": x},
        fits={f"reader / month|{k}": v for k, v in
              ((int(key.rsplit("|", 1)[1]), value) for key, value in lad.fits.items())}))
    through_fit = occlusion(fit, "reader / month", metric="roc_auc", permutations=3, seed=2)

    assert through_fit["part"] == through_ladder["part"]
    assert np.allclose(through_fit["weight"], through_ladder["weight"], equal_nan=True)
    # And it is the bin the signal was planted in that carries the weight.
    heaviest = int(np.nanargmax(np.nanmean(through_fit["weight"], axis=1)))
    assert through_fit["part"][heaviest].startswith("2021-09")


def test_a_profile_needs_the_models_the_fit_was_told_to_keep():
    x, y = occlusion_fixture(n_unit=24)
    folds = fold_map(y, v=3, seed=5)
    cells = scorable_cells(y, folds)
    oof = {"reader / month": np.full(y.values.shape, 0.5),
           "flat / month": np.full(y.values.shape, 0.4)}
    fit = fitted_object(oof, y, cells, folds, method=None,
                        extra=dict(representations={"month": x}, fits={}))
    with pytest.raises(ValueError, match="keep_fits=True"):
        occlusion(fit, "reader / month")
    fit.fits = {"other / month|1": None}
    with pytest.raises(KeyError, match="reader / month"):
        occlusion(fit, "reader / month")


def test_asking_for_a_candidate_a_fit_does_not_carry_says_which_it_does():
    x, y = occlusion_fixture(n_unit=24)
    folds = fold_map(y, v=3, seed=5)
    cells = scorable_cells(y, folds)
    oof = {"reader / month": np.full(y.values.shape, 0.5),
           "flat / month": np.full(y.values.shape, 0.4)}
    fit = fitted_object(oof, y, cells, folds, method=None, extra=dict(
        representations={"month": x}, fits={"nowhere / week|1": None}))
    with pytest.raises(KeyError, match="reader / month"):
        occlusion(fit, "nowhere / week")


def test_a_stack_says_what_it_is():
    made = Stack(method="stack", weights={"a / week": 0.75, "b / week": 0.25},
                 members=("a / week", "b / week"))
    assert "a / week 0.750" in repr(made)
    assert "median" in repr(Stack(method="median", weights={"a": 0.5, "b": 0.5},
                                  members=("a", "b")))


def test_roc_auc_is_the_metric_the_profile_was_asked_for():
    # A guard on the fixture rather than on the code: the profile above is read at roc_auc, so a
    # constant prediction has to leave it undefined rather than at some level.
    assert np.isnan(roc_auc([1, 1, 1], [0.5, 0.5, 0.5]))


def test_the_combiner_refuses_to_drop_a_scorable_cell_rather_than_fitting_on_fewer():
    oof, y, cells, folds = board()
    admits = {(v, int(k)): bool(ok)
              for v, k, ok in zip(cells.variable, cells.fold, cells.scorable)}
    f = align_folds(folds, y.units)
    inside = np.array([[admits[(v, int(k))] for v in y.variables] for k in f])
    a, b = np.argwhere(inside)[0]
    spoiled = {name: p.copy() for name, p in oof.items()}
    spoiled["forest / month"][a, b] = np.nan
    with pytest.raises(ValueError, match="did not settle"):
        ensemble_fit(spoiled, y, cells, folds, ensemble(), score_table(oof, y, folds, cells))


# ---- biomod2's combinations -------------------------------------------------------------------

FIXTURES = Path(__file__).resolve().parents[2] / "inst" / "spec" / "fixtures"


def rows(name):
    with open(FIXTURES / name, newline="") as fh:
        return list(csv.DictReader(fh))


def number(text):
    return float("nan") if text == "NA" else float(text)


def test_a_specification_refuses_a_setting_its_method_does_not_have():
    assert ensemble(method="weighted").decay == "proportional"
    assert ensemble(method="committee").rule == "youden"
    assert ensemble(method="mean").decay is None
    with pytest.raises(ValueError, match='"weighted" method'):
        ensemble(method="mean", decay=2)
    with pytest.raises(ValueError, match="at least one"):
        ensemble(method="weighted", decay=0.5)
    with pytest.raises(ValueError, match="at least one"):
        ensemble(method="weighted", decay="linear")
    with pytest.raises(ValueError, match="member of a committee"):
        ensemble(rule="kappa")
    with pytest.raises(ValueError, match="`rule` is one of"):
        ensemble(method="committee", rule="otsu")
    with pytest.raises(ValueError, match="one finite number"):
        ensemble(min_score=float("nan"))


def test_a_decay_weighs_each_rank_down_by_the_same_ratio_and_shares_it_across_a_tie():
    assert score_weights([0.9, 0.5, 0.7], 2) == pytest.approx(np.array([8, 2, 4]) / 14)
    assert score_weights([0.9, 0.1, 0.8], 2) == pytest.approx(np.array([8, 2, 4]) / 14)
    assert score_weights([0.9, 0.7, 0.7, 0.2], 2) == pytest.approx(np.array([16, 6, 6, 2]) / 30)
    assert score_weights([0.9, 0.7, -0.1], 3) == pytest.approx(np.array([9, 3, 0]) / 12)
    assert score_weights([-0.2, 0.0, float("nan")], 3) == pytest.approx(np.full(3, 1 / 3))
    assert score_weights([0.6, 0.2]) == pytest.approx([0.75, 0.25])


def test_a_minimum_score_is_applied_before_the_scope_picks_and_refuses_to_leave_one():
    oof, y, cells, folds = board()
    scores = score_table(oof, y, folds, cells)
    level = candidate_means(scores)
    cut = float(np.mean(sorted(level.values())[:2]))
    kept = ensemble_fit(oof, y, cells, folds, ensemble(method="mean", min_score=cut), scores)
    assert kept.members == tuple(n for n in oof if level[n] >= cut)
    with pytest.raises(ValueError, match="leaves 0 candidates"):
        ensemble_fit(oof, y, cells, folds,
                     ensemble(method="mean", min_score=max(level.values()) + 0.01), scores)
    with pytest.raises(ValueError, match="Neither was given"):
        ensemble_fit(oof, y, cells, folds, ensemble(method="mean", min_score=0.0))


def test_a_committee_member_votes_at_its_own_cut_and_one_holding_no_cut_does_not_vote():
    from timesift.metrics import decision_threshold
    oof, y, cells, folds = board()
    st = ensemble_fit(oof, y, cells, folds, ensemble(method="committee", rule="kappa"))
    names = list(oof)
    for i, name in enumerate(names):
        for j in range(len(y.variables)):
            want = decision_threshold(y.values[:, j], oof[name][:, j], "kappa")
            assert st.thresholds[i, j] == want or (np.isnan(want) and np.isnan(st.thresholds[i, j]))
    votes = sum((oof[n] >= st.thresholds[i]) for i, n in enumerate(names)) / 3
    combined = ensemble_combine(st, oof)
    finite = np.isfinite(st.thresholds).all(axis=0)
    assert np.array_equal(combined[:, finite], votes[:, finite])

    cut = st.thresholds.copy()
    cut[2, 0] = np.nan
    share = ensemble_combine(replace(st, thresholds=cut), oof)[:, 0]
    two = (oof[names[0]][:, 0] >= cut[0, 0]).astype(float) + (oof[names[1]][:, 0] >= cut[1, 0])
    assert np.array_equal(share, two / 2)
    cut[:, 0] = np.nan
    assert np.isnan(ensemble_combine(replace(st, thresholds=cut), oof)[:, 0]).all()


def test_a_spread_is_the_members_mean_spread_and_t_interval_under_the_stacks_weights():
    oof, y, cells, folds = board()
    plain = ensemble_fit(oof, y, cells, folds, ensemble(method="mean"))
    s = ensemble_spread(plain, oof, alpha=0.1)
    assert s.shape == y.values.shape + (5,)
    x = np.array([p[7, 2] for p in oof.values()])
    assert s[7, 2, 0] == pytest.approx(x.mean())
    assert s[7, 2, 1] == pytest.approx(x.std(ddof=1))
    assert s[7, 2, 2] == pytest.approx(x.std(ddof=1) / x.mean())
    half = 2.919985580355516 * x.std(ddof=1) / np.sqrt(3)
    assert s[7, 2, 3] == pytest.approx(max(x.mean() - half, 0.0))
    assert s[7, 2, 4] == pytest.approx(min(x.mean() + half, 1.0))
    assert (s[..., 3] >= 0).all() and (s[..., 4] <= 1).all()

    stacked = ensemble_fit(oof, y, cells, folds, ensemble(), score_table(oof, y, folds, cells))
    assert np.allclose(ensemble_spread(stacked, oof)[..., 0], ensemble_combine(stacked, oof))
    lone = replace(stacked, weights={n: float(i == 0) for i, n in enumerate(stacked.members)})
    assert np.isnan(ensemble_spread(lone, oof)[..., 1:]).all()
    with pytest.raises(ValueError, match="strictly between"):
        ensemble_spread(stacked, oof, alpha=1.0)


def test_every_combination_lands_on_the_numbers_the_contract_pins():
    y = read_response(FIXTURES / "response.csv")
    folds = read_folds(FIXTURES / "folds.csv", y.units)
    cells = scorable_cells(y, folds)
    where = {(u, v): (i, j) for i, u in enumerate(y.units) for j, v in enumerate(y.variables)}
    oof = {}
    for row in rows("ensemble_oof.csv"):
        p = oof.setdefault(row["candidate"], np.full(y.values.shape, np.nan))
        p[where[(row["id"], row["variable"])]] = float(row["p"])
    scores = score_table(oof, y, folds, cells)
    weights, thresholds, expected = (rows("ensemble_weights.csv"),
                                     rows("ensemble_thresholds.csv"),
                                     rows("ensemble_predict.csv"))
    for case in rows("ensemble_cases.csv"):
        spec = ensemble(method=case["method"], scope=case["scope"], metric=case["metric"] or None,
                        min_score=float(case["min_score"]) if case["min_score"] else None,
                        decay=float(case["decay"]) if case["decay"] else None,
                        rule=case["rule"] or None)
        st = ensemble_fit(oof, y, cells, folds, spec, scores)
        tol = 1e-10
        w = [r for r in weights if r["case"] == case["case"]]
        assert st.members == tuple(r["member"] for r in w), case["case"]
        for r in w:
            assert st.weights[r["member"]] == pytest.approx(float(r["weight"]), rel=tol, abs=1e-9)
        for r in (r for r in thresholds if r["case"] == case["case"]):
            got = st.thresholds[st.members.index(r["member"]), y.variables.index(r["variable"])]
            want = number(r["threshold"])
            assert got == pytest.approx(want, rel=1e-10, nan_ok=True), (case["case"], r)
        p = ensemble_combine(st, oof)
        s = ensemble_spread(st, oof, alpha=0.1)
        for r in (r for r in expected if r["case"] == case["case"]):
            i, j = where[(r["id"], r["variable"])]
            got = [p[i, j], *s[i, j]]
            want = [number(r[k]) for k in ("combined", "mean", "sd", "cv", "lower", "upper")]
            assert got == pytest.approx(want, rel=tol, abs=1e-9, nan_ok=True), (case["case"], r)


def test_the_combiner_under_a_count_response_lands_on_the_numbers_the_contract_pins():
    from timesift.metrics import regression_metric
    from timesift.response import Folds, numeric_cells
    response = rows("ensemble_count_response.csv")
    units = tuple(r["id"] for r in response)
    y = Response(np.array([[float(r["y"])] for r in response]), units, ("count",))
    folds = Folds(fold=np.array([int(r["fold"]) for r in response]), units=units)
    cells = numeric_cells(y, folds)
    oof = {}
    for row in rows("ensemble_count_oof.csv"):
        oof.setdefault(row["candidate"], []).append(float(row["p"]))
    oof = {k: np.array(v).reshape(-1, 1) for k, v in oof.items()}
    scores = score_table(oof, y, folds, cells,
                         score=lambda obs, p: -regression_metric(obs, p, "poisson_deviance"))
    weights, expected = rows("ensemble_count_weights.csv"), rows("ensemble_count_predict.csv")
    tol = 1e-9
    for case in rows("ensemble_count_cases.csv"):
        spec = ensemble(method=case["method"], scope=case["scope"], response="count")
        st = ensemble_fit(oof, y, cells, folds, spec, scores)
        assert st.loss == "poisson_deviance"
        w = [r for r in weights if r["case"] == case["case"]]
        assert st.members == tuple(r["member"] for r in w), case["case"]
        for r in w:
            assert st.weights[r["member"]] == pytest.approx(float(r["weight"]), rel=tol, abs=tol)
        p = ensemble_combine(st, oof)
        s = ensemble_spread(st, oof, alpha=0.1)
        for r in (r for r in expected if r["case"] == case["case"]):
            i = units.index(r["id"])
            got = [p[i, 0], s[i, 0][1], s[i, 0][3], s[i, 0][4]]
            want = [number(r[k]) for k in ("combined", "sd", "lower", "upper")]
            assert got == pytest.approx(want, rel=tol, abs=tol, nan_ok=True), (case["case"], r)


def test_the_poisson_deviance_loss_is_the_one_a_count_head_is_trained_under():
    loss = stack_loss("count")
    assert loss is stack_loss("count") and loss["range"] == (0.0, np.inf)
    y = np.array([0.0, 1.0, 2.0, 5.0, 3.0, 0.0, 4.0, 2.0])
    p = np.array([0.4, 1.3, 1.9, 4.1, 3.6, 0.3, 3.2, 2.4])
    with np.errstate(divide="ignore", invalid="ignore"):
        saturated = np.where(y > 0, y * np.log(y / p), 0.0)
    assert loss["value"](p, y) == pytest.approx(np.mean(2 * (saturated - (y - p))))
    step = 1e-6
    numeric = np.array([(loss["value"](p + step * e, y) - loss["value"](p - step * e, y)) / (2 * step)
                        for e in np.eye(len(p))])
    np.testing.assert_allclose(loss["gradient"](p, y), numeric, rtol=1e-6, atol=1e-9)
    # A mean at zero is read at the machine epsilon, so the value stays finite.
    assert np.isfinite(loss["value"](np.r_[0.0, p[1:]], y))
    # The solver reaches the weights an exhaustive search over the simplex reaches.
    rng = np.random.default_rng(8)
    pm = np.column_stack([p * np.exp(rng.normal(0, 0.1, 8)), p * np.exp(rng.normal(0, 0.4, 8)),
                          np.full(8, y.mean())])
    w = simplex_weights(pm, y, loss)
    grid = [(a, b, 1 - a - b) for a in np.linspace(0, 1, 101) for b in np.linspace(0, 1, 101)
            if a + b <= 1 + 1e-12]
    best = min(loss["value"](pm @ np.array(g), y) for g in grid)
    assert loss["value"](pm @ w, y) <= best + 1e-9
