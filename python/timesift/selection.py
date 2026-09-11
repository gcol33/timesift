"""Choose the grain inside the training data, and score the whole procedure.

``grain_ladder`` fits every candidate against one fold map and reports the grid, so reading the
best grain off it and quoting that grain's score quotes a number the held-out units helped
choose. This does the choosing inside the training data instead.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import interval as ncv_module
from ._stats import t_ppf
from .interval import check_interval, interval_target
from .ladder import (Ladder, aligned_predictions, concat_ladders, grain_ladder, ladder_from_rows,
                     learner_dict, mean_se, paired_contrast, per_variable, place, score_arm)
from .learners import fit_learner
from .metrics import THRESHOLD_RULES, decision_threshold, tss
from .registry import METRICS, RESPONSES, metrics
from .representation import TimesiftSet, timesift_set
from .response import Folds, align_folds, as_response, fold_map

# The selected procedure is one arm like any other, so it carries an arm label of the ladder's own
# shape and every reader that splits on "|" keeps working.
SELECTED = "selected"
SELECTED_ARM = "selected|selected"
RULES = ("argmax", "coarsest_adequate")
# The label the TSS read at a cut learned on the inner folds is reported under. It is not a
# registered metric: a cut learned elsewhere is not a function of one cell's (y, p).
INNER_CUT_METRIC = "tss_inner_cut"


@dataclass
class Selection:
    """What a nested selection chose, what it scores, and what it was searched over."""

    selected: list[dict]
    estimate: list[dict]
    contrast: list[dict] | None
    candidates: list[dict]
    scores: Ladder
    inner: list[dict]
    metric: str
    response: str
    rule: str = "argmax"
    threshold: str | None = None
    thresholds: list[dict] | None = None
    cut_scores: Ladder | None = None
    interval: str = "variables"
    nested_cv: list[dict] | None = None
    final: dict | None = None

    def __repr__(self) -> str:  # pragma: no cover - display only
        picked: dict[str, int] = {}
        for row in self.selected:
            arm = f"{row['grain']}|{row['learner']}"
            picked[arm] = picked.get(arm, 0) + 1
        lines = [f"<timesift selection> {len(self.candidates)} candidates over "
                 f"{len(self.selected)} outer folds by {self.rule}",
                 "selected: " + ", ".join(f"{a} x{n}" for a, n in picked.items())]
        for row in self.estimate:
            mark = " <- selected on" if row["metric"] == self.metric else ""
            lines.append(f"  {row['metric']:<12} {row['score']:.4f} "
                         f"({row['lower']:.4f} to {row['upper']:.4f}, {row['interval']}){mark}")
        for kind in dict.fromkeys(row["interval"] for row in self.estimate):
            lines.append(f"  {kind}: {interval_target(kind)}")
        return "\n".join(lines)


def select_grain(x, y, learners, folds=None, inner=5, rule: str = "argmax",
                 threshold: str | None = None, interval: str = "variables", repeats: int = 1,
                 response: str = "presence_absence", metric=None,
                 compare: Ladder | None = None, control=None, seed: int = 1,
                 verbose: bool = True) -> Selection:
    """Choose the grain inside each outer fold's training units, then score the whole procedure.

    Within each outer fold the training units are split again, every candidate is fitted on part of
    them and scored on the rest, the best is refitted on the whole outer training set, and the
    outer test fold is predicted once. The estimate that comes back is therefore of the procedure
    including its choice of grain, which is what an ecologist applying it to a new site would run.

    What the estimate is of: the expected held-out score of the whole pipeline, selection included,
    on units drawn as these were. What it is not: the score of the winning grain. That is higher,
    by the amount selection buys itself, and the difference between the two is the quantity this
    function exists to keep out of a reported number.

    The cost is the ladder's, multiplied by the number of inner folds: ``v_outer * (v_inner *
    candidates + 1)`` fits.

    ``control`` is the ``train_control`` every neural learner trains under, in the inner search and
    in the refit alike; a learner carrying settings of its own overrides it on the ones it names.

    Inside each outer fold every candidate carries an inner score, the mean over variables of its
    per-variable mean over the inner folds, and a standard error, the standard deviation over the
    inner folds of the fold's own score divided by the square root of their number. ``rule``
    chooses among them. ``"argmax"`` takes the highest score, and on an exact tie the candidate
    declared first. ``"coarsest_adequate"`` is the one-standard-error rule (Breiman, Friedman,
    Olshen and Stone 1984; Hastie, Tibshirani and Friedman 2009, section 7.10) with coarseness in
    place of complexity: every candidate scoring at least the highest minus its standard error is
    adequate, and the one with the fewest bins wins, then the fewest channels, then the higher
    score, then the one declared first. A standard error that cannot be computed is taken as zero.

    ``threshold`` names a rule of ``decision_threshold`` (``"youden"``, the cut that maximises
    TSS, ``"kappa"`` or ``"prevalence"``). With it set, each outer fold learns one cut per variable
    on the inner out-of-fold predictions of the candidate it selected, which cover the outer
    training units and nothing else, freezes it, and reads the outer test fold's predictions at it
    with ``tss``. The estimate then carries a row ``tss_inner_cut``, ``thresholds`` the cut of
    every outer fold and variable, and ``cut_scores`` the per-cell rows.

    The estimate carries the interval across the response variables, which is the spread between
    the variables of this dataset rather than an interval for what the procedure would score on a
    new sample. ``interval="nested_cv"`` adds one that is, by the nested cross-validation of Bates,
    Hastie and Tibshirani (2024): each outer training set is cross-validated again over the
    remaining folds of the same map, over ``repeats`` fold maps, which gives the mean squared error
    of a cross-validation estimate; ``final`` then holds the procedure fitted on every unit, whose
    risk the interval is for. One repetition costs one fit of the procedure per unordered pair of
    outer folds.
    """
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    if threshold is not None and threshold not in THRESHOLD_RULES:
        raise ValueError(f"threshold must be None or one of {THRESHOLD_RULES}, got {threshold!r}")
    check_interval(interval)
    grains = timesift_set(x)
    units = grains.units
    spec = RESPONSES.get(response)
    y = spec["prepare"](as_response(y)).align(units)
    if folds is None:
        folds = fold_map(y)
    folds = Folds.coerce(folds, units).align(units)
    f = align_folds(folds, units)
    cells = spec["cells"](y, Folds(fold=f, units=units))
    # The estimate is reported under every registered metric, and one selected on has to be a row
    # of that table, so this is the one door a function of (y, p) does not go through.
    if callable(metric):
        raise TypeError("select_grain() reports the estimate under every registered metric, so "
                        "the one it selects on has to be registered. register_metric() takes a "
                        "function of (y, p).")
    metric = metric or spec["metric"]
    score = METRICS.get(metric)
    split = _inner_splitter(inner, folds.group)
    _check_compare(compare, metric, interval)
    # The candidate set keeps the order its grains and its learners were declared in, so which
    # candidate an exact tie on the inner score falls to does not depend on how the names sort.
    learners = learner_dict(learners)
    candidates = [dict(grain=w, learner=ln) for w in grains for ln in learners]
    if len(candidates) < 2:
        raise ValueError("selection needs at least two candidates; "
                         "got one grain and one learner")
    size = {c["grain"]: grains[c["grain"]].values.shape[1:] for c in candidates}
    ctx = dict(grains=grains, y=y, learners=learners, candidates=candidates, size=size, rule=rule,
               split=split, response=response, metric=metric, control=control, group=folds.group)

    levels = np.unique(f)
    p = np.full(y.values.shape, np.nan)
    chosen: list[dict] = []
    inner_rows: list[dict] = []
    cuts: dict = {}

    for i, k in enumerate(levels, start=1):
        train = np.flatnonzero(f != k)
        test = np.flatnonzero(f == k)
        y_train = y.take_units(train)

        once = _select_once(ctx, train, test, seed + i, fold=int(k))
        lad, grid, best, won = once["lad"], once["grid"], once["best"], once["won"]
        p[test] = once["pred"]

        # The cut is learned on the selected candidate's inner out-of-fold predictions, which
        # cover the outer training units and nothing else.
        if threshold is not None:
            oof = lad.predictions[f"{won['grain']}|{won['learner']}"]
            for j, v in enumerate(y.variables):
                cuts[(int(k), v)] = decision_threshold(y_train.values[:, j], oof[:, j], threshold)

        chosen.append(dict(fold=int(k), grain=won["grain"], learner=won["learner"],
                           inner_score=won["score"], inner_best=best["score"],
                           inner_se=best["se"], n_train=len(train), n_test=len(test)))
        inner_rows.extend(grid)
        if verbose:
            print(f"fold {k} of {len(levels)} selected {won['grain']}|{won['learner']} "
                  f"at {metric} {won['score']:.3f}")

    scores = ladder_from_rows(
        score_arm(SELECTED, SELECTED, y, p, f, levels, cells, score),
        predictions={SELECTED_ARM: p}, cells=cells, folds=Folds(fold=f, units=units),
        metric=metric, scorer=score, response=response, fits={})

    estimate = _nested_estimate(y, p, f, levels, cells, response)
    ncv = nested = final = None
    if interval == "nested_cv":
        # The procedure fitted once on every unit is the model the interval is for.
        whole = _select_once(ctx, np.arange(len(units)), np.empty(0, dtype=int), seed)
        final = dict(grain=whole["won"]["grain"], learner=whole["won"]["learner"],
                     fit=whole["fit"], inner=whole["grid"])
        maps = ncv_module.ncv_maps(y, f, folds.group, repeats, seed)
        if verbose:
            print(f"nested cross-validation: {len(maps)} repetition(s) of {len(levels)} "
                  "outer folds")

        def fit_predict(train, test, tag):
            return _select_once(ctx, train, test,
                                seed + 10007 * tag[0] + 101 * tag[1] + tag[2])["pred"]

        ncv = ncv_module.record(y, maps,
                                {SELECTED_ARM: ncv_module.keep_runs(
                                    ncv_module.ncv_collect(fit_predict, y, maps, p))})
        scores = replace(scores, ncv=ncv)
        nested = _nested_cv_estimate(ncv, SELECTED_ARM, response)
        estimate = estimate + [{k: row[k] for k in estimate[0]} for row in nested]
    thresholds = cut_scores = None
    if threshold is not None:
        thresholds = [dict(fold=k, variable=v, threshold=c, rule=threshold)
                      for (k, v), c in cuts.items()]
        cut_scores = ladder_from_rows(
            score_arm(SELECTED, SELECTED, y, p, f, levels, cells, tss, at=cuts),
            predictions={}, cells=cells, folds=Folds(fold=f, units=units),
            metric=INNER_CUT_METRIC, scorer=tss, response=response, fits={})
        estimate.append(_estimate_row(INNER_CUT_METRIC, cut_scores))

    return Selection(selected=chosen, estimate=estimate,
                     contrast=_selection_contrast(scores, compare, interval),
                     candidates=candidates, scores=scores, inner=inner_rows,
                     metric=metric, response=response, rule=rule, threshold=threshold,
                     thresholds=thresholds, cut_scores=cut_scores, interval=interval,
                     nested_cv=nested, final=final)


def _select_once(ctx: dict, train, test, split_seed: int, fold: int | None = None) -> dict:
    """One fit of the whole procedure: the inner search on the training units, the rule, the refit
    of the chosen candidate on all of them, and its predictions for the test units. The outer folds
    of ``select_grain`` and every fit of its nested cross-validation go through this."""
    y_train = ctx["y"].take_units(train)
    # The selector sees the training units and nothing else: the inner map is drawn on them, and
    # the representation it searches over is cut to them before any fitting happens.
    lad = grain_ladder(_subset(ctx["grains"], train), y_train, ctx["learners"],
                       folds=ctx["split"](y_train, split_seed, train), response=ctx["response"],
                       metric=ctx["metric"], control=ctx["control"], verbose=False)
    grid = _join_candidates(ctx["candidates"], lad.summary(), _inner_se(lad),
                            -1 if fold is None else fold)
    if not any(np.isfinite(g["score"]) for g in grid):
        where = "a fit of the nested cross-validation" if fold is None else f"fold {fold}"
        raise ValueError(f"no candidate scored inside the training data of {where}. Widen the "
                         "inner folds or drop the variables that cannot be scored.")
    best = _first_best(grid)
    won = _choose_candidate(grid, ctx["size"], ctx["rule"])
    # The refit is the inner ladder's own fitting path, so the procedure's held-out predictions
    # are the ones its chosen candidate would have made rather than a second fitting path's.
    m = ctx["grains"][won["grain"]]
    group = ctx["group"]
    fit = fit_learner(ctx["learners"][won["learner"]], m.take_units(train), y_train,
                      response=ctx["response"], control=ctx["control"],
                      group=None if group is None else tuple(group[u] for u in train))
    return dict(lad=lad, grid=grid, best=best, won=won, fit=fit,
                pred=aligned_predictions(fit, m, ctx["y"], test))


def _subset(grains: TimesiftSet, index) -> TimesiftSet:
    return timesift_set({w: m.take_units(index) for w, m in grains.items()})


def _join_candidates(candidates: list[dict], summary: list[dict], se: dict,
                     fold: int) -> list[dict]:
    """Every candidate's inner score and its standard error, in the order the candidates were
    declared.

    A candidate nothing was scorable for carries no score rather than being dropped, so a fold
    where one grain could not be read is visible instead of quietly narrowing the search.
    """
    found = {(r["grain"], r["learner"]): r for r in summary}
    out = []
    for candidate in candidates:
        key = (candidate["grain"], candidate["learner"])
        hit = found.get(key)
        out.append(dict(fold=fold, grain=candidate["grain"], learner=candidate["learner"],
                        score=hit["score"] if hit else float("nan"),
                        se=se.get(key, float("nan")),
                        n_variable=hit["n_variable"] if hit else 0))
    return out


def _inner_se(lad: Ladder) -> dict:
    """The standard error of each candidate's inner score: the spread over the inner folds of the
    fold's own score, the mean over the variables scored in it, divided by the square root of their
    number. Keyed by ``(grain, learner)``."""
    keep = ~np.isnan(lad.score)
    acc: dict = {}
    for w, ln, k, s in zip(lad.grain[keep], lad.learner[keep], lad.fold[keep], lad.score[keep]):
        acc.setdefault((str(w), str(ln)), {}).setdefault(int(k), []).append(float(s))
    out = {}
    for key, per_fold in acc.items():
        fold_scores = np.asarray([np.mean(v) for v in per_fold.values()])
        out[key] = (float(fold_scores.std(ddof=1) / np.sqrt(len(fold_scores)))
                    if len(fold_scores) > 1 else float("nan"))
    return out


def _first_best(grid: list[dict]) -> dict:
    """The highest inner score, and on an exact tie the candidate declared first."""
    best, best_value = None, -np.inf
    for candidate in grid:
        value = candidate["score"] if np.isfinite(candidate["score"]) else -np.inf
        if best is None or value > best_value:
            best, best_value = candidate, value
    return best


def _choose_candidate(grid: list[dict], size: dict, rule: str) -> dict:
    """The candidate a rule chooses. Under ``"coarsest_adequate"`` every candidate within one
    standard error of the highest score is adequate, and the coarsest of them wins: fewest bins,
    then fewest channels, then the higher score, then the order of declaration."""
    best = _first_best(grid)
    if rule == "argmax":
        return best
    tolerance = best["se"] if np.isfinite(best["se"]) else 0.0

    def value(c):
        return c["score"] if np.isfinite(c["score"]) else -np.inf

    adequate = [(i, c) for i, c in enumerate(grid) if value(c) >= value(best) - tolerance]
    return min(adequate, key=lambda ic: (size[ic[1]["grain"]][0], size[ic[1]["grain"]][1],
                                         -value(ic[1]), ic[0]))[1]


def _inner_splitter(inner, group=None):
    """The inner map is drawn on the outer training units alone, either by ``fold_map`` at a given
    count, dealing by the grouping the outer map carries, or by a splitter of the caller's own."""
    if callable(inner):
        return lambda y_train, seed, train: inner(y_train)
    if not isinstance(inner, (int, np.integer)) or isinstance(inner, bool) or int(inner) < 2:
        raise ValueError("`inner` is a number of folds of at least 2, or a function of the "
                         f"training response, got {inner!r}")

    def split(y_train, seed, train):
        if group is None:
            return fold_map(y_train, v=int(inner), seed=seed)
        return fold_map(y_train, v=int(inner), seed=seed, strata=1,
                        group=[group[i] for i in train])

    return split


# Every registered metric reads the same held-out predictions, so the estimate is reported under
# all of them and the choice of selection metric does not decide what may be quoted.
def _nested_estimate(y, p, f, levels, cells, response) -> list[dict]:
    out = []
    for name in metrics():
        rows = ladder_from_rows(
            score_arm(SELECTED, SELECTED, y, p, f, levels, cells, METRICS.get(name)),
            predictions={}, cells=cells, folds=Folds(fold=f, units=y.units), metric=name,
            scorer=METRICS.get(name), response=response, fits={})
        out.append(_estimate_row(name, rows))
    return out


def _estimate_row(name: str, rows: Ladder) -> dict:
    """One row of the estimate: the mean over variables of each variable's mean over its cells,
    and the interval across variables, on Student's t with one degree of freedom fewer than there
    are variables."""
    by_variable = list(per_variable(rows).values())
    level, se = mean_se(by_variable)
    n = len(by_variable)
    half = t_ppf(0.975, n - 1) * se if n > 1 else float("nan")
    return dict(metric=name, score=level, center=level, se=se, lower=level - half,
                upper=level + half, n_variable=n, interval="variables")


def _nested_cv_estimate(ncv: dict, arm: str, response: str) -> list[dict]:
    """The nested cross-validation rows of the estimate, one per registered metric, each read off
    the same stored predictions, with the estimator's own quantities beside them."""
    out = []
    for name in metrics():
        read = ncv_module.ncv_read(ncv, [arm], response, METRICS.get(name))
        out.append(dict(metric=name, score=read["estimate"], center=read["center"],
                        se=read["se"], lower=read["lower"], upper=read["upper"],
                        n_variable=read["n_variable"], interval="nested_cv",
                        **{k: read[k] for k in ("bias", "err_ncv", "mse_ncv", "se_naive",
                                                "repeats", "folds")}))
    return out


def _check_compare(compare, metric, interval: str = "variables") -> None:
    if compare is None:
        return
    if not isinstance(compare, Ladder):
        raise ValueError(f"`compare` is a grain_ladder() result, got "
                         f"{type(compare).__name__}")
    if compare.metric != metric:
        raise ValueError(f"`compare` is scored by {compare.metric} and the selection by {metric}. "
                         "Score both by the same metric before contrasting them.")
    if interval == "nested_cv" and compare.ncv is None:
        raise ValueError("`compare` was fitted without nested cross-validation, so its contrast "
                         "with the selection has none to read. Fit it with "
                         'grain_ladder(interval="nested_cv") on the same folds, `repeats` and '
                         "`seed`.")


def _selection_contrast(scores: Ladder, compare: Ladder | None, interval: str = "variables"):
    """One contrast row against each arm of ``compare``, under the across-variable interval and,
    where the selection was fitted with it, the nested cross-validation one too.

    The contrast is the ladder's, run on one table holding both arms, so the pairing rule and the
    interval come from ``paired_contrast`` rather than from a second copy of it here.
    """
    if compare is None:
        return None
    both = concat_ladders(scores, compare)
    seen = []
    for w, ln in zip(compare.grain, compare.learner):
        arm = f"{w}|{ln}"
        if arm not in seen:
            seen.append(arm)
    kinds = ["variables"] if interval == "variables" else ["variables", interval]
    return [paired_contrast(both, SELECTED_ARM, arm, kind) for kind in kinds for arm in seen]
