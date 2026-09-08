"""Choose the grain inside the training data, and score the whole procedure.

``grain_ladder`` fits every candidate against one fold map and reports the grid, so reading the
best grain off it and quoting that grain's score quotes a number the held-out units helped
choose. This does the choosing inside the training data instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ladder import (Ladder, concat_ladders, ladder_from_rows, learner_dict, mean_se,
                     paired_contrast, per_variable, score_arm, grain_ladder)
from .learners import fit_learner
from .registry import METRICS, RESPONSES, metrics
from .representation import TimesiftSet, timesift_set
from .response import Folds, align_folds, as_response, fold_map

# The selected procedure is one arm like any other, so it carries an arm label of the ladder's own
# shape and every reader that splits on "|" keeps working.
SELECTED = "selected"
SELECTED_ARM = "selected|selected"


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

    def __repr__(self) -> str:  # pragma: no cover - display only
        picked: dict[str, int] = {}
        for row in self.selected:
            arm = f"{row['grain']}|{row['learner']}"
            picked[arm] = picked.get(arm, 0) + 1
        lines = [f"<timesift selection> {len(self.candidates)} candidates over "
                 f"{len(self.selected)} outer folds",
                 "selected: " + ", ".join(f"{a} x{n}" for a, n in picked.items())]
        for row in self.estimate:
            mark = " <- selected on" if row["metric"] == self.metric else ""
            lines.append(f"  {row['metric']:<12} {row['score']:.4f} "
                         f"(se {row['se']:.4f}, {row['n_variable']} variables){mark}")
        return "\n".join(lines)


def select_grain(x, y, learners, folds=None, inner=5, response: str = "presence_absence",
                 metric=None, compare: Ladder | None = None, control=None, seed: int = 1,
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
    """
    grains = timesift_set(x)
    units = grains.units
    spec = RESPONSES.get(response)
    y = spec["prepare"](as_response(y)).align(units)
    if folds is None:
        folds = fold_map(y)
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
    split = _inner_splitter(inner)
    _check_compare(compare, metric)
    # The candidate set keeps the order its grains and its learners were declared in, so which
    # candidate an exact tie on the inner score falls to does not depend on how the names sort.
    learners = learner_dict(learners)
    candidates = [dict(grain=w, learner=ln) for w in grains for ln in learners]
    if len(candidates) < 2:
        raise ValueError("selection needs at least two candidates; "
                         "got one grain and one learner")

    levels = np.unique(f)
    p = np.full(y.values.shape, np.nan)
    row = {u: i for i, u in enumerate(units)}
    column = {v: j for j, v in enumerate(y.variables)}
    chosen: list[dict] = []
    inner_rows: list[dict] = []

    for i, k in enumerate(levels, start=1):
        train = np.flatnonzero(f != k)
        test = np.flatnonzero(f == k)
        y_train = y.take_units(train)

        # The selector sees the outer training units and nothing else: the inner map is drawn on
        # them, and the representation it searches over is cut to them before any fitting happens.
        lad = grain_ladder(_subset(grains, train), y_train, learners,
                            folds=split(y_train, seed + i), response=response, metric=metric,
                            control=control, verbose=False)
        grid = _join_candidates(candidates, lad.summary(), int(k))
        if not any(np.isfinite(g["score"]) for g in grid):
            raise ValueError(f"no candidate scored inside the training data of fold {k}. Widen "
                             "the inner folds or drop the variables that cannot be scored.")
        won = _first_best(grid)

        fit = fit_learner(learners[won["learner"]], grains[won["grain"]].take_units(train),
                          y_train, response=response, control=control)
        held = grains[won["grain"]].take_units(test)
        predicted = fit.predict(held)
        for a, unit in enumerate(held.units):
            for b, variable in enumerate(fit.variables):
                p[row[unit], column[variable]] = predicted[a, b]

        chosen.append(dict(fold=int(k), grain=won["grain"], learner=won["learner"],
                           inner_score=won["score"], n_train=len(train), n_test=len(test)))
        inner_rows.extend(grid)
        if verbose:
            print(f"fold {k} of {len(levels)} selected {won['grain']}|{won['learner']} "
                  f"at {metric} {won['score']:.3f}")

    scores = ladder_from_rows(
        score_arm(SELECTED, SELECTED, y, p, f, levels, cells, score),
        predictions={SELECTED_ARM: p}, cells=cells, folds=Folds(fold=f, units=units),
        metric=metric, scorer=score, response=response, fits={})

    return Selection(selected=chosen,
                     estimate=_nested_estimate(y, p, f, levels, cells, response),
                     contrast=_selection_contrast(scores, compare),
                     candidates=candidates, scores=scores, inner=inner_rows,
                     metric=metric, response=response)


def _subset(grains: TimesiftSet, index) -> TimesiftSet:
    return timesift_set({w: m.take_units(index) for w, m in grains.items()})


def _join_candidates(candidates: list[dict], summary: list[dict], fold: int) -> list[dict]:
    """Every candidate's inner score, in the order the candidates were declared.

    A candidate nothing was scorable for carries no score rather than being dropped, so a fold
    where one grain could not be read is visible instead of quietly narrowing the search.
    """
    found = {(r["grain"], r["learner"]): r for r in summary}
    out = []
    for candidate in candidates:
        hit = found.get((candidate["grain"], candidate["learner"]))
        out.append(dict(fold=fold, grain=candidate["grain"], learner=candidate["learner"],
                        score=hit["score"] if hit else float("nan"),
                        n_variable=hit["n_variable"] if hit else 0))
    return out


def _first_best(grid: list[dict]) -> dict:
    """The highest inner score, and on an exact tie the candidate declared first."""
    best, best_value = None, -np.inf
    for candidate in grid:
        value = candidate["score"] if np.isfinite(candidate["score"]) else -np.inf
        if best is None or value > best_value:
            best, best_value = candidate, value
    return best


def _inner_splitter(inner):
    """The inner map is drawn on the outer training units alone, either by ``fold_map`` at a given
    count or by a splitter of the caller's own."""
    if callable(inner):
        return lambda y_train, seed: inner(y_train)
    if not isinstance(inner, (int, np.integer)) or isinstance(inner, bool) or int(inner) < 2:
        raise ValueError("`inner` is a number of folds of at least 2, or a function of the "
                         f"training response, got {inner!r}")
    return lambda y_train, seed: fold_map(y_train, v=int(inner), seed=seed)


# Every registered metric reads the same held-out predictions, so the estimate is reported under
# all of them and the choice of selection metric does not decide what may be quoted.
def _nested_estimate(y, p, f, levels, cells, response) -> list[dict]:
    out = []
    for name in metrics():
        rows = ladder_from_rows(
            score_arm(SELECTED, SELECTED, y, p, f, levels, cells, METRICS.get(name)),
            predictions={}, cells=cells, folds=Folds(fold=f, units=y.units), metric=name,
            scorer=METRICS.get(name), response=response, fits={})
        by_variable = list(per_variable(rows).values())
        level, se = mean_se(by_variable)
        out.append(dict(metric=name, score=level, se=se, n_variable=len(by_variable)))
    return out


def _check_compare(compare, metric) -> None:
    if compare is None:
        return
    if not isinstance(compare, Ladder):
        raise ValueError(f"`compare` is a grain_ladder() result, got "
                         f"{type(compare).__name__}")
    if compare.metric != metric:
        raise ValueError(f"`compare` is scored by {compare.metric} and the selection by {metric}. "
                         "Score both by the same metric before contrasting them.")


def _selection_contrast(scores: Ladder, compare: Ladder | None):
    """One contrast row against each arm of ``compare``.

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
    return [paired_contrast(both, SELECTED_ARM, arm) for arm in seen]
