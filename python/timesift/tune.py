"""A learner that chooses its own settings on the units it is fitted on."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .ladder import score_arm, variable_means
from .learners import Learner, _call_fit, flatten, get_learner
from .registry import TUNINGS, resolve_metric, tunings
from .response import Response, fold_map

__all__ = ["Tuned", "default_grids", "tune"]


@dataclass(frozen=True)
class Tuned:
    """What a tuned learner fitted: the model, the setting chosen, and every setting's score."""

    model: object
    chosen: dict
    table: list


def tune(learner, grid: dict | None = None, metric=None, n_inner: int = 5, seed: int = 1) -> Learner:
    """A learner that searches ``grid`` on the units it is fitted on and fits the best setting.

    The search is a cross-validation inside those units, so in a run the outer folds never see it:
    each fold chooses from its own training units, and the score it is then read at is not selected
    on. biomod2's ``BIOMOD_Tuning()`` searches a grid per algorithm by the same device.

    With ``grid`` left unset the learner is searched over the grid registered under its name by
    ``register_tuning``, which for the learners that ship is the one ``BIOMOD_Tuning()`` searches:
    ``mtry`` of a forest from 1 to the smaller of 10 and the number of columns; ``trees``,
    ``depth`` and ``shrinkage`` of a gbm-style boosting, ``shrinkage`` and ``colsample`` of the
    second-order one; ``degree`` and ``nprune`` of ``mars``; ``degree`` of ``discriminant``;
    ``regmult`` of ``maxent``; ``quantile`` of ``envelope``; and the layer width of ``mlp`` at 2, 4,
    6 and 8. biomod2's weight decay is a training setting here, which the control holds.

    A learner's settings are the ones it carries as ``params``. ``grid`` names some of them and
    gives the values to try, and the grid is every combination, the first setting varying fastest
    as in R, so a tie between two combinations falls to the same one on both sides. A setting is
    scored by the mean over the responses of the mean over inner folds of ``metric`` on the cells a
    score is defined on. The inner folds keep the grouping the outer
    fold map keeps whole. What was chosen is on the fitted model as ``chosen`` and ``table`` and in
    the ``settings`` column of a run's candidate table.
    """
    base = get_learner(learner)
    if not isinstance(n_inner, (int, np.integer)) or isinstance(n_inner, bool) or n_inner < 2:
        raise ValueError(f"`n_inner` is a number of folds of 2 or more, got {n_inner!r}")
    if grid is None and not TUNINGS.has(base.name):
        raise ValueError(f"no grid is registered for the {base.name} learner. Give `grid`, or "
                         f"register one with register_tuning(). Registered: "
                         f"{', '.join(tunings())}")
    if grid is not None:
        _points(base, grid)
    if metric is not None:
        resolve_metric(metric)

    def fit(x, y, head=None, control=None, group=None, variables=None, **given):
        points = _points(base, grid if grid is not None else _registered_grid(base, x))
        search = _search(base, points, x, y, head, control, group, variables, given, metric,
                         int(n_inner), seed)
        best = points[search["best"]]
        kept = {k: v for k, v in given.items() if k not in best}
        model = _call_fit(base, x, y, head, control, variables, group, {**kept, **best})
        return Tuned(model=model, chosen=best, table=search["table"])

    def predict(model, x):
        return base.predict(model.model, x)

    return Learner(name=base.name, fit=fit, predict=predict, needs=base.needs,
                   params=dict(base.params), data=base.data, reads=base.reads, multi=base.multi)


def _registered_grid(base: Learner, x) -> dict:
    grid = TUNINGS.get(base.name)
    return grid(base, x) if callable(grid) else grid


def default_grids() -> dict:
    """The grids ``BIOMOD_Tuning()`` searches, on the settings the learners carry."""
    def forest_grid(learner, x):
        return {"mtry": list(range(1, min(10, flatten(x).shape[1]) + 1))}

    def boosting_grid(learner, x):
        if learner.params.get("method") == "xgboost":
            return {"trees": [50], "depth": [1], "shrinkage": [0.3, 0.4], "min_leaf": [1.0],
                    "subsample": [0.5], "colsample": [0.6, 0.8], "gamma": [0.0]}
        return {"trees": [500, 1000, 2500], "depth": [2, 5, 8], "shrinkage": [0.001, 0.01, 0.1]}

    def mars_grid(learner, x):
        return {"degree": [1, 2], "nprune": list(range(2, max(21, 2 * flatten(x).shape[1] + 1) + 1))}

    return {"forest": forest_grid, "boosting": boosting_grid, "mars": mars_grid,
            "discriminant": {"degree": [1, 2]}, "maxent": {"regmult": [0.5, 1.0]},
            "envelope": {"quantile": [0.0, 0.0125, 0.025, 0.05, 0.1]},
            "mlp": {"hidden": [[2], [4], [6], [8]]}}


def _points(base: Learner, grid: dict) -> list:
    """Every combination of the grid, the first setting varying fastest."""
    if not isinstance(grid, dict) or not grid or any(not isinstance(k, str) for k in grid):
        raise ValueError("`grid` is a dict of the values to try for each setting")
    unknown = [k for k in grid if k not in base.params]
    if unknown:
        raise ValueError(f"the {base.name} learner carries no setting called "
                         f"{', '.join(unknown)}. Its settings are {', '.join(base.params)}")
    empty = [k for k, v in grid.items() if len(v) == 0]
    if empty:
        raise ValueError(f"`grid` holds no value for {', '.join(empty)}")
    names = list(grid)[::-1]
    return [dict(zip(names[::-1], combo[::-1]))
            for combo in itertools.product(*(grid[k] for k in names))]


def _search(base, points, x, y, head, control, group, variables, given, metric, inner, seed):
    units = tuple(x.units)
    frame = Response(values=np.asarray(y, dtype=np.float64), units=units,
                     variables=tuple(variables) if variables is not None
                     else tuple(f"y{j}" for j in range(np.asarray(y).shape[1])))
    folds = fold_map(frame, v=inner, seed=seed, strata=5 if group is None else 1,
                     group=None if group is None else list(group))
    f = np.asarray(folds.fold)
    levels = np.unique(f)
    cells = head["cells"](frame, folds)
    score = resolve_metric(metric, head["metric"])[0]
    scores = []
    for point in points:
        settings = {**{k: v for k, v in given.items() if k not in point}, **point}
        p = np.full(frame.values.shape, np.nan)
        for k in levels:
            train, test = np.flatnonzero(f != k), np.flatnonzero(f == k)
            model = _call_fit(base, x.take_units(train), frame.values[train], head, control,
                              frame.variables, None if group is None else [group[i] for i in train],
                              settings)
            p[test] = np.asarray(base.predict(model, x.take_units(test)), dtype=np.float64)
        rows = score_arm(None, None, frame, p, f, levels, cells, score)
        ok = [bool(s) and np.isfinite(v) for s, v in zip(rows["scorable"], rows["score"])]
        per = variable_means([0] * sum(ok), [v for v, k in zip(rows["variable"], ok) if k],
                             [v for v, k in zip(rows["score"], ok) if k])
        scores.append(float(np.mean(list(per[0].values()))) if per else float("nan"))
    if all(np.isnan(scores)):
        raise ValueError("no setting of the grid has a cell to be scored on: every response needs "
                         "both classes on each side of an inner split. Fewer inner folds, or a "
                         "response present somewhere")
    table = [dict(settings=label(p), score=s) for p, s in zip(points, scores)]
    return {"best": int(np.nanargmax(scores)), "table": table}


def label(point: dict) -> str:
    """A setting as the candidate table prints it."""
    return ", ".join(f"{k} = {_describe(v)}" for k, v in point.items())


def _describe(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, (list, tuple, np.ndarray)):
        return "/".join(_describe(e) for e in v)
    return f"{v:g}" if isinstance(v, (float, np.floating)) else str(v)
