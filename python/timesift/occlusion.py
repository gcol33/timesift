"""What part of the record a fitted model reads, and what an already-reduced feature table is."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .registry import RESPONSES, resolve_metric
from .representation import TimesiftMatrix
from .response import Response, align_folds, as_response

__all__ = ["feature_matrix"]


def feature_matrix(m, units=None, features=None, label: str = "features") -> TimesiftMatrix:
    """Bring an already-reduced feature table into a ladder as a one-channel representation.

    It carries no time axis, because it has none: the reduction already happened, elsewhere, and
    what reaches the model is a list of numbers per unit. That is the whole point of comparing
    against it.
    """
    m = np.asarray(m, dtype=np.float64)
    n_u, n_f = m.shape
    units = tuple(units) if units is not None else tuple(str(i) for i in range(n_u))
    features = tuple(features) if features is not None else tuple(f"f{j}" for j in range(n_f))
    empty = np.full(n_f, np.datetime64("NaT"), dtype="datetime64[s]")
    return TimesiftMatrix(values=m.reshape(n_u, n_f, 1), units=units, bins=features,
                        stats=(label,), grain=label, year_start="", bin_start=empty,
                        bin_end=empty, bin_n=np.zeros((n_u, n_f), dtype=np.int64),
                        bin_partial=np.zeros(n_f, dtype=bool))


def ladder_occlusion(ladder, x, y: Response, arm: str, over: str = "bin",
                  substitute: str = "permute", metric=None,
                  permutations: int = 20, seed: int = 1):
    """Hold one bin of the record back at a time and record the fall in score as its weight.

    Nothing is refitted: the models kept by ``grain_ladder(keep_fits=True)`` are the ones read.
    What a held-back bin is replaced by decides what the weight means, so the substitute is part of
    the answer: ``permute`` keeps the observed readings and cuts only the link between a reading
    and its unit, ``fold_mean`` removes all between-unit variation, ``unit_mean`` keeps how warm a
    unit is and removes only that bin's departure from it.
    """
    if not ladder.fits:
        raise ValueError("this ladder kept no fits; refit with grain_ladder(..., keep_fits=True)")
    if "|" in arm:
        grain, learner = arm.split("|", 1)
        m = x if isinstance(x, TimesiftMatrix) else dict(x)[grain]
    else:
        learner, m = arm, x
        if not isinstance(m, TimesiftMatrix):
            raise ValueError(f'"{arm}" names no grain, so it can only be read against one '
                             "representation rather than a set of them")
        grain = m.grain
    prefix = f"{grain}|{learner}|"
    fits = {int(key[len(prefix):]): fit for key, fit in ladder.fits.items()
            if key.startswith(prefix)}
    if not fits:
        raise KeyError(f'this ladder kept no fits for the arm "{grain}|{learner}"')
    # Left unset the profile is read by the metric the fit was scored under, so a weight is a fall
    # in the number the summary reports rather than in a second one.
    return occlusion_profile(fits, m, y, ladder.folds, over=over, substitute=substitute,
                             metric=ladder.scorer if metric is None else metric,
                             response=ladder.response, permutations=permutations, seed=seed)


def occlusion_profile(fits: dict, m: TimesiftMatrix, y, folds, over: str = "bin",
                      substitute: str = "permute", metric="roc_auc",
                      response: str = "presence_absence",
                      permutations: int = 20, seed: int = 1):
    """The occlusion itself: one fitted model per fold, read on the units that model held out.

    A ladder reaches it through ``ladder_occlusion`` and a fitted ``timesift`` through ``occlusion``,
    so what a withheld bin costs has one definition whichever door it is asked through.
    """
    if over not in ("bin", "channel"):
        raise ValueError(f"`over` must be 'bin' or 'channel', got {over!r}")
    if substitute not in ("permute", "fold_mean", "unit_mean"):
        raise ValueError(f"unknown substitute {substitute!r}")

    # The response reaches its own head's prepare, as it does everywhere else, so a head that is
    # not presence-absence can be occluded too.
    y = RESPONSES.get(response)["prepare"](as_response(y)).align(m.units)
    f = align_folds(folds, m.units)
    score = resolve_metric(metric)[0]

    n_parts = m.values.shape[1] if over == "bin" else m.values.shape[2]
    labels = m.bins if over == "bin" else m.stats
    rng = np.random.default_rng(seed)

    weight = np.full((n_parts, len(y.variables)), np.nan)
    counted = np.zeros((n_parts, len(y.variables)))
    for k in np.unique(f):
        fit = fits.get(int(k))
        if fit is None:
            continue
        test = np.flatnonzero(f == k)
        train = np.flatnonzero(f != k)
        sub = m.take_units(test)
        # The baseline is one prediction for the fold, not one per variable: the model reads the
        # whole block and an encoder would otherwise be rebuilt from its arrays once per response.
        full = fit.predict(sub)
        base = np.asarray([score(y.values[test, j], full[:, j])
                           for j in range(len(y.variables))])
        for i in range(n_parts):
            draws = permutations if substitute == "permute" else 1
            acc = np.zeros((draws, len(y.variables)))
            for r in range(draws):
                occluded = _occlude(m, sub, train, i, over, substitute, rng)
                p = fit.predict(occluded)
                acc[r] = [score(y.values[test, j], p[:, j]) for j in range(len(y.variables))]
            fall = base - acc.mean(axis=0)
            ok = np.isfinite(fall)
            weight[i, ok] = np.where(np.isnan(weight[i, ok]), 0, weight[i, ok]) + fall[ok]
            counted[i, ok] += 1

    with np.errstate(invalid="ignore"):
        weight = weight / np.where(counted > 0, counted, np.nan)
    return {"part": list(labels), "variable": list(y.variables), "weight": weight}


def _occlude(m: TimesiftMatrix, sub: TimesiftMatrix, train, i, over, substitute, rng) -> TimesiftMatrix:
    values = sub.values.copy()
    n = values.shape[0]
    if over == "channel":
        if substitute == "permute":
            values[:, :, i] = values[rng.permutation(n), :, i]
        elif substitute == "fold_mean":
            values[:, :, i] = m.values[train, :, i].mean(axis=0)[None, :]
        else:
            values[:, :, i] = values[:, :, i].mean(axis=1)[:, None]
        return replace(sub, values=values)
    for ch in range(values.shape[2]):
        if substitute == "permute":
            values[:, i, ch] = values[rng.permutation(n), i, ch]
        elif substitute == "fold_mean":
            values[:, i, ch] = m.values[train, i, ch].mean()
        else:
            values[:, i, ch] = values[:, :, ch].mean(axis=1)
    return replace(sub, values=values)
