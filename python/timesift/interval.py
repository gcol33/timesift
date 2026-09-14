"""The interval for the procedure's risk.

The nested cross-validation estimate of the mean squared error of a cross-validation estimate, of
Bates, Hastie and Tibshirani (2024), Algorithm 1 and equation (10). One implementation, read by
``select_grain`` for the selection and by ``grain_ladder`` and ``paired_contrast`` for a fixed arm
and for the difference between two arms.

The paper writes the procedure's error as the mean of a per-unit loss. Every metric here is read on
a whole cell and averaged over variables, so the three places the paper averages losses are read as
follows, each reducing to the paper's quantity when the metric is a mean of per-unit losses:

===================== =========================================================================
``mean(e_out)``       the score of outer fold k: the mean over the variables scorable there
``mean(e_in)``        the cross-validation estimate on the outer training units, over the other
                      folds of the same map, averaged as the reported estimate is
``var(e_out)/|I_k|``  the delete-one jackknife variance of the fold score over its units, which
                      for a mean of per-unit losses is exactly ``var(e_out) / |I_k|``
===================== =========================================================================

and the naive standard error the mean squared error is bounded by is the jackknife standard error
of the reported estimate with every prediction held fixed.

The centre of the paper's interval is the nested estimate less the bias of equation (15), and the
width is the mean squared error of the plain cross-validation estimate: the correction is taken as
a shift with no spread of its own. It has one. The correction is ``1 + (K - 2) / K`` times the gap
between two cross-validation estimates on the same units, and where the sample is small or the
signal absent that gap moves from sample to sample by more than the estimate it corrects. So the
width here is the holdout identity applied to the corrected estimator itself: inside every outer
training set the same nested cross-validation is run one level down, over the unordered triples of
folds, which gives the bias-corrected estimate that training set alone would report, and term (a)
is the squared gap between that and the held-out fold's score. The paper's width is kept beside it
as ``se_bates``.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from ._stats import norm_ppf
from .registry import RESPONSES
from .response import Folds, fold_map

INTERVAL_KINDS = ("variables", "nested_cv")


def check_interval(interval: str) -> str:
    if interval not in INTERVAL_KINDS:
        raise ValueError(f"`interval` is one of {INTERVAL_KINDS}, got {interval!r}")
    return interval


def interval_target(interval: str) -> str:
    """What each interval is an interval for, in the words the printed objects use."""
    if interval == "variables":
        return ("the spread across the variables of this dataset, fitted and scored on these "
                "units and folds")
    return "the procedure's risk on a new sample of this size, by nested cross-validation"


def ncv_maps(y, f, group, repeats: int, seed: int) -> list[np.ndarray]:
    """The fold maps the repetitions are read on.

    The first is the map the estimate was computed on, so its outer predictions are the ones
    already made; the others are drawn by ``fold_map`` at the same number of folds, dealing by the
    grouping the first map carries.
    """
    if not isinstance(repeats, (int, np.integer)) or int(repeats) < 1:
        raise ValueError(f"`repeats` is a whole number of at least 1, got {repeats!r}")
    k = len(np.unique(f))
    if k < 4:
        raise ValueError("nested cross-validation needs at least four outer folds: the "
                         "cross-validation inside each outer training set is run once more inside "
                         f"each of its own training sets, and a map of {k} leaves that innermost "
                         "fit nothing to fit on")
    maps = [np.asarray(f)]
    for r in range(2, int(repeats) + 1):
        drawn = fold_map(y, v=k, seed=seed + 7919 * r, strata=5 if group is None else 1,
                         group=group)
        maps.append(np.asarray(drawn.fold))
    return maps


def ncv_collect(fit_predict, y, maps, first_outer) -> list[dict]:
    """Every prediction nested cross-validation reads, for one procedure over every map.

    ``fit_predict`` takes training and test unit indices and a tag naming the fit,
    ``(repetition, fold, fold, fold)`` with a zero for a fold the fit does not leave out, and
    returns the test units' predictions as an array in the order it was given them. Inside
    repetition r, the cross-validation on the training units of outer fold k fits on all but folds
    k and j and predicts fold j; that training set is the one the cross-validation of fold j fits
    on to predict fold k, so each unordered pair of folds is fitted once and predicts both. One
    level down, the cross-validation on the training units of fold k less fold j fits on all but
    folds k, j and l and predicts fold l, and that fit serves every ordering of the three, so each
    unordered triple is fitted once and predicts all three.
    """
    shape = y.values.shape
    runs = []
    for r, m in enumerate(maps, start=1):
        levels = [int(k) for k in np.unique(m)]
        inner = {k: np.full(shape, np.nan) for k in levels}
        for a, b in combinations(levels, 2):
            train = np.flatnonzero((m != a) & (m != b))
            test = np.flatnonzero((m == a) | (m == b))
            pred = fit_predict(train, test, (r, a, b, 0))
            at = {int(u): i for i, u in enumerate(test)}
            for k, other in ((a, b), (b, a)):
                rows = np.flatnonzero(m == other)
                inner[k][rows] = pred[[at[int(u)] for u in rows]]
        deep = {k: {j: np.full(shape, np.nan) for j in levels if j != k} for k in levels}
        for triple in combinations(levels, 3):
            train = np.flatnonzero(~np.isin(m, triple))
            test = np.flatnonzero(np.isin(m, triple))
            pred = fit_predict(train, test, (r, *triple))
            at = {int(u): i for i, u in enumerate(test)}
            for k in triple:
                for j in triple:
                    if j == k:
                        continue
                    (l,) = [x for x in triple if x not in (k, j)]
                    rows = np.flatnonzero(m == l)
                    deep[k][j][rows] = pred[[at[int(u)] for u in rows]]
        if r == 1:
            outer = np.asarray(first_outer, dtype=float)
        else:
            outer = np.full(shape, np.nan)
            for k in levels:
                test = np.flatnonzero(m == k)
                outer[test] = fit_predict(np.flatnonzero(m != k), test, (r, k, 0, 0))
        runs.append(dict(map=np.asarray(m), inner=inner, deep=deep, outer=outer))
    return runs


def ncv_seed(seed: int, tag) -> int:
    """The seed one fit of the nested cross-validation draws under, from its tag: distinct for
    every repetition and every set of folds left out, and the same in both languages."""
    return int(seed + 10007 * tag[0] + 101 * tag[1] + tag[2] + 3001 * tag[3])


def keep_runs(runs: list[dict]) -> list[dict]:
    """One arm's runs as a table keeps them."""
    return [{"inner": run["inner"], "deep": run["deep"], "outer": run["outer"]} for run in runs]


def record(y, maps, arms) -> dict:
    """What a table keeps of the nested cross-validation of its arms."""
    return dict(y=y, maps=maps, arms=arms)


def join(a, b):
    """Two tables' records, read on one set of maps, joined so a contrast can be read."""
    if a is None or b is None:
        return None
    same = len(a["maps"]) == len(b["maps"]) and all(
        np.array_equal(x, z) for x, z in zip(a["maps"], b["maps"]))
    if not same or not np.array_equal(a["y"].values, b["y"].values):
        raise ValueError("the two tables were cross-validated on different fold maps or "
                         "responses, so their nested cross-validation cannot be paired. Give both "
                         "the same response, folds, `repeats` and `seed`.")
    arms = dict(a["arms"])
    for name, runs in b["arms"].items():
        arms.setdefault(name, runs)
    return record(a["y"], a["maps"], arms)


def runs_of(ncv, arm: str) -> list[dict]:
    """The runs of one arm, rebuilt with the maps they were read on."""
    if ncv is None or arm not in ncv["arms"]:
        raise KeyError(f'no nested cross-validation for the arm "{arm}". Fit it with '
                       'interval="nested_cv" in grain_ladder() or select_grain().')
    return [dict(map=ncv["maps"][r], **ncv["arms"][arm][r]) for r in range(len(ncv["maps"]))]


def cell_values(y, arms, f, levels, cells, score) -> dict:
    """The cell values an estimate is averaged from, for one arm or for the difference of two."""
    from .ladder import score_arm
    rows = [score_arm(None, None, y, np.asarray(p), f, levels, cells, score) for p in arms]
    value = np.asarray(rows[0]["score"], dtype=float)
    if len(rows) == 2:
        value = value - np.asarray(rows[1]["score"], dtype=float)
    return dict(variable=np.asarray(rows[0]["variable"]), fold=np.asarray(rows[0]["fold"]),
                score=value)


def level_of(values: dict) -> float:
    """The estimate as every level of the package averages it: within a variable over its folds,
    then over variables."""
    keep = ~np.isnan(values["score"])
    if not keep.any():
        return float("nan")
    per = {}
    for v, s in zip(values["variable"][keep], values["score"][keep]):
        per.setdefault(str(v), []).append(float(s))
    return float(np.mean([np.mean(s) for s in per.values()]))


def jackknife_var(n: int, stat) -> float:
    """The delete-one jackknife variance of a statistic of ``n`` units."""
    if n < 2:
        return float("nan")
    theta = np.asarray([stat(i) for i in range(n)], dtype=float)
    return float((n - 1) / n * np.sum((theta - theta.mean()) ** 2))


def cell_on(y, arms, rows, variables, score) -> np.ndarray:
    """One arm's, or one difference's, value on each variable over the units given."""
    out = []
    for v in variables:
        j = y.variables.index(str(v))
        s = [score(y.values[rows, j], np.asarray(p)[rows, j]) for p in arms]
        out.append(s[0] - s[1] if len(s) == 2 else s[0])
    return np.asarray(out, dtype=float)


def fold_jackknife(y, arms, rows, variables, score) -> float:
    """The jackknife variance of a fold's score over its own units.

    A deletion that leaves a cell undefined, a presence-absence cell left with one class, keeps
    the cell's full value.
    """
    full = cell_on(y, arms, rows, variables, score)

    def leave(i):
        left = cell_on(y, arms, np.delete(rows, i), variables, score)
        gone = np.isnan(left)
        left[gone] = full[gone]
        return float(np.mean(left))

    return jackknife_var(len(rows), leave)


def jackknife_of(theta) -> float:
    """The delete-one jackknife variance from the leave-one-out values themselves."""
    theta = np.asarray(theta, dtype=float)
    n = len(theta)
    return float((n - 1) / n * np.sum((theta - theta.mean()) ** 2))


def estimate_jackknife(y, arms, m, values, score) -> float:
    """The jackknife variance of the ordinary estimate, every prediction held fixed."""
    return jackknife_of(level_jackknife(y, arms, m, values, score))


def level_jackknife(y, arms, m, values, score) -> np.ndarray:
    """A level with each unit left out in turn, every prediction held fixed: leaving unit i out
    changes only the cells of its own fold."""
    keep = ~np.isnan(values["score"])
    variable = values["variable"][keep]
    fold = values["fold"][keep].astype(int)
    value = values["score"][keep]
    sums, counts = {}, {}
    for v, s in zip(variable, value):
        sums[str(v)] = sums.get(str(v), 0.0) + float(s)
        counts[str(v)] = counts.get(str(v), 0) + 1
    names = list(sums)
    n = y.values.shape[0]
    theta = np.empty(n)
    for i in range(n):
        k = int(m[i])
        here = fold == k
        totals = dict(sums)
        if here.any():
            rows = np.asarray([u for u in np.flatnonzero(np.asarray(m) == k) if u != i])
            left = cell_on(y, arms, rows, variable[here], score)
            gone = np.isnan(left)
            left[gone] = value[here][gone]
            for v, was, now in zip(variable[here], value[here], left):
                totals[str(v)] = totals[str(v)] - float(was) + float(now)
        theta[i] = float(np.mean([totals[v] / counts[v] for v in names]))
    return theta


def ncv_level(y, preds, m, levels, cells_fun, score) -> tuple[float, dict]:
    """The cross-validation estimate a set of units reports from predictions made inside it, and
    the cell values it is the level of."""
    values = cell_values(y, preds, m, levels, cells_fun(y, m), score)
    return level_of(values), values


def ncv_bias(k: int, err_ncv: float, estimate: float) -> float:
    """The bias of equation (15) for a cross-validation of ``k`` folds: the gap between the nested
    estimate and the plain one, scaled from the two sample sizes they were fitted at to the
    whole."""
    return (1 + (k - 2) / k) * (err_ncv - estimate)


def ncv_terms(y, runs, cells_fun, score) -> dict:
    """The quantities of Algorithm 1 for one arm or one difference, over every repetition and outer
    fold, and the bias-corrected nested estimate the outer training set would report on its own
    (``e_hat``), from the cross-validation one level down; beside them the ordinary estimate on the
    first map, its naive jackknife standard error, and the naive jackknife standard error of the
    bias-corrected centre, which is a fixed linear combination of that estimate and the mean of
    the inner estimates and is jackknifed as one, unit by unit, with every prediction held
    fixed."""
    n = y.values.shape[0]
    terms = []
    theta_in = np.zeros(n)
    for r in range(len(runs[0])):
        m = np.asarray(runs[0][r]["map"])
        levels = [int(k) for k in np.unique(m)]
        cells_all = cells_fun(y, m)
        outer = [arm[r]["outer"] for arm in runs]
        out_values = cell_values(y, outer, m, levels, cells_all, score)
        for k in levels:
            train = np.flatnonzero(m != k)
            test = np.flatnonzero(m == k)
            y_in = y.take_units(train)
            m_in = m[train]
            inner = [arm[r]["inner"][k][train] for arm in runs]
            s_in, in_values = ncv_level(y_in, inner, m_in, np.unique(m_in), cells_fun, score)
            here = (out_values["fold"].astype(int) == k) & ~np.isnan(out_values["score"])
            if not here.any() or not np.isfinite(s_in):
                continue
            # The nested estimate inside the training set of fold k: the cross-validation on that
            # set less fold j, from the fits that left out k, j and one more, averaged over j as
            # the estimate on the whole sample averages over its outer folds.
            s_deep = []
            for j in levels:
                if j == k:
                    continue
                keep = np.flatnonzero((m != k) & (m != j))
                preds = [arm[r]["deep"][k][j][keep] for arm in runs]
                s_deep.append(ncv_level(y.take_units(keep), preds, m[keep],
                                        [x for x in levels if x not in (k, j)], cells_fun,
                                        score)[0])
            s_deep = np.asarray(s_deep, dtype=float)
            err_deep = float(np.mean(s_deep[np.isfinite(s_deep)]))
            terms.append(dict(repeat=r + 1, fold=k, s_in=s_in,
                              s_out=float(np.mean(out_values["score"][here])),
                              b=fold_jackknife(y, outer, test, out_values["variable"][here],
                                               score),
                              e_hat=err_deep - ncv_bias(len(levels) - 1, err_deep, s_in)))
            # The inner estimate with each of its training units left out; a unit of the test
            # fold leaves it as it is.
            left = np.full(n, s_in)
            left[train] = level_jackknife(y_in, inner, m_in, in_values, score)
            theta_in += left
    theta_in /= len(terms)

    m = np.asarray(runs[0][0]["map"])
    k = len(np.unique(m))
    outer = [arm[0]["outer"] for arm in runs]
    values = cell_values(y, outer, m, np.unique(m), cells_fun(y, m), score)
    keep = ~np.isnan(values["score"])
    theta_est = level_jackknife(y, outer, m, values, score)
    theta_center = (1 + (k - 2) / k) * theta_est - (k - 2) / k * theta_in
    return dict(terms=terms, estimate=level_of(values),
                se_naive=float(np.sqrt(jackknife_of(theta_est))),
                se_naive_center=float(np.sqrt(jackknife_of(theta_center))),
                folds=k,
                n_variable=len({str(v) for v in values["variable"][keep]}))


def ncv_interval(terms: dict, level: float = 0.95) -> dict:
    """Equation (10) of the paper, with the rescaling and the bounds of its section 4.3.2 and the
    bias estimate of its Appendix C: the centre is the nested estimate less the bias of fitting on
    fewer units than the whole sample, a mean squared error estimated on n (K - 1) / K units is
    rescaled by (K - 1) / K, and its square root is held between the naive standard error and
    sqrt(K) times it. The width the interval carries is the mean squared error of the
    bias-corrected estimate, read off ``e_hat`` and held between the naive standard error of the
    corrected centre and sqrt(K) times it; the paper's, read off the plain estimate ``s_in`` and
    held by the plain estimate's naive standard error, is ``se_bates``."""
    t = terms["terms"]
    k = terms["folds"]
    s_in = np.asarray([row["s_in"] for row in t])
    s_out = np.asarray([row["s_out"] for row in t])
    b = np.asarray([row["b"] for row in t])
    e_hat = np.asarray([row["e_hat"] for row in t])
    err_ncv = float(s_in.mean())

    def bounded(mse: float, floor: float) -> float:
        se = float(np.sqrt(max((k - 1) / k * mse, 0.0)))
        if np.isfinite(floor):
            se = float(min(max(se, floor), np.sqrt(k) * floor))
        return se

    mse_ncv = float(np.mean((s_in - s_out) ** 2) - np.mean(b))
    mse_center = float(np.mean((e_hat - s_out) ** 2) - np.mean(b))
    se = bounded(mse_center, terms["se_naive_center"])
    bias = ncv_bias(k, err_ncv, terms["estimate"])
    centre = err_ncv - bias
    half = norm_ppf(1 - (1 - level) / 2) * se
    return dict(estimate=terms["estimate"], center=centre, se=se, lower=centre - half,
                upper=centre + half, bias=bias, err_ncv=err_ncv, mse_ncv=mse_ncv,
                mse_center=mse_center, se_bates=bounded(mse_ncv, terms["se_naive"]),
                se_naive=terms["se_naive"], se_naive_center=terms["se_naive_center"],
                repeats=len({row["repeat"] for row in t}), folds=k,
                n_variable=terms["n_variable"])


def ncv_read(ncv, arms, response: str, score) -> dict:
    """The interval for one arm or one difference, read off a table's record under one metric."""
    spec = RESPONSES.get(response)

    def cells_fun(y, m):
        return spec["cells"](y, Folds(fold=np.asarray(m), units=y.units))

    runs = [runs_of(ncv, a) for a in arms]
    return ncv_interval(ncv_terms(ncv["y"], runs, cells_fun, score))
