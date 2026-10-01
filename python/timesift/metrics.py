"""Threshold metrics on held-out predictions.

A cut may only fall between distinct predictions: units sharing a prediction are decided together,
so the same score comes back whatever order they arrived in and both language sides read every
threshold metric off the same rule.
"""

from __future__ import annotations

import math

import numpy as np

THRESHOLD_RULES = ("youden", "kappa", "prevalence", "mpa")


def _labels(y) -> np.ndarray:
    """The response as 0/1 integers, refused where it is not one: checked on the values as given,
    because coercing first would read 0.6 as 0 and pass a response that was never binary."""
    y = np.asarray(y, dtype=np.float64)
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("`y` must be presence-absence, 0/1 or logical.")
    return y.astype(np.int64)


def _sweep(y, p):
    y = _labels(y)
    p = np.asarray(p, dtype=np.float64)
    if y.shape != p.shape:
        raise ValueError("`y` and `p` must be the same length")
    if not np.isfinite(p).all():
        return None
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(-p, kind="stable")
    ys, ps = y[order], p[order]
    keep = np.empty(len(ps), dtype=bool)
    keep[:-1] = ps[:-1] != ps[1:]
    keep[-1] = True
    return dict(thr=ps[keep], tp=np.cumsum(ys)[keep].astype(float),
                fp=np.cumsum(1 - ys)[keep].astype(float), n_pos=n_pos, n_neg=n_neg)


def tss(y, p, threshold=None) -> float:
    """Sensitivity plus specificity minus one, at the threshold that maximises it.

    Given a ``threshold`` learned elsewhere, the score is read at that cut instead, presence being
    predicted at ``p >= threshold``.
    """
    if threshold is not None:
        s = _sweep(y, p)
        if s is None or not np.isfinite(threshold):
            return float("nan")
        y = _labels(y)
        hit = np.asarray(p, dtype=np.float64) >= threshold
        return float(hit[y == 1].mean() - hit[y == 0].mean())
    s = _sweep(y, p)
    if s is None:
        return float("nan")
    return float(np.max(s["tp"] / s["n_pos"] - s["fp"] / s["n_neg"]))


def average_precision(y, p) -> float:
    """The area under the precision-recall curve, as the step sum over the distinct predictions.

    At each distinct prediction, the precision of calling every unit at or above it a presence,
    weighted by the share of presences that cut adds. Units sharing a prediction enter together.
    Its floor is the prevalence rather than one half, which makes it the reading of how well
    presences are ranked above absences where presences are rare.
    """
    s = _sweep(y, p)
    if s is None:
        return float("nan")
    precision = s["tp"] / (s["tp"] + s["fp"])
    gained = np.diff(np.concatenate(([0.0], s["tp"]))) / s["n_pos"]
    return float(np.sum(gained * precision))


def roc_auc(y, p) -> float:
    """The area under the ROC curve, as the rank sum of the presences. Ties take the average rank."""
    y = _labels(y)
    p = np.asarray(p, dtype=np.float64)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0 or not np.isfinite(p).all():
        return float("nan")
    ranks = _average_ranks(p)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _average_ranks(x) -> np.ndarray:
    """Ranks from 1, each run of tied values taking the average of the ranks it spans."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    xs = x[order]
    start = 0
    for i in range(1, len(xs) + 1):
        if i == len(xs) or xs[i] != xs[start]:
            ranks[order[start:i]] = (start + i + 1) / 2
            start = i
    return ranks


# Each metric of a two-by-two table of decisions against observations, as a function of its four
# cells: hits `tp`, false alarms `fp`, misses `fn` and correct negatives `tn`.
TABLE_METRICS = {
    "pod": lambda tp, fp, fn, tn: tp / (tp + fn),
    "pofd": lambda tp, fp, fn, tn: fp / (fp + tn),
    "far": lambda tp, fp, fn, tn: fp / (tp + fp),
    "sr": lambda tp, fp, fn, tn: tp / (tp + fp),
    "accuracy": lambda tp, fp, fn, tn: (tp + tn) / (tp + fp + fn + tn),
    "bias": lambda tp, fp, fn, tn: (tp + fp) / (tp + fn),
    "or": lambda tp, fp, fn, tn: tp * tn / (fn * fp),
    "orss": lambda tp, fp, fn, tn: (tp * tn - fn * fp) / (tp * tn + fn * fp),
    "csi": lambda tp, fp, fn, tn: tp / (tp + fn + fp),
    "ets": lambda tp, fp, fn, tn: (
        (tp - (tp + fn) * (tp + fp) / (tp + fp + fn + tn))
        / (tp + fn + fp - (tp + fn) * (tp + fp) / (tp + fp + fn + tn))),
}


def table_metric(y, p, metric: str, rule: str = "youden", threshold=None,
                 perc: float = 0.9) -> float:
    """A metric of the two-by-two table of decisions against observations.

    ``metric`` is one of ``pod``, ``pofd``, ``far``, ``sr``, ``accuracy``, ``bias``, ``or``,
    ``orss``, ``csi`` and ``ets``, biomod2's evaluation statistics. The cut is the one ``rule`` of
    ``decision_threshold`` selects, or a ``threshold`` learned elsewhere, presence being predicted
    at ``p >= cut``. A value the table does not define, a zero denominator, is NaN, as is a cell
    of one class.
    """
    if metric not in TABLE_METRICS:
        raise ValueError(f"metric must be one of {tuple(TABLE_METRICS)}, got {metric!r}")
    if rule not in THRESHOLD_RULES:
        raise ValueError(f"rule must be one of {THRESHOLD_RULES}, got {rule!r}")
    cut = decision_threshold(y, p, rule, perc=perc) if threshold is None else threshold
    if _sweep(y, p) is None or not np.isfinite(cut):
        return float("nan")
    y = _labels(y)
    hit = np.asarray(p, dtype=np.float64) >= cut
    cells = dict(tp=float(np.sum(hit & (y == 1))), fp=float(np.sum(hit & (y == 0))),
                 fn=float(np.sum(~hit & (y == 1))), tn=float(np.sum(~hit & (y == 0))))
    try:
        value = TABLE_METRICS[metric](**cells)
    except ZeroDivisionError:
        return float("nan")
    return float(value) if np.isfinite(value) else float("nan")


def boyce_index(y, p, resolution: int = 100, width: float = 0.1) -> float:
    """The continuous Boyce index (Hirzel et al. 2006).

    A window of ``width`` times the range of the predictions slides over that range in
    ``resolution + 1`` equal steps. In each, the share of presences inside divided by the share of
    all units inside is the predicted-to-expected ratio, undefined where no unit falls inside, and
    the index is the Spearman correlation of that ratio with the window's midpoint. The units of
    ``p`` are the background. The windows are closed at both ends and the ratio is not thinned of
    repeated values. NaN where the cell defines none, the predictions are all equal, the ratio is
    defined in fewer than three windows or takes one value.
    """
    s = _sweep(y, p)
    if s is None:
        return float("nan")
    y = _labels(y)
    p = np.asarray(p, dtype=np.float64)
    lo, hi = float(p.min()), float(p.max())
    if hi == lo:
        return float("nan")
    w = (hi - lo) * width
    steps = np.arange(resolution + 1, dtype=np.float64)
    start = lo + (hi - w - lo) * steps / resolution
    end = start + w
    end[-1] = hi

    def inside(x):
        return ((x[None, :] >= start[:, None]) & (x[None, :] <= end[:, None])).sum(axis=1).astype(float)

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (inside(p[y == 1]) / np.sum(y == 1)) / (inside(p) / len(p))
    keep = np.isfinite(ratio)
    if keep.sum() < 3 or len(np.unique(ratio[keep])) < 2:
        return float("nan")
    a = _average_ranks(((start + end) / 2)[keep])
    b = _average_ranks(ratio[keep])
    return float(np.corrcoef(a, b)[0, 1])


def _r_squared(y, p):
    sst = np.sum((y - y.mean()) ** 2)
    return 1 - np.sum((y - p) ** 2) / sst if sst > 0 else float("nan")


def _pearson(y, p):
    return float(np.corrcoef(y, p)[0, 1]) if y.std() > 0 and p.std() > 0 else float("nan")


REGRESSION_METRICS = {
    "r_squared": _r_squared,
    "pearson": _pearson,
    "rmse": lambda y, p: float(np.sqrt(np.mean((y - p) ** 2))),
    "mse": lambda y, p: float(np.mean((y - p) ** 2)),
    "mae": lambda y, p: float(np.mean(np.abs(y - p))),
    "max_error": lambda y, p: float(np.max(np.abs(y - p))),
}


def regression_metric(y, p, metric: str) -> float:
    """A metric of a numeric response and its predictions, as biomod2 reads an abundance model.

    ``metric`` is ``r_squared`` (``1 - sum(e**2) / sum((y - mean(y))**2)``, NaN where ``y`` is
    constant), ``pearson``, ``rmse``, ``mse``, ``mae`` or ``max_error``. A comparison reads the
    highest score as the best, so the four errors are registered as ``neg_rmse``, ``neg_mse``,
    ``neg_mae`` and ``neg_max_error`` with their sign reversed.
    """
    if metric not in REGRESSION_METRICS:
        raise ValueError(f"metric must be one of {tuple(REGRESSION_METRICS)}, got {metric!r}")
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if y.shape != p.shape:
        raise ValueError("`y` and `p` must be the same length")
    if not len(y) or np.isnan(y).any() or not np.isfinite(p).all():
        return float("nan")
    value = float(REGRESSION_METRICS[metric](y, p))
    return value if np.isfinite(value) else float("nan")


def _ordinal_accuracy(m):
    return float(np.trace(m) / m.sum())


def _recall(m):
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.nansum(np.diag(m) / m.sum(axis=0)) / m.shape[0])


def _precision(m):
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.nansum(np.diag(m) / m.sum(axis=1)) / m.shape[0])


def _f1(m):
    r, q = _recall(m), _precision(m)
    return 2 * q * r / (q + r) if q + r > 0 else float("nan")


ORDINAL_METRICS = {"accuracy": _ordinal_accuracy, "recall": _recall, "precision": _precision,
                   "f1": _f1}


def ordinal_metric(y, p, metric: str) -> float:
    """A metric of ordinal classes, as biomod2 reads a model of an ordinal response.

    The response is a column of whole-number classes, the model predicts a number on the same
    scale, and each prediction is read as the observed class nearest to it, the lower class on a
    tie. With ``m[i, j]`` the units of observed class ``j`` predicted as class ``i``, over the
    ``k`` classes observed in the cell: ``accuracy`` is ``trace(m) / sum(m)``, ``recall`` the mean
    over the ``k`` classes of ``m[j, j]`` over the units of class ``j``, ``precision`` the mean of
    ``m[i, i]`` over the units predicted as ``i``, and ``f1`` is ``2 P R / (P + R)`` of the two
    means. A class with no unit, or in which nothing is predicted, adds zero to its mean. They are
    registered as ``ordinal_accuracy``, ``ordinal_recall``, ``ordinal_precision`` and
    ``ordinal_f1``.
    """
    if metric not in ORDINAL_METRICS:
        raise ValueError(f"metric must be one of {tuple(ORDINAL_METRICS)}, got {metric!r}")
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if y.shape != p.shape:
        raise ValueError("`y` and `p` must be the same length")
    if not len(y) or np.isnan(y).any() or not np.isfinite(p).all():
        return float("nan")
    classes = np.unique(y)
    predicted = np.argmin(np.abs(classes[None, :] - p[:, None]), axis=1)
    observed = np.searchsorted(classes, y)
    m = np.zeros((len(classes), len(classes)))
    np.add.at(m, (predicted, observed), 1.0)
    value = ORDINAL_METRICS[metric](m)
    return value if np.isfinite(value) else float("nan")


def decision_threshold(y, p=None, rule: str = "youden", candidate: str = "ensemble",
                       perc: float = 0.9):
    """The probability cut a rule selects. Presence is predicted at ``p >= threshold``.

    ``"mpa"`` is the minimum predicted area rule: the highest cut that still predicts presence at a
    share ``perc`` of the observed presences.

    Given a ``timesift()`` fit in place of ``y`` and no ``p``, one cut per response, learned from
    ``candidate``'s out-of-fold predictions of the fit's own targets, as a dict keyed by the
    response; that is the cut ``predict(type="binary")`` applies to new targets.
    """
    if rule not in THRESHOLD_RULES:
        raise ValueError(f"rule must be one of {THRESHOLD_RULES}, got {rule!r}")
    from .fit import Timesift
    if isinstance(y, Timesift):
        if p is not None:
            raise TypeError("a fit carries its own held-out predictions; `p` is not given with one")
        observed = np.asarray(y.y.values, dtype=np.float64)
        if not np.isin(observed, (0.0, 1.0)).all():
            raise ValueError("a cut is learned on a presence-absence response, and this fit was "
                             f'made under the "{y.response}" head.')
        held = np.asarray(y._held_out(candidate), dtype=np.float64)
        return {v: decision_threshold(observed[:, j], held[:, j], rule)
                for j, v in enumerate(y.y.variables)}
    if p is None:
        raise TypeError("`p` is the predictions for the units of `y`")
    s = _sweep(y, p)
    if s is None:
        return float("nan")
    if rule == "mpa":
        if not (np.isfinite(perc) and 0 < perc <= 1):
            raise ValueError("`perc` is a share of the presences, in (0, 1].")
        held = np.sort(np.asarray(p, dtype=np.float64)[_labels(y) == 1])[::-1]
        return float(held[max(1, math.ceil(perc * len(held) - 1e-9)) - 1])
    if rule == "prevalence":
        return float(np.quantile(np.asarray(p, dtype=np.float64),
                                 1 - s["n_pos"] / (s["n_pos"] + s["n_neg"])))
    if rule == "youden":
        return float(s["thr"][int(np.argmax(s["tp"] / s["n_pos"] - s["fp"] / s["n_neg"]))])
    n = s["n_pos"] + s["n_neg"]
    fn, tn = s["n_pos"] - s["tp"], s["n_neg"] - s["fp"]
    po = (s["tp"] + tn) / n
    pe = ((s["tp"] + s["fp"]) * s["n_pos"] + (fn + tn) * s["n_neg"]) / n ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(pe >= 1, -np.inf, (po - pe) / (1 - pe))
    return float(s["thr"][int(np.argmax(k))])


def cohen_kappa(a, b) -> float:
    """Chance-corrected agreement of two labellings of the same units, in either order."""
    a = np.asarray(a).astype(np.int64)
    b = np.asarray(b).astype(np.int64)
    n = len(a)
    both = int(np.sum((a == 1) & (b == 1)))
    neither = int(np.sum((a == 0) & (b == 0)))
    a_only = int(np.sum((a == 1) & (b == 0)))
    b_only = int(np.sum((a == 0) & (b == 1)))
    po = (both + neither) / n
    pe = ((both + a_only) * (both + b_only) + (b_only + neither) * (a_only + neither)) / n ** 2
    return float("nan") if pe >= 1 else float((po - pe) / (1 - pe))


def kappa_score(y, p, rule: str = "youden") -> float:
    """Cohen's kappa of a model's decisions against the observed response."""
    thr = decision_threshold(y, p, rule)
    if not np.isfinite(thr):
        return float("nan")
    return cohen_kappa(_labels(y), (np.asarray(p, dtype=np.float64) >= thr).astype(int))


def model_agreement(y, p_a, p_b, rule: str = "youden") -> dict:
    """Agreement between two models' decisions, with how often each is right where they differ."""
    y = _labels(y)
    ta, tb = decision_threshold(y, p_a, rule), decision_threshold(y, p_b, rule)
    if not (np.isfinite(ta) and np.isfinite(tb)):
        return dict(kappa=float("nan"), n=len(y), n_disagree=0, share_disagree=float("nan"),
                    a_right=0, b_right=0)
    da = (np.asarray(p_a, dtype=np.float64) >= ta).astype(int)
    db = (np.asarray(p_b, dtype=np.float64) >= tb).astype(int)
    diff = da != db
    return dict(kappa=cohen_kappa(da, db), n=len(y), n_disagree=int(diff.sum()),
                share_disagree=float(diff.mean()),
                a_right=int(np.sum(diff & (da == y))), b_right=int(np.sum(diff & (db == y))))
