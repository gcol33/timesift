"""Threshold metrics on held-out predictions.

A cut may only fall between distinct predictions: units sharing a prediction are decided together,
so the same score comes back whatever order they arrived in and both language sides read every
threshold metric off the same rule.
"""

from __future__ import annotations

import numpy as np

THRESHOLD_RULES = ("youden", "kappa", "prevalence")


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


def roc_auc(y, p) -> float:
    """The area under the ROC curve, as the rank sum of the presences. Ties take the average rank."""
    y = _labels(y)
    p = np.asarray(p, dtype=np.float64)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0 or not np.isfinite(p).all():
        return float("nan")
    order = np.argsort(p, kind="stable")
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    # average rank within each run of tied predictions
    ps = p[order]
    start = 0
    for i in range(1, len(ps) + 1):
        if i == len(ps) or ps[i] != ps[start]:
            ranks[order[start:i]] = (start + i + 1) / 2
            start = i
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def decision_threshold(y, p, rule: str = "youden") -> float:
    """The probability cut a rule selects. Presence is predicted at ``p >= threshold``."""
    if rule not in THRESHOLD_RULES:
        raise ValueError(f"rule must be one of {THRESHOLD_RULES}, got {rule!r}")
    s = _sweep(y, p)
    if s is None:
        return float("nan")
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
