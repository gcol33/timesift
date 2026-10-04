"""maxnet, over the core ``src/ts_maxnet.cpp`` compiles into both languages.

Nothing here decides anything: the design, the response, the case weights and the inner folds are
settled above, and what is left is to hand them over column-major. The R package's
``R/learners_maxnet.R`` is the same layer over the same core, so the two fit the same model for the
same input.

A fit is a plain dictionary of arrays, so it pickles and predicts on another machine, which is
what every other fitted object in the package is.
"""

from __future__ import annotations

import numpy as np

from . import _core

__all__ = ["maxnet_design", "maxnet_fit", "maxnet_predict"]


def _design(x) -> np.ndarray:
    return np.asfortranarray(np.asarray(x, dtype=np.float64))


def _vector(v) -> np.ndarray:
    return np.ascontiguousarray(v, dtype=np.float64)


def maxnet_design(x, y, classes=None, knots=50, regmult=1.0, formulation="background",
                  add_samples=True, max_design=2.0) -> dict:
    """The rows a fit reads, background rows added, their response, the features and each
    feature's penalty factor. ``rows`` are 0-based."""
    return _core.maxnet_design(_design(x), _vector(y), classes or "", int(knots), float(regmult),
                               formulation, bool(add_samples), float(max_design))


def maxnet_fits(x, y, w, classes=None, knots=50, regmult=1.0, formulation="background",
                add_samples=True, thresh=1e-8, max_pass=1e8, n_lambda=100, one_se=False,
                fold=None, n_fold=(), threads=1, max_design=2.0) -> list:
    """maxnet over the columns of ``x``, one fit per column of ``y``: the background formulation at
    the last of maxnet's penalties, or the absence formulation at the penalty its folds choose,
    where column ``s`` of ``fold`` gives response ``s`` one 0-based fold index per unit over
    ``n_fold[s]`` folds."""
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if y.ndim == 1:
        y, w = y[:, None], w[:, None]
    if fold is not None:
        fold = np.asarray(fold, dtype=np.int32)
        fold = np.asfortranarray(fold[:, None] if fold.ndim == 1 else fold)
    return _core.maxnet_fit(_design(x), np.asfortranarray(y), np.asfortranarray(w),
                            classes or "", int(knots), float(regmult), formulation,
                            bool(add_samples), float(thresh), float(max_pass), int(n_lambda),
                            bool(one_se), fold, [int(k) for k in np.atleast_1d(n_fold)],
                            int(threads), float(max_design))


def maxnet_fit(x, y, w, fold=None, n_fold=0, **settings) -> dict:
    """maxnet on the one response ``y``, ``fold`` one 0-based fold index per unit."""
    return maxnet_fits(x, y, w, fold=fold, n_fold=() if fold is None else [n_fold],
                       **settings)[0]


def maxnet_predict(fit: dict, newx, clamp=True, type="cloglog") -> np.ndarray:
    """maxnet's output at each row of ``newx``: ``"link"``, ``"exponential"``, ``"cloglog"`` or
    ``"logistic"`` under the background formulation, the link or the probability,
    ``"logistic"``, under the absence one."""
    return _core.maxnet_predict(fit, _design(newx), bool(clamp), type)
