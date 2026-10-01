"""The additive model, over the core ``src/ts_additive.cpp`` compiles into both languages.

Nothing here decides anything: the design, the responses and the case weights are settled above,
and what is left is to hand them over column-major. The R package's ``R/learners_additive.R`` is
the same layer over the same core, so the two fit the same model for the same input.

A fit is a plain dictionary of arrays, every column's basis and every response's coefficients, so
it pickles and predicts on another machine.
"""

from __future__ import annotations

import numpy as np

from . import _core

__all__ = ["additive_fit", "additive_predict"]


def additive_fit(x, y, w, family, k=10, gamma=1.0, max_knots=2000, threads=1,
                 sp=None) -> dict:
    """One additive model per column of ``y``, over the columns of ``x``, under the case weights
    ``w`` of the same shape as ``y``. ``sp``, one smoothing parameter per penalised column in
    order, fits at those parameters in place of the search."""
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
        w = w[:, None]
    return _core.additive_fit(np.asfortranarray(np.asarray(x, dtype=np.float64)),
                              np.asfortranarray(y), np.asfortranarray(w), family, int(k),
                              float(gamma), int(max_knots), int(threads),
                              None if sp is None else [float(v) for v in sp])


def additive_predict(fit: dict, newx) -> np.ndarray:
    """The fitted mean of every response at each row of ``newx``, ``[rows, responses]``."""
    newx = np.asfortranarray(np.asarray(newx, dtype=np.float64))
    out = np.asarray(_core.additive_predict(fit, newx))
    return out.reshape((newx.shape[0], int(fit["n_response"])), order="F")
