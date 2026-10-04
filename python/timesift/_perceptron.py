"""The one-hidden-layer network, over the core ``src/ts_perceptron.cpp`` compiles into both
languages.

Nothing here decides anything: the design, the response and the case weights are settled above,
and what is left is to hand them over column-major. The R package's ``R/learners_perceptron.R`` is
the same layer over the same core, so the two fit the same network for the same input.

A fit is a plain dictionary of its size, its family and its weights, so it pickles and predicts on
another machine.
"""

from __future__ import annotations

import numpy as np

from . import _core

__all__ = ["perceptron_fit", "perceptron_predict"]


def perceptron_fit(x, y, w, family, hidden=2, decay=0.0, range_=0.7, max_iter=100, skip=False,
                   abs_tol=1e-4, rel_tol=1e-8, seed=1, start=None) -> dict:
    """The network over the columns of ``x`` under the case weights ``w``; ``start``, where given,
    replaces the drawn starting weights."""
    return _core.perceptron_fit(np.asfortranarray(np.asarray(x, dtype=np.float64)),
                                np.ascontiguousarray(y, dtype=np.float64),
                                np.ascontiguousarray(w, dtype=np.float64), family, int(hidden),
                                bool(skip), float(decay), float(range_), int(max_iter),
                                float(abs_tol), float(rel_tol), int(seed),
                                np.ascontiguousarray([] if start is None else start,
                                                     dtype=np.float64))


def perceptron_predict(fit: dict, newx) -> np.ndarray:
    """The fitted mean at each row of ``newx``."""
    return _core.perceptron_predict(fit, np.asfortranarray(np.asarray(newx, dtype=np.float64)))
