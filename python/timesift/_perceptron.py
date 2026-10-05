"""The one-hidden-layer network, over the core ``src/ts_perceptron.cpp`` compiles into both
languages.

Nothing here decides anything: the design, the response and the case weights are settled above,
and what is left is to hand them over column-major. The R package's ``R/learners_perceptron.R`` is
the same layer over the same core, so the two fit the same network for the same input.

A fit is a plain dictionary of its size, its family, the centre and scale of its columns and its
weights, so it pickles and predicts on another machine.
"""

from __future__ import annotations

import numpy as np

from . import _core
from ._responses import as_design, as_responses

__all__ = ["perceptron_fit", "perceptron_fits", "perceptron_predict"]


def perceptron_fits(x, y, w, family, seeds, hidden=2, decay=0.0, range_=0.7, max_iter=100,
                    skip=False, standardise=False, abs_tol=1e-4, rel_tol=1e-8, threads=1,
                    start=None) -> list:
    """One network per column of ``y`` over the columns of ``x``, under the matching column of the
    case weights ``w`` and seed of ``seeds``; ``start``, where given, replaces every network's drawn
    starting weights."""
    y, w = as_responses(y, w)
    return _core.perceptron_fit(as_design(x), y, w, [int(s) for s in seeds], family, int(hidden),
                                bool(skip), bool(standardise), float(decay), float(range_),
                                int(max_iter), float(abs_tol), float(rel_tol), int(threads),
                                np.ascontiguousarray([] if start is None else start,
                                                     dtype=np.float64))


def perceptron_fit(x, y, w, family, seed=1, **settings) -> dict:
    """The network of the one response ``y``."""
    return perceptron_fits(x, y, w, family, [seed], **settings)[0]


def perceptron_predict(fit: dict, newx) -> np.ndarray:
    """The fitted mean at each row of ``newx``."""
    return _core.perceptron_predict(fit, as_design(newx))
