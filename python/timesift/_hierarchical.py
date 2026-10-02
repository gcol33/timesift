"""The hierarchical model, over the core ``src/ts_hierarchical.cpp`` compiles into both languages.

Nothing here decides anything: the design, the units' indices, the coordinates and the case weights
are settled above, and what is left is to hand them over column-major. The R package's
``R/hierarchical.R`` is the same layer over the same core, so the two fit the same model to the
same input.

A fit is a plain dictionary of arrays, so it pickles and predicts on another machine.
"""

from __future__ import annotations

import numpy as np

from . import _core

__all__ = ["COVARIANCES", "hierarchical_fit", "hierarchical_predict"]

COVARIANCES = ("exponential", "matern32", "matern52", "gaussian")


def hierarchical_fit(x, y, w, unit=None, n_unit=0, coords=None, field="none", m=6, boundary=1.5,
                     neighbours=15, cov=0, nodes=5, threads=1, beta_sd=2.5, theta=None) -> dict:
    """The model over the columns of ``x`` under the case weights ``w``. ``unit`` holds each
    target's 0-based unit and ``coords`` its pair of coordinates."""
    return _core.hierarchical_fit(
        np.asfortranarray(np.asarray(x, dtype=np.float64)),
        np.ascontiguousarray(y, dtype=np.float64), np.ascontiguousarray(w, dtype=np.float64),
        None if unit is None else np.ascontiguousarray(unit, dtype=np.int32), int(n_unit),
        None if coords is None else np.asfortranarray(np.asarray(coords, dtype=np.float64)),
        field, float(beta_sd), m=int(m), boundary=float(boundary), neighbours=int(neighbours),
        cov=int(cov), nodes=int(nodes), threads=int(threads),
        theta=None if theta is None else [float(t) for t in theta])


def hierarchical_predict(fit: dict, newx, unit=None, coords=None) -> np.ndarray:
    """The linear predictor at each row of ``newx``; ``unit`` is -1 for a unit the fit did not
    see, whose intercept is zero."""
    return _core.hierarchical_predict(
        fit, np.asfortranarray(np.asarray(newx, dtype=np.float64)),
        None if unit is None else np.ascontiguousarray(unit, dtype=np.int32),
        None if coords is None else np.asfortranarray(np.asarray(coords, dtype=np.float64)))
