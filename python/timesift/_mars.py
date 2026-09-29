"""MARS, over the core ``src/ts_mars.cpp`` compiles into both languages.

Nothing here decides anything: the design, the response and the case weights are settled above,
and what is left is to hand them over column-major. The R package's ``R/learners_mars.R`` is the
same layer over the same core, so the two keep the same terms for the same input.

A fit is a plain dictionary of arrays, every term of the forward pass as its factors and the kept
ones' coefficients, so it pickles and predicts on another machine.
"""

from __future__ import annotations

import numpy as np

from . import _core

__all__ = ["mars_fit", "mars_predict"]


def mars_fit(x, y, w, family, degree=1, penalty=None, nk=None, thresh=0.001, minspan=0,
             endspan=0, fast_k=20, fast_beta=1.0, prune=True, nprune=None, threads=1) -> dict:
    """The forward and pruning passes over the columns of ``x`` under the case weights ``w``."""
    return _core.mars_fit(np.asfortranarray(np.asarray(x, dtype=np.float64)),
                          np.ascontiguousarray(y, dtype=np.float64),
                          np.ascontiguousarray(w, dtype=np.float64), family, int(degree),
                          float("nan") if penalty is None else float(penalty),
                          0 if nk is None else int(nk), float(thresh), int(minspan),
                          int(endspan), int(fast_k), float(fast_beta), bool(prune),
                          0 if nprune is None else int(nprune), int(threads))


def mars_predict(fit: dict, newx) -> np.ndarray:
    """The fitted mean at each row of ``newx``."""
    return _core.mars_predict(fit, np.asfortranarray(np.asarray(newx, dtype=np.float64)))
