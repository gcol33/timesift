"""The envelope, over the core ``src/ts_envelope.cpp`` compiles into both languages.

A fit is each column's two bounds over the presences, a plain dictionary of arrays. The R
package's ``R/learners_envelope.R`` is the same layer over the same core.
"""

from __future__ import annotations

import numpy as np

from . import _core

__all__ = ["envelope_fit", "envelope_predict"]


def envelope_fit(x, y, quantile=0.025) -> dict:
    """Each column's ``quantile`` and ``1 - quantile`` quantiles over the rows where ``y`` is one."""
    return _core.envelope_fit(np.asfortranarray(np.asarray(x, dtype=np.float64)),
                              np.ascontiguousarray(y, dtype=np.float64), float(quantile))


def envelope_predict(fit: dict, newx) -> np.ndarray:
    """One where every column of a row lies inside its band, zero elsewhere."""
    return _core.envelope_predict(fit, np.asfortranarray(np.asarray(newx, dtype=np.float64)))
