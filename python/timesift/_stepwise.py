"""The stepwise model, over the core ``src/ts_stepwise.cpp`` compiles into both languages.

Nothing here decides anything: the design, the response and the case weights are settled above,
and what is left is to hand them over column-major. The R package's ``R/learners_linear.R`` is the
same layer over the same core, so the two select the same terms for the same input.

A fit is a plain dictionary of arrays, its terms as the column each reads and the recurrence of its
polynomial, so it pickles and predicts on another machine.
"""

from __future__ import annotations

import numpy as np

from . import _core

__all__ = ["stepwise_fit", "stepwise_predict"]


def stepwise_fit(x, y, w, family, max_terms=3, degree=2, direction="forward", terms="column",
                 threads=1) -> dict:
    """The search over the columns of ``x`` under the prior weights ``w``."""
    return _core.stepwise_fit(np.asfortranarray(np.asarray(x, dtype=np.float64)),
                              np.ascontiguousarray(y, dtype=np.float64),
                              np.ascontiguousarray(w, dtype=np.float64), family,
                              float(max_terms), int(degree), direction, terms, int(threads))


def stepwise_predict(fit: dict, newx) -> np.ndarray:
    """The fitted mean at each row of ``newx``."""
    return _core.stepwise_predict(fit, np.asfortranarray(np.asarray(newx, dtype=np.float64)))
