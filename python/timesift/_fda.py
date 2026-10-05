"""Flexible discriminant analysis, over the core ``src/ts_fda.cpp`` compiles into both languages.

Nothing here decides anything: the design, the response and the case weights are settled above,
and what is left is to hand them over column-major. The R package's ``R/learners_fda.R`` is the
same layer over the same core, so the two keep the same terms for the same input.

A fit is a plain dictionary of arrays, the kept terms as their factors, their coefficients, the
variate and the recalibration, so it pickles and predicts on another machine.
"""

from __future__ import annotations

import numpy as np

from . import _core
from ._responses import as_design, as_responses

__all__ = ["fda_fit", "fda_fits", "fda_predict"]


def fda_fits(x, y, w, degree=1, penalty=None, nk=None, thresh=0.001, prune=True, calibrate=True,
             threads=1) -> list:
    """The discriminant of each zero-one column of ``y`` on the columns of ``x``, under the
    matching column of the case weights ``w``."""
    y, w = as_responses(y, w)
    return _core.fda_fit(as_design(x), y, w, int(degree),
                         float("nan") if penalty is None else float(penalty),
                         0 if nk is None else int(nk), float(thresh), bool(prune),
                         bool(calibrate), int(threads))


def fda_fit(x, y, w, **settings) -> dict:
    """The discriminant of the one zero-one response ``y``."""
    return fda_fits(x, y, w, **settings)[0]


def fda_predict(fit: dict, newx) -> np.ndarray:
    """The probability of the second class at each row of ``newx``."""
    return _core.fda_predict(fit, as_design(newx))
