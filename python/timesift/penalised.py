"""The penalised fit, over the core ``src/ts_penalised.cpp`` compiles into both languages.

Nothing here decides anything: the design, the family, the case weights and the inner folds are
settled above, and what is left is to hand them over column-major and to shape what comes back.
The R package's ``R/penalised.R`` is the same layer over the same core, so the two return the same
coefficients for the same input.

A fit is a plain dictionary of arrays, so it pickles and predicts on another machine, which is
what every other fitted object in the package is.
"""

from __future__ import annotations

import numpy as np

from . import _core

__all__ = ["penalised_path", "penalised_cv", "penalised_predict", "penalised_coef"]


def _design(x: np.ndarray) -> np.ndarray:
    return np.asfortranarray(np.asarray(x, dtype=np.float64))


def penalised_path(x, y, w=None, family="binomial", alpha=1.0, n_lambda=100, lambda_=None,
                   thresh=1e-8, standardize=True, intercept=True, max_pass=1e6) -> dict:
    """One elastic-net path over ``x``, at ``n_lambda`` penalties down from the smallest that
    leaves every coefficient at zero, or at the penalties ``lambda_`` names."""
    m = _design(x)
    y = np.asarray(y, dtype=np.float64)
    w = np.ones(m.shape[0]) if w is None else np.asarray(w, dtype=np.float64)
    out = _core.penalised_path(m, y, w, family, alpha, int(n_lambda), 0.0,
                               None if lambda_ is None else [float(v) for v in lambda_],
                               float(thresh), bool(standardize), bool(intercept),
                               float(max_pass))
    return _shape(out)


def penalised_cv(x, y, w, family, alpha, fold, n_fold, n_lambda=100, thresh=1e-8,
                 standardize=True, intercept=True, max_pass=1e6) -> dict:
    """The path fitted on every unit, and the same penalties scored on units held out fold by
    fold. ``fold`` is one 0-based fold index per unit, which is what keeps a grouping whole: the
    caller deals the folds, not this."""
    m = _design(x)
    out = _core.penalised_cv(m, np.asarray(y, dtype=np.float64),
                             np.asarray(w, dtype=np.float64),
                             np.ascontiguousarray(fold, dtype=np.int32), int(n_fold), family,
                             alpha, int(n_lambda), 0.0, float(thresh), bool(standardize),
                             bool(intercept), float(max_pass))
    fit = _shape(out)
    fit["cv_mean"] = out["cv_mean"]
    fit["cv_sd"] = out["cv_sd"]
    fit["lambda_min"] = float(out["lambda"][out["index_min"]])
    fit["lambda_1se"] = float(out["lambda"][out["index_1se"]])
    return fit


def _shape(out: dict) -> dict:
    return dict(lambda_=out["lambda"], a0=out["a0"],
                beta=out["beta"].reshape(len(out["lambda"]), out["n_column"]).T,
                df=out["df"], dev_ratio=out["dev_ratio"],
                null_deviance=out["null_deviance"], passes=out["passes"],
                family=out["family"])


def _penalty_at(model: dict, s) -> float:
    """The penalty a fit is read at: a point of the path by name, or a number, which is
    interpolated between the two points around it the way the path itself is read."""
    if not isinstance(s, str):
        return float(s)
    named = {"lambda.min": "lambda_min", "lambda.1se": "lambda_1se"}
    if s not in named or named[s] not in model:
        raise ValueError('a penalised fit is read at "lambda.min", at "lambda.1se", or at a '
                         "penalty of its own, and a path fitted without a cross-validation "
                         "carries neither name.")
    return float(model[named[s]])


def penalised_predict(model: dict, newx, s="lambda.min") -> np.ndarray:
    return _core.penalised_predict(model["lambda_"], model["a0"],
                                   np.ascontiguousarray(model["beta"].T).reshape(-1),
                                   model["family"], _penalty_at(model, s), _design(newx))


def penalised_coef(model: dict, s="lambda.min") -> np.ndarray:
    return _core.penalised_coef(model["lambda_"], model["a0"],
                                np.ascontiguousarray(model["beta"].T).reshape(-1),
                                model["family"], _penalty_at(model, s))
