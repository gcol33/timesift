"""The tree, the forest and the boosted trees, over the core ``src/ts_tree.cpp`` and
``src/ts_boost.cpp`` compile into both languages.

Nothing here grows anything: the design, the family, the case weights and the inner folds are
settled above, and what is left is to hand them over column-major and to read the complexity table
that comes back. The R package's ``R/tree.R`` is the same layer over the same core, so the two grow
the same tree for the same input.

A fitted tree is a plain dictionary of arrays, so it pickles and predicts on another machine,
which is what every other fitted object in the package is.
"""

from __future__ import annotations

import numpy as np

from . import _core

__all__ = ["tree_fit", "tree_prune", "tree_predict", "tree_prune_cp", "forest_fit",
           "forest_predict", "forest_stream", "boost_fit", "boost_predict"]

PRUNE_RULES = ("se_sum", "one_se", "min", "none")


def _design(x) -> np.ndarray:
    return np.asfortranarray(np.asarray(x, dtype=np.float64))


def tree_fit(x, y, w, family, min_split, min_leaf, cp, max_depth, fold=None,
             n_fold=0, shrink=1.0) -> dict:
    """One tree grown under rpart's rules, with its complexity table, and that table's
    cross-validated error where ``fold`` gives one 0-based fold index per unit. ``shrink`` is the
    coefficient of variation of the gamma prior a Poisson leaf's rate is shrunk by."""
    return _core.tree_fit(_design(x), np.ascontiguousarray(y, dtype=np.float64),
                          np.ascontiguousarray(w, dtype=np.float64), family, int(min_split),
                          int(min_leaf), float(cp), int(max_depth),
                          None if fold is None else np.ascontiguousarray(fold, dtype=np.int32),
                          int(n_fold), float(shrink))


def tree_prune(tree: dict, cp: float) -> dict:
    """The tree with every split of complexity at or below ``cp`` collapsed, as rpart's
    ``prune()``. The complexity table is kept whole."""
    return _core.tree_prune(tree, float(cp))


def tree_predict(tree: dict, newx) -> np.ndarray:
    """The value of the leaf each row of ``newx`` falls into: the probability of a 1, or the
    mean."""
    return _core.tree_predict(tree, _design(newx))


def forest_fit(x, y, w, family, trees, mtry, min_leaf, balance, seed, threads=1) -> dict:
    """A random forest grown by the core: each tree on a bootstrap draw weighted by ``w``, each
    node split on the best of ``mtry`` columns drawn for it. ``seed`` is taken modulo 2^32."""
    return _core.forest_fit(_design(x), np.ascontiguousarray(y, dtype=np.float64),
                            np.ascontiguousarray(w, dtype=np.float64), family, int(trees),
                            int(mtry), int(min_leaf), bool(balance), int(seed) & 0xFFFFFFFF,
                            int(threads))


def forest_predict(forest: dict, newx) -> np.ndarray:
    """The mean over the forest's trees of the leaf each row of ``newx`` falls into."""
    return _core.forest_predict(forest, _design(newx))


def boost_fit(x, y, w, family, trees, depth, shrinkage, min_leaf, subsample, colsample, newton,
              lambda_, gamma, seed, fold=None, n_fold=0, threads=1) -> dict:
    """Gradient boosted trees grown by the core, gbm's or, under ``newton``, xgboost's. ``fold``
    gives one 0-based fold index per unit, and the number of trees kept is then the one of least
    held-out deviance. ``seed`` is taken modulo 2^32."""
    return _core.boost_fit(_design(x), np.ascontiguousarray(y, dtype=np.float64),
                           np.ascontiguousarray(w, dtype=np.float64), family, int(trees),
                           int(depth), float(shrinkage), float(min_leaf), float(subsample),
                           float(colsample), bool(newton), float(lambda_), float(gamma),
                           int(seed) & 0xFFFFFFFF,
                           None if fold is None else np.ascontiguousarray(fold, dtype=np.int32),
                           int(n_fold), int(threads))


def boost_predict(fit: dict, newx) -> np.ndarray:
    """The score of each row of ``newx``, through the logistic function under a binomial
    family."""
    return _core.boost_predict(fit, _design(newx))


def forest_stream(seed, tree, n) -> np.ndarray:
    """The first ``n`` outputs of the generator tree ``tree`` of a forest seeded ``seed`` draws
    from."""
    return _core.forest_stream(int(seed) & 0xFFFFFFFF, int(tree), int(n))


def tree_prune_cp(tree: dict, rule: str):
    """The complexity a tree is pruned back to, read off its cross-validated complexity table.

    ``"se_sum"`` is the row of least cross-validated error plus its standard error among the rows
    that keep a split, the last of them where several tie, which is how biomod2 prunes its
    classification tree. ``"one_se"`` is the smallest tree within one standard error of the least
    cross-validated error, and ``"min"`` the first row reaching the least error. ``None`` is no
    pruning: a table without a cross-validation, or a rule that finds no row, leaves the tree as
    it was grown.
    """
    if rule not in PRUNE_RULES:
        raise ValueError(f"a tree is pruned by one of {', '.join(PRUNE_RULES)}, got {rule!r}.")
    xerror = np.asarray(tree["xerror"])
    if rule == "none" or xerror.size == 0:
        return None
    cp = np.asarray(tree["cp"])
    xstd = np.asarray(tree["xstd"])
    if rule == "se_sum":
        keep = np.flatnonzero(np.asarray(tree["nsplit"]) > 0)
        if keep.size == 0:
            return None
        xsum = xerror[keep] + xstd[keep]
        return float(cp[keep[np.flatnonzero(xsum == xsum.min())[-1]]])
    best = int(np.argmin(xerror))
    if rule == "one_se":
        return float(cp[np.flatnonzero(xerror <= xerror[best] + xstd[best])[0]])
    return float(cp[best])
