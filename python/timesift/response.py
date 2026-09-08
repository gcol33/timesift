"""The response, the fold map, and the cells a score is defined on.

The fold map is an artifact rather than an algorithm: it is built once and read by everything that
scores, in either language, so that every arm sees identical splits and any two of them can be
compared cell by cell. ``fold_map`` builds one; ``timesift.read_folds`` reads one somebody else
built, from the file format ``inst/spec/representation.md`` defines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .representation import _unit_names


@dataclass(frozen=True)
class Response:
    """A ``[unit, variable]`` matrix of observed values, with the units named."""

    values: np.ndarray
    units: tuple[str, ...]
    variables: tuple[str, ...]

    @classmethod
    def from_columns(cls, data, id: str, variables=None) -> "Response":
        """A response from a table of one unit column and one column per variable."""
        units = list(_unit_names(data[id], id))
        variables = list(variables) if variables is not None else \
            [k for k in data.keys() if k != id]
        values = np.column_stack([np.asarray(data[v], dtype=np.float64) for v in variables])
        return cls(values=values, units=tuple(units), variables=tuple(variables))

    def take_units(self, index) -> "Response":
        """The response restricted to a subset of its units, in the order given."""
        index = np.asarray(index)
        return Response(self.values[index], tuple(np.asarray(self.units)[index]), self.variables)

    def take_variables(self, index) -> "Response":
        """The response restricted to a subset of its variables, in the order given.

        A learner covering one response at a time is handed them through this, so the matrix a
        candidate emits is assembled from the same names each part was fitted on.
        """
        index = np.asarray(index)
        return Response(self.values[:, index], self.units,
                        tuple(np.asarray(self.variables)[index]))

    def align(self, units) -> "Response":
        """Put the response into the row order of a representation, by unit and never by position."""
        if tuple(units) == self.units:
            return self
        position = {u: i for i, u in enumerate(self.units)}
        missing = [u for u in units if u not in position]
        if missing:
            raise ValueError(f"{len(missing)} units are in the representation but not the "
                             f"response, first: {missing[0]}")
        return self.take_units([position[u] for u in units])

    def check_presence_absence(self) -> "Response":
        """Error unless every value is 0 or 1 and none is missing."""
        if np.isnan(self.values).any():
            raise ValueError("the response holds missing values")
        if not np.isin(self.values, (0, 1)).all():
            raise ValueError("a presence-absence response must be 0/1 or logical")
        return self


@dataclass(frozen=True)
class Cells:
    """Which ``(variable, fold)`` cells admit a score, and the counts that decided it."""

    variable: np.ndarray
    fold: np.ndarray
    n_occ: np.ndarray
    pres_train: np.ndarray
    abs_train: np.ndarray
    pres_test: np.ndarray
    abs_test: np.ndarray
    scorable: np.ndarray

    def is_scorable(self, variable: str, fold: int) -> bool:
        """Whether one `(variable, fold)` cell admits a score."""
        hit = (self.variable == variable) & (self.fold == fold)
        return bool(self.scorable[hit][0]) if hit.any() else False

    def __repr__(self) -> str:  # pragma: no cover - display only
        by_variable = {v: False for v in self.variable}
        for v, ok in zip(self.variable, self.scorable):
            by_variable[v] = by_variable[v] or bool(ok)
        return (f"<timesift cells> {len(self.variable)} cells over {len(by_variable)} variables\n"
                f"scorable: {int(self.scorable.sum())} "
                f"({100 * self.scorable.mean():.1f}%); variables with at least one scorable fold: "
                f"{sum(by_variable.values())} of {len(by_variable)}")


@dataclass(frozen=True)
class Folds:
    """Which fold each unit is held out in, named by unit.

    Named rather than positional, because a fold map is aligned to a representation by unit and
    never by row: two tables of the same height are not two tables in the same order.
    """

    fold: np.ndarray
    units: tuple[str, ...]
    grouped: bool = False
    group: tuple[str, ...] | None = None

    def __post_init__(self):
        if len(self.fold) != len(self.units):
            raise ValueError(f"{len(self.fold)} folds for {len(self.units)} units")
        if self.group is not None and len(self.group) != len(self.units):
            raise ValueError(f"{len(self.group)} group values for {len(self.units)} units")

    @property
    def v(self) -> int:
        """How many folds the map holds."""
        return int(len(np.unique(self.fold)))

    @classmethod
    def coerce(cls, x, units=None) -> "Folds":
        """A ``Folds``, a mapping of unit to fold, or a bare vector read in ``units`` order."""
        if isinstance(x, cls):
            return x
        if isinstance(x, dict):
            return cls(fold=np.asarray(list(x.values()), dtype=np.int64),
                       units=tuple(str(u) for u in x))
        fold = np.asarray(x, dtype=np.int64)
        if units is None:
            raise ValueError("a bare fold map has no unit names, so it can only be read "
                             "alongside the units it is in the order of")
        if len(fold) != len(units):
            raise ValueError(f"an unnamed fold map must have one entry per unit, got "
                             f"{len(fold)} for {len(units)}")
        return cls(fold=fold, units=tuple(str(u) for u in units))

    def align(self, units) -> "Folds":
        """Put the map into the row order of a representation, by unit and never by position."""
        units = tuple(str(u) for u in units)
        if units == self.units:
            return self
        position = {u: i for i, u in enumerate(self.units)}
        missing = [u for u in units if u not in position]
        if missing:
            raise ValueError(f"{len(missing)} unit{'s have' if len(missing) > 1 else ' has'} "
                             f"no row in the fold map, first: {missing[0]}")
        order = [position[u] for u in units]
        return Folds(fold=self.fold[order], units=units, grouped=self.grouped,
                     group=None if self.group is None else tuple(self.group[i] for i in order))

    def as_dict(self) -> dict:
        """The map as unit to fold."""
        return {u: int(k) for u, k in zip(self.units, self.fold)}

    def __len__(self) -> int:
        return len(self.fold)

    def __iter__(self):
        return iter(self.fold.tolist())

    def __array__(self, dtype=None, copy=None):
        return self.fold if dtype is None else self.fold.astype(dtype)

    def __repr__(self) -> str:  # pragma: no cover - display only
        levels, counts = np.unique(self.fold, return_counts=True)
        return (f"<timesift folds> {len(self.fold)} units in {self.v} folds\n"
                + "  ".join(f"{k}: {n}" for k, n in zip(levels.tolist(), counts.tolist())))


def fold_map(y: Response, v: int = 10, seed: int = 1, strata: int = 5, by=None,
             group=None) -> Folds:
    """Assign units to folds, balanced within equal-count strata of a stratifying value.

    ``group`` keeps every unit sharing a value in one fold, which is what repeated targets on the
    same physical unit need: two visits to a plot are not two independent held-out units. The deal
    is then made over the groups rather than over the units, and a group carries the mean of the
    stratifying value of the units in it.

    The stream is numpy's, so a map built here is not the map the R side builds from the same seed.
    Where both languages must see identical splits, build the map once, write it with
    ``write_folds`` and read it in the other with ``read_folds``.
    """
    n = y.values.shape[0]
    value = y.values.sum(axis=1) if by is None else np.asarray(by, dtype=np.float64)
    if len(value) != n:
        raise ValueError(f"`by` must have one value per unit, got {len(value)} for {n}")
    if group is None:
        return Folds(fold=_deal(value, v, seed, strata, "units"), units=y.units)
    key = np.asarray([str(g) for g in group])
    if len(key) != n:
        raise ValueError(f"`group` must have one value per unit, got {len(key)} for {n}")
    names, index = np.unique(key, return_inverse=True)
    per_group = np.asarray([value[index == k].mean() for k in range(len(names))])
    # The grouping travels with the map, so every split drawn inside a fold, the encoders'
    # validation set and the penalised fit's inner folds, keeps whole what the outer folds kept
    # whole.
    return Folds(fold=_deal(per_group, v, seed, strata, "groups")[index], units=y.units,
                 grouped=True, group=tuple(key.tolist()))


def _deal(value: np.ndarray, v: int, seed: int, strata: int, what: str) -> np.ndarray:
    """Shuffle inside each stratum and deal the folds round-robin, so every fold carries the same
    mix of the stratifying value."""
    n = len(value)
    if not 2 <= v <= n:
        raise ValueError(f"`v` must be between 2 and the {n} {what}, got {v}")
    stratum = np.ones(n, dtype=int) if strata <= 1 else _quantile_strata(value, strata)

    # The fold labels are permuted once, and the deal runs on from one stratum into the next: a
    # counter restarted in every stratum hands the first labels one unit more than the last in
    # every stratum, and a stratum smaller than ``v`` never reaches the last labels at all.
    rng = np.random.default_rng(seed)
    labels = rng.permutation(v) + 1
    fold = np.empty(n, dtype=int)
    dealt = 0
    for s in np.unique(stratum):
        idx = np.flatnonzero(stratum == s)
        rng.shuffle(idx)
        fold[idx] = labels[(dealt + np.arange(len(idx))) % v]
        dealt += len(idx)
    return fold


def align_folds(folds, units) -> np.ndarray:
    """A fold map reaches the fitting path as an integer vector in the row order of the
    representation, whether it arrived as a ``Folds``, a mapping of unit to fold, or a bare vector
    already in that order."""
    return Folds.coerce(folds, units).align(units).fold


def scorable_cells(y: Response, folds) -> Cells:
    """Which cells admit a score, from the response and the fold map alone.

    A cell needs both classes among the held-out units and both classes among the units a model is
    fitted on. Computing the mask without a model is what lets every arm be restricted to the same
    cells, so their means share a denominator and every paired difference runs on matched cells.
    """
    folds = align_folds(folds, y.units)
    levels = np.unique(folds)
    n_occ = y.values.sum(axis=0).astype(np.int64)

    variable, fold, occ, p_test, a_test = [], [], [], [], []
    for k in levels:
        rows = folds == k
        pres = y.values[rows].sum(axis=0)
        for j, name in enumerate(y.variables):
            variable.append(name)
            fold.append(int(k))
            occ.append(n_occ[j])
            p_test.append(pres[j])
            a_test.append(int(rows.sum()) - pres[j])

    occ = np.asarray(occ, dtype=np.int64)
    p_test = np.asarray(p_test, dtype=np.int64)
    a_test = np.asarray(a_test, dtype=np.int64)
    p_train = occ - p_test
    a_train = (y.values.shape[0] - occ) - a_test
    order = np.lexsort((np.asarray(fold), np.asarray(variable)))
    return Cells(variable=np.asarray(variable)[order], fold=np.asarray(fold)[order],
                 n_occ=occ[order], pres_train=p_train[order], abs_train=a_train[order],
                 pres_test=p_test[order], abs_test=a_test[order],
                 scorable=((p_train >= 1) & (a_train >= 1) &
                           (p_test >= 1) & (a_test >= 1))[order])


PRESENCE_ABSENCE = dict(
    prepare=lambda y: as_response(y).check_presence_absence(),
    activation="sigmoid",
    loss="binary_cross_entropy",
    metric="tss",
    cells=lambda y, folds: scorable_cells(y, folds),
)


def as_response(y) -> Response:
    """The response reaches everything downstream as a ``Response``, whether it arrived as one, as
    a mapping of variable name to values, or as a two-dimensional array with no names at all."""
    if isinstance(y, Response):
        return y
    if isinstance(y, dict):
        variables = tuple(k for k in y if k != "id")
        values = np.column_stack([np.asarray(y[v], dtype=np.float64) for v in variables])
        units = tuple(str(u) for u in y["id"]) if "id" in y else \
            tuple(str(i + 1) for i in range(values.shape[0]))
        return Response(values=values, units=units, variables=variables)
    values = np.asarray(y, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError(f"the response must be two-dimensional, got {values.ndim} dimensions")
    return Response(values=values,
                    units=tuple(str(i + 1) for i in range(values.shape[0])),
                    variables=("y",) if values.shape[1] == 1 else
                    tuple(f"v{j + 1}" for j in range(values.shape[1])))


def _quantile_strata(value: np.ndarray, k: int) -> np.ndarray:
    edges = np.unique(np.quantile(value, np.linspace(0, 1, k + 1)))
    if len(edges) < 3:
        return np.ones(len(value), dtype=int)
    return np.clip(np.searchsorted(edges[1:-1], value, side="left"), 0, len(edges) - 2) + 1
