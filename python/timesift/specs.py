"""What to build and how to split: the representation and resampling specifications.

A specification is inert. It names a reduction and holds none of the data, so the same one builds
the representation a candidate was fitted on and the representation it is later asked to predict
from, which is what lets ``predict()`` mean anything on targets the fit never saw.

Everything the fitting layer builds comes back as a :class:`~timesift.representation.TimesiftMatrix`
of ``[target, bin, channel]`` in the targets' own row order, whether the bins came from the
calendar, from a lookback, or from columns carried alongside it. One array shape above the core
means the learners, the occlusion and the fold arithmetic have one thing to read.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from .learners import flatten
from .representation import (DAY_LEVEL_STATS, GRAINS, TimesiftMatrix, _check_stats,
                             _parse_year_start, _unit_names, bind_channels, grain_matrix,
                             lookback_matrix)
from .response import Folds, fold_map
from .select import column_names, select_columns

__all__ = [
    "Representation", "Resampling", "Sift", "TimesiftSpec", "as_resampling", "as_sift",
    "auto_grains", "build_representation", "cv", "expand_sift", "grain", "grains", "grouped_cv",
    "lookback", "lookbacks", "multigrain", "n_targets", "native", "resolve_folds",
    "target_labels",
]


@dataclass(frozen=True)
class Representation:
    """One reduction, named but not yet built.

    ``sequence`` says whether the bins are ordered in time and mean something to a convolution:
    the record unreduced, a calendar grain and a lookback cut into several bins are sequences; a
    block of features bound side by side is not.
    """

    label: str
    kind: str
    stats: tuple[str, ...] = ("mean",)
    grain: str | None = None
    grains: tuple[str, ...] | None = None
    span: object = None
    lag: object = "0 days"
    bins: int = 1
    sequence: bool = True
    year_start: str = "09-01"


def native(stats="mean", year_start="09-01") -> Representation:
    """The record unreduced: one bin per reading."""
    return grain("native", stats, year_start)


def grain(g: str, stats="mean", year_start="09-01") -> Representation:
    """One calendar grain."""
    name = _check_grain(g)
    return Representation(label=name, kind="grain", grain=g, stats=_stats(stats, name),
                          year_start=_year_start(year_start))


def multigrain(grains=None, stats="mean", year_start="09-01") -> Representation:
    """Several grains flattened and bound side by side into one block of features.

    Left at ``None`` the grains are the ones the record supports, the set :func:`auto_grains`
    names. A caller who does not want the record unreduced among them names the grains instead.
    """
    named = None if grains is None else tuple(_check_grain(g) for g in _flatten(grains))
    label = "multigrain" if named is None else f"multigrain({'+'.join(named)})"
    for one in named if named is not None else ("day",):
        _stats(stats, one)
    return Representation(label=label, kind="multigrain", grains=named, stats=_stats(stats, "day"),
                          sequence=False, year_start=_year_start(year_start))


def lookback(span, lag="0 days", bins=1, stats="mean") -> Representation:
    """A stretch of record of fixed length, ending a fixed lag before each target's own instant."""
    bins = _check_bins(bins)
    return Representation(label=_lookback_label(span, lag, bins), kind="lookback", span=span,
                          lag=lag, bins=bins, stats=_stats(stats, "lookback"), sequence=bins > 1)


class Sift(Mapping):
    """The representations a set of candidates runs across, as a mapping of label to spec."""

    def __init__(self, parts):
        if isinstance(parts, Representation):
            parts = {parts.label: parts}
        parts = dict(parts)
        if not parts:
            raise ValueError("a sift is a non-empty mapping of representations")
        bad = [str(k) for k, v in parts.items() if not isinstance(v, Representation)]
        if bad:
            raise ValueError(f"{', '.join(bad)} {'are' if len(bad) > 1 else 'is'} not a "
                             f"representation")
        self._parts = parts

    def __getitem__(self, key):
        return self._parts[key]

    def __iter__(self):
        return iter(self._parts)

    def __len__(self) -> int:
        return len(self._parts)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (f"<timesift sift> {len(self._parts)} representations\n  "
                + "\n  ".join(f"{k:<14} {v.kind}" for k, v in self._parts.items()))


def grains(*g, stats="mean", year_start="09-01") -> Sift:
    """A sift over calendar grains, named or read off the record with ``"auto"``."""
    named = [str(one) for one in _flatten(g)]
    if not named:
        raise ValueError("grains() names no grain")
    if "auto" in named:
        if len(named) > 1:
            raise ValueError('"auto" is the whole set the record supports and cannot be named '
                             "beside a grain.")
        return Sift({"auto": Representation(label="auto", kind="auto",
                                            stats=_stats(stats, "day"),
                                            year_start=_year_start(year_start))})
    parts = {}
    for one in named:
        rep = grain(one, stats, year_start)
        parts[rep.label] = rep
    return Sift(parts)


def lookbacks(*spans, lag="0 days", bins=1, stats="mean") -> Sift:
    """A sift over lookbacks of several lengths, all sharing a lag and a number of bins."""
    named = _flatten(spans)
    if not named:
        raise ValueError("lookbacks() names no span")
    parts = {}
    for span in named:
        rep = lookback(span, lag=lag, bins=bins, stats=stats)
        parts[rep.label] = rep
    return Sift(parts)


def as_sift(x) -> Sift:
    """A sift, whether it arrived as one, as a representation, as a grain name, or as a list."""
    if isinstance(x, Sift):
        return x
    if isinstance(x, Representation):
        return Sift(x)
    if isinstance(x, str):
        return grains(x)
    if isinstance(x, Mapping):
        return Sift(x)
    if isinstance(x, (list, tuple)):
        if all(isinstance(one, Representation) for one in x):
            return Sift({one.label: one for one in x})
        if all(isinstance(one, str) for one in x):
            return grains(*x)
        raise ValueError("a sift given as a list holds either grain names or representations, "
                         "not both")
    raise ValueError(f"`sift` is a representation, a set of them, or a grain name, got "
                     f"{type(x).__name__}")


# ---- resampling --------------------------------------------------------------------------------

@dataclass(frozen=True)
class Resampling:
    """How the targets are split, named but not yet drawn."""

    kind: str
    v: int = 10
    seed: int = 1
    strata: int = 5
    group: str | None = None
    folds: object = None


def cv(v: int = 10, seed: int = 1, strata: int = 5) -> Resampling:
    """Hold out single targets, balanced within equal-count strata of the response."""
    return Resampling(kind="cv", v=_whole(v, "v"), seed=seed, strata=_whole(strata, "strata"))


def grouped_cv(group, v: int = 10, seed: int = 1) -> Resampling:
    """Keep every target sharing a value of the ``group`` column in one fold.

    It is what repeated targets on the same unit need: two visits to a plot a fortnight apart are
    not two independent held-out units, and splitting them across folds scores a model on a unit
    it has already read.
    """
    return Resampling(kind="grouped_cv", v=_whole(v, "v"), seed=seed, group=group)


def as_resampling(x) -> Resampling:
    """A resampling spec, or a fold map somebody else built, read as a spec that returns it."""
    if isinstance(x, Resampling):
        return x
    if x is None:
        return cv()
    return Resampling(kind="given", folds=x)


def resolve_folds(resampling, y, targets, spec) -> Folds:
    """Draw the fold map a resampling spec names, in the row order of the response."""
    r = as_resampling(resampling)
    if r.kind == "given":
        return Folds.coerce(r.folds, y.units).align(y.units)
    if r.kind == "cv":
        return fold_map(y, v=r.v, seed=r.seed, strata=r.strata)
    named = select_columns(column_names(targets), r.group, "`group` of grouped_cv()")
    if len(named) != 1:
        raise ValueError(f"`group` of grouped_cv() names one column of `targets`, got "
                         f"{len(named)}")
    return fold_map(y, v=r.v, seed=r.seed, strata=r.strata,
                    group=[str(g) for g in targets[named[0]]])


# ---- building ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class TimesiftSpec:
    """How a fit was asked for: the columns each table plays, and the calendar they are read in."""

    y: tuple[str, ...]
    x: tuple[str, ...] = ()
    id: str | None = None
    time: str | None = None
    target_time: str | None = None
    static: tuple[str, ...] = ()
    tz: object = None
    response: str = "presence_absence"
    metric: object = None


def n_targets(targets, spec: TimesiftSpec) -> int:
    """How many rows of targets there are, read off a column the spec is sure of."""
    return len(targets[spec.id if spec.id is not None else spec.static[0]])


def target_labels(targets, spec: TimesiftSpec) -> tuple[str, ...]:
    """What names the rows of every array in one fit.

    The unit identifier names them where a unit carries one target. Where ``target_time`` lets a
    unit carry several, no identifier tells them apart, so the row's own position does, which is
    what :func:`~timesift.representation.lookback_matrix` already names its targets by.
    """
    if spec.id is not None and spec.target_time is None:
        return tuple(_unit_names(targets[spec.id], spec.id))
    return tuple(str(i + 1) for i in range(n_targets(targets, spec)))


def auto_grains(series, spec: TimesiftSpec, stats=("mean",),
                year_start="09-01") -> tuple[str, ...]:
    """The named grains that give the record at least two bins, from the finest to the coarsest.

    The count comes from the calendar in the core rather than from arithmetic here, so a grain is
    admitted on the same rule that will bin it. It is read off one reading per distinct instant,
    which carries the record's whole span and its gaps at the cost of a single unit's memory.
    """
    probe = _probe(series, spec)
    day_level = any(s in DAY_LEVEL_STATS for s in stats)
    out = []
    for g in GRAINS:
        if day_level and g in ("native", "halfday"):
            continue
        counted = grain_matrix(probe, "id", "time", "value", grain=g, stats=("mean",),
                               year_start=year_start, tz=spec.tz)
        if counted.values.shape[1] >= 2:
            out.append(g)
    return tuple(out)


def expand_sift(sift, series, spec: TimesiftSpec) -> Sift:
    """The sift with ``"auto"`` replaced by the grains the record supports."""
    out: dict[str, Representation] = {}
    for label, rep in as_sift(sift).items():
        if rep.kind != "auto":
            out[label] = rep
            continue
        for g in auto_grains(series, spec, rep.stats, rep.year_start):
            out.setdefault(g, grain(g, rep.stats, rep.year_start))
    if not out:
        raise ValueError("no grain gives this record more than one bin, so there is nothing to "
                         "sift over. Check the series covers more than a single bin, or name the "
                         "grains.")
    return Sift(out)


def build_representation(rep: Representation, series, targets, spec: TimesiftSpec) -> TimesiftMatrix:
    """The array one representation names, for these targets, in their own row order."""
    labels = target_labels(targets, spec)
    if rep.kind == "static":
        return _static_block(targets, spec, labels)
    if series is None:
        raise ValueError(f"the {rep.label} representation reads a series, and none was given")
    if rep.kind == "grain":
        built = _grain_block(rep.grain, rep.stats, rep.year_start, series, spec, labels)
    elif rep.kind == "multigrain":
        built = _multigrain_block(rep, series, spec, labels)
    elif rep.kind == "lookback":
        built = _lookback_block(rep, series, targets, spec, labels)
    else:
        raise ValueError(f"a {rep.kind} representation is expanded before it is built")
    return _with_static(built, targets, spec)


def _grain_block(name, stats, year_start, series, spec, labels) -> TimesiftMatrix:
    parts = [_channels(grain_matrix(series, spec.id, spec.time, v, grain=name, stats=stats,
                                    year_start=year_start, tz=spec.tz), v, spec)
             for v in spec.x]
    return _order(parts[0] if len(parts) == 1 else bind_channels(*parts), labels)


def _multigrain_block(rep, series, spec, labels) -> TimesiftMatrix:
    named = rep.grains if rep.grains is not None else auto_grains(series, spec, rep.stats,
                                                                  rep.year_start)
    blocks, channels, built = [], [], []
    for g in named:
        m = _grain_block(g, rep.stats, rep.year_start, series, spec, labels)
        built.append(m)
        blocks.append(flatten(m))
        # flatten() runs the bin fastest inside each channel, and the names have to say the same.
        channels += [f"{g}:{b}:{c}" for c in m.stats for b in m.bins]
    values = np.concatenate(blocks, axis=1)[:, None, :]
    return _derived(values, labels, ("features",), channels, "multigrain", spec,
                    bin_n=built[0].bin_n.sum(axis=1, keepdims=True),
                    bin_start=built[0].bin_start[:1], bin_end=built[0].bin_end[-1:])


def _needs_target_time(labels) -> None:
    """What a representation anchored on the target needs, said in one place: the run raises it
    before anything is fitted, and the builder raises it at the one door that reaches a lookback
    without going through that check."""
    raise ValueError(f"{labels} reads a stretch of record ending at each target's own instant, so "
                     "`target_time` has to name the column of `targets` holding it.")


def _lookback_block(rep, series, targets, spec, labels) -> TimesiftMatrix:
    if spec.target_time is None:
        _needs_target_time(rep.label)
    at = {"id": _unit_names(targets[spec.id], spec.id),
          "at": np.asarray(targets[spec.target_time], dtype="datetime64[s]")}
    parts = []
    for v in spec.x:
        w = lookback_matrix(series, spec.id, spec.time, v, at=at, span=rep.span, lag=rep.lag,
                            bins=rep.bins, stats=rep.stats, tz=spec.tz)
        parts.append(_channels(replace(w, units=tuple(labels)), v, spec))
    return parts[0] if len(parts) == 1 else bind_channels(*parts)


def _static_block(targets, spec, labels) -> TimesiftMatrix:
    if not spec.static:
        raise ValueError("a static representation needs the predictor columns named in `static`")
    values = _static_values(targets, spec)[:, None, :]
    return _derived(values, labels, ("features",), spec.static, "static", spec)


def _with_static(m: TimesiftMatrix, targets, spec) -> TimesiftMatrix:
    """Columns of the target table are carried as channels held constant over the bins, which is
    the constant an encoder reads beside the readings. Which channels those are is recorded, so
    that a learner reading the bins as a block of features takes one predictor from each of them
    rather than one per bin."""
    if not spec.static:
        return m
    block = _static_values(targets, spec)
    if block.shape[0] != m.values.shape[0]:
        raise ValueError(f"{block.shape[0]} target rows and {m.values.shape[0]} representation "
                         f"rows")
    names = tuple(m.stats) + tuple(spec.static)
    clash = [n for n in spec.static if n in m.stats]
    if clash:
        raise ValueError(f"the static column{'s' if len(clash) > 1 else ''} "
                         f"{', '.join(clash)} would take the name of a channel the "
                         f"representation already carries")
    extra = np.repeat(block[:, None, :], m.values.shape[1], axis=1)
    return replace(m, values=np.ascontiguousarray(np.concatenate([m.values, extra], axis=2)),
                   stats=names, static=tuple(m.static) + tuple(spec.static))


def _static_values(targets, spec) -> np.ndarray:
    columns = []
    for c in spec.static:
        try:
            values = np.asarray(targets[c], dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise ValueError(f"the static column {c} is not numeric. Encode it as numbers before "
                             f"naming it in `static`.") from e
        if np.isnan(values).any():
            raise ValueError(f"missing values in the static column {c}. "
                             "Fill or drop them before fitting.")
        columns.append(values)
    return np.column_stack(columns)


def _channels(m: TimesiftMatrix, value: str, spec) -> TimesiftMatrix:
    """One value column keeps the statistic's own name; several carry the column's name too, so
    the channels of a two-sensor record say which sensor each of them came from."""
    if len(spec.x) < 2:
        return m
    return replace(m, stats=tuple(f"{value}_{s}" for s in m.stats))


def _order(m: TimesiftMatrix, labels) -> TimesiftMatrix:
    position = {u: i for i, u in enumerate(m.units)}
    missing = [u for u in labels if u not in position]
    if missing:
        raise ValueError(f"{len(missing)} target{'s' if len(missing) > 1 else ''} name a unit the "
                         f"series does not carry, first: {missing[0]}.")
    return m.take_units([position[u] for u in labels])


def _derived(values, labels, bins, stats, name, spec, bin_n=None, bin_start=None,
             bin_end=None) -> TimesiftMatrix:
    """A representation the calendar did not bin: its bins are named by the reduction rather than
    by an instant, so it carries no bin boundaries."""
    n_b = len(bins)
    nat = np.full(n_b, np.datetime64("NaT"), dtype="datetime64[s]")
    return TimesiftMatrix(
        values=np.ascontiguousarray(values), units=tuple(labels), bins=tuple(bins),
        stats=tuple(stats), grain=name, year_start=None,
        bin_start=nat if bin_start is None else bin_start,
        bin_end=nat if bin_end is None else bin_end,
        bin_n=np.zeros((len(labels), n_b), dtype=np.int64) if bin_n is None else bin_n,
        bin_partial=np.zeros(n_b, dtype=bool))


def _probe(series, spec) -> dict:
    when = np.unique(np.asarray(series[spec.time], dtype="datetime64[s]"))
    return {"id": ["probe"] * len(when), "time": when, "value": np.zeros(len(when))}


# ---- checks ------------------------------------------------------------------------------------

def _stats(stats, grain) -> tuple[str, ...]:
    """The statistics, refused here rather than when the array is built: a specification that
    names a statistic the grain has no definition for is wrong before any record is read."""
    return tuple(_check_stats(stats, grain))


def _year_start(year_start: str) -> str:
    _parse_year_start(year_start)
    return year_start


def _check_grain(g) -> str:
    if g not in GRAINS:
        raise ValueError(f"unknown grain: {g}. Available: {', '.join(GRAINS)}")
    return str(g)


def _check_bins(bins) -> int:
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or int(bins) < 1:
        raise ValueError("`bins` must be a positive whole number")
    return int(bins)


def _whole(x, arg) -> int:
    if isinstance(x, bool) or not isinstance(x, (int, np.integer)):
        raise ValueError(f"`{arg}` must be a whole number, got {x!r}")
    return int(x)


def _lookback_label(span, lag, bins) -> str:
    parts = [str(span)]
    if lag not in (0, "0 days"):
        parts.append(f"lag {lag}")
    if bins != 1:
        parts.append(f"{bins} bins")
    return " ".join(parts)


def _flatten(args) -> list:
    out = []
    for a in args:
        if isinstance(a, (list, tuple)):
            out.extend(a)
        else:
            out.append(a)
    return out
