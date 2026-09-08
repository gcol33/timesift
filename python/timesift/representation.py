"""Reduce sensor series to a temporal grain.

Implements ``inst/spec/representation.md``. Where this file and that document disagree, the document is
right: it is also what the R side answers to, and the two are held together by the digests in
``inst/spec/fixtures/``.

The binning and the reduction are not written here. They are ``src/ts_core.cpp``, compiled into
``timesift._core`` and into the R package alike, so the two languages agree by construction rather
than by two implementations being checked against each other after the fact. What is written here
is the boundary: resolving the columns, resolving the zone, and putting the result into a
``TimesiftMatrix``.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime

import numpy as np

from . import _core

GRAINS = ("native", "halfday", "day", "week", "month", "season", "year")
DAY_LEVEL_STATS = ("cold_day", "warm_day", "mean_daily_min", "mean_daily_max")
STATS = ("mean", "min", "max") + DAY_LEVEL_STATS

# A year is 365 days and a month is 30 days here. A lookback of a fixed length is a fixed length,
# not a calendar step: every target has to read the same amount of record for the representations
# to be comparable, which a February and a leap year would take away.
DURATION_SECONDS = {"second": 1, "seconds": 1, "minute": 60, "minutes": 60, "hour": 3600,
                    "hours": 3600, "day": 86400, "days": 86400, "week": 604800, "weeks": 604800,
                    "month": 2592000, "months": 2592000, "year": 31536000, "years": 31536000}


@dataclass(frozen=True)
class TimesiftMatrix:
    """A ``[row, bin, channel]`` representation and the reduction that produced it.

    A row is a unit where the calendar did the binning and a target where a lookback did, since a
    unit carrying several targets cannot name a row on its own. ``span`` and ``lag`` are set by a
    lookback alone, and are what rebuilding one for new targets reads. ``static`` names the
    channels holding the same number in every bin, which :func:`flatten` reads once each.

    ``bin_start``, ``bin_end`` and ``bin_partial`` are the calendar's, and are ``None`` on a
    representation the calendar did not bin.
    """

    values: np.ndarray
    units: tuple[str, ...]
    bins: tuple[str, ...]
    stats: tuple[str, ...]
    grain: str
    year_start: str | None
    bin_start: np.ndarray | None
    bin_end: np.ndarray | None
    bin_n: np.ndarray = field(repr=False)
    bin_partial: np.ndarray | None = field(repr=False)
    span: int | None = None
    lag: int | None = None
    static: tuple[str, ...] = ()

    @property
    def shape(self) -> tuple[int, int, int]:
        """Rows, bins and channels."""
        return self.values.shape

    def channel(self, name: str) -> np.ndarray:
        """One statistic as a `[row, bin]` matrix."""
        return self.values[:, :, self.stats.index(name)]

    def take_units(self, index) -> "TimesiftMatrix":
        """The representation restricted to a subset of its rows, in the order given."""
        index = np.asarray(index)
        return replace(self, values=self.values[index], bin_n=self.bin_n[index],
                       units=tuple(np.asarray(self.units)[index]))

    def __repr__(self) -> str:  # pragma: no cover - display only
        n_u, n_b, n_c = self.values.shape
        row = "targets" if self.grain == "lookback" else "units"
        return (f"<timesift matrix> {n_u} {row} x {n_b} bins x {n_c} channels\n"
                f"grain: {self.grain}  stats: {', '.join(self.stats)}\n"
                f"from  : {self.bins[0]} to {self.bins[-1]}")


class TimesiftSet(Mapping):
    """A ladder of representations, one per grain.

    Naming several grains in :func:`grain_matrix` returns one of these: representations of the
    same units, differing only in how coarsely the record was read. It is what
    :func:`grain_ladder` fits across, and it reads as a mapping of grain name to representation.
    """

    def __init__(self, parts):
        if isinstance(parts, TimesiftMatrix):
            parts = {parts.grain: parts}
        parts = dict(parts)
        if not parts:
            raise ValueError("a timesift set is a non-empty mapping of grain_matrix() results")
        bad = [k for k, v in parts.items() if not isinstance(v, TimesiftMatrix)]
        if bad:
            raise ValueError(f"{', '.join(bad)} "
                             f"{'are' if len(bad) > 1 else 'is'} not a grain_matrix() result")
        units = next(iter(parts.values())).units
        differ = [k for k, v in parts.items() if v.units != units]
        if differ:
            raise ValueError(f"every grain in a set must cover the same units; "
                             f"{', '.join(differ)} does not")
        self._parts = parts

    def __getitem__(self, key):
        if isinstance(key, (list, tuple)):
            return TimesiftSet({k: self._parts[k] for k in key})
        return self._parts[key]

    def __iter__(self):
        return iter(self._parts)

    def __len__(self) -> int:
        return len(self._parts)

    @property
    def units(self) -> tuple[str, ...]:
        """The units the set covers, which every grain in it shares."""
        return next(iter(self._parts.values())).units

    def __repr__(self) -> str:  # pragma: no cover - display only
        lines = [f"<timesift set> {len(self._parts)} grains over {len(self.units)} units"]
        for name, m in self._parts.items():
            lines.append(f"  {name:<10} {m.values.shape[1]:5d} bins x {m.values.shape[2]} "
                         f"channels ({', '.join(m.stats)})")
        return "\n".join(lines)


def timesift_set(x) -> TimesiftSet:
    """Every entry point that fits across grains takes a representation, a set, or a bare mapping,
    and works on a set. One coercion, so no caller repeats the three cases."""
    return x if isinstance(x, TimesiftSet) else TimesiftSet(x)


def grain_matrix(data=None, id=None, time=None, value=None, *, grain="day", stats=("mean",),
                  year_start="09-01", partial="keep", tz=None):
    """Bin readings by the calendar and summarise every bin.

    ``data`` is a mapping of column name to sequence, or any object with ``__getitem__`` over the
    three column names given by ``id``, ``time`` and ``value``. Naming two or more grains returns
    a :class:`TimesiftSet`; naming one, whether as a string or as a sequence of one, returns the
    representation itself. ``grain`` may also be a callable, which is handed the reading instants
    and must return the start of each reading's bin.

    ``tz`` names the calendar to bin by. Left at ``None`` the instants are taken as already
    expressed in that calendar, which is what a zone-free ``datetime64`` says and what the R side
    does for a series carried in UTC. Given a zone name, the instants are read as UTC and binned by
    that zone's clock, which is what the R side does for a series carrying a ``tzone``: the same
    instants and the same zone give the same answer in both languages. A time column that carries
    a zone of its own names the calendar the same way, so a zone-aware column bins by its own
    clock without being told to; naming a different one in ``tz`` beside it is an error.

    ``partial`` says what becomes of a bin the record does not cover for its whole calendar span,
    which is what a record beginning or ending away from a bin boundary produces. ``"keep"``, the
    default, returns it alongside the full bins; ``"drop"`` removes it. Either way the verdict is
    carried on ``bin_partial``, so a kept partial bin is labelled rather than silent. A
    caller-supplied binning declares its own bins, so the package cannot know where the last one
    was meant to end and takes the record's end as its end.
    """
    unit, when, reading, carried = _columns(data, id, time, value)
    partial = _check_partial(partial)

    if not callable(grain) and not isinstance(grain, str):
        named = _check_grains(grain)
        parts = {w: grain_matrix(data, id, time, value, grain=w, stats=stats,
                                  year_start=year_start, partial=partial, tz=tz)
                 for w in named}
        # One grain is one representation, as it is when it is named as a string: a set of one is
        # a shape the caller then has to unwrap for no reason, and the R side does not make one.
        return parts[named[0]] if len(named) == 1 else TimesiftSet(parts)

    stats = _check_stats(stats, grain)
    ys = _parse_year_start(year_start)
    zone = _resolve_zone(tz, carried)

    instant = np.ascontiguousarray(when.astype("datetime64[s]").astype(np.int64))
    units, unit_ix = np.unique(unit, return_inverse=True)
    unit_ix = np.ascontiguousarray(unit_ix.astype(np.int32))
    _check_readings(units, unit_ix, instant, when)
    local = _naive_seconds(instant, zone)

    supplied = None
    if callable(grain):
        given = np.asarray(grain(when), dtype="datetime64[s]")
        if given.shape != when.shape:
            raise ValueError("a grain function must return one bin start per reading")
        supplied = _naive_seconds(np.ascontiguousarray(given.astype(np.int64)), zone)

    name = "custom" if callable(grain) else grain
    values, bin_start, bin_end, bin_n, bin_partial = _core.reduce(
        unit_ix, reading, instant, local, supplied, [str(u) for u in units],
        name, ys[0], ys[1], list(stats), _sampling_step(instant))

    n_u, n_b, n_c = len(units), len(bin_start), len(stats)
    starts = _local_to_instant(bin_start, zone).astype("datetime64[s]")
    x = TimesiftMatrix(
        values=np.ascontiguousarray(values.reshape(n_c, n_b, n_u).transpose(2, 1, 0)),
        units=tuple(str(u) for u in units), bins=tuple(_iso(b) for b in starts),
        stats=tuple(stats), grain=name, year_start=year_start,
        bin_start=starts, bin_end=bin_end.astype("datetime64[s]"),
        bin_n=bin_n.reshape(n_b, n_u).T, bin_partial=bin_partial.astype(bool),
    )
    return _drop_partial(x) if partial == "drop" else x


@dataclass(frozen=True)
class Coverage:
    """How many readings each unit has in each bin, over every bin the calendar tiles the record
    with. ``count`` is ``[unit, bin]``; a unit that started late, stopped early or lost a month is
    a row with zeros in it, and a bin the whole record skips is a column of zeros."""

    count: np.ndarray
    units: tuple[str, ...]
    bins: tuple[str, ...]
    grain: str
    bin_start: np.ndarray

    @property
    def empty(self) -> np.ndarray:
        """The ``[unit, bin]`` mask of cells holding no reading."""
        return self.count == 0

    def units_with_gaps(self) -> tuple[str, ...]:
        """The units that do not reach every bin."""
        return tuple(u for u, row in zip(self.units, self.empty) if row.any())

    def bins_no_unit_reaches(self) -> tuple[str, ...]:
        """The bins the whole record skips."""
        return tuple(b for b, col in zip(self.bins, self.empty.T) if col.all())

    def __repr__(self) -> str:  # pragma: no cover - display only
        head = (f"<timesift coverage> {len(self.units)} units x {len(self.bins)} bins at the "
                f"{self.grain} grain")
        if not self.empty.any():
            return head + "\nevery unit reaches every bin"
        gaps = self.units_with_gaps()
        lines = [head, f"{int(self.empty.sum())} empty (unit, bin) cells in {len(gaps)} units: "
                 + ", ".join(gaps[:5]) + (", ..." if len(gaps) > 5 else "")]
        skipped = self.bins_no_unit_reaches()
        if skipped:
            lines.append(f"{len(skipped)} bins no unit reaches: " + ", ".join(skipped[:5])
                         + (", ..." if len(skipped) > 5 else ""))
        return "\n".join(lines)


def coverage(data=None, id=None, time=None, *, grain="day", year_start="09-01",
             tz=None) -> Coverage:
    """Which units reach which bins.

    A representation needs every unit in every bin, and :func:`grain_matrix` refuses a record
    where one is missing rather than pad it. This is the same binning laid out so the gaps can be
    read: how many readings each unit has in each bin, over every bin the calendar tiles the
    record with from the first bin any unit touches to the last. What to do about a gap is the
    analyst's decision, and this is the table it is made on; nothing here fills a cell. ``grain``
    and ``tz`` read as they do for :func:`grain_matrix`.
    """
    if not (callable(grain) or isinstance(grain, str)):
        raise ValueError("`coverage()` reads one grain at a time")
    unit, when, _, carried = _columns(data, id, time, None)
    if not callable(grain):
        _check_grains([grain])
    ys = _parse_year_start(year_start)
    zone = _resolve_zone(tz, carried)

    instant = np.ascontiguousarray(when.astype("datetime64[s]").astype(np.int64))
    units, unit_ix = np.unique(unit, return_inverse=True)
    unit_ix = np.ascontiguousarray(unit_ix.astype(np.int32))
    _check_readings(units, unit_ix, instant, when)
    local = _naive_seconds(instant, zone)

    supplied = None
    if callable(grain):
        given = np.asarray(grain(when), dtype="datetime64[s]")
        if given.shape != when.shape:
            raise ValueError("a grain function must return one bin start per reading")
        supplied = _naive_seconds(np.ascontiguousarray(given.astype(np.int64)), zone)

    name = "custom" if callable(grain) else grain
    bin_start, count = _core.coverage(unit_ix, local, supplied, [str(u) for u in units], name,
                                      ys[0], ys[1])
    starts = _local_to_instant(bin_start, zone).astype("datetime64[s]")
    return Coverage(count=count.reshape(len(bin_start), len(units)).T.copy(),
                    units=tuple(str(u) for u in units), bins=tuple(_iso(b) for b in starts),
                    grain=name, bin_start=starts)


def _drop_partial(x: TimesiftMatrix) -> TimesiftMatrix:
    """Keeping or dropping a partial bin is the caller's choice, so the array is built over every
    bin the calendar produced and the unwanted ones are removed afterwards, which keeps one binning
    path rather than one per setting."""
    keep = np.flatnonzero(~x.bin_partial)
    if not len(keep):
        raise ValueError(f"dropping the partial bins leaves no bin: the record covers no whole "
                         f"{x.grain}. Use partial='keep' or a finer grain.")
    if len(keep) == len(x.bin_partial):
        return x
    return replace(x, values=x.values[:, keep, :], bins=tuple(x.bins[i] for i in keep),
                   bin_start=x.bin_start[keep], bin_end=x.bin_end[keep],
                   bin_n=x.bin_n[:, keep], bin_partial=x.bin_partial[keep])


def lookback_matrix(data=None, id=None, time=None, value=None, at=None, span=None, *,
                  lag="0 days", bins=1, stats="mean", tz=None):
    """Read a fixed length of record ending a fixed lag before each target's own instant.

    It is the reduction a calendar cannot express: two targets on the same unit a fortnight apart
    read two different stretches of the same series, so the bins are relative to the target rather
    than to a month or a week.

    ``at`` is a mapping with an ``"id"`` array of units and an ``"at"`` array of anchor instants,
    one row per target; a unit may carry any number of them. Bin ``b`` of a target anchored at
    ``a`` covers ``[a - lag - span + b * step, a - lag - span + (b + 1) * step)``, with ``step`` the
    span divided by ``bins`` and ``b`` counted from zero. Only the readings of the target's own
    unit are read, and every ``(target, bin)`` cell must hold at least one: a lookback reaching past
    the record is an error naming the target, never a padded row.

    ``span`` and ``lag`` are read from a count and a unit -- ``"30 days"``, ``"12 hours"``,
    ``"1 year"`` -- or from a bare number of seconds. A year is 365 days and a month is 30 days
    here, because a lookback of a fixed length is a fixed length rather than a calendar step.

    ``tz`` names the calendar, as it does for :func:`grain_matrix`. The anchors are instants and
    are read as a clock in that same calendar, so one record is binned by one calendar.
    """
    unit, when, reading, carried = _columns(data, id, time, value)
    span = _parse_duration(span, "span")
    lag = _parse_duration(lag, "lag")
    bins = _check_bins(bins)
    stats = _check_stats(stats, "lookback")
    zone = _resolve_zone(tz, carried)

    instant = np.ascontiguousarray(when.astype("datetime64[s]").astype(np.int64))
    units, unit_ix = np.unique(unit, return_inverse=True)
    unit_ix = np.ascontiguousarray(unit_ix.astype(np.int32))
    _check_readings(units, unit_ix, instant, when)
    local = _naive_seconds(instant, zone)

    target_unit, anchor, labels = _targets(at, units, zone)
    values, bin_n = _core.reduce_lookbacks(
        unit_ix, reading, local, [str(u) for u in units], target_unit, anchor, list(labels),
        span, lag, bins, list(stats))

    n_t = len(labels)
    # No bin_start, bin_end or bin_partial: a bin is a position relative to an anchor rather than
    # a span of the calendar, and a cell the record does not cover is an error rather than a
    # verdict. R carries no such attribute here, and a column of NaT beside a column of zeros
    # would read as an answer.
    return TimesiftMatrix(
        values=np.ascontiguousarray(values.reshape(len(stats), bins, n_t).transpose(2, 1, 0)),
        units=labels, bins=_bin_offsets(span, lag, bins), stats=tuple(stats),
        grain="lookback", year_start=None, bin_start=None, bin_end=None,
        bin_n=bin_n.reshape(bins, n_t).T, bin_partial=None, span=span, lag=lag)


def _targets(at, units, zone):
    """A target is a unit and an instant, and its identity is its position in ``at``: a unit may
    carry several targets, so the unit cannot name a row. The anchors go through the same boundary
    the readings do, so both are read as a clock in the series' own calendar."""
    if at is None or "id" not in at or "at" not in at:
        raise ValueError('`at` must give an "id" naming the unit and an "at" anchor instant for '
                         "every target")
    who = _unit_names(at["id"], "id")
    anchor = np.asarray(at["at"], dtype="datetime64[s]")
    if len(who) != len(anchor):
        raise ValueError("`at` must give one anchor per target")
    if not len(who):
        raise ValueError("`at` holds no target")
    if np.isnat(anchor).any():
        raise ValueError("missing values in `at`. "
                         "Fill or drop them before building a representation.")

    index = np.searchsorted(units, who)
    unknown = (index >= len(units)) | (units[np.minimum(index, len(units) - 1)] != who)
    if unknown.any():
        n = int(unknown.sum())
        raise ValueError(f"{n} target{'s' if n > 1 else ''} name a unit the series does not "
                         f"carry, first: {who[np.flatnonzero(unknown)[0]]}.")

    seconds = _naive_seconds(np.ascontiguousarray(anchor.astype(np.int64)), zone)
    return (np.ascontiguousarray(index.astype(np.int32)), np.ascontiguousarray(seconds),
            tuple(str(i + 1) for i in range(len(who))))


def _check_bins(bins):
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or int(bins) < 1:
        raise ValueError("`bins` must be a positive whole number")
    return int(bins)


def _parse_duration(x, arg):
    if isinstance(x, bool) or x is None:
        raise ValueError(f'`{arg}` must be a whole number of seconds or a count and a unit, '
                         f'like "30 days"')
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x) or float(x) != int(x):
            raise ValueError(f'`{arg}` must be a whole number of seconds or a count and a unit, '
                             f'like "30 days"')
        return int(x)
    if not isinstance(x, str):
        raise ValueError(f'`{arg}` must be a whole number of seconds or a count and a unit, '
                         f'like "30 days"')
    parts = re.fullmatch(r" *([0-9]+) *([A-Za-z]*) *", x)
    if parts is None:
        raise ValueError(f'`{arg}` must be a count and a unit, like "30 days", got "{x}"')
    if not parts.group(2):
        return int(parts.group(1))
    size = DURATION_SECONDS.get(parts.group(2).lower())
    if size is None:
        raise ValueError(f'unknown duration unit "{parts.group(2)}" in `{arg}`. Available: '
                         "seconds, minutes, hours, days, weeks, months, years")
    return int(parts.group(1)) * size


def _bin_offsets(span, lag, bins):
    """The second dimension is the lookback itself, so a bin is named by where it opens relative to
    the anchor rather than by an instant no two targets share."""
    step = span // bins
    return tuple(_format_duration(b * step - lag - span) for b in range(bins))


def _format_duration(x: int) -> str:
    for name, size in (("day", 86400), ("hour", 3600), ("minute", 60), ("second", 1)):
        if x % size == 0:
            n = x // size
            return f"{n} {name}" + ("" if abs(n) == 1 else "s")


def calendar_channels(x: TimesiftMatrix) -> TimesiftMatrix:
    """Where in the year each bin sits, as the sine and cosine of its fractional position."""
    mid = x.bin_start + (x.bin_end - x.bin_start) / 2
    year = mid.astype("datetime64[Y]")
    length = (year + 1).astype("datetime64[s]").astype(np.int64) \
        - year.astype("datetime64[s]").astype(np.int64)
    frac = (mid.astype("datetime64[s]").astype(np.int64)
            - year.astype("datetime64[s]").astype(np.int64)) / length

    n_u = x.values.shape[0]
    out = np.empty((n_u, len(frac), 2), dtype=np.float64)
    out[:, :, 0] = np.sin(2 * np.pi * frac)
    out[:, :, 1] = np.cos(2 * np.pi * frac)
    return replace(x, values=out, stats=("year_sin", "year_cos"))


def bind_channels(*parts: TimesiftMatrix) -> TimesiftMatrix:
    """Put the channels of several representations of the same units and bins side by side."""
    if len(parts) < 2:
        raise ValueError("bind_channels() needs at least two representations")
    first = parts[0]
    for k, p in enumerate(parts[1:], start=1):
        if p.units != first.units or p.bins != first.bins:
            raise ValueError(f"argument {k} covers different units or bins from the first")
    names = tuple(s for p in parts for s in p.stats)
    if len(set(names)) != len(names):
        raise ValueError("two representations carry a channel of the same name")
    return replace(first, values=np.concatenate([p.values for p in parts], axis=2), stats=names)


# ---- the zone ---------------------------------------------------------------------------------

def _column_zone(column):
    """The zone a time column carries, or ``None`` where it carries none.

    Reading the column as instants drops it: a zone-aware column becomes the UTC seconds it stands
    for and nothing says which clock it was written on, so it is read off the column itself here or
    it is not read at all.
    """
    for holder in (getattr(column, "dt", None), getattr(column, "dtype", None), column):
        zone = getattr(holder, "tz", None)
        if zone is not None:
            return zone
    if isinstance(column, (list, tuple)) and len(column):
        return getattr(column[0], "tzinfo", None)
    return None


def _zone_key(zone):
    """A zone as the name two of them are compared by. UTC is the calendar an instant already
    reads as, so here it is the same thing as no zone at all."""
    if zone is None:
        return None
    key = getattr(zone, "key", None) or str(zone)
    return None if key in ("UTC", "GMT") else key


def _resolve_zone(tz, carried):
    """The calendar to bin by.

    A time column that carries a zone names it, which is what makes a zone-aware column bin the
    same way in both languages. ``tz`` beside one must agree with it: two zones are two answers,
    and neither of them is the one to take without saying so.
    """
    given, held = _zone_key(tz), _zone_key(carried)
    if given is not None and held is not None and given != held:
        raise ValueError(f"the time column is written in {held} and `tz` names {given}. A record "
                         f"is binned by one calendar; give one of the two.")
    return _zone(carried) if held is not None else _zone(tz)


def _zone(tz):
    """The zone lives at this boundary and nowhere else; the core bins a calendar with no zone in
    it. ``None``, and UTC, mean the instants already read as the calendar to bin by."""
    if tz is None:
        return None
    if isinstance(tz, str):
        if tz in ("UTC", "GMT"):
            return None
        from zoneinfo import ZoneInfo
        return ZoneInfo(tz)
    return tz


def _naive_seconds(instant: np.ndarray, zone) -> np.ndarray:
    """The clock each instant reads in ``zone``, as seconds. Defined for every instant in every
    zone; it is the reverse direction that is not."""
    if zone is None:
        return instant
    u, inverse = np.unique(instant, return_inverse=True)
    offset = np.fromiter((_offset_at(int(t), zone) for t in u), dtype=np.int64, count=len(u))
    return np.ascontiguousarray((u + offset)[inverse])


def _local_to_instant(local: np.ndarray, zone) -> np.ndarray:
    """The instant whose clock in ``zone`` reads each given local time. The offsets in force a day
    either side bracket any transition, so one of the three candidates is it. A local time the
    clock skipped has no instant at all, and the answer is then the instant the clock jumped to; a
    local time the clock repeated has two, and the answer is the first of them."""
    if zone is None:
        return local
    out = np.empty(len(local), dtype=np.int64)
    for i, value in enumerate(local):
        target = int(value)
        candidate = [target - _offset_at(target + shift, zone) for shift in (-86400, 0, 86400)]
        valid = [c for c in candidate if c + _offset_at(c, zone) == target]
        out[i] = min(valid) if valid else max(candidate)
    return out


def _offset_at(instant: int, zone) -> int:
    return int(datetime.fromtimestamp(instant, zone).utcoffset().total_seconds())


def _sampling_step(instant: np.ndarray) -> int:
    u = np.unique(instant)
    return int(np.diff(u).min()) if len(u) > 1 else 0


# ---- checks ----------------------------------------------------------------------------------

def _unit_names(values, column) -> np.ndarray:
    """The names an identifier column gives its units, under the one rule both languages read it
    by.

    A string id is itself and a category is its label; a whole number is its digits, with no
    exponent and no decimal point, so an id read as 100000 from a file is ``100000`` rather than
    Python's ``100000.0`` beside R's ``1e+05``. A number that is not whole has no such writing and
    is refused, as is a value of any other type: a response aligned by name against a
    representation built in the other language would otherwise match none of its units.
    """
    out = []
    for v in values:
        if isinstance(v, (str, np.str_)):
            out.append(str(v))
        elif isinstance(v, (bool, np.bool_)):
            raise ValueError(f"`{column}` identifies a unit by {v!r}, which is not text, a "
                             "category or a whole number.")
        elif isinstance(v, (int, np.integer)):
            out.append(str(int(v)))
        elif isinstance(v, (float, np.floating)) and float(v).is_integer():
            out.append(str(int(v)))
        elif isinstance(v, (float, np.floating)):
            raise ValueError(f"`{column}` identifies a unit by {float(v)!r}, which is not a whole "
                             "number. An identifier is a name; round it or write it as text "
                             "before naming it.")
        else:
            raise ValueError(f"`{column}` must identify a unit by text, a category or a whole "
                             f"number, not {type(v).__name__}.")
    return np.asarray(out, dtype=object).astype(str)


def _columns(data, id, time, value):
    """The unit, instant and reading columns, and the zone the time column was carrying.

    ``reading`` is ``None`` where no value column is read. The instants come back as the UTC
    seconds every path below here works in, which is what numpy makes of a column in any zone.
    """
    raw = list(data[id])
    if any(v is None or (isinstance(v, float) and v != v) for v in raw):
        raise ValueError("missing values in the readings. "
                         "Fill or drop them before building a representation.")
    unit = _unit_names(raw, id)
    column = data[time]
    zone = _column_zone(column)
    with warnings.catch_warnings():
        # numpy warns that reading the column as instants keeps no zone, which is true of the
        # array and is why the zone is taken off the column first. Having taken it, the warning
        # is answered.
        if zone is not None:
            warnings.simplefilter("ignore", UserWarning)
        when = np.asarray(column, dtype="datetime64[s]")
    reading = None if value is None else \
        np.ascontiguousarray(np.asarray(data[value], dtype=np.float64))
    if len(unit) != len(when) or (reading is not None and len(reading) != len(unit)):
        raise ValueError("the columns must be the same length")
    return unit, when, reading, zone


def _check_grains(grain):
    grain = list(grain)
    bad = [w for w in grain if w not in GRAINS]
    if bad:
        raise ValueError(f"unknown grain: {', '.join(bad)}. Available: {', '.join(GRAINS)}")
    if len(set(grain)) != len(grain):
        raise ValueError("`grain` names a grain twice")
    return grain


def _check_stats(stats, grain):
    stats = [stats] if isinstance(stats, str) else list(stats)
    bad = [s for s in stats if s not in STATS]
    if bad:
        raise ValueError(f"unknown statistic: {', '.join(bad)}. Available: {', '.join(STATS)}")
    if len(set(stats)) != len(stats):
        raise ValueError("`stats` names a statistic twice")
    if isinstance(grain, str) and grain in ("native", "halfday"):
        day_level = [s for s in stats if s in DAY_LEVEL_STATS]
        if day_level:
            raise ValueError(f"`{grain}` bins are shorter than a day, so "
                             f"{' and '.join(day_level)} is not defined there. "
                             "Use a grain of `day` or coarser.")
    return stats


def _check_partial(partial):
    if partial not in ("keep", "drop"):
        raise ValueError(f'`partial` must be "keep" or "drop", got "{partial}"')
    return partial


def _parse_year_start(year_start: str):
    parts = year_start.split("-")
    if len(parts) != 2 or not all(len(p) == 2 and p.isdigit() for p in parts):
        raise ValueError(f'`year_start` must look like "MM-DD", got "{year_start}"')
    month, day = int(parts[0]), int(parts[1])
    if not (1 <= month <= 12 and 1 <= day <= 28):
        raise ValueError(f'`year_start` must be a month 01-12 and a day 01-28, got "{year_start}"')
    return month, day


def _check_readings(units, unit_ix, instant, when):
    """The instants are the whole seconds the calendar is read at, so two readings a fraction of a
    second apart are the same reading twice here. Sorting by (unit, time) and looking at neighbours
    costs no string per reading, which on a record of tens of millions matters.

    A reading's own value is not read here: whether it is a number a bin can hold is the core's
    guard, raised once for both languages.
    """
    if np.isnat(when).any():
        raise ValueError("missing values in the readings. "
                         "Fill or drop them before building a representation.")
    if len(instant) < 2:
        return
    order = np.lexsort((instant, unit_ix))
    same = ((unit_ix[order][1:] == unit_ix[order][:-1])
            & (instant[order][1:] == instant[order][:-1]))
    if same.any():
        first = order[int(np.flatnonzero(same)[0]) + 1]
        raise ValueError(f"{int(same.sum())} duplicated (unit, time) pairs, "
                         f"first: {units[unit_ix[first]]} at {_iso(when[first])}")


def _iso(t: np.datetime64) -> str:
    return np.datetime_as_string(t.astype("datetime64[s]"), unit="s") + "Z"
