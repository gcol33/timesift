"""The representation as it was written in NumPy alone, before the two languages shared a core.

Nothing imports this outside the suite and the package never reaches it at runtime. It is kept
because it was written from ``inst/spec/representation.md`` rather than from the R source, and one
shared binary would otherwise make the agreement between the two languages trivially true. One
implementation in production, two in evidence.

It reads the calendar off zone-free ``datetime64`` values. A series carried in a zone is first
relabelled into that zone's clock, instant by instant through ``zoneinfo``, and every grain but
``native`` is then binned on the relabelled series; ``native`` is binned on the instants
themselves, because its bin is the reading and two readings of an hour a zone repeats are two
bins.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np

DAY_LEVEL_STATS = ("cold_day", "warm_day", "mean_daily_min", "mean_daily_max")

_HOUR = np.timedelta64(1, "h")
_DAY = np.timedelta64(1, "D")


def oracle_local_clock(when: np.ndarray, tz) -> np.ndarray:
    """The instants read as a clock in ``tz``, as zone-free ``datetime64`` values."""
    if tz is None or tz in ("UTC", "GMT"):
        return when
    zone = ZoneInfo(tz)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    seconds = when.astype("datetime64[s]").astype(np.int64)
    clock = [(epoch + timedelta(seconds=int(t))).astimezone(zone).replace(tzinfo=None)
             for t in seconds]
    return np.asarray(clock, dtype="datetime64[s]")


def oracle_bin_start(when: np.ndarray, grain, ys) -> np.ndarray:
    if callable(grain):
        out = np.asarray(grain(when), dtype="datetime64[s]")
        if out.shape != when.shape:
            raise ValueError("a grain function must return one bin start per reading")
        return out
    if grain == "native":
        return when
    day = when.astype("datetime64[D]").astype("datetime64[s]")
    if grain == "halfday":
        hour = (when.astype("datetime64[h]") - when.astype("datetime64[D]")).astype(np.int64)
        return day + np.where(hour >= 12, 12, 0) * _HOUR
    if grain == "day":
        return day
    if grain == "week":
        days = when.astype("datetime64[D]")
        # 1970-01-01 was a Thursday, so (days since the epoch + 3) is zero on a Monday.
        weekday = (days.astype(np.int64) + 3) % 7
        return (days - weekday * _DAY).astype("datetime64[s]")
    if grain == "month":
        return when.astype("datetime64[M]").astype("datetime64[s]")
    step = 3 if grain == "season" else 12
    return oracle_anniversary(oracle_offset_months(when, ys) // step * step, ys)


def oracle_bin_partial(clock: np.ndarray, when: np.ndarray, bins: np.ndarray, grain,
                       ys) -> np.ndarray:
    """Which bins the record does not cover for their whole calendar span. The record covers from
    its first reading to its last plus one sampling interval, and a bin is partial when its own
    span reaches outside that. Only a bin at an end of the record can, because _check_grid() has
    already required every unit to hold readings in every bin between them. The record's ends are
    read on the clock the bins are; the sampling interval is the record's own, read off the
    instants."""
    covered_start = clock.min()
    covered_end = clock.max() + oracle_sampling_step(when)
    return (bins < covered_start) | (oracle_bin_next(bins, grain, ys, covered_end) > covered_end)


def oracle_sampling_step(when: np.ndarray) -> np.timedelta64:
    u = np.unique(when)
    return np.diff(u).min() if len(u) > 1 else np.timedelta64(0, "s")


def oracle_bin_next(bins: np.ndarray, grain, ys, covered_end) -> np.ndarray:
    """Where each bin ends, which is where the next one on the same calendar starts. The four
    coarse grains step by the calendar rather than by a count of seconds, so the successor is
    taken by landing well inside the following bin and flooring that, which is exact whatever the
    month length. A caller-supplied binning declares its own bins, so its successors are read off
    the bins themselves and its last bin is taken to end with the record."""
    if callable(grain) or grain == "native":
        return np.concatenate([bins[1:], np.asarray([covered_end], dtype="datetime64[s]")])
    if grain == "halfday":
        return bins + 12 * _HOUR
    if grain in ("day", "week", "month"):
        ahead = {"day": 36 * _HOUR, "week": 180 * _HOUR, "month": 40 * _DAY}[grain]
        return oracle_bin_start(bins + ahead, grain, ys)
    return oracle_anniversary(oracle_offset_months(bins, ys) + (3 if grain == "season" else 12), ys)


def oracle_offset_months(when: np.ndarray, ys) -> np.ndarray:
    months = when.astype("datetime64[M]")
    year = months.astype("datetime64[Y]").astype(int) + 1970
    month = months.astype(int) % 12 + 1
    day = (when.astype("datetime64[D]") - months).astype(np.int64) + 1
    return year * 12 + (month - 1) - (ys[0] - 1) - (day < ys[1]).astype(np.int64)


def oracle_anniversary(offset: np.ndarray, ys) -> np.ndarray:
    absolute = offset + (ys[0] - 1)
    month = (absolute - 1970 * 12).astype("datetime64[M]")
    return (month.astype("datetime64[D]") + (ys[1] - 1) * _DAY).astype("datetime64[s]")


def oracle_day_level(reading, unit_ix, when, bins, n_u, ys):
    """Reduce every (unit, calendar day) to its own mean, minimum and maximum, and say which bin
    cell each of those days falls in. The four day-level statistics are a second reduction over
    those days, which is what keeps an extreme day distinct from an extreme reading."""
    day_start = oracle_bin_start(when, "day", ys)
    days, day_ix = np.unique(day_start, return_inverse=True)
    dcell = day_ix * n_u + unit_ix
    n_dcell = n_u * len(days)

    count = np.bincount(dcell, minlength=n_dcell)
    present = np.flatnonzero(count)
    order = np.argsort(dcell, kind="stable")
    starts = np.searchsorted(dcell[order], np.arange(n_dcell), side="left")
    reading_sorted = reading[order]

    mean = (np.bincount(dcell, weights=reading, minlength=n_dcell)[present]
            / count[present])
    low = np.minimum.reduceat(reading_sorted, starts)[present]
    high = np.maximum.reduceat(reading_sorted, starts)[present]

    day_unit = present % n_u
    bin_of = np.searchsorted(bins, days[present // n_u], side="right") - 1
    return {"cell": bin_of * n_u + day_unit, "mean": mean, "min": low, "max": high}


def oracle_reduce(values, cell, n_cell, how):
    order = np.argsort(cell, kind="stable")
    starts = np.searchsorted(cell[order], np.arange(n_cell), side="left")
    if how == "mean":
        count = np.bincount(cell, minlength=n_cell)
        return np.bincount(cell, weights=values, minlength=n_cell) / count
    ufunc = np.minimum if how == "min" else np.maximum
    return ufunc.reduceat(values[order], starts)


def oracle_bin_extent(when, bin_ix, n_b):
    order = np.argsort(bin_ix, kind="stable")
    starts = np.searchsorted(bin_ix[order], np.arange(n_b), side="left")
    seconds = when.astype("datetime64[s]").astype(np.int64)
    return np.maximum.reduceat(seconds[order], starts).astype("datetime64[s]")


def oracle_duration(x):
    """A count and a unit, or a bare count of seconds. A year is 365 days and a month is 30 days:
    a lookback of a fixed length is a fixed length rather than a calendar step."""
    if not isinstance(x, str):
        return int(x)
    size = {"second": 1, "minute": 60, "hour": 3600, "day": 86400, "week": 604800,
            "month": 2592000, "year": 31536000}
    parts = x.strip().split()
    if len(parts) == 1:
        return int(parts[0])
    return int(parts[0]) * size[parts[1].lower().rstrip("s")]


def oracle_bin_offsets(span, lag, bins):
    step = span // bins
    out = []
    for b in range(bins):
        x = b * step - lag - span
        for name, size in (("day", 86400), ("hour", 3600), ("minute", 60), ("second", 1)):
            if x % size == 0:
                n = x // size
                out.append(f"{n} {name}" + ("" if abs(n) == 1 else "s"))
                break
    return tuple(out)


def oracle_cell_stat(name, v, t):
    """The seven statistics over the readings of one cell. The four day-level ones reduce each
    calendar day first and reduce again over the days of the cell, oldest first, which is what
    keeps an extreme day distinct from an extreme reading."""
    if name == "mean":
        return float(v.mean())
    if name == "min":
        return float(v.min())
    if name == "max":
        return float(v.max())
    parts = np.split(v, np.flatnonzero(np.diff(t // 86400)) + 1)
    if name == "cold_day":
        return float(min(p.mean() for p in parts))
    if name == "warm_day":
        return float(max(p.mean() for p in parts))
    if name == "mean_daily_min":
        return float(np.mean([p.min() for p in parts]))
    return float(np.mean([p.max() for p in parts]))


def oracle_lookback_matrix(data, id, time, value, at, span, lag="0 days", bins=1, stats=("mean",)):
    """The lookback, from the section of ``inst/spec/representation.md`` that describes it.

    It reads whole seconds and knows no zone, so it answers for a series already expressed in the
    calendar to bin by; that is what the tests hand it.
    """
    span = oracle_duration(span)
    lag = oracle_duration(lag)
    step = span // bins
    stats = [stats] if isinstance(stats, str) else list(stats)
    unit = np.asarray([str(v) for v in data[id]])
    when = np.asarray(data[time], dtype="datetime64[s]").astype(np.int64)
    reading = np.asarray(data[value], dtype=np.float64)
    who = np.asarray([str(v) for v in at["id"]])
    anchor = np.asarray(at["at"], dtype="datetime64[s]").astype(np.int64)
    n_t = len(anchor)

    # A day-level statistic is defined only where every calendar day lies whole inside one bin,
    # which for a lookback is a step of whole days and a lookback opening on a day boundary.
    if any(s in DAY_LEVEL_STATS for s in stats):
        if step % 86400:
            raise ValueError("needs bins of a calendar day or coarser")
        off = np.flatnonzero((anchor - lag - span) % 86400)
        if len(off):
            raise ValueError(f"needs bins that open on a day boundary: target {off[0] + 1}")

    out = np.empty((n_t, bins, len(stats)), dtype=np.float64)
    count = np.zeros((n_t, bins), dtype=np.int64)
    for i in range(n_t):
        opens = int(anchor[i]) - lag - span
        inside = (unit == who[i]) & (when >= opens) & (when < opens + span)
        order = np.argsort(when[inside], kind="stable")
        t = when[inside][order]
        v = reading[inside][order]
        b = (t - opens) // step
        for k in range(bins):
            take = b == k
            if not take.any():
                raise ValueError(f"(target, bin) cell holds no readings, first: target {i + 1}")
            count[i, k] = int(take.sum())
            for j, s in enumerate(stats):
                out[i, k, j] = oracle_cell_stat(s, v[take], t[take])

    return {"values": out, "bins": oracle_bin_offsets(span, lag, bins), "bin_n": count}


def oracle_grain_matrix(data, id, time, value, *, grain="day", stats=("mean",),
                         year_start="09-01", tz=None):
    unit = np.asarray([str(v) for v in data[id]])
    when = np.asarray(data[time], dtype="datetime64[s]")
    reading = np.asarray(data[value], dtype=np.float64)
    stats = [stats] if isinstance(stats, str) else list(stats)
    ys = (int(year_start.split("-")[0]), int(year_start.split("-")[1]))

    # The clock the bins are read on: the instant itself at `native`, the series' zone elsewhere.
    clock = when if grain == "native" else oracle_local_clock(when, tz)
    bin_start = oracle_bin_start(clock, grain, ys)
    units, unit_ix = np.unique(unit, return_inverse=True)
    bins, bin_ix = np.unique(bin_start, return_inverse=True)
    n_u, n_b = len(units), len(bins)

    cell = bin_ix * n_u + unit_ix
    count = np.bincount(cell, minlength=n_u * n_b)
    order = np.argsort(cell, kind="stable")
    reading_sorted = reading[order]
    starts = np.searchsorted(cell[order], np.arange(n_u * n_b), side="left")

    day = oracle_day_level(reading, unit_ix, clock, bins, n_u, ys) \
        if any(s in DAY_LEVEL_STATS for s in stats) else None

    out = np.empty((n_u, n_b, len(stats)), dtype=np.float64)
    for k, s in enumerate(stats):
        if s == "mean":
            flat = np.bincount(cell, weights=reading, minlength=n_u * n_b) / count
        elif s == "min":
            flat = np.minimum.reduceat(reading_sorted, starts)
        elif s == "max":
            flat = np.maximum.reduceat(reading_sorted, starts)
        elif s == "cold_day":
            flat = oracle_reduce(day["mean"], day["cell"], n_u * n_b, "min")
        elif s == "warm_day":
            flat = oracle_reduce(day["mean"], day["cell"], n_u * n_b, "max")
        elif s == "mean_daily_min":
            flat = oracle_reduce(day["min"], day["cell"], n_u * n_b, "mean")
        else:
            flat = oracle_reduce(day["max"], day["cell"], n_u * n_b, "mean")
        out[:, :, k] = flat.reshape(n_b, n_u).T

    return {"values": out, "units": units, "bin_start": bins,
            "bin_end": oracle_bin_extent(when, bin_ix, n_b),
            "bin_n": count.reshape(n_b, n_u).T,
            "bin_partial": oracle_bin_partial(clock, when, bins, grain, ys)}


def oracle_year_fraction(bin_start: np.ndarray, bin_end: np.ndarray) -> np.ndarray:
    """Where in the year a bin sits, from the instants a representation carries: the midpoint of
    the record the bin holds, floored to the second it began on, over the calendar year in UTC
    that midpoint falls in. It reads the year by truncating a datetime64, where the core counts
    days from the civil epoch, so the two arrive by different routes."""
    opens = bin_start.astype("datetime64[s]").astype(np.int64)
    closes = bin_end.astype("datetime64[s]").astype(np.int64)
    mid = opens + (closes - opens) // 2
    year = mid.astype("datetime64[s]").astype("datetime64[Y]")
    first = year.astype("datetime64[s]").astype(np.int64)
    length = (year + 1).astype("datetime64[s]").astype(np.int64) - first
    return (mid - first) / length
