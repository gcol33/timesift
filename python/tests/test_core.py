"""The shared core against the NumPy oracle, and the guards the core raises."""

from __future__ import annotations

import numpy as np
import pytest

from oracle import oracle_grain_matrix
from timesift import _core
from timesift.representation import coverage, grain_matrix, lookback_matrix

SCHEMES = [["min", "mean", "max"],
           ["mean_daily_min", "mean", "mean_daily_max"],
           ["cold_day", "mean", "warm_day"]]
DAY_LEVEL = {"cold_day", "warm_day", "mean_daily_min", "mean_daily_max"}


def series(start, days=400, units=("p1", "p2", "p3"), step="h", seed=0):
    when = np.arange(np.datetime64(start, "s"),
                     np.datetime64(start, "s") + np.timedelta64(24 * days, "h"),
                     np.timedelta64(1, step)).astype("datetime64[s]")
    rng = np.random.default_rng(seed)
    return {"id": np.repeat(np.asarray(units), len(when)),
            "t": np.tile(when, len(units)),
            "v": rng.normal(scale=5.0, size=len(when) * len(units))}


@pytest.mark.parametrize("start", ["2019-09-01", "2020-02-17T05:00:00", "2021-06-11T13:00:00"])
def test_core_reproduces_the_numpy_oracle(start):
    data = series(start)
    for grain in ("native", "halfday", "day", "week", "month", "season", "year"):
        for scheme in SCHEMES:
            if grain in ("native", "halfday") and DAY_LEVEL.intersection(scheme):
                continue
            x = grain_matrix(data, "id", "t", "v", grain=grain, stats=scheme)
            o = oracle_grain_matrix(data, "id", "t", "v", grain=grain, stats=scheme)
            np.testing.assert_array_equal(x.values, o["values"])
            np.testing.assert_array_equal(x.bin_start, o["bin_start"])
            np.testing.assert_array_equal(x.bin_end, o["bin_end"])
            np.testing.assert_array_equal(x.bin_n, o["bin_n"])
            np.testing.assert_array_equal(x.bin_partial, o["bin_partial"])


def test_the_seven_statistics_keep_the_two_orderings_in_the_core_and_the_oracle():
    data = series("2020-02-17T05:00:00", seed=8)
    stats = ["min", "mean_daily_min", "cold_day", "mean", "warm_day", "mean_daily_max", "max"]
    chains = [["min", "mean_daily_min", "mean", "mean_daily_max", "max"],
              ["min", "cold_day", "mean", "warm_day", "max"]]
    for grain in ("day", "week", "month", "season", "year"):
        x = grain_matrix(data, "id", "t", "v", grain=grain, stats=stats)
        o = oracle_grain_matrix(data, "id", "t", "v", grain=grain, stats=stats)["values"]
        for values in (x.values, o):
            for chain in chains:
                for a, b in zip(chain, chain[1:]):
                    lower, upper = values[:, :, stats.index(a)], values[:, :, stats.index(b)]
                    assert (lower <= upper + 1e-9).all(), f"{grain}: {a} <= {b}"


@pytest.mark.parametrize("year_start", ["01-01", "03-01", "09-01", "12-28"])
def test_core_reproduces_the_oracle_at_other_anniversaries(year_start):
    data = series("2019-01-01", days=500, units=("a", "b"))
    for grain in ("season", "year"):
        x = grain_matrix(data, "id", "t", "v", grain=grain,
                          stats=["cold_day", "mean", "warm_day"], year_start=year_start)
        o = oracle_grain_matrix(data, "id", "t", "v", grain=grain,
                                 stats=["cold_day", "mean", "warm_day"], year_start=year_start)
        np.testing.assert_array_equal(x.values, o["values"])
        np.testing.assert_array_equal(x.bin_start, o["bin_start"])


def test_core_reproduces_the_oracle_under_a_supplied_calendar():
    data = series("2019-09-01", units=("a", "b"))

    def ten_days(when):
        seconds = when.astype("datetime64[s]").astype(np.int64)
        return (seconds // (10 * 86400) * (10 * 86400)).astype("datetime64[s]")

    x = grain_matrix(data, "id", "t", "v", grain=ten_days,
                      stats=["cold_day", "mean", "warm_day"])
    o = oracle_grain_matrix(data, "id", "t", "v", grain=ten_days,
                             stats=["cold_day", "mean", "warm_day"])
    np.testing.assert_array_equal(x.values, o["values"])
    np.testing.assert_array_equal(x.bin_partial, o["bin_partial"])


def test_the_cores_calendar_agrees_with_the_oracles():
    from oracle import oracle_bin_next, oracle_bin_start

    when = np.arange(np.datetime64("2018-01-01", "s"),
                     np.datetime64("2018-01-01", "s") + np.timedelta64(20000 * 97, "m"),
                     np.timedelta64(97, "m")).astype("datetime64[s]")
    local = when.astype(np.int64)
    for grain in ("native", "halfday", "day", "week", "month", "season", "year"):
        expected = oracle_bin_start(when, grain, (9, 1)).astype(np.int64)
        np.testing.assert_array_equal(_core.bin_starts(local, grain, 9, 1), expected)

    bins = np.unique(oracle_bin_start(when, "month", (9, 1)))
    expected = oracle_bin_next(bins, "month", (9, 1), when.max()).astype(np.int64)
    np.testing.assert_array_equal(_core.bin_nexts(bins.astype(np.int64), "month", 9, 1), expected)


def test_a_gap_the_whole_record_shares_is_an_error():
    data = series("2021-12-01", days=150, units=("a", "b"))
    month = data["t"].astype("datetime64[M]")
    keep = month != np.datetime64("2022-02", "M")
    gap = {k: v[keep] for k, v in data.items()}

    with pytest.raises(ValueError, match="month bins are not contiguous"):
        grain_matrix(gap, "id", "t", "v", grain="month", stats="mean")
    with pytest.raises(ValueError, match="day bins are not contiguous"):
        grain_matrix(gap, "id", "t", "v", grain="day", stats="mean")

    # The record's own ends are not a gap, and at the `native` grain the bin is the reading itself,
    # so nothing there says what a bin between two others would have been.
    grain_matrix(data, "id", "t", "v", grain="month", stats="mean")
    hours = data["t"].astype("datetime64[h]").astype(np.int64) % 3 == 0
    grain_matrix({k: v[hours] for k, v in data.items()}, "id", "t", "v", grain="native",
                  stats="mean")


def test_a_day_level_statistic_needs_bins_of_a_day_or_coarser():
    data = series("2021-12-01", days=60, units=("a", "b"))

    def six_hours(when):
        seconds = when.astype("datetime64[s]").astype(np.int64)
        return (seconds // 21600 * 21600).astype("datetime64[s]")

    with pytest.raises(ValueError, match="need bins of a calendar day or coarser"):
        grain_matrix(data, "id", "t", "v", grain=six_hours, stats=["cold_day", "warm_day"])
    with pytest.raises(ValueError, match="mean_daily_min needs bins of a calendar day"):
        grain_matrix(data, "id", "t", "v", grain=six_hours, stats="mean_daily_min")
    grain_matrix(data, "id", "t", "v", grain=six_hours, stats=["min", "mean", "max"])


def test_a_zone_whose_local_midnight_does_not_exist():
    data = series("2018-11-01T15:00:00", days=8, units=("a", "b"))
    for grain in ("native", "halfday", "day", "week", "month"):
        x = grain_matrix(data, "id", "t", "v", grain=grain, stats="mean",
                          tz="America/Sao_Paulo")
        assert not np.isnan(x.values).any()

    x = grain_matrix(data, "id", "t", "v", grain="day", stats="mean", tz="America/Sao_Paulo")
    opens = np.datetime_as_string(x.bin_start, unit="s")
    assert "2018-11-04T03:00:00" in list(opens)
    assert int(x.bin_n[0][list(opens).index("2018-11-04T03:00:00")]) == 23


def test_a_series_binned_by_a_zones_calendar_matches_the_r_side():
    data = series("2021-12-20", days=40, units=("a", "b"), seed=6)
    utc = grain_matrix(data, "id", "t", "v", grain="day", stats=["min", "mean", "max"])
    vienna = grain_matrix(data, "id", "t", "v", grain="day", stats=["min", "mean", "max"],
                           tz="Europe/Vienna")

    assert utc.shape[1] == 40
    assert vienna.shape[1] == 41
    assert vienna.bins[0] == "2021-12-19T23:00:00Z"

    # The same instants, relabelled into their Vienna clock and binned as if that clock were the
    # calendar, give the same numbers: the zone is the whole of the difference.
    from zoneinfo import ZoneInfo
    from datetime import datetime
    zone = ZoneInfo("Europe/Vienna")
    seconds = data["t"].astype(np.int64)
    shifted = seconds + np.asarray(
        [int(datetime.fromtimestamp(int(t), zone).utcoffset().total_seconds()) for t in seconds])
    naive = grain_matrix({**data, "t": shifted.astype("datetime64[s]")}, "id", "t", "v",
                          grain="day", stats=["min", "mean", "max"])
    np.testing.assert_array_equal(vienna.values, naive.values)
    np.testing.assert_array_equal(vienna.bin_n, naive.bin_n)


def test_an_id_called_nan_is_a_unit_and_a_missing_id_is_an_error():
    when = np.arange(np.datetime64("2021-09-01", "s"),
                     np.datetime64("2021-09-03", "s"),
                     np.timedelta64(1, "h")).astype("datetime64[s]")
    data = {"id": ["nan"] * len(when) + ["a"] * len(when),
            "t": np.tile(when, 2),
            "v": np.arange(2 * len(when), dtype=float)}
    x = grain_matrix(data, "id", "t", "v", grain="day", stats="mean")
    assert x.units == ("a", "nan")

    data["id"] = [None] * len(when) + ["a"] * len(when)
    with pytest.raises(ValueError, match="missing values in the readings"):
        grain_matrix(data, "id", "t", "v", grain="day", stats="mean")


def test_readings_a_fraction_of_a_second_apart_are_the_same_reading_twice():
    data = {"id": ["a", "a", "a"],
            "t": np.asarray(["1970-01-01T00:00:00", "1970-01-01T00:00:00",
                             "1970-01-01T00:00:01"], dtype="datetime64[s]"),
            "v": [1.0, 2.0, 3.0]}
    with pytest.raises(ValueError, match=r"duplicated \(unit, time\) pairs"):
        grain_matrix(data, "id", "t", "v", grain="native", stats="mean")


def test_a_reading_that_is_not_a_finite_number_is_refused_and_named():
    when = np.arange(np.datetime64("2021-09-01", "s"),
                     np.datetime64("2021-09-09", "s"),
                     np.timedelta64(1, "h")).astype("datetime64[s]")
    rng = np.random.default_rng(0)
    base = {"id": np.repeat(np.asarray(["p1", "p2"]), len(when)),
            "t": np.tile(when, 2),
            "v": rng.normal(size=2 * len(when))}
    at = {"id": ["p1", "p2"], "at": [when[-1], when[-1]]}

    for hole in (np.nan, np.inf, -np.inf):
        data = dict(base, v=base["v"].copy())
        data["v"][29] = hole
        for call in (lambda d: grain_matrix(d, "id", "t", "v", grain="day"),
                     lambda d: lookback_matrix(d, "id", "t", "v", at=at, span="2 days")):
            with pytest.raises(ValueError, match="1 reading is not a finite number, "
                                                 "first: unit p1 at 2021-09-02T05:00:00"):
                call(data)

    two = dict(base, v=base["v"].copy())
    two["v"][[29, 30]] = np.inf
    with pytest.raises(ValueError, match="2 readings are not a finite number"):
        grain_matrix(two, "id", "t", "v", grain="day")

    # coverage() reads how many readings a unit has in each bin and never their values, so it is
    # not the guard's business.
    data = dict(base, v=base["v"].copy())
    data["v"][29] = np.nan
    assert int(coverage(data, "id", "t", grain="day").count.sum()) == len(base["v"])
