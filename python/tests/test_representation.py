from __future__ import annotations

import numpy as np
import pytest

from timesift import (GRAINS, TimesiftSet, TimesiftMatrix, bind_channels,
                       calendar_channels, timesift_set, grain_matrix)


def hourly(units=("a", "b"), hours=24 * 400, start="2021-09-01"):
    t = np.datetime64(start, "s") + np.arange(hours) * np.timedelta64(1, "h")
    rng = np.random.default_rng(1)
    return {"id": [u for u in units for _ in range(hours)],
            "time": list(t) * len(units),
            "value": rng.normal(size=hours * len(units))}


def brute(data, key, fun):
    out = {}
    for u, k, v in zip(data["id"], key, data["value"]):
        out.setdefault((u, k), []).append(v)
    return {k: fun(v) for k, v in out.items()}


def test_bins_tile_the_record_with_no_gap_and_no_overlap():
    d = hourly()
    for w in GRAINS:
        x = grain_matrix(d, "id", "time", "value", grain=w)
        assert x.bin_n.sum(axis=1).tolist() == [24 * 400, 24 * 400], w
        assert (x.bin_n > 0).all(), w


def test_the_mean_matches_an_independent_reduction():
    d = hourly()
    key = [str(np.datetime64(t, "D")) for t in d["time"]]
    ref = brute(d, key, np.mean)
    x = grain_matrix(d, "id", "time", "value", grain="day", stats="mean")
    for i, u in enumerate(x.units):
        for j, b in enumerate(x.bins):
            assert x.values[i, j, 0] == pytest.approx(ref[(u, b[:10])])


def test_an_extreme_day_is_not_an_extreme_reading():
    t = np.datetime64("2021-09-06T00:00:00", "s") + np.arange(24 * 7) * np.timedelta64(1, "h")
    temp = np.repeat([10.0, 5, 0, 5, 10, 10, 10], 24)
    temp[24 + 5] = -50
    d = {"id": ["a"] * len(t), "time": list(t), "value": list(temp)}
    x = grain_matrix(d, "id", "time", "value", grain="week",
                      stats=["min", "cold_day", "mean", "warm_day", "max"])
    assert x.channel("min")[0, 0] == -50
    assert x.channel("cold_day")[0, 0] == 0
    assert x.channel("warm_day")[0, 0] == 10
    assert x.channel("max")[0, 0] == 10


def test_the_average_daily_extremes_average_over_days():
    t = np.datetime64("2021-09-06T00:00:00", "s") + np.arange(48) * np.timedelta64(1, "h")
    temp = np.concatenate([np.repeat([-10.0, 10.0], 12), np.zeros(24)])
    d = {"id": ["a"] * len(t), "time": list(t), "value": list(temp)}
    x = grain_matrix(d, "id", "time", "value", grain="week",
                      stats=["min", "mean_daily_min", "mean", "mean_daily_max", "max"])
    assert x.channel("min")[0, 0] == -10
    assert x.channel("mean_daily_min")[0, 0] == -5
    assert x.channel("mean")[0, 0] == 0
    assert x.channel("mean_daily_max")[0, 0] == 5
    assert x.channel("max")[0, 0] == 10


def test_the_mean_of_the_daily_minima_is_not_the_coldest_day():
    t = np.datetime64("2021-09-06T00:00:00", "s") + np.arange(48) * np.timedelta64(1, "h")
    d = {"id": ["a"] * len(t), "time": list(t), "value": list(np.repeat([0.0, 10.0], 24))}
    x = grain_matrix(d, "id", "time", "value", grain="week",
                      stats=["cold_day", "mean_daily_min", "mean_daily_max", "warm_day"])
    assert x.channel("cold_day")[0, 0] == 0
    assert x.channel("mean_daily_min")[0, 0] == 5
    assert x.channel("warm_day")[0, 0] == 10


def test_coarse_bins_follow_the_calendar():
    d = hourly(units=("a",))
    x = grain_matrix(d, "id", "time", "value", grain="month")
    n = x.bin_n[0].tolist()
    assert n[0] == 30 * 24
    assert n[1] == 31 * 24
    assert 28 * 24 in n


def test_seasons_are_three_calendar_months_from_the_year_start():
    d = hourly(units=("a",))
    x = grain_matrix(d, "id", "time", "value", grain="season", year_start="09-01")
    assert [b[:10] for b in x.bins[:3]] == ["2021-09-01", "2021-12-01", "2022-03-01"]


def test_a_bin_the_record_does_not_fill_is_reported_and_can_be_dropped():
    # 1 September 2021 is a Wednesday, so three hydrological years from it fill every month, season
    # and year of the calendar and neither the first nor the last week of it.
    aligned = hourly(units=("a",), hours=26304, start="2021-09-01")
    for w in ("native", "halfday", "day", "month", "season", "year"):
        x = grain_matrix(aligned, "id", "time", "value", grain=w)
        assert not x.bin_partial.any(), w
    week = grain_matrix(aligned, "id", "time", "value", grain="week")
    assert np.flatnonzero(week.bin_partial).tolist() == [0, 156]
    assert grain_matrix(aligned, "id", "time", "value", grain="week",
                         partial="drop").values.shape[1] == 155

    # A logger deployed on no boundary at all carries one at the start of every grain that
    # aggregates, and the bin it starts is the one holding fewer readings than its neighbours.
    offset = hourly(units=("a",), hours=24 * 300, start="2021-10-17T05:00:00")
    for w in ("halfday", "day", "week", "month", "season"):
        p = grain_matrix(offset, "id", "time", "value", grain=w).bin_partial
        assert p[0], w
        assert not p[1:-1].any(), w
    m = grain_matrix(offset, "id", "time", "value", grain="month")
    assert m.bin_n[0, 0] < m.bin_n[0, 1]
    dropped = grain_matrix(offset, "id", "time", "value", grain="month", partial="drop")
    assert dropped.values.shape[1] == m.values.shape[1] - int(m.bin_partial.sum())
    assert not dropped.bin_partial.any()
    assert np.array_equal(dropped.values, m.values[:, ~m.bin_partial, :])


def test_a_caller_supplied_calendar_owns_its_own_bin_lengths():
    d = hourly(units=("a",), hours=24 * 200, start="2021-09-01")
    # Cut at the equinox rather than on the first of a month, with the first edge at the record's
    # own start: the leading bin is three weeks against a season's three months, and that is the
    # calendar the caller asked for rather than a bin the record failed to fill.
    edges = np.array(["2021-09-01", "2021-09-22", "2021-12-21", "2022-03-20"],
                     dtype="datetime64[s]")

    def astronomical(when):
        return edges[np.searchsorted(edges, when, side="right") - 1]

    x = grain_matrix(d, "id", "time", "value", grain=astronomical)
    assert x.values.shape[1] == 3
    assert not x.bin_partial.any()
    assert x.bin_n[0, 0] < x.bin_n[0, 1]

    # The same record on the named grain counts three calendar months from the anniversary, so it
    # is a different rule and a different number of bins, not a different implementation of one.
    named = grain_matrix(d, "id", "time", "value", grain="season")
    assert named.values.shape[1] == 3
    assert [b[:10] for b in named.bins] == ["2021-09-01", "2021-12-01", "2022-03-01"]


def test_the_hydrological_year_boundary_moves_with_year_start():
    t = np.datetime64("2021-08-25T00:00:00", "s") + np.arange(24 * 20) * np.timedelta64(1, "h")
    d = {"id": ["a"] * len(t), "time": list(t), "value": [1.0] * len(t)}
    assert grain_matrix(d, "id", "time", "value", grain="year").values.shape[1] == 2
    assert grain_matrix(d, "id", "time", "value", grain="year",
                         year_start="01-01").values.shape[1] == 1


def test_naming_several_grains_returns_one_representation_per_grain():
    d = hourly(hours=24 * 60)
    s = grain_matrix(d, "id", "time", "value", grain=["day", "week", "month"])
    assert list(s) == ["day", "week", "month"]
    one = grain_matrix(d, "id", "time", "value", grain="week")
    assert np.array_equal(s["week"].values, one.values)


def test_a_calendar_the_package_does_not_carry_can_be_passed_as_a_function():
    d = hourly(units=("a",), hours=24 * 120)
    edges = np.array(["2021-06-23", "2021-09-23", "2021-12-21", "2022-03-20"],
                     dtype="datetime64[s]")

    def astronomical(when):
        return edges[np.searchsorted(edges, when, side="right") - 1]

    x = grain_matrix(d, "id", "time", "value", grain=astronomical)
    assert x.grain == "custom"
    assert [b[:10] for b in x.bins] == ["2021-06-23", "2021-09-23", "2021-12-21"]
    assert x.bin_n.sum() == 24 * 120


def test_the_representation_refuses_input_it_cannot_reduce_honestly():
    d = hourly(units=("a",), hours=48)
    gap = {k: v[24:] for k, v in d.items()}
    gap = {k: list(gap[k]) + list(d[k][:24]) for k in d}
    gap["id"] = ["a"] * 24 + ["b"] * 24
    with pytest.raises(ValueError, match="no readings"):
        grain_matrix(gap, "id", "time", "value", grain="day")

    dup = {k: list(v) + [v[0]] for k, v in d.items()}
    with pytest.raises(ValueError, match="duplicated"):
        grain_matrix(dup, "id", "time", "value", grain="day")

    with pytest.raises(ValueError, match="not defined"):
        grain_matrix(d, "id", "time", "value", grain="native", stats="cold_day")
    with pytest.raises(ValueError, match="twice"):
        grain_matrix(d, "id", "time", "value", grain="day", stats=["mean", "mean"])
    with pytest.raises(ValueError, match="unknown statistic"):
        grain_matrix(d, "id", "time", "value", grain="day", stats="median")
    with pytest.raises(ValueError, match="MM-DD"):
        grain_matrix(d, "id", "time", "value", grain="day", year_start="9-1")


def test_the_calendar_channels_are_the_position_of_a_bin_in_the_year():
    d = hourly()
    x = grain_matrix(d, "id", "time", "value", grain="month")
    cc = calendar_channels(x)
    assert cc.stats == ("year_sin", "year_cos")
    assert np.allclose(cc.values[0], cc.values[1])
    assert np.allclose(cc.values[0, :, 0] ** 2 + cc.values[0, :, 1] ** 2, 1)


def test_channels_are_joined_in_the_order_they_are_given():
    d = hourly(hours=24 * 60)
    x = grain_matrix(d, "id", "time", "value", grain="week", stats=["cold_day", "mean"])
    b = bind_channels(x, calendar_channels(x))
    assert b.stats == ("cold_day", "mean", "year_sin", "year_cos")
    assert np.array_equal(b.channel("mean"), x.channel("mean"))
    with pytest.raises(ValueError, match="same name"):
        bind_channels(x, x)


def test_naming_one_grain_returns_one_representation_however_it_is_named():
    d = hourly(hours=24 * 40)
    one = grain_matrix(d, "id", "time", "value", grain="week")
    as_list = grain_matrix(d, "id", "time", "value", grain=["week"])
    # A set of one is a shape the caller would only have to unwrap, and the R side does not make
    # one either.
    assert isinstance(as_list, TimesiftMatrix)
    assert np.array_equal(as_list.values, one.values)


def test_a_set_reads_as_a_mapping_and_can_be_cut_to_some_of_its_grains():
    d = hourly(hours=24 * 60)
    s = grain_matrix(d, "id", "time", "value", grain=["day", "week", "month"])
    assert isinstance(s, TimesiftSet)
    assert len(s) == 3 and list(s) == ["day", "week", "month"]
    part = s[["week", "month"]]
    assert list(part) == ["week", "month"]
    assert part.units == s.units
    assert np.array_equal(part["week"].values, s["week"].values)


def test_a_set_must_cover_the_same_units_at_every_grain():
    d = hourly(hours=24 * 40)
    whole = grain_matrix(d, "id", "time", "value", grain="week")
    with pytest.raises(ValueError, match="same units"):
        timesift_set({"week": whole, "cut": whole.take_units([0])})
    with pytest.raises(ValueError, match="not a grain_matrix"):
        timesift_set({"week": whole.values})
    with pytest.raises(ValueError, match="non-empty"):
        timesift_set({})


def test_coverage_lays_a_refused_records_gaps_out_as_zeros():
    from timesift import coverage
    d = hourly(hours=24 * 40)
    t = np.asarray(d["time"], dtype="datetime64[s]")
    unit = np.asarray(d["id"])
    lost = (unit == "b") & (t >= np.datetime64("2021-09-06")) & (t < np.datetime64("2021-09-13"))
    kept = {k: [v for v, keep in zip(d[k], ~lost) if keep] for k in d}
    cov = coverage(kept, "id", "time", grain="week")
    assert cov.count.shape == (2, 6)
    assert cov.count[1].tolist() == [120, 0, 168, 168, 168, 168]
    assert cov.count[0].tolist() == [120, 168, 168, 168, 168, 168]
    assert cov.units_with_gaps() == ("b",)
    assert cov.bins_no_unit_reaches() == ()
    with pytest.raises(ValueError, match=r"coverage\(\) lists"):
        grain_matrix(kept, "id", "time", "value", grain="week")

    skipped = (t >= np.datetime64("2021-09-13")) & (t < np.datetime64("2021-09-20"))
    kept = {k: [v for v, keep in zip(d[k], ~skipped) if keep] for k in d}
    cov = coverage(kept, "id", "time", grain="week")
    assert cov.count.shape == (2, 6)
    assert (cov.count.sum(axis=0) == 0).tolist() == [False, False, True, False, False, False]
    assert cov.bins[2] == "2021-09-13T00:00:00Z"
    assert cov.bins_no_unit_reaches() == ("2021-09-13T00:00:00Z",)

    full = coverage(d, "id", "time", grain="week")
    assert not full.empty.any()
    assert (full.count == grain_matrix(d, "id", "time", "value", grain="week").bin_n).all()
    assert coverage(d, "id", "time", grain="native").count.shape == (2, 24 * 40)
    with pytest.raises(ValueError, match="one grain at a time"):
        coverage(d, "id", "time", grain=["day", "week"])


def test_a_numeric_identifier_is_written_by_its_digits():
    t = np.arange(np.datetime64("2021-09-01T00:00:00"),
                  np.datetime64("2021-09-04T00:00:00"), np.timedelta64(1, "h"))
    named = [100000, 9, 10]
    d = {"id": np.repeat(np.asarray(named, dtype=np.float64), len(t)),
         "time": np.tile(t, 3), "value": np.zeros(3 * len(t))}
    # The same three names R writes: not 100000.0, and not 1e+05 either.
    assert grain_matrix(d, "id", "time", "value", grain="day").units == ("10", "100000", "9")
    whole = dict(d, id=np.repeat(np.asarray(named, dtype=np.int64), len(t)))
    assert grain_matrix(whole, "id", "time", "value", grain="day").units == ("10", "100000", "9")

    with pytest.raises(ValueError, match="not a whole number"):
        grain_matrix(dict(d, id=d["id"] + 0.5), "id", "time", "value", grain="day")
    with pytest.raises(ValueError, match="must identify a unit by text"):
        grain_matrix(dict(d, id=np.tile(t, 3)), "id", "time", "value", grain="day")
