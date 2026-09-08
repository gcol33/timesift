"""The lookback: the same input reduced the same way gives the same numbers as R does.

A digest mismatch here is a bug in whichever implementation moved, never a fixture to regenerate.
Fixtures are regenerated on the R side, deliberately, in their own commit, and only when
``inst/spec/representation.md`` changed with them.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from timesift.digest import digest_array
from timesift.representation import _bin_offsets, _parse_duration, lookback_matrix

from oracle import oracle_lookback_matrix

FIXTURES = Path(__file__).resolve().parents[2] / "inst" / "spec" / "fixtures"

SERIES_FILE = {"aligned": "series.csv", "offset": "series_offset.csv",
               "zoned": "series_zoned.csv", "order": "series_order.csv"}


def read_series(name):
    with (FIXTURES / SERIES_FILE[name]).open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {"id": [r["id"] for r in rows],
            "time": [r["time"].replace("Z", "") for r in rows],
            "value": [float(r["value"]) for r in rows]}


def read_csv(name):
    with (FIXTURES / name).open(newline="") as fh:
        return list(csv.DictReader(fh))


TARGETS = read_csv("lookback_targets.csv")


def series_of(name):
    return next(r["series"] for r in TARGETS if r["set"] == name)


def anchors_of(name):
    rows = [r for r in TARGETS if r["set"] == name]
    return {"id": [r["id"] for r in rows],
            "at": np.asarray([r["at"].replace("Z", "") for r in rows], dtype="datetime64[s]")}


def record(days=200, seed=20260904):
    rng = np.random.default_rng(seed)
    t = np.arange(np.datetime64("2021-09-01T00:00:00"),
                  np.datetime64("2021-09-01T00:00:00") + np.timedelta64(24 * days, "h"),
                  np.timedelta64(1, "h"))
    return {"id": ["p1"] * len(t) + ["p2"] * len(t),
            "time": np.concatenate([t, t]),
            "value": rng.normal(scale=5.0, size=2 * len(t))}


ANCHORS = {"id": ["p1", "p2", "p1", "p2"],
           "at": np.asarray(["2022-01-01", "2022-02-10", "2022-03-05", "2022-03-15"],
                              dtype="datetime64[s]")}

SCHEMES = ["mean", "min", "max", "cold_day", "warm_day", "mean_daily_min", "mean_daily_max",
           ["min", "mean", "max"], ["cold_day", "mean", "warm_day"],
           ["mean_daily_min", "mean", "mean_daily_max"]]

GRID = [("30 days", "0 days", 1), ("30 days", "0 days", 3), ("30 days", "7 days", 3),
        ("7 days", "0 days", 7), ("60 days", "1 day", 2), ("2 days", "0 days", 2)]


@pytest.fixture(scope="module")
def series():
    return {name: read_series(name) for name in SERIES_FILE}


@pytest.mark.parametrize("span,lag,bins", GRID, ids=lambda v: str(v).replace(" ", "_"))
@pytest.mark.parametrize("stats", SCHEMES, ids=lambda s: s if isinstance(s, str) else "+".join(s))
def test_the_core_reproduces_the_pure_numpy_oracle(span, lag, bins, stats):
    d = record()
    x = lookback_matrix(d, "id", "time", "value", ANCHORS, span, lag=lag, bins=bins, stats=stats)
    o = oracle_lookback_matrix(d, "id", "time", "value", ANCHORS, span, lag=lag, bins=bins,
                             stats=stats)
    # NumPy sums pairwise where the core sums in order, so the two agree on the arithmetic rather
    # than on the last bit. What is asserted byte-exactly is the digest, off the core.
    np.testing.assert_allclose(x.values, o["values"], rtol=0, atol=1e-10)
    assert x.bins == o["bins"]
    assert (x.bin_n == o["bin_n"]).all()


@pytest.mark.parametrize(
    "row", read_csv("lookback_digests.csv"),
    ids=lambda r: (f"{r['set']}-{r['tz'].replace('/', '_')}-{r['span'].replace(' ', '')}"
                   f"-{r['lag'].replace(' ', '')}-{r['bins']}-{r['stat']}"))
def test_digest_matches_the_r_side(series, row):
    at = anchors_of(row["set"])
    x = lookback_matrix(series[series_of(row["set"])], "id", "time", "value", at,
                      row["span"], lag=row["lag"], bins=int(row["bins"]),
                      stats=row["stat"].split("+"), tz=row["tz"])
    # The shape and the bin naming are asserted before the digest, so a lookback cut differently is
    # reported as that rather than as an unexplained hash mismatch.
    assert x.values.shape[0] == int(row["n_target"])
    assert x.values.shape[1] == int(row["n_bin"])
    assert x.bins[0] == row["first_bin"]
    assert x.bins[-1] == row["last_bin"]
    assert at["id"][0] == row["first_unit"]
    assert at["id"][-1] == row["last_unit"]
    assert digest_array(x) == row["digest"]


def test_the_lookback_fixtures_carry_what_the_contract_says_they_carry():
    rows = read_csv("lookback_digests.csv")
    # An anchor that is a local midnight in one zone is not one in another, and an anchor on the
    # hour rules out the four day-level statistics a midnight allows, so the sets are what carry
    # both halves rather than one set per series.
    assert {r["set"] for r in TARGETS} == {"aligned", "offset", "hourly", "zoned"}
    assert {"UTC", "America/Sao_Paulo"} <= {r["tz"] for r in rows}
    assert any(r["set"] == "zoned" and r["tz"] == "America/Sao_Paulo" and "cold_day" in r["stat"]
               for r in rows)

    named = {s for r in rows for s in r["stat"].split("+")}
    assert {"mean", "min", "max", "cold_day", "warm_day",
            "mean_daily_min", "mean_daily_max"} <= named
    assert {"1", "3", "4", "7"} <= {r["bins"] for r in rows}
    assert len({r["lag"] for r in rows}) > 1
    # A span written as a bare count of seconds, and a step that is no whole number of days.
    assert any(r["span"].isdigit() for r in rows)
    assert any(r["span"] == "7 days" and r["bins"] == "3" for r in rows)


@pytest.mark.parametrize("row", read_csv("lookback_guards.csv"),
                         ids=lambda r: f"{r['set']}-{r['stat']}")
def test_both_guards_fire_with_the_message_the_fixtures_pin(series, row):
    with pytest.raises(ValueError) as raised:
        lookback_matrix(series[series_of(row["set"])], "id", "time", "value", anchors_of(row["set"]),
                      row["span"], lag=row["lag"], bins=int(row["bins"]), stats=row["stat"],
                      tz=row["tz"])
    assert row["message"] in str(raised.value)


def test_a_cell_the_record_cannot_fill_names_the_target_and_the_interval():
    d = record(days=30)
    at = {"id": ["p1"], "at": np.asarray(["2021-09-20"], dtype="datetime64[s]")}
    with pytest.raises(ValueError) as raised:
        lookback_matrix(d, "id", "time", "value", at, "30 days", bins=3, stats="mean")
    assert "1 (target, bin) cell hold no readings, first: target 1 over " in str(raised.value)
    assert "[2021-08-21T00:00:00, 2021-08-31T00:00:00)" in str(raised.value)


def test_a_day_level_statistic_needs_every_day_whole_inside_one_bin():
    d = record(days=60)
    at = {"id": ["p1"], "at": np.asarray(["2021-10-20"], dtype="datetime64[s]")}
    with pytest.raises(ValueError, match="cold_day needs bins of a calendar day or coarser"):
        lookback_matrix(d, "id", "time", "value", at, "7 days", bins=3, stats="cold_day")
    with pytest.raises(ValueError, match="cold_day and warm_day need bins of a calendar day"):
        lookback_matrix(d, "id", "time", "value", at, "7 days", bins=3,
                      stats=["cold_day", "warm_day"])
    lookback_matrix(d, "id", "time", "value", at, "7 days", bins=3, stats=["min", "mean", "max"])

    hour = {"id": ["p1"], "at": np.asarray(["2021-10-20T05:00:00"], dtype="datetime64[s]")}
    with pytest.raises(ValueError, match="warm_day needs bins that open on a day boundary"):
        lookback_matrix(d, "id", "time", "value", hour, "7 days", bins=7, stats="warm_day")
    lookback_matrix(d, "id", "time", "value", hour, "7 days", bins=7, stats="mean")


def test_a_duration_is_a_count_and_a_unit_or_a_count_of_seconds():
    assert _parse_duration("30 days", "span") == 2592000
    assert _parse_duration("7 days", "span") == 604800
    assert _parse_duration("12 hours", "span") == 43200
    assert _parse_duration("1 day", "span") == 86400
    assert _parse_duration("1 week", "span") == 604800
    # A year is 365 days and a month is 30 days: a lookback of a fixed length is a fixed length.
    assert _parse_duration("1 year", "span") == 31536000
    assert _parse_duration("1 month", "span") == 2592000
    assert _parse_duration("90", "span") == 90
    assert _parse_duration(90, "span") == 90
    assert _parse_duration("2 HOURS", "span") == 7200

    with pytest.raises(ValueError, match="must be a count and a unit"):
        _parse_duration("a fortnight", "span")
    with pytest.raises(ValueError, match="unknown duration unit"):
        _parse_duration("3 fortnights", "lag")
    with pytest.raises(ValueError, match="whole number of seconds"):
        _parse_duration(1.5, "span")
    with pytest.raises(ValueError, match="must be a count and a unit"):
        _parse_duration("-1 day", "lag")


def test_a_bin_is_named_by_where_it_opens_relative_to_the_anchor():
    assert _bin_offsets(2592000, 0, 3) == ("-30 days", "-20 days", "-10 days")
    assert _bin_offsets(86400, 0, 1) == ("-1 day",)
    assert _bin_offsets(43200, 43200, 3) == ("-1 day", "-20 hours", "-16 hours")
    assert _bin_offsets(604800, 43200, 1) == ("-180 hours",)


def test_a_lookback_states_what_it_was_built_from():
    d = record(days=60)
    at = {"id": ["p1", "p2"],
          "at": np.asarray(["2021-10-20", "2021-10-21"], dtype="datetime64[s]")}
    x = lookback_matrix(d, "id", "time", "value", at, "30 days", lag="12 hours", bins=3,
                      stats=["min", "mean", "max"])

    assert x.grain == "lookback"
    assert x.span == 2592000
    assert x.lag == 43200
    assert x.stats == ("min", "mean", "max")
    assert x.shape == (2, 3, 3)
    assert x.bin_n.shape == (2, 3)
    assert x.units == ("1", "2")
    assert (x.channel("mean") == x.values[:, :, 1]).all()


def test_a_lookback_reads_only_the_targets_own_unit_and_only_its_own_stretch():
    d = record(days=60)
    at = {"id": ["p1", "p2"],
          "at": np.asarray(["2021-10-20", "2021-10-20"], dtype="datetime64[s]")}
    x = lookback_matrix(d, "id", "time", "value", at, "10 days", stats="mean")

    unit = np.asarray(d["id"])
    when = np.asarray(d["time"], dtype="datetime64[s]")
    inside = ((when >= np.datetime64("2021-10-10")) & (when < np.datetime64("2021-10-20")))
    own = [np.asarray(d["value"])[inside & (unit == u)].mean() for u in ("p1", "p2")]
    np.testing.assert_allclose(x.values.ravel(), own, rtol=0, atol=1e-10)
    # The interval is closed at the left and open at the right.
    assert x.bin_n.ravel().tolist() == [240, 240]

    # Two targets on one unit a fortnight apart read two different stretches of one series, which
    # is the whole reason the reduction exists.
    pair = lookback_matrix(d, "id", "time", "value",
                         {"id": ["p1", "p1"],
                          "at": np.asarray(["2021-10-06", "2021-10-20"],
                                             dtype="datetime64[s]")},
                         "10 days", stats="mean")
    assert pair.values[0, 0, 0] != pair.values[1, 0, 0]


def test_a_lookback_refuses_an_input_it_cannot_answer_for():
    d = record(days=60)
    at = {"id": ["p1"], "at": np.asarray(["2021-10-20"], dtype="datetime64[s]")}

    with pytest.raises(ValueError, match="does not divide into 11 bins"):
        lookback_matrix(d, "id", "time", "value", at, "7 days", bins=11, stats="mean")
    with pytest.raises(ValueError, match="must be a positive whole number"):
        lookback_matrix(d, "id", "time", "value", at, "7 days", bins=0, stats="mean")
    with pytest.raises(ValueError, match="unknown statistic"):
        lookback_matrix(d, "id", "time", "value", at, "7 days", stats="warmest")
    with pytest.raises(ValueError, match="name a unit the series does not carry, first: p9"):
        lookback_matrix(d, "id", "time", "value",
                      {"id": ["p9"], "at": at["at"]}, "7 days", stats="mean")
    with pytest.raises(ValueError, match='must give an "id" naming the unit'):
        lookback_matrix(d, "id", "time", "value", {"unit": ["p1"]}, "7 days", stats="mean")


def test_the_anchors_are_read_as_a_clock_in_the_series_own_calendar():
    d = record(days=60)
    # An instant that opens a Vienna day, which is 23:00 the evening before in UTC.
    at = {"id": ["p1"], "at": np.asarray(["2021-10-19T22:00:00"], dtype="datetime64[s]")}
    vienna = lookback_matrix(d, "id", "time", "value", at, "7 days", bins=7, stats="cold_day",
                           tz="Europe/Vienna")

    # The same instants relabelled into their Vienna clock, with the anchor relabelled too, give
    # the same answer: the zone is the whole of the difference and it is resolved once, at the edge.
    relabelled = dict(d, time=np.asarray(d["time"], dtype="datetime64[s]")
                      + np.timedelta64(2, "h"))
    shifted = {"id": ["p1"], "at": at["at"] + np.timedelta64(2, "h")}
    naive = lookback_matrix(relabelled, "id", "time", "value", shifted, "7 days", bins=7,
                          stats="cold_day")
    np.testing.assert_array_equal(vienna.values, naive.values)

    # A lookback is placed relative to its anchor, so an offset the same at both ends of it cancels
    # and the zone shows only where it decides something: where a calendar day begins. The anchor
    # that opens a Vienna day opens no UTC one, and the day-level four are refused there.
    with pytest.raises(ValueError, match="open on a day boundary"):
        lookback_matrix(d, "id", "time", "value", at, "7 days", bins=7, stats="cold_day")


def test_the_target_table_is_read_by_name_and_carries_no_calendar_attribute():
    d = record(days=60)
    at = {"id": ["p1"], "at": np.asarray(["2021-10-20"], dtype="datetime64[s]")}
    x = lookback_matrix(d, "id", "time", "value", at, "10 days", stats="mean")
    # A bin is a position relative to an anchor rather than a span of the calendar, so the three
    # attributes the calendar owns are absent here as they are in R.
    assert x.bin_start is None and x.bin_end is None and x.bin_partial is None

    with pytest.raises(ValueError, match='must give an "id" naming the unit'):
        lookback_matrix(d, "id", "time", "value", {"id": ["p1"], "time": at["at"]},
                      "10 days", stats="mean")


def test_a_float_bins_that_is_a_whole_number_is_that_whole_number():
    from timesift.representation import _check_bins

    assert _check_bins(3.0) == 3
    assert _check_bins(np.float64(2)) == 2
    for bad in (2.5, 0.0, float("nan"), float("inf"), True, "3"):
        with pytest.raises(ValueError, match="positive whole number"):
            _check_bins(bad)
