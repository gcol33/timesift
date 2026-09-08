"""The cross-language contract: the same input reduced the same way gives the same numbers.

A digest mismatch here is a bug in whichever implementation moved, never a fixture to regenerate.
Fixtures are regenerated on the R side, deliberately, in their own commit, and only when
``inst/spec/representation.md`` changed with them.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from timesift import Response, coverage, digest_array, scorable_cells, grain_matrix

FIXTURES = Path(__file__).resolve().parents[2] / "inst" / "spec" / "fixtures"

SERIES_FILE = {"aligned": "series.csv", "offset": "series_offset.csv",
               "zoned": "series_zoned.csv", "order": "series_order.csv"}


def read_series(name):
    with (FIXTURES / SERIES_FILE[name]).open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {"id": [r["id"] for r in rows],
            "time": [r["time"].replace("Z", "") for r in rows],
            "value": [float(r["value"]) for r in rows]}


def read_digests():
    with (FIXTURES / "digests.csv").open(newline="") as fh:
        return list(csv.DictReader(fh))


def read_edges(name):
    with (FIXTURES / "seasons.csv").open(newline="") as fh:
        rows = [r["edge"].replace("Z", "") for r in csv.DictReader(fh) if r["series"] == name]
    return np.asarray(rows, dtype="datetime64[s]")


def read_coverage():
    with (FIXTURES / "coverage.csv").open(newline="") as fh:
        return list(csv.DictReader(fh))


def read_guards(name):
    with (FIXTURES / name).open(newline="") as fh:
        return list(csv.DictReader(fh))


def calendar(name):
    """The two calendars the guard fixture names, built from the name so that both suites build
    the same function rather than each writing one that happens to break the same rule."""
    if name == "late":
        def late(when):
            t = when.astype("datetime64[s]").astype(np.int64)
            return ((t // 86400 + 1) * 86400).astype("datetime64[s]")
        return late
    if name == "alternate":
        def alternate(when):
            t = when.astype("datetime64[s]").astype(np.int64)
            return (t.min() + 3600 * ((t - t.min()) // 3600 % 2)).astype("datetime64[s]")
        return alternate
    raise KeyError(f"no calendar called {name}")


def binning(name, grain):
    if grain != "astronomical":
        return grain
    edges = read_edges(name)

    def astronomical(when):
        return edges[np.searchsorted(edges, when, side="right") - 1]

    return astronomical


@pytest.fixture(scope="module")
def series():
    return {name: read_series(name) for name in SERIES_FILE}


@pytest.mark.parametrize(
    "row", read_digests(),
    ids=lambda r: (f"{r['series']}-{r['grain']}-{r['tz'].replace('/', '_')}"
                   f"-{r['year_start']}-{r['partial']}-{r['stat']}"))
def test_digest_matches_the_r_side(series, row):
    # The instants are the same bytes on disk whichever calendar reads them; the zone is the clock
    # laid over them, and a zone row asserts that both languages read that clock the same way.
    x = grain_matrix(series[row["series"]], "id", "time", "value",
                      grain=binning(row["series"], row["grain"]),
                      stats=row["stat"].split("+"), year_start=row["year_start"],
                      partial=row["partial"], tz=row["tz"])
    # The shape is asserted before the digest, so a binning that puts the record into a different
    # number of bins is reported as that rather than as an unexplained hash mismatch.
    assert x.values.shape[0] == int(row["n_unit"])
    assert x.values.shape[1] == int(row["n_bin"])
    assert x.bins[0] == row["first_bin"]
    assert x.bins[-1] == row["last_bin"]
    assert int(x.bin_partial.sum()) == int(row["n_partial"])
    # The row order is asserted by name before the digest, so an implementation that orders the
    # ids differently is reported as that rather than as an unexplained hash mismatch.
    assert x.units[0] == row["first_unit"]
    assert x.units[-1] == row["last_unit"]
    assert digest_array(x) == row["digest"]


def zoned_rows():
    """The digest rows read under a clock that is not UTC, which are the ones a column carrying a
    zone of its own has to reproduce."""
    return [r for r in read_digests() if r["tz"] != "UTC" and r["stat"] == "mean"]


def aware_column(times, zone):
    """The fixture's instants written as a zone-aware column, which is the shape a record arrives
    in when it was read from a source that kept the clock it was recorded on."""
    from zoneinfo import ZoneInfo
    here = ZoneInfo(zone)
    return [datetime.fromisoformat(t).replace(tzinfo=timezone.utc).astimezone(here)
            for t in times]


@pytest.mark.parametrize("row", zoned_rows(),
                          ids=lambda r: f"{r['series']}-{r['grain']}-{r['tz'].replace('/', '_')}")
def test_a_column_carrying_a_zone_names_the_calendar_and_reaches_the_same_digest(series, row):
    # Reading the column as instants drops the clock it was written on, so a column in a zone was
    # binned as UTC days however plainly it said otherwise. It names the calendar the `tz` argument
    # names, and lands on the digest the R side wrote for that zone.
    record = dict(series[row["series"]])
    record["time"] = aware_column(record["time"], row["tz"])
    x = grain_matrix(record, "id", "time", "value",
                      grain=binning(row["series"], row["grain"]), stats=row["stat"].split("+"),
                      year_start=row["year_start"], partial=row["partial"])
    assert x.values.shape[1] == int(row["n_bin"])
    assert digest_array(x) == row["digest"]


def test_a_zone_on_the_column_and_a_different_one_beside_it_is_refused():
    row = zoned_rows()[0]
    record = dict(read_series(row["series"]))
    record["time"] = aware_column(record["time"], row["tz"])
    with pytest.raises(ValueError, match="binned by one calendar"):
        grain_matrix(record, "id", "time", "value", grain="day", tz="America/Sao_Paulo")
    # Naming the zone the column already carries is the same thing said twice, not a disagreement.
    assert grain_matrix(record, "id", "time", "value", grain=binning(row["series"], row["grain"]),
                        stats=row["stat"].split("+"), year_start=row["year_start"],
                        partial=row["partial"], tz=row["tz"]) is not None


def test_a_pandas_column_in_a_zone_bins_by_that_zone():
    pd = pytest.importorskip("pandas")
    row = zoned_rows()[0]
    record = dict(read_series(row["series"]))
    record["time"] = pd.Series(aware_column(record["time"], row["tz"]))
    x = grain_matrix(record, "id", "time", "value",
                      grain=binning(row["series"], row["grain"]), stats=row["stat"].split("+"),
                      year_start=row["year_start"], partial=row["partial"])
    assert digest_array(x) == row["digest"]


def test_the_fixtures_cover_a_record_that_starts_on_no_bin_boundary():
    # A record beginning at midnight on the year_start anniversary puts every grain in phase with
    # it, which is the one input on which a rule that keeps a partial leading bin and a rule that
    # never makes one agree. The contract is only a contract if it also carries the other case.
    rows = read_digests()
    assert {r["series"] for r in rows} == {"aligned", "offset", "zoned", "order"}
    offset = [r for r in rows if r["series"] == "offset"]
    assert {r["grain"] for r in offset} == {"native", "halfday", "day", "week", "month", "season",
                                             "year", "astronomical"}
    assert sum(int(r["n_partial"]) for r in offset) > 0
    assert {r["partial"] for r in rows} == {"keep", "drop"}
    assert len({r["year_start"] for r in rows}) > 1

    # A contract checked only in UTC verifies the calendar on the one zone where the question does
    # not arise. Both a zone that moves its clock in the middle of the day and one that moves it at
    # midnight are pinned.
    assert {"UTC", "Europe/Vienna", "America/Sao_Paulo"} <= {r["tz"] for r in rows}
    assert any(r["tz"] == "America/Sao_Paulo" and r["year_start"] == "11-04" for r in rows)


def test_dropping_every_bin_is_an_error_rather_than_an_empty_representation(series):
    with pytest.raises(ValueError, match="no whole year"):
        grain_matrix(series["offset"], "id", "time", "value", grain="year", partial="drop")
    with pytest.raises(ValueError, match="must be"):
        grain_matrix(series["offset"], "id", "time", "value", grain="day", partial="sometimes")


def test_the_digest_is_the_lf_terminated_twelve_place_form_and_nothing_else():
    import hashlib
    values = np.array([1.0, -0.5]).reshape(2, 1, 1)
    expected = hashlib.md5(b"1.000000000000\n-0.500000000000\n").hexdigest()
    assert digest_array(values) == expected


def test_a_digest_is_refused_over_an_array_that_is_not_finite():
    for bad in (np.inf, -np.inf, np.nan):
        with pytest.raises(ValueError, match="not finite"):
            digest_array(np.array([1.0, bad]).reshape(2, 1, 1))
    with pytest.raises(ValueError, match="1 of 2"):
        digest_array(np.array([1.0, np.nan]).reshape(2, 1, 1))


def test_the_traversal_is_unit_fastest_then_bin_then_channel():
    values = np.arange(2 * 3 * 2, dtype=float).reshape(2, 3, 2)
    flat = values.flatten(order="F")
    assert list(flat[:2]) == [values[0, 0, 0], values[1, 0, 0]]
    assert flat[2] == values[0, 1, 0]


def test_the_row_order_is_c_collation_and_not_the_locales(series):
    # These five ids are the case that made the two languages disagree: an English locale orders
    # them _x a1 A1 P10 P9, C collation orders them A1 P10 P9 _x a1, and the contract names the
    # second. NumPy already sorts this way; the fixture is what keeps it that way in both.
    record = series["order"]
    x = grain_matrix(record, "id", "time", "value", grain="day")
    assert x.units == ("A1", "P10", "P9", "_x", "a1")

    # The input row order carries no meaning: the ids arrive in a third order again.
    seen = list(dict.fromkeys(record["id"]))
    assert seen == ["a1", "P9", "_x", "A1", "P10"]
    order = np.argsort(np.asarray(record["value"]), kind="stable")
    shuffled = {k: [record[k][i] for i in order] for k in ("id", "time", "value")}
    assert (digest_array(grain_matrix(shuffled, "id", "time", "value", grain="day"))
            == digest_array(x))


def coverage_input(row):
    """The readings a coverage case takes out, named in the fixture so that both suites build the
    same record rather than each writing one that happens to have a hole in it."""
    record = read_series(row["series"])
    if not row["unit"]:
        return record
    when = np.asarray(record["time"], dtype="datetime64[s]")
    lost = ((when >= np.datetime64(row["from"].replace("Z", ""), "s"))
            & (when < np.datetime64(row["to"].replace("Z", ""), "s")))
    if row["unit"] != "all":
        lost &= np.asarray(record["id"]) == row["unit"]
    keep = ~lost
    return {k: [v for v, take in zip(record[k], keep) if take]
            for k in ("id", "time", "value")}


@pytest.mark.parametrize("row", read_coverage(), ids=lambda r: r["case"])
def test_coverage_counts_the_readings_the_fixtures_pin(row):
    got = coverage(coverage_input(row), "id", "time",
                   grain=binning(row["series"], row["grain"]))
    assert got.count.shape == (int(row["n_unit"]), int(row["n_bin"]))
    assert got.bins[0] == row["first_bin"]
    assert got.bins[-1] == row["last_bin"]
    assert int(got.empty.sum()) == int(row["n_empty"])
    assert len(got.units_with_gaps()) == int(row["n_unit_gap"])
    assert len(got.bins_no_unit_reaches()) == int(row["n_bin_skipped"])
    assert digest_array(got.count.astype(float)) == row["digest"]


def test_the_coverage_fixtures_carry_a_gap_and_a_record_without_one():
    rows = read_coverage()
    assert any(int(r["n_bin_skipped"]) > 0 for r in rows)
    assert any(int(r["n_empty"]) == 0 for r in rows)
    assert any(r["grain"] == "astronomical" for r in rows)


def test_the_reduction_reads_the_same_record_however_its_rows_are_ordered(series):
    # Addition is not associative, so a reduction that accumulated in the order the caller wrote
    # its rows in would move in its last bits under this. The digest is a statement about the
    # representation, so the two are the same bytes and not merely close.
    record = series["aligned"]
    order = np.random.default_rng(11).permutation(len(record["id"]))
    shuffled = {k: [record[k][i] for i in order] for k in ("id", "time", "value")}
    for grain in ("native", "halfday", "day", "week", "month", "season", "year"):
        schemes = [["min", "mean", "max"]]
        if grain not in ("native", "halfday"):
            schemes += [["cold_day", "mean", "warm_day"],
                        ["mean_daily_min", "mean", "mean_daily_max"]]
        for stats in schemes:
            plain = grain_matrix(record, "id", "time", "value", grain=grain, stats=stats)
            mixed = grain_matrix(shuffled, "id", "time", "value", grain=grain, stats=stats)
            assert digest_array(mixed) == digest_array(plain), f"{grain} {'+'.join(stats)}"

def test_the_scorable_mask_orders_its_variables_by_c_collation_too():
    y = Response(values=np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0]]),
                 units=("u1", "u2", "u3", "u4"), variables=("a1", "P9", "_x"))
    cells = scorable_cells(y, {"u1": 1, "u2": 1, "u3": 2, "u4": 2})
    assert list(dict.fromkeys(cells.variable)) == ["P9", "_x", "a1"]


@pytest.mark.parametrize("row", read_guards("grain_guards.csv"),
                         ids=lambda r: f"{r['series']}-{r['calendar']}")
def test_a_supplied_calendar_that_breaks_its_guarantees_is_refused(series, row):
    with pytest.raises(ValueError) as raised:
        grain_matrix(series[row["series"]], "id", "time", "value",
                     grain=calendar(row["calendar"]), stats="mean")
    assert row["message"] in str(raised.value)
