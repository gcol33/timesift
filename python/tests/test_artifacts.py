"""The three artifacts that cross the language boundary, and the numbers read off them.

Every assertion here has a twin in tests/testthat/test-artifacts.R reading the same files. The
files were written by the R side; that this side reads them, recomputes what follows from them and
writes back the same bytes is what makes the handover a contract rather than a convention.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from timesift import (Ladder, Response, fold_map, paired_contrast, read_cells, read_folds,
                       read_response, scorable_cells, tss_inflation, write_cells, write_folds,
                       write_response)
from timesift.metrics import decision_threshold, kappa_score, roc_auc, tss

FIXTURES = Path(__file__).resolve().parents[2] / "inst" / "spec" / "fixtures"

METRIC_FNS = {
    "tss": tss,
    "roc_auc": roc_auc,
    "kappa": lambda y, p: kappa_score(y, p, "prevalence"),
    "kappa_youden": lambda y, p: kappa_score(y, p, "youden"),
    "threshold_youden": lambda y, p: decision_threshold(y, p, "youden"),
    "threshold_kappa": lambda y, p: decision_threshold(y, p, "kappa"),
    "threshold_prevalence": lambda y, p: decision_threshold(y, p, "prevalence"),
}


def read_fixture(name):
    with (FIXTURES / name).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def artifacts():
    y = read_response(FIXTURES / "response.csv")
    return y, read_folds(FIXTURES / "folds.csv", y.units)


def test_a_fold_map_a_response_and_a_mask_round_trip_byte_for_byte(tmp_path):
    rng = np.random.default_rng(3)
    y = Response(values=rng.binomial(1, 0.3, (25, 6)).astype(float),
                 units=tuple(f"p{i:02d}" for i in range(1, 26)),
                 variables=tuple(f"sp{j}" for j in range(1, 7)))
    f = fold_map(y, v=5)
    cells = scorable_cells(y, f)

    write_folds(f, tmp_path / "folds.csv")
    write_response(y, tmp_path / "response.csv")
    write_cells(cells, tmp_path / "cells.csv")

    assert read_folds(tmp_path / "folds.csv", y.units).fold.tolist() == f.fold.tolist()
    back = read_response(tmp_path / "response.csv", y.units)
    assert back.variables == y.variables
    assert np.array_equal(back.values, y.values)
    assert list(read_cells(tmp_path / "cells.csv").scorable) == list(cells.scorable)

    # Writing what was read gives the same bytes: a reader that quietly reordered or reformatted
    # would show up here.
    again = tmp_path / "again.csv"
    for name, read, write in (("folds", read_folds, write_folds),
                              ("response", read_response, write_response),
                              ("cells", read_cells, write_cells)):
        original = tmp_path / f"{name}.csv"
        write(read(original), again)
        assert again.read_bytes() == original.read_bytes(), name


def test_the_artifacts_are_written_with_lf_line_endings_and_twelve_significant_digits(tmp_path):
    y = Response(values=np.array([[0.0, 0.5], [1.0, 1 / 3]]), units=("b", "a"),
                 variables=("v2", "v1"))
    path = tmp_path / "response.csv"
    write_response(y, path)
    raw = path.read_bytes()
    assert b"\r" not in raw
    # Rows by the identifier under C collation, columns in the response's own order.
    assert raw.decode().splitlines() == ["id,v2,v1", "a,1,0.333333333333", "b,0,0.5"]


def test_a_unit_the_artifact_has_no_row_for_is_an_error_naming_it(tmp_path):
    rng = np.random.default_rng(4)
    y = Response(values=rng.binomial(1, 0.4, (10, 3)).astype(float),
                 units=tuple(f"p{i:02d}" for i in range(1, 11)),
                 variables=("sp1", "sp2", "sp3"))
    write_folds(fold_map(y, v=3), tmp_path / "folds.csv")
    write_response(y, tmp_path / "response.csv")

    with pytest.raises(ValueError, match="no row in the fold map, first: p99"):
        read_folds(tmp_path / "folds.csv", ("p01", "p99"))
    with pytest.raises(ValueError, match="p98"):
        read_response(tmp_path / "response.csv", ("p01", "p98", "p99"))
    # A unit the file carries beyond those asked for is dropped.
    assert read_folds(tmp_path / "folds.csv", ("p03", "p01")).units == ("p03", "p01")


def test_a_malformed_artifact_is_refused_rather_than_read_as_something_else(tmp_path):
    (tmp_path / "twice.csv").write_bytes(b"id,fold\np01,1\np01,2\n")
    (tmp_path / "word.csv").write_bytes(b"id,fold\np01,first\n")
    (tmp_path / "bare.csv").write_bytes(b"id\np01\n")
    with pytest.raises(ValueError, match="names p01 more than once"):
        read_folds(tmp_path / "twice.csv")
    with pytest.raises(ValueError, match="not a whole number"):
        read_folds(tmp_path / "word.csv")
    with pytest.raises(ValueError, match="has no fold column"):
        read_folds(tmp_path / "bare.csv")
    with pytest.raises(ValueError, match="no variable"):
        read_response(tmp_path / "bare.csv")
    with pytest.raises(FileNotFoundError):
        read_folds(tmp_path / "absent.csv")


def test_the_scorable_mask_the_fixtures_pin_is_the_mask_this_implementation_builds(artifacts):
    y, f = artifacts
    expected = read_cells(FIXTURES / "cells.csv")
    cells = scorable_cells(y, f)

    # Cell by cell rather than in aggregate: a mask that agreed on how many cells are scorable and
    # disagreed on which would score two arms on different units and report neither.
    assert list(cells.variable) == list(expected.variable)
    assert list(cells.fold) == list(expected.fold)
    assert list(cells.scorable) == list(expected.scorable)
    for name in ("n_occ", "pres_train", "abs_train", "pres_test", "abs_test"):
        assert list(getattr(cells, name)) == list(getattr(expected, name)), name

    # The fixture is only a test if it holds a cell of each verdict, and a variable that has no
    # scorable fold at all.
    assert expected.scorable.any() and not expected.scorable.all()
    by_variable = {v: False for v in expected.variable}
    for v, ok in zip(expected.variable, expected.scorable):
        by_variable[v] = by_variable[v] or bool(ok)
    assert not all(by_variable.values())


@pytest.mark.parametrize("row", read_fixture("metrics.csv"),
                         ids=lambda r: f"{r['case']}-{r['metric']}")
def test_every_threshold_metric_matches_the_value_the_fixtures_pin(row):
    cases = [r for r in read_fixture("metric_cases.csv") if r["case"] == row["case"]]
    y = [int(r["y"]) for r in cases]
    p = [float(r["p"]) for r in cases]
    got = METRIC_FNS[row["metric"]](y, p)
    if row["value"] == "NA":
        # A case a metric defines no value on is pinned as that, so a silent number is a failure.
        assert not np.isfinite(got)
    else:
        assert f"{got:.12g}" == row["value"]


def test_the_metric_cases_cover_where_the_tie_rule_is_the_whole_answer():
    cases = {r["case"] for r in read_fixture("metrics.csv")}
    assert {"all_tied", "some_tied", "tied_across_classes", "one_presence", "one_absence",
            "all_presence", "all_absence", "perfect", "reversed"} <= cases


def test_the_paired_contrast_matches_the_value_the_fixtures_pin():
    cells = read_fixture("contrast_cells.csv")
    expected = {r["quantity"]: r["value"] for r in read_fixture("contrast.csv")}
    variable = np.asarray([r["variable"] for r in cells] * 2)
    fold = np.asarray([int(r["fold"]) for r in cells] * 2)
    score = np.asarray([float(r["a"]) if r["a"] != "NA" else np.nan for r in cells]
                       + [float(r["b"]) if r["b"] != "NA" else np.nan for r in cells])
    ladder = Ladder(grain=np.asarray(["week"] * len(variable)),
                    learner=np.asarray(["a"] * len(cells) + ["b"] * len(cells)),
                    variable=variable, fold=fold, score=score, scorable=~np.isnan(score),
                    predictions={}, cells=None, folds=None, metric="tss", scorer=None,
                    response="presence_absence", fits={})
    got = paired_contrast(ladder, "week|a", "week|b")

    for quantity, value in expected.items():
        assert f"{float(got[quantity]):.12g}" == value, quantity
    # A table where both arms scored every cell would pin the pairing at its easiest.
    assert got["n_cell"] < len(cells)


def test_the_inflation_of_a_self_selected_threshold_lands_inside_the_band_the_spec_gives(artifacts):
    # The one thing here that cannot be a digest: it draws replicates, from this language's own
    # random stream. The fixture holds the R side's value and the band the contract allows.
    y, f = artifacts
    expected = read_fixture("inflation.csv")
    skill = [float(r["skill"]) for r in expected]
    tolerance = float(expected[0]["tolerance"])
    out = tss_inflation(y, f, skill=skill, replicates=int(expected[0]["replicates"]), seed=1)

    for got, row in zip(out, expected):
        assert got["skill"] == float(row["skill"])
        assert got["inflation"] > 0
        assert abs(got["inflation"] - float(row["inflation"])) <= tolerance, row["skill"]
