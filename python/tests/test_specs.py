"""The specifications: which columns an argument names, what a representation builds, how it splits.

A specification carries no data, so what is asserted here is that the same one builds the same
array for the targets it was fitted on and for targets it has never seen, and that every array it
builds comes back in the targets' own row order.
"""

from __future__ import annotations

import numpy as np
import pytest

from timesift.learners import flatten
from timesift.representation import grain_matrix
from timesift.response import Response, fold_map
from timesift.select import column_names, select_columns
from timesift.specs import (Representation, Sift, TimesiftSpec, as_resampling, as_sift,
                            auto_grains, build_representation, cv, expand_sift, grain, grains,
                            grouped_cv, lookback, lookbacks, multigrain, n_targets, native,
                            resolve_folds, target_labels)

PLOTS = ("p2", "p1", "p3", "p4")
START = np.datetime64("2021-09-01T00:00:00", "s")


def instants(days: int) -> np.ndarray:
    return START + np.arange(days * 24) * np.timedelta64(3600, "s")


def series(days: int = 45, plots=PLOTS, columns=("temp",)) -> dict:
    t = instants(days)
    out = {"plot": np.repeat(np.asarray(plots), len(t)), "when": np.tile(t, len(plots))}
    for k, name in enumerate(columns):
        out[name] = np.concatenate([np.sin(np.arange(len(t)) / 13.0) + i + 10 * k
                                    for i in range(len(plots))])
    return out


def targets(plots=PLOTS) -> dict:
    n = len(plots)
    return {"plot": list(plots), "sp_a": [i % 2 for i in range(n)],
            "sp_b": [(i // 2) % 2 for i in range(n)],
            "elevation": [1000.0 + 10 * i for i in range(n)],
            "aspect": [90.0 * i for i in range(n)]}


def spec(**given) -> TimesiftSpec:
    settings = dict(y=("sp_a", "sp_b"), x=("temp",), id="plot", time="when")
    settings.update(given)
    return TimesiftSpec(**settings)


# ---- selecting columns ---------------------------------------------------------------------------

def test_a_selection_is_a_name_a_list_a_glob_or_a_predicate():
    columns = ["plot", "sp_a", "sp_b", "elevation"]
    assert select_columns(columns, "sp_a", "`y`") == ["sp_a"]
    assert select_columns(columns, ["sp_b", "sp_a"], "`y`") == ["sp_b", "sp_a"]
    assert select_columns(columns, "sp_*", "`y`") == ["sp_a", "sp_b"]
    assert select_columns(columns, lambda c: c.startswith("sp"), "`y`") == ["sp_a", "sp_b"]


def test_a_list_keeps_the_callers_order_and_a_glob_keeps_the_tables():
    columns = ["b", "a", "c"]
    assert select_columns(columns, ["a", "b"], "`y`") == ["a", "b"]
    assert select_columns(columns, "*", "`y`") == ["b", "a", "c"]


def test_a_glob_matching_nothing_selects_nothing_and_a_name_must_exist():
    columns = ["plot", "sp_a"]
    assert select_columns(columns, "none_*", "`static`") == []
    with pytest.raises(ValueError, match="not a column of the table"):
        select_columns(columns, "sp_z", "`y`")


def test_a_column_already_read_as_something_else_cannot_be_a_predictor():
    with pytest.raises(ValueError, match="cannot be a predictor"):
        select_columns(["elevation"], "plot", "`static`", exclude=("plot",))
    assert select_columns(["elevation"], "*", "`static`", exclude=("plot",)) == ["elevation"]


def test_a_selection_naming_a_column_twice_is_refused():
    with pytest.raises(ValueError, match="names sp_a twice"):
        select_columns(["sp_a", "sp_b"], ["sp_a", "sp_*"], "`y`")


def test_the_column_names_of_a_mapping_are_its_keys():
    assert column_names(targets())[:2] == ["plot", "sp_a"]


# ---- the representation specifications -----------------------------------------------------------

def test_a_representation_says_whether_its_bins_are_a_sequence():
    assert native().sequence and native().grain == "native"
    assert grain("week").sequence and grain("week").label == "week"
    assert not multigrain().sequence
    assert not lookback("30 days").sequence
    assert lookback("30 days", bins=4).sequence


def test_an_unknown_grain_is_named_where_it_is_asked_for():
    with pytest.raises(ValueError, match="unknown grain: fortnight"):
        grain("fortnight")


def test_a_lookback_carries_its_lag_and_its_bins_in_its_label():
    assert lookback("30 days").label == "30 days"
    assert lookback("30 days", lag="7 days").label == "30 days lag 7 days"
    assert lookback("30 days", bins=3).label == "30 days 3 bins"
    with pytest.raises(ValueError, match="positive whole number"):
        lookback("30 days", bins=0)


def test_a_sift_reads_grain_names_representations_or_a_single_one():
    assert list(grains("day", "week")) == ["day", "week"]
    assert list(grains(["day", "week"])) == ["day", "week"]
    assert list(lookbacks("10 days", "20 days")) == ["10 days", "20 days"]
    assert list(as_sift("week")) == ["week"]
    assert list(as_sift(["day", "week"])) == ["day", "week"]
    assert list(as_sift(grain("month"))) == ["month"]
    assert list(as_sift([grain("day"), multigrain()])) == ["day", "multigrain"]
    with pytest.raises(ValueError, match="either grain names or representations"):
        as_sift(["day", grain("week")])


def test_auto_names_the_grains_the_record_gives_more_than_one_bin():
    assert auto_grains(series(), spec()) == ("native", "halfday", "day", "week", "month")
    assert auto_grains(series(days=3), spec()) == ("native", "halfday", "day")


def test_auto_leaves_out_the_grains_a_day_level_statistic_is_undefined_at():
    assert auto_grains(series(), spec(), stats=("cold_day",)) == ("day", "week", "month")


def test_a_set_of_grains_hands_every_member_its_statistics_and_its_year_start():
    named = grains("week", "year", stats=("cold_day", "mean"), year_start="01-01")
    assert [one.year_start for one in named.values()] == ["01-01", "01-01"]
    assert [one.stats for one in named.values()] == [("cold_day", "mean")] * 2
    auto = grains("auto", stats="cold_day", year_start="01-01")["auto"]
    assert (auto.year_start, auto.stats) == ("01-01", ("cold_day",))


def test_auto_is_the_whole_set_and_cannot_be_named_beside_a_grain():
    with pytest.raises(ValueError, match="cannot be named beside a grain"):
        grains("week", "auto")


def test_a_representation_refuses_what_its_grain_cannot_carry_when_it_is_named():
    with pytest.raises(ValueError, match="not defined there"):
        grain("halfday", stats="cold_day")
    with pytest.raises(ValueError, match="unknown statistic"):
        grains("week", stats="warmest")
    with pytest.raises(ValueError, match="month 01-12"):
        grains("week", year_start="13-01")
    with pytest.raises(ValueError, match='must look like "MM-DD"'):
        multigrain(year_start="0101")
    with pytest.raises(ValueError, match="unknown statistic"):
        lookback("10 days", stats="warmest")


def test_expanding_a_sift_replaces_auto_and_leaves_everything_else():
    expanded = expand_sift(grains("auto"), series(days=3), spec())
    assert list(expanded) == ["native", "halfday", "day"]
    assert list(expand_sift(grains("day", "week"), series(), spec())) == ["day", "week"]


# ---- what a representation builds ----------------------------------------------------------------

def test_a_grain_block_comes_back_in_the_targets_own_row_order():
    m = build_representation(grain("day"), series(), targets(), spec())
    assert m.units == PLOTS
    assert m.values.shape == (4, 45, 1)
    # The series sorts its units under C collation; the representation does not inherit that.
    assert m.units != tuple(sorted(PLOTS))
    direct = grain_matrix(series(), "plot", "when", "temp", grain="day")
    assert m.values[0] == pytest.approx(direct.values[direct.units.index("p2")])


def test_a_target_naming_a_unit_the_series_does_not_carry_is_named():
    t = targets()
    t["plot"] = ["p1", "p2", "p3", "p9"]
    with pytest.raises(ValueError, match="first: p9"):
        build_representation(grain("day"), series(), t, spec())


def test_several_value_columns_carry_the_column_name_into_the_channel():
    two = series(columns=("temp", "moisture"))
    m = build_representation(grain("week", stats=("mean", "max")), two, targets(),
                             spec(x=("temp", "moisture")))
    assert m.stats == ("temp_mean", "temp_max", "moisture_mean", "moisture_max")
    one = build_representation(grain("week", stats=("mean", "max")), series(), targets(), spec())
    assert one.stats == ("mean", "max")


def test_a_multigrain_block_is_one_bin_of_every_grains_columns():
    m = build_representation(multigrain(grains=("day", "week")), series(), targets(), spec())
    assert m.values.shape == (4, 1, 45 + 7)
    assert m.stats[0].startswith("day:")
    assert m.stats[-1].startswith("week:")
    # flatten() runs the bin fastest inside a channel, and the names say the same.
    day = build_representation(grain("day"), series(), targets(), spec())
    assert flatten(m)[:, :45] == pytest.approx(flatten(day))


def test_static_columns_are_carried_as_channels_held_constant_over_the_bins():
    m = build_representation(grain("week"), series(), targets(), spec(static=("elevation",)))
    assert m.stats == ("mean", "elevation")
    assert m.values[:, :, 1] == pytest.approx(np.repeat(
        np.asarray(targets()["elevation"])[:, None], m.values.shape[1], axis=1))


def test_a_static_column_is_one_predictor_of_the_flattened_design_not_one_per_bin():
    for rep in (grain("week"), multigrain(grains=("week", "month")), native()):
        m = build_representation(rep, series(), targets(), spec(static=("elevation",)))
        bare = build_representation(rep, series(), targets(), spec())
        design = flatten(m)
        assert design.shape[1] == flatten(bare).shape[1] + 1
        assert design[:, :-1] == pytest.approx(flatten(bare))
        assert design[:, -1] == pytest.approx(np.asarray(targets()["elevation"], dtype=float))


def test_a_static_column_that_is_not_a_number_is_refused_by_name():
    t = targets()
    t["soil"] = ["loam", "sand", "loam", "sand"]
    with pytest.raises(ValueError, match="the static column soil is not numeric"):
        build_representation(grain("week"), series(), t, spec(static=("soil",)))


def test_without_a_series_the_static_columns_are_the_whole_block():
    settings = TimesiftSpec(y=("sp_a", "sp_b"), id="plot", static=("elevation", "aspect"))
    m = build_representation(Representation(label="static", kind="static", sequence=False), None,
                             targets(), settings)
    assert m.values.shape == (4, 1, 2)
    assert m.stats == ("elevation", "aspect")
    assert m.units == PLOTS


def test_a_lookback_reads_one_row_per_target_in_the_tables_own_order():
    anchors = {"plot": ["p1", "p1", "p3"],
               "when": [START + np.timedelta64(20 * 86400, "s"),
                        START + np.timedelta64(40 * 86400, "s"),
                        START + np.timedelta64(30 * 86400, "s")],
               "sp_a": [1, 0, 1], "sp_b": [0, 1, 1]}
    settings = spec(target_time="when")
    m = build_representation(lookback("10 days", bins=2), series(), anchors, settings)
    assert m.values.shape == (3, 2, 1)
    assert m.units == ("1", "2", "3")
    assert target_labels(anchors, settings) == ("1", "2", "3")
    assert n_targets(anchors, settings) == 3


def test_a_lookback_without_an_anchor_names_the_representation_and_target_time():
    with pytest.raises(ValueError, match="`target_time` has to name the column"):
        build_representation(lookback("10 days"), series(), targets(), spec())


def test_the_rows_are_named_by_the_unit_where_a_unit_carries_one_target():
    assert target_labels(targets(), spec()) == PLOTS


# ---- resampling ----------------------------------------------------------------------------------

def test_a_resampling_is_a_spec_or_a_fold_map_somebody_else_built():
    assert as_resampling(None).kind == "cv"
    assert as_resampling(cv(v=4)).v == 4
    assert as_resampling([1, 2, 1, 2]).kind == "given"
    assert grouped_cv("plot", v=3).group == "plot"
    assert native().label == "native" == grain("native").label


def test_a_fold_map_given_directly_is_read_in_the_row_order_of_the_response():
    y = Response(values=np.eye(4), units=PLOTS, variables=("a", "b", "c", "d"))
    folds = resolve_folds([1, 2, 1, 2], y, targets(), spec())
    assert folds.units == PLOTS
    assert folds.fold.tolist() == [1, 2, 1, 2]


def test_grouped_cv_keeps_every_target_of_a_group_in_one_fold():
    units = tuple(str(i + 1) for i in range(12))
    values = np.zeros((12, 2))
    values[::2, 0] = 1.0
    values[::3, 1] = 1.0
    y = Response(values=values, units=units, variables=("a", "b"))
    group = [f"plot{i // 2}" for i in range(12)]
    folds = fold_map(y, v=3, seed=2, group=group)
    for g in set(group):
        assigned = {int(k) for k, one in zip(folds.fold, group) if one == g}
        assert len(assigned) == 1
    assert len(set(folds.fold.tolist())) == 3


def test_grouping_does_not_move_the_map_a_plain_call_draws():
    y = Response(values=np.eye(6), units=tuple("abcdef"), variables=tuple("uvwxyz"))
    # The maps a plain call draws, pinned so that giving `group` a meaning cannot move them.
    assert fold_map(y, v=3, seed=1, strata=1).fold.tolist() == [2, 1, 3, 3, 1, 2]
    assert fold_map(y, v=3, seed=1, strata=1, group=list("aabbcc")).fold.tolist() == \
        [3, 3, 1, 1, 2, 2]
    values = np.random.default_rng(7).integers(0, 2, size=(24, 4)).astype(float)
    stratified = Response(values=values, units=tuple(f"u{i:02d}" for i in range(24)),
                          variables=tuple("abcd"))
    assert fold_map(stratified, v=4, seed=3).fold.tolist() == [
        4, 1, 1, 4, 4, 3, 2, 3, 3, 2, 4, 1, 2, 4, 1, 3, 1, 3, 2, 2, 4, 3, 2, 2]


def test_a_group_needs_one_value_per_unit():
    y = Response(values=np.eye(4), units=PLOTS, variables=tuple("abcd"))
    with pytest.raises(ValueError, match="`group` must have one value per unit"):
        fold_map(y, v=2, group=["a", "b"])
    with pytest.raises(ValueError, match="between 2 and the 2 groups"):
        fold_map(y, v=3, group=["a", "a", "b", "b"])


def test_a_sift_is_a_mapping_of_representations_and_nothing_else():
    assert len(Sift(grain("day"))) == 1
    with pytest.raises(ValueError, match="not a representation"):
        Sift({"day": "day"})
    with pytest.raises(ValueError, match="non-empty"):
        Sift({})
