"""The tree core, against the reference the fixtures carry, and the learner above it.

The reference is rpart's, grown on the design the penalised fixtures carry under integer case
weights, where the core's class priors and rpart's are the same numbers. The node table, the
complexity table and its cross-validated error are asserted to rounding, and the R suite reads the
same files.
"""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np
import pytest

from counts import fixture_counts
from timesift import Response, fit_learner, grain_matrix, tree
from timesift.learners import flatten
from timesift.metrics import roc_auc
from timesift._tree import tree_fit, tree_predict, tree_prune, tree_prune_cp

FIXTURES = Path(__file__).resolve().parents[2] / "inst" / "spec" / "fixtures"
HELD = ("unit", "y_gaussian", "y_binomial", "w", "fold")


def read_rows(name):
    with open(FIXTURES / name, newline="") as handle:
        return list(csv.DictReader(handle))


CASES = read_rows("tree_cases.csv")


@pytest.fixture(scope="module")
def tree_input():
    rows = read_rows("penalised_input.csv")
    columns = [c for c in rows[0] if c not in HELD]
    counts = {r["unit"]: int(r["count"]) for r in read_rows("tree_weights.csv")}
    return dict(
        x=np.asfortranarray([[float(r[c]) for c in columns] for r in rows]),
        binomial=np.array([float(r["y_binomial"]) for r in rows]),
        gaussian=np.array([float(r["y_gaussian"]) for r in rows]),
        poisson=fixture_counts(FIXTURES, [r["unit"] for r in rows]),
        fold=np.array([int(r["fold"]) for r in rows], dtype=np.int32),
        count=np.array([counts[r["unit"]] for r in rows], dtype=float))


def case_fit(data, row):
    y = data[row["family"]]
    w = data["count"] if row["weights"] == "counts" else np.ones(len(y))
    return tree_fit(data["x"], y, w, row["family"], int(row["min_split"]), int(row["min_leaf"]),
                    float(row["cp"]), int(row["max_depth"]), data["fold"], 5,
                    float(row["shrink"]))


def reference(name, case):
    return [r for r in read_rows(name) if r["case"] == case]


@pytest.mark.parametrize("row", CASES, ids=lambda r: r["case"])
def test_the_tree_core_grows_rparts_tree_node_for_node(tree_input, row):
    fit = case_fit(tree_input, row)
    ref = reference("tree_nodes.csv", row["case"])
    for field in ("number", "column", "less_left", "n"):
        assert fit[field].tolist() == [int(r[field]) for r in ref], field
    for field in ("threshold", "weight", "risk", "complexity", "value"):
        np.testing.assert_allclose(fit[field], [float(r[field]) for r in ref], rtol=1e-12,
                                   atol=1e-14, err_msg=field)


@pytest.mark.parametrize("row", CASES, ids=lambda r: r["case"])
def test_the_tree_cores_complexity_table_and_its_cross_validated_error_are_rparts(tree_input,
                                                                                   row):
    fit = case_fit(tree_input, row)
    ref = reference("tree_cptable.csv", row["case"])
    assert fit["nsplit"].tolist() == [int(r["nsplit"]) for r in ref]
    for field in ("cp", "rel_error", "xerror", "xstd"):
        np.testing.assert_allclose(fit[field], [float(r[field]) for r in ref], rtol=1e-12,
                                   atol=1e-14, err_msg=field)


@pytest.mark.parametrize("row", CASES, ids=lambda r: r["case"])
def test_pruned_by_biomod2s_rule_the_tree_predicts_what_rparts_pruned_tree_does(tree_input, row):
    fit = case_fit(tree_input, row)
    at = tree_prune_cp(fit, "se_sum")
    pruned = fit if at is None else tree_prune(fit, at)
    ref = reference("tree_predict.csv", row["case"])
    np.testing.assert_allclose(tree_predict(pruned, tree_input["x"]),
                               [float(r["se_sum"]) for r in ref], rtol=1e-12, atol=1e-14)


def test_pruning_collapses_every_split_at_or_below_the_complexity_and_keeps_the_table(
        tree_input):
    row = next(r for r in CASES if r["case"] == "gaussian_counts_bigboss")
    fit = case_fit(tree_input, row)
    at = float(fit["cp"][2])
    pruned = tree_prune(fit, at)
    assert np.all(pruned["complexity"][pruned["column"] >= 0] > at)
    assert int((pruned["column"] >= 0).sum()) == int(fit["nsplit"][2])
    assert pruned["cp"].tolist() == fit["cp"].tolist()
    assert tree_prune(fit, 0.0)["number"].tolist() == fit["number"].tolist()


def test_each_pruning_rule_reads_the_complexity_table_the_way_it_says():
    table = dict(cp=np.array([0.5, 0.2, 0.1, 0.05, 0.01]), nsplit=np.array([0, 1, 2, 4, 7]),
                 xerror=np.array([1, 0.58, 0.5, 0.55, 0.5]),
                 xstd=np.array([0.1, 0.08, 0.09, 0.07, 0.09]))
    assert tree_prune_cp(table, "min") == 0.1
    assert tree_prune_cp(table, "one_se") == 0.2
    # 0.58 + 0.08, 0.5 + 0.09, 0.55 + 0.07, 0.5 + 0.09: the last of the two least.
    assert tree_prune_cp(table, "se_sum") == 0.01
    assert tree_prune_cp(table, "none") is None
    assert tree_prune_cp(dict(cp=np.array([1.0]), nsplit=np.array([0]), xerror=np.array([1.0]),
                              xstd=np.array([0.0])), "se_sum") is None
    assert tree_prune_cp(dict(cp=np.array([1.0]), nsplit=np.array([0]), xerror=np.array([]),
                              xstd=np.array([])), "min") is None


def test_the_tree_core_refuses_what_it_cannot_grow_on():
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    ones = np.ones(4)
    with pytest.raises(Exception, match="0 and 1 alone"):
        tree_fit(x, np.array([0, 1, 0, 2.0]), ones, "binomial", 2, 1, 0.0, 30)
    with pytest.raises(Exception, match="finite values"):
        tree_fit(x, np.array([0, 1, np.nan, 1]), ones, "binomial", 2, 1, 0.0, 30)
    with pytest.raises(Exception, match="zero or more"):
        tree_fit(x, np.array([0, 1, 0, 1.0]), np.array([1, -1, 1, 1.0]), "binomial", 2, 1, 0.0,
                 30)
    with pytest.raises(Exception, match="between 0 and 30"):
        tree_fit(x, np.array([0, 1, 0, 1.0]), ones, "binomial", 2, 1, 0.0, 31)
    fit = tree_fit(x, np.array([0, 0, 1, 1.0]), ones, "binomial", 2, 1, 0.0, 30)
    with pytest.raises(Exception, match="finite values"):
        tree_predict(fit, np.array([[np.nan]]))


def test_a_preset_fills_the_settings_left_open_as_rpart_and_biomod2_have_them():
    p = tree().params
    assert {k: p[k] for k in ("min_split", "min_leaf", "cp", "max_depth", "n_inner")} == dict(
        min_split=20, min_leaf=7, cp=0.01, max_depth=30, n_inner=10)
    assert tree(min_split=30).params["min_leaf"] == 10
    assert tree(min_leaf=4).params["min_split"] == 12
    b = tree(preset="bigboss").params
    assert {k: b[k] for k in ("min_split", "min_leaf", "cp", "max_depth", "n_inner")} == dict(
        min_split=5, min_leaf=5, cp=0.001, max_depth=10, n_inner=5)
    assert tree(preset="bigboss", min_split=12).params["min_leaf"] == 5
    with pytest.raises(ValueError, match="prune"):
        tree(prune="half")
    with pytest.raises(ValueError, match="preset"):
        tree(preset="tuned")


def planted(n_unit=60, days=56, seed=17):
    """A record whose weekly level carries the response, and nothing else does."""
    rng = np.random.default_rng(seed)
    t = np.datetime64("2021-09-01T00:00:00", "s") + np.arange(24 * days) * np.timedelta64(1, "h")
    units = [f"p{i:02d}" for i in range(n_unit)]
    warmth = rng.normal(size=n_unit)
    value = np.concatenate([w * 2.0 + rng.normal(0, 1.0, len(t)) for w in warmth])
    d = {"id": [u for u in units for _ in range(len(t))],
         "time": list(t) * n_unit, "value": list(value)}
    y = rng.binomial(1, 1 / (1 + np.exp(-3 * np.column_stack([warmth, -warmth])))).astype(float)
    x = grain_matrix(d, "id", "time", "value", grain="week")
    return x, Response(y, tuple(units), ("sp1", "sp2"))


def test_a_tree_fits_predicts_survives_a_round_trip_and_reads_the_flattened_columns():
    x, y = planted()
    fit = fit_learner(tree(), x, y)
    p = fit.predict(x)
    assert p.shape == (len(y.units), 2)
    assert np.all((p >= 0) & (p <= 1))
    assert roc_auc(y.values[:, 0], p[:, 0]) > 0.75
    assert fit.model["n_col"] == flatten(x).shape[1]
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(fit)).predict(x), p)


def test_an_unpruned_tree_keeps_at_least_the_leaves_a_pruned_one_does():
    x, y = planted(n_unit=80, seed=41)
    grown = fit_learner(tree(prune="none", cp=0.0), x, y).model["models"][0]
    pruned = fit_learner(tree(cp=0.0), x, y).model["models"][0]
    assert (grown["column"] < 0).sum() >= (pruned["column"] < 0).sum()
    assert grown["xerror"].size == 0


def test_a_tree_fits_the_family_the_response_heads_loss_names_and_a_constant_is_its_mean(
        temporary_response):
    from timesift.response import as_response, scorable_cells
    temporary_response("continuous_test", dict(
        prepare=as_response, activation="identity", loss="squared_error", metric="roc_auc",
        cells=lambda y, folds: scorable_cells(y, folds)))
    x, y = planted(seed=91)
    level = x.values[:, :, 0].mean(axis=1)
    level = 10 + 3 * (level - level.mean()) / level.std()
    fit = fit_learner(tree(min_split=6), x, Response(level.reshape(-1, 1), y.units, ("height",)),
                      response="continuous_test")
    assert fit.model["family"] == "gaussian"
    assert np.corrcoef(fit.predict(x)[:, 0], level)[0, 1] > 0.8

    flat = Response(np.zeros((len(y.units), 1)), y.units, ("absent",))
    assert np.unique(fit_learner(tree(), x, flat).predict(x)).tolist() == [0.0]
