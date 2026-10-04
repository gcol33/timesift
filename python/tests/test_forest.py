"""The forest core, against the forest the spec describes, and the learner above it.

The reference is the forest the R suite's oracle grows from the spec's text, written to the
fixtures by ``make_fixtures.R``; the generator is also checked against a reimplementation here,
written from the spec. The R suite reads the same files.
"""

from __future__ import annotations

import csv
import itertools
import pickle
from pathlib import Path

import numpy as np
import pytest

from counts import fixture_counts
from oracle import oracle_stream
from timesift import Response, fit_learner, forest, grain_matrix
from timesift.learners import _forest_settings, flatten
from timesift.metrics import roc_auc
from timesift._tree import forest_fit, forest_predict, forest_stream

FIXTURES = Path(__file__).resolve().parents[2] / "inst" / "spec" / "fixtures"
HELD = ("unit", "y_gaussian", "y_binomial", "w", "fold")


def read_rows(name):
    with open(FIXTURES / name, newline="") as handle:
        return list(csv.DictReader(handle))


CASES = read_rows("forest_cases.csv")


@pytest.fixture(scope="module")
def forest_input():
    rows = read_rows("penalised_input.csv")
    columns = [c for c in rows[0] if c not in HELD]
    counts = {r["unit"]: int(r["count"]) for r in read_rows("tree_weights.csv")}
    return dict(
        x=np.asfortranarray([[float(r[c]) for c in columns] for r in rows]),
        binomial=np.array([float(r["y_binomial"]) for r in rows]),
        gaussian=np.array([float(r["y_gaussian"]) for r in rows]),
        poisson=fixture_counts(FIXTURES, [r["unit"] for r in rows]),
        count=np.array([counts[r["unit"]] for r in rows], dtype=float))


def case_fit(data, row):
    y = data[row["family"]]
    w = data["count"] if row["weights"] == "counts" else np.ones(len(y))
    return forest_fit(data["x"], y, w, row["family"], int(row["trees"]), int(row["mtry"]),
                      int(row["min_leaf"]), row["balance"] == "1", int(row["seed"]))


def test_the_generator_is_the_specs_output_for_output():
    rows = read_rows("forest_stream.csv")
    for key, group in itertools.groupby(rows, key=lambda r: (r["seed"], r["tree"])):
        ref = [int(r["output"]) for r in group]
        seed, tree = int(key[0]), int(key[1])
        assert forest_stream(seed, tree, len(ref)).tolist() == ref, key
        assert list(itertools.islice(oracle_stream(seed, tree), len(ref))) == ref, key
    assert (forest_stream(7, 12, 50).tolist()
            == list(itertools.islice(oracle_stream(7, 12), 50)))


@pytest.mark.parametrize("row", CASES, ids=lambda r: r["case"])
def test_the_forest_core_grows_the_specs_forest_node_for_node(forest_input, row):
    fit = case_fit(forest_input, row)
    ref = [r for r in read_rows("forest_nodes.csv") if r["case"] == row["case"]]
    starts = [i for i, r in enumerate(ref) if r["node"] == "0"]
    assert fit["offset"].tolist() == starts + [len(ref)]
    for field in ("column", "less_left", "left", "right"):
        assert fit[field].tolist() == [int(r[field]) for r in ref], field
    for field in ("threshold", "value"):
        assert fit[field].tolist() == [float.fromhex(r[field]) for r in ref], field


@pytest.mark.parametrize("row", CASES, ids=lambda r: r["case"])
def test_the_forest_predicts_the_mean_of_its_trees_as_the_specs_does(forest_input, row):
    ref = [float.fromhex(r["value"]) for r in read_rows("forest_predict.csv")
           if r["case"] == row["case"]]
    assert forest_predict(case_fit(forest_input, row), forest_input["x"]).tolist() == ref


def test_a_forest_is_the_same_forest_on_any_number_of_threads():
    rng = np.random.default_rng(84)
    x = rng.normal(size=(60, 5))
    y = (x[:, 0] > 0).astype(float)
    one = forest_fit(x, y, np.ones(60), "binomial", 40, 2, 1, False, 5)
    many = forest_fit(x, y, np.ones(60), "binomial", 40, 2, 1, False, 5, threads=3)
    for field in ("offset", "column", "threshold", "less_left", "left", "right", "value"):
        np.testing.assert_array_equal(many[field], one[field])


def test_the_forest_core_refuses_what_it_cannot_grow_on():
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    with pytest.raises(ValueError, match="0 and 1 alone"):
        forest_fit(x, np.array([0, 1, 0, 2.0]), np.ones(4), "binomial", 2, 1, 1, False, 1)
    with pytest.raises(ValueError, match="between 1 and the 1 columns"):
        forest_fit(x, np.array([0, 1, 0, 1.0]), np.ones(4), "binomial", 2, 2, 1, False, 1)
    with pytest.raises(ValueError, match="binomial response"):
        forest_fit(x, np.array([0, 1, 0, 1.0]), np.ones(4), "gaussian", 2, 1, 1, True, 1)
    with pytest.raises(ValueError, match="one class weighs nothing"):
        forest_fit(x, np.array([0, 1, 0, 1.0]), np.array([1, 0, 1, 0.0]), "binomial", 2, 1, 1,
                   True, 1)


def test_a_preset_fills_the_settings_left_open_as_randomforest_and_biomod2_have_them():
    assert _forest_settings("default", "binomial", 471, None, None, None) == dict(
        trees=500, mtry=21, min_node=1)
    assert _forest_settings("default", "gaussian", 471, None, None, None) == dict(
        trees=500, mtry=157, min_node=5)
    assert _forest_settings("default", "gaussian", 2, None, None, None)["mtry"] == 1
    assert _forest_settings("bigboss", "binomial", 471, None, None, None) == dict(
        trees=500, mtry=2, min_node=5)
    assert _forest_settings("bigboss", "binomial", 1, 100, None, 3) == dict(
        trees=100, mtry=1, min_node=3)


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


def test_a_forest_fits_predicts_survives_a_round_trip_and_reads_the_flattened_columns():
    x, y = planted()
    fit = fit_learner(forest(trees=100), x, y)
    p = fit.predict(x)
    assert p.shape == (len(y.units), 2)
    assert np.all((p >= 0) & (p <= 1))
    assert roc_auc(y.values[:, 0], p[:, 0]) > 0.75
    assert fit.model["n_col"] == flatten(x).shape[1]
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(fit)).predict(x), p)
    balanced = fit_learner(forest(trees=100, balance=True), x, y).predict(x)
    assert roc_auc(y.values[:, 0], balanced[:, 0]) > 0.75


def test_a_forest_fits_the_family_the_response_heads_loss_names_and_a_constant_is_its_mean(
        temporary_response):
    from timesift.response import as_response, scorable_cells
    temporary_response("continuous_test", dict(
        prepare=as_response, activation="identity", loss="squared_error", metric="roc_auc",
        cells=lambda y, folds: scorable_cells(y, folds)))
    x, y = planted(seed=91)
    level = x.values[:, :, 0].mean(axis=1)
    level = 10 + 3 * (level - level.mean()) / level.std()
    continuous = Response(level.reshape(-1, 1), y.units, ("height",))
    fit = fit_learner(forest(trees=100), x, continuous, response="continuous_test")
    assert fit.model["family"] == "gaussian"
    assert np.corrcoef(fit.predict(x)[:, 0], level)[0, 1] > 0.8
    with pytest.raises(ValueError, match="binomial response"):
        fit_learner(forest(trees=10, balance=True), x, continuous, response="continuous_test")

    flat = Response(np.zeros((len(y.units), 1)), y.units, ("absent",))
    assert np.unique(fit_learner(forest(trees=10), x, flat).predict(x)).tolist() == [0.0]


def test_a_forest_without_a_preset_it_knows_is_refused():
    with pytest.raises(ValueError, match="preset"):
        forest(preset="tuned")
