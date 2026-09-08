"""The three registries, which are what the package is extended through in either language."""

from __future__ import annotations

import numpy as np
import pytest

from timesift.ladder import grain_ladder
from timesift.learners import Learner
from timesift.registry import (LEARNERS, METRICS, RESPONSES, get_learner, learners, metrics,
                               register_learner, register_metric, register_response, responses)
from timesift.representation import grain_matrix
from timesift.response import Response, fold_map, scorable_cells


@pytest.fixture(autouse=True)
def registries_restored():
    """What a test registers is registered for that test only, so the order they run in cannot
    decide what any of them sees."""
    kept = [(r, dict(r._entries)) for r in (LEARNERS, METRICS, RESPONSES)]
    yield
    for registry, entries in kept:
        registry._entries.clear()
        registry._entries.update(entries)


def constant_learner():
    """A learner with nothing in it: every unit is predicted the mean of the fitting units."""
    return Learner(name="constant",
                   fit=lambda x, y, **_: dict(level=y.mean(axis=0)),
                   predict=lambda model, x: np.tile(model["level"], (x.values.shape[0], 1)))


def readings(n_unit=12, days=40, seed=5):
    rng = np.random.default_rng(seed)
    t = np.datetime64("2021-09-01T00:00:00", "s") + np.arange(24 * days) * np.timedelta64(1, "h")
    units = [f"p{i:02d}" for i in range(n_unit)]
    warmth = rng.normal(size=n_unit)
    value = np.concatenate([w + rng.normal(0, 1.0, len(t)) for w in warmth])
    d = {"id": [u for u in units for _ in range(len(t))],
         "time": list(t) * n_unit, "value": list(value)}
    y = (warmth[:, None] > np.array([-0.5, 0.0])).astype(float)
    return d, Response(y, tuple(units), ("sp1", "sp2"))


def test_what_ships_is_registered_and_reachable_by_name():
    assert learners() == ["cnn", "elasticnet", "forest", "mlp", "rescnn", "stepwise"]
    assert metrics() == ["kappa", "kappa_youden", "roc_auc", "tss"]
    assert responses() == ["presence_absence"]
    assert get_learner("stepwise").name == "stepwise"


def test_a_registered_learner_arrives_with_what_it_reads_declared():
    assert (get_learner("cnn").reads, get_learner("cnn").multi) == ("sequence", "joint")
    assert (get_learner("forest").reads, get_learner("forest").multi) == ("tabular", "separate")


def test_a_registry_lists_by_c_collation_whatever_the_names_look_like():
    names = ("B_case", "a_case", "A_case", "_leading", "P10", "P9")
    for name in names:
        register_metric(name, lambda y, p: 0.0)
    assert [n for n in metrics() if n in names] == \
        ["A_case", "B_case", "P10", "P9", "_leading", "a_case"]


def test_registering_a_name_twice_needs_saying_so():
    register_metric("only_once", lambda y, p: 0.0)
    with pytest.raises(ValueError, match="already registered"):
        register_metric("only_once", lambda y, p: 1.0)
    register_metric("only_once", lambda y, p: 1.0, overwrite=True)
    assert METRICS.get("only_once")(None, None) == 1.0


def test_an_unknown_name_says_what_is_registered():
    with pytest.raises(KeyError, match="elasticnet"):
        get_learner("nope")
    with pytest.raises(KeyError, match="presence_absence"):
        RESPONSES.get("abundance")


def test_what_a_registration_must_carry():
    with pytest.raises(ValueError, match="constructor"):
        register_learner("not_a_learner", "elasticnet")
    with pytest.raises(ValueError, match=r"\(y, p\)"):
        register_metric("not_a_metric", 0.5)
    with pytest.raises(ValueError, match="activation"):
        register_response("half", dict(prepare=lambda y: y, loss="l2", metric="tss",
                                       cells=lambda y, f: None))
    with pytest.raises(ValueError, match="single non-empty name"):
        register_metric("", lambda y, p: 0.0)


def test_a_metric_of_ones_own_reaches_the_ladder_by_name():
    d, y = readings()
    register_metric("hit_rate", lambda y, p: float(np.mean((p >= 0.5) == (y == 1))))
    register_learner("constant", constant_learner)
    x = grain_matrix(d, "id", "time", "value", grain="week")
    lad = grain_ladder(x, y, ["constant"], folds=fold_map(y, v=3, seed=2), metric="hit_rate",
                        verbose=False)
    assert lad.metric == "hit_rate"
    scored = lad.score[np.isfinite(lad.score)]
    assert len(scored) and np.all((scored >= 0) & (scored <= 1))


def test_a_response_head_of_ones_own_decides_the_metric_and_the_cells():
    d, y = readings()
    register_response("every_cell", dict(
        prepare=lambda r: r,
        activation="sigmoid",
        loss="binary_cross_entropy",
        metric="roc_auc",
        cells=lambda r, folds: scorable_cells(r, folds)))
    register_learner("constant", constant_learner)
    x = grain_matrix(d, "id", "time", "value", grain="week")
    lad = grain_ladder(x, y, ["constant"], folds=fold_map(y, v=3, seed=2),
                        response="every_cell", verbose=False)
    # A ladder that names no metric is scored on the one its response head carries.
    assert lad.metric == "roc_auc"


def test_the_head_that_ships_refuses_a_response_that_is_not_presence_absence():
    d, y = readings()
    x = grain_matrix(d, "id", "time", "value", grain="week")
    counts = Response(y.values * 3, y.units, y.variables)
    with pytest.raises(ValueError, match="presence-absence"):
        grain_ladder(x, counts, ["elasticnet"], folds=fold_map(y, v=3), verbose=False)
    holes = Response(np.where(y.values > 0, np.nan, y.values), y.units, y.variables)
    with pytest.raises(ValueError, match="missing values"):
        grain_ladder(x, holes, ["elasticnet"], folds=fold_map(y, v=3), verbose=False)
