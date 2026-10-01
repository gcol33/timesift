from __future__ import annotations

import numpy as np
import pytest

from timesift import pseudo_absences


def pa_tables():
    n = np.arange(1, 101)
    pool = {"cell": n, "x": np.repeat(np.arange(1, 11), 10), "y": np.tile(np.arange(1, 11), 10),
            "e1": (7 * n) % 17, "e2": (3 * n) % 13}
    take = [11, 44, 45, 77]
    return pool, {k: v[take] for k, v in pool.items()}


def test_the_random_strategy_admits_every_unit_that_is_not_a_presence():
    pool, presences = pa_tables()
    out = pseudo_absences(pool, presences, n=20, id="cell")
    assert len(out["cell"]) == 20 and len(set(out["cell"].tolist())) == 20
    assert not set(out["cell"].tolist()) & set(presences["cell"].tolist())
    assert out["pseudo"].all() and set(out["set"].tolist()) == {1}
    from timesift.pseudo import pa_candidates
    assert pa_candidates(pool, presences, "random", "cell", None, 0.025, None, 0, np.inf,
                         False).sum() == 96
    assert pa_candidates(pool, presences, "random", None, None, 0.025, None, 0, np.inf,
                         False).sum() == 100


def test_a_draw_is_reproducible_and_each_repeat_is_its_own():
    pool, presences = pa_tables()
    a = pseudo_absences(pool, presences, n=10, id="cell", seed=4)
    b = pseudo_absences(pool, presences, n=10, id="cell", seed=4)
    assert a["cell"].tolist() == b["cell"].tolist()
    many = pseudo_absences(pool, presences, n=10, id="cell", repeats=3, seed=4)
    assert len(many["cell"]) == 30 and sorted(set(many["set"].tolist())) == [1, 2, 3]
    assert many["cell"][many["set"] == 1].tolist() == a["cell"].tolist()
    assert many["cell"][many["set"] == 1].tolist() != many["cell"][many["set"] == 2].tolist()


def test_the_disk_strategy_keeps_the_distance_to_the_nearest_presence_in_its_band():
    pool, presences = pa_tables()
    out = pseudo_absences(pool, presences, n=15, strategy="disk", id="cell",
                          coords=["x", "y"], dist_min=2, dist_max=3)
    for x, y in zip(out["x"], out["y"]):
        near = np.sqrt((presences["x"] - x) ** 2 + (presences["y"] - y) ** 2).min()
        assert 2 <= near <= 3
    with pytest.raises(ValueError, match="admits 0"):
        pseudo_absences(pool, presences, n=1, strategy="disk", coords=["x", "y"], dist_min=50,
                        dist_max=60)
    with pytest.raises(ValueError, match="0 <= dist_min"):
        pseudo_absences(pool, presences, n=1, strategy="disk", coords=["x", "y"], dist_min=3,
                        dist_max=2)
    with pytest.raises(ValueError, match="coordinate"):
        pseudo_absences(pool, presences, n=1, strategy="disk")


def test_the_sre_strategy_keeps_the_units_outside_the_envelope_of_the_presences():
    pool, presences = pa_tables()
    out = pseudo_absences(pool, presences, n=10, strategy="sre", id="cell", env=["e1", "e2"],
                          quantile=0.1)
    inside = np.ones(10, dtype=bool)
    for v in ("e1", "e2"):
        lo, hi = np.quantile(presences[v], [0.1, 0.9])
        inside &= (out[v] >= lo) & (out[v] <= hi)
    assert not inside.any()
    with pytest.raises(ValueError, match="env"):
        pseudo_absences(pool, presences, n=1, strategy="sre")
    with pytest.raises(ValueError, match=r"in \[0, 0.5\]"):
        pseudo_absences(pool, presences, n=1, strategy="sre", env=["e1"], quantile=0.9)


def test_a_draw_says_what_it_cannot_do():
    pool, presences = pa_tables()
    with pytest.raises(ValueError, match="admits 96"):
        pseudo_absences(pool, presences, n=99, id="cell")
    with pytest.raises(ValueError, match="1 or more"):
        pseudo_absences(pool, presences, n=0)
    with pytest.raises(ValueError, match="at least one"):
        pseudo_absences(pool, {k: v[:0] for k, v in presences.items()}, n=2)
    with pytest.raises(ValueError, match="both tables"):
        pseudo_absences(pool, presences, n=2, id="nope")
    with pytest.raises(ValueError, match="strategy is one of"):
        pseudo_absences(pool, presences, n=2, strategy="nope")
