"""The plots draw what the objects hold, and return it."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from timesift import Ladder, elasticnet, fold_map, grain_ladder, grain_matrix, plot  # noqa: E402
from timesift import select_grain, simulate_records  # noqa: E402
from timesift.plot import hcl_palette  # noqa: E402


@pytest.fixture(scope="module")
def run():
    sim = simulate_records(n=60, mechanism="season", variables=3, days=365, auc=0.9,
                           prevalence=0.3)
    x = grain_matrix(sim.readings, "unit", "time", "reading", grain=["month", "season", "year"])
    return x, sim.y


def test_the_palette_is_r_s():
    # hcl.colors(n, "Dark 3") in R 4.3.
    assert hcl_palette(2) == ["#E16A86", "#00AD9A"]
    assert hcl_palette(3) == ["#E16A86", "#50A315", "#009ADE"]
    assert hcl_palette(4) == ["#E16A86", "#909800", "#00AD9A", "#9183E6"]


def test_a_ladder_is_drawn_at_its_summary_levels(run):
    x, y = run
    lad = grain_ladder(x, y, elasticnet(n_inner=3), folds=fold_map(y, v=3), verbose=False)
    table = plot(lad)
    level = {(r["learner"], r["grain"]): r["score"] for r in lad.summary()}
    assert [r["grain"] for r in table] == ["month", "season", "year"]
    for r in table:
        assert r["score"] == pytest.approx(level[(r["learner"], r["grain"])])
        assert r["se"] > 0


def test_a_selection_marks_the_candidate_each_fold_chose(run):
    x, y = run
    sel = select_grain(x, y, elasticnet(n_inner=3), folds=fold_map(y, v=3), inner=3,
                       verbose=False)
    inner = plot(sel, title="inner scores")
    assert len(inner) == len(sel.inner)
    picked = [r for r in inner if r["selected"]]
    assert len(picked) == len(sel.selected)
    assert {r["at"] for r in inner} == {1, 2, 3}


def test_what_it_cannot_draw_it_names():
    with pytest.raises(TypeError, match="plot\\(\\) draws a grain_ladder\\(\\)"):
        plot({"grain": []})
    empty = Ladder(grain=np.array(["a"]), learner=np.array(["l"]), variable=np.array(["v"]),
                   fold=np.array([1]), score=np.array([np.nan]), scorable=np.array([False]),
                   predictions={}, cells=None, folds=None, metric="tss", scorer=None,
                   response="presence_absence", fits={})
    table = plot(empty)
    assert np.isnan(table[0]["score"])


def test_a_run_is_drawn_with_the_stack_across_it():
    from test_timesift import fitted

    from timesift import ensemble

    fit = fitted(ensemble=ensemble())
    table = plot(fit)
    assert any(r.get("arm") == "ensemble" for r in fit.estimate or [])
    assert {r["representation"] for r in table} == set(fit.candidates["representation"])
    assert all(np.isfinite(r["score"]) for r in table)
