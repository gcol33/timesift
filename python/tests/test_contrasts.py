"""The grain contrast held against R's table on the same scores.

``grain_contrast_cells.csv`` is a learner's per-cell scores and ``grain_contrast.csv`` what R's
``grain_contrasts()`` read off them, through lme4, lmerTest and emmeans. The differences are the
fixed effects of one restricted fit, which the two sides find to their optimisers' tolerance; the
interval and the p-value are integrals of a multivariate t each side evaluates by quasi-Monte
Carlo, so they agree to the integrator's error and no closer.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

from timesift import Ladder, grain_contrasts  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[2] / "inst" / "spec" / "fixtures"


def read_fixture(name):
    with (FIXTURES / name).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def ladder(rows, learner="cnn"):
    score = np.array([np.nan if r["score"] == "NA" else float(r["score"]) for r in rows])
    return Ladder(grain=np.array([r["grain"] for r in rows]),
                  learner=np.array([learner] * len(rows)),
                  variable=np.array([r["variable"] for r in rows]),
                  fold=np.array([int(r["fold"]) for r in rows]), score=score,
                  scorable=~np.isnan(score), predictions={}, cells=None, folds=None,
                  metric="tss", scorer=None, response="presence_absence", fits={})


@pytest.fixture(scope="module")
def pinned():
    return ladder(read_fixture("grain_contrast_cells.csv")), read_fixture("grain_contrast.csv")


def test_the_contrast_is_r_s_to_the_optimiser_and_the_integrator(pinned):
    lad, expected = pinned
    got = grain_contrasts(lad)
    assert [r["grain"] for r in got] == [r["grain"] for r in expected]
    for g, e in zip(got, expected):
        assert g["learner"] == e["learner"]
        assert g["reference"] == e["reference"]
        assert g["diff"] == pytest.approx(float(e["diff"]), abs=1e-6)
        assert g["lower"] == pytest.approx(float(e["lower"]), abs=2e-4)
        assert g["upper"] == pytest.approx(float(e["upper"]), abs=2e-4)
        assert g["p_value"] == pytest.approx(float(e["p_value"]), abs=2e-3)


def test_one_ladder_returns_one_table(pinned):
    lad, _ = pinned
    assert grain_contrasts(lad) == grain_contrasts(lad)


def test_a_named_reference_is_the_one_every_grain_is_read_against(pinned):
    lad, _ = pinned
    got = grain_contrasts(lad, reference="day")
    assert [r["grain"] for r in got] == ["week", "month"]
    assert {r["reference"] for r in got} == {"day"}
    against_week = {r["grain"]: r["diff"] for r in grain_contrasts(lad)}
    assert got[0]["diff"] == pytest.approx(-against_week["day"], abs=1e-9)


def test_unadjusted_intervals_are_student_s(pinned):
    lad, _ = pinned
    adjusted = grain_contrasts(lad)
    plain = grain_contrasts(lad, adjust="none")
    for a, p in zip(adjusted, plain):
        assert p["diff"] == pytest.approx(a["diff"])
        assert p["upper"] - p["lower"] < a["upper"] - a["lower"]
        assert p["p_value"] <= a["p_value"] + 1e-12


def test_what_it_refuses_it_names(pinned):
    lad, _ = pinned
    with pytest.raises(ValueError, match='"hour" is not a grain of this ladder'):
        grain_contrasts(lad, reference="hour")
    with pytest.raises(ValueError, match='no scored cell for learner "rf"'):
        grain_contrasts(lad, learner="rf")
    with pytest.raises(TypeError, match="expected a grain_ladder"):
        grain_contrasts({"grain": []})
    one = ladder([r for r in read_fixture("grain_contrast_cells.csv") if r["grain"] == "day"])
    with pytest.raises(ValueError, match="at least two grains, this ladder has 1"):
        grain_contrasts(one)
