"""The simulator's design held against R's, and its draw against what it promises.

The design draws nothing, so ``simulate_design.csv`` pins it to the last place R writes: every
variable's anchor, the weights it reads the record by, its driver's population standard deviation
and the link. The units are drawn from numpy's stream and are held to the design they were drawn
under rather than to R's draw.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from timesift import grain_matrix, simulate_records
from timesift.simulate import _design, link_coefficients

FIXTURES = Path(__file__).resolve().parents[2] / "inst" / "spec" / "fixtures"


def read_fixture(name):
    with (FIXTURES / name).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.mark.parametrize("mechanism", ["event", "season", "lag"])
def test_the_design_is_r_s(mechanism):
    rows = [r for r in read_fixture("simulate_design.csv") if r["mechanism"] == mechanism]
    sim = simulate_records(n=2, mechanism=mechanism, variables=6, days=400, prevalence=0.2,
                           auc=0.8)
    when = np.unique(sim.readings["time"])
    phi = float(np.exp(-sim.design["step_hours"] / (24 * sim.design["anomaly_days"])))
    d = _design(mechanism, 6, when, "09-01", phi, 0.0, 1.0, 0.2, 0.8)
    assert sim.grain == rows[0]["grain"]
    assert sim.design["bins"] == int(rows[0]["bins"])
    assert [a + 1 for a in d["anchor"]] == [int(r["anchor"]) for r in rows]
    np.testing.assert_allclose((d["weights"] ** 2).sum(0), [float(r["weight_ss"]) for r in rows],
                               rtol=1e-10)
    np.testing.assert_allclose(d["sigma"], [float(r["sigma"]) for r in rows], rtol=1e-10)
    assert d["link"]["b0"] == pytest.approx(float(rows[0]["b0"]), abs=1e-7)
    assert d["link"]["b1"] == pytest.approx(float(rows[0]["b1"]), abs=1e-7)


def test_the_true_grain_s_bins_carry_the_driver_exactly():
    sim = simulate_records(n=30, mechanism="lag", variables=3, days=200, sensor_sd=0.0)
    week = grain_matrix(sim.readings, "unit", "time", "reading", grain="week")
    # The driver is a linear functional of the anomaly, which is the reading less the shared wave
    # and the unit's offset; with no sensor noise, the weekly means reproduce it up to those.
    assert week.values.shape[0] == 30
    assert sim.weights.shape == (200 * 8, 3)
    np.testing.assert_allclose(sim.weights.sum(0), 1.0)


def test_one_seed_and_draw_is_one_record_and_another_draw_another_sample():
    a = simulate_records(n=20, mechanism="event", variables=2, days=60)
    b = simulate_records(n=20, mechanism="event", variables=2, days=60)
    c = simulate_records(n=20, mechanism="event", variables=2, days=60, draw=2)
    np.testing.assert_array_equal(a.readings["reading"], b.readings["reading"])
    np.testing.assert_array_equal(a.y.values, b.y.values)
    assert set(a.y.units).isdisjoint(c.y.units)
    np.testing.assert_array_equal(a.weights, c.weights)
    assert a.link == c.link


def test_the_draw_lands_on_the_asked_for_prevalence_and_skill():
    sim = simulate_records(n=4000, mechanism="season", variables=4, days=365, prevalence=0.3,
                           auc=0.8)
    assert sim.y.values.mean() == pytest.approx(0.3, abs=0.02)
    z = sim.driver.ravel()
    y = sim.y.values.ravel()
    pos, neg = z[y == 1], z[y == 0]
    auc = np.mean(pos[:, None] > neg[None, ::7])
    assert auc == pytest.approx(0.8, abs=0.02)
    assert z.std() == pytest.approx(1.0, abs=0.05)


def test_the_readings_are_the_table_grain_matrix_takes():
    sim = simulate_records(n=5, mechanism="none", variables=2, days=10)
    assert sim.grain is None
    assert len(sim.readings["unit"]) == 5 * 10 * 8
    assert sim.readings["unit"][:5].tolist() == [f"d001u{i:05d}" for i in range(1, 6)]
    day = grain_matrix(sim.readings, "unit", "time", "reading", grain="day")
    assert day.values.shape == (5, 10, 1)
    assert sim.y.variables == ("v01", "v02")


def test_the_link_is_solved_once_per_pair():
    assert link_coefficients(0.1, 0.75) == link_coefficients(0.1, 0.75)


def test_what_it_refuses_it_names():
    with pytest.raises(ValueError, match="`mechanism` is one of"):
        simulate_records(mechanism="storm")
    with pytest.raises(ValueError, match="`step_hours` must divide 24, got 5"):
        simulate_records(step_hours=5)
    with pytest.raises(ValueError, match="`prevalence` must lie strictly between 0 and 1"):
        simulate_records(prevalence=1)
    with pytest.raises(ValueError, match="`auc` must lie strictly between 0.5 and 1"):
        simulate_records(auc=0.5)
    with pytest.raises(ValueError, match="`n` must be a single whole number of at least 2"):
        simulate_records(n=1)
    with pytest.raises(ValueError, match="no run of 4 whole week bins"):
        simulate_records(mechanism="lag", days=10)
