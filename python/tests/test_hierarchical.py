"""The hierarchical model on the core, against tulpa's own fits and against a dense implementation of
the same Laplace marginal, and the learner above it.

The tulpa reference is written by ``inst/spec/make_fixtures.R`` at the levels where its answer is
deterministic: the posterior mode, the empirical-Bayes fit, and each node of the nested grid. The R
suite reads the same files.
"""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np
import pytest

from timesift import Response, fold_map, grains, hierarchical, timesift
from timesift._hierarchical import COVARIANCES, hierarchical_fit, hierarchical_predict

FIXTURES = Path(__file__).resolve().parents[2] / "inst" / "spec" / "fixtures"


def read_rows(name):
    with open(FIXTURES / name, newline="") as handle:
        return list(csv.DictReader(handle))


def floats(row, keys):
    return np.array([float(row[k]) for k in keys])


@pytest.fixture(scope="module")
def fx():
    d = read_rows("hierarchical_input.csv")
    column = lambda name: np.array([float(r[name]) for r in d])  # noqa: E731
    units = sorted({r["unit"] for r in d})
    return dict(y=column("y"), weight=column("weight"),
                X=np.column_stack([np.ones(len(d)), column("x1"), column("x2")]),
                coords=np.column_stack([column("lon"), column("lat")]), units=units,
                index=np.array([units.index(r["unit"]) for r in d], dtype=np.int32),
                cases=read_rows("hierarchical_cases.csv"), ranef=read_rows("hierarchical_ranef.csv"),
                hsgp=read_rows("hierarchical_hsgp_nodes.csv"),
                nngp=read_rows("hierarchical_nngp_nodes.csv"))


# ---------------------------------------------------------------------------------------------
# The same marginal written out densely.

def kernel(cov, d, sigma2, rng):
    if cov == "exponential":
        return sigma2 * np.exp(-d / rng)
    if cov == "matern32":
        return sigma2 * (1 + np.sqrt(3) * d / rng) * np.exp(-np.sqrt(3) * d / rng)
    if cov == "matern52":
        return sigma2 * (1 + np.sqrt(5) * d / rng + 5 * d ** 2 / (3 * rng ** 2)) * np.exp(-np.sqrt(5) * d / rng)
    return sigma2 * np.exp(-(d / rng) ** 2)


def standardise(coords):
    centre = coords.mean(axis=0)
    scale = np.sqrt(coords.var(axis=0, ddof=1).mean())
    return (coords - centre) / scale, centre, scale


def locations(xy):
    loc = np.unique(xy, axis=0)
    loc = loc[np.lexsort((loc[:, 1], loc[:, 0]))]
    place = {tuple(p): i for i, p in enumerate(loc)}
    return loc, np.array([place[tuple(p)] for p in xy])


def vecchia(loc, sigma, rng, cov, nn):
    n = len(loc)
    a = np.zeros((n, n))
    dvar = np.zeros(n)
    dvar[0] = sigma ** 2
    for i in range(1, n):
        earlier = loc[:i]
        d = np.sqrt(((earlier - loc[i]) ** 2).sum(axis=1))
        nb = np.argsort(d, kind="stable")[:min(nn, i)]
        pair = np.sqrt(((earlier[nb][:, None] - earlier[nb][None]) ** 2).sum(axis=2))
        c = kernel(cov, pair, sigma ** 2, rng) + 1e-8 * np.eye(len(nb))
        cv = kernel(cov, d[nb], sigma ** 2, rng)
        coef = np.linalg.solve(c, cv)
        a[i, nb] = coef
        dvar[i] = max(sigma ** 2 - cv @ coef, 1e-10)
    ia = np.eye(n) - a
    return ia.T @ (ia / dvar[:, None]), -np.log(dvar).sum()


def laplace(x, y, w, field_design, field_precision, field_log_det, unit, sigma_u, beta_sd):
    n, p = x.shape
    m = field_design.shape[1]
    g = 0 if unit is None else int(unit.max()) + 1
    u = np.zeros((n, g))
    if g:
        u[np.arange(n), unit] = 1
    z = np.hstack([x, field_design, u])
    tau_u = 1 / sigma_u ** 2 if g else 1.0
    q = np.zeros((p + m + g, p + m + g))
    q[:p, :p] = np.eye(p) / beta_sd ** 2
    if m:
        q[p:p + m, p:p + m] = field_precision
    if g:
        q[p + m:, p + m:] = np.eye(g) * tau_u

    def value(th):
        eta = z @ th
        return (w * (y * eta - np.logaddexp(0, eta))).sum() - 0.5 * th @ q @ th

    th = np.zeros(p + m + g)
    for _ in range(200):
        pr = 1 / (1 + np.exp(-(z @ th)))
        grad = z.T @ (w * (y - pr)) - q @ th
        h = z.T @ ((w * pr * (1 - pr))[:, None] * z) + q
        step = np.linalg.solve(h, grad)
        t = 1.0
        while value(th + t * step) < value(th) - 1e-12 and t > 1e-8:
            t /= 2
        th = th + t * step
        if np.abs(t * step).max() < 1e-13:
            break
    pr = 1 / (1 + np.exp(-(z @ th)))
    h = z.T @ ((w * pr * (1 - pr))[:, None] * z) + q
    log_prior_det = p * np.log(1 / beta_sd ** 2) + g * np.log(tau_u) + field_log_det
    return th, value(th) + 0.5 * log_prior_det - 0.5 * np.linalg.slogdet(h)[1]


def hsgp_design(xy, sigma, rng, m=6, boundary=1.5):
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    centre = (lo + hi) / 2
    half = np.maximum(boundary * (hi - lo) / 2, 0.1)
    out = np.zeros((len(xy), m * m))
    lam = np.zeros(m * m)
    for j1 in range(1, m + 1):
        for j2 in range(1, m + 1):
            col = (j1 - 1) * m + (j2 - 1)
            out[:, col] = (np.sin(np.pi * j1 * (xy[:, 0] - centre[0] + half[0]) / (2 * half[0])) / np.sqrt(half[0])
                           * np.sin(np.pi * j2 * (xy[:, 1] - centre[1] + half[1]) / (2 * half[1])) / np.sqrt(half[1]))
            lam[col] = (np.pi * j1 / (2 * half[0])) ** 2 + (np.pi * j2 / (2 * half[1])) ** 2
    s = np.sqrt(sigma ** 2 * 2 * np.pi * rng ** 2 * np.exp(-0.5 * rng ** 2 * lam))
    return out * s


# ---------------------------------------------------------------------------------------------

def test_the_learner_declares_its_settings():
    learner = hierarchical(spatial="hsgp", m=8)
    assert learner.params["spatial"] == "hsgp" and learner.params["m"] == 8
    with pytest.raises(ValueError, match="spatial"):
        hierarchical(spatial="gp")
    with pytest.raises(ValueError, match="`random`"):
        hierarchical(random=None)
    with pytest.raises(ValueError, match="`neighbours`"):
        hierarchical(neighbours=0)
    with pytest.raises(ValueError, match="`m`"):
        hierarchical(m=2)
    with pytest.raises(ValueError, match="`boundary`"):
        hierarchical(boundary=0.5)
    with pytest.raises(ValueError, match="`cov`"):
        hierarchical(cov="spherical")


def test_the_posterior_mode_of_the_coefficients_is_tulpas_weighted_and_not(fx):
    for row in fx["cases"]:
        if row["case"] not in ("map", "map_weighted"):
            continue
        w = fx["weight"] if row["weighted"] == "TRUE" else np.ones(len(fx["y"]))
        fit = hierarchical_fit(fx["X"], fx["y"], w)
        tol = float(row["tolerance"])
        np.testing.assert_allclose(fit["beta"], floats(row, ("b0", "b1", "b2")), rtol=tol, atol=tol)
        assert fit["log_marginal"] == pytest.approx(float(row["log_marginal"]), rel=tol)


def test_the_empirical_bayes_fit_with_an_intercept_for_each_unit_is_tulpas(fx):
    row = next(r for r in fx["cases"] if r["case"] == "eb")
    fit = hierarchical_fit(fx["X"], fx["y"], np.ones(len(fx["y"])), unit=fx["index"],
                           n_unit=len(fx["units"]))
    assert np.exp(fit["theta_hat"][0]) == pytest.approx(float(row["sd_unit"]), rel=1e-5)
    np.testing.assert_allclose(fit["beta"], floats(row, ("b0", "b1", "b2")), atol=1e-6)
    assert fit["log_marginal"] == pytest.approx(float(row["log_marginal"]), rel=1e-6)
    effect = {r["unit"]: float(r["effect"]) for r in fx["ranef"]}
    np.testing.assert_allclose(fit["unit_effect"], [effect[u] for u in fx["units"]], atol=1e-5)


@pytest.mark.parametrize("field", ["hsgp", "nngp"])
def test_each_node_of_a_fields_grid_is_tulpas_conditional_fit_and_the_weights_follow(fx, field):
    xy, _, _ = standardise(fx["coords"])
    _, place = locations(xy)
    nodes = fx[field]
    gap = []
    for row in nodes:
        fit = hierarchical_fit(fx["X"], fx["y"], np.ones(len(fx["y"])), coords=fx["coords"],
                               field=field, beta_sd=100.0,
                               theta=np.log([np.sqrt(float(row["sigma2"])), float(row["range"])]))
        coef = floats(row, [k for k in row if k.startswith("c") and k[1:].isdigit()])
        mine = np.asarray(fit["node_field"])
        if field == "nngp":
            mine = mine[place]
        tol = float(row["tolerance"])
        np.testing.assert_allclose(fit["beta"], floats(row, ("b0", "b1", "b2")), atol=tol)
        np.testing.assert_allclose(mine, coef, atol=tol)
        gap.append(fit["node_log_post"][0] - float(row["log_weight"]))
    assert max(gap) - min(gap) < float(nodes[0]["weight_tolerance"])


@pytest.mark.parametrize("cov", COVARIANCES)
def test_unit_intercepts_and_a_nearest_neighbour_field_are_the_dense_laplace_marginal(fx, cov):
    xy, _, _ = standardise(fx["coords"])
    loc, place = locations(xy)
    sigma_u, sigma, rng = 0.8, 1.3, 0.45
    fit = hierarchical_fit(fx["X"], fx["y"], fx["weight"], unit=fx["index"], n_unit=len(fx["units"]),
                           coords=fx["coords"], field="nngp", neighbours=5,
                           cov=COVARIANCES.index(cov), theta=np.log([sigma_u, sigma, rng]))
    precision, log_det = vecchia(loc, sigma, rng, cov, 5)
    s = np.zeros((len(fx["y"]), len(loc)))
    s[np.arange(len(fx["y"])), place] = 1
    latent, marginal = laplace(fx["X"], fx["y"], fx["weight"], s, precision, log_det, fx["index"],
                               sigma_u, 2.5)
    assert fit["log_marginal"] == pytest.approx(marginal, rel=1e-8)
    np.testing.assert_allclose(
        np.concatenate([fit["beta"], fit["node_field"], fit["unit_effect"]]), latent, atol=1e-7)


def test_unit_intercepts_and_a_hilbert_space_field_are_the_dense_laplace_marginal(fx):
    xy, _, _ = standardise(fx["coords"])
    sigma_u, sigma, rng = 0.8, 1.3, 0.45
    fit = hierarchical_fit(fx["X"], fx["y"], fx["weight"], unit=fx["index"], n_unit=len(fx["units"]),
                           coords=fx["coords"], field="hsgp", theta=np.log([sigma_u, sigma, rng]))
    design = hsgp_design(xy, sigma, rng)
    latent, marginal = laplace(fx["X"], fx["y"], fx["weight"], design, np.eye(design.shape[1]), 0.0,
                               fx["index"], sigma_u, 2.5)
    assert fit["log_marginal"] == pytest.approx(marginal, rel=1e-8)
    np.testing.assert_allclose(
        np.concatenate([fit["beta"], fit["node_field"], fit["unit_effect"]]), latent, atol=1e-7)


def test_a_nearest_neighbour_field_interpolates_to_new_places_as_its_conditional_mean(fx):
    xy, centre, scale = standardise(fx["coords"])
    loc, place = locations(xy)
    sigma, rng = 1.3, 0.45
    fit = hierarchical_fit(fx["X"], fx["y"], np.ones(len(fx["y"])), coords=fx["coords"],
                           field="nngp", neighbours=6, theta=np.log([sigma, rng]))
    gen = np.random.default_rng(5)
    newx = np.column_stack([np.ones(8), gen.normal(size=8), gen.normal(size=8)])
    new_coords = gen.uniform(size=(8, 2))
    got = hierarchical_predict(fit, newx, coords=new_coords)
    field = np.asarray(fit["node_field"])
    for i, p in enumerate((new_coords - centre) / scale):
        d = np.sqrt(((loc - p) ** 2).sum(axis=1))
        nb = np.argsort(d, kind="stable")[:6]
        pair = np.sqrt(((loc[nb][:, None] - loc[nb][None]) ** 2).sum(axis=2))
        c = kernel("exponential", pair, sigma ** 2, rng) + 1e-6 * np.eye(6)
        cv = kernel("exponential", d[nb], sigma ** 2, rng)
        assert got[i] == pytest.approx(newx[i] @ fit["beta"] + cv @ np.linalg.solve(c, field[nb]),
                                       abs=1e-10)
    at_training = hierarchical_predict(fit, fx["X"], coords=fx["coords"])
    np.testing.assert_allclose(at_training, fx["X"] @ fit["beta"] + field[place], atol=1e-5)


def test_a_fit_refuses_what_it_cannot_read(fx):
    w = np.ones(len(fx["y"]))
    with pytest.raises(Exception, match="zero and one"):
        hierarchical_fit(fx["X"], fx["y"] * 2, w)
    with pytest.raises(Exception, match="zero or more"):
        hierarchical_fit(fx["X"], fx["y"], -w)
    with pytest.raises(Exception, match="coordinates"):
        hierarchical_fit(fx["X"], fx["y"], w, field="nngp")
    with pytest.raises(Exception, match="covariance"):
        hierarchical_fit(fx["X"], fx["y"], w, coords=fx["coords"], field="nngp", cov=9)
    with pytest.raises(Exception, match="hyperparameters"):
        hierarchical_fit(fx["X"], fx["y"], w, coords=fx["coords"], field="hsgp", theta=[0.0])


# ---------------------------------------------------------------------------------------------
# The learner inside a run.

START = np.datetime64("2020-01-01T00:00:00", "s")


def run(spatial, coords=True, random=False, n=60, seed=2):
    gen = np.random.default_rng(seed)
    plots = [f"p{i:03d}" for i in range(n)]
    lon, lat = gen.uniform(size=n), gen.uniform(size=n)
    shift = np.repeat(gen.normal(size=n // 2), 2) if random else 0.0
    tg = {"plot": plots, "lon": lon.tolist(), "lat": lat.tolist(),
          "s1": gen.binomial(1, 1 / (1 + np.exp(-(-0.3 + 2 * np.sin(5 * lon) + shift)))).tolist(),
          "s2": gen.binomial(1, 1 / (1 + np.exp(-1.5 * (lat - 0.5)))).tolist()}
    t = START + np.arange(240) * np.timedelta64(12 * 3600, "s")
    series = {"plot": np.repeat(np.asarray(plots), len(t)), "when": np.tile(t, n),
              "v": gen.normal(size=n * len(t))}
    y = Response(np.column_stack([tg["s1"], tg["s2"]]).astype(float), tuple(plots), ("s1", "s2"))
    fit = timesift(tg, series, y=("s1", "s2"), id="plot", time="when", x="v",
                   coords=("lon", "lat") if coords else None,
                   models=[hierarchical(spatial=spatial, random=random)], sift=grains("month"),
                   resampling=fold_map(y, v=3, seed=2), inner=None, ensemble=False, verbose=False)
    return tg, series, fit


def test_a_field_needs_the_coordinates_and_a_fit_without_one_does_not():
    with pytest.raises(ValueError, match="coords="):
        run("hsgp", coords=False)
    tg, series, fit = run("none", coords=False)
    sub = {k: v[:5] for k, v in tg.items()}
    p = fit.predict(sub, series, candidate="selected")
    assert p.shape == (5, 2) and ((p > 0) & (p < 1)).all()


@pytest.mark.parametrize("spatial", ["hsgp", "nngp"])
def test_a_field_predicts_from_the_coordinates_of_the_new_units_and_pickles(spatial):
    tg, series, fit = run(spatial)
    new = {k: v[:6] for k, v in tg.items()}
    p = fit.predict(new, series, candidate="selected")
    assert p.shape == (6, 2)
    moved = {**new, "lon": [1 - v for v in new["lon"]]}
    assert not np.allclose(p, fit.predict(moved, series, candidate="selected"))
    again = pickle.loads(pickle.dumps(fit))
    np.testing.assert_allclose(again.predict(new, series, candidate="selected"), p)


def test_an_intercept_for_each_unit_needs_the_unit_and_a_held_out_unit_is_predicted_at_the_population_level():
    tg, series, fit = run("none", coords=False, random=True)
    p = fit.predict(tg, series, candidate="selected")
    assert p.shape == (60, 2) and ((p > 0) & (p < 1)).all()
    _, _, plain = run("none", coords=False, random=False)
    assert not np.allclose(p, plain.predict(tg, series, candidate="selected"))
