"""The learners that ship on the tabular side: the penalised fit, the selector and the forest."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from timesift.control import train_control
from timesift.learners import (Learner, elasticnet, fit_learner, flatten, forest, stepwise,
                               _apply_basis, _design, _logistic, _poly_basis)
from timesift.metrics import roc_auc
from timesift.representation import grain_matrix
from timesift.response import Response
from timesift.specs import grain

needs_sklearn = pytest.mark.skipif(importlib.util.find_spec("sklearn") is None,
                                   reason="scikit-learn is not installed")


def planted(n_unit=40, days=56, noise=1.0, seed=17):
    """A record whose weekly level carries the response, and nothing else does."""
    rng = np.random.default_rng(seed)
    t = np.datetime64("2021-09-01T00:00:00", "s") + np.arange(24 * days) * np.timedelta64(1, "h")
    units = [f"p{i:02d}" for i in range(n_unit)]
    warmth = rng.normal(size=n_unit)
    value = np.concatenate([w * 2.0 + rng.normal(0, noise, len(t)) for w in warmth])
    d = {"id": [u for u in units for _ in range(len(t))],
         "time": list(t) * n_unit, "value": list(value)}
    y = rng.binomial(1, 1 / (1 + np.exp(-3 * np.column_stack([warmth, -warmth])))).astype(float)
    x = grain_matrix(d, "id", "time", "value", grain="week")
    return x, Response(y, tuple(units), ("sp1", "sp2"))


def test_the_selector_fits_predicts_and_finds_the_planted_signal():
    x, y = planted()
    fit = fit_learner(stepwise(max_terms=2), x, y)
    p = fit.predict(x)
    assert p.shape == (len(y.units), 2)
    assert np.all((p >= 0) & (p <= 1))
    assert roc_auc(y.values[:, 0], p[:, 0]) > 0.75


def test_the_selector_admits_no_more_columns_than_its_budget():
    x, y = planted()
    for budget in (1, 3):
        model = fit_learner(stepwise(max_terms=budget), x, y).model
        for f in model["models"]:
            assert len(f.get("columns", [])) <= budget


def test_a_variable_with_one_outcome_is_predicted_as_its_share():
    x, y = planted()
    flat = Response(np.column_stack([y.values[:, 0], np.zeros(len(y.units))]),
                    y.units, ("sp1", "absent"))
    fit = fit_learner(stepwise(), x, flat)
    p = fit.predict(x)
    assert np.allclose(p[:, 1], 0.0)


def test_predicting_a_single_unit_returns_one_row_and_not_one_column():
    x, y = planted()
    fit = fit_learner(stepwise(max_terms=2), x, y)
    one = x.take_units([0])
    assert fit.predict(one).shape == (1, 2)


def test_the_selector_refuses_a_representation_it_was_not_fitted_on():
    x, y = planted()
    fit = fit_learner(stepwise(max_terms=1), x, y)
    with pytest.raises(ValueError, match="different channels or bins"):
        fit.predict(grain_matrix(
            {"id": [u for u in y.units for _ in range(24)],
             "time": list(np.datetime64("2021-09-01T00:00:00", "s")
                          + np.arange(24) * np.timedelta64(1, "h")) * len(y.units),
             "value": list(np.zeros(24 * len(y.units)))},
            "id", "time", "value", grain="day"))


def test_the_polynomial_basis_is_orthonormal_and_travels_with_the_fit():
    rng = np.random.default_rng(2)
    v = rng.normal(size=50)
    basis = _poly_basis(v, 3)
    z = basis["values"]
    # Orthonormal columns, and orthogonal to the constant: that is what makes a degree admitted
    # after another one carry only what the earlier one does not.
    assert np.allclose(z.T @ z, np.eye(3), atol=1e-10)
    assert np.allclose(z.sum(axis=0), 0.0, atol=1e-10)
    # New units are mapped through the basis the fit carries, never through one re-derived from
    # themselves: a subset re-derived would give a different basis.
    part = _apply_basis(basis, v[:10])
    assert np.allclose(part, z[:10])
    assert not np.allclose(_poly_basis(v[:10], 3)["values"], part)


def test_a_degree_beyond_what_the_readings_distinguish_is_dropped():
    assert _poly_basis(np.array([1.0, 1.0, 2.0, 2.0]), 3)["degree"] == 1
    assert _poly_basis(np.array([1.0, 2.0, 3.0]), 5)["degree"] == 2


def test_a_column_holding_one_value_is_not_offered_to_the_forward_search(temporary_response):
    from dataclasses import replace

    from timesift.response import as_response, scorable_cells
    temporary_response("constant_continuous_test", dict(
        prepare=as_response, activation="identity", loss="squared_error", metric="roc_auc",
        cells=lambda y, folds: scorable_cells(y, folds)))

    x, y = planted(seed=37)
    # A channel every unit reads the same value on: what a sensor that never moved records, or a
    # predictor that is constant over the units this fold holds.
    values = np.concatenate([x.values, np.full((*x.values.shape[:2], 1), 7.0)], axis=2)
    flat = replace(x, values=values, stats=(*x.stats, "flat"))
    n_bin = x.values.shape[1]

    level = x.values[:, :, 0].mean(axis=1)
    level = 10 + 3 * (level - level.mean()) / level.std()
    arms = [(y, "presence_absence"),
            (Response(level.reshape(-1, 1), y.units, ("height",)), "constant_continuous_test")]
    for response, name in arms:
        fit = fit_learner(stepwise(max_terms=2), flat, response, response=name)
        chosen = fit.model["models"][0]["columns"]
        assert chosen, name
        # flatten() lays the channels out one block of bins after another, so the constant one is
        # every column from the last block on.
        assert max(chosen) < n_bin * len(x.stats), name
        assert fit.predict(flat).shape == (len(response.units), len(response.variables))

    with pytest.raises(ValueError, match="no polynomial basis"):
        _poly_basis(np.full(20, 7.0), 2)


def test_a_separated_or_unsettled_fit_is_refused_rather_than_returned():
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    separating = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]).reshape(-1, 1)
    assert _logistic(separating, y) is None
    overlapping = np.array([-1.0, 1.0, -0.5, 0.5, -0.2, 0.7]).reshape(-1, 1)
    fit = _logistic(overlapping, y)
    assert fit is not None and np.isfinite(fit["aic"])


@needs_sklearn
def test_the_penalised_fit_uses_the_mixing_it_was_given_and_a_standardised_design():
    x, y = planted(n_unit=30, days=28, seed=41)
    for alpha in (0.2, 0.9):
        model = fit_learner(elasticnet(alpha=alpha, n_inner=3), x, y).model
        scaler, penalised = model["models"][0][0], model["models"][0][-1]
        assert float(np.asarray(penalised.l1_ratio_).ravel()[0]) == alpha

        # The scaler travels with the fit, so new units are mapped through the centre and the
        # spread the model was fitted at rather than through their own.
        design = scaler.transform(_design(x, model["squares"]))
        assert np.allclose(design.mean(axis=0), 0, atol=1e-8)
        assert np.allclose(design.std(axis=0), 1, atol=1e-8)


@needs_sklearn
def test_a_setting_given_at_fit_time_overrides_the_one_the_learner_carries():
    x, y = planted(n_unit=24, days=28)
    overridden = fit_learner(elasticnet(), x, y, squares=False)
    built = fit_learner(elasticnet(squares=False), x, y)
    assert overridden.model["squares"] is False
    assert np.allclose(overridden.predict(x), built.predict(x))


@needs_sklearn
def test_the_settings_a_learner_carries_are_the_ones_it_reports():
    learner = elasticnet(alpha=0.25, n_inner=3)
    assert learner.params["alpha"] == 0.25
    assert learner.params["n_inner"] == 3
    assert learner.params["weight_positives"] is True


def test_every_learner_declares_what_it_reads_and_how_it_covers_the_responses():
    declared = {"elasticnet": ("tabular", "separate"), "stepwise": ("tabular", "separate"),
                "forest": ("tabular", "separate")}
    for build in (elasticnet, stepwise, forest):
        learner = build()
        assert (learner.reads, learner.multi) == declared[learner.name]
        assert learner.data is None
        # What a learner is pinned to is a representation, never the name of one: the layer above
        # reads its kind and its label before anything is built, and a string carries neither.
        assert build(data=grain("week")).data == grain("week")
        with pytest.raises(ValueError, match="must be a representation"):
            build(data="week")


def test_a_learner_can_only_declare_what_the_layer_above_knows_how_to_read():
    with pytest.raises(ValueError, match="`reads` is one of"):
        Learner(name="x", fit=lambda *a, **k: None, predict=lambda *a, **k: None, reads="raw")
    with pytest.raises(ValueError, match="`multi` is one of"):
        Learner(name="x", fit=lambda *a, **k: None, predict=lambda *a, **k: None, multi="both")


def test_a_learner_names_the_install_it_needs():
    learner = Learner(name="imaginary", fit=lambda *a, **k: None,
                      predict=lambda *a, **k: None, needs=("no_such_package",))
    with pytest.raises(ImportError, match="pip install no_such_package"):
        learner.require()


@needs_sklearn
def test_the_forest_fits_predicts_and_finds_the_planted_signal():
    x, y = planted()
    fit = fit_learner(forest(trees=50), x, y)
    p = fit.predict(x)
    assert p.shape == (len(y.units), 2)
    assert np.all((p >= 0) & (p <= 1))
    assert roc_auc(y.values[:, 0], p[:, 0]) > 0.75


@needs_sklearn
def test_the_forest_reads_the_flattened_representation_and_refuses_another_shape():
    x, y = planted(n_unit=24, days=28)
    fit = fit_learner(forest(trees=20), x, y)
    assert fit.model["n_col"] == flatten(x).shape[1]
    with pytest.raises(ValueError, match="different channels or bins"):
        fit.predict(grain_matrix(
            {"id": [u for u in y.units for _ in range(24)],
             "time": list(np.datetime64("2021-09-01T00:00:00", "s")
                          + np.arange(24) * np.timedelta64(1, "h")) * len(y.units),
             "value": list(np.zeros(24 * len(y.units)))},
            "id", "time", "value", grain="day"))


@needs_sklearn
def test_the_forest_carries_its_own_seed_and_repeats_itself():
    x, y = planted(n_unit=24, days=28)
    a = fit_learner(forest(trees=30, seed=4), x, y).predict(x)
    b = fit_learner(forest(trees=30, seed=4), x, y).predict(x)
    c = fit_learner(forest(trees=30, seed=9), x, y).predict(x)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)
    # The run's control carries the seed of the encoders' optimiser, not the forest's bootstrap.
    d = fit_learner(forest(trees=30, seed=4), x, y, control=train_control(seed=9)).predict(x)
    assert np.allclose(a, d)


@needs_sklearn
def test_each_response_draws_its_own_bootstrap_from_a_seed_of_its_own_name():
    # Two responses with the same observations under different names would otherwise be the
    # same forest, and a response fitted alone would differ from the same response fitted beside
    # others.
    x, y = planted(n_unit=24, days=28)
    twin = Response(values=np.column_stack([y.values[:, 0], y.values[:, 0]]), units=y.units,
                    variables=("sp_a", "sp_b"))
    both = fit_learner(forest(trees=30, seed=4), x, twin).predict(x)
    assert not np.allclose(both[:, 0], both[:, 1])
    alone = fit_learner(forest(trees=30, seed=4), x,
                        Response(values=y.values[:, :1], units=y.units,
                                 variables=("sp_a",))).predict(x)
    assert np.allclose(both[:, 0], alone[:, 0])
    from timesift.learners import _variable_seeds
    # The offsets R's `.name_offset()` gives the same two names.
    assert _variable_seeds(1, ("sp_a", "sp_b")) == [1 + 80582, 1 + 80583]


@needs_sklearn
def test_a_response_with_one_outcome_is_its_share_whichever_model_covers_it():
    x, y = planted(n_unit=24, days=28)
    flat = Response(np.column_stack([y.values[:, 0], np.ones(len(y.units))]),
                    y.units, ("sp1", "everywhere"))
    for learner in (forest(trees=20), elasticnet()):
        p = fit_learner(learner, x, flat).predict(x)
        assert np.allclose(p[:, 1], 1.0)


def test_the_fit_is_the_maximum_likelihood_one_and_not_merely_a_settled_one():
    """At the coefficients returned, the step a maximum-likelihood fitter would take next is
    nothing.

    The stopping rule is the relative change in the deviance, which is the one R's ``glm`` uses, so
    the point reached is not exactly stationary. What has to hold is that it is inside any
    precision a criterion is read at: the largest remaining step, measured across six designs, is
    7e-8 in coefficient units against coefficients of order one. That is what makes the criterion
    the selector reads comparable to another implementation's rather than to this fitter's own
    tolerance.
    """
    for seed in range(6):
        rng = np.random.default_rng(seed + 1)
        n = 60
        x1, x2 = rng.normal(size=n), rng.uniform(-2, 2, n)
        y = rng.binomial(1, 1 / (1 + np.exp(-(-0.4 + 1.3 * x1 - 0.8 * x2)))).astype(float)
        fit = _logistic(np.column_stack([x1, x2]), y)
        assert fit is not None

        x = np.column_stack([np.ones(n), x1, x2])
        mu = 1.0 / (1.0 + np.exp(-(x @ fit["beta"])))
        score = x.T @ (y - mu)
        w = mu * (1.0 - mu)
        step = np.linalg.solve((x * w[:, None]).T @ x, score)
        assert np.max(np.abs(step)) < 1e-6

        # The deviance reported is the one those coefficients give, so a criterion built on it
        # counts the right number of parameters.
        deviance = -2 * float(np.sum(y * np.log(mu) + (1 - y) * np.log1p(-mu)))
        assert fit["deviance"] == pytest.approx(deviance)
        assert fit["aic"] == pytest.approx(deviance + 2 * 3)

        # And no coefficients a thousand times further out than that step do better, which is what
        # a maximum means.
        for _ in range(20):
            nudged = fit["beta"] + rng.normal(0, 1e-3, len(fit["beta"]))
            m = 1.0 / (1.0 + np.exp(-(x @ nudged)))
            assert -2 * float(np.sum(y * np.log(m) + (1 - y) * np.log1p(-m))) >= deviance - 1e-9


def test_the_basis_spans_the_polynomials_of_its_own_degree():
    """Orthonormal columns alone would not make it a polynomial basis. Every raw power up to the
    degree has to be a combination of the columns and the constant, which is what lets a term be
    non-monotone in the reading the way a niche optimum is."""
    rng = np.random.default_rng(6)
    v = rng.normal(size=40)
    z = np.column_stack([np.ones(len(v)), _poly_basis(v, 3)["values"]])
    for power in (1, 2, 3):
        raw = v ** power
        fitted = z @ np.linalg.lstsq(z, raw, rcond=None)[0]
        assert np.allclose(fitted, raw, atol=1e-9)
    # And a fourth power is not, so the degree is the degree asked for.
    fourth = v ** 4
    assert not np.allclose(z @ np.linalg.lstsq(z, fourth, rcond=None)[0], fourth, atol=1e-6)


@needs_sklearn
def test_the_per_response_learners_fit_the_family_the_response_heads_loss_names(temporary_response):
    from timesift.response import as_response, scorable_cells
    temporary_response("continuous_test", dict(
        prepare=as_response, activation="identity", loss="squared_error", metric="roc_auc",
        cells=lambda y, folds: scorable_cells(y, folds)))
    x, y = planted(seed=91)
    level = x.values[:, :, 0].mean(axis=1)
    level = 10 + 3 * (level - level.mean()) / level.std()
    yc = Response(level.reshape(-1, 1), y.units, ("height",))
    for learner in (elasticnet(squares=False), forest(trees=100), stepwise(max_terms=1)):
        fit = fit_learner(learner, x, yc, response="continuous_test")
        p = fit.predict(x)
        assert (p > 1).all(), learner.name
        assert np.corrcoef(p[:, 0], level)[0, 1] > 0.8, learner.name
        assert fit.model["family"] == "gaussian"

    temporary_response("poisson_test", dict(
        prepare=as_response, activation="exp", loss="poisson", metric="roc_auc",
        cells=lambda y, folds: scorable_cells(y, folds)))
    with pytest.raises(ValueError, match="no family for the 'poisson'"):
        fit_learner(stepwise(), x, yc, response="poisson_test")


def test_a_learners_fit_is_handed_the_head_only_where_it_declares_one():
    x, y = planted(n_unit=10, days=14, seed=92)
    seen = {}

    def plain(x, y, **params):
        seen.update(params)
        return 1

    def aware(x, y, head, **params):
        return head["loss"]

    constant = lambda model, x: np.full((x.values.shape[0], 1), 0.5)  # noqa: E731
    fit_learner(Learner("plain", fit=plain, predict=constant), x, y.take_variables([0]))
    assert "head" not in seen
    fit = fit_learner(Learner("aware", fit=aware, predict=constant), x, y.take_variables([0]))
    assert fit.model == "binary_cross_entropy"


def test_a_learners_fit_is_handed_the_control_only_where_it_declares_one():
    x, y = planted(n_unit=10, days=14, seed=94)
    one = y.take_variables([0])
    constant = lambda model, x: np.full((x.values.shape[0], 1), 0.5)  # noqa: E731

    def two_arguments(x, y):
        return "fitted"

    def trained(x, y, control=None):
        return control

    control = train_control(epochs=2)
    # A fit that declares neither is called with neither, whatever the run carries.
    assert fit_learner(Learner("plain", fit=two_arguments, predict=constant), x, one,
                       control=control).model == "fitted"
    assert fit_learner(Learner("trained", fit=trained, predict=constant), x, one,
                       control=control).model is control
