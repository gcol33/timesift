from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from timesift.control import TrainControl, train_control
from timesift.learners import cnn, fit_learner, mlp, rescnn
from timesift.metrics import roc_auc
from timesift.representation import grain_matrix
from timesift.response import Response

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None,
                                reason="torch is not installed")


def fixture(n_unit=40, days=90, noise=0.5, level=3.0, seed=81):
    rng = np.random.default_rng(seed)
    t = np.datetime64("2021-09-01T00:00:00", "s") + np.arange(24 * days) * np.timedelta64(1, "h")
    units = [f"p{i:03d}" for i in range(n_unit)]
    warmth = rng.normal(size=n_unit)
    shape = 5 * np.sin(np.arange(len(t)) / (24 * 30))
    value = np.concatenate([level * w + shape + rng.normal(0, noise, len(t)) for w in warmth])
    readings = {"id": [u for u in units for _ in range(len(t))],
                "time": list(t) * n_unit, "value": list(value)}
    y = rng.binomial(1, 1 / (1 + np.exp(-4 * np.outer(warmth, [1, -1]))))
    x = grain_matrix(readings, "id", "time", "value", grain="week",
                      stats=["cold_day", "mean", "warm_day"])
    return x, Response(y.astype(float), tuple(units), ("sp0", "sp1")), readings


@pytest.mark.parametrize("build", [
    lambda: mlp(epochs=3),
    lambda: cnn(epochs=3),
    lambda: rescnn(epochs=3, channels=(16, 32)),
])
def test_every_encoder_fits_and_returns_one_probability_per_unit_and_variable(build):
    x, y, _ = fixture()
    p = fit_learner(build(), x, y).predict(x)
    assert p.shape == (40, 2)
    assert ((p > 0) & (p < 1)).all()


def test_the_same_seed_gives_the_same_fit():
    x, y, _ = fixture(n_unit=24, days=60)
    a = fit_learner(cnn(epochs=3, seed=4), x, y).predict(x)
    b = fit_learner(cnn(epochs=3, seed=4), x, y).predict(x)
    assert np.allclose(a, b)


def test_the_stack_still_runs_where_the_record_is_one_bin_per_year():
    _, y, readings = fixture(n_unit=24, days=400, seed=83)
    for w in ("season", "year"):
        x = grain_matrix(readings, "id", "time", "value", grain=w)
        assert x.values.shape[1] <= 5
        for build in (lambda: cnn(epochs=2),
                      lambda: rescnn(epochs=2, channels=(16, 32))):
            assert fit_learner(build(), x, y).predict(x).shape == (24, 2)


def test_an_encoder_refuses_a_representation_it_was_not_fitted_on():
    x, y, readings = fixture(n_unit=24, days=60)
    fit = fit_learner(cnn(epochs=2), x, y)
    other = grain_matrix(readings, "id", "time", "value", grain="month")
    with pytest.raises(ValueError, match="different channels or bins"):
        fit.predict(other)


def test_a_fully_connected_encoder_recovers_a_planted_signal():
    x, y, _ = fixture(n_unit=90, days=90, noise=0.3, seed=85)
    p = fit_learner(mlp(epochs=40, seed=3), x, y).predict(x)
    assert roc_auc(y.values[:, 0], p[:, 0]) > 0.8


def test_weight_averaging_runs_the_whole_averaging_grain_and_returns_a_usable_fit():
    x, y, _ = fixture(n_unit=30, days=60)
    p = fit_learner(cnn(epochs=6, swa=True, swa_start=0.5), x, y).predict(x)
    assert p.shape == (30, 2)
    assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all()


def test_averaging_the_tail_is_not_the_same_fit_as_keeping_one_epoch_of_it():
    x, y, _ = fixture(n_unit=30, days=60)
    averaged = fit_learner(cnn(epochs=6, swa=True, swa_start=0.5, seed=7), x, y).predict(x)
    single = fit_learner(cnn(epochs=6, seed=7), x, y).predict(x)
    assert not np.allclose(averaged, single)


def test_a_setting_given_at_fit_time_overrides_the_one_the_learner_carries():
    x, y, _ = fixture(n_unit=20, days=40)
    wide = fit_learner(cnn(epochs=2), x, y, channels=(8, 16)).predict(x)
    narrow = fit_learner(cnn(epochs=2, channels=(8, 16)), x, y).predict(x)
    assert np.allclose(wide, narrow)
    assert not np.allclose(wide, fit_learner(cnn(epochs=2), x, y).predict(x))


def test_an_architecture_carries_architecture_and_the_control_carries_the_training():
    learner = cnn(channels=(8, 16), epochs=4)
    assert learner.params["channels"] == (8, 16)
    # The settings a learner was not told about are the control's, so there is one place they are
    # defaulted and a learner carries only what it was asked to change.
    assert learner.params["epochs"] == 4
    assert "batch_size" not in learner.params
    assert TrainControl().batch_size == train_control().batch_size == 64


def test_a_training_setting_reaches_a_learner_through_the_control():
    x, y, _ = fixture(n_unit=20, days=40)
    control = fit_learner(cnn(channels=(8, 16)), x, y,
                          control=train_control(epochs=2, seed=5)).predict(x)
    given = fit_learner(cnn(channels=(8, 16), epochs=2, seed=5), x, y).predict(x)
    assert np.allclose(control, given)


def test_a_setting_on_a_learner_wins_over_the_control_it_is_fitted_under():
    x, y, _ = fixture(n_unit=20, days=40)
    pinned = fit_learner(cnn(channels=(8, 16), epochs=2, seed=5), x, y,
                         control=train_control(epochs=2, seed=11)).predict(x)
    same = fit_learner(cnn(channels=(8, 16), epochs=2, seed=5), x, y).predict(x)
    assert np.allclose(pinned, same)


def test_a_setting_no_learner_and_no_control_carries_is_refused():
    with pytest.raises(TypeError, match="no training setting called learning_rat"):
        cnn(learning_rat=1e-3)
    x, y, _ = fixture(n_unit=20, days=40)
    with pytest.raises(TypeError, match="no setting called kernal"):
        fit_learner(cnn(epochs=2), x, y, kernal=3)


def test_no_training_batch_holds_one_row():
    """Batch normalisation has no variance to standardise a single row by, and the layer's running
    statistics take a NaN from it. The batches are as equal as they can be for that reason, and
    none holds more than the batch size."""
    from timesift.learners import _batches
    for n in (17, 20, 33, 65):
        parts = _batches(np.arange(n), 8)
        assert all(1 < len(p) <= 8 for p in parts)
        assert sum(len(p) for p in parts) == n
    # Three rows under a batch size of two would cut into two and one; they stay one batch.
    assert [len(p) for p in _batches(np.arange(3), 2)] == [3]
    assert [len(p) for p in _batches(np.arange(65), 64)] == [33, 32]


def test_standardisation_is_per_channel_and_computed_on_the_units_handed():
    x, y, _ = fixture(n_unit=24, days=60)
    half = np.arange(12)
    fit = fit_learner(cnn(epochs=2), x.take_units(half), y.take_units(half))
    values = x.values[half]
    assert np.allclose(fit.model["centre"].ravel(), values.mean(axis=(0, 1)))
    assert np.allclose(fit.model["scale"].ravel(), values.std(axis=(0, 1), ddof=1))
    assert not np.allclose(fit.model["centre"].ravel()[0], x.values[:, :, 0].mean())


def test_a_static_predictor_appended_as_a_channel_is_put_on_the_readings_footing():
    from dataclasses import replace
    x, y, _ = fixture(n_unit=24, days=60)
    elevation = 2000 + 500 * np.arange(24)
    block = np.broadcast_to(elevation[:, None, None], (24, x.values.shape[1], 1))
    with_static = replace(x, values=np.concatenate([x.values, block], axis=2),
                          stats=tuple(x.stats) + ("elevation",))
    fit = fit_learner(cnn(epochs=2), with_static, y)
    # Standardising every channel by one global scale would crush the readings to a constant
    # beside metres of elevation; per channel, the readings keep their spread.
    scaled = (np.transpose(with_static.values, (0, 2, 1)) - fit.model["centre"]) / fit.model["scale"]
    assert scaled[:, 0, :].std() > 0.5
    assert np.isclose(fit.model["centre"].ravel()[3], elevation.mean())
    assert np.isfinite(fit.predict(with_static)).all()


def test_a_snapshot_of_the_weights_is_a_copy_and_not_the_optimisers_storage():
    import torch
    from timesift.learners import _snapshot
    net = torch.nn.Linear(3, 1)
    before = _snapshot(net)
    opt = torch.optim.AdamW(net.parameters(), lr=0.1)
    for _ in range(3):
        opt.zero_grad()
        net(torch.randn(8, 3)).pow(2).mean().backward()
        opt.step()
    assert not torch.allclose(before["weight"], net.weight)
    net.load_state_dict(before)
    assert torch.equal(net.weight, before["weight"])


def test_early_stopping_restores_the_best_epoch_rather_than_the_last_one():
    from timesift.learners import _validation_split
    x, y, _ = fixture(n_unit=30, days=60)
    one = fit_learner(mlp(epochs=1, learning_rate=0.5, seed=2, val_frac=0.3), x, y)
    stopped = fit_learner(mlp(epochs=12, early_stopping=1, learning_rate=0.5, seed=2,
                              val_frac=0.3), x, y)
    val = _validation_split(y.values, 0.3, np.random.default_rng(2))
    fit_idx = np.setdiff1d(np.arange(30), val)
    pos = y.values[fit_idx].sum(axis=0)
    w = np.clip(np.where(pos > 0, (len(fit_idx) - pos) / np.maximum(pos, 1), 1), 1, 50)

    def loss(fit):
        p = np.clip(fit.predict(x)[val], 1e-6, 1 - 1e-6)
        yv = y.values[val]
        return -np.mean(w * yv * np.log(p) + (1 - yv) * np.log(1 - p))

    assert loss(stopped) <= loss(one) * (1 + 1e-5)


def test_a_fitted_encoder_pickles_and_predicts_the_same_after_loading():
    import pickle
    x, y, _ = fixture(n_unit=24, days=60)
    fit = fit_learner(rescnn(epochs=2, channels=(8, 16)), x, y)
    p = fit.predict(x)
    back = pickle.loads(pickle.dumps(fit))
    assert np.allclose(back.predict(x), p)
    assert all(isinstance(v, np.ndarray) for v in back.model["state"].values())


def test_the_encoders_train_under_the_loss_the_response_head_names(temporary_response):
    from timesift.response import as_response, scorable_cells
    x, y, _ = fixture(n_unit=40, days=60)
    temporary_response("continuous_test", dict(
        prepare=as_response, activation="identity", loss="squared_error", metric="roc_auc",
        cells=lambda y, folds: scorable_cells(y, folds)))
    level = x.values[:, :, 1].mean(axis=1)
    level = 3 * (level - level.mean()) / level.std()
    yc = Response(level.reshape(-1, 1), y.units, ("height",))
    fit = fit_learner(mlp(epochs=80, learning_rate=0.01, seed=3, val_frac=0), x, yc,
                      response="continuous_test")
    p = fit.predict(x)
    assert fit.model["activation"] == "identity"
    assert (np.abs(p) > 1).any()
    assert np.corrcoef(p[:, 0], level)[0, 1] > 0.8


def test_a_loss_the_encoders_do_not_know_is_refused_by_name(temporary_response):
    from timesift.response import as_response, scorable_cells
    x, y, _ = fixture(n_unit=20, days=40)
    temporary_response("poisson_test", dict(
        prepare=as_response, activation="exp", loss="poisson", metric="roc_auc",
        cells=lambda y, folds: scorable_cells(y, folds)))
    with pytest.raises(ValueError, match="do not train under the 'poisson' loss"):
        fit_learner(cnn(epochs=1), x, y, response="poisson_test")


def test_the_averaged_weights_batch_norm_statistics_are_the_plain_mean_over_the_pass():
    import torch
    from timesift.learners import _refresh_batchnorm
    net = torch.nn.Sequential(torch.nn.BatchNorm1d(2))
    xt = torch.randn(20, 2, 5) * 4 + 7
    net.train()
    net(xt[:10] * 0 - 100)
    _refresh_batchnorm(net, xt, np.arange(20), 5, "cpu")
    assert torch.allclose(net[0].running_mean, xt.mean(dim=(0, 2)), atol=1e-5)
    assert net[0].momentum == 0.1


def test_the_inner_validation_set_is_drawn_from_every_level_of_the_response():
    from timesift.learners import _validation_split
    y = np.column_stack([np.r_[np.ones(6), np.zeros(54)], np.zeros(60)])
    rng = np.random.default_rng(1)
    for _ in range(20):
        val = _validation_split(y, 1 / 6, rng)
        assert len(val) == 10
        # The last of the ten strata is the six presences alone, so every draw carries exactly
        # one presence into the validation set.
        assert y[val, 0].sum() == 1
    assert len(_validation_split(y, 0, rng)) == 0
    assert len(_validation_split(y[:2], 0.5, rng)) == 0


def test_the_control_ranges_are_the_ones_the_r_side_checks():
    with pytest.raises(ValueError, match="at least 1"):
        train_control(pos_weight_cap=0.5)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        train_control(swa_start=1)
    assert train_control(swa_start=0).swa_start == 0
    assert train_control(device="cpu").device == "cpu"


def test_a_fitted_encoder_names_its_module_builder_rather_than_carrying_it():
    pytest.importorskip("torch")
    from timesift.learners import _torch_module
    x, y, _ = fixture(n_unit=20, days=40)
    fit = fit_learner(cnn(epochs=2, channels=(8, 16)), x, y)
    assert fit.model["module"] == "cnn"
    assert not any(callable(v) for v in fit.model.values())
    # A saved fit rebuilds the network from this version's builder, so one naming a builder the
    # package no longer carries says so rather than loading its weights into another architecture.
    with pytest.raises(ValueError, match="gru"):
        _torch_module("gru")

def test_the_inner_validation_set_keeps_a_grouping_whole_as_the_outer_folds_do():
    from timesift.learners import _validation_split
    rng = np.random.default_rng(3)
    y = np.column_stack([rng.binomial(1, 0.3, 40), rng.binomial(1, 0.5, 40)]).astype(float)
    group = [f"g{i:02d}" for i in range(20) for _ in (0, 1)]
    for _ in range(10):
        val = _validation_split(y, 0.25, rng, group)
        # Five of twenty groups, so ten rows, and never one row of a group without the other.
        assert len(val) == 10
        drawn = [group[i] for i in val]
        assert all(drawn.count(g) == 2 for g in set(drawn))
