from __future__ import annotations

import numpy as np
import pytest

from timesift import fit_learner, forest, grains, timesift, tune
from timesift.learners import Learner
from timesift.representation import grain_matrix
from timesift.response import Response, fold_map
from timesift.tune import Tuned


def tune_data(n_unit=40, seed=81):
    rng = np.random.default_rng(seed)
    t = np.datetime64("2021-09-01T00:00:00", "s") + np.arange(24 * 90) * np.timedelta64(1, "h")
    units = [f"p{i:02d}" for i in range(n_unit)]
    warmth = rng.normal(size=n_unit)
    value = np.concatenate([w + rng.normal(0, 0.5, len(t)) for w in warmth])
    readings = {"plot": [u for u in units for _ in range(len(t))], "time": list(t) * n_unit,
                "value": list(value)}
    x = grain_matrix({"id": readings["plot"], "time": readings["time"], "value": readings["value"]},
                     "id", "time", "value", grain="month")
    s = x.values[:, :, 0].mean(axis=1)
    y = Response(np.column_stack([s > np.median(s), s < np.median(s)]).astype(float),
                 tuple(units), ("sp1", "sp2"))
    return x, y, readings, s


def k_learner():
    """Predicts the ranking of the unit means exactly at k = 3 and buries it in a fixed jitter that
    grows with the distance from 3, so the setting that scores best is known."""
    def predict(model, x):
        s = x.values[:, :, 0].mean(axis=1)
        noise = np.sin(np.arange(1, len(s) + 1) * 12.9898) * abs(model["k"] - 3) * 10
        return np.column_stack([s + noise, -(s + noise)])

    return Learner(name="kl", multi="joint", params=dict(k=1, other="kept"),
                   fit=lambda x, y, k, other, **kw: dict(k=k, other=other), predict=predict)


def test_the_setting_that_scores_best_on_the_inner_folds_is_the_one_fitted():
    x, y, _, s = tune_data()
    tuned = tune(k_learner(), {"k": [1, 2, 3, 4, 5]}, n_inner=3)
    assert tuned.name == "kl"
    fit = fit_learner(tuned, x, y)
    assert isinstance(fit.model, Tuned)
    assert fit.model.chosen == {"k": 3}
    table = fit.model.table
    assert len(table) == 5
    assert [r["score"] for r in table if r["settings"] == "k = 3"] == [1.0]
    assert all(r["score"] < 1 for r in table if r["settings"] != "k = 3")
    assert fit.model.model["k"] == 3 and fit.model.model["other"] == "kept"
    assert np.allclose(fit.predict(x)[:, 0], s)


def test_a_grid_is_every_combination_and_a_setting_given_to_the_fit_overrides_the_carried_one():
    x, y, _, _ = tune_data()
    fit = fit_learner(tune(k_learner(), {"k": [3, 9], "other": ["a", "b"]}, n_inner=3), x, y)
    assert [r["settings"] for r in fit.model.table] == [
        "k = 3, other = a", "k = 9, other = a", "k = 3, other = b", "k = 9, other = b"]
    # Ties go to the first combination in the grid.
    assert fit.model.chosen == {"k": 3, "other": "a"}
    given = fit_learner(tune(k_learner(), {"k": [3]}, n_inner=3), x, y, other="given")
    assert given.model.model["other"] == "given"


def test_the_inner_folds_keep_a_grouping_whole_and_a_metric_of_its_own_is_used():
    x, y, _, _ = tune_data()
    group = [f"g{i // 4:02d}" for i in range(40)]
    fit = fit_learner(tune(k_learner(), {"k": [1, 2, 3, 4, 5]}, n_inner=3), x, y, group=group)
    assert fit.model.chosen == {"k": 3}
    by_tss = fit_learner(tune(k_learner(), {"k": [1, 2, 3, 4, 5]}, metric="tss", n_inner=3), x, y)
    assert by_tss.model.chosen == {"k": 3} and by_tss.model.table[2]["score"] == 1.0
    by_function = fit_learner(
        tune(k_learner(), {"k": [1, 2, 3, 4, 5]}, n_inner=3,
             metric=lambda yy, p: -abs(float(np.mean(p)) - 0.2)), x, y)
    assert by_function.model.chosen["k"] in (1, 2, 3, 4, 5)


def test_a_shipped_learner_is_tuned_the_same_way():
    x, y, _, _ = tune_data()
    fit = fit_learner(tune(forest(), {"trees": [5, 15]}, n_inner=3), x, y)
    assert fit.model.chosen["trees"] in (5, 15)
    assert len(fit.model.table) == 2
    assert fit.predict(x).shape == y.values.shape


def test_a_run_records_what_each_tuned_candidate_chose():
    x, y, readings, _ = tune_data()
    targets = {"plot": list(y.units), "sp1": y.values[:, 0].tolist(),
               "sp2": y.values[:, 1].tolist()}
    plain = k_learner()
    plain = Learner(name="plain", multi=plain.multi, params=plain.params, fit=plain.fit,
                    predict=plain.predict)
    run = timesift(targets, readings, y=["sp1", "sp2"], id="plot", time="time", x="value",
                   learners=[tune(k_learner(), {"k": [1, 2, 3, 4, 5]}, n_inner=3), plain],
                   sift=grains("month"), resampling=fold_map(y, v=3, seed=2), n_inner=None,
                   ensemble=False, verbose=False)
    by_learner = dict(zip(run.candidates["learner"], run.candidates["settings"]))
    assert by_learner["kl"] == "k = 3" and by_learner["plain"] == ""


def test_a_grid_says_what_it_cannot_search():
    with pytest.raises(ValueError, match="carries no setting called nope"):
        tune(k_learner(), {"nope": [1, 2]})
    with pytest.raises(ValueError, match="no value for k"):
        tune(k_learner(), {"k": []})
    with pytest.raises(ValueError, match="dict"):
        tune(k_learner(), [1, 2, 3])
    with pytest.raises(ValueError, match="2 or more"):
        tune(k_learner(), {"k": [1, 2]}, n_inner=1)


def test_a_learner_with_no_grid_of_its_own_is_searched_over_the_one_registered_for_it():
    from timesift import (boosting, discriminant, elasticnet, envelope, mars, mlp, tunings)
    from timesift.learners import flatten
    from timesift.tune import _registered_grid, default_grids

    x, y, _, _ = tune_data()
    columns = flatten(x).shape[1]
    assert set(default_grids()) <= set(tunings())
    assert _registered_grid(forest(), x)["mtry"] == list(range(1, min(10, columns) + 1))
    fit = fit_learner(tune(forest(trees=10), n_inner=3), x, y)
    assert len(fit.model.table) == min(10, columns)

    assert _registered_grid(boosting(), x) == {"trees": [500, 1000, 2500], "depth": [2, 5, 8],
                                               "shrinkage": [0.001, 0.01, 0.1]}
    xgb = _registered_grid(boosting(method="xgboost"), x)
    assert xgb["shrinkage"] == [0.3, 0.4] and xgb["colsample"] == [0.6, 0.8]
    assert _registered_grid(mars(), x)["nprune"] == list(range(2, max(21, 2 * columns + 1) + 1))
    assert _registered_grid(envelope(), x)["quantile"] == [0.0, 0.0125, 0.025, 0.05, 0.1]
    labels = [r["settings"] for r in fit_learner(tune(envelope(), n_inner=3), x, y).model.table]
    assert labels == [f"quantile = {q}" for q in ("0", "0.0125", "0.025", "0.05", "0.1")]
    assert _registered_grid(mlp(), x)["hidden"] == [[2], [4], [6], [8]]
    assert _registered_grid(discriminant(), x) == {"degree": [1, 2]}
    with pytest.raises(ValueError, match="no grid is registered for the elasticnet learner"):
        tune(elasticnet())


def test_a_grid_can_be_registered_for_a_learner_of_ones_own_as_a_dict_or_as_a_function():
    from timesift import register_tuning
    from timesift.registry import TUNINGS

    x, y, _, _ = tune_data()
    register_tuning("kl", {"k": [1, 2, 3, 4, 5]}, overwrite=True)
    try:
        assert fit_learner(tune(k_learner(), n_inner=3), x, y).model.chosen == {"k": 3}
        register_tuning("kl", lambda learner, x: {"k": [2, 3]}, overwrite=True)
        assert len(fit_learner(tune(k_learner(), n_inner=3), x, y).model.table) == 2
        with pytest.raises(ValueError, match="dict of values"):
            register_tuning("kl", [1, 2], overwrite=True)
        with pytest.raises(ValueError, match="already registered"):
            register_tuning("kl", {"k": [1]})
    finally:
        TUNINGS.remove("kl")
