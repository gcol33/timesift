from __future__ import annotations

import numpy as np
import pytest

import timesift as ts
from timesift import (boosting, discriminant, elasticnet, ensemble, envelope, fit_learner, forest,
                      grains, mars, maxnet, ordinal_metric, regression_metric, stepwise, timesift,
                      tree)
from timesift.registry import RESPONSES
from timesift.representation import grain_matrix
from timesift.response import Folds, Response, fold_map, numeric_cells


def numeric_data(n=50, seed=5):
    rng = np.random.default_rng(seed)
    t = np.datetime64("2021-09-01T00:00:00", "s") + np.arange(24 * 90) * np.timedelta64(1, "h")
    units = [f"p{i:02d}" for i in range(n)]
    warmth = rng.normal(size=n)
    value = np.concatenate([w + rng.normal(0, 0.5, len(t)) for w in warmth])
    readings = {"plot": [u for u in units for _ in range(len(t))], "time": list(t) * n,
                "value": list(value)}
    x = grain_matrix({"id": readings["plot"], "time": readings["time"], "value": readings["value"]},
                     "id", "time", "value", grain="week")
    height = 10 + 3 * warmth + rng.normal(0, 0.3, n)
    return x, readings, units, height, Response(height[:, None], tuple(units), ("height",))


def test_the_heads_and_metrics_of_a_numeric_response_are_registered():
    assert {"continuous", "abundance", "ordinal", "count"} <= set(ts.responses())
    assert {"r_squared", "pearson", "neg_rmse", "neg_mse", "neg_mae", "neg_max_error",
            "neg_poisson_deviance", "ordinal_accuracy", "ordinal_recall", "ordinal_precision",
            "ordinal_f1"} <= set(ts.metrics())
    for h, metric in (("continuous", "r_squared"), ("abundance", "r_squared"),
                      ("ordinal", "ordinal_f1")):
        head = RESPONSES.get(h)
        assert head["metric"] == metric and head["loss"] == "squared_error"
        assert head["activation"] == "identity"


def test_each_statistic_of_a_numeric_response_reads_the_errors():
    y, p = [1, 2, 3, 4], [1.5, 2.5, 3.5, 4.5]
    assert regression_metric(y, p, "rmse") == pytest.approx(0.5)
    assert regression_metric(y, p, "mse") == pytest.approx(0.25)
    assert regression_metric(y, p, "mae") == pytest.approx(0.5)
    assert regression_metric(y, p, "max_error") == pytest.approx(0.5)
    assert regression_metric(y, p, "pearson") == pytest.approx(1)
    # The sum of squares about the mean is 5, and the squared errors sum to 1.
    assert regression_metric(y, p, "r_squared") == pytest.approx(1 - 1 / 5)
    assert np.isnan(regression_metric([3, 3, 3], [1, 2, 3], "r_squared"))
    assert np.isnan(regression_metric(y, [2, 2, 2, 2], "pearson"))
    assert np.isnan(regression_metric(y, [1, 2, np.nan, 4], "rmse"))
    with pytest.raises(ValueError, match="same length"):
        regression_metric(y, p[1:], "rmse")
    # The errors are registered with their sign reversed, so a higher score is a better one.
    assert ts.resolve_metric("neg_rmse")[0](y, p) == pytest.approx(-0.5)


def test_ordinal_classes_are_read_as_the_observed_class_nearest_to_the_prediction():
    y = [1, 1, 2, 2, 3, 3, 3]
    p = [1.2, 2.4, 2, 1.4, 2.8, 3.4, 2.2]
    # Observed 1, 1, 2, 2, 3, 3, 3 against read 1, 2, 2, 1, 3, 3, 2: four hits of seven, and the
    # classes are recalled 1/2, 1/2 and 2/3 and predicted rightly 1/2, 1/3 and 1.
    assert ordinal_metric(y, p, "accuracy") == pytest.approx(4 / 7)
    r, q = (1 / 2 + 1 / 2 + 2 / 3) / 3, (1 / 2 + 1 / 3 + 1) / 3
    assert ordinal_metric(y, p, "recall") == pytest.approx(r)
    assert ordinal_metric(y, p, "precision") == pytest.approx(q)
    assert ordinal_metric(y, p, "f1") == pytest.approx(2 * q * r / (q + r))
    # A prediction halfway between two classes is read as the lower one.
    assert ordinal_metric([1, 2, 3, 3], [1.5, 2.5, 3.5, 2.5], "accuracy") == pytest.approx(3 / 4)
    with pytest.raises(ValueError, match="metric must be one of"):
        ordinal_metric(y, p, "nope")


def test_a_numeric_cell_is_scorable_where_both_sides_of_the_split_hold_two_values():
    y = Response(np.column_stack([[1, 2, 3, 4, 5, 6], [1, 1, 1, 1, 2, 3]]).astype(float),
                 tuple(f"p{i}" for i in range(1, 7)), ("a", "b"))
    folds = Folds(fold=np.array([1, 1, 2, 2, 3, 3]), units=y.units)
    cells = numeric_cells(y, folds)
    assert list(cells.variable) == ["a"] * 3 + ["b"] * 3
    assert cells.scorable.tolist() == [True, True, True, False, False, False]
    assert cells.abs_test[3:].tolist() == [1, 1, 2]
    assert cells.abs_train[3:].tolist() == [3, 3, 1]


def test_a_head_refuses_a_response_it_cannot_hold():
    y = Response(np.array([[1.0], [2.0], [3.0], [4.0]]), ("a", "b", "c", "d"), ("a",))

    def with_values(v):
        return Response(np.asarray(v, dtype=float), y.units, y.variables)

    with pytest.raises(ValueError, match="not negative"):
        RESPONSES.get("abundance")["prepare"](with_values(-y.values))
    with pytest.raises(ValueError, match="whole-number"):
        RESPONSES.get("ordinal")["prepare"](with_values(y.values + 0.5))
    with pytest.raises(ValueError, match="finite"):
        RESPONSES.get("continuous")["prepare"](with_values([[1.0], [np.nan], [3.0], [4.0]]))
    assert RESPONSES.get("ordinal")["prepare"](y).values.tolist() == y.values.tolist()


def test_a_run_under_a_numeric_head_is_scored_by_that_heads_metric_and_stacked():
    x, readings, units, height, y = numeric_data()
    targets = {"plot": units, "height": height.tolist()}
    run = timesift(targets, readings, y="height", id="plot", time="time", x="value",
                   response="continuous",
                   models=[elasticnet(squares=False), forest(trees=30)], sift=grains("week"),
                   resampling=fold_map(y, v=4, seed=2), inner=None, ensemble=ensemble("stack"),
                   verbose=False)
    assert run.metric == "r_squared"
    scored = np.asarray(run.scores["score"], dtype=float)[np.asarray(run.scores["scorable"])]
    assert np.isfinite(scored).all() and scored.max() > 0.7
    p = run.predict(targets, readings)
    assert np.corrcoef(p[:, 0], height)[0, 1] > 0.9
    with pytest.raises(ValueError, match="presence-absence"):
        run.predict(targets, readings, type="binary")


def test_the_learners_that_need_presences_and_absences_refuse_a_numeric_head():
    x, _, _, height, y = numeric_data()
    for learner in (discriminant(), envelope(), maxnet()):
        with pytest.raises(ValueError, match="squared_error"):
            fit_learner(learner, x, y, response="continuous")
    for learner in (elasticnet(), forest(trees=20), boosting(trees=20), tree(), mars()):
        fit = fit_learner(learner, x, y, response="continuous")
        assert np.corrcoef(fit.predict(x)[:, 0], height)[0, 1] > 0.8, learner.name


# ---- a count response ------------------------------------------------------------------------

def count_data(n=80, seed=5):
    rng = np.random.default_rng(seed)
    t = np.datetime64("2021-09-01T00:00:00", "s") + np.arange(24 * 90) * np.timedelta64(1, "h")
    units = [f"p{i:02d}" for i in range(n)]
    warmth = rng.normal(size=n)
    value = np.concatenate([w + rng.normal(0, 0.5, len(t)) for w in warmth])
    readings = {"plot": [u for u in units for _ in range(len(t))], "time": list(t) * n,
                "value": list(value)}
    x = grain_matrix({"id": readings["plot"], "time": readings["time"], "value": readings["value"]},
                     "id", "time", "value", grain="week")
    rate = np.exp(0.8 + 0.7 * warmth)
    count = rng.poisson(rate).astype(float)
    return x, readings, units, rate, count, Response(count[:, None], tuple(units), ("count",))


def poisson_deviance_of(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        saturated = np.where(y > 0, y * np.log(y / p), 0.0)
    return float(np.mean(2 * (saturated - (y - p))))


def test_the_count_head_names_the_poisson_deviance_and_an_exponential_output():
    head = RESPONSES.get("count")
    assert head["loss"] == "poisson_deviance" and head["activation"] == "exp"
    assert head["metric"] == "neg_poisson_deviance"


def test_the_poisson_deviance_reads_counts_and_refuses_a_mean_outside_its_support():
    y = [0, 1, 2, 5, 3, 0]
    p = [0.5, 1.2, 1.8, 4.2, 3.5, 0.2]
    assert regression_metric(y, p, "poisson_deviance") == pytest.approx(poisson_deviance_of(y, p))
    assert regression_metric([1, 2, 3], [1, 2, 3], "poisson_deviance") == pytest.approx(0)
    assert regression_metric([0, 2], [0, 2], "poisson_deviance") == pytest.approx(0)
    assert np.isfinite(regression_metric([0, 0, 2, 1], [0, 0.4, 1.5, 1.5], "poisson_deviance"))
    assert np.isnan(regression_metric([0, 0, 2, 1], [0, 0.4, 0, 1.5], "poisson_deviance"))
    assert np.isnan(regression_metric([1, 2], [1, -1], "poisson_deviance"))
    assert (ts.resolve_metric("neg_poisson_deviance")[0](y, p)
            == pytest.approx(-poisson_deviance_of(y, p)))


def test_the_count_head_refuses_a_response_it_cannot_hold():
    def with_values(v):
        return Response(np.asarray(v, dtype=float), ("a", "b", "c"), ("a",))

    prepare = RESPONSES.get("count")["prepare"]
    with pytest.raises(ValueError, match="whole numbers of zero or more"):
        prepare(with_values([[1.0], [-2.0], [3.0]]))
    with pytest.raises(ValueError, match="whole numbers of zero or more"):
        prepare(with_values([[1.0], [2.5], [3.0]]))
    with pytest.raises(ValueError, match="finite"):
        prepare(with_values([[1.0], [np.nan], [3.0]]))
    assert prepare(with_values([[0.0], [1.0], [7.0]])).values.ravel().tolist() == [0.0, 1.0, 7.0]


def test_each_learner_that_fits_a_family_fits_a_count_under_the_poisson_one():
    x, _, _, rate, count, y = count_data()
    for learner in (elasticnet(), forest(trees=30), boosting(trees=30),
                    boosting(trees=30, newton=True, depth=2), tree(), stepwise(), mars(),
                    ts.additive(k=5)):
        fit = fit_learner(learner, x, y, response="count")
        p = fit.predict(x)[:, 0]
        assert (p > 0).all(), learner.name
        assert np.corrcoef(p, rate)[0, 1] > 0.5, learner.name
        assert poisson_deviance_of(count, p) < poisson_deviance_of(count, np.full(len(p), count.mean()))
    assert fit_learner(elasticnet(), x, y, response="count").model["models"][0]["family"] == "poisson"


def test_the_learners_that_need_presences_and_absences_refuse_a_count_head():
    x, _, _, _, _, y = count_data(n=40)
    for learner in (discriminant(), envelope(), maxnet()):
        with pytest.raises(ValueError, match="poisson_deviance"):
            fit_learner(learner, x, y, response="count")


def test_a_poisson_shrinkage_reaches_the_tree_only_through_a_count_head():
    x, _, _, _, _, y = count_data(n=60)
    tight = fit_learner(tree(shrink=0), x, y, response="count").predict(x)
    loose = fit_learner(tree(shrink=1), x, y, response="count").predict(x)
    assert not np.allclose(tight, loose)
    with pytest.raises(ValueError, match="zero or more"):
        tree(shrink=-1)


def test_a_run_under_the_count_head_is_scored_by_the_poisson_deviance_and_stacked():
    x, readings, units, rate, count, y = count_data()
    targets = {"plot": units, "count": count.tolist()}
    run = timesift(targets, readings, y="count", id="plot", time="time", x="value",
                   response="count",
                   models=[elasticnet(squares=False), forest(trees=30)], sift=grains("week"),
                   resampling=fold_map(y, v=4, seed=2), inner=None, ensemble=ensemble("stack"),
                   verbose=False)
    assert run.metric == "neg_poisson_deviance"
    scored = np.asarray(run.scores["score"], dtype=float)[np.asarray(run.scores["scorable"])]
    assert np.isfinite(scored).all() and (scored < 0).all()
    assert run.stack.loss == "poisson_deviance"
    assert sum(run.stack.weights.values()) == pytest.approx(1.0)
    p = run.predict(targets, readings)
    assert (p > 0).all() and np.corrcoef(p[:, 0], rate)[0, 1] > 0.5
