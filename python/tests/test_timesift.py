"""The entry point: what it refuses, what it fits, and what the fitted object carries.

Every candidate in one fit is scored on one fold map and one mask of cells, so what is asserted
here is mostly that: the same cells, the same rows, one prediction matrix per candidate whichever
way its learner covered the responses.
"""

from __future__ import annotations

import numpy as np
import pytest

from timesift.fit import SEPARATOR, timesift
from timesift.learners import Learner, flatten
from timesift.specs import (grain, grains, grouped_cv, lookback, lookbacks, multigrain,
                            native)

PLOTS = tuple(f"p{i:02d}" for i in range(1, 13))
START = np.datetime64("2021-09-01T00:00:00", "s")
FOLDS = [1, 2, 3] * 4


def series(days: int = 45, plots=PLOTS, columns=("temp",)) -> dict:
    t = START + np.arange(days * 24) * np.timedelta64(3600, "s")
    out = {"plot": np.repeat(np.asarray(plots), len(t)), "when": np.tile(t, len(plots))}
    for k, name in enumerate(columns):
        out[name] = np.concatenate([np.sin(np.arange(len(t)) / 13.0) + i + 10 * k
                                    for i in range(len(plots))])
    return out


def targets(plots=PLOTS) -> dict:
    n = len(plots)
    return {"plot": list(plots),
            "sp_a": [i % 2 for i in range(n)],
            "sp_b": [(i // 2) % 2 for i in range(n)],
            "sp_c": [(i // 3) % 2 for i in range(n)],
            "elevation": [1000.0 + 10 * i for i in range(n)],
            "aspect": [1.0 * (i % 4) for i in range(n)]}


# ---- a learner of the shape the fitting layer is written against -------------------------------

def learner(name="stub", reads="tabular", multi="separate", data=None) -> Learner:
    return Learner(name=name, fit=_fit, predict=_predict, data=data, reads=reads, multi=multi)


def _summary(x) -> np.ndarray:
    return flatten(x).mean(axis=1)


def _fit(x, y, control=None, **_):
    d = np.column_stack([np.ones(x.values.shape[0]), _summary(x)])
    return dict(beta=np.linalg.lstsq(d, y, rcond=None)[0], control=control,
                bins=x.values.shape[1], channels=x.values.shape[2])


def _predict(model, x):
    d = np.column_stack([np.ones(x.values.shape[0]), _summary(x)])
    return 1.0 / (1.0 + np.exp(-(d @ model["beta"])))


def fitted(**given):
    settings = dict(y="sp_*", id="plot", time="when", models=[learner()],
                    sift=grains("day", "week"), resampling=FOLDS, ensemble=False, verbose=False)
    settings.update(given)
    return timesift(targets(), series(), **settings)


# ---- what a fit produces -------------------------------------------------------------------------

def test_one_candidate_per_representation_and_learner():
    fit = fitted()
    assert list(fit.candidates["candidate"]) == [f"stub{SEPARATOR}day", f"stub{SEPARATOR}week"]
    assert list(fit.candidates["bins"]) == [45, 7]
    assert list(fit.candidates["multi"]) == ["separate", "separate"]
    assert set(fit.oof) == set(fit.models) == set(fit.candidates["candidate"])
    assert fit.representations["day"].values.shape == (12, 45, 1)


def test_every_candidate_emits_a_prediction_for_every_target_and_response():
    fit = fitted()
    for name, p in fit.oof.items():
        assert p.shape == (12, 3), name
        assert np.isfinite(p).all(), name


def test_every_candidate_is_scored_on_the_same_cells():
    fit = fitted(models=[learner("a"), learner("b")])
    columns = fit.scores
    cells = {}
    for name, variable, fold, ok in zip(columns["candidate"], columns["variable"],
                                        columns["fold"], columns["scorable"]):
        cells.setdefault(name, set()).add((str(variable), int(fold), bool(ok)))
    assert len(cells) == 4
    assert len(set(map(frozenset, cells.values()))) == 1


def test_the_response_and_the_folds_are_in_the_targets_own_row_order():
    fit = fitted()
    assert fit.y.units == PLOTS
    assert fit.folds.units == PLOTS
    assert fit.folds.fold.tolist() == FOLDS


def test_a_separate_learner_and_a_joint_one_emit_the_same_shaped_matrix():
    """Both are handed the whole response matrix and both emit one column per response.

    A learner that fits one model per response does that inside its own fit, so the block of
    predictors is built once for the fit rather than once for every response of it.
    """
    fit = fitted(models=[learner("apart", multi="separate"), learner("together", multi="joint")])
    apart = fit.models[f"apart{SEPARATOR}day"]
    together = fit.models[f"together{SEPARATOR}day"]
    assert apart.variables == together.variables == ("sp_a", "sp_b", "sp_c")
    assert apart.predict(fit.representations["day"]).shape == (12, 3)
    assert together.predict(fit.representations["day"]).shape == (12, 3)


def test_the_predictor_block_is_built_once_for_a_fit_rather_than_once_per_response():
    seen = []

    def fit(x, y, **_):
        seen.append(x.values.shape)
        return dict(rate=y.mean(axis=0))

    counting = Learner(name="counting", multi="separate", fit=fit,
                       predict=lambda model, x: np.tile(model["rate"], (x.values.shape[0], 1)))
    run = fitted(models=[counting], sift=grains("day"))
    # Three folds and the refit on all targets, whatever the number of responses. Fitting per
    # response inside the layer instead would build the flattened block once for every one of
    # them, which on the published grid is 101 copies of a 47 MB matrix per fold and per arm.
    assert len(seen) == 4
    assert len(run.y.variables) > 1


def test_the_models_kept_per_fold_are_keyed_by_candidate_and_fold():
    fit = fitted(keep_fits=True)
    assert set(fit.fits) == {f"stub{SEPARATOR}{w}|{k}" for w in ("day", "week") for k in (1, 2, 3)}
    assert not fitted().fits


def test_the_control_reaches_the_learner():
    from timesift.control import train_control

    control = train_control(epochs=3)
    fit = fitted(control=control)
    assert fit.models[f"stub{SEPARATOR}day"].model["control"] is control
    assert fitted().models[f"stub{SEPARATOR}day"].model["control"] is None


# ---- the rules the entry point enforces ----------------------------------------------------------

def test_static_is_never_implicit():
    plain = fitted()
    assert plain.representations["day"].values.shape[2] == 1
    named = fitted(static=["elevation", "aspect"])
    assert named.representations["day"].stats == ("mean", "elevation", "aspect")


def test_a_response_column_cannot_also_be_a_static_predictor():
    with pytest.raises(ValueError, match="cannot be a predictor"):
        fitted(static="sp_a")


def test_one_target_row_per_id_unless_the_targets_are_anchored_in_time():
    t = targets()
    t["plot"] = list(PLOTS[:11]) + ["p01"]
    with pytest.raises(ValueError, match="more than one row of `targets`: p01"):
        timesift(t, series(), y="sp_*", id="plot", time="when", models=[learner()],
                 sift=grains("day"), resampling=FOLDS, ensemble=False, verbose=False)


def test_a_calendar_grain_is_refused_where_the_targets_are_anchored_in_time():
    t = repeated_targets()
    with pytest.raises(ValueError, match="must be anchored on the target"):
        timesift(t, series(), y="sp_*", id="plot", time="when", target_time="visit",
                 models=[learner()], sift=grains("day"), resampling=[1, 2, 3] * 4,
                 ensemble=False, verbose=False)


def test_there_is_no_default_set_of_spans():
    with pytest.raises(ValueError, match="no defensible default set of spans"):
        timesift(repeated_targets(), series(), y="sp_*", id="plot", time="when",
                 target_time="visit", models=[learner()], resampling=[1, 2, 3] * 4,
                 ensemble=False, verbose=False)


def test_a_lookback_needs_the_column_holding_each_targets_instant():
    with pytest.raises(ValueError, match="`target_time` has to name the column"):
        fitted(sift=lookbacks("10 days"))

    # The same check sees a representation a learner pinned itself to, which no sift carries.
    with pytest.raises(ValueError, match="`target_time` has to name the column"):
        fitted(models=[learner(data=lookback("10 days"))], sift=grains("day"))


def test_a_lookback_fits_on_repeated_targets_grouped_by_their_unit():
    t = repeated_targets()
    fit = timesift(t, series(), y="sp_*", id="plot", time="when", target_time="visit",
                   models=[learner()], sift=lookbacks("10 days", "20 days"),
                   resampling=grouped_cv("plot", v=3, seed=2), ensemble=False, verbose=False)
    assert fit.y.units == tuple(str(i + 1) for i in range(12))
    assert list(fit.candidates["candidate"]) == [f"stub{SEPARATOR}10 days",
                                                 f"stub{SEPARATOR}20 days"]
    assigned = {}
    for plot, k in zip(t["plot"], fit.folds.fold):
        assigned.setdefault(plot, set()).add(int(k))
    assert all(len(k) == 1 for k in assigned.values())


def test_without_a_series_static_is_the_whole_predictor_block_and_the_sift_is_ignored():
    fit = timesift(targets(), y="sp_*", id="plot", static=["elevation", "aspect"],
                   models=[learner()], sift=grains("day"), resampling=FOLDS, ensemble=False,
                   verbose=False)
    assert list(fit.representations) == ["static"]
    assert fit.representations["static"].values.shape == (12, 1, 2)
    with pytest.raises(ValueError, match="nothing to predict from"):
        timesift(targets(), y="sp_*", id="plot", models=[learner()], resampling=FOLDS,
                 ensemble=False, verbose=False)


def test_the_columns_of_each_table_are_named_where_they_are_wrong():
    with pytest.raises(ValueError, match="`id` names plt, which is not a column of `targets`"):
        fitted(id="plt")
    with pytest.raises(ValueError, match="`y` names no column of `targets`"):
        fitted(y="none_*")
    with pytest.raises(ValueError, match="`time` names the column of reading instants"):
        timesift(targets(), series(), y="sp_*", id="plot", models=[learner()], sift=grains("day"),
                 resampling=FOLDS, ensemble=False, verbose=False)


# ---- what can read what --------------------------------------------------------------------------

def test_a_tabular_learner_is_refused_the_unreduced_record():
    fit = fitted(sift=grains("native", "week"))
    reason = dict(zip(fit.candidates["candidate"], fit.candidates["reason"]))
    assert "one column per reading" in reason[f"stub{SEPARATOR}native"]
    assert reason[f"stub{SEPARATOR}week"] == ""
    assert set(fit.oof) == {f"stub{SEPARATOR}week"}
    assert "native" not in fit.representations


def test_the_same_pair_named_through_data_is_an_error_rather_than_a_skip():
    with pytest.raises(ValueError, match="one column per reading"):
        fitted(models=[learner(data=native())])


def test_a_sequence_learner_is_refused_a_block_of_one_bin():
    fit = fitted(models=[learner("seq", reads="sequence"), learner("flat")],
                 sift=[multigrain(grains=("day", "week")), grain("week")])
    reason = dict(zip(fit.candidates["candidate"], fit.candidates["reason"]))
    assert "gives one row of" in reason[f"seq{SEPARATOR}multigrain(day+week)"]
    assert reason[f"seq{SEPARATOR}week"] == ""
    assert reason[f"flat{SEPARATOR}multigrain(day+week)"] == ""


def test_a_learner_pinned_to_a_representation_runs_on_that_one_alone():
    fit = fitted(models=[learner("pinned", data=grain("month")), learner("free")],
                 sift=grains("day", "week"))
    assert list(fit.candidates["candidate"]) == [f"pinned{SEPARATOR}month",
                                                 f"free{SEPARATOR}day", f"free{SEPARATOR}week"]
    assert fit.representations["month"].values.shape[1] == 2


def test_a_fit_with_nothing_left_to_read_says_so():
    with pytest.raises(ValueError, match="no learner can read any of the representations"):
        fitted(sift=grains("native"))


# ---- predicting new targets ----------------------------------------------------------------------

def test_predict_rebuilds_the_representation_for_new_targets():
    fit = fitted()
    later = {k: v[:4] for k, v in targets().items()}
    p = fit.predict(later, series(plots=PLOTS[:4]), candidate=f"stub{SEPARATOR}week")
    assert p.shape == (4, 3)
    assert np.isfinite(p).all()
    with pytest.raises(KeyError, match="no candidate called"):
        fit.predict(later, series(plots=PLOTS[:4]), candidate="nobody")
    with pytest.raises(ValueError, match="no ensemble"):
        fit.predict(later, series(plots=PLOTS[:4]))


def test_the_representation_a_candidate_reads_is_read_off_the_candidates_table():
    fit = fitted()
    assert fit.representation_of(f"stub{SEPARATOR}week") == "week"
    with pytest.raises(KeyError, match="no candidate called"):
        fit.representation_of("nobody")


# ---- the ensemble --------------------------------------------------------------------------------

def test_one_candidate_leaves_no_ensemble_to_fit():
    fit = fitted(sift=grains("week"), ensemble=True)
    assert fit.stack is None and fit.weights is None


def test_the_ensemble_is_fitted_on_the_out_of_fold_predictions_and_predicts_through_the_refits():
    fit = fitted(models=[learner("a"), learner("b", multi="joint")], ensemble=True)
    assert fit.stack is not None
    assert set(fit.weights) <= set(fit.oof)
    assert sum(fit.weights.values()) == pytest.approx(1.0)
    later = {k: v[:4] for k, v in targets().items()}
    assert fit.predict(later, series(plots=PLOTS[:4])).shape == (4, 3)


def test_a_learner_that_declares_no_control_is_not_handed_the_runs():
    from timesift.control import train_control

    def two_arguments(x, y):
        return dict(beta=np.zeros((2, 3)))

    plain = Learner(name="plain", fit=two_arguments, predict=_predict, reads="tabular",
                    multi="joint")
    fit = fitted(models=[plain], sift=grains("week"), control=train_control(epochs=2))
    assert set(fit.oof) == {f"plain{SEPARATOR}week"}


def test_the_combiner_minimises_the_loss_of_the_head_the_run_was_fitted_under(temporary_response,
                                                                              monkeypatch):
    from timesift import stack as stack_module
    from timesift.response import scorable_cells
    from timesift.stack import ensemble
    temporary_response("gauss_test", dict(
        prepare=lambda r: r, activation="identity", loss="squared_error", metric="roc_auc",
        cells=lambda r, folds: scorable_cells(r, folds)))
    seen = []
    minimised = stack_module.stack_loss

    def record(name):
        seen.append(name)
        return minimised(name)

    monkeypatch.setattr(stack_module, "stack_loss", record)
    two = [learner("a"), learner("b", multi="joint")]

    fitted(models=two, ensemble=True, response="gauss_test")
    assert seen == ["gauss_test"]
    fitted(models=two, ensemble=True)
    assert seen[-1] == "presence_absence"
    # Naming the run's own head is the same thing said twice, and is not a contradiction.
    fitted(models=two, ensemble=ensemble(response="gauss_test"), response="gauss_test")
    assert seen[-1] == "gauss_test"
    with pytest.raises(ValueError, match="names another"):
        fitted(models=two, ensemble=ensemble(response="presence_absence"), response="gauss_test")


def test_the_summary_reads_the_fit_and_lists_what_could_not_be_paired():
    fit = fitted(sift=grains("native", "day", "week"), ensemble=True)
    text = repr(fit)
    assert "not applicable" in text
    assert f"stub{SEPARATOR}week" in text
    assert "ensemble" in text


def test_a_metric_given_as_a_function_scores_the_run_and_everything_that_rescores_it():
    from timesift.metrics import tss
    from timesift.selection import select_grain
    from timesift.representation import grain_matrix
    from timesift.response import Response

    settings = dict(models=[learner("a"), learner("b")], sift=grains("day", "week"),
                    ensemble=True)
    fit = fitted(metric=lambda y, p: tss(y, p), **settings)
    by_name = fitted(metric="tss", **settings)

    # The function is what scores, so every number is the registered metric's; only the name
    # differs, and it is a name rather than whatever this language calls an anonymous function.
    np.testing.assert_array_equal(fit.scores["score"], by_name.scores["score"])
    assert fit.metric == "<function>"
    assert fit.stack is not None
    assert repr(fit).splitlines() == [line.replace(", tss", ", <function>")
                                      for line in repr(by_name).splitlines()]

    # The selection reports the estimate under every registered metric, so its own has to be one.
    t = targets()
    x = grain_matrix(series(), "plot", "when", "temp", grain="week")
    y = Response(np.column_stack([t["sp_a"], t["sp_b"]]).astype(float), tuple(t["plot"]),
                 ("sp_a", "sp_b"))
    with pytest.raises(TypeError, match="has to be registered"):
        select_grain(x, y, [learner("a")], metric=tss)
    with pytest.raises(TypeError, match="name of a registered metric or a function"):
        fitted(metric=3)


def repeated_targets() -> dict:
    plots = [p for p in PLOTS[:6] for _ in (0, 1)]
    visit = [START + np.timedelta64(d * 86400, "s") for d in (20, 40) for _ in range(6)]
    return {"plot": plots, "visit": [visit[i % 2 * 6] for i in range(12)],
            "sp_a": [i % 2 for i in range(12)], "sp_b": [(i // 2) % 2 for i in range(12)],
            "sp_c": [(i // 3) % 2 for i in range(12)]}
