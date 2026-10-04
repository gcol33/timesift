"""timesift: learn predictive representations of time-varying data.

A target row, a series belonging to it, a representation of that series and a learner is the whole
contract; species distribution modelling from microclimate loggers is one application of it.

The representation answers to ``inst/spec/representation.md``, the same document the R package
answers to, and the test suite asserts the digests in ``inst/spec/fixtures/``. Where this
implementation and that document disagree, the document is right.

That document's last section says what each language carries, so a difference between the two is a
decision recorded there rather than something to be discovered at the call site.
"""

from importlib.metadata import version as _installed_version

from .artifacts import (read_cells, read_folds, read_response, write_cells,
                        write_folds, write_response)
from .contrasts import grain_contrasts
from .control import TrainControl, train_control
from .curves import ResponseCurve, response_curve
from .digest import digest_array
from .fit import Timesift, timesift
from .ladder import (Ladder, grain_ladder, implied_skill, paired_contrast,
                     score_predictions, tss_inflation)
from .learners import (Fit, Learner, additive, boosting, cnn, discriminant, elasticnet, envelope,
                       fit_learner, flatten, forest, hierarchical, linear, mars, maxent, mlp, perceptron,
                       rescnn, tree)
from .metrics import (ORDINAL_METRICS, TABLE_METRICS, average_precision, boyce_index,
                      cohen_kappa, decision_threshold, kappa_score, model_agreement,
                      ordinal_metric, regression_metric, roc_auc, table_metric, tss)
from .occlusion import feature_matrix
from .pseudo import pseudo_absences
from .raster import RangeChange, project, range_change
from .registry import (get_learner, learners, metrics, register_learner, register_metric,
                       register_response, register_tuning, resolve_metric, responses, tunings)
from .report import (candidate_table, ensemble_weights, occlusion, procedure_table,
                     summary)
from .representation import (DAY_LEVEL_STATS, GRAINS, STATS, Coverage, TimesiftMatrix,
                             TimesiftSet, bind_channels, calendar_channels, coverage,
                             grain_matrix, lookback_matrix, timesift_set)
from .response import (ABUNDANCE, CONTINUOUS, COUNT, ORDINAL, PRESENCE_ABSENCE, Cells, Folds, Response, align_folds, as_response,
                       fold_map, positive_weights, scorable_cells)
from .plot import plot
from .select import column_names, select_columns
from .selection import Selection, select_grain
from .simulate import Simulation, simulate_records
from .specs import (Representation, Resampling, Sift, TimesiftSpec, as_resampling, as_sift,
                    auto_grains, block_cv, build_representation, cv, env_cv, expand_sift, grain,
                    grains, grouped_cv, lookback, lookbacks, multigrain, n_targets, native,
                    resolve_folds, target_labels)
from .tune import Tuned, default_grids, tune
from .stack import (SPREAD_STATISTICS, EnsembleSpec, Stack, ensemble, ensemble_combine,
                    ensemble_fit, ensemble_spread)

# The version the wheel was built with, which `pyproject.toml` reads from `DESCRIPTION`. It is
# not written here as well: two literals are two versions the moment one bump misses one.
__version__ = _installed_version("timesift")

# The learners, response heads and metrics that ship are registered here, through the same public
# calls a user registers their own with. There is no second, privileged path into the registries.
# The R package does the same at load, in `zzz.R`.
register_metric("tss", tss)
register_metric("roc_auc", roc_auc)
register_metric("average_precision", average_precision)
register_metric("kappa", lambda y, p: kappa_score(y, p, "prevalence"))
register_metric("kappa_youden", lambda y, p: kappa_score(y, p, "youden"))
register_metric("boyce", boyce_index)
for _name in TABLE_METRICS:
    register_metric(_name, lambda y, p, _name=_name: table_metric(y, p, _name))
for _name in ("r_squared", "pearson"):
    register_metric(_name, lambda y, p, _name=_name: regression_metric(y, p, _name))
for _name in ("rmse", "mse", "mae", "max_error", "poisson_deviance"):
    register_metric("neg_" + _name, lambda y, p, _name=_name: -regression_metric(y, p, _name))
for _name in ORDINAL_METRICS:
    register_metric("ordinal_" + _name, lambda y, p, _name=_name: ordinal_metric(y, p, _name))

register_response("presence_absence", PRESENCE_ABSENCE)
register_response("continuous", CONTINUOUS)
register_response("abundance", ABUNDANCE)
register_response("ordinal", ORDINAL)
register_response("count", COUNT)

for _name, _grid in default_grids().items():
    register_tuning(_name, _grid)

register_learner("elasticnet", elasticnet)
register_learner("linear", linear)
register_learner("forest", forest)
register_learner("tree", tree)
register_learner("boosting", boosting)
register_learner("maxent", maxent)
register_learner("envelope", envelope)
register_learner("mars", mars)
register_learner("discriminant", discriminant)
register_learner("additive", additive)
register_learner("perceptron", perceptron)
register_learner("hierarchical", hierarchical)
register_learner("mlp", mlp)
register_learner("cnn", cnn)
register_learner("rescnn", rescnn)

__all__ = [
    "Cells", "Coverage", "DAY_LEVEL_STATS", "EnsembleSpec", "Fit", "Folds",
    "GRAINS",
    "ABUNDANCE", "CONTINUOUS", "COUNT", "Ladder", "Learner", "ORDINAL", "PRESENCE_ABSENCE", "RangeChange", "ResponseCurve", "Representation", "Resampling", "Response",
    "SPREAD_STATISTICS", "STATS",
    "Selection", "Sift", "Simulation", "Stack", "Timesift", "TimesiftMatrix", "Tuned", "TimesiftSet", "TimesiftSpec",
    "TrainControl", "additive", "align_folds", "as_resampling", "as_response", "as_sift", "auto_grains",
    "average_precision", "boyce_index", "ordinal_metric", "regression_metric",
    "bind_channels", "block_cv", "boosting", "build_representation", "calendar_channels", "candidate_table", "cnn",
    "cohen_kappa", "column_names", "coverage", "cv", "decision_threshold", "digest_array", "discriminant",
    "elasticnet", "envelope", "env_cv",
    "ensemble", "ensemble_combine", "ensemble_fit", "ensemble_spread", "ensemble_weights",
    "expand_sift", "feature_matrix", "fit_learner", "flatten", "fold_map", "forest",
    "get_learner", "grain", "grain_contrasts",
    "grain_ladder", "grain_matrix", "grains", "grouped_cv", "hierarchical", "implied_skill",
    "kappa_score",
    "learners", "linear", "lookback", "lookback_matrix", "lookbacks", "mars", "maxent", "metrics", "mlp",
    "model_agreement",
    "multigrain", "n_targets", "native", "occlusion", "paired_contrast", "perceptron", "plot", "positive_weights", "project", "pseudo_absences",
    "procedure_table",
    "range_change", "read_cells",
    "read_folds", "read_response", "register_learner", "register_metric", "register_response", "register_tuning",
    "rescnn", "resolve_folds", "response_curve", "resolve_metric", "responses", "roc_auc", "scorable_cells",
    "score_predictions", "select_columns", "table_metric",
    "select_grain", "simulate_records", "summary", "target_labels", "timesift", "timesift_set",
    "train_control", "tree", "tss", "tune", "tss_inflation", "tunings", "write_cells", "write_folds", "write_response",
]
