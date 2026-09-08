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
from .control import TrainControl, train_control
from .digest import digest_array
from .fit import Timesift, timesift
from .ladder import (Ladder, grain_ladder, implied_skill, paired_contrast,
                     score_predictions, tss_inflation)
from .learners import (Fit, Learner, cnn, elasticnet, fit_learner, flatten, forest, mlp,
                       rescnn, stepwise)
from .metrics import (cohen_kappa, decision_threshold, kappa_score, model_agreement,
                      roc_auc, tss)
from .occlusion import feature_matrix
from .registry import (get_learner, learners, metrics, register_learner, register_metric,
                       register_response, resolve_metric, responses)
from .report import (candidate_table, ensemble_row, ensemble_weights, occlusion,
                     summary)
from .representation import (DAY_LEVEL_STATS, GRAINS, STATS, Coverage, TimesiftMatrix,
                             TimesiftSet, bind_channels, calendar_channels, coverage,
                             grain_matrix, lookback_matrix, timesift_set)
from .response import (PRESENCE_ABSENCE, Cells, Folds, Response, align_folds, as_response,
                       fold_map, positive_weights, scorable_cells)
from .select import column_names, select_columns
from .selection import Selection, select_grain
from .specs import (Representation, Resampling, Sift, TimesiftSpec, as_resampling, as_sift,
                    auto_grains, build_representation, cv, expand_sift, grain, grains,
                    grouped_cv, lookback, lookbacks, multigrain, n_targets, native,
                    resolve_folds, target_labels)
from .stack import EnsembleSpec, Stack, ensemble, ensemble_combine, ensemble_fit

# The version the wheel was built with, which `pyproject.toml` reads from `DESCRIPTION`. It is
# not written here as well: two literals are two versions the moment one bump misses one.
__version__ = _installed_version("timesift")

# The learners, response heads and metrics that ship are registered here, through the same public
# calls a user registers their own with. There is no second, privileged path into the registries.
# The R package does the same at load, in `zzz.R`.
register_metric("tss", tss)
register_metric("roc_auc", roc_auc)
register_metric("kappa", lambda y, p: kappa_score(y, p, "prevalence"))
register_metric("kappa_youden", lambda y, p: kappa_score(y, p, "youden"))

register_response("presence_absence", PRESENCE_ABSENCE)

register_learner("elasticnet", elasticnet)
register_learner("stepwise", stepwise)
register_learner("forest", forest)
register_learner("mlp", mlp)
register_learner("cnn", cnn)
register_learner("rescnn", rescnn)

__all__ = [
    "Cells", "Coverage", "DAY_LEVEL_STATS", "EnsembleSpec", "Fit", "Folds",
    "GRAINS",
    "Ladder", "Learner", "PRESENCE_ABSENCE", "Representation", "Resampling", "Response", "STATS",
    "Selection", "Sift", "Stack", "Timesift", "TimesiftMatrix", "TimesiftSet", "TimesiftSpec",
    "TrainControl", "align_folds", "as_resampling", "as_response", "as_sift", "auto_grains",
    "bind_channels", "build_representation", "calendar_channels", "candidate_table", "cnn",
    "cohen_kappa", "column_names", "coverage", "cv", "decision_threshold", "digest_array",
    "elasticnet",
    "ensemble", "ensemble_combine", "ensemble_fit", "ensemble_row", "ensemble_weights",
    "expand_sift", "feature_matrix", "fit_learner", "flatten", "fold_map", "forest",
    "get_learner", "grain",
    "grain_ladder", "grain_matrix", "grains", "grouped_cv", "implied_skill", "kappa_score",
    "learners", "lookback", "lookback_matrix", "lookbacks", "metrics", "mlp", "model_agreement",
    "multigrain", "n_targets", "native", "occlusion", "paired_contrast", "positive_weights",
    "read_cells",
    "read_folds", "read_response", "register_learner", "register_metric", "register_response",
    "rescnn", "resolve_folds", "resolve_metric", "responses", "roc_auc", "scorable_cells",
    "score_predictions", "select_columns",
    "select_grain", "stepwise", "summary", "target_labels", "timesift", "timesift_set",
    "train_control", "tss", "tss_inflation", "write_cells", "write_folds", "write_response",
]
