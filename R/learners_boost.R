#' Gradient boosted trees on the flattened representation
#'
#' One boosted model per response, over every bin-by-channel column of the representation: a
#' logistic model under a presence-absence head, a squared-error one under a head with a
#' squared-error loss and a Poisson one with a log link under a count head. The score starts at the
#' log-odds of the weighted share of presences, the weighted mean, or the log of the weighted mean
#' count, and each tree is fitted to the loss's gradient at the current score and added to
#' it scaled by `shrinkage`. Each tree is grown on a subsample of the units drawn without
#' replacement, and reads a subsample of the columns.
#'
#' `method` picks the trees. Under `"gbm"` they are Friedman's gradient boosting machine as the gbm
#' package grows it, which is what biomod2 fits as `GBM`: `depth` splits grown best first, each the one that most reduces the weighted squared error of the
#' working response with at least `min_leaf` units on each side, and a leaf that takes one Newton
#' step on the loss. Under `"xgboost"` they are XGBoost's exact greedy trees (Chen and Guestrin 2016),
#' which biomod2 fits as `XGBOOST`:
#' grown level by level to `depth`, each split chosen by the second-order gain under the L2 penalty
#' `lambda` with at least `min_leaf` of hessian on each side, pruned where a split gains less than
#' `gamma`, and a leaf the step `-G / (H + lambda)`. Either way `depth` is the order of interaction
#' a tree can hold. With `subsample = 1` the first-order fit is gbm's own to rounding, and the
#' second-order one xgboost's to its single-precision storage; the model is grown by the core the
#' Python package calls, so the two languages fit the same model.
#'
#' `n_inner` folds, when above zero, choose how many trees are kept: the fit is repeated on each
#' fold's complement, and the number of trees of least held-out deviance, summed over the folds and
#' weighted by how many units each holds, is kept, as gbm's `cv.folds` chooses it. The folds are
#' dealt for each response and stratified on it, as the elastic net's are.
#'
#' `preset` says whose defaults the settings left `NULL` take. `"default"` is the fitting
#' package's own, which is what biomod2's default option set fits: under gbm 100 trees of one split,
#' `shrinkage = 0.1`, `min_leaf = 10` and `subsample = 0.5`; under xgboost 100 trees of depth 6,
#' `shrinkage = 0.3`, `min_leaf = 1`, `lambda = 1` and every unit and column. `"bigboss"` is
#' biomod2's tuned option set: under gbm 2500 trees of seven splits, `shrinkage = 0.001`,
#' `min_leaf = 5`, `subsample = 0.5` and three inner folds; under xgboost four trees of depth 2 at
#' `shrinkage = 1`. A setting given explicitly beats either.
#'
#' The case weights are the response head's, [positive_weights()] under presence-absence, and
#' weigh the gradient and every sum a tree is grown on; `min_leaf` counts units under gbm, as
#' `n.minobsinnode` does.
#' Under the shipped presence-absence head those weights are on, so a default `boosting()` is
#' gbm's specification fitted under them; a head registered without `weights` fits it unweighted.
#'
#' @inheritParams elasticnet
#' @param trees Trees fitted.
#' @param method `"gbm"` for first-order trees grown best first, `"xgboost"` for second-order trees
#'   grown level by level.
#' @param depth Splits in a tree under `"gbm"`, its depth under `"xgboost"`.
#' @param shrinkage The scale each tree is added to the score at.
#' @param min_leaf Units each side of a split keeps under `"gbm"`, hessian under `"xgboost"`.
#' @param subsample Share of the units each tree is grown on.
#' @param colsample Share of the columns each tree reads.
#' @param lambda,gamma The L2 penalty on a leaf and the least gain of a split under `"xgboost"`;
#'   zero under `"gbm"`.
#' @param n_inner Folds of the inner cross-validation choosing how many trees are kept, or 0 to
#'   keep them all.
#' @param preset Whose defaults the settings left `NULL` take: `"default"` or `"bigboss"`.
#' @param seed Seed for the subsamples and the inner folds.
#' @param threads Fits of one response's inner cross-validation run at once. The model is the same
#'   on any number.
#'
#' @return A [learner()].
#'
#' @examples
#' boosting()
#' boosting(preset = "bigboss")
#' boosting(method = "xgboost", depth = 3L)
#'
#' @export
boosting <- function(data = NULL, method = c("gbm", "xgboost"), trees = NULL, depth = NULL,
                     shrinkage = NULL, min_leaf = NULL, subsample = NULL, colsample = NULL,
                     lambda = NULL, gamma = NULL, n_inner = NULL,
                     preset = c("default", "bigboss"), seed = 1L, threads = 1L) {
  method <- match.arg(method)
  preset <- match.arg(preset)
  settings <- .boost_settings(preset, method, trees, depth, shrinkage, min_leaf,
                              subsample, colsample, lambda, gamma, n_inner)
  learner(
    name = "boosting",
    data = data, reads = "tabular", multi = "separate",
    params = c(settings, list(method = method, seed = as.integer(seed),
                              threads = as.integer(threads))),
    fit = function(x, y, trees, depth, shrinkage, min_leaf, subsample, colsample, method, lambda,
                   gamma, n_inner, seed, threads, head, weights, group = NULL, ...) {
      family <- .head_family(head)
      m <- .flatten(x)
      seeds <- .variable_seeds(seed, y)
      models <- lapply(seq_len(ncol(y)), function(j) {
        yj <- y[, j]
        if (length(unique(yj)) < 2L) {
          return(mean(yj))
        }
        fold <- NULL
        n_fold <- 0L
        if (n_inner > 0L) {
          inner <- .inner_folds(yj, n_inner, seeds[j], group)
          labels <- sort(unique(inner))
          fold <- match(inner, labels) - 1L
          n_fold <- length(labels)
        }
        .boost_fit(m, yj, weights[, j], family, trees, depth, shrinkage, min_leaf, subsample,
                   colsample, method == "xgboost", lambda, gamma, seeds[j], fold, n_fold,
                   threads)
      })
      list(models = models, columns = colnames(m), family = family)
    },
    predict = function(model, x) {
      m <- .flatten(x)
      .as_predictions(vapply(model$models, function(f) {
        if (is.numeric(f)) rep(f, nrow(m)) else .boost_predict(f, m)
      }, numeric(nrow(m))), nrow(m))
    }
  )
}

# The settings boosted trees are fitted under: those given, and the preset's for the rest, which
# are gbm's or xgboost's as `method` picks. gbm's trees take no penalty and no least gain.
.boost_settings <- function(preset, method, trees, depth, shrinkage, min_leaf, subsample,
                            colsample, lambda, gamma, n_inner) {
  if (method == "gbm" && ((!is.null(lambda) && lambda != 0) || (!is.null(gamma) && gamma != 0))) {
    stop("`lambda` and `gamma` are the second-order trees' settings; ",
         "set `method = \"xgboost\"` to use them.", call. = FALSE)
  }
  base <- switch(
    paste(preset, method),
    "default gbm" = list(trees = 100L, depth = 1L, shrinkage = 0.1, min_leaf = 10,
                         subsample = 0.5, colsample = 1, lambda = 0, gamma = 0, n_inner = 0L),
    "bigboss gbm" = list(trees = 2500L, depth = 7L, shrinkage = 0.001, min_leaf = 5,
                         subsample = 0.5, colsample = 1, lambda = 0, gamma = 0, n_inner = 3L),
    "default xgboost" = list(trees = 100L, depth = 6L, shrinkage = 0.3, min_leaf = 1,
                             subsample = 1, colsample = 1, lambda = 1, gamma = 0, n_inner = 0L),
    "bigboss xgboost" = list(trees = 4L, depth = 2L, shrinkage = 1, min_leaf = 1, subsample = 1,
                             colsample = 1, lambda = 1, gamma = 0, n_inner = 0L)
  )
  given <- list(trees = trees, depth = depth, shrinkage = shrinkage, min_leaf = min_leaf,
                subsample = subsample, colsample = colsample, lambda = lambda, gamma = gamma,
                n_inner = n_inner)
  out <- Map(function(g, b) g %||% b, given, base)
  for (k in c("trees", "depth", "n_inner")) out[[k]] <- as.integer(out[[k]])
  for (k in c("shrinkage", "min_leaf", "subsample", "colsample", "lambda", "gamma")) {
    out[[k]] <- as.numeric(out[[k]])
  }
  out
}
