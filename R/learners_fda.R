#' Flexible discriminant analysis on the flattened representation
#'
#' One discriminant per response, over every bin-by-channel column of the representation, fitted as
#' mda's `fda(method = mars)` fits it and as biomod2 fits `FDA`. Optimal scoring gives presence and
#' absence one score each and regresses the scored response on a MARS basis of the columns; the
#' fitted score is the one canonical variate, and a unit's prediction is its posterior probability
#' of presence under two normal classes around the class centroids on that variate, with the
#' classes' shares among the fitting units as priors.
#'
#' The basis is mda's own MARS, which biomod2 reaches through `fda()` and which differs from earth's
#' that [mars()] ports. Each forward step adds a column linearly or a pair of hinges on it,
#' `max(0, x - t)` and `max(0, t - x)`, choosing by Friedman's running updates, and the pass stops
#' when a step lowers the residuals by less than `thresh` of them, when they fall to `thresh` of
#' the null model's, when the generalised cross-validation passes ten times the null model's, or at
#' `nk` terms. The pruning pass drops
#' the term of least t statistic, one at a time, and keeps the subset of least generalised
#' cross-validation, which counts each term beyond the intercept as `1 + penalty / 2` degrees of
#' freedom. The defaults are mda's, which biomod2
#' passes unchanged under its default option set and under `"bigboss"`: degree one,
#' `penalty = 2`, `thresh = 0.001` and `nk = max(21, 2 p + 1)` for `p` columns.
#'
#' The case weights are the response head's, [positive_weights()] under presence-absence. They set
#' the classes' scores and the variate, as `fda()` reads them, and not the basis: mda's forward
#' pass sets them to one.
#'
#' biomod2 always recalibrates an `FDA` posterior by a probit regression of the response on it,
#' under the case weights, and predicts through that regression. `calibrate = TRUE` does the same,
#' on the units the discriminant was fitted on; `FALSE` predicts the posterior itself. biomod2
#' rounds the posterior to three decimals before recalibrating it, which is not reproduced.
#'
#' The forward pass, the pruning, the scoring and the recalibration are ported from mda and R into
#' the core the Python package calls, so the two languages keep the same terms and predict the same
#' probabilities. A response holding one value is predicted its mean, and so is one whose scored
#' response the basis does not reach; both are named in `unfitted`. The learner needs a
#' presence-absence response, and a head whose loss is not the binary cross-entropy is refused.
#'
#' @inheritParams elasticnet
#' @param degree The most hinges a term of the basis multiplies.
#' @param penalty The generalised cross-validation's charge per term beyond its own degree of
#'   freedom, taken at half. `NULL` is mda's, 2 at degree one and 3 above.
#' @param nk The most terms the forward pass reaches, the intercept included. `NULL` is
#'   `max(21, 2 p + 1)` for `p` columns; an even number is taken one lower.
#' @param thresh The least share of the residuals a forward step is kept for.
#' @param prune Whether the pruning pass runs. Without it every term of the forward pass is kept.
#' @param calibrate Whether the posterior is recalibrated by a probit regression, as biomod2 does.
#' @param threads Columns searched at once. The model is the same on any number.
#'
#' @return A [learner()].
#'
#' @examples
#' discriminant()
#' discriminant(data = grain("season"), calibrate = FALSE)
#'
#' @export
discriminant <- function(data = NULL, degree = 1L, penalty = NULL, nk = NULL, thresh = 0.001,
                         prune = TRUE, calibrate = TRUE, threads = 1L) {
  .check_count(degree, "degree", 1)
  if (!is.null(nk)) .check_count(nk, "nk", 3)
  .check_count(threads, "threads", 1)
  if (!is.null(penalty) && (!is.numeric(penalty) || length(penalty) != 1L || is.na(penalty) ||
                            penalty < 0)) {
    stop("`penalty` is one number of zero or more, got ", .describe(penalty), ".", call. = FALSE)
  }
  if (!is.numeric(thresh) || length(thresh) != 1L || is.na(thresh) || thresh < 0 ||
      thresh >= 1) {
    stop("`thresh` is one number in [0, 1), got ", .describe(thresh), ".", call. = FALSE)
  }
  if (!is.logical(prune) || length(prune) != 1L || is.na(prune)) {
    stop("`prune` is TRUE or FALSE.", call. = FALSE)
  }
  if (!is.logical(calibrate) || length(calibrate) != 1L || is.na(calibrate)) {
    stop("`calibrate` is TRUE or FALSE.", call. = FALSE)
  }
  learner(
    name = "discriminant",
    data = data, reads = "tabular", multi = "separate",
    params = list(degree = as.integer(degree), penalty = penalty, nk = nk, thresh = thresh,
                  prune = prune, calibrate = calibrate, threads = as.integer(threads)),
    fit = function(x, y, degree, penalty, nk, thresh, prune, calibrate, threads, head, weights,
                   ...) {
      if (!identical(.head_family(head), "binomial")) {
        stop("a discriminant separates presences from absences, under a head whose loss is the ",
             "binary cross-entropy; this head's loss is ", .describe(head$loss), ".",
             call. = FALSE)
      }
      m <- .flatten(x)
      models <- lapply(seq_len(ncol(y)), function(j) {
        yj <- y[, j]
        if (length(unique(yj)) < 2L) {
          return(mean(yj))
        }
        .fda_fit(m, yj, weights[, j], degree = degree, penalty = penalty, nk = nk,
                 thresh = thresh, prune = prune, calibrate = calibrate, threads = threads)
      })
      unfitted <- vapply(models, function(f) is.numeric(f) || !f$discriminates, logical(1L))
      stopped <- vapply(models, function(f) is.list(f) && !f$converged, logical(1L))
      list(models = models, columns = colnames(m), unfitted = colnames(y)[unfitted],
           stopped = colnames(y)[stopped])
    },
    predict = function(model, x) {
      m <- .flatten(x)
      .as_predictions(vapply(model$models, function(f) {
        if (is.numeric(f)) rep(f, nrow(m)) else .fda_predict(f, m)
      }, numeric(nrow(m))), nrow(m))
    }
  )
}

# The discriminant, over the core `src/ts_fda.cpp` compiles into both languages. A fit is a plain
# list of numbers, the kept terms as their factors, their coefficients, the variate and the
# recalibration, so it round trips through `saveRDS()` and predicts on another machine.
.fda_fit <- function(x, y, w, degree = 1L, penalty = NULL, nk = NULL, thresh = 0.001,
                     prune = TRUE, calibrate = TRUE, threads = 1L) {
  ts_fda_fit_(as.numeric(x), as.numeric(y), as.numeric(w), nrow(x), ncol(x), as.integer(degree),
              if (is.null(penalty)) NA_real_ else as.numeric(penalty),
              if (is.null(nk)) 0L else as.integer(nk), as.numeric(thresh), isTRUE(prune),
              isTRUE(calibrate), as.integer(threads))
}

.fda_predict <- function(fit, newx) {
  ts_fda_predict_(fit, as.numeric(newx), nrow(newx), ncol(newx))
}
