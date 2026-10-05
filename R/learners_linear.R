#' Penalised regression on the flattened representation
#'
#' One elastic net per variable, over every bin-by-channel column of the representation and, by
#' default, their squares. There is no discrete selection step: the penalty path uses every column
#' and shrinks, and the penalty itself is chosen by an inner cross-validation on the fitting units,
#' so nothing about the model is decided outside the fold it is fitted in.
#'
#' The family is the response head's: a binary cross-entropy loss fits a logistic model, a
#' squared-error loss a linear one and a Poisson-deviance loss a log-linear one, so the learner is
#' the same under a presence-absence head, a continuous one and a count one. So are the case
#' weights: the head's `weights`, [positive_weights()] for presence-absence, are what every learner
#' that ships fits under.
#'
#' The inner folds are dealt for each response and stratified on it, so a rare outcome is spread
#' over them as evenly as its count allows. A presence-absence response whose inner training sets
#' cannot each hold two of each outcome, the fewest a logistic path is fitted to, has too few of
#' one outcome to choose a penalty on. It is predicted its share among the fitting units, as a
#' response holding one outcome is, and the fit names every such response in `unfitted`.
#'
#' A reweighted fit that does not settle at a penalty ends its path there and keeps the points
#' before it, which is what glmnet does with the same event, and the penalty is chosen over the
#' points fitted. It happens where a rare outcome is nearly separable at the small end of the
#' path. The fit names every response whose path on the fitting units, or on any inner fold, ended
#' that way in `stopped`.
#'
#' This is the aggregate-feature side of the comparison the package was built for, and it is the
#' fair opponent for a network: a per-fold discrete selector pays selection variance a network
#' never pays, so beating that one is not a matched result.
#'
#' The path is fitted by the same core the Python package calls, so the two return the same
#' coefficients for the same input. Its conventions are glmnet's, which is what the arm is
#' measured against: weights normalised to sum to one, columns centred and scaled by their
#' weighted mean and weighted standard deviation, a hundred penalties down from the smallest that
#' leaves every coefficient at zero, and the held-out deviance read fold by fold.
#'
#' @param data A representation the learner is pinned to, or `NULL` to run across every
#'   representation of the run.
#' @param alpha Elastic-net mixing, `1` lasso and `0` ridge.
#' @param n_inner Folds of the inner cross-validation that chooses the penalty, ten as in
#'   `cv.glmnet()`.
#' @param squares Add the square of every column, giving the same quadratic capacity a
#'   second-order polynomial term would.
#' @param s Which penalty of the inner path to predict at: `"lambda.min"`, `"lambda.1se"`, or a
#'   penalty of its own, which is interpolated between the two points of the path around it.
#'   `"lambda.1se"`, the largest penalty within one standard error of the least held-out
#'   deviance, is glmnet's default.
#' @param n_lambda Points of the penalty path.
#' @param tol Where the coordinate descent stops, read off the largest coefficient move of a
#'   pass. The default leaves the fit as close to the optimum as glmnet's own default does; a
#'   looser one is faster and a tighter one costs time roughly in proportion.
#' @param threads How many responses are fitted at once, or, with one response to fit, how many
#'   fits of its inner cross-validation: the path on every fitting unit and the path of each inner
#'   fold are one independent fit each, so `n_inner + 1` threads is as many as a lone response can
#'   use. The default is serial, because a package does not take a machine's cores without being
#'   asked. What comes back does not depend on it.
#' @param seed Seed for the inner cross-validation's fold draw, which is random and would otherwise
#'   make the fit irreproducible.
#'
#' @return A [learner()].
#'
#' @examples
#' elasticnet(alpha = 0.5)
#'
#' @export
elasticnet <- function(data = NULL, alpha = 0.5, n_inner = 10L, squares = TRUE,
                       s = "lambda.1se", n_lambda = 100L, tol = 1e-8, threads = 1L, seed = 1L) {
  learner(
    name = "elasticnet",
    data = data, reads = "tabular", multi = "separate",
    params = list(alpha = alpha, n_inner = n_inner, squares = squares, s = s,
                  n_lambda = n_lambda, tol = tol, threads = threads, seed = seed),
    fit = function(x, y, alpha, n_inner, squares, s, n_lambda, tol, threads, seed, head,
                   group = NULL, ...) {
      family <- .head_family(head)
      m <- .design(x, squares)
      if (ncol(m) < 2L) {
        stop("the elastic net needs at least two columns to penalise over, and this ",
             "representation flattens to ", ncol(m), ". Give it more bins or channels, or ",
             "leave `squares = TRUE`.", call. = FALSE)
      }
      old <- .seed_state()
      on.exit(.restore_seed(old), add = TRUE)
      weights <- .head_weights(head, y)
      models <- as.list(unname(colMeans(y)))
      fit <- .varies(y)
      if (any(fit)) {
        # The inner folds are dealt here rather than inside the path, so a grouping the outer
        # folds keep whole stays whole where the penalty is chosen, and a rare outcome is spread
        # over them rather than left to a plain deal.
        folds <- .response_folds(y[, fit, drop = FALSE], n_inner,
                                 .variable_seeds(seed, y)[fit], group)
        keep <- if (identical(family, "binomial")) folds$fittable else rep(TRUE, sum(fit))
        fit[fit] <- keep
      }
      if (any(fit)) {
        models[fit] <- .penalised_cvs(m, y[, fit, drop = FALSE], weights[, fit, drop = FALSE],
                                      family, alpha, folds$fold[, keep, drop = FALSE],
                                      folds$n_fold[keep], n_lambda = n_lambda, thresh = tol,
                                      threads = threads)
      }
      unfitted <- colnames(y)[vapply(models, is.numeric, logical(1L))]
      stopped <- colnames(y)[vapply(models, .penalised_stopped, logical(1L))]
      list(models = models, squares = squares, s = s, columns = colnames(m),
           unfitted = unfitted, stopped = stopped)
    },
    predict = function(model, x) {
      m <- .design(x, model$squares)
      .as_predictions(vapply(model$models, function(f) {
        if (is.numeric(f)) rep(f, nrow(m))
        else .penalised_predict(f, m, model$s)
      }, numeric(nrow(m))), nrow(m))
    }
  )
}

#' A generalised linear model on the flattened representation
#'
#' One generalised linear model per variable over every bin-by-channel column, its terms chosen by
#' Akaike's criterion. The family is the response head's: logistic under a binary cross-entropy
#' loss, Gaussian under a squared-error one and Poisson under a Poisson-deviance one. So are the
#' case weights, so a rare response weighs here what it weighs in every other learner.
#'
#' The defaults are biomod2's `GLM`: every column enters as `x + I(x^2)`, and the terms are
#' searched in both directions by AIC as `MASS::stepAIC()` searches them, with no bound on how many
#' are kept.
#' Under the shipped presence-absence head those weights are on, so a default `linear()` is
#' biomod2's `GLM` specification fitted under them; a head registered without `weights` fits it
#' unweighted.
#'
#' `terms` says what one term is. Under `"power"` each power of a column, `x`, `x^2` and so on, is
#' a term of its own, which is how biomod2 writes a quadratic formula and how `stepAIC()` walks it.
#' Under `"column"` a term is a column's orthogonal polynomial of degree `degree`, so a column enters
#' with its curvature at once and can be non-monotone in the reading the way a niche optimum is. A
#' column holding one value over the fitting units is not a term.
#'
#' `select` is the search. `"both"` starts from the intercept and at every step takes the move that
#' lowers the criterion most, adding a term or dropping one it holds, while one does and the model
#' holds fewer than `max_terms`. `"forward"` only adds, and `"backward"` starts from every term and
#' only drops. `"none"` fits every term and selects nothing. The two-way and backward searches are
#' `stepAIC()`, step for step: the model as it stands wins a tie, a term whose removal leaves the
#' rank unchanged is dropped first, and an addition that does not raise the rank is not offered.
#' `max_terms` bounds what a forward or two-way search adds, and the backward and unselected fits
#' start from every term whatever it is.
#'
#' Each fit is R's `glm.fit`: iteratively reweighted least squares, the rank read off the same
#' pivoted decomposition, and the same stopping rule. A move whose fit does not settle within its
#' 25 iterations is refused rather than taken, and the fit names every response whose final model
#' did not settle in `stopped`.
#'
#' A model with nothing but the intercept predicts the response's share among the fitting units.
#' Selection happens inside whichever units the learner is handed, so under [grain_ladder()] it is
#' redone in every fold. Reported beside a penalised fit it also prices discrete selection:
#' choosing a handful of columns out of hundreds is high variance, and that variance is a cost of
#' the selector rather than of the features. An unselected fit over hundreds of columns separates
#' any response it is given and is meant for a coarse grain, `data = grain("season")`.
#'
#' The search and the fits run on the core the Python package calls, so the two select the same
#' terms and return the same coefficients.
#'
#' @inheritParams elasticnet
#' @param max_terms Terms a forward or two-way search holds at most; `Inf` for no bound.
#' @param degree Polynomial degree each column enters at.
#' @param select `"both"`, `"forward"`, `"backward"` or `"none"`.
#' @param terms `"power"` for each power of a column its own term, `"column"` for a column's
#'   polynomial as one term.
#' @param threads How many responses are searched at once, or, with one response to fit, how many
#'   of one step's candidate fits run at once. What comes back does not depend on it.
#'
#' @return A [learner()].
#'
#' @examples
#' linear()
#' linear(select = "forward", terms = "column", max_terms = 3)
#' linear(data = grain("season"), select = "none")
#'
#' @export
linear <- function(data = NULL, select = c("both", "forward", "backward", "none"),
                   terms = c("power", "column"), max_terms = Inf, degree = 2L, threads = 1L) {
  select <- match.arg(select)
  terms <- match.arg(terms)
  if (!is.numeric(max_terms) || length(max_terms) != 1L || is.na(max_terms) || max_terms < 0) {
    stop("`max_terms` is one number of zero or more, or Inf, got ", .describe(max_terms), ".",
         call. = FALSE)
  }
  if (!is.numeric(degree) || length(degree) != 1L || is.na(degree) || degree < 1 ||
      degree != round(degree)) {
    stop("`degree` is one whole number of one or more, got ", .describe(degree), ".",
         call. = FALSE)
  }
  learner(
    name = "linear",
    data = data, reads = "tabular", multi = "separate",
    params = list(select = select, terms = terms, max_terms = max_terms,
                  degree = as.integer(degree), threads = as.integer(threads)),
    fit = function(x, y, select, terms, max_terms, degree, threads, head, ...) {
      family <- .head_family(head)
      m <- .flatten(x)
      weights <- .head_weights(head, y)
      fit <- .varies(y)
      models <- as.list(unname(colMeans(y)))
      if (any(fit)) {
        models[fit] <- .stepwise_fits(m, y[, fit, drop = FALSE], weights[, fit, drop = FALSE],
                                      family, max_terms = max_terms, degree = degree,
                                      direction = select, terms = terms, threads = threads)
      }
      stopped <- colnames(y)[vapply(models, function(f) is.list(f) && !f$converged, logical(1L))]
      list(models = models, columns = colnames(m), family = family, stopped = stopped)
    },
    predict = function(model, x) {
      m <- .flatten(x)
      .as_predictions(vapply(model$models, function(f) {
        if (is.numeric(f)) rep(f, nrow(m)) else .stepwise_predict(f, m)
      }, numeric(nrow(m))), nrow(m))
    }
  )
}
# The stepwise model, over the core `src/ts_stepwise.cpp` compiles into both languages. The design,
# the response and the case weights are settled above; a fit is a plain list of numbers, its terms
# as the column each reads and the recurrence of its polynomial, so it round trips through
# `saveRDS()` and predicts on another machine.
.stepwise_fits <- function(x, y, w, family, max_terms = 3, degree = 2L, direction = "forward",
                           terms = "column", threads = 1L) {
  y <- as.matrix(y)
  ts_stepwise_fit_(as.numeric(x), as.numeric(y), as.numeric(as.matrix(w)), nrow(x), ncol(x),
                   ncol(y), family, as.numeric(max_terms), as.integer(degree), direction, terms,
                   as.integer(threads))
}

# The stepwise model of one response.
.stepwise_fit <- function(x, y, w, family, ...) {
  .stepwise_fits(x, y, w, family, ...)[[1L]]
}

.stepwise_predict <- function(fit, newx) {
  ts_stepwise_predict_(fit, as.numeric(newx), nrow(newx), ncol(newx))
}

.design <- function(x, squares) {
  m <- .flatten(x)
  if (!squares) {
    return(m)
  }
  out <- cbind(m, m^2)
  colnames(out) <- c(colnames(m), paste0(colnames(m), "^2"))
  out
}

# vapply drops to a vector when there is one unit to predict, which would reach the caller as one
# row per variable instead of one column. The shape is restored here so a single new site predicts
# the same way a thousand do.
.as_predictions <- function(p, units) {
  matrix(p, nrow = units)
}

