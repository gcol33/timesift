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
#' @param n_inner Folds of the inner cross-validation that chooses the penalty.
#' @param squares Add the square of every column, giving the same quadratic capacity a
#'   second-order polynomial term would.
#' @param s Which penalty of the inner path to predict at: `"lambda.min"`, `"lambda.1se"`, or a
#'   penalty of its own, which is interpolated between the two points of the path around it.
#' @param n_lambda Points of the penalty path.
#' @param thresh Where the coordinate descent stops, read off the largest coefficient move of a
#'   pass. The default leaves the fit as close to the optimum as glmnet's own default does; a
#'   looser one is faster and a tighter one costs time roughly in proportion.
#' @param threads How many fits of one response's inner cross-validation run at once. The path on
#'   every fitting unit and the path of each inner fold are one independent fit each, so they
#'   parallelise without sharing anything, and `n_inner + 1` threads is as many as a response can
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
elasticnet <- function(data = NULL, alpha = 0.5, n_inner = 5L, squares = TRUE, s = "lambda.min",
                       n_lambda = 100L, thresh = 1e-8, threads = 1L, seed = 1L) {
  learner(
    name = "elasticnet",
    data = data, reads = "tabular", multi = "separate",
    params = list(alpha = alpha, n_inner = n_inner, squares = squares, s = s,
                  n_lambda = n_lambda, thresh = thresh, threads = threads, seed = seed),
    fit = function(x, y, alpha, n_inner, squares, s, n_lambda, thresh, threads, seed, head,
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
      seeds <- .variable_seeds(seed, y)
      weights <- .head_weights(head, y)
      models <- lapply(seq_len(ncol(y)), function(j) {
        yj <- y[, j]
        if (length(unique(yj)) < 2L) {
          return(mean(yj))
        }
        # The inner folds are dealt here rather than inside the path, so a grouping the outer
        # folds keep whole stays whole where the penalty is chosen, and a rare outcome is spread
        # over them rather than left to a plain deal.
        inner <- .inner_folds(yj, n_inner, seeds[j], group)
        if (identical(family, "binomial") && !.inner_fittable(yj, inner)) {
          return(mean(yj))
        }
        labels <- sort(unique(inner))
        .penalised_cv(m, yj, weights[, j], family, alpha, match(inner, labels) - 1L,
                      length(labels), n_lambda = n_lambda, thresh = thresh, threads = threads)
      })
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

#' Stepwise selection by AIC on the flattened representation
#'
#' One generalised linear model per variable, its terms chosen by Akaike's criterion over every
#' bin-by-channel column. The family is the response head's: logistic under a binary cross-entropy
#' loss, Gaussian under a squared-error one and Poisson under a Poisson-deviance one. So are the
#' case weights, so a rare response weighs here what it weighs in every other learner.
#'
#' `terms` says what one term is. Under `"column"` it is a column's orthogonal polynomial of degree
#' `degree`, so a column enters with its curvature at once and can be non-monotone in the reading
#' the way a niche optimum is. Under `"power"` each power of a column, `x`, `x^2` and so on, is a
#' term of its own, which is how biomod2 writes a quadratic formula and how `MASS::stepAIC()` walks
#' it. A column holding one value over the fitting units is not a term.
#'
#' `direction` is the search. `"forward"` starts from the intercept and admits the term that lowers
#' the criterion most, while one does and the model holds fewer than `max_terms`. `"both"` does the
#' same but also weighs dropping each term it holds at every step, and `"backward"` starts from
#' every term and drops alone. `"none"` fits every term and selects nothing: with
#' `terms = "power"` and `degree = 2` that is the model biomod2's GLM fits, `y ~ x + I(x^2)` over
#' every column. The two-way and backward searches are MASS's `stepAIC()`, step for step: the model as it
#' stands wins a tie, a term whose removal leaves the rank unchanged is dropped first, and an
#' addition that does not raise the rank is not offered. `max_terms` bounds what a forward or
#' two-way search adds, and the backward and unselected fits start from every term whatever it is.
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
#' @param direction `"forward"`, `"both"`, `"backward"` or `"none"`.
#' @param terms `"column"` for a column's polynomial as one term, `"power"` for each power its own.
#' @param threads How many of one step's candidate fits run at once. What comes back does not
#'   depend on it.
#'
#' @return A [learner()].
#'
#' @examples
#' stepwise(max_terms = 3)
#' stepwise(direction = "both", terms = "power", max_terms = Inf)
#' stepwise(data = grain("season"), direction = "none", terms = "power")
#'
#' @export
stepwise <- function(data = NULL, max_terms = 3L, degree = 2L,
                     direction = c("forward", "both", "backward", "none"),
                     terms = c("column", "power"), threads = 1L) {
  direction <- match.arg(direction)
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
    name = "stepwise",
    data = data, reads = "tabular", multi = "separate",
    params = list(max_terms = max_terms, degree = as.integer(degree), direction = direction,
                  terms = terms, threads = as.integer(threads)),
    fit = function(x, y, max_terms, degree, direction, terms, threads, head, ...) {
      family <- .head_family(head)
      m <- .flatten(x)
      weights <- .head_weights(head, y)
      models <- lapply(seq_len(ncol(y)), function(j) {
        yj <- y[, j]
        if (length(unique(yj)) < 2L) {
          return(mean(yj))
        }
        .stepwise_fit(m, yj, weights[, j], family, max_terms = max_terms, degree = degree,
                      direction = direction, terms = terms, threads = threads)
      })
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
.stepwise_fit <- function(x, y, w, family, max_terms = 3, degree = 2L, direction = "forward",
                          terms = "column", threads = 1L) {
  ts_stepwise_fit_(as.numeric(x), as.numeric(y), as.numeric(w), nrow(x), ncol(x), family,
                   as.numeric(max_terms), as.integer(degree), direction, terms,
                   as.integer(threads))
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

