#' Multivariate adaptive regression splines on the flattened representation
#'
#' One MARS model per response, over every bin-by-channel column of the representation, fitted as
#' the earth package fits it and as biomod2 fits `MARS`. The forward pass starts from the intercept
#' and at each step multiplies a term already in the model by a pair of hinges on one column,
#' `max(0, x - t)` and `max(0, t - x)`, taking the parent, the column and the knot `t` that most
#' reduce the residual sum of squares of a least-squares fit to the response. A knot at a column's
#' least value enters the column linearly. `degree` bounds how many hinges a term multiplies, so
#' `degree = 1` is an additive model and `2` admits pairwise interactions. The pass stops at `nk`
#' terms, when a step raises the R-squared by less than `thresh`, or when no term reduces the
#' residuals.
#'
#' The pruning pass then removes terms one at a time, each time the one whose loss raises the
#' residuals least, and keeps the subset of least generalised cross-validation, which charges
#' `penalty` for each knot. Under a presence-absence head the kept terms are refitted as a logistic
#' model, as earth's `glm = list(family = binomial)` refits them and biomod2 asks it to, and the
#' prediction is that model's probability; under a squared-error head they are refitted by least
#' squares.
#'
#' The defaults are earth's own, which biomod2 uses under its default option set and under
#' `"bigboss"` alike: degree one, `penalty = 2`, `thresh = 0.001`, `nk = min(200, max(20, 2 p)) + 1`
#' for `p` columns, Friedman's rules for the spans between knots, and Fast MARS over the 20 best
#' parents. The forward pass, the pruning pass and the refit are ported from earth and leaps into
#' the core the Python package calls, so the two languages keep the same terms and return the same
#' coefficients.
#'
#' The case weights are the response head's, [positive_weights()] under presence-absence, and weigh
#' both passes and the refit as earth weighs them. earth refits the whole basis by QR at every
#' candidate knot of a weighted fit; the core reaches the same residual sums with Friedman's running
#' updates, which is what makes a weighted fit over hundreds of columns affordable.
#'
#' A response holding one value is predicted its mean.
#'
#' @inheritParams elasticnet
#' @param degree The most hinges a term multiplies.
#' @param penalty The generalised cross-validation's charge per knot. `NULL` is earth's, 2 at
#'   degree one and 3 above; -1 charges nothing.
#' @param nk The most terms the forward pass reaches, the intercept included. `NULL` is
#'   `min(200, max(20, 2 p)) + 1` for `p` columns.
#' @param thresh The least rise in R-squared a forward step is kept for.
#' @param minspan,endspan Units between knots, and units at either end of a column no knot is
#'   placed among. 0 is Friedman's rule; a negative `minspan` asks for that many knots per column.
#' @param fast_k,fast_beta Fast MARS: parents tried at each step, and how fast an untried parent
#'   ages up the queue. `fast_k = 0` tries every term.
#' @param prune Whether the pruning pass runs. Without it every term of the forward pass is kept.
#' @param nprune The most terms kept, the intercept included; `NULL` for no bound.
#' @param threads Columns searched at once. The model is the same on any number.
#'
#' @return A [learner()].
#'
#' @examples
#' mars()
#' mars(degree = 2L)
#' mars(data = grain("season"), nprune = 10L)
#'
#' @export
mars <- function(data = NULL, degree = 1L, penalty = NULL, nk = NULL, thresh = 0.001,
                 minspan = 0L, endspan = 0L, fast_k = 20L, fast_beta = 1, prune = TRUE,
                 nprune = NULL, threads = 1L) {
  .check_count <- function(v, name, least) {
    if (!is.numeric(v) || length(v) != 1L || is.na(v) || v != round(v) || v < least) {
      stop("`", name, "` is one whole number of ", least, " or more, got ", .describe(v), ".",
           call. = FALSE)
    }
  }
  .check_count(degree, "degree", 1)
  if (!is.null(nk)) .check_count(nk, "nk", 1)
  if (!is.null(nprune)) .check_count(nprune, "nprune", 1)
  .check_count(endspan, "endspan", 0)
  .check_count(fast_k, "fast_k", 0)
  if (!is.numeric(minspan) || length(minspan) != 1L || is.na(minspan) ||
      minspan != round(minspan)) {
    stop("`minspan` is one whole number, got ", .describe(minspan), ".", call. = FALSE)
  }
  if (!is.null(penalty) && (!is.numeric(penalty) || length(penalty) != 1L || is.na(penalty) ||
                            (penalty < 0 && penalty != -1))) {
    stop("`penalty` is one number of zero or more, or -1, got ", .describe(penalty), ".",
         call. = FALSE)
  }
  if (!is.numeric(thresh) || length(thresh) != 1L || is.na(thresh) || thresh < 0 ||
      thresh >= 1) {
    stop("`thresh` is one number in [0, 1), got ", .describe(thresh), ".", call. = FALSE)
  }
  if (!is.numeric(fast_beta) || length(fast_beta) != 1L || is.na(fast_beta) || fast_beta < 0) {
    stop("`fast_beta` is one number of zero or more, got ", .describe(fast_beta), ".",
         call. = FALSE)
  }
  if (!is.logical(prune) || length(prune) != 1L || is.na(prune)) {
    stop("`prune` is TRUE or FALSE.", call. = FALSE)
  }
  learner(
    name = "mars",
    data = data, reads = "tabular", multi = "separate",
    params = list(degree = as.integer(degree), penalty = penalty, nk = nk, thresh = thresh,
                  minspan = as.integer(minspan), endspan = as.integer(endspan),
                  fast_k = as.integer(fast_k), fast_beta = fast_beta, prune = prune,
                  nprune = nprune, threads = as.integer(threads)),
    fit = function(x, y, degree, penalty, nk, thresh, minspan, endspan, fast_k, fast_beta, prune,
                   nprune, threads, head, weights, ...) {
      family <- .head_family(head)
      m <- .flatten(x)
      models <- lapply(seq_len(ncol(y)), function(j) {
        yj <- y[, j]
        if (length(unique(yj)) < 2L) {
          return(mean(yj))
        }
        .mars_fit(m, yj, weights[, j], family, degree = degree, penalty = penalty, nk = nk,
                  thresh = thresh, minspan = minspan, endspan = endspan, fast_k = fast_k,
                  fast_beta = fast_beta, prune = prune, nprune = nprune, threads = threads)
      })
      stopped <- colnames(y)[vapply(models, function(f) is.list(f) && !f$converged, logical(1L))]
      list(models = models, columns = colnames(m), family = family, stopped = stopped)
    },
    predict = function(model, x) {
      m <- .flatten(x)
      .as_predictions(vapply(model$models, function(f) {
        if (is.numeric(f)) rep(f, nrow(m)) else .mars_predict(f, m)
      }, numeric(nrow(m))), nrow(m))
    }
  )
}

# MARS, over the core `src/ts_mars.cpp` compiles into both languages. A fit is a plain list of
# numbers, every forward term as its factors and the kept ones' coefficients, so it round trips
# through `saveRDS()` and predicts on another machine.
.mars_fit <- function(x, y, w, family, degree = 1L, penalty = NULL, nk = NULL, thresh = 0.001,
                      minspan = 0L, endspan = 0L, fast_k = 20L, fast_beta = 1, prune = TRUE,
                      nprune = NULL, threads = 1L) {
  ts_mars_fit_(as.numeric(x), as.numeric(y), as.numeric(w), nrow(x), ncol(x), family,
               as.integer(degree), if (is.null(penalty)) NA_real_ else as.numeric(penalty),
               if (is.null(nk)) 0L else as.integer(nk), as.numeric(thresh),
               as.integer(minspan), as.integer(endspan), as.integer(fast_k),
               as.numeric(fast_beta), isTRUE(prune),
               if (is.null(nprune)) 0L else as.integer(nprune), as.integer(threads))
}

.mars_predict <- function(fit, newx) {
  ts_mars_predict_(fit, as.numeric(newx), nrow(newx), ncol(newx))
}
