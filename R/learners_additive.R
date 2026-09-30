#' Generalised additive model on the flattened representation
#'
#' One additive model per response, a smooth function of every bin-by-channel column of the
#' representation, fitted as mgcv's `gam(y ~ s(x1) + s(x2) + ..., method = "GCV.Cp")` fits it and
#' as biomod2 fits `GAM`. Each column enters as a thin plate regression spline of `k` basis
#' functions (Wood 2003): a cubic radial function centred on each of the column's distinct values,
#' reduced to its `k - 2` directions of greatest eigenvalue, together with the linear function,
#' which the penalty on the spline's squared second derivative leaves free. Each smooth sums to zero
#' over the units, beside one intercept.
#'
#' The coefficients maximise the penalised likelihood, by penalised iteratively reweighted least
#' squares, and the smoothing parameters, one per column, minimise the unbiased risk estimator
#' under presence-absence and the generalised cross-validation score under a squared-error head, by
#' Newton's method with the exact derivatives (Wood 2008). `gamma` multiplies the charge each
#' effective degree of freedom adds to that criterion, as mgcv's `gamma` does; above one it gives
#' smoother fits. Over columns as alike as neighbouring weeks the criterion can have more than one
#' local minimum; the one the search settles in then depends on where it starts, and mgcv, starting
#' from a rule of its own parametrisation, can settle in another.
#'
#' The defaults are mgcv's, which biomod2 passes unchanged: `k = 10` and a criterion charge of one.
#' Above `max_knots` distinct values, the radial functions are centred on that many of them, drawn
#' as mgcv draws them, so that the basis is the one mgcv fits. A column of fewer than `k` distinct
#' values takes as many basis functions as it holds values, a column of two enters linearly, and a
#' column of one is left out; mgcv refuses the first and the last of those. A column whose linear
#' part the columns before it already span keeps its penalised part and loses the linear one.
#'
#' The case weights are the response head's, [positive_weights()] under presence-absence, and enter
#' the likelihood as mgcv's prior weights. The model holds at most as many coefficients as there are
#' units, as mgcv's does: one for the intercept and `k - 1` per column.
#'
#' The basis, the fit and the choice of the smoothing parameters live in the core the Python package
#' calls, pinned against mgcv in the fixtures, so the two languages fit the same model. A response
#' holding one value is predicted its mean and named in `unfitted`; one whose smoothing parameter
#' search stopped short of its tolerance is named in `stopped`.
#'
#' @inheritParams elasticnet
#' @param k The basis dimension of a column's smooth, the constant included, at least 3.
#' @param gamma The criterion's charge per effective degree of freedom.
#' @param max_knots The most distinct values of a column the radial functions are centred on.
#' @param threads Columns and responses worked at once. The model is the same on any number.
#'
#' @return A [learner()].
#'
#' @examples
#' additive()
#' additive(data = grain("season"), k = 5)
#'
#' @export
additive <- function(data = NULL, k = 10L, gamma = 1, max_knots = 2000L, threads = 1L) {
  .check_count(k, "k", 3)
  .check_count(max_knots, "max_knots", 3)
  .check_count(threads, "threads", 1)
  if (max_knots < k) {
    stop("`max_knots` is at least `k`.", call. = FALSE)
  }
  if (!is.numeric(gamma) || length(gamma) != 1L || is.na(gamma) || gamma <= 0) {
    stop("`gamma` is one positive number, got ", .describe(gamma), ".", call. = FALSE)
  }
  learner(
    name = "additive",
    data = data, reads = "tabular", multi = "separate",
    params = list(k = as.integer(k), gamma = gamma, max_knots = as.integer(max_knots),
                  threads = as.integer(threads)),
    fit = function(x, y, k, gamma, max_knots, threads, head, weights, ...) {
      family <- .head_family(head)
      m <- .flatten(x)
      fittable <- vapply(seq_len(ncol(y)), function(j) length(unique(y[, j])) > 1L, logical(1L))
      fit <- NULL
      if (any(fittable)) {
        fit <- .additive_fit(m, y[, fittable, drop = FALSE], weights[, fittable, drop = FALSE],
                             family, k = k, gamma = gamma, max_knots = max_knots,
                             threads = threads)
      }
      stopped <- rep(FALSE, ncol(y))
      if (!is.null(fit)) stopped[fittable] <- fit$converged == 0L
      list(fit = fit, fittable = fittable, means = colMeans(y), columns = colnames(m),
           family = family, unfitted = colnames(y)[!fittable], stopped = colnames(y)[stopped])
    },
    predict = function(model, x) {
      m <- .flatten(x)
      out <- matrix(rep(model$means, each = nrow(m)), nrow(m), length(model$means))
      if (!is.null(model$fit)) {
        out[, model$fittable] <- .additive_predict(model$fit, m)
      }
      .as_predictions(out, nrow(m))
    }
  )
}

# The additive model, over the core `src/ts_additive.cpp` compiles into both languages. A fit is a
# plain list of numbers, every column's basis and every response's coefficients, so it round trips
# through `saveRDS()` and predicts on another machine.
.additive_fit <- function(x, y, w, family, k = 10L, gamma = 1, max_knots = 2000L, threads = 1L) {
  y <- as.matrix(y)
  w <- as.matrix(w)
  ts_additive_fit_(as.numeric(x), as.numeric(y), as.numeric(w), nrow(x), ncol(x), ncol(y),
                   family, as.integer(k), as.numeric(gamma), as.integer(max_knots),
                   as.integer(threads))
}

.additive_predict <- function(fit, newx) {
  matrix(ts_additive_predict_(fit, as.numeric(newx), nrow(newx), ncol(newx)), nrow(newx),
         fit$n_response)
}
