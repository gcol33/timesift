#' Single-hidden-layer network on the flattened representation
#'
#' One network per response, over every bin-by-channel column of the representation, fitted as the
#' nnet package fits it and as biomod2 fits `ANN`. Each of `hidden` logistic units takes a bias and
#' every column, and the output takes a bias, every hidden unit and, with `skip`, every column
#' again. The weights start uniform on `[-range, range]` and are fitted by the variable metric
#' (BFGS) method of Nash (1990), the minimiser nnet uses, on the response head's loss plus `decay`
#' times the sum of the squared weights, biases included. The fit stops after `max_iter`
#' iterations, when the objective falls below `abs_tol`, or when an iteration lowers it by no more
#' than `rel_tol` of itself.
#'
#' Under a presence-absence head the output is the logistic function of its sum and the loss the
#' cross-entropy, nnet's `entropy = TRUE`; under a head with a squared-error loss the output is the
#' sum itself and the loss the sum of squares, nnet's `linout = TRUE`; under a count head the output
#' is the exponential of the sum and the loss the Poisson deviance, which nnet does not offer.
#' biomod2 leaves nnet's own `entropy = FALSE`, fitting a presence-absence response by least squares
#' on the logistic output; the learner fits the head's loss, as every learner does.
#'
#' `preset` says whose defaults the settings left `NULL` take. `"default"` is what biomod2's default
#' option set fits: two hidden units, as biomod2 sets them, and nnet's own `decay = 0`,
#' `range = 0.7` and `max_iter = 100`. `"bigboss"` is biomod2's tuned option set: five hidden units,
#' `decay = 0.1`, `range = 0.1` and `max_iter = 200`. A setting given explicitly beats either.
#'
#' nnet reads the columns as given, and so does the default: a record in its own units saturates
#' the hidden units sooner the wider its range. With `standardise = TRUE` each column is centred on
#' its mean and divided by its sample standard deviation over the fitting units, and a prediction
#' centres and scales by the fit's own.
#'
#' The network, its objective and the minimiser live in the core the Python package calls, pinned
#' against nnet in the fixtures from the same starting weights, so the two languages fit the same
#' network. The starting weights are drawn from the core's own generator, so a fit does not repeat
#' nnet's from the same R seed. `threads` fit that many responses at once, each network the same as
#' when fitted alone. The minimiser holds an approximate inverse Hessian of one number per pair of
#' weights for every network fitted at once; a fit that would need more than `max_hessian`
#' gigabytes for them is refused with the size, and a coarser grain, fewer hidden units or fewer
#' threads shrinks it.
#'
#' The case weights are the response head's, [positive_weights()] under presence-absence, and weigh
#' each unit's term of the loss as nnet's `weights` do.
#' Under the shipped presence-absence head those weights are on, so a default `perceptron()` is
#' biomod2's `ANN` specification fitted under them; a head registered without `weights` fits it
#' unweighted.
#'
#' A response holding one value is predicted its mean.
#'
#' @inheritParams elasticnet
#' @param hidden Hidden units.
#' @param decay The weight of the squared weights' sum in the objective.
#' @param range The starting weights are uniform on `[-range, range]`.
#' @param max_iter The most iterations of the minimiser.
#' @param skip Whether the output also takes every column directly.
#' @param standardise Whether each column is centred and scaled before the fit.
#' @param abs_tol,rel_tol The fit stops when the objective falls below `abs_tol`, or when an
#'   iteration lowers it by no more than `rel_tol` of itself.
#' @param preset Whose defaults the settings left `NULL` take: `"default"` or `"bigboss"`.
#' @param max_hessian Gigabytes the minimisers' approximate inverse Hessians may take together.
#' @param threads Responses fitted at once.
#' @param seed Seed for the starting weights.
#'
#' @return A [learner()].
#'
#' @examples
#' perceptron()
#' perceptron(preset = "bigboss")
#' perceptron(hidden = 4L, decay = 0.01)
#'
#' @export
perceptron <- function(data = NULL, hidden = NULL, decay = NULL, range = NULL, max_iter = NULL,
                       skip = FALSE, standardise = FALSE, abs_tol = 1e-4, rel_tol = 1e-8,
                       preset = c("default", "bigboss"), max_hessian = 2, threads = 1L,
                       seed = 1L) {
  preset <- match.arg(preset)
  settings <- .perceptron_settings(preset, hidden, decay, range, max_iter)
  .check_count(settings$hidden, "hidden", 1)
  .check_count(settings$max_iter, "max_iter", 0)
  numbers <- list(decay = settings$decay, range = settings$range, abs_tol = abs_tol,
                  rel_tol = rel_tol)
  for (nm in names(numbers)) {
    v <- numbers[[nm]]
    if (!is.numeric(v) || length(v) != 1L || is.na(v) || v < 0) {
      stop("`", nm, "` is one number of zero or more, got ", .describe(v), ".", call. = FALSE)
    }
  }
  for (nm in c("skip", "standardise")) {
    v <- get(nm)
    if (!is.logical(v) || length(v) != 1L || is.na(v)) {
      stop("`", nm, "` is TRUE or FALSE.", call. = FALSE)
    }
  }
  .check_count(threads, "threads", 1)
  if (!is.numeric(max_hessian) || length(max_hessian) != 1L || is.na(max_hessian) ||
      max_hessian <= 0) {
    stop("`max_hessian` is one positive number, got ", .describe(max_hessian), ".", call. = FALSE)
  }
  learner(
    name = "perceptron",
    data = data, reads = "tabular", multi = "separate",
    params = c(settings, list(skip = skip, standardise = standardise,
                              abs_tol = as.numeric(abs_tol), rel_tol = as.numeric(rel_tol),
                              max_hessian = as.numeric(max_hessian),
                              threads = as.integer(threads), seed = as.integer(seed))),
    fit = function(x, y, hidden, decay, range, max_iter, skip, standardise, abs_tol, rel_tol,
                   max_hessian, threads, seed, head, weights, ...) {
      family <- .head_family(head)
      m <- .flatten(x)
      fittable <- vapply(seq_len(ncol(y)), function(j) length(unique(y[, j])) > 1L, logical(1L))
      at_once <- max(min(threads, sum(fittable)), 1L)
      n_weight <- hidden * (ncol(m) + 1) + hidden + 1 + if (skip) ncol(m) else 0
      need <- at_once * n_weight * (n_weight + 1) / 2 * 8 / 2^30
      if (need > max_hessian) {
        stop(sprintf(paste0("%d network(s) of %d hidden units over %d columns, %d weights each, ",
                            "fitted at once would hold approximate inverse Hessians of %.1f GB, ",
                            "above `max_hessian = %g`. A coarser grain, fewer hidden units or ",
                            "fewer threads shrinks it."),
                     at_once, hidden, ncol(m), n_weight, need, max_hessian), call. = FALSE)
      }
      models <- as.list(unname(colMeans(y)))
      if (any(fittable)) {
        models[fittable] <- .perceptron_fits(
          m, y[, fittable, drop = FALSE], weights[, fittable, drop = FALSE], family,
          seeds = .variable_seeds(seed, y)[fittable], hidden = hidden, decay = decay,
          range = range, max_iter = max_iter, skip = skip, standardise = standardise,
          abs_tol = abs_tol, rel_tol = rel_tol, threads = threads)
      }
      stopped <- colnames(y)[vapply(models, function(f) is.list(f) && !f$converged, logical(1L))]
      list(models = models, columns = colnames(m), family = family, stopped = stopped)
    },
    predict = function(model, x) {
      m <- .flatten(x)
      .as_predictions(vapply(model$models, function(f) {
        if (is.numeric(f)) rep(f, nrow(m)) else .perceptron_predict(f, m)
      }, numeric(nrow(m))), nrow(m))
    }
  )
}

# The settings a network is fitted under: those given, and the preset's for the rest.
.perceptron_settings <- function(preset, hidden, decay, range, max_iter) {
  base <- if (identical(preset, "bigboss")) {
    list(hidden = 5L, decay = 0.1, range = 0.1, max_iter = 200L)
  } else {
    list(hidden = 2L, decay = 0, range = 0.7, max_iter = 100L)
  }
  list(hidden = as.integer(hidden %||% base$hidden), decay = as.numeric(decay %||% base$decay),
       range = as.numeric(range %||% base$range),
       max_iter = as.integer(max_iter %||% base$max_iter))
}

# The one-hidden-layer networks, over the core `src/ts_perceptron.cpp` compiles into both
# languages: one per column of `y`, under the matching column of `w` and seed of `seeds`. A fit is a
# plain list of its size, its family, the centre and scale of its columns and its weights, so it
# round trips through `saveRDS()` and predicts on another machine. `start`, where given, replaces
# every network's drawn starting weights.
.perceptron_fits <- function(x, y, w, family, seeds, hidden = 2L, decay = 0, range = 0.7,
                             max_iter = 100L, skip = FALSE, standardise = FALSE, abs_tol = 1e-4,
                             rel_tol = 1e-8, threads = 1L, start = NULL) {
  y <- as.matrix(y)
  w <- as.matrix(w)
  ts_perceptron_fit_(as.numeric(x), as.numeric(y), as.numeric(w), nrow(x), ncol(x), ncol(y),
                     as.integer(seeds), family, as.integer(hidden), isTRUE(skip),
                     isTRUE(standardise), as.numeric(decay), as.numeric(range),
                     as.integer(max_iter), as.numeric(abs_tol), as.numeric(rel_tol),
                     as.integer(threads), as.numeric(start %||% numeric(0)))
}

# The network of one response.
.perceptron_fit <- function(x, y, w, family, ..., seed = 1L) {
  .perceptron_fits(x, y, w, family, seeds = seed, ...)[[1L]]
}

.perceptron_predict <- function(fit, newx) {
  ts_perceptron_predict_(fit, as.numeric(newx), nrow(newx), ncol(newx))
}
