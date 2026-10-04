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
#' The network, its objective and the minimiser live in the core the Python package calls, pinned
#' against nnet in the fixtures from the same starting weights, so the two languages fit the same
#' network. The starting weights are drawn from the core's own generator, so a fit does not repeat
#' nnet's from the same R seed. The minimiser holds an approximate inverse Hessian of one number per
#' pair of weights; a network that would need more than `max_hessian` gigabytes for it is refused
#' with the size, and a coarser grain or fewer hidden units shrinks it.
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
#' @param abs_tol,rel_tol The fit stops when the objective falls below `abs_tol`, or when an
#'   iteration lowers it by no more than `rel_tol` of itself.
#' @param preset Whose defaults the settings left `NULL` take: `"default"` or `"bigboss"`.
#' @param max_hessian Gigabytes the minimiser's approximate inverse Hessian may take.
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
                       skip = FALSE, abs_tol = 1e-4, rel_tol = 1e-8,
                       preset = c("default", "bigboss"), max_hessian = 2, seed = 1L) {
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
  if (!is.logical(skip) || length(skip) != 1L || is.na(skip)) {
    stop("`skip` is TRUE or FALSE.", call. = FALSE)
  }
  if (!is.numeric(max_hessian) || length(max_hessian) != 1L || is.na(max_hessian) ||
      max_hessian <= 0) {
    stop("`max_hessian` is one positive number, got ", .describe(max_hessian), ".", call. = FALSE)
  }
  learner(
    name = "perceptron",
    data = data, reads = "tabular", multi = "separate",
    params = c(settings, list(skip = skip, abs_tol = as.numeric(abs_tol),
                              rel_tol = as.numeric(rel_tol), max_hessian = as.numeric(max_hessian),
                              seed = as.integer(seed))),
    fit = function(x, y, hidden, decay, range, max_iter, skip, abs_tol, rel_tol, max_hessian,
                   seed, head, weights, ...) {
      family <- .head_family(head)
      m <- .flatten(x)
      n_weight <- hidden * (ncol(m) + 1) + hidden + 1 + if (skip) ncol(m) else 0
      need <- n_weight * (n_weight + 1) / 2 * 8 / 2^30
      if (need > max_hessian) {
        stop(sprintf(paste0("A network of %d hidden units over %d columns has %d weights, and its ",
                            "approximate inverse Hessian would take %.1f GB, above `max_hessian = ",
                            "%g`. A coarser grain or fewer hidden units shrinks it."),
                     hidden, ncol(m), n_weight, need, max_hessian), call. = FALSE)
      }
      seeds <- .variable_seeds(seed, y)
      models <- lapply(seq_len(ncol(y)), function(j) {
        yj <- y[, j]
        if (length(unique(yj)) < 2L) {
          return(mean(yj))
        }
        .perceptron_fit(m, yj, weights[, j], family, hidden = hidden, decay = decay,
                        range = range, max_iter = max_iter, skip = skip, abs_tol = abs_tol,
                        rel_tol = rel_tol, seed = seeds[j])
      })
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

# The one-hidden-layer network, over the core `src/ts_perceptron.cpp` compiles into both
# languages. A fit is a plain list of its size, its family and its weights, so it round trips
# through `saveRDS()` and predicts on another machine. `start`, where given, replaces the drawn
# starting weights.
.perceptron_fit <- function(x, y, w, family, hidden = 2L, decay = 0, range = 0.7,
                            max_iter = 100L, skip = FALSE, abs_tol = 1e-4, rel_tol = 1e-8,
                            seed = 1L, start = NULL) {
  ts_perceptron_fit_(as.numeric(x), as.numeric(y), as.numeric(w), nrow(x), ncol(x), family,
                     as.integer(hidden), isTRUE(skip), as.numeric(decay), as.numeric(range),
                     as.integer(max_iter), as.numeric(abs_tol), as.numeric(rel_tol),
                     as.integer(seed), as.numeric(start %||% numeric(0)))
}

.perceptron_predict <- function(fit, newx) {
  ts_perceptron_predict_(fit, as.numeric(newx), nrow(newx), ncol(newx))
}
