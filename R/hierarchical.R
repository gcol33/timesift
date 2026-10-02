#' Bayesian logistic model with a spatial field
#'
#' One model per response: a logistic regression on every bin-by-channel column of the
#' representation, standardised so the prior on a coefficient means the same thing for each,
#' optionally with an intercept for each unit and a Gaussian-process field over the targets'
#' coordinates. A unit then carries what its own record says and what its neighbours' presences
#' say, under the folds every other learner is scored on.
#'
#' The coefficients, the intercept included, have a `N(0, 2.5^2)` prior. A unit's intercept is
#' `N(0, sd^2)` and the field has a marginal standard deviation and a range, each standard
#' deviation under a penalised-complexity prior (Simpson et al. 2017) with
#' `P(sd > 3) = 0.01`, and the range under one anchored at a fifth of the coordinates' extent with
#' `P(range < anchor) = 0.5` (Fuglstad et al. 2019). Coordinates are centred and divided by one
#' factor, so distances keep their proportions.
#'
#' The field is a Gaussian process on the two columns named by `coords` in [timesift()]. `"hsgp"` is
#' the Hilbert-space approximation (Solin and Sarkka 2020) with `m` Laplacian eigenfunctions per
#' axis, and `"nngp"` the nearest-neighbour process (Datta et al. 2016) over the distinct
#' locations, each conditioned on its `neighbours` nearest among those before it in lexicographic
#' order of the coordinates, with the covariance `cov`, whose range is the field's. Its sparse
#' precision is factored once per set of hyperparameters, so the fit scales with the number of
#' locations rather than their square.
#'
#' Inference is Laplace's method over the coefficients, the intercepts and the field together, and
#' the hyperparameters are integrated over on a grid of `nodes` points per hyperparameter, centred
#' on the mode of their posterior and weighted by it (Rue, Martino and Chopin 2009). With an
#' intercept alone its standard deviation is set at the mode of its posterior; with neither the
#' fit is the posterior mode of the coefficients. A prediction at new units interpolates the field
#' to their coordinates, so [predict.timesift()] is given targets that carry the same coordinate
#' columns. The head's case weights enter the likelihood in every configuration. A response holding
#' one value is predicted its mean and named in `unfitted`. The learner fits a presence-absence
#' head.
#'
#' With `random = TRUE` each unit, as named by `id` in [timesift()], gets an intercept, which
#' absorbs what its several targets share beyond the record: it is identified where a unit carries
#' more than one target, as an anchored fit's units do. A prediction adds a unit's intercept where
#' the unit was in the fit and leaves it at zero, the population level, where it was not, so a unit
#' held out whole is predicted from its record and its place alone.
#'
#' Both languages call one C++ core, `src/ts_hierarchical.cpp`.
#'
#' @inheritParams elasticnet
#' @param spatial `"none"` for no field, `"nngp"` or `"hsgp"` for a Gaussian-process field over
#'   the coordinates.
#' @param random An intercept for each unit.
#' @param cov The covariance of the `"nngp"` field: `"exponential"`, `"matern32"`, `"matern52"` or
#'   `"gaussian"`, `sigma^2 exp(-d / range)` for the first.
#' @param neighbours The neighbours each location of an `"nngp"` field is conditioned on.
#' @param m Eigenfunctions per axis of an `"hsgp"` field.
#' @param boundary The factor by which an `"hsgp"` field's box is wider than the coordinates.
#' @param nodes Grid points per hyperparameter.
#'
#' @return A [learner()].
#'
#' @examples
#' hierarchical()
#' hierarchical(data = grain("month"), spatial = "hsgp")
#'
#' @export
hierarchical <- function(data = NULL, spatial = c("none", "nngp", "hsgp"), random = FALSE,
                         cov = c("exponential", "matern32", "matern52", "gaussian"),
                         neighbours = 15L, m = 6L, boundary = 1.5, nodes = 5L, threads = 1L) {
  spatial <- match.arg(spatial)
  cov <- match.arg(cov)
  if (!is.logical(random) || length(random) != 1L || is.na(random)) {
    stop("`random` is TRUE or FALSE, got ", .describe(random), ".", call. = FALSE)
  }
  .check_whole(neighbours, "neighbours", 1L, 200L)
  .check_whole(m, "m", 3L, 50L)
  .check_whole(nodes, "nodes", 1L, 99L)
  if (!is.numeric(boundary) || length(boundary) != 1L || is.na(boundary) || boundary < 1) {
    stop("`boundary` is one number of one or more, got ", .describe(boundary), ".", call. = FALSE)
  }
  learner(
    name = "hierarchical",
    data = data, reads = "tabular", multi = "separate",
    params = list(spatial = spatial, random = random, cov = cov, neighbours = as.integer(neighbours),
                  m = as.integer(m), boundary = boundary, nodes = as.integer(nodes),
                  threads = as.integer(threads)),
    fit = function(x, y, spatial, random, cov, neighbours, m, boundary, nodes, threads, head, ...) {
      if (!identical(.head_family(head), "binomial")) {
        stop("the hierarchical learner fits a presence-absence head.", call. = FALSE)
      }
      mat <- .flatten(x)
      weights <- .head_weights(head, y)
      coords <- .hier_coords(x, spatial)
      units <- .hier_units(x, random)
      sd <- apply(mat, 2L, stats::sd)
      keep <- is.finite(sd) & sd > 0
      centre <- colMeans(mat)[keep]
      scale <- sd[keep]
      design <- .hier_design(mat[, keep, drop = FALSE], centre, scale)
      levels <- if (random) sort(unique(units), method = "radix")
      index <- if (random) match(units, levels) - 1L
      fits <- lapply(seq_len(ncol(y)), function(j) {
        yj <- y[, j]
        if (length(unique(yj)) < 2L) {
          return(mean(yj))
        }
        .hier_fit(design, yj, weights[, j], unit = index, n_unit = length(levels), coords = coords,
                  field = spatial, m = m, boundary = boundary, neighbours = neighbours,
                  cov = match(cov, .hier_covs) - 1L, nodes = nodes, threads = threads)
      })
      list(fits = fits, columns = colnames(mat)[keep], centre = centre, scale = scale,
           levels = levels, spatial = spatial, random = random,
           unfitted = colnames(y)[vapply(fits, is.numeric, logical(1L))],
           stopped = colnames(y)[vapply(fits, function(f) is.list(f) && !f$converged,
                                        logical(1L))])
    },
    predict = function(model, x) {
      mat <- .flatten(x)
      design <- .hier_design(mat[, model$columns, drop = FALSE], model$centre, model$scale)
      coords <- .hier_coords(x, model$spatial)
      index <- if (model$random) {
        found <- match(.hier_units(x, TRUE), model$levels) - 1L
        ifelse(is.na(found), -1L, found)
      }
      .as_predictions(vapply(model$fits, function(f) {
        if (is.numeric(f)) rep(f, nrow(mat))
        else stats::plogis(.hier_predict(f, design, unit = index, coords = coords))
      }, numeric(nrow(mat))), nrow(mat))
    }
  )
}

.hier_covs <- c("exponential", "matern32", "matern52", "gaussian")

.check_whole <- function(v, name, lo, hi) {
  if (!is.numeric(v) || length(v) != 1L || is.na(v) || v != round(v) || v < lo || v > hi) {
    stop("`", name, "` is one whole number from ", lo, " to ", hi, ", got ", .describe(v), ".",
         call. = FALSE)
  }
}

# The coordinates a field is placed by, which `timesift(coords = )` carries on the array.
.hier_coords <- function(x, spatial) {
  if (identical(spatial, "none")) {
    return(NULL)
  }
  coords <- attr(x, "coords")
  if (is.null(coords)) {
    stop("a hierarchical learner with a `", spatial, "` field places each unit by its coordinates; ",
         "name the two columns in `timesift(coords = )`.", call. = FALSE)
  }
  coords
}

# The unit each target belongs to, which `timesift(id = )` carries on the array.
.hier_units <- function(x, random) {
  if (!random) {
    return(NULL)
  }
  units <- attr(x, "units")
  if (is.null(units)) {
    stop("a hierarchical learner with an intercept for each unit names each target's unit; give ",
         "the identifier in `timesift(id = )`.", call. = FALSE)
  }
  units
}

# The intercept's column, then the predictors centred and scaled by the fit's own.
.hier_design <- function(m, centre, scale) {
  cbind(1, sweep(sweep(m, 2L, centre), 2L, scale, "/"))
}

# The hierarchical model, over the core `src/ts_hierarchical.cpp` compiles into both languages.
# Nothing here decides anything: the design, the units' indices, the coordinates and the case
# weights are settled above, and what is left is to hand them over column-major and to read what
# comes back.
#
# A fit is a plain list of numbers, so it round trips through `saveRDS()` and predicts on another
# machine, which is what every other fitted object in the package is.

.hier_fit <- function(x, y, w, unit = NULL, n_unit = 0L, coords = NULL, field = "none",
                      beta_sd = 2.5, sd_u = 3, sd_alpha = 0.01, range_fraction = 0.2,
                      range_alpha = 0.5, m = 6L, boundary = 1.5, neighbours = 15L, cov = 0L,
                      nodes = 5L, step = 1.25, threads = 1L, theta = NULL) {
  ts_hierarchical_fit_(as.numeric(x), as.numeric(y), as.numeric(w), nrow(x), ncol(x),
                       if (is.null(unit)) NULL else as.integer(unit), as.integer(n_unit),
                       if (is.null(coords)) NULL else as.numeric(coords), field, beta_sd, sd_u,
                       sd_alpha, range_fraction, range_alpha, as.integer(m), boundary,
                       as.integer(neighbours), as.integer(cov), as.integer(nodes), step,
                       as.integer(threads), if (is.null(theta)) NULL else as.numeric(theta))
}

.hier_predict <- function(fit, newx, unit = NULL, coords = NULL) {
  ts_hierarchical_predict_(fit, as.numeric(newx), nrow(newx), ncol(newx),
                           if (is.null(unit)) NULL else as.integer(unit),
                           if (is.null(coords)) NULL else as.numeric(coords))
}
