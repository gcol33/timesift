#' Bayesian logistic model with a spatial field, through tulpa
#'
#' One model per response, fitted by the tulpa package: a logistic regression on every
#' bin-by-channel column of the representation, standardised so the prior on a coefficient means the
#' same thing for each, and optionally a Gaussian-process field over the targets' coordinates. A
#' unit then carries what its own record says and what its neighbours' presences say, under the
#' folds every other learner is scored on.
#'
#' The field is a Gaussian process on the two columns named by `coords` in [timesift()]:
#' `"nngp"` is the nearest-neighbour approximation and `"hsgp"` the Hilbert-space one, and the
#' field's hyperparameters are integrated by nested Laplace. A prediction at new units interpolates
#' the field to their coordinates, so [predict.timesift()] is given targets that carry the same
#' coordinate columns. Without a field no coordinates are needed and the model is a Bayesian
#' logistic regression, conditioned on the posterior mode by Laplace's method.
#'
#' tulpa's nested Laplace and its random-effect integrators carry no case weights on every route, so
#' a fit with a field or an intercept is unweighted whatever the response head weighs; without
#' either, the head's weights enter the likelihood. A response holding
#' one value is predicted its mean and named in `unfitted`. The learner fits a presence-absence
#' head: tulpa holds a Gaussian response's dispersion fixed unless it is estimated by empirical
#' Bayes, which a field does not allow.
#'
#' With `random = TRUE` each unit, as named by `id` in [timesift()], gets a random intercept, which
#' absorbs what its several targets share beyond the record: it is identified where a unit carries
#' more than one target, as an anchored fit's units do. Its standard deviation is estimated, not
#' conditioned on. A prediction adds a unit's intercept where
#' the unit was in the fit and leaves it at zero, the population level, where it was not, so a unit
#' held out whole is predicted from its record and its place alone.
#'
#' This learner is in the R package only, because tulpa is an R package.
#'
#' @inheritParams elasticnet
#' @param spatial `"none"` for no field, `"nngp"` or `"hsgp"` for a Gaussian-process field over
#'   the coordinates.
#' @param random A random intercept for each unit.
#' @param inference tulpa's `mode`. `NULL` takes `"laplace"` without a field or an intercept,
#'   `"eb"` with an intercept alone, which estimates its standard deviation by empirical Bayes, and
#'   `"auto"` with a field, which integrates the field's hyperparameters.
#'
#' @return A [learner()].
#'
#' @examples
#' hierarchical()
#' hierarchical(data = grain("month"), spatial = "hsgp")
#'
#' @export
hierarchical <- function(data = NULL, spatial = c("none", "nngp", "hsgp"), random = FALSE,
                         inference = NULL) {
  spatial <- match.arg(spatial)
  if (!is.logical(random) || length(random) != 1L || is.na(random)) {
    stop("`random` is TRUE or FALSE, got ", .describe(random), ".", call. = FALSE)
  }
  if (!is.null(inference) && (!is.character(inference) || length(inference) != 1L)) {
    stop("`inference` is one tulpa mode, or NULL, got ", .describe(inference), ".", call. = FALSE)
  }
  learner(
    name = "hierarchical",
    data = data, reads = "tabular", multi = "separate", needs = "tulpa",
    params = list(spatial = spatial, random = random, inference = inference),
    fit = function(x, y, spatial, random, inference, head, weights, ...) {
      if (!identical(.head_family(head), "binomial")) {
        stop("the hierarchical learner fits a presence-absence head.", call. = FALSE)
      }
      m <- .flatten(x)
      coords <- .tulpa_coords(x, spatial)
      units <- .tulpa_units(x, random)
      sd <- apply(m, 2L, stats::sd)
      keep <- is.finite(sd) & sd > 0
      centre <- colMeans(m)[keep]
      scale <- sd[keep]
      design <- .tulpa_design(m[, keep, drop = FALSE], centre, scale, coords, units)
      predictors <- names(design)[seq_len(sum(keep))]
      formula <- stats::reformulate(c(predictors, if (random) "(1 | ts_unit)"), response = "y")
      field <- if (identical(spatial, "none")) NULL else {
        tulpa::spatial_gp(~ ts_coord_1 + ts_coord_2,
                          approx = if (identical(spatial, "hsgp")) "hsgp")
      }
      mode <- inference %||% if (!is.null(field)) "auto" else if (random) "eb" else "laplace"
      fittable <- vapply(seq_len(ncol(y)), function(j) length(unique(y[, j])) > 1L, logical(1L))
      fits <- lapply(which(fittable), function(j) {
        design$y <- y[, j]
        args <- list(formula = formula, data = design, family = "binomial", mode = mode)
        if (is.null(field) && !random) {
          args$weights <- weights[, j]
        }
        args$spatial <- field
        fit <- do.call(tulpa::tulpa, args)
        list(fit = fit, effects = if (random) .tulpa_effects(fit) else NULL)
      })
      list(fits = fits, fittable = fittable, means = colMeans(y), keep = keep, centre = centre,
           scale = scale, columns = colnames(m)[keep], coords = colnames(coords),
           spatial = spatial, random = random, unfitted = colnames(y)[!fittable])
    },
    predict = function(model, x) {
      m <- .flatten(x)
      coords <- .tulpa_coords(x, model$spatial)
      units <- .tulpa_units(x, model$random)
      design <- .tulpa_design(m[, model$columns, drop = FALSE], model$centre, model$scale, coords,
                              units)
      out <- matrix(rep(model$means, each = nrow(m)), nrow(m), length(model$means))
      for (k in seq_along(model$fits)) {
        eta <- as.numeric(stats::predict(model$fits[[k]]$fit, design, type = "link"))
        if (model$random) {
          effect <- model$fits[[k]]$effects[units]
          eta <- eta + ifelse(is.na(effect), 0, effect)
        }
        out[, which(model$fittable)[k]] <- stats::plogis(eta)
      }
      .as_predictions(out, nrow(m))
    }
  )
}

# The coordinates a field is placed by, which `timesift(coords = )` carries on the array.
.tulpa_coords <- function(x, spatial) {
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
.tulpa_units <- function(x, random) {
  if (!random) {
    return(NULL)
  }
  units <- attr(x, "units")
  if (is.null(units)) {
    stop("a hierarchical learner with a random intercept names each target's unit; give the ",
         "identifier in `timesift(id = )`.", call. = FALSE)
  }
  units
}

# A unit's intercept, by the unit's name. tulpa names a term `ts_unit[<level>]`.
.tulpa_effects <- function(fit) {
  r <- tulpa::ranef(fit)
  prefix <- "ts_unit["
  stats::setNames(r$estimate, substr(r$term, nchar(prefix) + 1L, nchar(r$term) - 1L))
}

# Standardised predictors under names a formula can hold, and the field's coordinates and the
# units' labels beside them.
.tulpa_design <- function(m, centre, scale, coords, units = NULL) {
  z <- sweep(sweep(m, 2L, centre), 2L, scale, "/")
  out <- as.data.frame(z, check.names = FALSE, optional = TRUE)
  names(out) <- paste0("x", seq_len(ncol(z)))
  if (!is.null(coords)) {
    out$ts_coord_1 <- coords[, 1L]
    out$ts_coord_2 <- coords[, 2L]
  }
  if (!is.null(units)) {
    out$ts_unit <- factor(units)
  }
  out
}
