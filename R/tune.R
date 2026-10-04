#' Tune a learner's settings on the training units
#'
#' Returns a learner that, whenever it is fitted, searches `grid` on the units it is handed and
#' fits the setting that scored best. The search is a cross-validation inside those units, so in
#' a run the outer folds never see it: each fold chooses from its own training units, and the score
#' it is then read at is not selected on. biomod2's `BIOMOD_Tuning()` searches a grid per algorithm
#' by the same device.
#'
#' A learner's settings are the ones it carries as `params`: the arguments of its constructor, as
#' `mtry` and `trees` are for [forest()]. `grid` names some of them and gives the values to try; the
#' grid is every combination. A value that is itself a vector, as the layer widths of [mlp()] are,
#' is given as an element of a list. The inner folds are dealt by [fold_map()] and keep the
#' grouping the outer fold map keeps whole. A setting is scored by the mean over the responses of
#' the mean over inner folds of `metric` on the cells a score is defined on, and ties go to the
#' first combination in the grid.
#'
#' What was chosen is recorded: on the fitted model as `$model$chosen` and `$model$table`, and in
#' the `settings` column of the candidate table of a run, which reads the model fitted on all
#' targets.
#'
#' With `grid` left unset the learner is searched over the grid registered under its name by
#' [register_tuning()], which for the learners that ship is the one `BIOMOD_Tuning()` searches:
#' `mtry` of a [forest()] from 1 to the smaller of 10 and the number of columns; `trees`, `depth`
#' and `shrinkage` of a gbm-style [boosting()], `shrinkage` and `colsample` of the second-order one;
#' `degree` and `nprune` of [mars()]; `degree` of [discriminant()]; `regmult` of [maxent()];
#' `quantile` of [envelope()]; and the layer width of [mlp()] at 2, 4, 6 and 8. biomod2's weight
#' decay is a training setting here, which [train_control()] holds and a grid does not reach.
#'
#' @param learner A [learner()], or the name of a registered one.
#' @param grid A named list of values to try, one element per setting, or `NULL` for the grid
#'   registered for the learner.
#' @param metric The registered metric, or a function of `(y, p)`, a setting is scored by. Left
#'   unset it is the response head's own.
#' @param n_inner Number of inner folds.
#' @param seed Random seed of the inner folds.
#'
#' @return A [learner()] reporting under the name of the one it wraps.
#'
#' @examples
#' tuned <- tune(forest(), list(mtry = c(2, 4), min_node = c(1, 5)), n_inner = 3L)
#' tuned
#'
#' @export
tune <- function(learner, grid = NULL, metric = NULL, n_inner = 5L, seed = 1L) {
  base <- .as_learner(learner)
  .check_count(n_inner, "n_inner", 2L)
  if (is.null(grid) && !.tuning_reg$has(base$name)) {
    stop("no grid is registered for the ", base$name, " learner. Give `grid`, or register one ",
         "with register_tuning(). Registered: ", .listing(tunings()), ".", call. = FALSE)
  }
  if (!is.null(grid)) {
    .tune_points(base, grid)
  }
  if (!is.null(metric)) {
    .as_metric(metric)
  }
  n_inner <- as.integer(n_inner)
  fit <- function(x, y, head, group = NULL, control = NULL, ...) {
    given <- list(...)
    points <- .tune_points(base, grid %||% .registered_grid(base, x))
    search <- .tune_search(base, points, x, y, head, control, group, given, metric, n_inner, seed)
    best <- points[[search$best]]
    kept <- given[setdiff(names(given), names(best))]
    model <- .call_fit(base, x, y, head, control, group, c(kept, best))
    structure(list(model = model, chosen = best, table = search$table),
              class = "timesift_tuned")
  }
  predict <- function(model, x) {
    base$predict(model$model, x)
  }
  learner(name = base$name, fit = fit, predict = predict, data = base$data, reads = base$reads,
          multi = base$multi, control = base$control, needs = base$needs, params = base$params)
}

# Every combination of the grid as a list of settings, the first setting varying fastest, after
# checking that each is a setting the learner carries.
.tune_points <- function(base, grid) {
  if (!is.list(grid) || is.null(names(grid)) || any(!nzchar(names(grid))) || !length(grid)) {
    stop("`grid` is a named list of the values to try for each setting.", call. = FALSE)
  }
  unknown <- setdiff(names(grid), names(base$params))
  if (length(unknown)) {
    stop("the ", base$name, " learner carries no setting called ", .listing(unknown),
         ". Its settings are ", .listing(names(base$params)), ".", call. = FALSE)
  }
  empty <- names(grid)[vapply(grid, length, integer(1L)) == 0L]
  if (length(empty)) {
    stop("`grid` holds no value for ", .listing(empty), ".", call. = FALSE)
  }
  index <- expand.grid(lapply(grid, seq_along), KEEP.OUT.ATTRS = FALSE)
  lapply(seq_len(nrow(index)), function(i) {
    stats::setNames(lapply(names(grid), function(nm) {
      v <- grid[[nm]]
      if (is.list(v)) v[[index[[nm]][i]]] else v[index[[nm]][i]]
    }), names(grid))
  })
}

# Each point of the grid cross-validated on the units handed to the fit, and the table of scores.
.tune_search <- function(base, points, x, y, head, control, group, given, metric, inner, seed) {
  n <- dim(x)[1L]
  folds <- fold_map(y, v = inner, seed = seed, strata = if (is.null(group)) 5L else 1L,
                    group = group)
  f <- as.integer(unclass(folds))
  levels <- sort(unique(f))
  cells <- head$cells(y, folds)
  score <- .as_metric(metric, head$metric)$fn
  scores <- vapply(points, function(point) {
    p <- matrix(NA_real_, nrow = n, ncol = ncol(y), dimnames = dimnames(y))
    settings <- c(given[setdiff(names(given), names(point))], point)
    for (k in levels) {
      train <- which(f != k)
      test <- which(f == k)
      model <- .call_fit(base, .subset_units(x, train), y[train, , drop = FALSE], head, control,
                         if (!is.null(group)) group[train], settings)
      held <- as.matrix(base$predict(model, .subset_units(x, test)))
      p[test, ] <- held
    }
    table <- .score_cells(y, p, f, levels, cells, score)
    per_response <- .cell_means(table)
    if (nrow(per_response)) mean(per_response$score) else NA_real_
  }, numeric(1L))
  if (all(is.na(scores))) {
    stop("no setting of the grid has a cell to be scored on: every response needs both classes ",
         "on each side of an inner split. Fewer inner folds, or a response present somewhere.",
         call. = FALSE)
  }
  labels <- vapply(points, .tune_label, character(1L))
  list(best = which.max(scores),
       table = data.frame(settings = labels, score = scores, stringsAsFactors = FALSE))
}

.tune_label <- function(point) {
  paste(names(point), vapply(point, .describe, character(1L)), sep = " = ", collapse = ", ")
}

# The setting a tuned model chose, as the candidate table prints it; NA for a model that was not
# tuned.
.chosen_settings <- function(fit) {
  model <- fit$model
  if (inherits(model, "timesift_tuned")) .tune_label(model$chosen) else NA_character_
}

.tuning_reg <- .new_registry("tuning grid")

#' Register the grid a learner is tuned over
#'
#' Makes `tune(learner)` search `grid` when it is given no grid of its own. The grids of the
#' learners that ship are registered the same way.
#'
#' @param name Name of the learner the grid belongs to, as it reports under.
#' @param grid A named list of values to try, or a function of `(learner, x)` returning one, where
#'   `x` is the representation the learner is fitted on. The second form is for a grid that depends
#'   on the data, as the number of columns does, or on a setting the learner carries.
#' @param overwrite Replace an existing registration.
#'
#' @return The grid, invisibly.
#'
#' @examples
#' register_tuning("flat_glm", list(thresh = c(1e-4, 1e-6)), overwrite = TRUE)
#' tunings()
#'
#' @export
register_tuning <- function(name, grid, overwrite = FALSE) {
  if (!is.function(grid) && !(is.list(grid) && length(grid) && !is.null(names(grid)))) {
    stop("a tuning grid is a named list of values, or a function of (learner, x) returning one.",
         call. = FALSE)
  }
  .tuning_reg$set(name, grid, overwrite)
}

#' @rdname register_tuning
#' @export
tunings <- function() .tuning_reg$names()

.registered_grid <- function(base, x) {
  grid <- .tuning_reg$get(base$name)
  if (is.function(grid)) grid(base, x) else grid
}

# The grids BIOMOD_Tuning() searches, on the settings the learners carry.
.default_grids <- function() {
  list(
    forest = function(learner, x) list(mtry = seq_len(min(10L, ncol(.flatten(x))))),
    boosting = function(learner, x) {
      if (identical(learner$params$method, "xgboost")) {
        list(trees = 50L, depth = 1L, shrinkage = c(0.3, 0.4), min_leaf = 1, subsample = 0.5,
             colsample = c(0.6, 0.8), gamma = 0)
      } else {
        list(trees = c(500L, 1000L, 2500L), depth = c(2L, 5L, 8L),
             shrinkage = c(0.001, 0.01, 0.1))
      }
    },
    mars = function(learner, x) list(degree = 1:2, nprune = 2:max(21L, 2L * ncol(.flatten(x)) + 1L)),
    discriminant = list(degree = 1:2),
    maxent = list(regmult = c(0.5, 1)),
    envelope = list(quantile = c(0, 0.0125, 0.025, 0.05, 0.1)),
    mlp = list(hidden = list(2L, 4L, 6L, 8L))
  )
}
