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
#' @param learner A [learner()], or the name of a registered one.
#' @param grid A named list of values to try, one element per setting.
#' @param metric The registered metric, or a function of `(y, p)`, a setting is scored by. Left
#'   unset it is the response head's own.
#' @param inner Number of inner folds.
#' @param seed Random seed of the inner folds.
#'
#' @return A [learner()] reporting under the name of the one it wraps.
#'
#' @examples
#' tuned <- tune(forest(), list(mtry = c(2, 4), min_node = c(1, 5)), inner = 3L)
#' tuned
#'
#' @export
tune <- function(learner, grid, metric = NULL, inner = 5L, seed = 1L) {
  base <- .as_learner(learner)
  .check_count(inner, "inner", 2L)
  points <- .tune_points(base, grid)
  if (!is.null(metric)) {
    .as_metric(metric)
  }
  inner <- as.integer(inner)
  fit <- function(x, y, head, group = NULL, control = NULL, ...) {
    given <- list(...)
    search <- .tune_search(base, points, x, y, head, control, group, given, metric, inner, seed)
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
