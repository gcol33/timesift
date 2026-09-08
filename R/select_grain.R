#' Choose the grain inside the training data, and score the whole procedure
#'
#' [grain_ladder()] fits every candidate against one fold map and reports the grid, so reading the
#' best grain off it and quoting that grain's score quotes a number the held-out units helped
#' choose. This does the choosing inside the training data instead. Within each outer fold the
#' training units are split again, every candidate is fitted on part of them and scored on the rest,
#' the best is refitted on the whole outer training set, and the outer test fold is predicted once.
#' The estimate that comes back is therefore of the procedure including its choice of grain, which
#' is what an ecologist applying it to a new site would run.
#'
#' A candidate is a `(grain, learner)` pair: the grains are the elements of the representation set,
#' which is where a grain and the statistic its grains are summarised by are both named, and the
#' learners are the ones passed. Both are registry entries or objects built by [learner()], so a new
#' grain, a new grain summary or a new candidate model widens the search with no change here.
#'
#' What the estimate is of: the expected held-out score of the whole pipeline, selection included,
#' on units drawn as these were. What it is not: the score of the winning grain. That is higher, by
#' the amount selection buys itself, and the difference between the two is the quantity this
#' function exists to keep out of a reported number. It also does not say the selected grain is the
#' one a mechanism acts at; it says that grain predicted best on the units the selector saw.
#'
#' The cost is the ladder's, multiplied by the number of inner folds: `v_outer * (v_inner *
#' candidates + 1)` fits. With a neural learner that is where an overnight run goes.
#'
#' @param x A [grain_matrix()] result, a [timesift_set()], or a named list of representations.
#'   Its names are the grains being chosen between.
#' @param y The response for the same units.
#' @param learners A learner, a list of them, or names of registered ones, as [grain_ladder()]
#'   takes. Named alongside the grains they form the candidate set.
#' @param folds The outer fold map, from [fold_map()] or any named integer vector. Built with the
#'   defaults of [fold_map()] when not given.
#' @param inner Number of inner folds the selection is made on, or a function of the outer training
#'   response returning a fold map for those units. A count deals the inner folds by the grouping
#'   the outer fold map carries, so what [grouped_cv()] kept whole outside stays whole inside.
#' @param response Name of the registered response head.
#' @param metric Name of a registered metric the selection is made on, or `NULL` for the
#'   response's own. The estimate is reported under every registered metric whichever this is.
#' @param compare A [grain_ladder()] result on the same units, response and outer fold map, whose
#'   arms the selected procedure is contrasted against cell by cell. `NULL` for no contrast.
#' @param control [train_control()], the training settings every neural learner reads, in the
#'   inner search and in the refit alike. A learner carrying a control of its own overrides it on
#'   the settings that control names.
#' @param seed Seed for the inner splits. Each outer fold splits under `seed` plus its own number,
#'   so no two outer folds inherit the same inner partition.
#' @param verbose Report each outer fold and what it selected as it runs.
#'
#' @return A `timesift_selection`: a list carrying `selected`, one row per outer fold with the
#'   candidate it chose and the inner score it chose on; `estimate`, the nested score under every
#'   registered metric with its standard error across variables; `contrast`, one
#'   [paired_contrast()] row against each arm of `compare`, or `NULL`; `candidates`, the set that
#'   was searched; and `scores`, the per-cell rows of the selected procedure under the selection
#'   metric, in the layout [grain_ladder()] returns. The held-out prediction of every unit is in
#'   the `predictions` attribute and the scorable-cell mask in `cells`.
#'
#' @seealso [grain_ladder()] for the grid this selects from, and [paired_contrast()] for the
#'   comparison the `contrast` element holds.
#'
#' @examplesIf requireNamespace("glmnet", quietly = TRUE)
#' set.seed(1)
#' t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24 * 200)
#' units <- sprintf("p%02d", 1:60)
#' warmth <- rnorm(60)
#' d <- data.frame(
#'   plot = rep(units, each = length(t)), t = rep(t, length(units)),
#'   temp = as.numeric(vapply(warmth, function(w) w + sin(seq_along(t) / 300) + rnorm(length(t)),
#'                            numeric(length(t)))))
#' y <- matrix(rbinom(120, 1, plogis(c(warmth, -warmth))), nrow = 60,
#'             dimnames = list(units, c("sp1", "sp2")))
#' x <- grain_matrix(d, plot, t, temp, grain = c("week", "month"))
#' sel <- select_grain(x, y, elasticnet(), folds = fold_map(y, v = 3), inner = 3,
#'                     verbose = FALSE)
#' sel
#' sel$estimate
#'
#' @export
select_grain <- function(x, y, learners, folds = NULL, inner = 5L,
                         response = "presence_absence", metric = NULL, compare = NULL,
                         control = train_control(), seed = 1L, verbose = TRUE) {
  set <- .as_set(x)
  units <- dimnames(set[[1L]])[[1L]]
  spec <- .responses_reg$get(response)
  y <- .align_response(spec$prepare(y), units)
  if (is.null(folds)) {
    folds <- fold_map(y)
  }
  f <- .as_folds(folds, units)
  cells <- spec$cells(y, stats::setNames(f, units))
  # The estimate is reported under every registered metric, and one selected on has to be a row
  # of that table, so this is the one door a function of (y, p) does not go through.
  if (is.function(metric)) {
    stop("`select_grain()` reports the estimate under every registered metric, so the one it ",
         "selects on has to be registered. register_metric() takes a function of (y, p).",
         call. = FALSE)
  }
  metric <- metric %||% spec$metric
  score <- .metrics_reg$get(metric)
  learners <- .learner_list(learners)
  group <- .fold_group(folds, units)
  inner_split <- .inner_splitter(inner, group)
  .check_compare(compare, metric)
  candidates <- expand.grid(grain = names(set), learner = names(learners),
                            KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  if (nrow(candidates) < 2L) {
    stop("selection needs at least two candidates; got one grain and one learner.", call. = FALSE)
  }

  levels <- sort(unique(f))
  p <- matrix(NA_real_, nrow = length(units), ncol = ncol(y),
              dimnames = list(units, colnames(y)))
  chosen <- vector("list", length(levels))
  inner_scores <- vector("list", length(levels))

  for (i in seq_along(levels)) {
    k <- levels[i]
    train <- which(f != k)
    test <- which(f == k)
    y_train <- y[train, , drop = FALSE]

    # The selector sees the outer training units and nothing else: the inner map is drawn on them,
    # and the representation it searches over is cut to them before any fitting happens.
    lad <- grain_ladder(.subset_set(set, train), y_train, learners,
                         folds = inner_split(y_train, seed + i, train), response = response,
                         metric = metric, control = control, verbose = FALSE)
    grid <- .join_candidates(candidates, summary(lad), k)
    if (all(!is.finite(grid$score))) {
      stop("no candidate scored inside the training data of fold ", k,
           ". Widen the inner folds or drop the variables that cannot be scored.", call. = FALSE)
    }
    won <- which.max(ifelse(is.finite(grid$score), grid$score, -Inf))

    # The refit is the inner ladder's own fitting path, so the procedure's held-out predictions are
    # the ones its chosen candidate would have made rather than a second fitting path's.
    fit <- fit_learner(learners[[grid$learner[won]]],
                       .subset_units(set[[grid$grain[won]]], train), y_train,
                       response = response, control = control, group = group[train])
    held_out <- stats::predict(fit, .subset_units(set[[grid$grain[won]]], test))
    p[rownames(held_out), colnames(held_out)] <- held_out

    chosen[[i]] <- data.frame(fold = k, grain = grid$grain[won], learner = grid$learner[won],
                              inner_score = grid$score[won], n_train = length(train),
                              n_test = length(test), stringsAsFactors = FALSE)
    inner_scores[[i]] <- grid[c("fold", "grain", "learner", "score", "n_variable")]
    if (verbose) {
      message(sprintf("fold %s of %d selected %s|%s at %s %.3f", k, length(levels),
                      grid$grain[won], grid$learner[won], metric, grid$score[won]))
    }
  }

  selected <- do.call(rbind, chosen)
  scores <- .as_grain_rows(.score_arm(.selected_label, "selected", y, p, f, levels, cells, score))
  scores <- structure(scores, class = c("timesift_ladder", "data.frame"),
                      predictions = stats::setNames(list(p), .selected_arm),
                      cells = cells, folds = stats::setNames(f, units),
                      metric = metric, scorer = score, response = response)

  out <- list(
    selected = selected,
    estimate = .nested_estimate(y, p, f, levels, cells),
    contrast = .selection_contrast(scores, compare),
    candidates = candidates,
    scores = scores,
    inner = do.call(rbind, inner_scores)
  )
  structure(out, class = "timesift_selection", metric = metric, response = response,
            folds = stats::setNames(f, units), cells = cells,
            predictions = stats::setNames(list(p), .selected_arm))
}

# The selected procedure is one arm like any other, so it carries an arm label of the ladder's own
# shape and every reader that splits on "|" keeps working.
.selected_label <- "selected"
.selected_arm <- "selected|selected"

#' @export
print.timesift_selection <- function(x, ...) {
  cat("<timesift selection>", .plural(nrow(x$selected), "outer fold"), "over",
      .plural(nrow(x$candidates), "candidate"), "\n")
  est <- x$estimate[x$estimate$metric == attr(x, "metric"), , drop = FALSE]
  cat(sprintf("%s: %.3f (se %.3f) for the procedure, selection included\n",
              attr(x, "metric"), est$score, est$se))
  print(summary(x))
  invisible(x)
}

#' @param object A selection.
#' @param ... Ignored.
#' @rdname select_grain
#' @export
summary.timesift_selection <- function(object, ...) {
  out <- object$candidates
  key <- paste(out$grain, out$learner)
  picked <- paste(object$selected$grain, object$selected$learner)
  out$n_selected <- as.integer(table(factor(picked, levels = key)))
  out$share <- out$n_selected / nrow(object$selected)
  inner <- object$inner
  mean_inner <- tapply(inner$score, paste(inner$grain, inner$learner), mean, na.rm = TRUE)
  out$inner_score <- as.numeric(mean_inner[key])
  out <- out[order(-out$n_selected, -out$inner_score), ]
  rownames(out) <- NULL
  out
}

#' Draw how stable the choice of grain was
#'
#' Every candidate's inner score in every outer fold, one line per outer fold, with an open circle
#' on the candidate that fold selected. A selection that lands on the same candidate each time draws
#' its circles in one column; one that wanders says the grid is flat enough that the choice is
#' arbitrary, which is worth seeing beside the estimate rather than after it.
#'
#' @param x A [select_grain()] result.
#' @param col One colour per outer fold, recycled.
#' @param ... Passed to [graphics::plot()].
#'
#' @return The table of inner scores the plot is drawn from, invisibly.
#'
#' @examplesIf requireNamespace("glmnet", quietly = TRUE)
#' set.seed(1)
#' t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24 * 200)
#' units <- sprintf("p%02d", 1:60)
#' warmth <- rnorm(60)
#' d <- data.frame(
#'   plot = rep(units, each = length(t)), t = rep(t, length(units)),
#'   temp = as.numeric(vapply(warmth, function(w) w + sin(seq_along(t) / 300) + rnorm(length(t)),
#'                            numeric(length(t)))))
#' y <- matrix(rbinom(120, 1, plogis(c(warmth, -warmth))), nrow = 60,
#'             dimnames = list(units, c("sp1", "sp2")))
#' x <- grain_matrix(d, plot, t, temp, grain = c("week", "month"))
#' sel <- select_grain(x, y, elasticnet(), folds = fold_map(y, v = 3), inner = 3,
#'                     verbose = FALSE)
#' plot(sel)
#'
#' @export
plot.timesift_selection <- function(x, col = NULL, ...) {
  inner <- x$inner
  label <- paste(x$candidates$grain, x$candidates$learner, sep = "|")
  at <- seq_along(label)
  folds <- sort(unique(inner$fold))
  if (is.null(col)) {
    col <- grDevices::hcl.colors(max(length(folds), 2L), "Dark 3")[seq_along(folds)]
  }
  col <- rep_len(col, length(folds))

  span <- range(inner$score[is.finite(inner$score)])
  args <- list(x = at, y = rep(NA_real_, length(at)), ylim = span, xaxt = "n",
               xlab = "candidate", ylab = paste(attr(x, "metric"), "inside the training data"))
  do.call(graphics::plot, utils::modifyList(args, list(...)))
  graphics::axis(1L, at = at, labels = label, las = 2L, cex.axis = 0.8)
  graphics::grid(nx = NA, ny = NULL, col = "grey90", lty = 1L)

  for (i in seq_along(folds)) {
    rows <- inner[inner$fold == folds[i], , drop = FALSE]
    rows <- rows[match(label, paste(rows$grain, rows$learner)), , drop = FALSE]
    graphics::lines(at, rows$score, col = col[i], lwd = 1.5)
    graphics::points(at, rows$score, col = col[i], pch = 19L, cex = 0.8)
    won <- x$selected[x$selected$fold == folds[i], , drop = FALSE]
    mark <- match(paste(won$grain, won$learner), label)
    graphics::points(at[mark], rows$score[mark], col = col[i], pch = 1L, cex = 2.2, lwd = 2)
  }
  invisible(inner)
}

# Every registered metric reads the same held-out predictions, so the estimate is reported under all
# of them and the choice of selection metric does not decide what may be quoted.
.nested_estimate <- function(y, p, f, levels, cells) {
  out <- lapply(metrics(), function(nm) {
    rows <- .score_arm(.selected_label, "selected", y, p, f, levels, cells,
                       .metrics_reg$get(nm))
    per_variable <- .cell_means(rows)
    ms <- .mean_se(per_variable$score)
    data.frame(metric = nm, score = ms[1L], se = ms[2L], n_variable = nrow(per_variable),
               stringsAsFactors = FALSE)
  })
  out <- do.call(rbind, out)
  rownames(out) <- NULL
  out
}

# The contrast is the ladder's, run on one table holding both arms, so the pairing rule and the
# interval come from paired_contrast() rather than from a second copy of it here.
.check_compare <- function(compare, metric) {
  if (is.null(compare)) {
    return(invisible(TRUE))
  }
  if (!inherits(compare, "timesift_ladder")) {
    stop("`compare` is a grain_ladder() result, got ", class(compare)[1L], ".", call. = FALSE)
  }
  if (!identical(attr(compare, "metric"), metric)) {
    stop("`compare` is scored by ", attr(compare, "metric"), " and the selection by ", metric,
         ". Score both by the same metric before contrasting them.", call. = FALSE)
  }
  invisible(TRUE)
}

.selection_contrast <- function(scores, compare) {
  if (is.null(compare)) {
    return(NULL)
  }
  shared <- intersect(names(scores), names(compare))
  both <- rbind(scores[shared], compare[shared])
  both <- structure(both, class = c("timesift_ladder", "data.frame"),
                    metric = attr(scores, "metric"))
  arms <- unique(paste(compare$grain, compare$learner, sep = "|"))
  out <- lapply(arms, function(a) paired_contrast(both, .selected_arm, a))
  out <- do.call(rbind, out)
  rownames(out) <- NULL
  out
}

# The candidate set keeps the order its grains and its learners were declared in, so which
# candidate an exact tie on the inner score falls to does not depend on the session's collation the
# way a join on the names would.
.join_candidates <- function(candidates, grid, fold) {
  i <- match(paste(candidates$grain, candidates$learner, sep = "|"),
             paste(grid$grain, grid$learner, sep = "|"))
  candidates$score <- grid$score[i]
  candidates$n_variable <- grid$n_variable[i]
  candidates$fold <- fold
  candidates
}

# The inner map is drawn on the outer training units alone, either by fold_map() at a given count,
# dealing by the grouping the outer map carries, or by a splitter of the caller's own.
.inner_splitter <- function(inner, group = NULL) {
  if (is.function(inner)) {
    return(function(y_train, seed, train) inner(y_train))
  }
  if (!is.numeric(inner) || length(inner) != 1L || is.na(inner) || inner < 2L ||
        inner != trunc(inner)) {
    stop("`inner` is a number of folds of at least 2, or a function of the training response, got ",
         paste(deparse(inner), collapse = ""), ".", call. = FALSE)
  }
  inner <- as.integer(inner)
  function(y_train, seed, train) {
    fold_map(y_train, v = inner, seed = seed, strata = if (is.null(group)) 5L else 1L,
             group = group[train])
  }
}

.subset_set <- function(set, idx) {
  timesift_set(lapply(set, .subset_units, idx = idx))
}
