#' Fit and compare representations of time-varying data
#'
#' One call from two tables to a scored comparison. `targets` is one row per thing to predict and
#' `series` is the long, time-stamped record belonging to those rows. Every representation in
#' `sift` is built, every learner in `models` is fitted on the ones it can read, each on the same
#' folds and restricted to the same scorable cells, and the out-of-fold predictions are stacked
#' into an ensemble. What comes back says where predictive skill saturates as the record is read
#' more coarsely, which is the measurement the package exists for.
#'
#' @param targets A data frame, one row per prediction target.
#' @param series A long data frame of readings, or `NULL` to fit on `static` alone.
#' @param y Columns of `targets` holding the response, as a tidyselect expression such as
#'   `starts_with("sp_")`.
#' @param x Columns of `series` holding the readings, as a tidyselect expression. Defaults to
#'   every numeric column but `id` and `time`.
#' @param id Column naming the unit, present in both tables. A bare column name or a string.
#' @param time Column of reading instants in `series`, `POSIXct`.
#' @param target_time Column of `targets` anchoring each row in time, `POSIXct`. Optional, and
#'   what a unit carrying several targets through time needs.
#' @param static Columns of `targets` carried alongside the representation, as a tidyselect
#'   expression. None by default.
#' @param models A learner, a set of them from [c()], or a list. Defaults to [elasticnet()].
#' @param sift The representations a learner without a `data =` of its own is run across.
#'   A [grains()] or [lookbacks()] set, a set from [c()], a bare vector of grain names, a single
#'   representation, or a list of them. Defaults to `grains("auto")`.
#' @param ensemble `TRUE` for the default stack, `FALSE` for none, or an [ensemble()] spec.
#' @param resampling [cv()], [grouped_cv()], a fold vector, or a [fold_map()] result.
#' @param response Name of the registered response head.
#' @param metric Name of a registered metric, or a function of `(y, p)`, or `NULL` for the
#'   response head's own. Whichever it is, it travels with the fit and is what every later
#'   rescoring reads; a function is reported as `<function>`.
#' @param control [train_control()], the training settings every neural learner reads.
#' @param keep_fits Keep every per-fold fitted candidate beside the refits.
#' @param verbose Report each candidate as it runs.
#'
#' @section Rules the entry point enforces:
#' `static` is never implicit: a column of `targets` that is neither the response, the identifier
#' nor the anchor is ignored unless `static` names it, because a predictor nobody asked for is
#' worse than one that is missing.
#'
#' One target row per `id`, unless `target_time` says where in time each row sits. Repeated
#' identifiers without an anchor are an error naming them.
#'
#' With `target_time`, every representation has to be anchored on the target, so [native()],
#' [grain()] and [multigrain()] are refused and `sift` must be given as [lookbacks()]. There is no
#' default set of spans, because there is no defensible one.
#'
#' Without `series`, `static` is the whole predictor block and `sift` is ignored.
#'
#' @section What a learner may be handed:
#' A learner declares whether it reads a tabular block or a sequence. A tabular learner given
#' [native()] is refused before anything is built, and a sequence learner given a representation of
#' one bin is refused once the array says how many bins it has. Inside a `sift` expansion such a
#' pair is skipped and reported once by name; named explicitly through a learner's `data =` it is
#' an error.
#'
#' @return A `timesift` object: a list carrying `candidates`, `scores`, `oof`, `representations`,
#'   `stack`, `weights`, `models`, `folds`, `cells`, `y`, and the `metric`, `response`, `spec` and
#'   `call` it was asked for.
#'
#' @seealso [build_representation()] for the array a candidate reads, [fold_map()] for the splits.
#'
#' @examples
#' set.seed(1)
#' t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24 * 90)
#' units <- sprintf("p%02d", 1:30)
#' warmth <- rnorm(30)
#' logger <- data.frame(
#'   plot = rep(units, each = length(t)), datetime = rep(t, 30),
#'   temp = as.numeric(vapply(warmth, function(w) w + sin(seq_along(t) / 300), numeric(length(t)))))
#' plots <- data.frame(plot = units,
#'                     sp_a = rbinom(30, 1, plogis(2 * warmth)),
#'                     sp_b = rbinom(30, 1, plogis(-2 * warmth)))
#' \donttest{
#' fit <- timesift(plots, logger, y = starts_with("sp_"), id = plot, time = datetime,
#'                 sift = grains("week", "month"), resampling = cv(v = 3L),
#'                 ensemble = FALSE, verbose = FALSE)
#' fit
#' }
#'
#' @export
timesift <- function(targets, series = NULL, y, x = NULL, id = NULL, time = NULL,
                     target_time = NULL, static = NULL,
                     models = NULL, sift = NULL, ensemble = TRUE,
                     resampling = cv(), response = "presence_absence", metric = NULL,
                     control = train_control(), keep_fits = FALSE, verbose = TRUE) {
  call <- match.call()
  env <- parent.frame()
  if (!is.data.frame(targets)) {
    stop("`targets` must be a data frame, one row per prediction target, got ",
         class(targets)[1L], ".", call. = FALSE)
  }
  if (!is.null(series) && !is.data.frame(series)) {
    stop("`series` must be a long data frame of readings, or NULL, got ", class(series)[1L], ".",
         call. = FALSE)
  }

  id_col <- .optional_column(substitute(id), targets, env)
  target_time_col <- .optional_column(substitute(target_time), targets, env)
  time_col <- if (is.null(series)) NULL else .optional_column(substitute(time), series, env)
  y_cols <- .select_columns(rlang::enquo(y), targets, "y")
  static_cols <- .select_columns(rlang::enquo(static), targets, "static")

  if (!length(y_cols)) {
    stop("`y` must name the response column", if (ncol(targets) > 1L) "(s)" else "",
         " in `targets`.", call. = FALSE)
  }
  clash <- intersect(static_cols, c(y_cols, id_col, target_time_col))
  if (length(clash)) {
    stop("`static` names ", .listing(clash), ", which the response, the identifier or the anchor ",
         "already carries. A column is a predictor or it is one of those, not both.",
         call. = FALSE)
  }
  if (!is.null(series)) {
    if (is.null(id_col)) {
      stop("`id` must name the column linking `targets` to `series`.", call. = FALSE)
    }
    if (is.null(time_col)) {
      stop("`time` must name the column of reading instants in `series`.", call. = FALSE)
    }
    if (!id_col %in% names(series)) {
      stop("`", id_col, "` is in `targets` but not in `series`, so no target reaches a record.",
           call. = FALSE)
    }
  } else if (!length(static_cols)) {
    stop("without `series` there is nothing to fit on unless `static` names predictors in ",
         "`targets`.", call. = FALSE)
  }
  if (!is.null(target_time_col) && !inherits(targets[[target_time_col]], "POSIXct")) {
    stop("`", target_time_col, "` must be POSIXct, not ", class(targets[[target_time_col]])[1L],
         ".", call. = FALSE)
  }

  value_cols <- .series_values(rlang::enquo(x), series, id_col, time_col)
  spec <- list(id = id_col, time = time_col, value = value_cols, target_time = target_time_col,
               static = static_cols, y = y_cols, partial = "keep", response = response,
               control = control, sift = NULL)
  tf <- .target_frame(targets, spec)
  targets <- targets[tf$order, , drop = FALSE]
  .check_targets_unique(tf, spec)
  if (!is.null(series)) {
    .check_series_reaches(tf, series, spec)
  }

  head <- .responses_reg$get(response)
  y_matrix <- head$prepare(.response_block(targets, tf, y_cols))
  # Refused here rather than after the fitting, because a contradiction between the run's head and
  # the combiner's is not worth a grid of fits to find out about.
  if (!isFALSE(ensemble)) {
    ensemble <- .run_ensemble(ensemble, response)
  }

  learners <- .learner_list(models %||% list(elasticnet()))
  for (ln in names(learners)) .learner_contract(learners[[ln]], ln)
  .refuse_pinned(learners)
  if (is.null(series) && !is.null(sift) && verbose) {
    message("without `series` the predictor block is `static`, so `sift` is ignored.")
  }
  sift <- .sift_specs(sift, series)
  .check_anchored(sift, learners, spec)

  store <- .build_sift(sift, series, targets, spec, learners, verbose)
  spec$sift <- store$sift
  built <- timesift_set(store$built)
  grid <- .candidate_grid(learners, store$sift, built, store$labels)
  fitted <- grid[grid$status == "fitted", , drop = FALSE]
  if (!nrow(fitted)) {
    stop("no learner can read any representation in the sift:\n  ",
         paste(grid$note, collapse = "\n  "), call. = FALSE)
  }
  if (verbose && any(grid$status != "fitted")) {
    message("skipping ", .plural(sum(grid$status != "fitted"), "candidate"),
            " no learner can read: ", .listing(grid$candidate[grid$status != "fitted"]))
  }

  folds <- .as_fold_map(resampling, y_matrix, targets, tf)
  f <- .as_folds(folds, tf$label)
  cells <- head$cells(y_matrix, folds)
  metric <- .as_metric(metric, head$metric)
  score <- metric$fn
  levels <- sort(unique(f))

  oof <- list()
  scores <- list()
  models_out <- list()
  fits <- list()
  for (i in seq_len(nrow(fitted))) {
    cand <- fitted$candidate[i]
    learner <- learners[[fitted$learner[i]]]
    x_array <- built[[fitted$representation[i]]]
    if (verbose) {
      message("fitting ", fitted$learner[i], " on the ", fitted$representation[i],
              " representation")
    }
    run <- .fit_candidate(learner, x_array, y_matrix, f, levels, response, control, keep_fits,
                          verbose)
    oof[[cand]] <- run$oof
    scores[[cand]] <- .candidate_scores(cand, fitted$representation[i], fitted$learner[i],
                                        y_matrix, run$oof, f, levels, cells, score)
    if (keep_fits) {
      fits[[cand]] <- run$fits
    }
    models_out[[cand]] <- .fit_candidate_once(learner, x_array, y_matrix, response, control)
  }
  scores <- do.call(rbind, scores)
  rownames(scores) <- NULL

  stack <- NULL
  if (!isFALSE(ensemble)) {
    if (length(oof) < 2L) {
      if (verbose) {
        message("no ensemble: stacking needs at least two candidates.")
      }
    } else {
      stack <- ensemble_fit(oof = oof, y = y_matrix, cells = cells, folds = folds,
                            spec = ensemble, scores = scores)
    }
  }

  structure(list(candidates = grid, scores = scores, oof = oof, representations = built,
                 stack = stack, weights = stack$weights, models = models_out,
                 fits = if (keep_fits) fits else NULL,
                 folds = folds, cells = cells, y = y_matrix,
                 metric = metric$name, scorer = score, response = response, spec = spec,
                 call = call),
            class = "timesift")
}

# The value columns of the series. A caller who names them is held to them; a caller who does not
# gets the numeric ones, because the default cannot know that a text column is a note rather than
# a reading and an explicit choice is a decision rather than a guess.
.series_values <- function(quo, series, id_col, time_col) {
  if (is.null(series)) {
    return(character())
  }
  chosen <- .select_columns(quo, series, "x")
  if (!length(chosen)) {
    rest <- setdiff(names(series), c(id_col, time_col))
    chosen <- rest[vapply(series[rest], is.numeric, logical(1L))]
    if (!length(chosen)) {
      stop("`series` carries no numeric column beside `", id_col, "` and `", time_col,
           "`. Name the readings with `x`.", call. = FALSE)
    }
    return(chosen)
  }
  clash <- intersect(chosen, c(id_col, time_col))
  if (length(clash)) {
    stop("`x` names ", .listing(clash), ", which is the identifier or the time column.",
         call. = FALSE)
  }
  bad <- chosen[!vapply(series[chosen], is.numeric, logical(1L))]
  if (length(bad)) {
    stop("`x` names ", .listing(bad), ", which ", if (length(bad) > 1L) "are" else "is",
         " not numeric.", call. = FALSE)
  }
  chosen
}

.check_targets_unique <- function(tf, spec) {
  if (is.null(tf$id) || !is.null(spec$target_time)) {
    return(invisible(TRUE))
  }
  dup <- unique(tf$id[duplicated(tf$id)])
  if (length(dup)) {
    stop("`targets` holds more than one row for ", .plural(length(dup), "identifier"), ": ",
         .listing(dup), ". Give `target_time` to say where in time each row sits.", call. = FALSE)
  }
  invisible(TRUE)
}

.check_series_reaches <- function(tf, series, spec) {
  missing <- setdiff(unique(tf$id), unique(.unit_names(series[[spec$id]], spec$id)))
  if (length(missing)) {
    stop(.plural(length(missing), "target"), " name a unit `series` does not carry: ",
         .listing(missing), ".", call. = FALSE)
  }
  invisible(TRUE)
}

.response_block <- function(targets, tf, y_cols) {
  out <- as.matrix(targets[, y_cols, drop = FALSE])
  storage.mode(out) <- "double"
  dimnames(out) <- list(tf$label, y_cols)
  out
}

# A learner declares what it can be handed and whether one fitted model covers every response.
# Both decide how the fitting layer calls it, so neither has a default to fall back on.
.learner_contract <- function(learner, label) {
  if (!isTRUE(learner$reads %in% c("tabular", "sequence"))) {
    stop("the ", label, " learner does not declare `reads`, which is \"tabular\" or \"sequence\".",
         call. = FALSE)
  }
  if (!isTRUE(learner$multi %in% c("joint", "separate"))) {
    stop("the ", label, " learner does not declare `multi`, which is \"joint\" or \"separate\".",
         call. = FALSE)
  }
  if (!is.null(learner$data) && !inherits(learner$data, "timesift_representation")) {
    stop("the ", label, " learner's `data` must be a representation or NULL, got ",
         class(learner$data)[1L], ".", call. = FALSE)
  }
  invisible(TRUE)
}

# The sift as representations, before any record is read: without a series there is one block and
# it is the static one, and the automatic set is still a promise the record has to keep.
.sift_specs <- function(sift, series) {
  if (is.null(series)) {
    return(timesift_sift(list(.static_representation())))
  }
  if (is.null(sift)) {
    return(grains("auto"))
  }
  timesift_sift(sift)
}

# Whether the targets are anchored in time and whether a representation is has to be one answer,
# and the check sees every representation the run will build: the members of the sift and the ones
# learners pinned themselves to through `data`. A lookback left without `target_time` reaches the
# builder with no anchor to place its bins against, which is a row count that does not add up
# rather than a message.
.check_anchored <- function(sift, learners, spec) {
  pinned <- Filter(Negate(is.null), lapply(learners, function(l) l$data))
  reps <- c(unclass(sift), pinned)
  if (is.null(spec$target_time)) {
    wrong <- Filter(function(r) identical(r$kind, "lookback"), reps)
    if (length(wrong)) {
      .needs_target_time(vapply(wrong, function(r) r$label, character(1L)))
    }
    return(invisible(TRUE))
  }
  if (isTRUE(attr(sift, "auto"))) {
    stop("`target_time` anchors every target in time, so every representation has to be a ",
         "lookback and there is no default set of spans. Give `sift = lookbacks(...)`.",
         call. = FALSE)
  }
  wrong <- Filter(function(r) !identical(r$kind, "lookback"), reps)
  if (length(wrong)) {
    labels <- vapply(wrong, function(r) r$label, character(1L))
    stop("`target_time` needs a target-anchored representation, and ", .listing(unique(labels)),
         " follows the calendar. Give `sift = lookbacks(...)`, or drop `target_time`.",
         call. = FALSE)
  }
  invisible(TRUE)
}

# Every array a candidate could read, built once and shared: two learners on one representation
# read one array, and the automatic set keeps the grains it built to decide with. A learner
# pinned to a representation of its own adds it to the same store under its own label.
.build_sift <- function(sift, series, targets, spec, learners, verbose) {
  if (isTRUE(attr(sift, "auto"))) {
    tf <- .target_frame(targets, spec)
    if (verbose) {
      message("choosing the grains the record carries")
    }
    built <- .auto_grains(attr(sift, "stats"), attr(sift, "year_start"), series, tf, spec)
    reps <- lapply(names(built), grain, stats = attr(sift, "stats"),
                   year_start = attr(sift, "year_start"))
    sift <- timesift_sift(stats::setNames(reps, names(built)))
  } else {
    built <- lapply(sift, build_representation, series = series, targets = targets, spec = spec)
    names(built) <- names(sift)
  }
  labels <- names(sift)
  known <- as.list(sift)
  for (ln in names(learners)) {
    rep <- learners[[ln]]$data
    if (is.null(rep)) {
      next
    }
    if (!is.null(known[[rep$label]])) {
      if (!identical(known[[rep$label]], rep)) {
        stop("two representations are reported under the name \"", rep$label,
             "\": the one the ", ln, " learner is pinned to and the one in the sift. Name the ",
             "sift to tell them apart.", call. = FALSE)
      }
      next
    }
    known[[rep$label]] <- rep
    built[[rep$label]] <- build_representation(rep, series, targets, spec)
  }
  list(built = built, sift = timesift_sift(known), labels = labels)
}

# One row per (learner, representation) pair considered, whether or not it was fitted, so what was
# skipped is reported rather than absent. A pair the caller asked for by name is an error instead:
# a representation named through `data =` is a decision, not one arm of an expansion.
.candidate_grid <- function(learners, sift, built, expansion) {
  rows <- list()
  for (ln in names(learners)) {
    learner <- learners[[ln]]
    pinned <- !is.null(learner$data)
    labels <- if (pinned) learner$data$label else expansion
    for (label in labels) {
      rep <- sift[[label]]
      x <- built[[label]]
      note <- .compatibility(learner, ln, rep, dim(x)[2L])
      if (pinned && !is.null(note)) {
        stop(note, call. = FALSE)
      }
      rows[[paste(ln, label)]] <- data.frame(
        candidate = paste(ln, label, sep = " / "), representation = label, learner = ln,
        grain = .grain_of(rep), bins = dim(x)[2L], channels = dim(x)[3L],
        status = if (is.null(note)) "fitted" else "not applicable",
        note = note %||% NA_character_, stringsAsFactors = FALSE)
    }
  }
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

.grain_of <- function(rep) {
  switch(rep$kind,
    grain = if (is.function(rep$grain)) "custom" else rep$grain,
    multigrain = if (is.null(rep$grains)) "multigrain" else paste(rep$grains, collapse = "+"),
    rep$kind)
}

# What a learner can be handed. The unreduced record is refused to a tabular learner from the
# representation alone, before any array exists; a sequence learner is refused a representation
# that declares itself a block of features the same way, and one whose record turned out to hold a
# single bin once `bins` is known, so the message names the representation rather than the setting.
.compatibility <- function(learner, label, rep, bins = NA_integer_) {
  if (identical(learner$reads, "tabular") && identical(rep$kind, "grain") &&
        identical(rep$grain, "native")) {
    return(paste0("`", label, "()` reads a tabular representation; `native()` gives it one column ",
                  "per reading. Use `grain()`, `multigrain()` or `lookback()`."))
  }
  one_bin <- !isTRUE(rep$sequence) || (!is.na(bins) && bins < 2L)
  if (identical(learner$reads, "sequence") && one_bin) {
    return(paste0("`", label, "()` reads a sequence; `", .constructor_of(rep),
                  "` gives one row of features."))
  }
  NULL
}

# A learner pinned to a representation it cannot read is refused before the record is touched,
# because building the array it would have been given is the expensive half of the call.
.refuse_pinned <- function(learners) {
  for (ln in names(learners)) {
    rep <- learners[[ln]]$data
    if (is.null(rep)) {
      next
    }
    note <- .compatibility(learners[[ln]], ln, rep)
    if (!is.null(note)) {
      stop(note, call. = FALSE)
    }
  }
  invisible(TRUE)
}

.constructor_of <- function(rep) {
  switch(rep$kind,
    grain = if (identical(rep$grain, "native")) "native()" else
      paste0("grain(\"", .grain_of(rep), "\")"),
    multigrain = "multigrain()",
    lookback = "lookback()",
    static = "static predictors alone")
}

# One candidate, over every fold, into the out-of-fold matrix the layers above read. Nothing above
# this loop knows whether the learner covered the responses jointly or one at a time.
.fit_candidate <- function(learner, x, y, f, levels, response, control, keep_fits,
                           verbose = FALSE) {
  p <- matrix(NA_real_, nrow = nrow(y), ncol = ncol(y), dimnames = dimnames(y))
  fits <- list()
  for (k in levels) {
    started <- Sys.time()
    train <- which(f != k)
    test <- which(f == k)
    fit <- .fit_candidate_once(learner, .subset_units(x, train), y[train, , drop = FALSE],
                               response, control)
    held_out <- stats::predict(fit, .subset_units(x, test))
    if (verbose) {
      message(sprintf("  fold %s of %d, %.0f s", k, length(levels),
                      as.numeric(difftime(Sys.time(), started, units = "secs"))))
    }
    p[rownames(held_out), colnames(held_out)] <- held_out
    if (keep_fits) {
      fits[[as.character(k)]] <- fit
    }
  }
  list(oof = p, fits = if (keep_fits) fits else NULL)
}

# A learner that covers the responses jointly is handed the matrix; one that does not is fitted
# once per column and the columns are put back together here. Either way one candidate is one
# fitted object emitting one [target, response] matrix.
.fit_candidate_once <- function(learner, x, y, response, control) {
  columns <- if (identical(learner$multi, "separate")) {
    as.list(seq_len(ncol(y)))
  } else {
    list(seq_len(ncol(y)))
  }
  fits <- lapply(columns, function(j) {
    fit_learner(learner, x, y[, j, drop = FALSE], response = response, control = control)
  })
  structure(list(fits = fits, learner = learner, variables = colnames(y), response = response,
                 multi = learner$multi),
            class = "timesift_candidate")
}

#' @export
predict.timesift_candidate <- function(object, newdata, ...) {
  parts <- lapply(object$fits, function(f) stats::predict(f, newdata))
  out <- do.call(cbind, parts)
  out[, object$variables, drop = FALSE]
}

#' @export
print.timesift_candidate <- function(x, ...) {
  cat("<timesift candidate>", x$learner$name, "on",
      .plural(length(x$variables), "response"), "\n")
  cat("fitted  :", .plural(length(x$fits), "model"),
      if (identical(x$multi, "separate")) "(one per response)" else "(one over all)", "\n")
  invisible(x)
}

.candidate_scores <- function(candidate, representation, learner, y, p, f, levels, cells, score) {
  out <- .score_arm(representation, learner, y, p, f, levels, cells, score)
  cbind(candidate = candidate, out[c("representation", "learner", "variable", "fold", "score",
                                     "scorable")],
        stringsAsFactors = FALSE)
}

#' Predict from a fitted timesift
#'
#' Rebuilds every member's representation for the new targets from the settings its own arm was
#' built with, predicts with the model refitted on all targets, and combines them where the
#' ensemble is asked for.
#'
#' @param object A [timesift()] fit.
#' @param targets A data frame of targets, carrying the identifier, the anchor and the static
#'   columns the fit was given.
#' @param series The long table of readings for those targets, or `NULL` for a targets-only fit.
#' @param candidate `"ensemble"`, or the name of one candidate.
#' @param ... Ignored.
#'
#' @return A `[target, response]` matrix of predictions, named by target and in the order the fit
#'   carries its own targets: sorted by identifier, or the targets' own order where `target_time`
#'   anchors them.
#'
#' @export
predict.timesift <- function(object, targets, series = NULL, candidate = "ensemble", ...) {
  spec <- object$spec
  if (identical(candidate, "ensemble")) {
    if (is.null(object$stack)) {
      stop("this fit carries no ensemble; name a candidate: ",
           .listing(names(object$models)), ".", call. = FALSE)
    }
    members <- names(object$oof)
  } else {
    members <- candidate
  }
  unknown <- setdiff(members, names(object$models))
  if (length(unknown)) {
    stop("unknown candidate: ", .listing(unknown), ". This fit carries ",
         .listing(names(object$models)), ".", call. = FALSE)
  }
  labels <- object$candidates$representation[match(members, object$candidates$candidate)]
  built <- lapply(stats::setNames(unique(labels), unique(labels)), function(label) {
    build_representation(spec$sift[[label]], series, targets, spec)
  })
  preds <- stats::setNames(lapply(seq_along(members), function(i) {
    stats::predict(object$models[[members[i]]], built[[labels[i]]])
  }), members)
  if (identical(candidate, "ensemble")) {
    ensemble_combine(object$stack, preds)
  } else {
    preds[[1L]]
  }
}
