#' What a run found
#'
#' The mean score of every candidate, how many responses each of them scored highest on, whether
#' one fitted model covered those responses or one was fitted per response, and the level the
#' combined prediction reached with the weights it reached it under.
#'
#' Both columns beside the mean are worth reading. A candidate can carry the ensemble without
#' winning a single response, which is what `won` shows and a mean alone hides; and a joint model
#' and a per-response one reach the same `[target, response]` matrix by different routes, which is
#' what `responses` records.
#'
#' A candidate the run built no representation for, because its learner cannot read the
#' representation it was paired with, is listed with no mean rather than dropped, so the report says
#' what was asked for as well as what ran.
#'
#' @param object A `timesift` result.
#' @param x A `timesift` result, or the table this returns.
#' @param ... Ignored, so that the methods take the arguments their generics declare.
#'
#' @return A data frame of one row per candidate and one for the ensemble, of class
#'   `timesift_summary`, carrying the mean score, the responses won and how the responses were
#'   covered. The weights are in the `weights` attribute.
#'
#' @name timesift_report
NULL

#' @rdname timesift_report
#' @export
summary.timesift <- function(object, ...) {
  per_response <- .per_response(object)
  candidates <- object$candidates$candidate
  mean_score <- vapply(candidates, function(cd) {
    v <- per_response$score[per_response$candidate == cd]
    if (!length(v)) NA_real_ else mean(v)
  }, numeric(1L))
  out <- data.frame(candidate = candidates, mean = as.numeric(mean_score),
                    won = .responses_won(per_response, candidates),
                    responses = .candidate_multi(object, candidates),
                    stringsAsFactors = FALSE)
  out <- out[order(out$mean, na.last = FALSE, decreasing = FALSE), , drop = FALSE]
  level <- .ensemble_level(object)
  if (is.finite(level)) {
    out <- rbind(out, data.frame(candidate = "ensemble", mean = level, won = NA_integer_,
                                 responses = "", stringsAsFactors = FALSE))
  }
  rownames(out) <- NULL
  structure(out, class = c("timesift_summary", "data.frame"),
            header = .run_header(object), weights = ensemble_weights(object))
}

#' @rdname timesift_report
#' @export
print.timesift <- function(x, ...) {
  print(summary(x))
  invisible(x)
}

#' @rdname timesift_report
#' @export
print.timesift_summary <- function(x, ...) {
  cat(attr(x, "header"), "\n\n", sep = "")
  if (!nrow(x)) {
    cat("no candidate scored a cell.\n")
    return(invisible(x))
  }
  width <- max(nchar(c(x$candidate, "candidate")))
  row <- function(...) cat(trimws(sprintf(...), which = "right"), "\n", sep = "")
  row("%-*s %14s %6s  %s", width, "candidate", "mean", "won", "responses")
  for (i in seq_len(nrow(x))) {
    row("%-*s %14s %6s  %s", width, x$candidate[i],
        if (is.na(x$mean[i])) "not applicable" else sprintf("%.3f", x$mean[i]),
        if (is.na(x$won[i])) "-" else format(x$won[i]),
        x$responses[i])
  }
  weights <- attr(x, "weights")
  if (!is.null(weights)) {
    # A member whose weight rounds to nothing is not a member of the combination in any way a
    # reader can act on; ensemble_weights() still carries every one of them.
    weights <- sort(weights[weights >= 0.005], decreasing = TRUE)
    cat("\nweights  ",
        paste(sprintf("%s %.2f", names(weights), weights), collapse = "   "), "\n", sep = "")
  }
  invisible(x)
}

#' @rdname occlusion
#'
#' @details
#' Reached through a [timesift()] run rather than through a ladder, the profile reads the per-fold
#' models the run was told to keep, so every bin is held back from a model that never saw the units
#' it is rescored on. The candidate is named as `summary()` reports it.
#'
#' @param candidate Name of the candidate to read, for a run.
#'
#' @export
occlusion.timesift <- function(x, candidate, over = c("bin", "channel"), ...) {
  over <- match.arg(over)
  fit <- x
  .check_run(fit)
  row <- .candidate_row(fit, candidate)
  arm <- paste(row$representation, row$learner, sep = "|")
  fits <- .fold_fits(fit, candidate, arm)
  occlusion(.ladder_view(fit, candidate, row, fits),
            fit$representations[row$representation], fit$y, arm, over = over, ...)
}

# ---- reading the fitted object ---------------------------------------------------------------

.check_run <- function(fit) {
  if (!inherits(fit, "timesift")) {
    stop("expected a timesift() result, got ", class(fit)[1L], ".", call. = FALSE)
  }
  invisible(TRUE)
}

.candidate_row <- function(fit, candidate) {
  i <- match(candidate, fit$candidates$candidate)
  if (length(candidate) != 1L || is.na(i)) {
    stop("no candidate called \"", paste(candidate, collapse = ", "), "\" in this run. It fitted ",
         paste(fit$candidates$candidate, collapse = ", "), ".", call. = FALSE)
  }
  fit$candidates[i, , drop = FALSE]
}

# A response is the independent replicate, so a candidate's cells are averaged within a response
# over its folds before anything is averaged across responses. It is the ladder's rule, on the
# columns a run reports under.
.per_response <- function(fit) {
  scores <- fit$scores
  keep <- scores[!is.na(scores$score) & scores$candidate != "ensemble", , drop = FALSE]
  empty <- data.frame(candidate = character(), variable = character(), score = numeric(),
                      representation = character(), learner = character(),
                      stringsAsFactors = FALSE)
  if (!nrow(keep)) {
    return(empty)
  }
  per <- stats::aggregate(list(score = keep$score), keep[c("candidate", "variable")], mean)
  i <- match(per$candidate, fit$candidates$candidate)
  per$representation <- fit$candidates$representation[i]
  per$learner <- fit$candidates$learner[i]
  # The order the run declared its candidates in, so which candidate an exact tie falls to does not
  # depend on the session's collation.
  per[order(match(per$candidate, fit$candidates$candidate), per$variable, method = "radix"),
      names(empty), drop = FALSE]
}

.responses_won <- function(per_response, candidates) {
  won <- stats::setNames(integer(length(candidates)), candidates)
  if (!nrow(per_response)) {
    return(unname(won))
  }
  for (v in unique(per_response$variable)) {
    rows <- per_response[per_response$variable == v, , drop = FALSE]
    top <- rows$candidate[which.max(rows$score)]
    won[[top]] <- won[[top]] + 1L
  }
  unname(won)
}

# Whether one fitted model covered every response is the learner's own declaration, read off the
# model the run refit on all targets rather than recorded a second time beside it.
.candidate_multi <- function(fit, candidates) {
  vapply(candidates, function(cd) {
    model <- fit$models[[cd]]
    if (is.null(model) || is.null(model$learner)) NA_character_ else model$learner$multi
  }, character(1L), USE.NAMES = FALSE)
}

# The level the combined out-of-fold prediction reaches, scored on the same cells and by the same
# metric as every candidate, so the ensemble row of the report is comparable with the rows above it.
.ensemble_level <- function(fit) {
  if (is.null(fit$stack)) {
    return(NA_real_)
  }
  combined <- ensemble_combine(fit$stack, fit$oof)
  f <- .as_folds(fit$folds, rownames(fit$y))
  rows <- .score_arm("ensemble", "ensemble", fit$y, combined, f, sort(unique(f)), fit$cells,
                     fit$scorer)
  per <- .arm_means(rows)
  if (!nrow(per)) NA_real_ else mean(per$score)
}

.run_header <- function(fit) {
  v <- length(unique(.as_folds(fit$folds, rownames(fit$y))))
  kind <- if (isTRUE(attr(fit$folds, "grouped"))) "grouped" else "random"
  sprintf("timesift  %s, %s, %d-fold %s CV, %s", .plural(nrow(fit$y), "target"),
          .plural(ncol(fit$y), "response"), v, kind, fit$metric)
}

# The per-fold models the run kept, rekeyed onto the arm label the occlusion reads them under. A
# run that kept none says so rather than reading the model refit on every target, which would
# rescore each fold on units that model was fitted on.
.fold_fits <- function(fit, candidate, arm) {
  if (is.null(fit$fits) || !length(fit$fits)) {
    stop("this run kept no per-fold fits. Refit with timesift(..., keep_fits = TRUE).",
         call. = FALSE)
  }
  kept <- fit$fits[[candidate]]
  if (is.null(kept) || !length(kept) || is.null(names(kept))) {
    stop("no fold of \"", candidate, "\" is among the fits this run kept.", call. = FALSE)
  }
  stats::setNames(kept, paste(arm, names(kept), sep = "|"))
}

# One candidate of a run, in the layout a ladder carries, so occlusion() reads a run through
# exactly the reader it already has.
.ladder_view <- function(fit, candidate, row, fits) {
  rows <- fit$scores[fit$scores$candidate == candidate, , drop = FALSE]
  arm <- paste(row$representation, row$learner, sep = "|")
  out <- data.frame(grain = row$representation, learner = row$learner, variable = rows$variable,
                    fold = rows$fold, score = rows$score, scorable = rows$scorable,
                    stringsAsFactors = FALSE)
  structure(out, class = c("timesift_ladder", "data.frame"),
            predictions = stats::setNames(list(fit$oof[[candidate]]), arm),
            cells = fit$cells, folds = fit$folds, fits = fits,
            metric = fit$metric, scorer = fit$scorer, response = fit$response)
}
