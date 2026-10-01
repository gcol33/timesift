#' How the candidates are combined
#'
#' Every candidate emits an out-of-fold prediction for every scorable cell over the same folds, so
#' the combination is arithmetic on those predictions and nothing else. `ensemble()` says which
#' arithmetic.
#'
#' `"stack"` fits non-negative weights summing to one on the out-of-fold predictions alone, never on
#' in-sample ones, minimising the response head's loss over the scorable cells: binomial deviance
#' for presence-absence. One weight vector covers every response, because per-response weights would
#' be fitted on the handful of cells a rare response has. `"mean"` and `"median"` combine without
#' fitting anything. `"weighted"` takes each candidate's own mean score, keeps its non-negative
#' part and rescales those to sum to one, so a candidate scoring at or below zero is left out and
#' the rest are weighted by how well they scored; where no candidate scores above zero the scores
#' order nothing worth weighting by, and every candidate weighs the same.
#'
#' `decay` changes how `"weighted"` turns scores into weights, as biomod2's `EMwmean.decay` does.
#' Given a number `d`, the candidates scoring above zero are ranked from the best down, the one
#' ranked `r` of `K` takes `d^(K - r + 1)`, candidates on the same score share the mean of their
#' ranks' weights, and the weights are rescaled to sum to one. Each rank down weighs `1 / d` of the
#' one above it, whatever the gap in score between them.
#'
#' `"committee"` is biomod2's committee averaging. Each member cuts each response at its own
#' threshold, learned by [decision_threshold()] under `rule` from that member's out-of-fold
#' predictions of every target, and the combined prediction is the share of members voting
#' presence. A member whose predictions of a response give no cut does not vote on it. Under
#' [timesift()] the thresholds of each outer fold are learned from the inner out-of-fold
#' predictions of its training targets, as a stack's weights are, so the estimate never reads a cut
#' chosen on the targets it scores.
#'
#' `min_score` is biomod2's `metric.select.thresh`: a candidate whose mean score is below it is
#' not eligible, and the filter is applied before `scope` picks among what is left.
#'
#' `scope` says which candidates are eligible. `"all"` is every candidate. `"learners"` keeps the
#' several learners that read the representation of the best-scoring candidate, and
#' `"representations"` keeps the one learner of the best-scoring candidate across the
#' representations it ran on; both are read off the same mean scores the report shows.
#'
#' @param method How the members are combined.
#' @param scope Which candidates are eligible.
#' @param metric Name of the registered metric the eligibility, `min_score` and the `"weighted"`
#'   weights are read by, or `NULL` for the score the run already carries.
#' @param response Name of the registered response head whose loss `"stack"` minimises, or `NULL`
#'   for the head the run was fitted under. Naming a head the run does not fit toward is an error
#'   rather than an override, and [ensemble_fit()] called on its own reads `NULL` as
#'   `"presence_absence"`.
#' @param min_score `NULL`, or the mean score a candidate needs to be eligible.
#' @param decay For `"weighted"`, `"proportional"` (the default) or a number of at least one, the
#'   ratio of one rank's weight to the next one's.
#' @param rule For `"committee"`, the rule of [decision_threshold()] each member's cut is learned
#'   by; `"youden"` by default.
#'
#' @return A `timesift_ensemble`.
#'
#' @examples
#' ensemble()
#' ensemble("weighted", scope = "learners")
#' ensemble("weighted", decay = 1.6, min_score = 0.4)
#' ensemble("committee", rule = "kappa")
#'
#' @export
ensemble <- function(method = c("stack", "mean", "median", "weighted", "committee"),
                     scope = c("all", "learners", "representations"), metric = NULL,
                     response = NULL, min_score = NULL, decay = NULL, rule = NULL) {
  method <- match.arg(method)
  scope <- match.arg(scope)
  if (!is.null(metric)) {
    .metrics_reg$get(metric)
  }
  if (!is.null(response)) {
    .responses_reg$get(response)
  }
  if (!is.null(min_score) &&
      (!is.numeric(min_score) || length(min_score) != 1L || !is.finite(min_score))) {
    stop("`min_score` is one finite number, or NULL for no filter.", call. = FALSE)
  }
  if (!is.null(decay)) {
    if (!identical(method, "weighted")) {
      stop("`decay` sets how the \"weighted\" method turns scores into weights, and the ",
           "method here is \"", method, "\".", call. = FALSE)
    }
    if (!identical(decay, "proportional") &&
        (!is.numeric(decay) || length(decay) != 1L || !is.finite(decay) || decay < 1)) {
      stop("`decay` is \"proportional\" or one number of at least one.", call. = FALSE)
    }
  }
  if (!is.null(rule)) {
    if (!identical(method, "committee")) {
      stop("`rule` sets where each member of a committee cuts its prediction, and the method ",
           "here is \"", method, "\".", call. = FALSE)
    }
    rule <- match.arg(rule, .threshold_rules)
  }
  structure(list(method = method, scope = scope, metric = metric, response = response,
                 min_score = min_score,
                 decay = if (identical(method, "weighted")) decay %||% "proportional",
                 rule = if (identical(method, "committee")) rule %||% "youden"),
            class = "timesift_ensemble")
}

#' @export
print.timesift_ensemble <- function(x, ...) {
  how <- switch(x$method,
    weighted = paste0(" (", if (is.numeric(x$decay)) paste("decay", x$decay) else x$decay, ")"),
    committee = paste0(" (cut by ", x$rule, ")"),
    "")
  cat("<timesift ensemble> ", x$method, how, " over the ", x$scope, " candidates",
      if (!is.null(x$min_score)) paste(" scoring at least", x$min_score), "\n", sep = "")
  cat("response:", x$response %||% "the run's own", "; metric:", x$metric %||% "the run's own",
      "\n")
  invisible(x)
}

# The run was fitted toward one response head and the combiner minimises that head's loss, so the
# run's response is what reaches the spec. A spec naming another head is a contradiction rather
# than an override: it would stack an abundance run under a presence-absence loss.
.run_ensemble <- function(spec, response) {
  spec <- .as_ensemble(spec)
  if (is.null(spec$response)) {
    spec$response <- response
    return(spec)
  }
  if (!identical(spec$response, response)) {
    stop("the run fits the ", response, " response and `ensemble(response = \"", spec$response,
         "\")` names another. The combiner minimises the loss of the head the run was fitted ",
         "under.", call. = FALSE)
  }
  spec
}

.as_ensemble <- function(spec) {
  if (inherits(spec, "timesift_ensemble")) {
    return(spec)
  }
  if (isTRUE(spec)) {
    return(ensemble())
  }
  if (is.character(spec) && length(spec) == 1L) {
    return(ensemble(spec))
  }
  stop("expected an ensemble() specification, got ", class(spec)[1L], ".", call. = FALSE)
}

#' Fit the combiner on the out-of-fold predictions
#'
#' The combiner sees the out-of-fold predictions, the response, the mask of scorable cells and the
#' fold map, and never a model, so it cannot read anything a candidate fitted in-sample.
#'
#' The weights are fitted to the response on those predictions, so the combined prediction scored
#' against the same response is scored on the data its weights were fitted to, and that score is
#' optimistic. [timesift()] evaluates the stack the other way: each outer fold's weights are fitted
#' on inner out-of-fold predictions of its training targets and applied to the outer test fold.
#'
#' @param oof Named list of `[target, response]` matrices, one per candidate.
#' @param y The response matrix.
#' @param cells The scorable-cell mask from [scorable_cells()].
#' @param folds The fold map.
#' @param spec An [ensemble()] specification.
#' @param scores The per-cell scores of the run, a data frame carrying `candidate`, `variable`,
#'   `fold`, `score` and `scorable`. Read only where `spec` names no metric of its own.
#'
#' @return A `timesift_stack`: `method`, the named `weights`, and what they were fitted on.
#'
#' @examples
#' set.seed(1)
#' y <- matrix(rbinom(200, 1, 0.4), nrow = 50,
#'             dimnames = list(sprintf("p%02d", 1:50), paste0("sp", 1:4)))
#' folds <- fold_map(y, v = 5)
#' truth <- matrix(runif(200), nrow = 50, dimnames = dimnames(y))
#' oof <- list(good = 0.8 * y + 0.2 * truth, noise = truth)
#' ensemble_fit(oof, y, scorable_cells(y, folds), folds)
#'
#' @export
ensemble_fit <- function(oof, y, cells, folds, spec = ensemble(), scores = NULL) {
  # Fitted from a run, the spec arrives carrying the run's head; fitted on its own, there is no
  # run to read one off and the shipped default is the one the package defaults to everywhere.
  spec <- .as_ensemble(spec)
  spec$response <- spec$response %||% "presence_absence"
  y <- .as_response(y)
  oof <- .check_oof(oof, y)
  if (!any(cells$scorable)) {
    stop("no cell is scorable, so there is nothing to fit a combiner on. Every (response, fold) ",
         "cell needs both classes on each side of its split; see scorable_cells().",
         call. = FALSE)
  }
  mean_score <- .member_scores(oof, y, cells, folds, spec, scores,
                               required = !identical(spec$scope, "all") ||
                                 identical(spec$method, "weighted") || !is.null(spec$min_score))
  members <- .eligible_members(.passing_members(names(oof), mean_score, spec$min_score),
                               spec$scope, mean_score)

  mask <- .scorable_matrix(y, cells, folds)
  block <- .stacking_block(oof[members], y, mask)
  loss <- .stack_loss(spec$response)

  equal <- list(weights = rep(1 / length(members), length(members)), value = NA_real_,
                iterations = 0L)
  fitted <- switch(
    spec$method,
    stack = .simplex_weights(block$predictions, block$response, loss),
    mean = equal,
    median = equal,
    committee = equal,
    weighted = list(weights = .score_weights(mean_score[members], spec$decay %||% "proportional"),
                    value = NA_real_, iterations = 0L)
  )
  weights <- stats::setNames(fitted$weights, members)
  structure(list(method = spec$method, weights = weights, scope = spec$scope,
                 metric = spec$metric, response = spec$response, loss = loss$name,
                 score = mean_score[members], n_cell = length(block$response),
                 value = fitted$value, iterations = fitted$iterations,
                 min_score = spec$min_score, decay = spec$decay, rule = spec$rule,
                 thresholds = if (identical(spec$method, "committee")) {
                   .member_thresholds(oof[members], y, spec$rule %||% "youden")
                 }),
            class = "timesift_stack")
}

# A committee member's cut on each response, learned from its own out-of-fold predictions of
# every target, as decision_threshold() learns the cut of a fit's candidate: a [member, response]
# matrix, NA where a member's predictions of a response give no cut.
.member_thresholds <- function(oof, y, rule) {
  t(vapply(oof, function(p) {
    vapply(colnames(y), function(v) decision_threshold.default(y[, v], p[, v], rule),
           numeric(1L))
  }, numeric(ncol(y))))
}

# biomod2's metric.select.thresh, read as the name says: a candidate scoring at least the minimum
# is kept, and one carrying no score is not.
.passing_members <- function(members, score, min_score) {
  if (is.null(min_score)) {
    return(members)
  }
  keep <- members[is.finite(score[members]) & score[members] >= min_score]
  if (length(keep) < 2L) {
    stop("`min_score = ", min_score, "` leaves ", .plural(length(keep), "candidate"),
         " to combine; the best scores ", format(max(score, na.rm = TRUE), digits = 3), ".",
         call. = FALSE)
  }
  keep
}

#' @export
print.timesift_stack <- function(x, ...) {
  cat("<timesift stack>", x$method, "over", .plural(length(x$weights), "candidate"), "\n")
  cat("fitted on", .plural(x$n_cell, "scorable cell"),
      if (is.finite(x$value)) sprintf(", %s %.4f", x$loss, x$value) else "", "\n")
  w <- sort(x$weights, decreasing = TRUE)
  cat(paste(sprintf("  %-28s %.3f", names(w), w), collapse = "\n"), "\n")
  invisible(x)
}

#' Combine one prediction per member into one prediction
#'
#' @param stack A [ensemble_fit()] result.
#' @param preds Named list of `[target, response]` matrices, one per member of the stack.
#'
#' @return One `[target, response]` matrix.
#'
#' @examples
#' set.seed(1)
#' y <- matrix(rbinom(200, 1, 0.4), nrow = 50,
#'             dimnames = list(sprintf("p%02d", 1:50), paste0("sp", 1:4)))
#' folds <- fold_map(y, v = 5)
#' truth <- matrix(runif(200), nrow = 50, dimnames = dimnames(y))
#' oof <- list(good = 0.8 * y + 0.2 * truth, noise = truth)
#' st <- ensemble_fit(oof, y, scorable_cells(y, folds), folds)
#' dim(ensemble_combine(st, oof))
#'
#' @export
ensemble_combine <- function(stack, preds) {
  parts <- .member_parts(stack, preds)
  d <- dim(parts[[1L]])
  out <- switch(stack$method,
    median = apply(array(unlist(parts, use.names = FALSE), dim = c(d, length(parts))), c(1L, 2L),
                   stats::median),
    committee = .committee_share(parts, stack$weights, stack$thresholds),
    Reduce(`+`, Map(function(p, w) p * w, parts, as.numeric(stack$weights))))
  matrix(out, nrow = d[1L], ncol = d[2L], dimnames = dimnames(parts[[1L]]))
}

# The members' predictions in the stack's order, each a matrix of the one shape.
.member_parts <- function(stack, preds) {
  if (!inherits(stack, "timesift_stack")) {
    stop("expected an ensemble_fit() result, got ", class(stack)[1L], ".", call. = FALSE)
  }
  members <- names(stack$weights)
  missing <- setdiff(members, names(preds))
  if (length(missing)) {
    stop("the stack was fitted on ", paste(missing, collapse = ", "),
         ", which `preds` does not carry.", call. = FALSE)
  }
  parts <- lapply(preds[members], as.matrix)
  d <- dim(parts[[1L]])
  same <- vapply(parts, function(p) identical(dim(p), d), logical(1L))
  if (!all(same)) {
    stop("every member's prediction must have the same shape; ",
         paste(members[!same], collapse = ", "), " does not.", call. = FALSE)
  }
  parts
}

# Each response's vote: the weighted share of the members holding a cut on it that read presence,
# `p >= cut`. A response no member holds a cut on has no vote to share and predicts NA.
.committee_share <- function(parts, weights, thresholds) {
  out <- matrix(NA_real_, nrow = nrow(parts[[1L]]), ncol = ncol(parts[[1L]]))
  for (j in seq_len(ncol(out))) {
    v <- colnames(parts[[1L]])[j]
    voting <- which(is.finite(thresholds[, v]))
    if (!length(voting)) {
      next
    }
    w <- as.numeric(weights[voting]) / sum(weights[voting])
    votes <- vapply(voting, function(m) as.numeric(parts[[m]][, j] >= thresholds[m, v]),
                    numeric(nrow(out)))
    out[, j] <- as.numeric(matrix(votes, nrow = nrow(out)) %*% w)
  }
  out
}

#' How far the members of an ensemble disagree
#'
#' biomod2's `EMcv` and `EMci`, read on the members' predictions under the weights the stack
#' carries. For each target and response: the weighted mean `m` of the members' predictions; their
#' weighted standard deviation `s`, the square root of `sum(w * (p - m)^2) / (1 - sum(w^2))`, which
#' is the sample standard deviation when the weights are equal; the coefficient of variation `s /
#' m`; and the interval `m -+ qt(1 - alpha / 2, n - 1) * s * sqrt(sum(w^2))`, `n` the number of
#' members carrying weight, which is the t interval of a mean of `n` members when the weights are
#' equal. Under a response head whose predictions are probabilities, the interval is held inside
#' zero and one. An uncertainty map is this on one target per map cell.
#'
#' A committee's and a median's members are read at equal weight, and a committee's spread is that
#' of the members' predictions rather than of their votes.
#'
#' @param stack A [ensemble_fit()] result.
#' @param preds Named list of `[target, response]` matrices, one per member of the stack.
#' @param alpha One minus the interval's coverage.
#'
#' @return A `[target, response, statistic]` array, the statistics being `mean`, `sd`, `cv`,
#'   `lower` and `upper`. `sd`, `cv` and the interval are `NA` where fewer than two members carry
#'   weight.
#'
#' @examples
#' set.seed(1)
#' y <- matrix(rbinom(200, 1, 0.4), nrow = 50,
#'             dimnames = list(sprintf("p%02d", 1:50), paste0("sp", 1:4)))
#' folds <- fold_map(y, v = 5)
#' truth <- matrix(runif(200), nrow = 50, dimnames = dimnames(y))
#' oof <- list(good = 0.8 * y + 0.2 * truth, fair = 0.6 * y + 0.4 * truth, noise = truth)
#' st <- ensemble_fit(oof, y, scorable_cells(y, folds), folds, ensemble("mean"))
#' ensemble_spread(st, oof)[1:3, "sp1", ]
#'
#' @export
ensemble_spread <- function(stack, preds, alpha = 0.05) {
  if (!is.numeric(alpha) || length(alpha) != 1L || !(alpha > 0 && alpha < 1)) {
    stop("`alpha` is one number strictly between 0 and 1.", call. = FALSE)
  }
  parts <- .member_parts(stack, preds)
  d <- dim(parts[[1L]])
  w <- if (stack$method %in% c("median", "committee")) {
    rep(1 / length(parts), length(parts))
  } else {
    as.numeric(stack$weights) / sum(stack$weights)
  }
  m <- Reduce(`+`, Map(function(p, wk) wk * p, parts, w))
  sq <- sum(w^2)
  n <- sum(w > 0)
  s <- if (n >= 2L) {
    sqrt(Reduce(`+`, Map(function(p, wk) wk * (p - m)^2, parts, w)) / (1 - sq))
  } else {
    matrix(NA_real_, nrow = d[1L], ncol = d[2L])
  }
  half <- if (n >= 2L) stats::qt(1 - alpha / 2, n - 1L) * s * sqrt(sq) else s
  range <- .stack_losses[[stack$loss]]$range
  array(c(m, s, s / m, pmax(m - half, range[1L]), pmin(m + half, range[2L])),
        dim = c(d, 5L),
        dimnames = c(dimnames(parts[[1L]]), list(c("mean", "sd", "cv", "lower", "upper"))))
}

#' The weights the combiner fitted
#'
#' @param fit A `timesift` result, or the stack itself.
#'
#' @return A named numeric vector, or `NULL` where the run fitted no combiner.
#'
#' @examples
#' set.seed(1)
#' y <- matrix(rbinom(200, 1, 0.4), nrow = 50,
#'             dimnames = list(sprintf("p%02d", 1:50), paste0("sp", 1:4)))
#' folds <- fold_map(y, v = 5)
#' truth <- matrix(runif(200), nrow = 50, dimnames = dimnames(y))
#' oof <- list(good = 0.8 * y + 0.2 * truth, noise = truth)
#' ensemble_weights(ensemble_fit(oof, y, scorable_cells(y, folds), folds))
#'
#' @export
ensemble_weights <- function(fit) {
  if (inherits(fit, "timesift_stack")) {
    return(fit$weights)
  }
  if (!inherits(fit, "timesift")) {
    stop("expected a timesift() result or an ensemble_fit() one, got ", class(fit)[1L], ".",
         call. = FALSE)
  }
  if (is.null(fit$stack)) {
    return(NULL)
  }
  fit$stack$weights
}

# ---- the simplex ---------------------------------------------------------------------------

# Non-negative weights summing to one, by exponentiated gradient. The multiplicative update keeps
# every weight positive and the renormalisation keeps the sum at one, so the iterate never leaves
# the simplex and no projection step is needed. The step is halved until the loss falls, which
# makes the sequence of losses monotone and the stopping point the same on every machine; the
# gradient is divided by its largest entry, so the step means the same thing whatever scale the
# loss is on. The loop stops when a step buys less than `tol` of the loss it is on; near the
# minimum the loss is flat to second order in the weights, so the stop settles the gradient to a
# precision of about the square root of `tol`.
.simplex_weights <- function(P, y, loss, iterations = 500L, tol = 1e-14) {
  k <- ncol(P)
  w <- rep(1 / k, k)
  value <- loss$value(as.numeric(P %*% w), y)
  step <- 1
  used <- 0L
  for (i in seq_len(iterations)) {
    g <- as.numeric(crossprod(P, loss$gradient(as.numeric(P %*% w), y)))
    largest <- max(abs(g))
    if (!is.finite(largest) || largest <= 0) {
      break
    }
    g <- g / largest
    repeat {
      candidate <- w * exp(-step * g)
      candidate <- candidate / sum(candidate)
      moved <- loss$value(as.numeric(P %*% candidate), y)
      if (is.finite(moved) && moved <= value) {
        break
      }
      step <- step / 2
      if (step < 1e-12) {
        break
      }
    }
    if (step < 1e-12) {
      break
    }
    gain <- value - moved
    w <- candidate
    value <- moved
    used <- i
    if (gain <= tol * max(1, abs(value))) {
      break
    }
    step <- step * 1.5
  }
  list(weights = w, value = value, iterations = used)
}

# A loss reaches the solver as its value and its derivative in the combined prediction, both
# averaged over the cells, so the solver is the same forty lines whatever the response head is.
.stack_losses <- list(
  binary_cross_entropy = list(
    range = c(0, 1),
    value = function(p, y) {
      p <- .clamp_unit(p)
      -mean(y * log(p) + (1 - y) * log(1 - p))
    },
    gradient = function(p, y) {
      p <- .clamp_unit(p)
      (p - y) / (p * (1 - p)) / length(y)
    }
  ),
  squared_error = list(
    range = c(-Inf, Inf),
    value = function(p, y) mean((p - y)^2),
    gradient = function(p, y) 2 * (p - y) / length(y)
  )
)

.clamp_unit <- function(p) pmin(pmax(p, 1e-7), 1 - 1e-7)

.stack_loss <- function(response) {
  name <- .responses_reg$get(response)$loss
  if (!name %in% names(.stack_losses)) {
    stop("the ", response, " response is trained under \"", name,
         "\", which the combiner cannot minimise. It knows ",
         paste(names(.stack_losses), collapse = " and "), ".", call. = FALSE)
  }
  c(.stack_losses[[name]], list(name = name))
}

# biomod2's EMwmean. Proportional weights are the positive part of each score; with a decay, the
# K candidates scoring above zero take decay^K for the best down to decay^1 for the K-th, tied
# scores share the mean of their ranks' weights, and a candidate at or below zero takes none.
.score_weights <- function(score, decay = "proportional") {
  score <- ifelse(is.finite(score), score, 0)
  positive <- score > 0
  if (!any(positive)) {
    return(rep(1 / length(score), length(score)))
  }
  w <- if (identical(decay, "proportional")) {
    pmax(score, 0)
  } else {
    k <- sum(positive)
    rank_weight <- numeric(length(score))
    top <- order(-score)[seq_len(k)]
    rank_weight[top] <- decay^(k - seq_len(k) + 1)
    tied <- stats::ave(rank_weight, match(score, unique(score)), FUN = mean)
    ifelse(positive, tied, 0)
  }
  as.numeric(w / sum(w))
}

# ---- what the combiner is handed ------------------------------------------------------------

.check_oof <- function(oof, y) {
  if (!is.list(oof) || !length(oof) || is.null(names(oof)) || anyDuplicated(names(oof)) ||
      any(!nzchar(names(oof)))) {
    stop("`oof` is a non-empty list of prediction matrices, each under its own candidate name.",
         call. = FALSE)
  }
  lapply(stats::setNames(names(oof), names(oof)), function(nm) {
    p <- as.matrix(oof[[nm]])
    if (is.null(rownames(p)) || is.null(colnames(p))) {
      stop("candidate ", nm, " emitted a prediction with no target or response names.",
           call. = FALSE)
    }
    if (!setequal(rownames(p), rownames(y)) || !setequal(colnames(p), colnames(y))) {
      stop("candidate ", nm, " emitted predictions for other targets or responses than the ",
           "response carries.", call. = FALSE)
    }
    p[rownames(y), colnames(y), drop = FALSE]
  })
}

# The mask, as a [target, response] logical: a cell is stacked on where the (response, fold) it
# falls in is one every candidate was scored on.
.scorable_matrix <- function(y, cells, folds) {
  f <- .as_folds(folds, rownames(y))
  ok <- cells$scorable[match(paste(rep(colnames(y), each = nrow(y)), rep(f, times = ncol(y))),
                             paste(cells$variable, cells$fold))]
  ok[is.na(ok)] <- FALSE
  matrix(ok, nrow = nrow(y), ncol = ncol(y), dimnames = dimnames(y))
}

# Every scorable cell, and no other: the mask says which cells every candidate was scored on, and
# the combiner is fitted on exactly those. Dropping the ones a candidate left without a number
# would fit the stack on fewer cells than the report says it was, and the report would not show it.
.stacking_block <- function(oof, y, mask) {
  P <- vapply(oof, function(p) as.numeric(p[mask]), numeric(sum(mask)))
  P <- matrix(P, nrow = sum(mask), ncol = length(oof), dimnames = list(NULL, names(oof)))
  bad <- vapply(seq_len(ncol(P)), function(j) sum(!is.finite(P[, j])), integer(1L))
  if (any(bad > 0L)) {
    stop("the combiner is fitted on every scorable cell, and ",
         .listing(paste0(colnames(P)[bad > 0L], " (", bad[bad > 0L], ")")),
         " hold no number on some of them. A candidate that cannot predict a scorable cell is a ",
         "fit that did not settle; drop it from the run rather than from the cells.",
         call. = FALSE)
  }
  list(predictions = P, response = as.numeric(y[mask]))
}

# A candidate is reported as "learner / representation", which is what makes a scope readable off
# the names alone. The combiner is handed predictions and their names and nothing else, so this is
# where the two halves come from.
.candidate_parts <- function(candidate) {
  parts <- strsplit(candidate, " / ", fixed = TRUE)
  data.frame(
    candidate = candidate,
    learner = vapply(parts, function(p) p[1L], character(1L)),
    representation = vapply(parts, function(p) if (length(p) > 1L) p[2L] else NA_character_,
                            character(1L)),
    stringsAsFactors = FALSE)
}

.eligible_members <- function(members, scope, score) {
  if (identical(scope, "all")) {
    return(members)
  }
  parts <- .candidate_parts(members)
  side <- if (identical(scope, "learners")) parts$representation else parts$learner
  if (anyNA(side)) {
    stop("scope \"", scope, "\" reads the learner and the representation off each candidate's ",
         "name, and ", paste(members[is.na(side)], collapse = ", "),
         " is not named \"learner / representation\".", call. = FALSE)
  }
  score <- score[members]
  best <- which.max(ifelse(is.finite(score), score, -Inf))
  keep <- members[side == side[best]]
  if (length(keep) < 2L) {
    stop("scope \"", scope, "\" leaves ", .plural(length(keep), "candidate"),
         " to combine. Widen the run or use scope = \"all\".", call. = FALSE)
  }
  keep
}

# The mean score of each candidate: recomputed from the out-of-fold predictions where the
# specification names its own metric, and read off the run's own scores where it does not. A
# combination that does not weigh candidates against each other needs neither, and says so rather
# than being handed a column of missing numbers to pick a maximum out of.
.member_scores <- function(oof, y, cells, folds, spec, scores, required) {
  if (!is.null(spec$metric)) {
    score <- .metrics_reg$get(spec$metric)
    f <- .as_folds(folds, rownames(y))
    levels <- sort(unique(f))
    return(vapply(oof, function(p) {
      per <- .cell_means(.score_arm("member", "member", y, p, f, levels, cells, score))
      if (!nrow(per)) NA_real_ else mean(per$score)
    }, numeric(1L)))
  }
  if (!is.null(scores)) {
    level <- .level_means(.cell_means(scores, "candidate"), "candidate")
    return(stats::setNames(level$score[match(names(oof), level$candidate)], names(oof)))
  }
  if (required) {
    stop("a ", spec$method, " combination over the ", spec$scope,
         " candidates weighs them by their score, which is read off `scores` or recomputed from ",
         "a metric named in ensemble(metric = ). Neither was given.", call. = FALSE)
  }
  stats::setNames(rep(NA_real_, length(oof)), names(oof))
}
