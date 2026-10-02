# A repeated resampling is the run made once per repeat, each on a fold map of its own, and the
# runs read together. The scores of a response are averaged over its folds and its repeats, which
# keeps a cell one fit's held-out score and a response the replicate; the combiner is fitted on
# the out-of-fold predictions of all the repeats at once.

.timesift_repeated <- function(call, env, resampling, ensemble, response, verbose) {
  total <- resampling$repeats
  runs <- lapply(seq_len(total), function(r) {
    if (verbose) {
      message("repeat ", r, " of ", total)
    }
    again <- call
    again[[1L]] <- quote(timesift::timesift)
    again$resampling <- .nth_repeat(resampling, r)
    again$verbose <- verbose && r == 1L
    again$.refit <- r == 1L
    eval(again, env)
  })
  .combine_repeats(runs, call, ensemble, response)
}

.combine_repeats <- function(runs, call, ensemble, response) {
  total <- length(runs)
  first <- runs[[1L]]
  units <- rownames(first$y)
  variables <- colnames(first$y)
  folds_of <- lapply(runs, function(run) .as_folds(run$folds, units))
  width <- max(unlist(folds_of))
  shift <- function(r) (r - 1L) * width
  tag <- function(table, r) {
    table <- as.data.frame(table)
    table$fold <- table$fold + shift(r)
    table[["repeat"]] <- r
    table
  }
  stack_rows <- function(parts) {
    out <- do.call(rbind, parts)
    rownames(out) <- NULL
    out
  }
  pool <- function(m, r) {
    m <- as.matrix(m)[units, variables, drop = FALSE]
    rownames(m) <- paste0(units, "@", r)
    m
  }
  pooled <- function(per_run) do.call(rbind, lapply(seq_len(total), function(r) pool(per_run[[r]], r)))
  mean_over <- function(per_run) {
    Reduce(`+`, lapply(per_run, function(m) as.matrix(m)[units, variables, drop = FALSE])) / total
  }

  scores <- stack_rows(Map(function(run, r) tag(run$scores, r), runs, seq_len(total)))
  cells <- structure(stack_rows(Map(function(run, r) tag(run$cells, r), runs, seq_len(total))),
                     class = class(first$cells))
  y_pool <- pooled(rep(list(first$y), total))
  f_pool <- stats::setNames(unlist(lapply(seq_len(total), function(r) folds_of[[r]] + shift(r))),
                            rownames(y_pool))
  levels <- sort(unique(f_pool))

  candidates <- names(first$oof)
  oof_pool <- stats::setNames(lapply(candidates, function(cd) {
    pooled(lapply(runs, function(run) run$oof[[cd]]))
  }), candidates)
  oof <- stats::setNames(lapply(candidates, function(cd) {
    mean_over(lapply(runs, function(run) run$oof[[cd]]))
  }), candidates)

  stack <- if (!is.null(first$stack)) {
    ensemble_fit(oof = oof_pool, y = y_pool, cells = cells, folds = f_pool,
                 spec = .run_ensemble(ensemble, response), scores = scores)
  }

  estimate <- selected <- inner <- fold_weights <- predictions <- NULL
  if (!is.null(first$estimate)) {
    metric <- list(fn = first$scorer, name = first$metric)
    predictions <- list(selected = mean_over(lapply(runs, function(run) run$predictions$selected)))
    estimate <- .run_estimate("selected", y_pool, pooled(lapply(runs, function(run) {
      run$predictions$selected
    })), f_pool, levels, cells, metric)
    if (!is.null(first$predictions$ensemble)) {
      predictions$ensemble <- mean_over(lapply(runs, function(run) run$predictions$ensemble))
      estimate <- rbind(estimate, .run_estimate("ensemble", y_pool, pooled(lapply(runs, function(run) {
        run$predictions$ensemble
      })), f_pool, levels, cells, metric))
    }
    selected <- stack_rows(Map(function(run, r) tag(run$selected, r), runs, seq_len(total)))
    if (!is.null(first$inner)) {
      inner <- stack_rows(Map(function(run, r) tag(run$inner, r), runs, seq_len(total)))
    }
    if (!is.null(first$fold_weights)) {
      fold_weights <- stack_rows(Map(function(run, r) tag(run$fold_weights, r), runs,
                                     seq_len(total)))
    }
  }

  votes <- table(factor(vapply(runs, function(run) run$choice, character(1L)),
                        levels = unique(vapply(runs, function(run) run$choice, character(1L)))))
  folds <- first$folds
  attr(folds, "repeats") <- total

  structure(list(estimate = estimate, selected = selected, inner = inner,
                 fold_weights = fold_weights, predictions = predictions,
                 candidates = first$candidates, scores = scores, oof = oof,
                 choice = names(votes)[which.max(votes)], models = first$models, stack = stack,
                 weights = stack$weights, representations = first$representations,
                 fits = first$fits, folds = folds, cells = cells, y = first$y,
                 metric = first$metric, scorer = first$scorer, response = first$response,
                 spec = first$spec, call = call, repeats = total),
            class = "timesift")
}
