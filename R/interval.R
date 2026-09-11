# The interval for the procedure's risk: the nested cross-validation estimate of the mean squared
# error of a cross-validation estimate, of Bates, Hastie and Tibshirani (2024), Algorithm 1 and
# equation (10). One implementation, read by select_grain() for the selection and by
# grain_ladder() and paired_contrast() for a fixed arm and for the difference between two arms.
#
# The paper writes the procedure's error as the mean of a per-unit loss. Every metric here is read
# on a whole cell and averaged over variables, so the three places the paper averages losses are
# read as follows, each reducing to the paper's quantity when the metric is a mean of per-unit
# losses:
#   mean(e_out)            the score of outer fold k: the mean over the variables scorable there
#   mean(e_in)             the cross-validation estimate on the outer training units, over the
#                          other folds of the same map, averaged as the reported estimate is
#   var(e_out) / |I_k|     the delete-one jackknife variance of the fold score over its units,
#                          which for a mean of per-unit losses is exactly var(e_out) / |I_k|
# and the naive standard error the MSE is bounded by is the jackknife standard error of the
# reported estimate with every prediction held fixed.

.interval_kinds <- c("variables", "nested_cv")

# The fold maps the repetitions are read on. The first is the map the estimate was computed on, so
# its outer predictions are the ones already made; the others are drawn by fold_map() at the same
# number of folds, dealing by the grouping the first map carries.
.ncv_maps <- function(y, f, group, repeats, seed) {
  repeats <- .whole(repeats, "repeats", 1L)
  k <- length(unique(f))
  if (k < 3L) {
    stop("nested cross-validation needs at least three outer folds, so that the cross-validation ",
         "inside each outer training set has two; the fold map has ", k, ".", call. = FALSE)
  }
  maps <- list(stats::setNames(as.integer(f), rownames(y)))
  for (r in seq_len(repeats - 1L) + 1L) {
    m <- fold_map(y, v = k, seed = seed + 7919L * r, strata = if (is.null(group)) 5L else 1L,
                  group = group)
    maps[[r]] <- stats::setNames(.as_folds(m, rownames(y)), rownames(y))
  }
  maps
}

# Every prediction nested cross-validation reads, for one procedure over every map. `fit_predict`
# takes training and test unit indices and a tag naming the fit, and returns the test units'
# predictions with their unit and variable names. Inside repetition r, the cross-validation on the
# training units of outer fold k fits on all but folds k and j and predicts fold j; that training
# set is the one the cross-validation of fold j fits on to predict fold k, so each unordered pair
# of folds is fitted once and predicts both.
.ncv_collect <- function(fit_predict, y, maps, first_outer) {
  units <- rownames(y)
  lapply(seq_along(maps), function(r) {
    m <- maps[[r]]
    levels <- sort(unique(m))
    inner <- lapply(levels, function(k) {
      matrix(NA_real_, nrow = sum(m != k), ncol = ncol(y),
             dimnames = list(units[m != k], colnames(y)))
    })
    names(inner) <- as.character(levels)
    for (pair in utils::combn(levels, 2L, simplify = FALSE)) {
      a <- pair[1L]
      b <- pair[2L]
      pred <- fit_predict(which(m != a & m != b), which(m == a | m == b), c(r, a, b))
      in_a <- units[m == a]
      in_b <- units[m == b]
      inner[[as.character(a)]][in_b, ] <- pred[in_b, colnames(y), drop = FALSE]
      inner[[as.character(b)]][in_a, ] <- pred[in_a, colnames(y), drop = FALSE]
    }
    outer <- if (r == 1L) {
      first_outer
    } else {
      out <- matrix(NA_real_, nrow = nrow(y), ncol = ncol(y), dimnames = dimnames(y))
      for (k in levels) {
        pred <- fit_predict(which(m != k), which(m == k), c(r, k, 0L))
        out[units[m == k], ] <- pred[units[m == k], colnames(y), drop = FALSE]
      }
      out
    }
    list(map = m, inner = inner, outer = outer)
  })
}

# The cell values an estimate is averaged from, for one arm or for the difference of two. `arms`
# is a list of one or two prediction matrices over the same units; a cell counts where every arm
# scored it.
.cell_values <- function(y, arms, f, levels, cells, score) {
  rows <- lapply(arms, function(p) .score_cells(y, p, f, levels, cells, score))
  value <- rows[[1L]]$score
  if (length(rows) == 2L) {
    value <- value - rows[[2L]]$score
  }
  data.frame(variable = rows[[1L]]$variable, fold = rows[[1L]]$fold, score = value,
             stringsAsFactors = FALSE)
}

# The estimate as every level of the package averages it: within a variable over its folds, then
# over variables.
.level_of <- function(values) {
  keep <- values[!is.na(values$score), , drop = FALSE]
  if (!nrow(keep)) {
    return(NA_real_)
  }
  mean(tapply(keep$score, keep$variable, mean))
}

# The delete-one jackknife variance of a statistic of `n` units, `stat(i)` being its value with
# unit i left out.
.jackknife_var <- function(n, stat) {
  if (n < 2L) {
    return(NA_real_)
  }
  theta <- vapply(seq_len(n), stat, numeric(1L))
  (n - 1) / n * sum((theta - mean(theta))^2)
}

# One arm's (or one difference's) score on the units of one fold, with each variable's cell value
# recomputed with one unit left out. A deletion that leaves a cell undefined, a presence-absence
# cell left with one class, keeps the cell's full value.
.fold_jackknife <- function(y, arms, rows, variables, score) {
  full <- .cell_on(y, arms, rows, variables, score)
  .jackknife_var(length(rows), function(i) {
    left <- .cell_on(y, arms, rows[-i], variables, score)
    left[is.na(left)] <- full[is.na(left)]
    mean(left)
  })
}

.cell_on <- function(y, arms, rows, variables, score) {
  vapply(variables, function(v) {
    s <- vapply(arms, function(p) score(y[rows, v], p[rows, v]), numeric(1L))
    if (length(s) == 2L) s[1L] - s[2L] else s[1L]
  }, numeric(1L))
}

# The quantities of Algorithm 1 for one arm or one difference, over every repetition and outer
# fold: the inner estimate, the outer fold score and the jackknife variance of the latter; beside
# them the ordinary estimate on the first map and its naive jackknife standard error.
.ncv_terms <- function(y, runs, cells_fun, score) {
  units <- rownames(y)
  rows <- list()
  for (r in seq_along(runs[[1L]])) {
    m <- runs[[1L]][[r]]$map
    levels <- sort(unique(m))
    cells_all <- cells_fun(y, m)
    outer <- lapply(runs, function(arm) arm[[r]]$outer)
    out_values <- .cell_values(y, outer, m, levels, cells_all, score)
    for (k in levels) {
      train <- which(m != k)
      test <- which(m == k)
      y_in <- y[train, , drop = FALSE]
      m_in <- m[train]
      inner <- lapply(runs, function(arm) arm[[r]]$inner[[as.character(k)]][units[train], ,
                                                                           drop = FALSE])
      s_in <- .level_of(.cell_values(y_in, inner, m_in, setdiff(levels, k),
                                     cells_fun(y_in, m_in), score))
      here <- out_values[out_values$fold == k & !is.na(out_values$score), , drop = FALSE]
      if (!nrow(here) || !is.finite(s_in)) {
        next
      }
      rows[[length(rows) + 1L]] <- data.frame(
        repeat_ = r, fold = k, s_in = s_in, s_out = mean(here$score),
        b = .fold_jackknife(y, outer, test, here$variable, score))
    }
  }
  terms <- do.call(rbind, rows)
  names(terms)[names(terms) == "repeat_"] <- "repeat"

  first <- runs[[1L]][[1L]]
  m <- first$map
  levels <- sort(unique(m))
  outer <- lapply(runs, function(arm) arm[[1L]]$outer)
  values <- .cell_values(y, outer, m, levels, cells_fun(y, m), score)
  list(terms = terms, estimate = .level_of(values),
       se_naive = sqrt(.estimate_jackknife(y, outer, m, values, score)),
       folds = length(levels),
       n_variable = length(unique(values$variable[!is.na(values$score)])))
}

# The jackknife variance of the ordinary estimate, every prediction held fixed: leaving unit i out
# changes only the cells of its own fold.
.estimate_jackknife <- function(y, arms, m, values, score) {
  keep <- values[!is.na(values$score), , drop = FALSE]
  sums <- tapply(keep$score, keep$variable, sum)
  counts <- tapply(keep$score, keep$variable, length)
  variables <- names(sums)
  n <- nrow(y)
  theta <- numeric(n)
  for (i in seq_len(n)) {
    k <- m[i]
    at_k <- keep[keep$fold == k, , drop = FALSE]
    s <- sums
    if (nrow(at_k)) {
      rows <- setdiff(which(m == k), i)
      left <- .cell_on(y, arms, rows, at_k$variable, score)
      left[is.na(left)] <- at_k$score[is.na(left)]
      s[at_k$variable] <- s[at_k$variable] - at_k$score + left
    }
    theta[i] <- mean(s[variables] / counts[variables])
  }
  (n - 1) / n * sum((theta - mean(theta))^2)
}

# Equation (10) of the paper, with the rescaling and the bounds of its section 4.3.2 and the bias
# estimate of its Appendix C: the MSE estimated on n (K - 1) / K units is rescaled by (K - 1) / K,
# its square root is held between the naive standard error and sqrt(K) times it, and the centre is
# the nested estimate less the bias of fitting on fewer units than the whole sample.
.ncv_interval <- function(terms, level = 0.95) {
  t <- terms$terms
  k <- terms$folds
  err_ncv <- mean(t$s_in)
  mse_ncv <- mean((t$s_in - t$s_out)^2) - mean(t$b)
  mse <- (k - 1) / k * mse_ncv
  se_naive <- terms$se_naive
  se <- sqrt(max(mse, 0))
  if (is.finite(se_naive)) {
    se <- min(max(se, se_naive), sqrt(k) * se_naive)
  }
  bias <- (1 + (k - 2) / k) * (err_ncv - terms$estimate)
  centre <- err_ncv - bias
  half <- stats::qnorm(1 - (1 - level) / 2) * se
  data.frame(estimate = terms$estimate, center = centre, se = se, lower = centre - half,
             upper = centre + half, bias = bias, err_ncv = err_ncv, mse_ncv = mse_ncv,
             se_naive = se_naive, repeats = length(unique(t[["repeat"]])), folds = k,
             n_variable = terms$n_variable, stringsAsFactors = FALSE)
}

# The arms a table carries nested cross-validation for, on one set of maps.
.ncv_of <- function(ladder) {
  attr(ladder, "ncv", exact = TRUE)
}

# Two tables' nested cross-validation, read on one set of maps, joined so a contrast between an
# arm of each can be read; a pair drawn on different maps has no pairing to read.
.ncv_join <- function(a, b) {
  if (is.null(a) || is.null(b)) {
    return(NULL)
  }
  if (!identical(lapply(a$maps, unname), lapply(b$maps, unname)) || !identical(a$y, b$y)) {
    stop("the two tables were cross-validated on different fold maps or responses, so their ",
         "nested cross-validation cannot be paired. Give both the same response, folds, ",
         "`repeats` and `seed`.", call. = FALSE)
  }
  list(y = a$y, maps = a$maps,
       arms = c(a$arms, b$arms[setdiff(names(b$arms), names(a$arms))]))
}

# What a table keeps of the nested cross-validation of its arms: the response, the maps, and for
# every arm the inner and outer predictions of every repetition.
.ncv_record <- function(y, maps, arms) {
  list(y = y, maps = maps, arms = arms)
}

# The runs of one arm, rebuilt with the maps they were read on.
.ncv_runs <- function(ncv, arm) {
  if (is.null(ncv) || !arm %in% names(ncv$arms)) {
    stop("no nested cross-validation for the arm \"", arm, "\". Fit it with ",
         "`interval = \"nested_cv\"` in grain_ladder() or select_grain().", call. = FALSE)
  }
  lapply(seq_along(ncv$maps), function(r) {
    c(list(map = ncv$maps[[r]]), ncv$arms[[arm]][[r]])
  })
}

# One arm's runs as a table keeps them.
.ncv_keep <- function(runs) {
  lapply(runs, function(x) x[c("inner", "outer")])
}

# The interval for one arm or one difference, read off a table's record under one metric.
.ncv_read <- function(ncv, arms, response, score) {
  spec <- .responses_reg$get(response)
  cells_fun <- function(y, m) spec$cells(y, stats::setNames(m, rownames(y)))
  runs <- lapply(arms, function(a) .ncv_runs(ncv, a))
  .ncv_interval(.ncv_terms(ncv$y, runs, cells_fun, score))
}

.check_interval <- function(interval) {
  if (!is.character(interval) || length(interval) != 1L || !interval %in% .interval_kinds) {
    stop("`interval` is one of ", paste0("\"", .interval_kinds, "\"", collapse = ", "), ", got ",
         paste(deparse(interval), collapse = ""), ".", call. = FALSE)
  }
  interval
}

# What each interval is an interval for, in the words the printed objects use.
.interval_target <- function(interval) {
  switch(interval,
         variables = "the spread across the variables of this dataset, fitted and scored on these units and folds",
         nested_cv = "the procedure's risk on a new sample of this size, by nested cross-validation")
}
