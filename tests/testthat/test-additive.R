# The additive model on the core, against mgcv, and the learner above it.
#
# The reference is mgcv's own fit on the fixture designs, written by `inst/spec/make_fixtures.R`
# at a tight tolerance: the criterion at its minimum, every column's effective degrees of freedom
# and the fitted mean on the design scaled by 1.01. Two searches of the same criterion settle at
# the same point to the tolerance they are run at, so each is asserted as a distance, to the
# tolerance its case carries. The Python suite reads the same files.

additive_fixture <- function() {
  dir <- fixture_dir()
  if (is.null(dir)) skip("the fixtures ship with the package and are not installed here")
  read <- function(f) utils::read.csv(file.path(dir, f), stringsAsFactors = FALSE,
                                      check.names = FALSE)
  d <- read("penalised_input.csv")
  held <- c("unit", "y_gaussian", "y_binomial", "w", "fold")
  x <- as.matrix(d[, setdiff(names(d), held), drop = FALSE])
  x <- unname(x[, !endsWith(colnames(x), "^2"), drop = FALSE])
  few <- read("additive_input.csv")
  knots <- read("additive_knots_input.csv")
  list(designs = list(weekly = x[, 1:4], first = x[, 1:3], late = x[, 5:8], all = x,
                      few = unname(as.matrix(few[, -1L])),
                      knots = unname(as.matrix(knots[, c("v01", "v02")]))),
       y = list(y_binomial = d$y_binomial, y_gaussian = d$y_gaussian, y = knots$y),
       w = d$w, cases = read("additive_cases.csv"), predict = read("additive_predict.csv"))
}

additive_case_fit <- function(fx, row) {
  x <- fx$designs[[row$design]]
  y <- fx$y[[row$response]]
  w <- if (isTRUE(row$weighted)) fx$w else rep(1, length(y))
  family <- if (row$response == "y_gaussian") "gaussian" else "binomial"
  list(x = x, fit = .additive_fit(x, y, w, family, k = row$k, gamma = row$gamma))
}

test_that("the core settles where mgcv does, to the criterion, the degrees of freedom and the fit", {
  fx <- additive_fixture()
  for (i in seq_len(nrow(fx$cases))) {
    row <- fx$cases[i, ]
    cf <- additive_case_fit(fx, row)
    f <- cf$fit
    expect_identical(f$converged, 1L, info = row$case)
    expect_equal(f$score, row$score, tolerance = row$score_tolerance, info = row$case)
    edf <- as.numeric(strsplit(row$edf, " ", fixed = TRUE)[[1L]])
    expect_equal(f$edf, edf, tolerance = row$edf_tolerance, info = row$case)
    pred <- fx$predict[fx$predict$case == row$case, ]
    expect_equal(drop(.additive_predict(f, cf$x[pred$row, , drop = FALSE] * 1.01)), pred$fitted,
                 tolerance = row$prediction_tolerance, info = row$case)
  }
})

test_that("past 2000 distinct values the knots are the ones set.seed(1) and sample() draw", {
  fx <- additive_fixture()
  x <- fx$designs$knots
  f <- .additive_fit(x, fx$y$y, rep(1, nrow(x)), "binomial")
  for (j in 1:2) {
    u <- sort(unique(x[, j]))
    expected <- withr::with_seed(1L, sort(u[sample(length(u), 2000L)]))
    got <- f$knots[(f$knot_start[j] + 1L):f$knot_start[j + 1L]]
    expect_identical(got, expected)
  }
})

test_that("responses fitted together are the responses fitted alone, on any number of threads", {
  fx <- additive_fixture()
  x <- fx$designs$weekly
  y <- cbind(fx$y$y_binomial, 1 - fx$y$y_binomial)
  w <- cbind(fx$w, rev(fx$w))
  both <- .additive_fit(x, y, w, "binomial")
  for (s in 1:2) {
    alone <- .additive_fit(x, y[, s], w[, s], "binomial")
    expect_identical(both$beta[(s - 1L) * both$n_coef + seq_len(both$n_coef)], alone$beta)
    expect_identical(both$score[s], alone$score)
  }
  expect_identical(.additive_fit(x, y, w, "binomial", threads = 4L), both)
  expect_identical(.additive_fit(fx$designs$all, fx$y$y_gaussian, fx$w, "gaussian", k = 5L,
                                 threads = 4L),
                   .additive_fit(fx$designs$all, fx$y$y_gaussian, fx$w, "gaussian", k = 5L))
})

test_that("a column of two values enters linearly and a column of one is left out", {
  set.seed(3)
  x <- cbind(stats::rnorm(60), rep(c(2, 5), 30), rep(1, 60))
  y <- stats::rbinom(60, 1, stats::plogis(x[, 1]))
  f <- .additive_fit(x, y, rep(1, 60), "binomial")
  expect_identical(f$term_column, c(0L, 1L))
  expect_identical(f$term_size, c(9L, 1L))
  expect_identical(f$term_penalised, c(8L, 0L))
  expect_equal(f$edf[2L], 1)
  expect_identical(f$n_coef, 11L)
})

test_that("a linear part the columns before it span is held at zero", {
  set.seed(4)
  a <- stats::rnorm(60)
  x <- cbind(a, 2 * a + 1)
  y <- stats::rbinom(60, 1, stats::plogis(a))
  f <- .additive_fit(x, y, rep(1, 60), "binomial")
  expect_identical(f$aliased, 18L)
  expect_identical(f$beta[19L], 0)
})

test_that("the additive model refuses settings it has no fit for", {
  expect_error(additive(k = 2), "whole number")
  expect_error(additive(k = 10, max_knots = 5), "at least `k`")
  expect_error(additive(gamma = 0), "positive")
  expect_error(additive(threads = 0), "whole number")
  x <- matrix(stats::rnorm(60), 20)
  expect_error(.additive_fit(x, rep(0:1, 10), c(-1, rep(1, 19)), "binomial"), "positive")
  expect_error(.additive_fit(x, rep(0:1, 10), rep(1, 20), "binomial"), "more than the 20 units")
  f <- .additive_fit(x[, 1, drop = FALSE], rep(0:1, 10), rep(1, 20), "binomial", k = 5L)
  expect_error(.additive_predict(f, x), "fitted on 1 columns")
})

test_that("the learner fits under both heads, round trips, and a constant response is its mean", {
  sim <- sim_series(n_unit = 60L, days = 120L, seed = 43L)
  y <- sim_response(sim, n_var = 2L, seed = 44L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "season")
  for (l in list(additive(), additive(k = 5L, gamma = 1.4))) {
    fit <- fit_learner(l, x, y)
    p <- stats::predict(fit, x)
    expect_equal(dim(p), c(60L, 2L))
    expect_true(all(p >= 0 & p <= 1))
    path <- tempfile(fileext = ".rds")
    saveRDS(fit, path)
    expect_identical(stats::predict(readRDS(path), x), p)
    unlink(path)
  }
  flat <- y
  flat[, 2L] <- 0
  fit <- fit_learner(additive(), x, flat)
  expect_identical(fit$model$unfitted, colnames(y)[2L])
  expect_equal(unique(stats::predict(fit, x)[, 2L]), 0)

  local_response("continuous_test", list(
    prepare = function(y) .as_response(y), activation = "identity",
    loss = "squared_error", metric = "roc_auc",
    cells = function(y, folds) scorable_cells(y > stats::median(y), folds)))
  level <- 10 + 3 * scale(rowMeans(x[, , 1L]))[, 1L]
  yc <- matrix(level, ncol = 1L, dimnames = list(dimnames(x)[[1L]], "height"))
  fit <- fit_learner(additive(), x, yc, response = "continuous_test")
  expect_equal(fit$model$family, "gaussian")
  expect_gt(stats::cor(stats::predict(fit, x)[, 1L], yc[, 1L]), 0.95)
})

# A fit at given smoothing parameters, against the penalised problem solved outright.
#
# The design and the penalty are read off the fit, each term's columns being its raw basis times
# its map and its penalty a diagonal, and the problem is solved by a singular value decomposition
# of the weighted design stacked on the root of the penalty, which takes no rank decision on the
# way. Under the binomial family the solve is repeated at the working weights until the
# coefficients stop moving. The core is asserted to land on the same coefficients, fitted means,
# effective degrees of freedom and criterion, at smoothing parameters spread over sixteen orders of
# magnitude: a solve that loses the directions a column's penalty holds, because another column's
# is so much larger, reads wrong in the fitted means here.
additive_model_columns <- function(f, x) {
  cols <- list(matrix(1, nrow(x), 1L))
  for (t in seq_along(f$term_column)) {
    xv <- x[, f$term_column[t] + 1L]
    basis <- f$term_basis[t]
    raw <- if (basis > 2L) {
      at <- (f$knot_start[t] + 1L):f$knot_start[t + 1L]
      rad <- matrix(f$radial[(f$radial_start[t] + 1L):f$radial_start[t + 1L]], length(at))
      cbind((abs(outer(xv, f$knots[at], "-"))^3 / 12) %*% rad, 1, xv - f$term_shift[t])
    } else {
      cbind(1, xv - f$term_shift[t])
    }
    map <- matrix(f$map[(f$map_start[t] + 1L):f$map_start[t + 1L]], basis)
    cols[[t + 1L]] <- raw %*% map
  }
  do.call(cbind, cols)
}

additive_model_penalty <- function(f, sp) {
  pen <- numeric(f$n_coef)
  at <- 1L
  from <- 0L
  smooth <- 0L
  for (t in seq_along(f$term_column)) {
    m <- f$term_penalised[t]
    if (m > 0L) {
      smooth <- smooth + 1L
      pen[at + seq_len(m)] <- sp[smooth] * f$penalty[from + seq_len(m)]
      from <- from + m
    }
    at <- at + f$term_size[t]
  }
  pen
}

additive_exact <- function(f, x, y, w, sp, gamma = 1) {
  xm <- additive_model_columns(f, x)
  pen <- additive_model_penalty(f, sp)
  keep <- setdiff(seq_len(f$n_coef), f$aliased + 1L)
  xm <- xm[, keep, drop = FALSE]
  pen <- pen[keep]
  binomial <- f$family == "binomial"
  n <- nrow(xm)
  solve_at <- function(root, z) {
    a <- rbind(root * xm, diag(sqrt(pen), length(pen)))
    s <- svd(a)
    list(beta = drop(s$v %*% ((crossprod(s$u, c(root * z, numeric(length(pen))))) / s$d)),
         influence = s$v %*% (t(s$u[seq_len(n), , drop = FALSE]) / s$d) %*% (root * xm))
  }
  deviance <- function(mu) {
    if (binomial) {
      2 * sum(w * (ifelse(y > 0, y * log(y / mu), 0) +
                   ifelse(y < 1, (1 - y) * log((1 - y) / (1 - mu)), 0)))
    } else {
      sum(w * (y - mu)^2)
    }
  }
  penalised <- function(beta, mu) deviance(mu) + sum(pen * beta^2)
  mu <- if (binomial) (w * y + 0.5) / (w + 1) else y
  eta <- if (binomial) stats::qlogis(mu) else mu
  beta <- NULL
  for (i in seq_len(if (binomial) 500L else 1L)) {
    wt <- if (binomial) w * mu * (1 - mu) else w
    z <- if (binomial) eta + (y - mu) / (mu * (1 - mu)) else y
    sol <- solve_at(sqrt(wt), z)
    next_beta <- sol$beta
    if (!is.null(beta)) {
      before <- penalised(beta, mu)
      for (h in 1:40) {
        next_mu <- stats::plogis(drop(xm %*% next_beta))
        if (penalised(next_beta, next_mu) <= before) break
        next_beta <- (next_beta + beta) / 2
      }
    }
    moved <- if (is.null(beta)) Inf else max(abs(next_beta - beta))
    beta <- next_beta
    eta <- drop(xm %*% beta)
    mu <- if (binomial) stats::plogis(eta) else eta
    if (!binomial || moved < 1e-11 * (1 + max(abs(beta)))) break
  }
  if (binomial) {
    expect_lt(moved, 1e-8 * (1 + max(abs(beta))))
    sol <- solve_at(sqrt(w * mu * (1 - mu)), eta + (y - mu) / (mu * (1 - mu)))
  }
  dev <- deviance(mu)
  per_coef <- numeric(f$n_coef)
  per_coef[keep] <- diag(sol$influence)
  edf <- vapply(seq_along(f$term_column), function(t) {
    first <- 1L + sum(f$term_size[seq_len(t - 1L)])
    sum(per_coef[first + seq_len(f$term_size[t]) - 1L])
  }, numeric(1L))
  tau <- sum(diag(sol$influence))
  score <- if (binomial) dev / n + 2 * gamma * tau / n - 1 else n * dev / (n - gamma * tau)^2
  list(mu = mu, edf = edf, score = score)
}

additive_sp_patterns <- function(m) {
  list(flat = rep(1, m),
       ramp = 10^seq(-4, 12, length.out = m),
       reversed = 10^seq(12, -4, length.out = m),
       alternating = 10^rep(c(-4, 12), length.out = m),
       one_large = c(1e12, rep(1e-4, m - 1L)),
       one_small = c(1e-4, rep(1e3, m - 1L)),
       all_large = rep(1e10, m))
}

test_that("a fit at given smoothing parameters is the exact penalised solve, whatever their spread", {
  fx <- additive_fixture()
  cases <- list(
    list(design = "weekly", response = "y_binomial", family = "binomial", k = 10L, gamma = 1),
    list(design = "all", response = "y_gaussian", family = "gaussian", k = 5L, gamma = 1.4),
    list(design = "few", response = "y_binomial", family = "binomial", k = 5L, gamma = 1))
  for (cs in cases) {
    x <- fx$designs[[cs$design]]
    y <- fx$y[[cs$response]]
    w <- fx$w
    start <- .additive_fit(x, y, w, cs$family, k = cs$k, gamma = cs$gamma)
    patterns <- additive_sp_patterns(length(start$sp))
    # The stacked system's condition number is the root of the smoothing parameters' spread, about
    # 1e8 here, and an inner fit under the binomial family stops where the penalised deviance moves
    # by 1e-13, which in a nearly flat direction is a coefficient error of 1e-8 or so.
    tol <- list(mu = 1e-7, edf = 1e-6, score = 1e-7)
    for (nm in names(patterns)) {
      sp <- patterns[[nm]]
      info <- paste(cs$design, nm)
      f <- .additive_fit(x, y, w, cs$family, k = cs$k, gamma = cs$gamma, sp = sp)
      ref <- additive_exact(f, x, y, w, sp, cs$gamma)
      expect_identical(f$converged, 1L, info = info)
      expect_identical(f$outer, 0L, info = info)
      expect_identical(f$sp, sp, info = info)
      expect_equal(drop(.additive_predict(f, x)), ref$mu, tolerance = tol$mu, info = info)
      expect_equal(f$edf, ref$edf, tolerance = tol$edf, info = info)
      expect_equal(f$score, ref$score, tolerance = tol$score, info = info)
    }
  }
})

test_that("smoothing parameters given to a fit are counted and positive", {
  fx <- additive_fixture()
  x <- fx$designs$weekly
  m <- length(.additive_fit(x, fx$y$y_binomial, fx$w, "binomial")$sp)
  expect_error(.additive_fit(x, fx$y$y_binomial, fx$w, "binomial", sp = rep(1, m + 1L)),
               "smoothing parameters")
  expect_error(.additive_fit(x, fx$y$y_binomial, fx$w, "binomial", sp = c(0, rep(1, m - 1L))),
               "positive and finite")
})
