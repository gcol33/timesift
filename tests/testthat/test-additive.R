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
