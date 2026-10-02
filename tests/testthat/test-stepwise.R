# The stepwise model on the core, against MASS's `stepAIC()`, `glm()` and the forward search over
# column terms in R alone, and the learner above it.
#
# The references are written separately from the core: MASS's search over biomod2's formula for
# every direction, `glm()` on every term for the unselected fit, and the oracle in
# `helper-oracle-stepwise.R` under fractional case weights. The terms chosen are asserted exactly,
# in the order the final model holds them, and the deviance and the fitted means to the tolerance
# each case carries. The Python suite reads the same files.

stepwise_fixture <- function() {
  dir <- fixture_dir()
  if (is.null(dir)) skip("the fixtures ship with the package and are not installed here")
  d <- utils::read.csv(file.path(dir, "penalised_input.csv"), stringsAsFactors = FALSE,
                       check.names = FALSE)
  held <- c("unit", "y_gaussian", "y_binomial", "w", "fold")
  x <- as.matrix(d[, setdiff(names(d), held), drop = FALSE])
  x <- x[, !endsWith(colnames(x), "^2"), drop = FALSE]
  r <- utils::read.csv(file.path(dir, "maxnet_response.csv"), stringsAsFactors = FALSE)
  r <- r[match(d$unit, r$unit), ]
  list(x = x, w = d$w,
       y = list(y_binomial = d$y_binomial, y_8 = r$y_8, y_12 = r$y_12, y_gaussian = d$y_gaussian,
                y_poisson = fixture_counts(dir, d$unit)),
       cases = utils::read.csv(file.path(dir, "stepwise_cases.csv"), stringsAsFactors = FALSE),
       predict = utils::read.csv(file.path(dir, "stepwise_predict.csv"),
                                 stringsAsFactors = FALSE))
}

stepwise_case_fit <- function(fx, row) {
  x <- if (isTRUE(row$duplicate)) cbind(fx$x, fx$x[, 1L]) else fx$x
  y <- fx$y[[row$response]]
  w <- if (isTRUE(row$weighted)) fx$w else rep(1, length(y))
  family <- switch(row$response, y_gaussian = "gaussian", y_poisson = "poisson", "binomial")
  list(x = x, fit = .stepwise_fit(x, y, w, family, max_terms = row$max_terms,
                                  degree = row$degree, direction = row$direction,
                                  terms = row$terms))
}

test_that("the core's search chooses the terms MASS, glm and the R oracle choose", {
  fx <- stepwise_fixture()
  for (i in seq_len(nrow(fx$cases))) {
    row <- fx$cases[i, ]
    got <- stepwise_case_fit(fx, row)
    f <- got$fit
    chosen <- paste(sprintf("%d:%d", f$term_column + 1L, f$term_power), collapse = " ")
    expect_identical(chosen, row$chosen, info = row$case)
    expect_identical(f$rank, row$rank, info = row$case)
    expect_identical(f$converged, row$converged, info = row$case)
    expect_identical(f$steps, row$steps, info = row$case)
    expect_equal(f$deviance, row$deviance, tolerance = row$deviance_tolerance, info = row$case)
    pred <- fx$predict[fx$predict$case == row$case, ]
    expect_equal(.stepwise_predict(f, got$x), pred$fitted, tolerance = row$prediction_tolerance,
                 info = row$case)
    expect_equal(.stepwise_predict(f, got$x * 1.01), pred$fitted_out,
                 tolerance = row$prediction_tolerance, info = row$case)
  }
})

test_that("the forward search over column terms is the R oracle's on a simulated record", {
  sim <- sim_series(n_unit = 60L, days = 120L, seed = 33L)
  y <- sim_response(sim, n_var = 2L, seed = 34L)
  m <- .flatten(grain_matrix(sim$readings, plot, t, temp, grain = "week",
                             stats = c("min", "mean", "max")))
  w <- ifelse(y == 1, 2.5, 1)
  for (j in 1:2) {
    oracle <- oracle_forward_aic(m, y[, j], 3L, 2L, "binomial", w[, j])
    core <- .stepwise_fit(m, y[, j], w[, j], "binomial", max_terms = 3L, degree = 2L)
    expect_identical(core$term_column + 1L, oracle$columns)
    expect_equal(.stepwise_predict(core, m * 1.01), oracle_predict_forward(oracle, m * 1.01),
                 tolerance = 1e-8)
  }
})

test_that("the thread count does not change what the search returns", {
  fx <- stepwise_fixture()
  y <- fx$y$y_binomial
  one <- .stepwise_fit(fx$x, y, fx$w, "binomial", max_terms = Inf, degree = 3L,
                       direction = "both", terms = "power", threads = 1L)
  four <- .stepwise_fit(fx$x, y, fx$w, "binomial", max_terms = Inf, degree = 3L,
                        direction = "both", terms = "power", threads = 4L)
  expect_identical(four, one)
})

test_that("a move whose fit does not settle is refused, and a final fit that does not is named", {
  # Twenty units split by the first column: its fit runs the coefficient outward at every iteration
  # and has not settled after 25, so the search keeps the intercept, as R's own fitter does.
  x <- cbind(1:20, sin(1:20))
  y <- as.numeric(1:20 > 10)
  f <- .stepwise_fit(x, y, rep(1, 20L), "binomial", terms = "power", degree = 1L)
  expect_length(f$term_column, 0L)
  expect_null(oracle_forward_aic(x, y, 3L, 1L, "binomial")$columns)
  full <- .stepwise_fit(x, y, rep(1, 20L), "binomial", direction = "none", terms = "power",
                        degree = 1L)
  expect_false(full$converged)

  units <- sprintf("u%02d", 1:20)
  features <- feature_matrix(matrix(x, ncol = 2L, dimnames = list(units, c("a", "b"))))
  resp <- matrix(y, ncol = 1L, dimnames = list(units, "sp"))
  learned <- fit_learner(stepwise(direction = "none", terms = "power", degree = 1L), features,
                         resp)
  expect_identical(learned$model$stopped, "sp")
})

test_that("stepwise refuses settings it has no search for", {
  expect_error(stepwise(direction = "sideways"), "should be one of")
  expect_error(stepwise(terms = "pair"), "should be one of")
  expect_error(stepwise(max_terms = -1), "zero or more")
  expect_error(stepwise(degree = 1.5), "whole number")
  x <- matrix(1:6, ncol = 1L)
  expect_error(.stepwise_fit(x, c(0, 1, 2, 0, 1, 0), rep(1, 6L), "binomial"), "zero and one")
  expect_error(.stepwise_fit(x, c(0, 1, -2, 0, 1, 0), rep(1, 6L), "poisson"),
               "none of them negative")
  expect_error(.stepwise_fit(x, c(0, 1, 1, 0, 1, 0), c(1, 1, 0, 1, 1, 1), "binomial"),
               "positive weights")
  f <- .stepwise_fit(x, c(0, 1, 1, 0, 1, 0), rep(1, 6L), "binomial")
  expect_error(.stepwise_predict(f, cbind(x, x)), "read 1 columns")
})

test_that("stepwise fits under both heads, round trips, and a model of no term is the share", {
  sim <- sim_series(n_unit = 60L, days = 120L, seed = 41L)
  y <- sim_response(sim, n_var = 2L, seed = 42L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "month")
  for (l in list(stepwise(), stepwise(direction = "both", terms = "power", max_terms = Inf),
                 stepwise(direction = "backward", terms = "power"),
                 stepwise(direction = "none"))) {
    fit <- fit_learner(l, x, y)
    p <- stats::predict(fit, x)
    expect_equal(dim(p), c(60L, 2L))
    expect_true(all(p >= 0 & p <= 1))
    path <- tempfile(fileext = ".rds")
    saveRDS(fit, path)
    expect_identical(stats::predict(readRDS(path), x), p)
    unlink(path)
  }
  empty <- .stepwise_fit(.flatten(x), y[, 1], rep(1, 60L), "binomial", max_terms = 0)
  expect_length(empty$term_column, 0L)
  expect_equal(.stepwise_predict(empty, .flatten(x)), rep(mean(y[, 1]), 60L))
})
