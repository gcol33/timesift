# The discriminant on the core, against the mda package, and the learner above it.
#
# The reference is mda's own `fda(method = mars)` on the fixture design, written by
# `inst/spec/make_fixtures.R`: the forward terms counted, the kept terms, their coefficients, and
# on the design scaled by 1.01 the posterior and biomod2's probit recalibration of it. The kept
# terms are asserted exactly, each cut to the tolerance its case carries; the coefficients and the
# predictions likewise. The Python suite reads the same files.

fda_fixture <- function() {
  dir <- fixture_dir()
  if (is.null(dir)) skip("the fixtures ship with the package and are not installed here")
  d <- utils::read.csv(file.path(dir, "penalised_input.csv"), stringsAsFactors = FALSE,
                       check.names = FALSE)
  held <- c("unit", "y_gaussian", "y_binomial", "w", "fold")
  squares <- as.matrix(d[, setdiff(names(d), held), drop = FALSE])
  list(x = list(columns = squares[, !endsWith(colnames(squares), "^2"), drop = FALSE],
                squares = squares),
       y = d$y_binomial, w = d$w,
       cases = utils::read.csv(file.path(dir, "fda_cases.csv"), stringsAsFactors = FALSE),
       coef = utils::read.csv(file.path(dir, "fda_coef.csv"), stringsAsFactors = FALSE),
       predict = utils::read.csv(file.path(dir, "fda_predict.csv"), stringsAsFactors = FALSE))
}

fda_case_fit <- function(fx, row, calibrate) {
  w <- if (isTRUE(row$weighted)) fx$w else rep(1, length(fx$y))
  .fda_fit(fx$x[[row$design]], fx$y, w, degree = row$degree, prune = isTRUE(row$prune),
           calibrate = calibrate)
}

test_that("the core keeps the terms mda keeps and predicts its posterior and biomod2's", {
  fx <- fda_fixture()
  for (i in seq_len(nrow(fx$cases))) {
    row <- fx$cases[i, ]
    x <- fx$x[[row$design]]
    f <- fda_case_fit(fx, row, calibrate = FALSE)
    expect_hinge_terms(f, row$kept, row$cut_tolerance, row$case)
    expect_identical(f$forward_terms, row$n_forward, info = row$case)
    expect_equal(f$gcv, row$gcv, tolerance = row$coef_tolerance, info = row$case)
    coef <- fx$coef[fx$coef$case == row$case, ]
    expect_equal(f$coef, coef$coefficient, tolerance = row$coef_tolerance, info = row$case)
    pred <- fx$predict[fx$predict$case == row$case, ]
    expect_equal(.fda_predict(f, x * 1.01), pred$posterior,
                 tolerance = row$prediction_tolerance, info = row$case)
    calibrated <- fda_case_fit(fx, row, calibrate = TRUE)
    expect_true(calibrated$converged, info = row$case)
    expect_equal(.fda_predict(calibrated, x * 1.01), pred$probability,
                 tolerance = row$prediction_tolerance, info = row$case)
  }
})

test_that("the case weights move the scores and not the basis", {
  fx <- fda_fixture()
  weighted <- .fda_fit(fx$x$columns, fx$y, fx$w, calibrate = FALSE)
  plain <- .fda_fit(fx$x$columns, fx$y, rep(1, length(fx$y)), calibrate = FALSE)
  expect_identical(hinge_terms(weighted), hinge_terms(plain))
  expect_false(isTRUE(all.equal(weighted$coef, plain$coef)))
})

test_that("the thread count does not change what the discriminant returns", {
  fx <- fda_fixture()
  for (degree in 1:2) {
    one <- .fda_fit(fx$x$squares, fx$y, fx$w, degree = degree, threads = 1L)
    four <- .fda_fit(fx$x$squares, fx$y, fx$w, degree = degree, threads = 4L)
    expect_identical(four, one)
  }
})

test_that("discriminant refuses settings and responses it has no fit for", {
  expect_error(discriminant(degree = 0), "whole number")
  expect_error(discriminant(nk = 2), "whole number")
  expect_error(discriminant(penalty = -1), "zero or more")
  expect_error(discriminant(thresh = 1), "\\[0, 1\\)")
  expect_error(discriminant(prune = NA), "TRUE or FALSE")
  expect_error(discriminant(calibrate = "yes"), "TRUE or FALSE")
  x <- matrix(c(1, 3, 2, 5, 4, 6), ncol = 1L)
  expect_error(.fda_fit(x, c(0, 1, 0, 2, 0, 1), rep(1, 6L)), "presences from absences")
  expect_error(.fda_fit(x, rep(1, 6L), rep(1, 6L)), "both classes")
  expect_error(.fda_fit(x, c(0, 1, 0, 1, 0, 1), c(1, 1, 0, 1, 1, 1)), "above zero")
  f <- .fda_fit(x, c(0, 1, 0, 1, 0, 1), rep(1, 6L))
  expect_error(.fda_predict(f, cbind(x, x)), "fitted on 1 columns")
})

test_that("discriminant fits, round trips, and a constant response is its mean", {
  sim <- sim_series(n_unit = 60L, days = 120L, seed = 43L)
  y <- sim_response(sim, n_var = 2L, seed = 44L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "month")
  for (l in list(discriminant(), discriminant(degree = 2L), discriminant(calibrate = FALSE),
                 discriminant(prune = FALSE))) {
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
  flat[, 1L] <- 0
  fit <- fit_learner(discriminant(), x, flat)
  expect_equal(unique(stats::predict(fit, x)[, 1L]), 0)
  expect_identical(fit$model$unfitted, colnames(y)[1L])
})

test_that("discriminant refuses a head whose loss is not the binary cross-entropy", {
  local_response("continuous_test", list(
    prepare = function(y) .as_response(y), activation = "identity",
    loss = "squared_error", metric = "roc_auc",
    cells = function(y, folds) scorable_cells(y > stats::median(y), folds)))
  sim <- sim_series(n_unit = 30L, days = 60L, seed = 91L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "month")
  y <- matrix(stats::rnorm(30L), ncol = 1L, dimnames = list(dimnames(x)[[1L]], "height"))
  expect_error(fit_learner(discriminant(), x, y, response = "continuous_test"),
               "binary cross-entropy")
})
