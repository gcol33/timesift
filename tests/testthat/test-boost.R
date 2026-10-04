# The boosting core, against gbm, xgboost and the spec's own draws, and the learner above it.
#
# gbm is the reference for the first-order trees wherever nothing is drawn, its cross-validation
# included; xgboost for the second-order ones, to its single-precision storage; and
# `helper-oracle-boost.R` for the subsample and the column draw. The Python suite reads the same
# files.

boost_fixture <- function() {
  dir <- fixture_dir()
  if (is.null(dir)) skip("the fixtures ship with the package and are not installed here")
  d <- utils::read.csv(file.path(dir, "penalised_input.csv"), stringsAsFactors = FALSE,
                       check.names = FALSE)
  held <- c("unit", "y_gaussian", "y_binomial", "w", "fold")
  counts <- utils::read.csv(file.path(dir, "tree_weights.csv"), stringsAsFactors = FALSE)
  random <- utils::read.csv(file.path(dir, "boost_weights.csv"), stringsAsFactors = FALSE)
  list(x = as.matrix(d[, setdiff(names(d), held), drop = FALSE]), binomial = d$y_binomial,
       gaussian = d$y_gaussian, poisson = fixture_counts(dir, d$unit), fold = d$fold,
       weights = list(flat = rep(1, nrow(d)), counts = counts$count[match(d$unit, counts$unit)],
                      random = random$weight[match(d$unit, random$unit)]),
       cases = utils::read.csv(file.path(dir, "boost_cases.csv"), stringsAsFactors = FALSE,
                               colClasses = c(seed = "numeric")),
       predict = utils::read.csv(file.path(dir, "boost_predict.csv"), stringsAsFactors = FALSE),
       cv = utils::read.csv(file.path(dir, "boost_cv.csv"), stringsAsFactors = FALSE))
}

boost_case_fit <- function(fx, row) {
  y <- fx[[row$family]]
  cross <- row$cv == 1L
  .boost_fit(fx$x, y, fx$weights[[row$weights]], row$family, row$trees, row$depth, row$shrinkage,
             row$min_leaf, row$subsample, row$colsample, row$newton == 1L, row$lambda, row$gamma,
             row$seed, if (cross) fx$fold else NULL, if (cross) 5L else 0L)
}

test_that("boosting predicts what gbm, xgboost and the spec's draws predict", {
  fx <- boost_fixture()
  for (i in seq_len(nrow(fx$cases))) {
    row <- fx$cases[i, ]
    ref <- fx$predict$value[fx$predict$case == row$case]
    tolerance <- if (row$reference == "xgboost") 1e-5 else 1e-12
    expect_equal(.boost_predict(boost_case_fit(fx, row), fx$x), ref, tolerance = tolerance,
                 info = row$case)
  }
})

test_that("the inner cross-validation reads gbm's held-out error and keeps gbm's number", {
  fx <- boost_fixture()
  for (case in unique(fx$cv$case)) {
    row <- fx$cases[fx$cases$case == case, ]
    fit <- boost_case_fit(fx, row)
    ref <- fx$cv[fx$cv$case == case, ]
    expect_equal(fit$cv_error, ref$cv_error, tolerance = 1e-12, info = case)
    expect_identical(length(fit$offset) - 1L, which.min(ref$cv_error), info = case)
  }
})

test_that("the core and the oracle agree off the fixtures too", {
  set.seed(90)
  n <- 50L
  x <- cbind(round(stats::rnorm(n), 1), stats::rnorm(n), sample(1:4, n, replace = TRUE))
  yb <- as.numeric(x[, 1] + x[, 3] / 2 + stats::rnorm(n) > 1)
  w <- stats::runif(n, 0.5, 2)
  core <- .boost_fit(x, yb, w, "binomial", 8L, 2L, 0.2, 2, 0.8, 0.7, FALSE, 0, 0, 33)
  ref <- oracle_boost(x, yb, w, "binomial", 8L, 2L, 0.2, 2, 0.8, 0.7, 33)
  expect_equal(core$init, ref$init)
  expect_equal(.boost_predict(core, x), oracle_boost_predict(ref, x, TRUE), tolerance = 1e-12)
})

test_that("a boosted fit is the same on any number of threads", {
  set.seed(91)
  x <- matrix(stats::rnorm(400), 80L)
  y <- as.numeric(x[, 1] + stats::rnorm(80L) > 0)
  fold <- rep_len(0:3, 80L)
  one <- .boost_fit(x, y, rep(1, 80L), "binomial", 30L, 2L, 0.1, 3, 0.5, 1, FALSE, 0, 0, 5,
                    fold, 4L)
  expect_identical(.boost_fit(x, y, rep(1, 80L), "binomial", 30L, 2L, 0.1, 3, 0.5, 1, FALSE, 0,
                              0, 5, fold, 4L, threads = 3L), one)
})

test_that("the boosting core refuses what it cannot fit", {
  x <- matrix(c(1, 2, 3, 4), ncol = 1L)
  fit <- function(y, w = rep(1, 4), ...) {
    args <- utils::modifyList(list(trees = 2L, depth = 1L, shrinkage = 0.1, min_leaf = 1,
                                   subsample = 1, colsample = 1, newton = FALSE, lambda = 0,
                                   gamma = 0, seed = 1), list(...))
    .boost_fit(x, y, w, "binomial", args$trees, args$depth, args$shrinkage, args$min_leaf,
               args$subsample, args$colsample, args$newton, args$lambda, args$gamma, args$seed)
  }
  expect_error(fit(c(0, 1, 0, 2)), "0 and 1 alone")
  expect_error(.boost_fit(x, c(0, 1, -2, 2), rep(1, 4), "poisson", 2L, 1L, 0.1, 1, 1, 1, FALSE,
                          0, 0, 1), "zero or more")
  expect_error(.boost_fit(x, c(0, 0, 0, 0), rep(1, 4), "poisson", 2L, 1L, 0.1, 1, 1, 1, FALSE,
                          0, 0, 1), "count above zero")
  expect_error(fit(c(0, 1, NA, 1)), "finite values")
  expect_error(fit(c(0, 1, 0, 1), c(1, -1, 1, 1)), "zero or more")
  expect_error(fit(c(0, 1, 0, 1), c(1, 0, 1, 0)), "both classes")
  expect_error(fit(c(0, 1, 0, 1), subsample = 1.5), "subsample")
  expect_error(fit(c(0, 1, 0, 1), subsample = 0.1), "holds no observation")
  expect_error(fit(c(0, 1, 0, 1), shrinkage = 0), "shrinkage")
  expect_error(fit(c(0, 1, 0, 1), depth = 0L), "depth")
  expect_error(.boost_predict(fit(c(0, 0, 1, 1)), matrix(NA_real_, 1L, 1L)), "finite values")
})

test_that("a preset fills the settings left open, as gbm, xgboost and biomod2 have them", {
  p <- boosting()$params
  expect_identical(p[c("trees", "depth", "n_inner")],
                   list(trees = 100L, depth = 1L, n_inner = 0L))
  expect_equal(unlist(p[c("shrinkage", "min_leaf", "subsample", "lambda")]),
               c(shrinkage = 0.1, min_leaf = 10, subsample = 0.5, lambda = 0))
  b <- boosting(preset = "bigboss")$params
  expect_identical(b[c("trees", "depth", "n_inner")],
                   list(trees = 2500L, depth = 7L, n_inner = 3L))
  expect_equal(unlist(b[c("shrinkage", "min_leaf")]), c(shrinkage = 0.001, min_leaf = 5))
  xg <- boosting(method = "xgboost")$params
  expect_identical(xg[c("trees", "depth")], list(trees = 100L, depth = 6L))
  expect_equal(unlist(xg[c("shrinkage", "min_leaf", "lambda", "subsample")]),
               c(shrinkage = 0.3, min_leaf = 1, lambda = 1, subsample = 1))
  xb <- boosting(method = "xgboost", preset = "bigboss")$params
  expect_identical(xb[c("trees", "depth")], list(trees = 4L, depth = 2L))
  expect_equal(xb$shrinkage, 1)
  expect_identical(boosting(preset = "bigboss", trees = 300L)$params$trees, 300L)
  expect_error(boosting(lambda = 1), "method = \"xgboost\"", fixed = TRUE)
})

test_that("boosting fits, predicts, survives a round trip and refuses a different representation", {
  sim <- sim_series(n_unit = 60L, days = 60L, seed = 35L)
  y <- sim_response(sim, n_var = 2L, seed = 36L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "week")
  fit <- fit_learner(boosting(), x, y)
  p <- stats::predict(fit, x)
  expect_equal(dim(p), c(60L, 2L))
  expect_true(all(p > 0 & p < 1))
  expect_gt(tss(y[, 1], p[, 1]), 0.4)
  expect_equal(fit$model$columns, colnames(.flatten(x)))
  path <- tempfile(fileext = ".rds")
  on.exit(unlink(path), add = TRUE)
  saveRDS(fit, path)
  expect_identical(stats::predict(readRDS(path), x), p)
  other <- grain_matrix(sim$readings, plot, t, temp, grain = "month")
  expect_error(stats::predict(fit, other), "different channels or bins")
  second <- stats::predict(fit_learner(boosting(method = "xgboost", trees = 30L), x, y), x)
  expect_gt(tss(y[, 1], second[, 1]), 0.4)
  cv <- fit_learner(boosting(trees = 60L, n_inner = 3L), x, y)$model$models[[1L]]
  expect_length(cv$cv_error, 60L)
  expect_identical(length(cv$offset) - 1L, which.min(cv$cv_error))
})

test_that("boosting fits the family the response head's loss names, and a constant is its mean", {
  local_response("continuous_test", list(
    prepare = function(y) .as_response(y), activation = "identity",
    loss = "squared_error", metric = "roc_auc",
    cells = function(y, folds) scorable_cells(y > stats::median(y), folds)))
  sim <- sim_series(n_unit = 60L, days = 60L, seed = 91L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "week")
  level <- 10 + 3 * scale(rowMeans(x[, , 1L]))[, 1L]
  y <- matrix(level, ncol = 1L, dimnames = list(dimnames(x)[[1L]], "height"))
  fit <- fit_learner(boosting(depth = 2L, min_leaf = 3), x, y, response = "continuous_test")
  expect_equal(fit$model$family, "gaussian")
  expect_gt(stats::cor(stats::predict(fit, x)[, 1L], y[, 1L]), 0.8)

  flat <- matrix(0, nrow = 60L, ncol = 1L, dimnames = list(dimnames(x)[[1L]], "absent"))
  expect_equal(unique(stats::predict(fit_learner(boosting(), x, flat), x)[, 1L]), 0)
})
