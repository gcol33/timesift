# The envelope on the core, against biomod2's own `bm_SRE()`, and the learner above it.
#
# The reference is biomod2's, on the weekly columns maxnet's fixtures read: the bounds, pinned to
# rounding, and the projection onto the fixture's rows and onto every reading scaled by 1.02,
# asserted exactly. The Python suite reads the same files.

envelope_fixture <- function() {
  dir <- fixture_dir()
  if (is.null(dir)) skip("the fixtures ship with the package and are not installed here")
  d <- utils::read.csv(file.path(dir, "penalised_input.csv"), stringsAsFactors = FALSE,
                       check.names = FALSE)
  held <- c("unit", "y_gaussian", "y_binomial", "w", "fold")
  x <- as.matrix(d[, setdiff(names(d), held), drop = FALSE])
  x <- x[, !endsWith(colnames(x), "^2"), drop = FALSE]
  r <- utils::read.csv(file.path(dir, "maxnet_response.csv"), stringsAsFactors = FALSE)
  r <- r[match(d$unit, r$unit), ]
  list(x = x, y = list(y_binomial = d$y_binomial, y_8 = r$y_8, y_12 = r$y_12),
       cases = utils::read.csv(file.path(dir, "envelope_cases.csv"), stringsAsFactors = FALSE),
       bounds = utils::read.csv(file.path(dir, "envelope_bounds.csv"), stringsAsFactors = FALSE),
       predict = utils::read.csv(file.path(dir, "envelope_predict.csv"),
                                 stringsAsFactors = FALSE))
}

test_that("the core draws the envelope biomod2's bm_SRE draws", {
  fx <- envelope_fixture()
  for (i in seq_len(nrow(fx$cases))) {
    row <- fx$cases[i, ]
    y <- fx$y[[row$response]]
    fit <- .envelope_fit(fx$x, y, row$quantile)
    ref <- fx$bounds[fx$bounds$case == row$case, ]
    expect_equal(fit$n_presence, row$n_presence, info = row$case)
    expect_equal(fit$lo, ref$lo, tolerance = row$bound_tolerance, info = row$case)
    expect_equal(fit$hi, ref$hi, tolerance = row$bound_tolerance, info = row$case)
    pred <- fx$predict[fx$predict$case == row$case, ]
    expect_identical(.envelope_predict(fit, fx$x), as.numeric(pred$inside), info = row$case)
    expect_identical(.envelope_predict(fit, fx$x * 1.02), as.numeric(pred$inside_out),
                     info = row$case)
  }
})

test_that("a quantile of zero keeps every presence, ends included, and one half the medians", {
  x <- matrix(c(1, 2, 3, 4, 5, 10, 20, 30, 40, 50), ncol = 2L)
  y <- c(0, 1, 1, 1, 0)
  fit <- .envelope_fit(x, y, 0)
  expect_equal(fit$lo, c(2, 20))
  expect_equal(fit$hi, c(4, 40))
  expect_identical(.envelope_predict(fit, x), c(0, 1, 1, 1, 0))
  mid <- .envelope_fit(x, y, 0.5)
  expect_equal(mid$lo, mid$hi)
  expect_identical(.envelope_predict(mid, x), c(0, 0, 1, 0, 0))
  # Type 7 interpolates between neighbours: over 2, 3, 4 the 0.25 quantile is 2.5.
  expect_equal(.envelope_fit(x, y, 0.25)$lo, c(2.5, 25))
})

test_that("the envelope refuses what it has no envelope for", {
  x <- matrix(1:4, ncol = 1L)
  expect_error(.envelope_fit(x, c(0, 0, 0, 0), 0.025), "at least one presence")
  expect_error(.envelope_fit(x, c(0, 1, 2, 0), 0.025), "zero and one")
  expect_error(.envelope_fit(x, c(0, 1, 1, 0), 0.6), "\\[0, 0.5\\]")
  fit <- .envelope_fit(x, c(0, 1, 1, 0), 0.025)
  expect_error(.envelope_predict(fit, cbind(x, x)), "drawn over 1 columns")
  expect_error(envelope(quantile = 0.7), "\\[0, 0.5\\]")
  expect_error(envelope(quantile = c(0.1, 0.2)), "one number")
})

test_that("the envelope fits, predicts zero or one, ignores the weights and round trips", {
  sim <- sim_series(n_unit = 60L, days = 120L, seed = 35L)
  y <- sim_response(sim, n_var = 2L, seed = 36L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "month")
  fit <- fit_learner(envelope(quantile = 0.05), x, y)
  p <- stats::predict(fit, x)
  expect_equal(dim(p), c(60L, 2L))
  expect_true(all(p %in% c(0, 1)))
  # Every presence inside the band of every column is predicted present.
  expect_gt(mean(p[y[, 1] == 1, 1]), mean(p[y[, 1] == 0, 1]))
  head <- .responses_reg$get("presence_absence")
  local_response("unweighted_test", head[setdiff(names(head), "weights")])
  plain <- stats::predict(fit_learner(envelope(quantile = 0.05), x, y, response = "unweighted_test"),
                          x)
  expect_identical(plain, p)
  path <- tempfile(fileext = ".rds")
  saveRDS(fit, path)
  expect_identical(stats::predict(readRDS(path), x), p)
  unlink(path)
})

test_that("the envelope needs a presence-absence head, and a response of one value is its share", {
  local_response("continuous_test", list(
    prepare = function(y) .as_response(y), activation = "identity",
    loss = "squared_error", metric = "roc_auc",
    cells = function(y, folds) scorable_cells(y > stats::median(y), folds)))
  sim <- sim_series(n_unit = 40L, days = 60L, seed = 91L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "month")
  level <- matrix(rowMeans(x[, , 1L]), ncol = 1L, dimnames = list(dimnames(x)[[1L]], "height"))
  expect_error(fit_learner(envelope(), x, level, response = "continuous_test"), "presences")
  none <- matrix(0, 40L, 1L, dimnames = list(dimnames(x)[[1L]], "absent"))
  fit <- fit_learner(envelope(), x, none)
  expect_identical(fit$model$unfitted, "absent")
  expect_equal(unique(stats::predict(fit, x)[, 1L]), 0)
})
