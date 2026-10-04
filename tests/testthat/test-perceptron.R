# The one-hidden-layer network on the core, against the nnet package, and the learner above it.
#
# The reference is nnet's own fit on the fixture design, written by `inst/spec/make_fixtures.R`
# from starting weights written beside it: the objective it ends at, whether it stopped at its
# iteration cap, its weights and its predictions on the design scaled by 1.01. The core is started
# from the same weights and asserted against all four. The Python suite reads the same files.

perceptron_fixture <- function() {
  dir <- fixture_dir()
  if (is.null(dir)) skip("the fixtures ship with the package and are not installed here")
  d <- utils::read.csv(file.path(dir, "penalised_input.csv"), stringsAsFactors = FALSE,
                       check.names = FALSE)
  held <- c("unit", "y_gaussian", "y_binomial", "w", "fold")
  x <- as.matrix(d[, setdiff(names(d), held), drop = FALSE])
  x <- unname(x[, !endsWith(colnames(x), "^2"), drop = FALSE])
  list(x = x, w = d$w, y = list(y_binomial = d$y_binomial, y_gaussian = d$y_gaussian),
       cases = utils::read.csv(file.path(dir, "perceptron_cases.csv"), stringsAsFactors = FALSE),
       weights = utils::read.csv(file.path(dir, "perceptron_weights.csv"),
                                 stringsAsFactors = FALSE),
       predict = utils::read.csv(file.path(dir, "perceptron_predict.csv"),
                                 stringsAsFactors = FALSE))
}

perceptron_case_fit <- function(fx, row) {
  y <- fx$y[[row$response]]
  w <- if (isTRUE(row$weighted)) fx$w else rep(1, length(y))
  family <- if (row$response == "y_gaussian") "gaussian" else "binomial"
  start <- fx$weights$start[fx$weights$case == row$case]
  .perceptron_fit(fx$x, y, w, family, hidden = row$hidden, decay = row$decay,
                  max_iter = row$max_iter, skip = isTRUE(row$skip), start = start)
}

test_that("the core ends where nnet ends from the same starting weights", {
  fx <- perceptron_fixture()
  for (i in seq_len(nrow(fx$cases))) {
    row <- fx$cases[i, ]
    f <- perceptron_case_fit(fx, row)
    wtol <- row$weight_tolerance
    ptol <- row$prediction_tolerance
    expect_equal(f$value, row$value, tolerance = ptol, info = row$case)
    expect_identical(!f$converged, row$stopped, info = row$case)
    expect_equal(f$weights, fx$weights$fitted[fx$weights$case == row$case], tolerance = wtol,
                 info = row$case)
    expect_equal(.perceptron_predict(f, fx$x * 1.01),
                 fx$predict$fitted_out[fx$predict$case == row$case], tolerance = ptol,
                 info = row$case)
  }
})

test_that("the gradient is the objective's, under every family", {
  set.seed(3)
  n <- 30L
  p <- 3L
  x <- matrix(rnorm(n * p), n)
  w <- runif(n, 0.5, 2)
  ys <- list(binomial = rbinom(n, 1, 0.4), gaussian = rnorm(n), poisson = rpois(n, 2))
  for (family in names(ys)) {
    for (skip in c(FALSE, TRUE)) {
      # A fit of no iterations returns the objective at the starting weights, so central
      # differences of it are the gradient the minimiser is handed.
      start <- runif(2L * (p + 1L) + 3L + if (skip) p else 0L, -0.5, 0.5)
      at <- function(v) {
        .perceptron_fit(x, ys[[family]], w, family, decay = 0.05, max_iter = 0L, skip = skip,
                        start = v)$value
      }
      h <- 1e-6
      numeric_g <- vapply(seq_along(start), function(a) {
        up <- start
        down <- start
        up[a] <- up[a] + h
        down[a] <- down[a] - h
        (at(up) - at(down)) / (2 * h)
      }, numeric(1L))
      # One iteration from the start moves along minus the gradient by the step the line search
      # accepts, so its first step's direction is the analytic gradient's.
      one <- .perceptron_fit(x, ys[[family]], w, family, decay = 0.05, max_iter = 2L,
                             skip = skip, start = start)
      step <- start - one$weights
      expect_equal(step / sqrt(sum(step^2)), numeric_g / sqrt(sum(numeric_g^2)),
                   tolerance = 1e-5, info = paste(family, skip))
    }
  }
})

test_that("a fit round trips through saveRDS and predicts the same", {
  fx <- perceptron_fixture()
  f <- .perceptron_fit(fx$x, fx$y$y_binomial, fx$w, "binomial", decay = 0.1)
  path <- tempfile(fileext = ".rds")
  saveRDS(f, path)
  expect_identical(.perceptron_predict(readRDS(path), fx$x), .perceptron_predict(f, fx$x))
})

test_that("the drawn starting weights follow the seed and stay inside the range", {
  fx <- perceptron_fixture()
  a <- .perceptron_fit(fx$x, fx$y$y_binomial, fx$w, "binomial", range = 0.3, max_iter = 0L,
                       seed = 7L)
  b <- .perceptron_fit(fx$x, fx$y$y_binomial, fx$w, "binomial", range = 0.3, max_iter = 0L,
                       seed = 7L)
  c <- .perceptron_fit(fx$x, fx$y$y_binomial, fx$w, "binomial", range = 0.3, max_iter = 0L,
                       seed = 8L)
  expect_identical(a$weights, b$weights)
  expect_false(identical(a$weights, c$weights))
  expect_true(all(abs(a$weights) <= 0.3))
})

test_that("the presets give biomod2's settings and an explicit setting beats them", {
  expect_identical(perceptron()$params[c("hidden", "decay", "range", "max_iter")],
                   list(hidden = 2L, decay = 0, range = 0.7, max_iter = 100L))
  expect_identical(perceptron(preset = "bigboss")$params[c("hidden", "decay", "range",
                                                           "max_iter")],
                   list(hidden = 5L, decay = 0.1, range = 0.1, max_iter = 200L))
  expect_identical(perceptron(preset = "bigboss", hidden = 3L)$params$hidden, 3L)
  expect_error(perceptron(hidden = 0L), "hidden")
  expect_error(perceptron(decay = -1), "decay")
  expect_error(perceptron(skip = NA), "skip")
})

test_that("perceptron fits under every head, round trips, and a constant is its mean", {
  sim <- sim_series(n_unit = 60L, days = 120L, seed = 43L)
  y <- sim_response(sim, n_var = 2L, seed = 44L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "month")
  for (l in list(perceptron(), perceptron(preset = "bigboss"), perceptron(skip = TRUE))) {
    fit <- fit_learner(l, x, y)
    p <- stats::predict(fit, x)
    expect_equal(dim(p), c(60L, 2L))
    expect_true(all(p >= 0 & p <= 1))
    path <- tempfile(fileext = ".rds")
    saveRDS(fit, path)
    expect_identical(stats::predict(readRDS(path), x), p)
    unlink(path)
  }
  flat <- matrix(0, nrow = 60L, ncol = 1L, dimnames = list(dimnames(x)[[1L]], "absent"))
  expect_equal(unique(stats::predict(fit_learner(perceptron(), x, flat), x)[, 1L]), 0)

  local_response("continuous_test", list(
    prepare = function(y) .as_response(y), activation = "identity",
    loss = "squared_error", metric = "roc_auc",
    cells = function(y, folds) scorable_cells(y > stats::median(y), folds)))
  level <- 10 + 3 * scale(rowMeans(x[, , 1L]))[, 1L]
  yc <- matrix(level, ncol = 1L, dimnames = list(dimnames(x)[[1L]], "height"))
  fit <- fit_learner(perceptron(decay = 0.01), x, yc, response = "continuous_test")
  expect_equal(fit$model$family, "gaussian")
  expect_gt(stats::cor(stats::predict(fit, x)[, 1L], yc[, 1L]), 0.8)
})

test_that("a network too large for its inverse Hessian is refused with its size", {
  sim <- sim_series(n_unit = 20L, days = 60L, seed = 5L)
  y <- sim_response(sim, n_var = 1L, seed = 6L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "native")
  expect_error(fit_learner(perceptron(hidden = 10L, max_hessian = 0.001), x, y),
               "approximate inverse Hessian")
})

test_that("networks fitted on several threads are the networks each response gets alone", {
  fx <- perceptron_fixture()
  y <- cbind(fx$y$y_binomial, 1 - fx$y$y_binomial, rev(fx$y$y_binomial))
  w <- cbind(fx$w, rep(1, nrow(y)), fx$w)
  seeds <- c(3L, 11L, 29L)
  many <- .perceptron_fits(fx$x, y, w, "binomial", seeds = seeds, decay = 0.05, threads = 3L)
  for (j in seq_len(ncol(y))) {
    one <- .perceptron_fit(fx$x, y[, j], w[, j], "binomial", decay = 0.05, seed = seeds[j])
    expect_identical(many[[j]], one)
  }
  expect_error(perceptron(threads = 0L), "threads")
})

test_that("a standardised network is the network on the standardised columns", {
  fx <- perceptron_fixture()
  a <- .perceptron_fit(fx$x, fx$y$y_binomial, fx$w, "binomial", decay = 0.05,
                       standardise = TRUE)
  expect_equal(a$centre, unname(colMeans(fx$x)), tolerance = 1e-12)
  expect_equal(a$scale, unname(apply(fx$x, 2L, stats::sd)), tolerance = 1e-12)
  scaled <- function(m) sweep(sweep(m, 2L, a$centre), 2L, a$scale, "/")
  b <- .perceptron_fit(scaled(fx$x), fx$y$y_binomial, fx$w, "binomial", decay = 0.05)
  expect_identical(a$weights, b$weights)
  expect_identical(.perceptron_predict(a, fx$x * 1.01),
                   .perceptron_predict(b, scaled(fx$x * 1.01)))
  expect_length(.perceptron_fit(fx$x, fx$y$y_binomial, fx$w, "binomial")$centre, 0L)
  expect_error(perceptron(standardise = NA), "standardise")
})
