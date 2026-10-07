count_data <- function(n = 80L, seed = 5L) {
  set.seed(seed)
  t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24L * 90L)
  units <- sprintf("p%02d", seq_len(n))
  warmth <- stats::rnorm(n)
  readings <- data.frame(
    plot = rep(units, each = length(t)), t = rep(t, times = n),
    temp = as.numeric(vapply(warmth, function(w) w + stats::rnorm(length(t), sd = 0.5),
                             numeric(length(t)))))
  x <- grain_matrix(readings, plot, t, temp, grain = "week")
  rate <- exp(0.8 + 0.7 * warmth)
  count <- stats::rpois(n, rate)
  y <- cbind(count = count)
  rownames(y) <- units
  list(x = x, readings = readings, units = units, rate = rate, count = count, y = y)
}

poisson_deviance_of <- function(y, p) {
  mean(2 * (ifelse(y > 0, y * log(y / p), 0) - (y - p)))
}

test_that("the count head and its metric are registered", {
  expect_true("count" %in% responses())
  expect_true("neg_poisson_deviance" %in% metrics())
  h <- .responses_reg$get("count")
  expect_identical(h$loss, "poisson_deviance")
  expect_identical(h$activation, "exp")
  expect_identical(h$metric, "neg_poisson_deviance")
})

test_that("the Poisson deviance reads counts and refuses a mean outside its support", {
  y <- c(0, 1, 2, 5, 3, 0)
  p <- c(0.5, 1.2, 1.8, 4.2, 3.5, 0.2)
  expect_equal(regression_metric(y, p, "poisson_deviance"), poisson_deviance_of(y, p))
  expect_equal(regression_metric(y, y + 0.5 * (y == 0), "poisson_deviance"),
               poisson_deviance_of(y, y + 0.5 * (y == 0)))
  expect_equal(regression_metric(c(1, 2, 3), c(1, 2, 3), "poisson_deviance"), 0)
  # A zero mean beside a zero count is the saturated cell and scores nothing against it.
  expect_equal(regression_metric(c(0, 2), c(0, 2), "poisson_deviance"), 0)
  expect_equal(regression_metric(c(0, 0, 2, 1), c(0, 0.4, 1.5, 1.5), "poisson_deviance"),
               poisson_deviance_of(c(0, 0, 2, 1), c(1e-300, 0.4, 1.5, 1.5)), tolerance = 1e-6)
  expect_true(is.na(regression_metric(c(0, 0, 2, 1), c(0, 0.4, 0, 1.5), "poisson_deviance")))
  expect_true(is.na(regression_metric(c(1, 2), c(1, -1), "poisson_deviance")))
  expect_equal(timesift:::.as_metric("neg_poisson_deviance")$fn(y, p), -poisson_deviance_of(y, p))
})

test_that("the count head refuses a response it cannot hold", {
  prepare <- .responses_reg$get("count")$prepare
  expect_error(prepare(cbind(a = c(1, -2, 3))), "whole numbers of zero or more")
  expect_error(prepare(cbind(a = c(1, 2.5, 3))), "whole numbers of zero or more")
  expect_error(prepare(cbind(a = c(1, NA, 3))), "missing")
  expect_equal(unname(prepare(cbind(a = c(0, 1, 7)))), unname(cbind(a = c(0, 1, 7))))
})

test_that("each learner that fits a family fits a count under the Poisson one", {
  skip_on_cran()
  d <- count_data()
  for (l in list(elasticnet(), forest(trees = 30L), boosting(trees = 30L),
                 boosting(method = "xgboost", trees = 30L, depth = 2L), tree(), linear(), mars(),
                 additive(k = 5L))) {
    fit <- fit_learner(l, d$x, d$y, response = "count")
    p <- stats::predict(fit, d$x)[, 1L]
    expect_true(all(p > 0), info = l$name)
    expect_gt(stats::cor(p, d$rate), 0.5, label = l$name)
    expect_lt(poisson_deviance_of(d$count, p), poisson_deviance_of(d$count, mean(d$count)),
              label = l$name)
  }
  expect_identical(fit_learner(elasticnet(), d$x, d$y, response = "count")$model$models[[1L]]$family,
                   "poisson")
  expect_identical(fit_learner(boosting(trees = 10L), d$x, d$y, response = "count")$model$family,
                   "poisson")
})

test_that("the learners that need presences and absences refuse a count head", {
  d <- count_data(n = 40L)
  for (l in list(discriminant(), envelope(), maxent())) {
    expect_error(fit_learner(l, d$x, d$y, response = "count"), "poisson_deviance", info = l$name)
  }
})

test_that("a Poisson shrinkage reaches the tree only through a count head", {
  d <- count_data(n = 60L)
  tight <- fit_learner(tree(shrink = 0), d$x, d$y, response = "count")
  loose <- fit_learner(tree(shrink = 1), d$x, d$y, response = "count")
  expect_false(isTRUE(all.equal(stats::predict(tight, d$x), stats::predict(loose, d$x))))
  expect_error(tree(shrink = -1), "zero or more")
  expect_error(tree(shrink = c(1, 2)), "one number")
})

test_that("a run under the count head is scored by the Poisson deviance and stacked", {
  skip_on_cran()
  d <- count_data()
  targets <- data.frame(plot = d$units, count = d$count, stringsAsFactors = FALSE)
  run <- suppressWarnings(timesift(
    targets, d$readings, y = count, id = plot, time = t, x = temp, response = "count",
    learners = list(en = elasticnet(squares = FALSE), rf = forest(trees = 30L)),
    sift = grains("week"), resampling = cv(v = 4L), n_inner = NULL,
    ensemble = ensemble("stack"), verbose = FALSE))
  expect_identical(run$metric, "neg_poisson_deviance")
  expect_true(all(is.finite(run$scores$score[run$scores$scorable])))
  expect_true(all(run$scores$score[run$scores$scorable] < 0))
  expect_identical(run$stack$loss, "poisson_deviance")
  expect_equal(sum(run$stack$weights), 1)
  p <- stats::predict(run, targets, d$readings)
  expect_true(all(p > 0))
  expect_gt(stats::cor(p[, 1L], d$rate), 0.5)
})

test_that("the combiner minimises the Poisson deviance of the combined mean", {
  loss <- .stack_loss("count")
  expect_identical(loss$name, "poisson_deviance")
  y <- c(0, 1, 2, 5, 3, 0, 4, 2)
  p <- c(0.4, 1.3, 1.9, 4.1, 3.6, 0.3, 3.2, 2.4)
  expect_equal(loss$value(p, y), poisson_deviance_of(y, p))
  # The gradient is the derivative of the value in the combined mean, cell by cell.
  step <- 1e-6
  numeric <- vapply(seq_along(p), function(i) {
    up <- p
    up[i] <- up[i] + step
    down <- p
    down[i] <- down[i] - step
    (loss$value(up, y) - loss$value(down, y)) / (2 * step)
  }, numeric(1L))
  expect_equal(loss$gradient(p, y), numeric, tolerance = 1e-6)
  # A mean at zero is read at the machine epsilon, so the value stays finite.
  expect_true(is.finite(loss$value(c(0, p[-1L]), y)))
  # The solver reaches the weights a direct search over the simplex reaches.
  set.seed(8)
  P <- cbind(a = p * exp(stats::rnorm(8, sd = 0.1)), b = p * exp(stats::rnorm(8, sd = 0.4)),
             c = rep(mean(y), 8L))
  w <- .simplex_weights(P, y, loss)$weights
  grid <- as.matrix(expand.grid(a = seq(0, 1, by = 0.01), b = seq(0, 1, by = 0.01)))
  grid <- cbind(grid, c = 1 - rowSums(grid))
  grid <- grid[grid[, "c"] >= -1e-12, , drop = FALSE]
  best <- min(apply(grid, 1L, function(g) loss$value(as.numeric(P %*% g), y)))
  expect_lte(loss$value(as.numeric(P %*% w), y), best + 1e-9)
})

test_that("the encoders train under the Poisson deviance and predict a positive mean", {
  skip_on_cran()
  skip_if_not_installed("torch")
  skip_if_not(torch::torch_is_installed(), "the torch runtime is not installed")
  d <- count_data(n = 60L)
  fit <- fit_learner(mlp(epochs = 60L, learning_rate = 0.01, seed = 3L, val_frac = 0), d$x, d$y,
                     response = "count")
  p <- stats::predict(fit, d$x)[, 1L]
  expect_identical(fit$model$activation, "exp")
  expect_true(all(p > 0))
  expect_gt(stats::cor(p, d$rate), 0.5)
  expect_lt(poisson_deviance_of(d$count, p), poisson_deviance_of(d$count, mean(d$count)))
})
