numeric_data <- function(n = 50L, seed = 5L) {
  set.seed(seed)
  t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24L * 90L)
  units <- sprintf("p%02d", seq_len(n))
  warmth <- stats::rnorm(n)
  readings <- data.frame(
    plot = rep(units, each = length(t)), t = rep(t, times = n),
    temp = as.numeric(vapply(warmth, function(w) w + stats::rnorm(length(t), sd = 0.5),
                             numeric(length(t)))))
  x <- grain_matrix(readings, plot, t, temp, grain = "week")
  height <- 10 + 3 * warmth + stats::rnorm(n, sd = 0.3)
  y <- cbind(height = height)
  rownames(y) <- units
  list(x = x, readings = readings, units = units, warmth = warmth, height = height, y = y)
}

test_that("the heads and metrics of a numeric response are registered", {
  expect_true(all(c("continuous", "abundance", "ordinal") %in% responses()))
  expect_true(all(c("r_squared", "pearson", "neg_rmse", "neg_mse", "neg_mae", "neg_max_error",
                    "ordinal_accuracy", "ordinal_recall", "ordinal_precision", "ordinal_f1")
                  %in% metrics()))
  for (h in c("continuous", "abundance")) {
    expect_identical(.responses_reg$get(h)$metric, "r_squared")
  }
  expect_identical(.responses_reg$get("ordinal")$metric, "ordinal_f1")
  for (h in c("continuous", "abundance", "ordinal")) {
    expect_identical(.responses_reg$get(h)$loss, "squared_error")
    expect_identical(.responses_reg$get(h)$activation, "identity")
  }
})

test_that("each statistic of a numeric response reads the errors", {
  y <- c(1, 2, 3, 4)
  p <- c(1.5, 2.5, 3.5, 4.5)
  expect_equal(regression_metric(y, p, "rmse"), 0.5)
  expect_equal(regression_metric(y, p, "mse"), 0.25)
  expect_equal(regression_metric(y, p, "mae"), 0.5)
  expect_equal(regression_metric(y, p, "max_error"), 0.5)
  expect_equal(regression_metric(y, p, "pearson"), 1)
  # The sum of squares about the mean is 5, and the squared errors sum to 1.
  expect_equal(regression_metric(y, p, "r_squared"), 1 - 1 / 5)
  expect_true(is.na(regression_metric(c(3, 3, 3), c(1, 2, 3), "r_squared")))
  expect_true(is.na(regression_metric(y, c(2, 2, 2, 2), "pearson")))
  expect_true(is.na(regression_metric(y, c(1, 2, NA, 4), "rmse")))
  expect_error(regression_metric(y, p[-1L], "rmse"), "same length")
  # The errors are registered with their sign reversed, so a higher score is a better one.
  expect_equal(timesift:::.as_metric("neg_rmse")$fn(y, p), -0.5)
  expect_equal(timesift:::.as_metric("r_squared")$fn(y, p), 0.8)
})

test_that("ordinal classes are read as the observed class nearest to the prediction", {
  y <- c(1, 1, 2, 2, 3, 3, 3)
  p <- c(1.2, 2.4, 2, 1.4, 2.8, 3.4, 2.2)
  # Observed 1, 1, 2, 2, 3, 3, 3 against read 1, 2, 2, 1, 3, 3, 2: four hits of seven, and the
  # classes are recalled 1/2, 1/2 and 2/3 and predicted rightly 1/2, 1/3 and 1.
  expect_equal(ordinal_metric(y, p, "accuracy"), 4 / 7)
  expect_equal(ordinal_metric(y, p, "recall"), (1 / 2 + 1 / 2 + 2 / 3) / 3)
  expect_equal(ordinal_metric(y, p, "precision"), (1 / 2 + 1 / 3 + 1) / 3)
  r <- (1 / 2 + 1 / 2 + 2 / 3) / 3
  q <- (1 / 2 + 1 / 3 + 1) / 3
  expect_equal(ordinal_metric(y, p, "f1"), 2 * q * r / (q + r))
  # A prediction halfway between two classes is read as the lower one.
  expect_equal(ordinal_metric(c(1, 2, 3, 3), c(1.5, 2.5, 3.5, 2.5), "accuracy"), 3 / 4)
  expect_equal(timesift:::.as_metric("ordinal_accuracy")$fn(y, p), 4 / 7)
  expect_error(ordinal_metric(y, p, "nope"), "should be one of")
})

test_that("a numeric cell is scorable where both sides of the split hold two values", {
  y <- cbind(a = c(1, 2, 3, 4, 5, 6), b = c(1, 1, 1, 1, 2, 3))
  rownames(y) <- sprintf("p%d", 1:6)
  folds <- stats::setNames(c(1L, 1L, 2L, 2L, 3L, 3L), rownames(y))
  cells <- timesift:::.numeric_cells(y, folds)
  expect_s3_class(cells, "timesift_cells")
  expect_equal(cells$variable, rep(c("a", "b"), each = 3L))
  expect_equal(cells$scorable, c(TRUE, TRUE, TRUE, FALSE, FALSE, FALSE))
  # Folds 1 and 2 of `b` each hold the value 1 twice, and fold 3 is fitted on four units of one
  # value: none of the three can be read.
  expect_equal(cells$abs_test[cells$variable == "b"], c(1L, 1L, 2L))
  expect_equal(cells$abs_train[cells$variable == "b"], c(3L, 3L, 1L))
})

test_that("a head refuses a response it cannot hold", {
  y <- cbind(a = c(1, 2, 3, 4))
  expect_error(.responses_reg$get("abundance")$prepare(-y), "not negative")
  expect_error(.responses_reg$get("ordinal")$prepare(y + 0.5), "whole-number")
  expect_error(.responses_reg$get("continuous")$prepare(cbind(a = c(1, NA, 3))), "missing")
  expect_equal(unname(.responses_reg$get("ordinal")$prepare(y)), unname(y))
})

test_that("a run under a numeric head is scored by that head's metric and stacked", {
  d <- numeric_data()
  targets <- data.frame(plot = d$units, height = d$height, stringsAsFactors = FALSE)
  run <- suppressWarnings(timesift(
    targets, d$readings, y = height, id = plot, time = t, x = temp, response = "continuous",
    learners = list(en = elasticnet(squares = FALSE), rf = forest(trees = 30L)),
    sift = grains("week"), resampling = cv(v = 4L), n_inner = NULL,
    ensemble = ensemble("stack"), verbose = FALSE))
  expect_identical(run$metric, "r_squared")
  expect_true(all(is.finite(run$scores$score[run$scores$scorable])))
  best <- max(run$scores$score, na.rm = TRUE)
  expect_gt(best, 0.7)
  p <- stats::predict(run, targets, d$readings)
  expect_gt(stats::cor(p[, 1L], d$height), 0.9)
  expect_error(stats::predict(run, targets, d$readings, type = "binary"), "presence-absence")
  expect_true(all(.as_folds(run$folds, d$units) %in% 1:4))
})

test_that("the learners that need presences and absences refuse a numeric head", {
  d <- numeric_data()
  for (l in list(discriminant(), envelope(), maxent())) {
    expect_error(fit_learner(l, d$x, d$y, response = "continuous"), "squared_error",
                 info = l$name)
  }
  for (l in list(elasticnet(), forest(trees = 20L), boosting(trees = 20L), tree(), linear(),
                 mars())) {
    fit <- fit_learner(l, d$x, d$y, response = "continuous")
    expect_gt(stats::cor(stats::predict(fit, d$x)[, 1L], d$height), 0.8, label = l$name)
  }
})
