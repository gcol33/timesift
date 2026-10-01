tune_data <- function(n_unit = 40L, seed = 81L) {
  set.seed(seed)
  t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24L * 90L)
  units <- sprintf("p%02d", seq_len(n_unit))
  warmth <- stats::rnorm(n_unit)
  readings <- data.frame(
    plot = rep(units, each = length(t)), t = rep(t, times = n_unit),
    temp = as.numeric(vapply(warmth, function(w) w + stats::rnorm(length(t), sd = 0.5),
                             numeric(length(t)))))
  x <- grain_matrix(readings, plot, t, temp, grain = "month")
  s <- rowMeans(matrix(x[, , 1L], nrow = n_unit))
  y <- cbind(sp1 = as.integer(s > stats::median(s)), sp2 = as.integer(s < stats::median(s)))
  rownames(y) <- units
  list(x = x, y = y, readings = readings, units = units, score = s)
}

# Predicts the ranking of the unit means exactly at k = 3 and buries it in a fixed jitter that
# grows with the distance from 3, so the setting that scores best is known.
k_learner <- function() {
  learner(
    "kl", multi = "joint", params = list(k = 1, other = "kept"),
    fit = function(x, y, k, other, ...) list(k = k, other = other),
    predict = function(model, x) {
      s <- rowMeans(matrix(x[, , 1L], nrow = dim(x)[1L]))
      noise <- sin(seq_len(nrow(x)) * 12.9898) * abs(model$k - 3) * 10
      cbind(s + noise, -(s + noise))
    })
}

test_that("the setting that scores best on the inner folds is the one fitted", {
  d <- tune_data()
  tuned <- tune(k_learner(), list(k = 1:5), inner = 3L)
  expect_s3_class(tuned, "timesift_learner")
  expect_identical(tuned$name, "kl")
  fit <- fit_learner(tuned, d$x, d$y)
  expect_s3_class(fit$model, "timesift_tuned")
  expect_equal(fit$model$chosen$k, 3L)
  tbl <- fit$model$table
  expect_equal(nrow(tbl), 5L)
  expect_equal(tbl$score[tbl$settings == "k = 3"], 1)
  expect_true(all(tbl$score[tbl$settings != "k = 3"] < 1))
  # What it fitted is the base learner at the chosen setting, and the other settings stay as given.
  expect_equal(fit$model$model$k, 3L)
  expect_identical(fit$model$model$other, "kept")
  expect_equal(unname(stats::predict(fit, d$x)[, 1L]), d$score)
})

test_that("a grid is every combination, and a setting given to the fit overrides the carried one", {
  d <- tune_data()
  tuned <- tune(k_learner(), list(k = c(3, 9), other = c("a", "b")), inner = 3L)
  fit <- fit_learner(tuned, d$x, d$y)
  expect_equal(fit$model$table$settings,
               c("k = 3, other = a", "k = 9, other = a", "k = 3, other = b", "k = 9, other = b"))
  expect_equal(fit$model$chosen$k, 3)
  # Ties go to the first combination in the grid.
  expect_identical(fit$model$chosen$other, "a")
  given <- fit_learner(tune(k_learner(), list(k = 3), inner = 3L), d$x, d$y, other = "given")
  expect_identical(given$model$model$other, "given")
})

test_that("the inner folds keep a grouping whole, and a metric of its own is used", {
  d <- tune_data()
  group <- rep(sprintf("g%02d", 1:10), each = 4L)
  fit <- fit_learner(tune(k_learner(), list(k = 1:5), inner = 3L), d$x, d$y, group = group)
  expect_equal(fit$model$chosen$k, 3L)
  by_tss <- fit_learner(tune(k_learner(), list(k = 1:5), metric = "tss", inner = 3L), d$x, d$y)
  expect_equal(by_tss$model$chosen$k, 3L)
  expect_equal(by_tss$model$table$score[3L], 1)
  by_function <- fit_learner(
    tune(k_learner(), list(k = 1:5), metric = function(y, p) -abs(mean(p) - 0.2), inner = 3L),
    d$x, d$y)
  expect_true(by_function$model$chosen$k %in% 1:5)
})

test_that("a shipped learner is tuned the same way", {
  d <- tune_data()
  fit <- fit_learner(tune(forest(), list(trees = c(5, 15)), inner = 3L), d$x, d$y)
  expect_true(fit$model$chosen$trees %in% c(5, 15))
  expect_equal(nrow(fit$model$table), 2L)
  expect_equal(dim(stats::predict(fit, d$x)), dim(d$y))
})

test_that("a run records what each tuned candidate chose", {
  d <- tune_data()
  targets <- data.frame(plot = d$units, d$y, stringsAsFactors = FALSE)
  run <- suppressWarnings(timesift(
    targets, d$readings, y = c("sp1", "sp2"), id = plot, time = t, x = temp,
    models = list(tuned = tune(k_learner(), list(k = 1:5), inner = 3L), plain = k_learner()),
    sift = grains("month"), resampling = cv(v = 3L), inner = NULL, ensemble = FALSE,
    verbose = FALSE))
  expect_true("settings" %in% names(run$candidates))
  expect_identical(run$candidates$settings[run$candidates$learner == "tuned"], "k = 3")
  expect_true(is.na(run$candidates$settings[run$candidates$learner == "plain"]))
})

test_that("a grid says what it cannot search", {
  expect_error(tune(k_learner(), list(nope = 1:2)), "carries no setting called nope")
  expect_error(tune(k_learner(), list(k = integer())), "no value for k")
  expect_error(tune(k_learner(), 1:3), "named list")
  expect_error(tune(k_learner(), list()), "named list")
  expect_error(tune(k_learner(), list(k = 1:2), inner = 1L), "2 or more")
  expect_error(tune("nope", list(k = 1)), "nope")
})
