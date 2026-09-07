# A record in which the units differ from each other in one stretch of the calendar and nowhere
# else, so the profile has a known answer to be checked against.
planted_series <- function(planted_month = "2021-11", n_unit = 60L, seed = 61L) {
  set.seed(seed)
  t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24L * 210L)
  in_month <- format(t, "%Y-%m", tz = "UTC") == planted_month
  units <- sprintf("p%03d", seq_len(n_unit))
  warmth <- stats::rnorm(n_unit)
  shape <- 5 * sin(seq_along(t) / (24 * 40))
  value <- as.numeric(vapply(warmth, function(w) {
    shape + w * in_month * 6 + stats::rnorm(length(t), sd = 0.3)
  }, numeric(length(t))))
  list(units = units, warmth = warmth, planted = planted_month,
       readings = data.frame(plot = rep(units, each = length(t)), t = rep(t, times = n_unit),
                             temp = value, stringsAsFactors = FALSE))
}

test_that("occlusion needs the fits the ladder was told to keep", {
  sim <- sim_series(n_unit = 30L, days = 60L)
  y <- sim_response(sim, n_var = 2L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "month")
  lad <- suppressWarnings(grain_ladder(x, y, elasticnet(), folds = fold_map(y, v = 3L),
                                        verbose = FALSE))
  expect_error(occlusion(lad, x, y, "month|elasticnet"), "kept no fits")
})

test_that("the bin a signal was planted in is the bin the profile weights", {
  skip_if_not_installed("glmnet")
  sim <- planted_series()
  y <- matrix(stats::rbinom(length(sim$warmth) * 2L, 1L,
                            stats::plogis(3 * c(sim$warmth, -sim$warmth))),
              ncol = 2L, dimnames = list(sim$units, c("sp1", "sp2")))
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "month")
  lad <- suppressWarnings(grain_ladder(x, y, elasticnet(),
                                        folds = fold_map(y, v = 4L, seed = 6L),
                                        keep_fits = TRUE, verbose = FALSE))
  oc <- occlusion(lad, x, y, "month|elasticnet", permutations = 5L, seed = 4L)

  weight <- stats::aggregate(list(weight = oc$weight), oc["part"], mean)
  heaviest <- weight$part[which.max(weight$weight)]
  expect_equal(substr(heaviest, 1L, 7L), sim$planted)
  expect_gt(max(weight$weight), 2 * stats::median(weight$weight))
})

test_that("every substitute runs and reports the same parts", {
  skip_if_not_installed("glmnet")
  sim <- planted_series(n_unit = 40L, seed = 62L)
  y <- matrix(stats::rbinom(length(sim$warmth) * 2L, 1L,
                            stats::plogis(3 * c(sim$warmth, -sim$warmth))),
              ncol = 2L, dimnames = list(sim$units, c("sp1", "sp2")))
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "month",
                     stats = c("cold_day", "mean", "warm_day"))
  lad <- suppressWarnings(grain_ladder(x, y, elasticnet(),
                                        folds = fold_map(y, v = 3L, seed = 6L),
                                        keep_fits = TRUE, verbose = FALSE))
  parts <- lapply(c("permute", "fold_mean", "unit_mean"), function(s)
    occlusion(lad, x, y, "month|elasticnet", substitute = s, permutations = 3L))
  expect_true(all(vapply(parts, function(p) identical(p$part, parts[[1L]]$part), logical(1L))))
  expect_true(all(vapply(parts, function(p) all(is.finite(p$weight)), logical(1L))))
})

test_that("the profile is read by the metric the fit was scored under", {
  skip_if_not_installed("glmnet")
  sim <- planted_series(n_unit = 40L, seed = 64L)
  y <- matrix(stats::rbinom(length(sim$warmth) * 2L, 1L,
                            stats::plogis(3 * c(sim$warmth, -sim$warmth))),
              ncol = 2L, dimnames = list(sim$units, c("sp1", "sp2")))
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "month")
  folds <- fold_map(y, v = 3L, seed = 6L)
  lad <- suppressWarnings(grain_ladder(x, y, elasticnet(), folds = folds, metric = "tss",
                                        keep_fits = TRUE, verbose = FALSE))

  # The ladder was scored by TSS, so a weight is a fall in TSS unless another metric is named.
  under_tss <- occlusion(lad, x, y, "month|elasticnet", permutations = 3L, seed = 4L)
  expect_identical(attr(under_tss, "metric"), "tss")
  named <- occlusion(lad, x, y, "month|elasticnet", metric = "tss", permutations = 3L, seed = 4L)
  expect_equal(under_tss$weight, named$weight)

  under_auc <- occlusion(lad, x, y, "month|elasticnet", metric = "roc_auc",
                         permutations = 3L, seed = 4L)
  expect_identical(attr(under_auc, "metric"), "roc_auc")
  expect_false(isTRUE(all.equal(under_tss$weight, under_auc$weight)))

  # A run reaches the same profile through its own door, under its own metric.
  run <- suppressWarnings(timesift(
    data.frame(plot = sim$units, y, stringsAsFactors = FALSE), sim$readings,
    y = c("sp1", "sp2"), id = plot, time = t, x = temp, models = elasticnet(),
    sift = grains("month"), resampling = folds, metric = "tss", ensemble = FALSE,
    keep_fits = TRUE, verbose = FALSE))
  through_run <- occlusion(run, "elasticnet / month", permutations = 3L, seed = 4L)
  expect_identical(attr(through_run, "metric"), "tss")
  expect_equal(through_run$weight, under_tss$weight)
})

test_that("holding a channel back asks what the statistic carries", {
  skip_if_not_installed("glmnet")
  sim <- planted_series(n_unit = 40L, seed = 63L)
  y <- matrix(stats::rbinom(length(sim$warmth) * 2L, 1L,
                            stats::plogis(3 * c(sim$warmth, -sim$warmth))),
              ncol = 2L, dimnames = list(sim$units, c("sp1", "sp2")))
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "month",
                     stats = c("cold_day", "mean", "warm_day"))
  lad <- suppressWarnings(grain_ladder(x, y, elasticnet(),
                                        folds = fold_map(y, v = 3L, seed = 6L),
                                        keep_fits = TRUE, verbose = FALSE))
  oc <- occlusion(lad, x, y, "month|elasticnet", over = "channel", permutations = 3L)
  expect_setequal(unique(oc$part), c("cold_day", "mean", "warm_day"))
})
