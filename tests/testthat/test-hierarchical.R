tulpa_run <- function(spatial, ...) {
  set.seed(2)
  n <- 60L
  targets <- data.frame(plot = sprintf("p%03d", seq_len(n)), lon = stats::runif(n),
                        lat = stats::runif(n))
  targets$s1 <- stats::rbinom(n, 1L, stats::plogis(-0.3 + 2 * sin(5 * targets$lon)))
  targets$s2 <- stats::rbinom(n, 1L, stats::plogis(1.5 * (targets$lat - 0.5)))
  days <- seq(as.POSIXct("2020-01-01", tz = "UTC"), by = "12 hours", length.out = 240L)
  series <- do.call(rbind, lapply(targets$plot, function(p) {
    data.frame(plot = p, t = days, v = stats::rnorm(length(days)))
  }))
  list(targets = targets, series = series,
       fit = timesift(targets, series, y = dplyr::starts_with("s"), id = plot, time = t, ...,
                      models = hierarchical(spatial = spatial), sift = grains("month"),
                      resampling = cv(v = 3L), inner = NULL, ensemble = FALSE, verbose = FALSE))
}

test_that("a hierarchical learner declares tulpa and its settings", {
  l <- hierarchical(spatial = "hsgp")
  expect_equal(l$needs, "tulpa")
  expect_equal(l$params$spatial, "hsgp")
  expect_error(hierarchical(spatial = "gp"), "should be one of")
  expect_error(hierarchical(inference = 1), "one tulpa mode")
})

test_that("coords names two numeric columns and reaches the array, split with its units", {
  expect_error(.check_coords(data.frame(a = 1, b = 2, c = 3), c("a", "b", "c")), "two columns")
  expect_error(.check_coords(data.frame(a = 1, b = "x"), c("a", "b")), "must be numeric")
  expect_error(.check_coords(data.frame(a = 1, b = NA_real_), c("a", "b")), "must be numeric")
  x <- feature_matrix(matrix(seq_len(8), 4L, 2L, dimnames = list(letters[1:4], c("u", "v"))))
  attr(x, "coords") <- matrix(1:8, 4L, 2L, dimnames = list(letters[1:4], c("lon", "lat")))
  sub <- .subset_units(x, c(3L, 1L))
  expect_equal(attr(sub, "coords"), attr(x, "coords")[c(3L, 1L), ])
})

test_that("a field needs the coordinates, and a fit without one does not", {
  skip_if_not_installed("tulpa")
  expect_error(tulpa_run("hsgp"), "coords = ")
  run <- tulpa_run("none")
  p <- stats::predict(run$fit, run$targets[1:5, ], run$series, candidate = "selected")
  expect_equal(dim(p), c(5L, 2L))
  expect_true(all(p > 0 & p < 1))
})

test_that("a field predicts from the coordinates of the new units, and survives saveRDS", {
  skip_if_not_installed("tulpa")
  run <- tulpa_run("hsgp", coords = c(lon, lat))
  new <- run$targets[1:6, ]
  p <- stats::predict(run$fit, new, run$series, candidate = "selected")
  expect_equal(dim(p), c(6L, 2L))
  moved <- new
  moved$lon <- 1 - moved$lon
  expect_false(isTRUE(all.equal(p, stats::predict(run$fit, moved, run$series,
                                                  candidate = "selected"))))
  path <- tempfile(fileext = ".rds")
  saveRDS(run$fit, path)
  expect_equal(stats::predict(readRDS(path), new, run$series, candidate = "selected"), p)
})

test_that("a random intercept needs the unit, and a held-out unit is predicted at the population level", {
  skip_if_not_installed("tulpa")
  expect_error(hierarchical(random = NA), "TRUE or FALSE")
  set.seed(3)
  n <- 40L
  targets <- data.frame(plot = rep(sprintf("p%02d", 1:20), each = 2L), when = rep(0:1, 20L))
  targets$t0 <- as.POSIXct("2020-06-01", tz = "UTC") + targets$when * 86400
  shift <- stats::rnorm(20L)[rep(1:20, each = 2L)]
  targets$s1 <- stats::rbinom(n, 1L, stats::plogis(shift))
  series <- do.call(rbind, lapply(unique(targets$plot), function(p) {
    data.frame(plot = p, t = as.POSIXct("2020-01-01", tz = "UTC") + 3600 * (0:3000),
               v = stats::rnorm(3001L))
  }))
  fit <- timesift(targets, series, y = s1, id = plot, time = t, target_time = t0,
                  models = hierarchical(random = TRUE), sift = lookbacks("30 days"),
                  resampling = grouped_cv("plot", v = 2L), inner = NULL, ensemble = FALSE, verbose = FALSE)
  p <- stats::predict(fit, targets, series, candidate = "selected")
  expect_equal(dim(p), c(n, 1L))
  expect_true(all(p > 0 & p < 1))
  none <- timesift(targets, series, y = s1, id = plot, time = t, target_time = t0,
                   models = hierarchical(), sift = lookbacks("30 days"),
                   resampling = grouped_cv("plot", v = 2L), inner = NULL, ensemble = FALSE, verbose = FALSE)
  expect_false(isTRUE(all.equal(p, stats::predict(none, targets, series, candidate = "selected"))))
})
