projection_case <- function(n_row = 4L, n_col = 5L, days = 120L, hole = NULL) {
  skip_if_not_installed("terra")
  set.seed(91L)
  n <- n_row * n_col
  r <- terra::rast(nrows = n_row, ncols = n_col, xmin = 0, xmax = n_col, ymin = 0, ymax = n_row,
                   crs = "EPSG:4326")
  when <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "day", length.out = days)
  warmth <- stats::rnorm(n)
  temp <- terra::rast(r, nlyrs = days)
  readings <- outer(warmth, rep(1, days)) + matrix(stats::rnorm(n * days, sd = 0.5), n)
  elev <- stats::rnorm(n)
  if (!is.null(hole)) {
    readings[hole, ] <- NA
    elev[hole] <- NA
  }
  terra::values(temp) <- readings
  terra::time(temp) <- when
  static <- terra::setValues(r, elev)
  names(static) <- "elev"
  cells <- seq_len(n)
  usable <- if (is.null(hole)) cells else cells[-hole]
  targets <- data.frame(cell = usable, elev = elev[usable],
                        sp1 = stats::rbinom(length(usable), 1L, stats::plogis(warmth[usable])),
                        sp2 = stats::rbinom(length(usable), 1L, stats::plogis(-warmth[usable])))
  series <- data.frame(cell = rep(usable, each = days), t = rep(when, times = length(usable)),
                       temp = as.numeric(t(readings[usable, , drop = FALSE])))
  list(raster = temp, static = static, targets = targets, series = series, n = n, usable = usable)
}

mean_reader <- function() {
  learner(
    "mean_reader", multi = "joint",
    fit = function(x, y, ...) list(),
    predict = function(model, x) {
      s <- rowMeans(matrix(x[, , "mean"], nrow = dim(x)[1L]))
      cbind(sp1 = 1 / (1 + exp(-s)), sp2 = 1 / (1 + exp(s)))
    })
}

projection_fit <- function(case) {
  suppressWarnings(timesift(
    case$targets, case$series, y = c("sp1", "sp2"), id = cell, time = t, x = temp,
    learners = list(rd = mean_reader()), sift = grains("month"), resampling = cv(v = 3L),
    n_inner = NULL, ensemble = FALSE, verbose = FALSE))
}

test_that("a projection is the fit applied to one target per cell", {
  skip_on_cran()
  case <- projection_case()
  fit <- projection_fit(case)
  map <- project(fit, series = case$raster, candidate = "rd / month")
  expect_s4_class(map, "SpatRaster")
  expect_equal(names(map), c("sp1", "sp2"))
  expect_equal(terra::ncell(map), case$n)
  expect_true(terra::compareGeom(map, case$raster, stopOnError = FALSE))
  expected <- stats::predict(fit, case$targets, case$series, candidate = "rd / month")
  expect_equal(unname(terra::values(map)[, "sp1"]), unname(expected[as.character(1:case$n), "sp1"]))
  # In chunks of a few cells it is the same map.
  chunked <- project(fit, series = case$raster, candidate = "rd / month", chunk = 3L)
  expect_equal(terra::values(chunked), terra::values(map))
})

test_that("a cell with a missing input is missing in every layer", {
  case <- projection_case(hole = c(2L, 11L))
  fit <- projection_fit(case)
  map <- project(fit, series = case$raster, candidate = "rd / month")
  values <- terra::values(map)
  expect_true(all(is.na(values[c(2L, 11L), ])))
  expect_false(anyNA(values[-c(2L, 11L), ]))
})

test_that("binary and spread maps follow predict", {
  case <- projection_case()
  fit <- projection_fit(case)
  binary <- project(fit, series = case$raster, candidate = "rd / month", type = "binary")
  want <- stats::predict(fit, case$targets, case$series, candidate = "rd / month", type = "binary")
  expect_equal(unname(terra::values(binary)[, "sp2"]), unname(want[as.character(1:case$n), "sp2"]))
  expect_true(all(terra::values(binary) %in% c(0, 1)))

  two <- suppressWarnings(timesift(
    case$targets, case$series, y = c("sp1", "sp2"), id = cell, time = t, x = temp,
    learners = list(a = mean_reader(), b = mean_reader()), sift = grains("month"),
    resampling = cv(v = 3L), n_inner = NULL, ensemble = ensemble("mean"), verbose = FALSE))
  spread <- project(two, series = case$raster, type = "spread")
  expect_true(all(c("sp1.mean", "sp1.sd", "sp2.upper") %in% names(spread)))
  ensemble_map <- project(two, series = case$raster)
  expect_equal(terra::values(spread)[, "sp1.mean"], terra::values(ensemble_map)[, "sp1"])
})

test_that("static predictors are read from the layers of a raster", {
  case <- projection_case()
  fit <- suppressWarnings(timesift(
    case$targets, y = c("sp1", "sp2"), id = cell, static = elev, learners = elasticnet(),
    ensemble = FALSE, n_inner = NULL, resampling = cv(v = 3L), verbose = FALSE))
  map <- project(fit, static = case$static, candidate = "elasticnet / static")
  want <- stats::predict(fit, case$targets, candidate = "elasticnet / static")
  expect_equal(unname(terra::values(map)[, "sp1"]), unname(want[as.character(1:case$n), "sp1"]))
})

test_that("a projection says what it was not given", {
  case <- projection_case()
  fit <- projection_fit(case)
  expect_error(project(fit), "so `series` is needed")
  expect_error(project(fit, series = list(nope = case$raster)), "list of rasters named")
  expect_error(project(fit, series = case$raster, chunk = 0), "1 or more")
  expect_error(project(1), "expected a timesift")
  static_fit <- suppressWarnings(timesift(
    case$targets, y = c("sp1", "sp2"), id = cell, static = elev, learners = elasticnet(),
    ensemble = FALSE, n_inner = NULL, resampling = cv(v = 3L), verbose = FALSE))
  expect_error(project(static_fit), "SpatRaster with layers named")
  other <- case$static
  names(other) <- "other"
  expect_error(project(static_fit, static = other), "layers named")
})

test_that("the range change counts the cells lost, kept and gained", {
  now <- cbind(sp1 = c(1, 1, 1, 0, 0, NA), sp2 = c(0, 0, 1, 1, 1, 1))
  later <- cbind(sp1 = c(1, 0, 0, 1, 1, 1), sp2 = c(0, 0, 1, 1, 0, 0))
  rc <- range_change(now, later)
  expect_s3_class(rc, "timesift_range_change")
  sp1 <- rc$table[rc$table$variable == "sp1", ]
  # sp1 over the five cells both maps hold: kept, lost, lost, gained, gained.
  expect_equal(c(sp1$lost, sp1$kept, sp1$gained, sp1$absent), c(2, 1, 2, 0))
  expect_equal(sp1$current, 3)
  expect_equal(sp1$later, 3)
  expect_equal(sp1$percent_loss, 100 * 2 / 3)
  expect_equal(sp1$percent_gain, 100 * 2 / 3)
  expect_equal(sp1$change, 0)
  sp2 <- rc$table[rc$table$variable == "sp2", ]
  expect_equal(c(sp2$lost, sp2$kept, sp2$gained, sp2$absent), c(2, 2, 0, 2))
  expect_equal(sp2$change, -50)
  expect_equal(unname(rc$map[, "sp1"]), c(-1, -2, -2, 1, 1, NA))
  expect_equal(unname(rc$map[, "sp2"]), c(0, 0, -1, -1, -2, -2))
  expect_output(print(rc), "range change")
})

test_that("a map that is not 0 and 1 is cut at a threshold", {
  now <- cbind(sp = c(0.1, 0.6, 0.9, 0.4))
  later <- cbind(sp = c(0.7, 0.2, 0.8, 0.3))
  expect_error(range_change(now, later), "give `threshold`")
  rc <- range_change(now, later, threshold = 0.5)
  expect_equal(unname(rc$map[, "sp"]), c(1, -2, -1, 0))
  expect_error(range_change(now, later[1:3, , drop = FALSE], threshold = 0.5), "same cells")
  expect_error(range_change(now, later, threshold = c(0.5, 0.5)), "one cut")
})

test_that("the range change of two projections is a raster of codes", {
  case <- projection_case()
  fit <- projection_fit(case)
  now <- project(fit, series = case$raster, candidate = "rd / month", type = "binary")
  warmer <- case$raster + 2
  terra::time(warmer) <- terra::time(case$raster)
  later <- project(fit, series = warmer, candidate = "rd / month", type = "binary")
  rc <- range_change(now, later)
  expect_s4_class(rc$map, "SpatRaster")
  expect_equal(names(rc$map), c("sp1", "sp2"))
  expect_true(all(rc$table$gained >= 0))
  grDevices::pdf(NULL)
  on.exit(grDevices::dev.off(), add = TRUE)
  expect_no_error(plot(rc, "sp1"))
})
