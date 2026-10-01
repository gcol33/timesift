curve_reader <- function(name, scale = 1) {
  learner(
    name = name, multi = "joint",
    fit = function(x, y, ...) list(),
    predict = function(model, x) {
      s <- scale * rowMeans(matrix(x[, , "warm_day"], nrow = dim(x)[1L]))
      cbind(s, -s)
    })
}

curve_run <- function(models, ensemble = FALSE) {
  set.seed(71L)
  t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24L * 120L)
  units <- sprintf("p%02d", 1:40)
  warmth <- stats::rnorm(40)
  readings <- data.frame(
    plot = rep(units, each = length(t)), t = rep(t, times = 40L),
    temp = as.numeric(vapply(warmth, function(w) w + sin(seq_along(t) / 300) +
                               stats::rnorm(length(t)), numeric(length(t)))))
  targets <- data.frame(plot = units, sp1 = stats::rbinom(40, 1L, stats::plogis(warmth)),
                        sp2 = stats::rbinom(40, 1L, stats::plogis(-warmth)),
                        stringsAsFactors = FALSE)
  suppressWarnings(timesift(
    targets, readings, y = c("sp1", "sp2"), id = plot, time = t, x = temp, models = models,
    sift = grains("month", stats = c("cold_day", "mean", "warm_day")),
    resampling = cv(v = 3L), inner = NULL, ensemble = ensemble, verbose = FALSE))
}

test_that("a curve in a channel is the prediction as that statistic moves in every bin", {
  run <- curve_run(list(w = curve_reader("w")))
  m <- run$representations[["month"]]
  rc <- response_curve(run, "w / month", "warm_day", n = 7L)
  expect_s3_class(rc, "timesift_response_curve")
  expect_named(rc, c("value", "variable", "prediction"))
  expect_equal(nrow(rc), 7L * 2L)
  warm <- as.numeric(m[, , "warm_day"])
  expect_equal(range(rc$value), range(warm))
  # The model reads the mean of the warmest day over the bins, so with the statistic set to one
  # value in every bin its prediction is that value.
  sp1 <- rc[rc$variable == "sp1", ]
  expect_equal(sp1$prediction, sp1$value)
  expect_equal(rc$prediction[rc$variable == "sp2"], -sp1$value)
  # A statistic the model never reads leaves its prediction where it was.
  flat <- response_curve(run, "w / month", "cold_day", n = 5L)
  expect_equal(length(unique(round(flat$prediction[flat$variable == "sp1"], 12))), 1L)
})

test_that("a curve in one cell moves one bin and holds the others at the reference", {
  run <- curve_run(list(w = curve_reader("w")))
  m <- run$representations[["month"]]
  bin <- dimnames(m)[[2L]][2L]
  rc <- response_curve(run, "w / month", list(bin = bin, channel = "warm_day"), n = 4L)
  reference <- apply(m[, , "warm_day"], 2L, mean)
  sp1 <- rc[rc$variable == "sp1", ]
  expect_equal(sp1$prediction, (sum(reference[-2L]) + sp1$value) / length(reference))
  # The same reference at the median, the minimum and the maximum.
  for (f in c("median", "min", "max")) {
    ref <- apply(m[, , "warm_day"], 2L, switch(f, median = stats::median, min = min, max = max))
    got <- response_curve(run, "w / month", list(bin = bin, channel = "warm_day"), fixed = f,
                          n = 4L)
    s1 <- got[got$variable == "sp1", ]
    expect_equal(s1$prediction, (sum(ref[-2L]) + s1$value) / length(ref), info = f)
  }
})

test_that("two predictors are read over the grid of the two", {
  run <- curve_run(list(w = curve_reader("w")))
  surface <- response_curve(run, "w / month", "warm_day", with = "cold_day", n = 5L)
  expect_named(surface, c("value", "value_with", "variable", "prediction"))
  expect_equal(nrow(surface), 5L * 5L * 2L)
  sp1 <- surface[surface$variable == "sp1", ]
  expect_equal(sp1$prediction, sp1$value)
  expect_equal(length(unique(sp1$value_with)), 5L)
})

test_that("the ensemble is moved through every member and combined by the stack", {
  run <- curve_run(list(a = curve_reader("a"), b = curve_reader("b", 2)),
                   ensemble = ensemble("mean"))
  rc <- response_curve(run, "ensemble", "warm_day", n = 6L, spread = TRUE)
  sp1 <- rc[rc$variable == "sp1", ]
  expect_equal(sp1$prediction, 1.5 * sp1$value)
  # Two members at equal weight, predicting v and 2 v: the standard deviation is |v| / sqrt(2).
  expect_equal(sp1$sd, abs(sp1$value) / sqrt(2))
  # The interval is held inside zero and one under a probability head.
  expect_true(all(sp1$lower <= sp1$upper))
})

test_that("a curve says what it cannot draw", {
  run <- curve_run(list(w = curve_reader("w")))
  expect_error(response_curve(run, "w / month"), "names what to vary")
  expect_error(response_curve(run, "w / month", "nope"), "neither a channel nor a bin")
  expect_error(response_curve(run, "w / month", "warm_day", n = 1L), "2 or more")
  expect_error(response_curve(run, "ensemble", "warm_day"), "no ensemble")
  expect_error(response_curve(run, "w / month", "warm_day", spread = TRUE), "members")
  expect_error(response_curve(run, "x / month", "warm_day"), "no candidate called")
  expect_error(response_curve(1, "a"), "expected a timesift")
  expect_output(print(response_curve(run, "w / month", "warm_day", n = 3L)),
                "response curve")
})
