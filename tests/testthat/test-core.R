test_that("the core reproduces the pure-R oracle on every grain and statistic", {
  set.seed(20260903)
  starts <- c("2019-09-01", "2020-02-17 05:00:00", "2021-06-11 13:00:00")
  schemes <- list(c("min", "mean", "max"),
                  c("mean_daily_min", "mean", "mean_daily_max"),
                  c("cold_day", "mean", "warm_day"))

  for (start in starts) {
    t <- seq(as.POSIXct(start, tz = "UTC"), by = "hour", length.out = 24 * 400)
    d <- data.frame(id = rep(c("p1", "p2", "p3"), each = length(t)),
                    t = rep(t, 3),
                    v = rnorm(3 * length(t), sd = 5))
    for (w in c("native", "halfday", "day", "week", "month", "season", "year")) {
      for (scheme in schemes) {
        if (w %in% c("native", "halfday") &&
              any(scheme %in% c("cold_day", "warm_day", "mean_daily_min", "mean_daily_max"))) {
          next
        }
        x <- grain_matrix(d, id, t, v, grain = w, stats = scheme)
        o <- oracle_grain_matrix(d, "id", "t", "v", grain = w, stats = scheme)
        expect_identical(as.vector(unclass(x)), as.vector(o$values),
                         info = paste(start, w, paste(scheme, collapse = "+")))
        expect_identical(dimnames(x), dimnames(o$values))
        expect_identical(as.vector(attr(x, "bin_n")), as.vector(o$bin_n))
        expect_identical(attr(x, "bin_partial"), o$bin_partial)
        expect_equal(as.numeric(attr(x, "bin_start")), as.numeric(o$bin_start))
        expect_equal(as.numeric(attr(x, "bin_end")), as.numeric(o$bin_end))
      }
    }
  }
})

test_that("the seven statistics keep the two orderings the definitions imply, in the core and the oracle", {
  set.seed(20260908)
  t <- seq(as.POSIXct("2020-02-17 05:00:00", tz = "UTC"), by = "hour", length.out = 24 * 400)
  d <- data.frame(id = rep(c("p1", "p2", "p3"), each = length(t)), t = rep(t, 3),
                  v = rnorm(3 * length(t), sd = 5))
  stats <- c("min", "mean_daily_min", "cold_day", "mean", "warm_day", "mean_daily_max", "max")
  ordered <- function(m, chain) {
    for (i in seq_len(length(chain) - 1L)) {
      expect_true(all(m[, , chain[i]] <= m[, , chain[i + 1L]] + 1e-9),
                  info = paste(attr(m, "grain"), chain[i], "<=", chain[i + 1L]))
    }
  }
  for (w in c("day", "week", "month", "season", "year")) {
    x <- grain_matrix(d, id, t, v, grain = w, stats = stats)
    o <- oracle_grain_matrix(d, "id", "t", "v", grain = w, stats = stats)$values
    attr(o, "grain") <- w
    for (m in list(x, o)) {
      ordered(m, c("min", "mean_daily_min", "mean", "mean_daily_max", "max"))
      ordered(m, c("min", "cold_day", "mean", "warm_day", "max"))
    }
  }
})

test_that("the core reproduces the oracle at anniversaries other than the default", {
  set.seed(11)
  t <- seq(as.POSIXct("2019-01-01", tz = "UTC"), by = "hour", length.out = 24 * 500)
  d <- data.frame(id = rep(c("a", "b"), each = length(t)), t = rep(t, 2),
                  v = rnorm(2 * length(t)))
  for (ys in c("01-01", "03-01", "09-01", "12-28")) {
    for (w in c("season", "year")) {
      x <- grain_matrix(d, id, t, v, grain = w, stats = c("cold_day", "mean", "warm_day"),
                         year_start = ys)
      o <- oracle_grain_matrix(d, "id", "t", "v", grain = w,
                                stats = c("cold_day", "mean", "warm_day"), year_start = ys)
      expect_identical(as.vector(unclass(x)), as.vector(o$values), info = paste(w, ys))
      expect_identical(dimnames(x)[[2]], dimnames(o$values)[[2]])
    }
  }
})

test_that("the core reproduces the oracle under a supplied calendar", {
  set.seed(12)
  t <- seq(as.POSIXct("2019-09-01", tz = "UTC"), by = "hour", length.out = 24 * 400)
  d <- data.frame(id = rep(c("a", "b"), each = length(t)), t = rep(t, 2),
                  v = rnorm(2 * length(t)))
  ten_days <- function(when) {
    .POSIXct(floor(as.numeric(when) / (10 * 86400)) * 10 * 86400, tz = "UTC")
  }
  x <- grain_matrix(d, id, t, v, grain = ten_days, stats = c("cold_day", "mean", "warm_day"))
  o <- oracle_grain_matrix(d, "id", "t", "v", grain = ten_days,
                            stats = c("cold_day", "mean", "warm_day"))
  expect_identical(as.vector(unclass(x)), as.vector(o$values))
  expect_identical(attr(x, "bin_partial"), o$bin_partial)
})

test_that("the core's calendar agrees with the oracle's, instant by instant", {
  t <- seq(as.POSIXct("2018-01-01", tz = "UTC"), by = "97 min", length.out = 20000)
  ys <- list(month = 9L, day = 1L)
  for (w in c("native", "halfday", "day", "week", "month", "season", "year")) {
    expect_equal(ts_bin_starts_(as.numeric(t), w, 9L, 1L),
                 as.numeric(oracle_bin_start(t, w, ys, "UTC")), info = w)
  }
  bins <- unique(oracle_bin_start(t, "month", ys, "UTC"))
  expect_equal(ts_bin_nexts_(as.numeric(bins), "month", 9L, 1L),
               as.numeric(oracle_bin_next(bins, "month", ys, "UTC", max(as.numeric(t)))))
})

test_that("a gap the whole record shares is an error, not four adjacent bins", {
  set.seed(3)
  t <- seq(as.POSIXct("2021-12-01", tz = "UTC"), by = "hour", length.out = 24 * 150)
  d <- data.frame(id = rep(c("a", "b"), each = length(t)), t = rep(t, 2),
                  v = rnorm(2 * length(t)))

  gap <- d[format(d$t, "%Y-%m") != "2022-02", ]
  expect_error(grain_matrix(gap, id, t, v, grain = "month", stats = "mean"),
               "month bins are not contiguous")
  expect_error(grain_matrix(gap, id, t, v, grain = "day", stats = "mean"),
               "day bins are not contiguous")

  # The record's own ends are not a gap: a partial bin at either end is reported, not rejected.
  expect_silent(grain_matrix(d, id, t, v, grain = "month", stats = "mean"))

  # At the `native` grain the bin is the reading itself, so the calendar cannot say what a bin
  # between two others would have been and none is asserted.
  sparse <- d[as.integer(format(d$t, "%H")) %% 3L == 0L, ]
  expect_silent(grain_matrix(sparse, id, t, v, grain = "native", stats = "mean"))
})

test_that("a day-level statistic needs bins of a day or coarser, whoever supplied them", {
  set.seed(4)
  t <- seq(as.POSIXct("2021-12-01", tz = "UTC"), by = "hour", length.out = 24 * 60)
  d <- data.frame(id = rep(c("a", "b"), each = length(t)), t = rep(t, 2),
                  v = rnorm(2 * length(t)))
  six <- function(when) .POSIXct(floor(as.numeric(when) / 21600) * 21600, tz = "UTC")

  expect_error(grain_matrix(d, id, t, v, grain = six, stats = c("cold_day", "warm_day")),
               "need bins of a calendar day or coarser")
  expect_error(grain_matrix(d, id, t, v, grain = six, stats = "mean_daily_min"),
               "mean_daily_min needs bins of a calendar day or coarser")
  expect_silent(grain_matrix(d, id, t, v, grain = six, stats = c("min", "mean", "max")))

  # A calendar that cuts on the day boundary is fine however unusual its bin lengths are.
  ten <- function(when) .POSIXct(floor(as.numeric(when) / (10 * 86400)) * 10 * 86400, tz = "UTC")
  expect_silent(grain_matrix(d, id, t, v, grain = ten, stats = c("cold_day", "warm_day")))
})

test_that("a zone whose local midnight does not exist bins without an NA", {
  set.seed(5)
  t <- seq(as.POSIXct("2018-11-01 12:00:00", tz = "America/Sao_Paulo"), by = "hour",
           length.out = 24 * 8)
  d <- data.frame(id = rep(c("a", "b"), each = length(t)), t = rep(t, 2),
                  v = rnorm(2 * length(t)))

  for (w in c("native", "halfday", "day", "week", "month")) {
    x <- grain_matrix(d, id, t, v, grain = w, stats = "mean")
    expect_false(anyNA(x), info = w)
    expect_false(anyNA(attr(x, "bin_start")), info = w)
  }

  # 4 November 2018 is 23 hours long in Sao Paulo, and the day it opens is the instant the clock
  # jumped to rather than a midnight that never happened.
  x <- grain_matrix(d, id, t, v, grain = "day", stats = "mean")
  n <- attr(x, "bin_n")["a", ]
  expect_identical(unname(n[format(attr(x, "bin_start"), "%Y-%m-%d") == "2018-11-04"]), 23L)
  expect_identical(format(attr(x, "bin_start"), "%H:%M:%S"),
                   c("00:00:00", "00:00:00", "00:00:00", "01:00:00", "00:00:00",
                     "00:00:00", "00:00:00", "00:00:00", "00:00:00"))

  # A year boundary landing on that date is an argument, not an error.
  expect_silent(grain_matrix(d, id, t, v, grain = "year", year_start = "11-04", stats = "mean"))
})

test_that("a series carried in a zone bins by that zone's calendar", {
  set.seed(6)
  t <- seq(as.POSIXct("2021-12-20", tz = "UTC"), by = "hour", length.out = 24 * 40)
  v <- rnorm(2 * length(t))
  d <- data.frame(id = rep(c("a", "b"), each = length(t)), t = rep(t, 2), v = v)
  utc <- grain_matrix(d, id, t, v, grain = "day", stats = c("min", "mean", "max"))

  attr(t, "tzone") <- "Europe/Vienna"
  d$t <- rep(t, 2)
  vienna <- grain_matrix(d, id, t, v, grain = "day", stats = c("min", "mean", "max"))

  expect_identical(dim(utc)[2L], 40L)
  expect_identical(dim(vienna)[2L], 41L)
  expect_identical(dimnames(vienna)[[2]][1], "2021-12-19T23:00:00Z")

  # The same instants, relabelled into their Vienna clock and binned as if that clock were UTC,
  # give the same numbers: the zone is the whole of the difference.
  relabelled <- data.frame(
    id = d$id,
    t = as.POSIXct(format(d$t, "%Y-%m-%d %H:%M:%S", tz = "Europe/Vienna"), tz = "UTC"),
    v = d$v)
  naive <- grain_matrix(relabelled, id, t, v, grain = "day", stats = c("min", "mean", "max"))
  expect_identical(as.vector(unclass(vienna)), as.vector(unclass(naive)))
  expect_identical(as.vector(attr(vienna, "bin_n")), as.vector(attr(naive, "bin_n")))
})

test_that("readings a fraction of a second apart are the same reading twice", {
  t <- as.POSIXct(c(0, 0.25, 1, 2), origin = "1970-01-01", tz = "UTC")
  d <- data.frame(id = "a", t = t, v = c(1, 2, 3, 4))
  expect_error(grain_matrix(d, id, t, v, grain = "native", stats = "mean"),
               "duplicated \\(unit, time\\) pair")
})

test_that("a reading that is not a finite number is refused, and named", {
  t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24 * 8)
  d <- data.frame(id = rep(c("p1", "p2"), each = length(t)), t = rep(t, 2),
                  v = rnorm(2 * length(t)))
  at <- data.frame(id = c("p1", "p2"), at = rep(max(t), 2))

  for (hole in list(NA_real_, NaN, Inf, -Inf)) {
    bad <- d
    bad$v[30L] <- hole
    expect_error(grain_matrix(bad, id, t, v, grain = "day"),
                 "1 reading is not a finite number, first: unit p1 at 2021-09-02T05:00:00")
    expect_error(lookback_matrix(bad, id, t, v, at = at, span = "2 days"),
                 "1 reading is not a finite number, first: unit p1 at 2021-09-02T05:00:00")
  }

  two <- d
  two$v[c(30L, 31L)] <- Inf
  expect_error(grain_matrix(two, id, t, v, grain = "day"), "2 readings are not a finite number")

  # coverage() reads how many readings a unit has in each bin and never their values, so it is
  # not the guard's business.
  bad <- d
  bad$v[30L] <- NA_real_
  expect_identical(sum(coverage(bad, id, t, grain = "day")), nrow(d))
})
