fixture_series <- function(dir, name) {
  file <- switch(name, aligned = "series.csv", offset = "series_offset.csv",
                 zoned = "series_zoned.csv", order = "series_order.csv")
  s <- read.csv(file.path(dir, file), stringsAsFactors = FALSE)
  s$time <- as.POSIXct(s$time, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  s
}

fixture_binning <- function(dir, name, grain) {
  if (grain != "astronomical") {
    return(grain)
  }
  edges <- read.csv(file.path(dir, "seasons.csv"), stringsAsFactors = FALSE)
  edges <- as.POSIXct(edges$edge[edges$series == name], format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  function(when) edges[findInterval(as.numeric(when), as.numeric(edges))]
}

# The two calendars the guard fixture names, built from the name so that both suites build the
# same function rather than each writing one that happens to break the same rule.
fixture_calendar <- function(name) {
  switch(name,
    late = function(when) {
      as.POSIXct((floor(as.numeric(when) / 86400) + 1) * 86400, origin = "1970-01-01", tz = "UTC")
    },
    alternate = function(when) {
      t0 <- min(as.numeric(when))
      as.POSIXct(t0 + 3600 * (((as.numeric(when) - t0) / 3600) %% 2), origin = "1970-01-01",
                 tz = "UTC")
    },
    stop("no calendar called ", name))
}

test_that("every representation matches the digest the Python side reads", {
  dir <- fixture_dir()
  skip_if(is.null(dir), "fixtures are not in the built package")

  expected <- read.csv(file.path(dir, "digests.csv"), stringsAsFactors = FALSE)
  series <- lapply(stats::setNames(nm = unique(expected$series)), fixture_series, dir = dir)

  for (i in seq_len(nrow(expected))) {
    row <- expected[i, ]
    label <- paste(row$series, row$grain, row$tz, row$year_start, row$partial, row$stat)
    # The instants are the same bytes on disk whichever calendar reads them; the zone is the clock
    # laid over them, and a zone row asserts that both languages read that clock the same way.
    record <- series[[row$series]]
    attr(record$time, "tzone") <- row$tz
    x <- grain_matrix(record, id, time, value,
                       grain = fixture_binning(dir, row$series, row$grain),
                       stats = strsplit(row$stat, "+", fixed = TRUE)[[1L]],
                       year_start = row$year_start, partial = row$partial)
    start <- attr(x, "bin_start")
    # The shape is asserted before the digest, so a binning that puts the record into a different
    # number of bins is reported as that rather than as an unexplained hash mismatch.
    expect_equal(dim(x)[1], row$n_unit, info = label)
    expect_equal(dim(x)[2], row$n_bin, info = label)
    expect_equal(format(start[1], "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"), row$first_bin, info = label)
    expect_equal(format(start[length(start)], "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
                 row$last_bin, info = label)
    expect_equal(sum(attr(x, "bin_partial")), row$n_partial, info = label)
    # The row order is asserted by name before the digest, so a session whose collation orders the
    # ids differently is reported as that rather than as an unexplained hash mismatch.
    expect_identical(dimnames(x)[[1L]][1L], row$first_unit, info = label)
    expect_identical(dimnames(x)[[1L]][dim(x)[1]], row$last_unit, info = label)
    expect_identical(digest_array(x), row$digest, info = label)
  }
})

test_that("the fixtures cover a record that starts on no bin boundary", {
  dir <- fixture_dir()
  skip_if(is.null(dir), "fixtures are not in the built package")
  expected <- read.csv(file.path(dir, "digests.csv"), stringsAsFactors = FALSE)

  # A record beginning at midnight on the year_start anniversary puts every grain in phase with
  # it, which is the one input on which a rule that keeps a partial leading bin and a rule that
  # never makes one agree. The contract is only a contract if it also carries the other case.
  expect_true(all(c("aligned", "offset", "zoned", "order") %in% expected$series))
  offset <- expected[expected$series == "offset", ]
  expect_true(all(c("native", "halfday", "day", "week", "month", "season", "year", "astronomical")
                  %in% offset$grain))
  expect_gt(sum(offset$n_partial), 0)
  expect_true(all(c("keep", "drop") %in% expected$partial))
  expect_gt(length(unique(expected$year_start)), 1L)

  # A contract checked only in UTC verifies the calendar on the one zone where the question does
  # not arise. Both a zone that moves its clock in the middle of the day and one that moves it at
  # midnight are pinned.
  expect_true(all(c("UTC", "Europe/Vienna", "America/Sao_Paulo") %in% expected$tz))
  expect_true(any(expected$tz == "America/Sao_Paulo" & expected$year_start == "11-04"))

  # Every grain whose bin count the offset record splits differently from the aligned one is
  # pinned by a row of its own, so a change to either binning rule moves a digest here.
  aligned <- expected[expected$series == "aligned" & expected$stat == "mean" &
                        expected$partial == "keep" & expected$year_start == "09-01" &
                        expected$tz == "UTC", ]
  expect_setequal(aligned$grain,
                  c("native", "halfday", "day", "week", "month", "season", "year", "astronomical"))
})

test_that("dropping every bin is an error rather than an empty representation", {
  t <- seq(as.POSIXct("2021-10-17 05:00:00", tz = "UTC"), by = "hour", length.out = 24 * 40)
  d <- data.frame(plot = "a", t = t, temp = sin(seq_along(t) / 24), stringsAsFactors = FALSE)
  expect_error(grain_matrix(d, plot, t, temp, grain = "year", partial = "drop"),
               "no whole year")
  expect_error(grain_matrix(d, plot, t, temp, grain = "day", partial = "sometimes"),
               "'arg' should be one of")
})

test_that("the digest is the LF-terminated twelve-place form and nothing else", {
  x <- array(c(1, -0.5), dim = c(2L, 1L, 1L))
  body <- charToRaw("1.000000000000\n-0.500000000000\n")
  f <- tempfile()
  con <- file(f, open = "wb")
  writeBin(body, con)
  close(con)
  expect_identical(digest_array(x), unname(tools::md5sum(f)))
  unlink(f)
})

test_that("a digest is refused over an array that is not finite", {
  expect_error(digest_array(array(c(1, Inf), dim = c(2L, 1L, 1L))), "not finite")
  expect_error(digest_array(array(c(1, -Inf), dim = c(2L, 1L, 1L))), "not finite")
  expect_error(digest_array(array(c(1, NaN), dim = c(2L, 1L, 1L))), "not finite")
  expect_error(digest_array(array(c(1, NA_real_), dim = c(2L, 1L, 1L))), "1 of 2")
})

test_that("the row order is C collation and not the session's", {
  dir <- fixture_dir()
  skip_if(is.null(dir), "fixtures are not in the built package")
  record <- fixture_series(dir, "order")

  # These five ids are the case that made the two languages disagree: an English locale orders
  # them _x a1 A1 P10 P9, C collation orders them A1 P10 P9 _x a1, and NumPy gives the second.
  # The representation must give the second whatever LC_COLLATE the session runs in.
  x <- grain_matrix(record, id, time, value, grain = "day")
  expect_identical(dimnames(x)[[1L]], c("A1", "P10", "P9", "_x", "a1"))

  # The input row order carries no meaning: the ids arrive in a third order again.
  expect_identical(unique(record$id), c("a1", "P9", "_x", "A1", "P10"))
  shuffled <- record[order(record$value), , drop = FALSE]
  expect_identical(digest_array(grain_matrix(shuffled, id, time, value, grain = "day")),
                   digest_array(x))
})

test_that("the scorable mask orders its variables by C collation too", {
  y <- matrix(c(1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1), nrow = 4,
              dimnames = list(paste0("u", 1:4), c("a1", "P9", "_x")))
  cells <- scorable_cells(y, c(u1 = 1L, u2 = 1L, u3 = 2L, u4 = 2L))
  expect_identical(unique(cells$variable), c("P9", "_x", "a1"))
})

test_that("a supplied calendar that breaks its guarantees is refused, as the fixtures pin", {
  dir <- fixture_dir()
  skip_if(is.null(dir), "fixtures are not in the built package")
  guards <- read.csv(file.path(dir, "grain_guards.csv"), stringsAsFactors = FALSE)
  expect_gte(nrow(guards), 2L)

  for (i in seq_len(nrow(guards))) {
    row <- guards[i, ]
    record <- fixture_series(dir, row$series)
    expect_error(grain_matrix(record, id, time, value,
                              grain = fixture_calendar(row$calendar), stats = "mean"),
                 row$message, fixed = TRUE)
  }
})
