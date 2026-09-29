#!/usr/bin/env Rscript
# Regenerates spec/fixtures/. This is a deliberate act with its own commit: a digest that moves
# means the representation moved, and the question is which implementation is wrong.

# Run from the package root: Rscript inst/spec/make_fixtures.R
if (!file.exists("DESCRIPTION")) {
  stop("run this from the package root", call. = FALSE)
}
suppressMessages(pkgload::load_all(".", quiet = TRUE))
out_dir <- "inst/spec/fixtures"

# Every fixture is written through here, over a binary connection: a text one translates the line
# feed to CRLF on Windows, which would make a regeneration there a diff of every byte rather than
# of the digests that moved.
write_fixture <- function(x, file) {
  holds_comma <- vapply(x, function(col) any(grepl(",", as.character(col), fixed = TRUE)),
                        logical(1L))
  if (any(holds_comma)) {
    stop(file, " is written unquoted and a field of ", names(x)[holds_comma][1L],
         " holds a comma.", call. = FALSE)
  }
  con <- file(file.path(out_dir, file), open = "wb")
  on.exit(close(con), add = TRUE)
  write.csv(x, con, row.names = FALSE, quote = FALSE, eol = "\n")
}

# Four series. The first begins at midnight on the default year_start, so every coarse grain is in
# phase with it from the first reading. The second begins at an arbitrary hour of an arbitrary day,
# which is what a logger deployed when someone could walk to it gives, and puts every grain out of
# phase: a record that starts on a bin boundary cannot tell two binning rules apart. The third and
# the fourth are described where they are built.
make_series <- function(from, units, days, seed) {
  t <- seq(as.POSIXct(from, tz = "UTC"), by = "hour", length.out = 24 * days)
  set.seed(seed)
  value <- unlist(lapply(seq_along(units), function(k) {
    season <- 8 * sin(2 * pi * (seq_along(t) / (24 * 365.25)) - pi / 2)
    diurnal <- 3 * sin(2 * pi * seq_along(t) / 24)
    round(season + diurnal + k + rnorm(length(t), sd = 0.5), 6)
  }))
  data.frame(id = rep(units, each = length(t)), time = rep(t, length(units)),
             value = value, stringsAsFactors = FALSE)
}

write_series <- function(series, file) {
  write_fixture(
    data.frame(id = series$id,
               time = format(series$time, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
               value = sprintf("%.6f", series$value)),
    file
  )
}

# The third is short and sits across 4 November 2018, the night America/Sao_Paulo moved its clock
# at midnight and that local day began at 01:00. It is the record that tells a calendar read by
# arithmetic apart from one read by writing a local midnight and parsing it back.
# The fourth carries ids that C collation and an English locale's collation order differently:
# C gives A1 P10 P9 _x a1, an English locale gives _x a1 A1 P10 P9. The ids arrive in a third
# order again, so a fixture that passes is evidence both that the input order carries no meaning
# and that the output order is the one the contract names. Ids like p01 to p03 agree under every
# rule and pin nothing.
SERIES <- list(
  aligned = make_series("2021-09-01 00:00:00", c("p01", "p02", "p03"), 400, 20260902L),
  offset = make_series("2021-10-17 05:00:00", c("p01", "p02"), 200, 20260903L),
  zoned = make_series("2018-11-01 00:00:00", c("p01", "p02"), 10, 20260904L),
  order = make_series("2021-09-01 00:00:00", c("a1", "P9", "_x", "A1", "P10"), 30, 20260905L)
)
write_series(SERIES$aligned, "series.csv")
write_series(SERIES$offset, "series_offset.csv")
write_series(SERIES$zoned, "series_zoned.csv")
write_series(SERIES$order, "series_order.csv")

# A calendar the package does not carry, cut where the deposit cuts its seasons: at the equinoxes
# and the solstices rather than on the first of a month. The first edge is the series' own first
# reading, so every reading falls at or after an edge and the two languages' interval lookups agree
# on every one of them.
EDGES <- list(
  aligned = c("2021-09-01", "2021-09-22", "2021-12-21", "2022-03-20", "2022-06-21", "2022-09-23"),
  offset = c("2021-10-17", "2021-12-21", "2022-03-20")
)
write_fixture(
  data.frame(series = rep(names(EDGES), lengths(EDGES)),
             edge = paste0(unlist(EDGES, use.names = FALSE), "T00:00:00Z")),
  "seasons.csv"
)

astronomical <- function(name) {
  edges <- as.POSIXct(EDGES[[name]], tz = "UTC")
  function(when) edges[findInterval(as.numeric(when), as.numeric(edges))]
}

# Three calendars that break what a supplied one has to satisfy, each named so both suites build
# the same function. `late` gives every reading the midnight after it, which is what a calendar
# shifted by one boundary returns and what an interval lookup wraps to below its first edge;
# `alternate` sends consecutive readings to two bins an hour apart, which interleaves them;
# `missing` gives the first reading of the second day no bin start at all.
BAD_CALENDARS <- list(
  late = function(when) {
    as.POSIXct((floor(as.numeric(when) / 86400) + 1) * 86400, origin = "1970-01-01", tz = "UTC")
  },
  alternate = function(when) {
    t0 <- min(as.numeric(when))
    as.POSIXct(t0 + 3600 * (((as.numeric(when) - t0) / 3600) %% 2), origin = "1970-01-01",
               tz = "UTC")
  },
  missing = function(when) {
    out <- as.POSIXct(floor(as.numeric(when) / 86400) * 86400, origin = "1970-01-01", tz = "UTC")
    out[as.numeric(when) == min(as.numeric(when)) + 86400] <- NA
    out
  }
)

# The three-channel schemes a caller asks for by name in the literature this package serves: the
# grain's own extremes, its typical day, and its coldest and warmest day.
schemes <- list(
  c("min", "mean", "max"),
  c("mean_daily_min", "mean", "mean_daily_max"),
  c("cold_day", "mean", "warm_day")
)

fine <- c("native", "halfday")
coarse <- c("month", "season", "year")
grains <- c("native", "halfday", "day", "week", "month", "season", "year")

rows <- list()
digest_row <- function(name, w, stats, year_start = "09-01", partial = "keep", tz = "UTC") {
  series <- SERIES[[name]]
  attr(series$time, "tzone") <- tz
  binning <- if (w == "astronomical") astronomical(name) else w
  x <- grain_matrix(series, id, time, value, grain = binning, stats = stats,
                     year_start = year_start, partial = partial)
  start <- attr(x, "bin_start")
  data.frame(series = name, grain = w, tz = tz, year_start = year_start, partial = partial,
             stat = paste(stats, collapse = "+"), n_unit = dim(x)[1], n_bin = dim(x)[2],
             first_bin = format(start[1], "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
             last_bin = format(start[length(start)], "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
             n_partial = sum(attr(x, "bin_partial")),
             first_unit = dimnames(x)[[1L]][1L],
             last_unit = dimnames(x)[[1L]][dim(x)[1]],
             digest = digest_array(x),
             stringsAsFactors = FALSE)
}
add <- function(...) rows[[length(rows) + 1L]] <<- digest_row(...)

for (name in c("aligned", "offset")) {
  for (w in grains) {
    available <- if (w %in% fine) c("mean", "min", "max") else
      c("mean", "min", "max", "cold_day", "warm_day", "mean_daily_min", "mean_daily_max")
    for (s in available) {
      add(name, w, s)
    }
    for (scheme in schemes) {
      if (w %in% fine && any(scheme %in% .day_level_stats())) {
        next
      }
      add(name, w, scheme)
    }
  }
  # The anniversary sets the phase of the two grains that count from it, so a contract checked at
  # the default alone pins only the phase the fixtures happen to start on.
  for (w in coarse) {
    for (ys in c("01-01", "03-01", "07-15")) {
      add(name, w, "mean", year_start = ys)
    }
  }
  # The bin the record does not fill is a choice, so both settings are pinned rather than the one
  # that happens to be the default. A grain the record holds no whole one of has nothing to pin:
  # it errors, and the test suites assert that separately.
  for (w in setdiff(grains, "native")) {
    whole <- !attr(grain_matrix(SERIES[[name]], id, time, value, grain = w), "bin_partial")
    if (any(whole)) {
      add(name, w, "mean", partial = "drop")
    }
  }
  # A calendar the package does not carry takes the same statistics a named grain does, so it is
  # pinned over the same grid rather than over the two schemes it happened to be introduced with.
  for (s in c("mean", "min", "max", "cold_day", "warm_day", "mean_daily_min", "mean_daily_max")) {
    add(name, "astronomical", s)
  }
  for (scheme in schemes) {
    add(name, "astronomical", scheme)
  }
}

# The zone. The instants are the same bytes on disk whichever calendar reads them, so a zone row
# is the same series with a different clock over it, and the digest is what says the two languages
# read that clock the same way. Europe/Vienna moves its clock twice inside the aligned record;
# America/Sao_Paulo moves it at midnight inside the short one, which is the case that has no local
# midnight to parse.
for (w in grains) {
  add("aligned", w, "mean", tz = "Europe/Vienna")
}
add("aligned", "week", c("cold_day", "mean", "warm_day"), tz = "Europe/Vienna")
add("aligned", "week", c("mean_daily_min", "mean", "mean_daily_max"), tz = "Europe/Vienna")
add("aligned", "day", c("min", "mean", "max"), tz = "Europe/Vienna")
add("zoned", "day", "mean")
for (w in c("native", "halfday", "day", "week")) {
  add("zoned", w, "mean", tz = "America/Sao_Paulo")
}
add("zoned", "day", c("min", "mean", "max"), tz = "America/Sao_Paulo")
add("zoned", "day", c("cold_day", "mean", "warm_day"), tz = "America/Sao_Paulo")
add("zoned", "day", c("mean_daily_min", "mean", "mean_daily_max"), tz = "America/Sao_Paulo")
add("zoned", "year", "mean", year_start = "11-04", tz = "America/Sao_Paulo")

# The row order. Every unit holds a different level, so reading the units in the wrong order moves
# the digest rather than leaving it as it was, and the first and last unit are named in the row so
# a mismatch is reported as an order rather than as an unexplained hash.
for (w in c("day", "week", "month")) {
  add("order", w, "mean")
}
add("order", "day", c("cold_day", "mean", "warm_day"))

write_fixture(do.call(rbind, rows), "digests.csv")
cat("wrote", length(rows), "digests\n")

# ---- the lookback -----------------------------------------------------------------------
# The reduction anchored on a target rather than on the calendar. The anchors come in named sets
# rather than one set per series: an anchor that is a local midnight in one zone is not one in
# another, and an anchor on the hour rules out the four day-level statistics that a midnight
# allows, so the same series carries two sets that pin different halves of the contract.
TARGETS <- list(
  aligned = list(series = "aligned", units = c("p01", "p02", "p03"), tz = "UTC",
                 at = c("2022-01-01 00:00:00", "2022-04-15 00:00:00", "2022-09-01 00:00:00")),
  offset = list(series = "offset", units = c("p01", "p02"), tz = "UTC",
                at = c("2022-01-01 00:00:00", "2022-03-01 00:00:00")),
  hourly = list(series = "offset", units = c("p01", "p02"), tz = "UTC",
                at = c("2022-01-15 05:00:00", "2022-02-20 13:00:00")),
  zoned = list(series = "zoned", units = c("p01", "p02"), tz = "America/Sao_Paulo",
               at = c("2018-11-08 00:00:00", "2018-11-10 00:00:00"))
)

lookback_targets <- do.call(rbind, lapply(names(TARGETS), function(name) {
  spec <- TARGETS[[name]]
  grid <- expand.grid(at = spec$at, id = spec$units,
                      KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  data.frame(set = name, series = spec$series, id = grid$id,
             at = format(as.POSIXct(grid$at, tz = spec$tz), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
             stringsAsFactors = FALSE)
}))
write_fixture(lookback_targets, "lookback_targets.csv")

lookback_at <- function(set) {
  taken <- lookback_targets[lookback_targets$set == set, ]
  data.frame(id = taken$id,
             at = as.POSIXct(taken$at, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
             stringsAsFactors = FALSE)
}

lookback_rows <- list()
lookback_row <- function(set, span, lag, bins, stats, tz = "UTC") {
  series <- SERIES[[TARGETS[[set]]$series]]
  attr(series$time, "tzone") <- tz
  at <- lookback_at(set)
  x <- lookback_matrix(series, id, time, value, at = at, span = span, lag = lag,
                     bins = bins, stats = stats)
  data.frame(set = set, tz = tz, span = span, lag = lag, bins = bins,
             stat = paste(stats, collapse = "+"), n_target = dim(x)[1], n_bin = dim(x)[2],
             first_bin = dimnames(x)[[2L]][1L], last_bin = dimnames(x)[[2L]][dim(x)[2]],
             first_unit = at$id[1L], last_unit = at$id[nrow(at)],
             digest = digest_array(x), stringsAsFactors = FALSE)
}
add_lookback <- function(...) lookback_rows[[length(lookback_rows) + 1L]] <<- lookback_row(...)

every_stat <- c("mean", "min", "max", "cold_day", "warm_day", "mean_daily_min", "mean_daily_max")
reading_stats <- c("min", "mean", "max")

# Anchors on a day boundary, so every statistic is available and the bin count is what decides.
for (s in every_stat) add_lookback("aligned", "30 days", "0 days", 1L, s)
for (scheme in schemes) add_lookback("aligned", "30 days", "0 days", 1L, scheme)
for (s in every_stat) add_lookback("aligned", "30 days", "0 days", 3L, s)
for (scheme in schemes) add_lookback("aligned", "30 days", "0 days", 3L, scheme)
for (scheme in schemes) add_lookback("aligned", "30 days", "7 days", 3L, scheme)
for (scheme in schemes) add_lookback("aligned", "7 days", "0 days", 7L, scheme)
for (scheme in schemes) add_lookback("aligned", "120 days", "0 days", 4L, scheme)
# A step of no whole number of days, and a lookback opening off one: the two the day-level four are
# refused for, pinned here with the statistics they are allowed for.
add_lookback("aligned", "7 days", "0 days", 3L, reading_stats)
add_lookback("aligned", "1 week", "12 hours", 1L, "mean")

for (s in every_stat) add_lookback("offset", "30 days", "0 days", 3L, s)
for (scheme in schemes) add_lookback("offset", "30 days", "0 days", 3L, scheme)
for (scheme in schemes) add_lookback("offset", "60 days", "7 days", 1L, scheme)
add_lookback("offset", "7 days", "0 days", 7L, c("cold_day", "mean", "warm_day"))

for (s in reading_stats) add_lookback("hourly", "12 hours", "0 seconds", 1L, s)
add_lookback("hourly", "12 hours", "0 seconds", 1L, reading_stats)
add_lookback("hourly", "12 hours", "12 hours", 3L, reading_stats)
add_lookback("hourly", "3 days", "0 days", 1L, "mean")
add_lookback("hourly", "86400", "0 days", 2L, "mean")

# The zone. The anchors are local midnights in a clock that moved at midnight four days earlier, so
# they are day boundaries on that calendar and on no other; the same anchors read as UTC are not,
# and the row that reads them that way carries the one statistic that does not care.
for (s in every_stat) add_lookback("zoned", "2 days", "0 days", 2L, s, tz = "America/Sao_Paulo")
for (scheme in schemes) {
  add_lookback("zoned", "2 days", "0 days", 2L, scheme, tz = "America/Sao_Paulo")
}
add_lookback("zoned", "2 days", "1 day", 1L, c("cold_day", "mean", "warm_day"),
           tz = "America/Sao_Paulo")
add_lookback("zoned", "2 days", "0 days", 2L, "mean")

write_fixture(do.call(rbind, lookback_rows), "lookback_digests.csv")

# What a lookback refuses. A digest cannot carry a case that errors, so each guard is pinned by the
# input that fires it and the part of the message both languages must raise.
WINDOW_GUARDS <- list(
  list(set = "aligned", tz = "UTC", span = "1 year", lag = "0 days", bins = 5L, stat = "mean",
       message = "first: target 1"),
  list(set = "aligned", tz = "UTC", span = "7 days", lag = "0 days", bins = 3L, stat = "cold_day",
       message = "needs bins of a calendar day or coarser"),
  list(set = "hourly", tz = "UTC", span = "2 days", lag = "0 days", bins = 1L, stat = "warm_day",
       message = "opens at 2022-01-13T05:00:00")
)
for (guard in WINDOW_GUARDS) {
  series <- SERIES[[TARGETS[[guard$set]]$series]]
  attr(series$time, "tzone") <- guard$tz
  raised <- tryCatch({
    lookback_matrix(series, id, time, value, at = lookback_at(guard$set), span = guard$span,
                  lag = guard$lag, bins = guard$bins, stats = guard$stat)
    ""
  }, error = function(e) conditionMessage(e))
  if (!grepl(guard$message, raised, fixed = TRUE)) {
    stop("the guard on ", guard$set, " raised \"", raised, "\", not \"", guard$message, "\"",
         call. = FALSE)
  }
}
write_fixture(do.call(rbind, lapply(WINDOW_GUARDS, as.data.frame, stringsAsFactors = FALSE)),
          "lookback_guards.csv")
cat("wrote", length(lookback_rows), "lookback digests and", length(WINDOW_GUARDS), "guards\n")

# What a supplied calendar refuses, pinned the same way: the series, the calendar that breaks it,
# and the part of the message both languages must raise.
GRAIN_GUARDS <- list(
  list(series = "aligned", calendar = "late", message = "beginning after it"),
  list(series = "offset", calendar = "alternate", message = "interleaves two of its bins"),
  # The count and the first reading are two rows of one calendar: the fixtures are written
  # unquoted, so a field holds no comma, and the message joins the two with one.
  list(series = "aligned", calendar = "missing", message = "gives no bin start for 3 readings"),
  list(series = "aligned", calendar = "missing",
       message = "first: unit p01 at 2021-09-02T00:00:00Z")
)
for (guard in GRAIN_GUARDS) {
  raised <- tryCatch({
    grain_matrix(SERIES[[guard$series]], id, time, value,
                 grain = BAD_CALENDARS[[guard$calendar]], stats = "mean")
    ""
  }, error = function(e) conditionMessage(e))
  if (!grepl(guard$message, raised, fixed = TRUE)) {
    stop("the ", guard$calendar, " calendar raised \"", raised, "\", not \"", guard$message, "\"",
         call. = FALSE)
  }
}
write_fixture(do.call(rbind, lapply(GRAIN_GUARDS, as.data.frame, stringsAsFactors = FALSE)),
          "grain_guards.csv")
cat("wrote", length(GRAIN_GUARDS), "grain guards\n")

# ---- what reaches which bin ---------------------------------------------------------------------
# `coverage()` lays the same binning out as a count of readings per (unit, bin), and it is the one
# reduction no digest above reaches: every series here is complete, and a gap is what the table
# exists to show. Each case names the readings it takes out, by unit and by span, so both suites
# build the same input rather than each writing a record that happens to have a hole in it.
COVERAGE_CASES <- list(
  list(case = "complete", series = "aligned", unit = "", from = "", to = "", grain = "month"),
  list(case = "custom", series = "aligned", unit = "", from = "", to = "",
       grain = "astronomical"),
  list(case = "late_start", series = "aligned", unit = "p02", from = "2021-09-01T00:00:00Z",
       to = "2021-10-01T00:00:00Z", grain = "week"),
  list(case = "lost_month", series = "offset", unit = "p01", from = "2021-12-01T00:00:00Z",
       to = "2021-12-11T00:00:00Z", grain = "day"),
  # Every unit loses the same span, so the calendar tiles over a bin no unit reaches at all, which
  # is the case a count of readings per cell cannot report as a row of zeros.
  list(case = "skipped_bin", series = "aligned", unit = "all", from = "2022-02-01T00:00:00Z",
       to = "2022-03-01T00:00:00Z", grain = "month")
)

coverage_input <- function(spec) {
  series <- SERIES[[spec$series]]
  if (!nzchar(spec$unit)) {
    return(series)
  }
  from <- as.POSIXct(spec$from, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  to <- as.POSIXct(spec$to, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  lost <- series$time >= from & series$time < to &
    (spec$unit == "all" | series$id == spec$unit)
  series[!lost, , drop = FALSE]
}

coverage_rows <- lapply(COVERAGE_CASES, function(spec) {
  got <- coverage(coverage_input(spec), id, time,
                  grain = if (spec$grain == "astronomical") astronomical(spec$series)
                          else spec$grain)
  empty <- got == 0L
  data.frame(case = spec$case, series = spec$series, unit = spec$unit, from = spec$from,
             to = spec$to, grain = spec$grain, n_unit = nrow(got), n_bin = ncol(got),
             first_bin = colnames(got)[1L], last_bin = colnames(got)[ncol(got)],
             n_empty = sum(empty), n_unit_gap = sum(rowSums(empty) > 0L),
             n_bin_skipped = sum(colSums(empty) == nrow(got)),
             digest = digest_array(array(as.numeric(got), dim = c(dim(got), 1L))),
             stringsAsFactors = FALSE)
})
write_fixture(do.call(rbind, coverage_rows), "coverage.csv")
cat("wrote", length(COVERAGE_CASES), "coverage cases\n")

# ---- where a bin sits in the year ---------------------------------------------------------------
# `calendar_channels()` and `bind_channels()` return an array a learner reads, so they are pinned
# the way the reductions are rather than tested on each side alone. A `calendar` row is a
# representation's two channels; a `bound` row is its readings with those channels beside them,
# which is what pins the order the two are joined in and the attributes the joined array keeps.
#
# What is digested is the fraction of the year each bin sits at, not the sine and the cosine of it.
# The fraction is exact arithmetic on the calendar and the same bits wherever it is computed; the
# sine and the cosine are the platform's library, accurate to about an ulp and no further, and
# twelve decimal places is close enough to that to make a hash of them a statement about the
# machine. The suites assert the two channels against the fraction to the tolerance the contract
# states, which is what the fraction is pinned here for.
channel_rows <- list()
channel_row <- function(name, w, stats = "mean", kind = "calendar", year_start = "09-01",
                        partial = "keep", tz = "UTC", cycle = "year") {
  series <- SERIES[[name]]
  attr(series$time, "tzone") <- tz
  binning <- if (w == "astronomical") astronomical(name) else w
  x <- grain_matrix(series, id, time, value, grain = binning, stats = stats,
                     year_start = year_start, partial = partial)
  cc <- calendar_channels(x, cycles = cycle)
  got <- if (kind == "bound") bind_channels(x, cc) else cc
  start <- attr(got, "bin_start")
  frac <- ts_cycle_fraction_(as.numeric(attr(got, "bin_start")),
                             as.numeric(attr(got, "bin_end")), cycle)
  data.frame(series = name, grain = w, tz = tz, year_start = year_start, partial = partial,
             stat = paste(stats, collapse = "+"), kind = kind, cycle = cycle,
             n_unit = dim(got)[1],
             n_bin = dim(got)[2],
             channels = paste(dimnames(got)[[3L]], collapse = "+"),
             first_bin = format(start[1], "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
             last_bin = format(start[length(start)], "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
             tolerance = 1e-12, digest = digest_array(frac), stringsAsFactors = FALSE)
}
add_channels <- function(...) channel_rows[[length(channel_rows) + 1L]] <<- channel_row(...)

# The grain sets the width of a bin and so the midpoint the phase is read at, and a supplied
# calendar's bins are neither a month nor a week wide.
for (w in c("day", "week", "month", "season", "year", "astronomical")) {
  add_channels("aligned", w)
}
# A record out of phase with every grain, where a bin the record only partly covers is read at the
# phase it was measured over rather than at the phase of a whole one.
for (w in c("day", "week", "month")) {
  add_channels("offset", w)
}
# The anniversary moves the bins of the two grains that count from it, and the phase with them.
for (w in c("season", "year")) {
  add_channels("aligned", w, year_start = "03-01")
}
add_channels("aligned", "week", partial = "drop")
# The zone. The bins move by the offset and the phase moves with them, so a zone row is what says
# the two languages read the same instants off the same bins.
add_channels("aligned", "week", tz = "Europe/Vienna")
# `native` is the bin whose start and end are one instant, and `halfday` the one narrower than the
# day the four day-level statistics need: two bin widths no other row reaches.
for (w in c("native", "halfday", "day")) {
  add_channels("zoned", w, tz = "America/Sao_Paulo")
}
add_channels("aligned", "week", stats = c("cold_day", "mean", "warm_day"), kind = "bound")
add_channels("offset", "week", stats = c("min", "mean", "max"), kind = "bound")
add_channels("zoned", "day", kind = "bound", tz = "America/Sao_Paulo")
# The place in the day, on the two grains finer than a day. The zone row is what says the phase is
# read on the instants rather than on the clock the bins were placed on.
for (w in c("native", "halfday")) {
  add_channels("zoned", w, tz = "America/Sao_Paulo", cycle = "day")
}
add_channels("zoned", "native", kind = "bound", tz = "America/Sao_Paulo", cycle = "day")

write_fixture(do.call(rbind, channel_rows), "channels_digests.csv")
cat("wrote", length(channel_rows), "channel digests\n")

# What the two refuse. Each case is named rather than written out, so both suites build the same
# input: every one reads the aligned series, `x` at the weekly grain, `other` at the monthly one,
# and `back` a 30-day lookback anchored on the last reading of each unit.
CHANNEL_GUARDS <- list(
  list(case = "lookback", message = "no position in the year"),
  list(case = "not_a_representation", message = "not a representation"),
  list(case = "one_argument", message = "at least two representations"),
  list(case = "duplicate", message = "carry a channel of the same name: mean"),
  list(case = "different_bins", message = "argument 2 covers different units or bins"),
  list(case = "day_coarse", message = "sit a day or more apart"),
  list(case = "unknown_cycle", message = "unknown cycle: month")
)
channel_guard <- function(case) {
  series <- SERIES$aligned
  x <- grain_matrix(series, id, time, value, grain = "week")
  switch(case,
    lookback = {
      at <- data.frame(id = unique(series$id), at = max(series$time), stringsAsFactors = FALSE)
      calendar_channels(lookback_matrix(series, id, time, value, at = at, span = "30 days"))
    },
    not_a_representation = bind_channels(x, 1),
    one_argument = bind_channels(x),
    duplicate = bind_channels(x, x),
    different_bins = bind_channels(x, grain_matrix(series, id, time, value, grain = "month")),
    day_coarse = calendar_channels(x, cycles = "day"),
    unknown_cycle = calendar_channels(x, cycles = "month"),
    stop("no channel guard called ", case))
}
for (guard in CHANNEL_GUARDS) {
  raised <- tryCatch({
    channel_guard(guard$case)
    ""
  }, error = function(e) conditionMessage(e))
  if (!grepl(guard$message, raised, fixed = TRUE)) {
    stop("the ", guard$case, " guard raised \"", raised, "\", not \"", guard$message, "\"",
         call. = FALSE)
  }
}
write_fixture(do.call(rbind, lapply(CHANNEL_GUARDS, as.data.frame, stringsAsFactors = FALSE)),
          "channels_guards.csv")
cat("wrote", length(CHANNEL_GUARDS), "channel guards\n")

# ---- what crosses the boundary above the representation ----------------------------------------
# The three artifacts, in the format the contract defines, plus the numbers read off them that are
# deterministic: the scorable mask, every threshold metric, and a paired contrast. None of these
# has a fitted model in it, so each can be pinned exactly rather than compared by hand once and
# left that way.

set.seed(20260906L)
resp_units <- sprintf("u%02d", 1:40)
# Variable names that C collation and an English locale order differently, so the cell order of the
# mask is pinned by the same case the representation's row order is.
resp_vars <- c("a1", "P9", "_x", "sp1", "sp2", "sp3")
# Prevalences chosen so the mask is not all TRUE: a species present nowhere and one present
# everywhere have no scorable cell at all, and a rare one has some folds and not others.
prevalence <- c(0.5, 0.05, 0, 1, 0.2, 0.3)
y_fix <- vapply(prevalence, function(pr) as.numeric(stats::rbinom(length(resp_units), 1L, pr)),
                numeric(length(resp_units)))
dimnames(y_fix) <- list(resp_units, resp_vars)
f_fix <- fold_map(y_fix, v = 5, seed = 11L)

write_folds(f_fix, file.path(out_dir, "folds.csv"))
write_response(y_fix, file.path(out_dir, "response.csv"))
write_cells(scorable_cells(y_fix, f_fix), file.path(out_dir, "cells.csv"))

# The threshold metrics and the sweep they all read off. Ties, a single presence, a single absence,
# a cell of one class, a perfect separation and a reversed one: the cases where the rule about
# where a cut may fall is the whole answer.
METRIC_CASES <- list(
  plain = list(y = c(0, 0, 0, 1, 1, 1, 0, 1),
               p = c(0.10, 0.20, 0.35, 0.40, 0.60, 0.90, 0.55, 0.70)),
  all_tied = list(y = c(0, 1, 0, 1, 1, 0), p = rep(0.5, 6)),
  some_tied = list(y = c(0, 1, 1, 0, 1, 0), p = c(0.2, 0.2, 0.8, 0.8, 0.5, 0.5)),
  tied_across_classes = list(y = c(0, 1, 0, 1), p = c(0.3, 0.3, 0.7, 0.7)),
  one_presence = list(y = c(0, 0, 0, 1, 0, 0), p = c(0.10, 0.30, 0.20, 0.90, 0.40, 0.05)),
  one_absence = list(y = c(1, 1, 1, 0, 1), p = c(0.9, 0.8, 0.7, 0.2, 0.6)),
  all_presence = list(y = c(1, 1, 1, 1), p = c(0.1, 0.4, 0.6, 0.9)),
  all_absence = list(y = c(0, 0, 0, 0), p = c(0.1, 0.4, 0.6, 0.9)),
  perfect = list(y = c(0, 0, 1, 1), p = c(0.1, 0.2, 0.8, 0.9)),
  reversed = list(y = c(1, 1, 0, 0), p = c(0.1, 0.2, 0.8, 0.9))
)
write_fixture(
  do.call(rbind, lapply(names(METRIC_CASES), function(nm) {
    d <- METRIC_CASES[[nm]]
    data.frame(case = nm, y = d$y, p = sprintf("%.12g", d$p), stringsAsFactors = FALSE)
  })),
  "metric_cases.csv"
)

METRIC_FNS <- list(
  tss = tss, roc_auc = roc_auc, average_precision = average_precision,
  kappa = function(y, p) kappa_score(y, p, "prevalence"),
  kappa_youden = function(y, p) kappa_score(y, p, "youden"),
  threshold_youden = function(y, p) decision_threshold(y, p, "youden"),
  threshold_kappa = function(y, p) decision_threshold(y, p, "kappa"),
  threshold_prevalence = function(y, p) decision_threshold(y, p, "prevalence")
)
# A case a metric defines no value on is written NA rather than left out, so a suite that quietly
# skipped it would fail rather than pass.
write_fixture(
  do.call(rbind, lapply(names(METRIC_CASES), function(nm) {
    d <- METRIC_CASES[[nm]]
    data.frame(case = nm, metric = names(METRIC_FNS),
               value = vapply(METRIC_FNS, function(fn) {
                 v <- fn(d$y, d$p)
                 if (is.finite(v)) sprintf("%.12g", v) else "NA"
               }, character(1L)),
               stringsAsFactors = FALSE)
  })),
  "metrics.csv"
)

# The paired contrast, from a fixed table of per-cell scores rather than from a fit: the pairing,
# the per-variable mean and the signed-rank p-value are the part both languages own, and a fitted
# model is the part they are not required to share. Cells one arm scored and the other did not are
# in the table, because dropping those is what the function is for. Six variables with no tied
# per-variable difference keeps the p-value on the exact branch of the signed-rank distribution,
# where the two implementations agree to the last place rather than to the normal approximation.
contrast_cells <- expand.grid(fold = 1:5, variable = resp_vars,
                              KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
contrast_cells <- contrast_cells[order(contrast_cells$variable, contrast_cells$fold,
                                       method = "radix"), c("variable", "fold")]
set.seed(20260907L)
contrast_cells$a <- round(stats::runif(nrow(contrast_cells), 0.30, 0.85), 6)
contrast_cells$b <- round(contrast_cells$a - stats::rnorm(nrow(contrast_cells), 0.04, 0.06), 6)
contrast_cells$a[c(2L, 17L)] <- NA_real_
contrast_cells$b[c(5L, 17L, 23L)] <- NA_real_
rownames(contrast_cells) <- NULL
write_fixture(
  data.frame(variable = contrast_cells$variable, fold = contrast_cells$fold,
             a = ifelse(is.na(contrast_cells$a), "NA", sprintf("%.12g", contrast_cells$a)),
             b = ifelse(is.na(contrast_cells$b), "NA", sprintf("%.12g", contrast_cells$b)),
             stringsAsFactors = FALSE),
  "contrast_cells.csv"
)

as_ladder <- function(cells) {
  arm <- function(name, score) {
    data.frame(grain = "week", learner = name, variable = cells$variable, fold = cells$fold,
               score = score, scorable = !is.na(score), stringsAsFactors = FALSE)
  }
  structure(rbind(arm("a", cells$a), arm("b", cells$b)),
            class = c("timesift_ladder", "data.frame"))
}
QUANTITIES <- c("diff", "lower", "upper", "n_variable", "n_cell", "n_favour", "p_value",
                "p_method")
contrast <- paired_contrast(as_ladder(contrast_cells), "week|a", "week|b")
write_fixture(
  data.frame(quantity = QUANTITIES,
             value = vapply(QUANTITIES, function(nm) {
               v <- contrast[[nm]]
               if (is.character(v)) v else sprintf("%.12g", v)
             }, character(1L)),
             stringsAsFactors = FALSE),
  "contrast.csv"
)

# The one number here that cannot be a digest: it draws replicates, from each language's own random
# stream, and aligning those streams would be the wrong fix for the same reason it is for a fold
# map. What is pinned is this side's value, and what the contract requires is that the other side
# lands within the band the document states.
inflation <- tss_inflation(y_fix, f_fix, skill = c(0.6, 0.7, 0.9), replicates = 200L, seed = 1L)
write_fixture(
  data.frame(skill = sprintf("%.12g", inflation$skill),
             reported = sprintf("%.12g", inflation$reported),
             inflation = sprintf("%.12g", inflation$inflation),
             replicates = 200L, tolerance = 0.02, stringsAsFactors = FALSE),
  "inflation.csv"
)

cat("wrote the response, the fold map, the mask,",
    nrow(utils::read.csv(file.path(out_dir, "metrics.csv"))),
    "metric values, the contrast and the inflation", "\n")

# ---- the combiner ---------------------------------------------------------------------------------
# Out-of-fold predictions of the response above for five candidates, rounded so the file holds them
# exactly, and every combination read off them. "forest / week" is "elasticnet / week" again, so
# two candidates score the same to the last bit and the decay's rule for a tie is exercised; the
# response carries a variable present nowhere and one present everywhere, on which no member holds
# a cut and a committee has no vote. Each side scores the candidates under its own `tss`, which the
# metric fixtures pin, and hands those scores to the combiner as a run does.
set.seed(20260929L)
ens_signal <- c("cnn / week" = 1.6, "cnn / month" = 1.1, "elasticnet / week" = 0.8,
                "mlp / week" = 0.2)
ens_oof <- lapply(ens_signal, function(a) {
  z <- a * (2 * y_fix - 1) + matrix(stats::rnorm(length(y_fix), sd = 1), nrow = nrow(y_fix))
  matrix(round(stats::plogis(z), 6), nrow = nrow(y_fix), dimnames = dimnames(y_fix))
})
ens_oof[["forest / week"]] <- ens_oof[["elasticnet / week"]]
write_fixture(
  do.call(rbind, lapply(names(ens_oof), function(nm) {
    p <- ens_oof[[nm]]
    data.frame(candidate = nm, id = rep(rownames(p), ncol(p)),
               variable = rep(colnames(p), each = nrow(p)), p = sprintf("%.6f", as.numeric(p)),
               stringsAsFactors = FALSE)
  })),
  "ensemble_oof.csv"
)

ens_cells <- scorable_cells(y_fix, f_fix)
ens_fold <- .as_folds(f_fix, rownames(y_fix))
ens_scores <- do.call(rbind, lapply(names(ens_oof), function(nm) {
  out <- .score_arm(nm, nm, y_fix, ens_oof[[nm]], ens_fold, sort(unique(ens_fold)), ens_cells,
                    tss)
  data.frame(candidate = nm, variable = out$variable, fold = out$fold, score = out$score,
             scorable = out$scorable, stringsAsFactors = FALSE)
}))

# One row per case. An empty field is the argument left at its default.
ENSEMBLE_CASES <- list(
  stack = ensemble("stack"),
  mean = ensemble("mean"),
  median = ensemble("median"),
  weighted = ensemble("weighted"),
  weighted_auc = ensemble("weighted", metric = "roc_auc"),
  decay = ensemble("weighted", decay = 1.6),
  decay_min_score = ensemble("weighted", decay = 2, min_score = 0.6),
  committee = ensemble("committee"),
  committee_kappa = ensemble("committee", rule = "kappa"),
  committee_prevalence = ensemble("committee", rule = "prevalence"),
  min_score_learners = ensemble("mean", scope = "learners", min_score = 0.7),
  stack_representations = ensemble("stack", scope = "representations")
)
ens_field <- function(x) if (is.null(x)) "" else as.character(x)
write_fixture(
  do.call(rbind, lapply(names(ENSEMBLE_CASES), function(nm) {
    s <- ENSEMBLE_CASES[[nm]]
    data.frame(case = nm, method = s$method, scope = s$scope, metric = ens_field(s$metric),
               min_score = ens_field(s$min_score),
               decay = if (identical(s$decay, "proportional")) "" else ens_field(s$decay),
               rule = if (identical(s$rule, "youden")) "" else ens_field(s$rule),
               stringsAsFactors = FALSE)
  })),
  "ensemble_cases.csv"
)

ens_fits <- lapply(ENSEMBLE_CASES, function(s) {
  ensemble_fit(ens_oof, y_fix, ens_cells, f_fix, s, ens_scores)
})
ens_number <- function(v) ifelse(is.finite(v), sprintf("%.12g", v), "NA")
write_fixture(
  do.call(rbind, lapply(names(ens_fits), function(nm) {
    st <- ens_fits[[nm]]
    data.frame(case = nm, member = names(st$weights), weight = ens_number(st$weights),
               stringsAsFactors = FALSE)
  })),
  "ensemble_weights.csv"
)
write_fixture(
  do.call(rbind, lapply(names(ens_fits), function(nm) {
    thr <- ens_fits[[nm]]$thresholds
    if (is.null(thr)) {
      return(NULL)
    }
    data.frame(case = nm, member = rep(rownames(thr), ncol(thr)),
               variable = rep(colnames(thr), each = nrow(thr)),
               threshold = ens_number(as.numeric(thr)), stringsAsFactors = FALSE)
  })),
  "ensemble_thresholds.csv"
)
# The combination and the spread, each read on the out-of-fold predictions themselves.
write_fixture(
  do.call(rbind, lapply(names(ens_fits), function(nm) {
    p <- ensemble_combine(ens_fits[[nm]], ens_oof)
    s <- ensemble_spread(ens_fits[[nm]], ens_oof, alpha = 0.1)
    data.frame(case = nm, id = rep(rownames(p), ncol(p)),
               variable = rep(colnames(p), each = nrow(p)),
               combined = ens_number(as.numeric(p)),
               mean = ens_number(as.numeric(s[, , "mean"])),
               sd = ens_number(as.numeric(s[, , "sd"])),
               cv = ens_number(as.numeric(s[, , "cv"])),
               lower = ens_number(as.numeric(s[, , "lower"])),
               upper = ens_number(as.numeric(s[, , "upper"])),
               stringsAsFactors = FALSE)
  })),
  "ensemble_predict.csv"
)
cat("wrote", length(ens_fits), "combiner cases\n")

# ---------------------------------------------------------------------------------------------
# The penalised fit.
#
# These are not digests. Two implementations of a coordinate descent settle at the same point to
# the tolerance they are run at, never to the last bit, so what is pinned is the reference itself
# and the distance a suite may sit from it. The reference is glmnet's, because the penalised arm
# is the baseline the networks are measured against and the point of the shared core is that it
# does not weaken it: a suite that reproduces these numbers reproduces glmnet.
#
# The design is the one `elasticnet()` penalises over: a weekly representation and the square of
# every column beside it. The squares are the scale case the standardisation exists for, and the
# weekly bins of one record are collinear the way adjacent bins are, so a suite that skipped the
# standardisation fails and one that stops the descent early fails on the split between two
# collinear columns.
if (!requireNamespace("glmnet", quietly = TRUE)) {
  stop("the penalised fixtures are the reference glmnet gives, so it has to be installed to ",
       "regenerate them.", call. = FALSE)
}

PEN_THRESH <- 1e-14
# glmnet's own pass budget, raised well above its default: at the tolerance the reference is read
# at, a collinear design runs past the default and glmnet then warns, truncates the path and
# returns the penalties it did reach. A reference that quietly carried one of those would be a
# fixture both implementations had to reproduce a failure to match.
PEN_MAXIT <- 1e7
# The objective both implementations minimise, on glmnet's own scale: the mean deviance halved
# for a Gaussian family and the mean negative log likelihood for a binomial one, plus the penalty.
# It is what the fixture pins tightly. A coefficient is not: two nearly identical columns split
# one coefficient between them differently in any two descents, and the split is not determined
# to the precision the fit is, so that is asserted at a looser tolerance beside this one.
pen_objective <- function(x, y, w, family, alpha, lambda, a0, beta) {
  wn <- w / sum(w)
  eta <- a0 + as.numeric(x %*% beta)
  fit <- if (family == "gaussian") sum(wn * (y - eta)^2) / 2 else
    -sum(wn * (y * eta - log1p(exp(eta))))
  fit + lambda * (alpha * sum(abs(beta)) + (1 - alpha) / 2 * sum(beta^2))
}

settled <- function(expr) {
  withCallingHandlers(expr, warning = function(cond) {
    stop("glmnet did not settle while the penalised reference was generated: ",
         conditionMessage(cond), call. = FALSE)
  })
}

# The design is a representation rather than a table of draws: a weekly grain over a simulated
# record, flattened, with the square of every column beside it, which is what `elasticnet()`
# penalises over. The squares are the scale case the standardisation exists for, a reading of ten
# degrees and its square of a hundred, and the weekly bins of one record are collinear the way
# adjacent bins are. Both suites read the design from the fixture, so the Python side needs no
# record simulator of its own to sit on the same numbers.
#
# The records come from `simulate_records()` rather than from the suite's own helper, whose units
# differ by one number and whose weekly bins are therefore a rank-one design no coordinate
# descent settles on.
pen_sim <- simulate_records(n = 80L, mechanism = "lag", variables = 1L, prevalence = 0.35,
                            auc = 0.8, days = 70L, step_hours = 6, seed = 20260918L, draw = 3L)
pen_x <- .design(grain_matrix(pen_sim$readings, unit, time, reading, grain = "week"),
                 squares = TRUE)
pen_x[] <- round(pen_x, 6)
PEN_N <- nrow(pen_x)
PEN_P <- ncol(pen_x)
pen_units <- rownames(pen_x)
pen_y_binomial <- as.numeric(pen_sim$y[, 1L])
# The continuous case is the driver the response was generated from, which is a real function of
# the record rather than a second draw beside it.
pen_y_gaussian <- round(as.numeric(pen_sim$driver[, 1L]), 6)
set.seed(20260918L)
pen_w <- round(stats::runif(PEN_N, 0.3, 3), 6)
pen_fold <- sample(rep_len(0:4, PEN_N))

# The column names carry the bin and the channel each predictor came from, which is what says the
# design is a representation and not a matrix of numbers.
pen_input <- data.frame(unit = pen_units, stringsAsFactors = FALSE)
for (j in seq_len(PEN_P)) pen_input[[colnames(pen_x)[j]]] <- sprintf("%.12g", pen_x[, j])
pen_input$y_gaussian <- sprintf("%.12g", pen_y_gaussian)
pen_input$y_binomial <- pen_y_binomial
pen_input$w <- sprintf("%.12g", pen_w)
pen_input$fold <- pen_fold
write_fixture(pen_input, "penalised_input.csv")

PEN_CASES <- do.call(rbind, lapply(c("gaussian", "binomial"), function(family) {
  do.call(rbind, lapply(c(1, 0.5, 0), function(alpha) {
    do.call(rbind, lapply(c(FALSE, TRUE), function(weighted) {
      data.frame(case = sprintf("%s_a%02d_%s", family, round(alpha * 10),
                                if (weighted) "weighted" else "flat"),
                 family = family, alpha = alpha, weighted = weighted,
                 stringsAsFactors = FALSE)
    }))
  }))
}))

pen_path_rows <- list()
pen_cv_rows <- list()
for (i in seq_len(nrow(PEN_CASES))) {
  row <- PEN_CASES[i, ]
  y <- if (row$family == "binomial") pen_y_binomial else pen_y_gaussian
  w <- if (row$weighted) pen_w else rep(1, PEN_N)
  fit <- settled(glmnet::glmnet(pen_x, y, family = row$family, alpha = row$alpha, weights = w,
                                control = list(thresh = PEN_THRESH, maxit = PEN_MAXIT)))
  beta <- as.matrix(fit$beta)
  # The path's first point is left out. glmnet fits it at a penalty of 9.9e35 and reports it
  # under the largest penalty that leaves every coefficient at zero, which is the same fit for
  # any mixing above zero and is not the same fit for a ridge, where nothing is ever exactly
  # zero. The core solves that point, so a ridge disagrees with glmnet there by construction.
  points <- unique(c(seq(2L, length(fit$lambda), by = 10L), length(fit$lambda)))
  pen_path_rows[[i]] <- data.frame(
    case = row$case, point = points,
    lambda = sprintf("%.12g", fit$lambda[points]),
    a0 = sprintf("%.12g", as.numeric(fit$a0)[points]),
    objective = sprintf("%.12g", vapply(points, function(k)
      pen_objective(pen_x, y, w, row$family, row$alpha, fit$lambda[k], fit$a0[k], beta[, k]),
      numeric(1L))),
    matrix(sprintf("%.12g", beta[, points]), nrow = length(points), byrow = TRUE,
           dimnames = list(NULL, paste0("b", seq_len(PEN_P)))),
    stringsAsFactors = FALSE)
  cv <- settled(glmnet::cv.glmnet(pen_x, y, family = row$family, alpha = row$alpha, weights = w,
                                  foldid = pen_fold + 1L, type.measure = "deviance",
                                  control = list(thresh = PEN_THRESH, maxit = PEN_MAXIT)))
  pen_cv_rows[[i]] <- data.frame(
    case = row$case, n_point = length(fit$lambda),
    lambda_min = sprintf("%.12g", cv$lambda.min), lambda_1se = sprintf("%.12g", cv$lambda.1se),
    index_min = which(cv$lambda == cv$lambda.min), index_1se = which(cv$lambda == cv$lambda.1se),
    cv_min = sprintf("%.12g", min(cv$cvm)),
    cv_sd_min = sprintf("%.12g", cv$cvsd[which.min(cv$cvm)]),
    stringsAsFactors = FALSE)
}

write_fixture(
  data.frame(PEN_CASES[, c("case", "family", "alpha", "weighted")],
             thresh = sprintf("%.12g", PEN_THRESH), max_pass = sprintf("%.12g", PEN_MAXIT),
             tolerance = 1e-4, objective_tolerance = 1e-6, stringsAsFactors = FALSE),
  "penalised_cases.csv"
)
write_fixture(do.call(rbind, pen_path_rows), "penalised_path.csv")
write_fixture(do.call(rbind, pen_cv_rows), "penalised_cv.csv")

cat("wrote", nrow(PEN_CASES), "penalised cases,",
    nrow(do.call(rbind, pen_path_rows)), "reference coefficients\n")

# A penalised fit that does not settle at a penalty. `penalised_stall_input.csv` is one inner
# cross-validation of the selection benchmark's `elasticnet-none-n300` cell, captured as it was
# handed over (replicate 102, #78): 96 units, 7 presences weighted 12.7, eight columns. It is read
# here rather than written, because it is the benchmark's own call and not a design this script
# draws. glmnet does not settle on it: at the threshold the call ran at, one fold's path stops at
# penalty 86 and another at 100, and glmnet returns the points before each. The reference is the
# penalty cv.glmnet chooses over those truncated folds, so it is read with glmnet's warnings on,
# which is the event being pinned.
stall <- utils::read.csv(file.path(out_dir, "penalised_stall_input.csv"), check.names = FALSE)
stall_x <- as.matrix(stall[, setdiff(names(stall), c("unit", "y", "w", "fold"))])
STALL_THRESH <- 1e-8
stall_warnings <- character()
stall_cv <- withCallingHandlers(
  glmnet::cv.glmnet(stall_x, stall$y, family = "binomial", alpha = 0.5, weights = stall$w,
                    foldid = stall$fold + 1L, type.measure = "deviance",
                    control = list(thresh = STALL_THRESH)),
  warning = function(cond) {
    stall_warnings <<- c(stall_warnings, conditionMessage(cond))
    invokeRestart("muffleWarning")
  })
stall_stops <- sum(grepl("Convergence for [0-9]+th lambda value not reached", stall_warnings))
if (stall_stops == 0L) {
  stop("glmnet settled on every fold of the captured stall, so it no longer pins the event it ",
       "was captured for.", call. = FALSE)
}
write_fixture(
  data.frame(family = "binomial", alpha = 0.5, n_fold = 5L,
             thresh = sprintf("%.12g", STALL_THRESH),
             lambda_min = sprintf("%.12g", stall_cv$lambda.min),
             lambda_1se = sprintf("%.12g", stall_cv$lambda.1se),
             glmnet_stops = stall_stops, stringsAsFactors = FALSE),
  "penalised_stall_cv.csv")
cat("wrote the captured stall's reference,", stall_stops, "glmnet folds stopping early\n")

# ---- the grain contrast and the simulator's design ------------------------------------------------
# The mixed model of the grain contrast is fitted by each language's own optimiser and its adjusted
# p-values and intervals are integrals each language evaluates by quasi-Monte Carlo, so what is
# pinned here is R's table and the tolerance each column is read at is the spec's: the differences
# to the optimiser's, the rest to the integrator's. The design is small, twelve variables over
# four folds, so the degrees of freedom are low enough that reading them as emmeans does moves the
# critical value, and cells are missing, so the fit is not the balanced one a shortcut would get
# right by accident.
set.seed(20260925L)
gc_vars <- sprintf("s%02d", 1:12)
gc_cells <- expand.grid(fold = 1:4, variable = gc_vars, grain = c("day", "week", "month"),
                        KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
gc_cells$score <- round(0.62 + stats::rnorm(12, sd = 0.08)[match(gc_cells$variable, gc_vars)] +
                          stats::rnorm(4, sd = 0.015)[gc_cells$fold] +
                          c(day = 0, week = 0.025, month = 0.012)[gc_cells$grain] +
                          stats::rnorm(nrow(gc_cells), sd = 0.03), 6)
gc_cells$score[c(3L, 20L, 41L, 77L, 130L)] <- NA_real_
gc_cells <- gc_cells[c("grain", "variable", "fold", "score")]
write_fixture(
  data.frame(gc_cells[c("grain", "variable", "fold")],
             score = ifelse(is.na(gc_cells$score), "NA", sprintf("%.12g", gc_cells$score)),
             stringsAsFactors = FALSE),
  "grain_contrast_cells.csv")
gc_ladder <- structure(cbind(learner = "cnn", gc_cells, scorable = !is.na(gc_cells$score),
                             stringsAsFactors = FALSE),
                       class = c("timesift_ladder", "data.frame"), metric = "tss")
gc_out <- grain_contrasts(gc_ladder)
write_fixture(
  data.frame(gc_out[c("learner", "grain", "reference")],
             lapply(gc_out[c("diff", "lower", "upper", "p_value")], sprintf, fmt = "%.12g"),
             stringsAsFactors = FALSE),
  "grain_contrast.csv")

# The simulator's design draws nothing, so every number of it is pinned: where each variable's
# stretch of bins is anchored, the weights it reads the readings by, the population standard
# deviation of its driver and the link solved for the asked-for prevalence and skill.
sim_rows <- list()
for (m in c("event", "season", "lag")) {
  sim <- simulate_records(n = 2L, mechanism = m, variables = 6L, days = 400L, prevalence = 0.2,
                          auc = 0.8)
  when <- sort(unique(sim$readings$time))
  phi <- exp(-sim$design$step_hours / (24 * sim$design$anomaly_days))
  d <- .simulate_design(m, 6L, when, "09-01", phi, 0, 1, 0.2, 0.8, 1L)
  sim_rows[[m]] <- data.frame(mechanism = m, grain = sim$grain, bins = sim$design$bins,
                              variable = 1:6, anchor = d$anchor,
                              weight_ss = sprintf("%.12g", colSums(d$weights^2)),
                              sigma = sprintf("%.12g", d$sigma),
                              b0 = sprintf("%.12g", d$link$b0), b1 = sprintf("%.12g", d$link$b1),
                              stringsAsFactors = FALSE)
}
write_fixture(do.call(rbind, sim_rows), "simulate_design.csv")
cat("wrote the grain contrast and the simulator's design\n")

# The tree.
#
# The reference is rpart's, grown on the design the penalised fixtures carry, read back from its
# own file so both sides and rpart see the same doubles. rpart's class priors are the data's class
# shares, which it computes with R's extended-precision sums; the core takes them as exactly those
# shares. Under integer weights the two are the same numbers and the tree is rpart's to the last
# bit, so the weights here are counts. Under fractional weights a split whose complexity ties the
# threshold exactly can fall the other way, and on which side depends on the width of R's long
# double on the machine rpart runs on.
tree_x <- as.matrix(utils::read.csv(file.path(out_dir, "penalised_input.csv"),
                                    check.names = FALSE)[, colnames(pen_x)])
set.seed(20260928L)
tree_count <- sample(1:4, PEN_N, replace = TRUE)
write_fixture(data.frame(unit = pen_units, count = tree_count, stringsAsFactors = FALSE),
              "tree_weights.csv")

TREE_CASES <- do.call(rbind, lapply(c("binomial", "gaussian"), function(family) {
  do.call(rbind, lapply(c("flat", "counts"), function(weights) {
    rbind(
      data.frame(case = sprintf("%s_%s_package", family, weights), family = family,
                 weights = weights, min_split = 20L, min_leaf = 7L, cp = 0.01, max_depth = 30L,
                 stringsAsFactors = FALSE),
      data.frame(case = sprintf("%s_%s_bigboss", family, weights), family = family,
                 weights = weights, min_split = 5L, min_leaf = 5L, cp = 0.001, max_depth = 10L,
                 stringsAsFactors = FALSE),
      data.frame(case = sprintf("%s_%s_deep", family, weights), family = family,
                 weights = weights, min_split = 4L, min_leaf = 2L, cp = 0, max_depth = 4L,
                 stringsAsFactors = FALSE))
  }))
}))
write_fixture(TREE_CASES, "tree_cases.csv")

tree_nodes <- list()
tree_tables <- list()
tree_predictions <- list()
for (i in seq_len(nrow(TREE_CASES))) {
  row <- TREE_CASES[i, ]
  y <- if (row$family == "binomial") pen_y_binomial else pen_y_gaussian
  w <- if (row$weights == "counts") tree_count else rep(1, PEN_N)
  d <- data.frame(y = if (row$family == "binomial") factor(y, levels = c(0, 1)) else y,
                  tree_x, check.names = TRUE)
  fit <- rpart::rpart(y ~ ., data = d, weights = w,
                      method = if (row$family == "binomial") "class" else "anova",
                      control = rpart::rpart.control(minsplit = row$min_split,
                                                     minbucket = row$min_leaf, cp = row$cp,
                                                     maxdepth = row$max_depth, maxcompete = 0L,
                                                     maxsurrogate = 0L, xval = pen_fold + 1L))
  f <- fit$frame
  inner <- f$var != "<leaf>"
  column <- rep(-1L, nrow(f))
  column[inner] <- match(as.character(f$var[inner]), make.names(colnames(tree_x))) - 1L
  threshold <- rep(0, nrow(f))
  less_left <- rep(0L, nrow(f))
  if (any(inner)) {
    threshold[inner] <- fit$splits[, "index"]
    less_left[inner] <- as.integer(fit$splits[, "ncat"] == -1)
  }
  value <- if (row$family == "binomial") f$yval2[, 5L] else f$yval
  tree_nodes[[i]] <- data.frame(case = row$case, number = as.integer(rownames(f)),
                                column = column, threshold = sprintf("%.17g", threshold),
                                less_left = less_left, n = f$n,
                                weight = sprintf("%.17g", f$wt), risk = sprintf("%.17g", f$dev),
                                complexity = sprintf("%.17g", f$complexity),
                                value = sprintf("%.17g", value), stringsAsFactors = FALSE)
  cpt <- fit$cptable
  tree_tables[[i]] <- data.frame(case = row$case, row = seq_len(nrow(cpt)),
                                 cp = sprintf("%.17g", cpt[, "CP"]),
                                 nsplit = as.integer(cpt[, "nsplit"]),
                                 rel_error = sprintf("%.17g", cpt[, "rel error"]),
                                 xerror = sprintf("%.17g", cpt[, "xerror"]),
                                 xstd = sprintf("%.17g", cpt[, "xstd"]),
                                 stringsAsFactors = FALSE)
  # biomod2's pruning: the least cross-validated error plus its standard error among the rows
  # that keep a split, the last of them where several tie.
  kept <- as.data.frame(cpt)
  kept$xsum <- kept$xerror + kept$xstd
  kept <- kept[kept$nsplit > 0, ]
  pruned <- if (nrow(kept)) rpart::prune(fit, cp = kept$CP[max(which(kept$xsum == min(kept$xsum)))])
            else fit
  p <- stats::predict(pruned, d)
  p <- if (row$family == "binomial") p[, "1"] else as.numeric(p)
  tree_predictions[[i]] <- data.frame(case = row$case, unit = pen_units,
                                      se_sum = sprintf("%.17g", p), stringsAsFactors = FALSE)
}
write_fixture(do.call(rbind, tree_nodes), "tree_nodes.csv")
write_fixture(do.call(rbind, tree_tables), "tree_cptable.csv")
write_fixture(do.call(rbind, tree_predictions), "tree_predict.csv")
cat("wrote the tree reference\n")

# The forest.
#
# No package grows a forest from this generator, so the reference is the forest grown from the
# spec's own text in R alone, `tests/testthat/helper-oracle-forest.R`, on the tree's design and
# counts. The core is asserted against it on both sides.
source("tests/testthat/helper-oracle-forest.R")
FOREST_CASES <- data.frame(
  case = c("binomial_flat", "binomial_counts", "binomial_balance", "binomial_bagged",
           "gaussian_flat", "gaussian_counts"),
  family = c(rep("binomial", 4L), rep("gaussian", 2L)),
  weights = c("flat", "counts", "counts", "flat", "flat", "counts"),
  trees = 3L,
  mtry = c(4L, 4L, 4L, ncol(tree_x), 7L, 7L),
  min_leaf = c(1L, 2L, 1L, 1L, 5L, 3L),
  balance = c(FALSE, FALSE, TRUE, FALSE, FALSE, FALSE),
  seed = c(11, 12, 13, 14, 15, 4294967295),
  stringsAsFactors = FALSE)
write_fixture(transform(FOREST_CASES, seed = sprintf("%.0f", seed),
                        balance = as.integer(balance)), "forest_cases.csv")

forest_nodes <- list()
forest_predictions <- list()
for (i in seq_len(nrow(FOREST_CASES))) {
  row <- FOREST_CASES[i, ]
  y <- if (row$family == "binomial") pen_y_binomial else pen_y_gaussian
  w <- if (row$weights == "counts") tree_count else rep(1, PEN_N)
  grown <- oracle_forest(tree_x, y, w, row$family, row$trees, row$mtry, row$min_leaf,
                         row$balance, row$seed)
  forest_nodes[[i]] <- do.call(rbind, lapply(seq_along(grown), function(t) {
    g <- grown[[t]]
    data.frame(case = row$case, tree = t - 1L, node = seq_along(g$column) - 1L,
               column = g$column, threshold = sprintf("%.17g", g$threshold),
               less_left = g$less_left, left = g$left, right = g$right,
               value = sprintf("%.17g", g$value), stringsAsFactors = FALSE)
  }))
  # The forest's prediction is the mean over its trees, summed in tree order.
  leaf_value <- function(g, i) {
    at <- 1L
    while (g$column[at] >= 0L) {
      below <- tree_x[i, g$column[at] + 1L] < g$threshold[at]
      at <- 1L + if (below == (g$less_left[at] == 1L)) g$left[at] else g$right[at]
    }
    g$value[at]
  }
  predicted <- vapply(seq_len(PEN_N), function(i) {
    oracle_serial_sum(vapply(grown, leaf_value, numeric(1L), i = i)) / length(grown)
  }, numeric(1L))
  forest_predictions[[i]] <- data.frame(case = row$case, unit = pen_units,
                                        value = sprintf("%.17g", predicted),
                                        stringsAsFactors = FALSE)
}
write_fixture(do.call(rbind, forest_nodes), "forest_nodes.csv")
write_fixture(do.call(rbind, forest_predictions), "forest_predict.csv")

# The generator's first outputs, for a seed at each end of its range and a tree far from the first.
forest_stream <- do.call(rbind, lapply(list(c(1, 0), c(1, 1), c(0, 0), c(4294967295, 3),
                                            c(20260929, 499)), function(st) {
  s <- oracle_stream(st[1L], st[2L])
  data.frame(seed = sprintf("%.0f", st[1L]), tree = st[2L], index = 0:7,
             output = sprintf("%.0f", vapply(1:8, function(k) s$next_word(), numeric(1L))),
             stringsAsFactors = FALSE)
}))
write_fixture(forest_stream, "forest_stream.csv")
cat("wrote the forest reference\n")

# Boosting.
#
# The first-order trees are gbm's, and gbm fits them with nothing drawn when every unit is in the
# bag, so gbm is the reference there, cross-validated folds included: each fold's fit is gbm's own
# `gbm.fit()` with the fold held out behind `nTrain`, as `gbmDoFold` arranges it, and the folds'
# held-out deviance is combined as `gbmCrossValErr` combines it. The second-order trees are
# xgboost's exact ones, which store the design in single precision, so they agree to that and no
# closer; the binomial cases carry weights that are not whole numbers, because under equal weights
# the first round's gradients take two values, many splits then tie exactly, and the two libraries'
# sums break the tie differently. The draws, which neither library makes as the spec does, are
# pinned by the fit grown from the spec's text in R alone, `tests/testthat/helper-oracle-boost.R`.
source("tests/testthat/helper-oracle-boost.R")
set.seed(20260929L)
boost_weight <- round(stats::runif(PEN_N, 0.5, 2), 3)
write_fixture(data.frame(unit = pen_units, weight = sprintf("%.3f", boost_weight),
                         stringsAsFactors = FALSE), "boost_weights.csv")
BOOST_CASES <- data.frame(
  case = c("gbm_binomial_flat", "gbm_binomial_counts", "gbm_gaussian_flat", "gbm_gaussian_counts",
           "gbm_binomial_cv", "gbm_gaussian_cv", "xgboost_binomial_random",
           "xgboost_binomial_gamma", "xgboost_gaussian_flat", "xgboost_gaussian_random",
           "oracle_binomial_drawn", "oracle_gaussian_drawn"),
  reference = c(rep("gbm", 6L), rep("xgboost", 4L), rep("oracle", 2L)),
  family = c("binomial", "binomial", "gaussian", "gaussian", "binomial", "gaussian",
             "binomial", "binomial", "gaussian", "gaussian", "binomial", "gaussian"),
  weights = c("flat", "counts", "flat", "counts", "flat", "counts", "random", "random", "flat",
              "random", "counts", "random"),
  newton = c(rep(0L, 6L), rep(1L, 4L), 0L, 0L),
  trees = c(50L, 50L, 40L, 40L, 60L, 60L, 20L, 20L, 20L, 20L, 20L, 20L),
  depth = c(1L, 3L, 2L, 4L, 2L, 3L, 2L, 3L, 3L, 4L, 2L, 3L),
  shrinkage = c(0.1, 0.05, 0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.3, 0.1, 0.1),
  min_leaf = c(10, 5, 5, 3, 5, 5, 1, 1, 1, 2, 3, 3),
  subsample = c(rep(1, 10L), 0.6, 0.7),
  colsample = c(rep(1, 10L), 0.5, 0.4),
  lambda = c(rep(0, 6L), 1, 1, 1, 2, 0, 0),
  gamma = c(rep(0, 6L), 0, 0.5, 0, 1, 0, 0),
  cv = c(0L, 0L, 0L, 0L, 1L, 1L, rep(0L, 6L)),
  seed = c(rep(1, 10L), 7, 4294967295),
  stringsAsFactors = FALSE)
write_fixture(transform(BOOST_CASES, seed = sprintf("%.0f", seed)), "boost_cases.csv")

boost_x <- tree_x
colnames(boost_x) <- make.names(colnames(boost_x))
gbm_fit <- function(x, y, w, dist, row, n_train = nrow(x)) {
  gbm::gbm.fit(x, y, w = w, distribution = dist, n.trees = row$trees,
               interaction.depth = row$depth, shrinkage = row$shrinkage,
               n.minobsinnode = row$min_leaf, bag.fraction = 1, nTrain = n_train,
               verbose = FALSE, keep.data = FALSE)
}
boost_predictions <- list()
boost_cv <- list()
for (i in seq_len(nrow(BOOST_CASES))) {
  row <- BOOST_CASES[i, ]
  y <- if (row$family == "binomial") pen_y_binomial else pen_y_gaussian
  w <- switch(row$weights, flat = rep(1, PEN_N), counts = tree_count, random = boost_weight)
  if (row$reference == "gbm") {
    dist <- if (row$family == "binomial") "bernoulli" else "gaussian"
    n_trees <- row$trees
    if (row$cv == 1L) {
      folds <- sort(unique(pen_fold))
      valid <- sapply(folds, function(g) {
        i_fold <- order(pen_fold == g)
        gbm_fit(boost_x[i_fold, , drop = FALSE], y[i_fold], w[i_fold], dist, row,
                n_train = sum(pen_fold != g))$valid.error
      })
      held <- tabulate(match(pen_fold, folds), nbins = length(folds))
      cv_error <- rowSums(sweep(valid, 2L, held, `*`)) / PEN_N
      n_trees <- which.min(cv_error)
      boost_cv[[length(boost_cv) + 1L]] <- data.frame(
        case = row$case, tree = seq_along(cv_error), cv_error = sprintf("%.17g", cv_error),
        stringsAsFactors = FALSE)
    }
    fit <- gbm_fit(boost_x, y, w, dist, row)
    p <- gbm:::predict.gbm(fit, as.data.frame(boost_x), n.trees = n_trees, type = "response")
  } else if (row$reference == "xgboost") {
    dm <- xgboost::xgb.DMatrix(boost_x, label = y, weight = w)
    fit <- xgboost::xgb.train(
      params = list(objective = if (row$family == "binomial") "binary:logistic" else
                      "reg:squarederror",
                    tree_method = "exact", max_depth = row$depth, eta = row$shrinkage,
                    lambda = row$lambda, gamma = row$gamma, min_child_weight = row$min_leaf,
                    nthread = 1),
      data = dm, nrounds = row$trees, verbose = 0)
    p <- stats::predict(fit, dm)
  } else {
    fit <- oracle_boost(boost_x, y, w, row$family, row$trees, row$depth, row$shrinkage,
                        row$min_leaf, row$subsample, row$colsample, row$seed)
    p <- oracle_boost_predict(fit, boost_x, row$family == "binomial")
  }
  boost_predictions[[i]] <- data.frame(case = row$case, unit = pen_units,
                                       value = sprintf("%.17g", p), stringsAsFactors = FALSE)
}
write_fixture(do.call(rbind, boost_predictions), "boost_predict.csv")
write_fixture(do.call(rbind, boost_cv), "boost_cv.csv")
cat("wrote the boosting reference\n")

# maxnet.
#
# The reference is the maxnet package's own fit, on the design the penalised fixtures carry
# without its squares: the weekly columns, read back from their file so every side sees the same
# doubles. maxnet's features and its regularisation are arithmetic and are pinned to rounding. Its
# lasso is glmnet's, run here at the tolerance the penalised reference is, and pinned the way that
# one is: by the objective at maxnet's last penalty, which both sides reach to that tolerance, and by
# the predictions, which the few features nearly collinear with each other leave slightly less
# determined. maxnet moves glmnet's own controls for the whole session (`pmin = 1e-8, fdev = 0`),
# so they are set back after every fit.
if (!requireNamespace("maxnet", quietly = TRUE)) {
  stop("the maxnet fixtures are the reference the maxnet package gives, so it has to be ",
       "installed to regenerate them.", call. = FALSE)
}
mx_x <- tree_x[, !grepl("\\^2$", colnames(tree_x)), drop = FALSE]
mx_names <- sprintf("v%02d", seq_len(ncol(mx_x)))
mx_frame <- function(x) stats::setNames(as.data.frame(x), mx_names)
# Two thinner responses, the first 8 and the first 12 presences of the binomial one in unit order,
# put maxnet's own choice of classes at its two smallest presence counts.
mx_first <- function(k) {
  y <- pen_y_binomial
  y[which(y == 1)[-seq_len(k)]] <- 0
  y
}
mx_y <- list(y_binomial = pen_y_binomial, y_8 = mx_first(8L), y_12 = mx_first(12L))
write_fixture(data.frame(unit = pen_units, y_8 = mx_y$y_8, y_12 = mx_y$y_12,
                         stringsAsFactors = FALSE), "maxnet_response.csv")

MX_THRESH <- 1e-14
mx_case <- function(case, formulation = "background", response = "y_binomial", classes = "",
                    regmult = 1, add_samples = TRUE, duplicate = FALSE, weighted = FALSE) {
  data.frame(case = case, formulation = formulation, response = response, classes = classes,
             regmult = regmult, add_samples = add_samples, duplicate = duplicate,
             weighted = weighted, stringsAsFactors = FALSE)
}
MX_CASES <- rbind(
  mx_case("default"),
  mx_case("default_8", response = "y_8"),
  mx_case("default_12", response = "y_12"),
  mx_case("lqph", classes = "lqph"),
  mx_case("lqpht", classes = "lqpht"),
  mx_case("thresholds", classes = "t"),
  mx_case("hinges_regmult2", classes = "h", regmult = 2),
  mx_case("lq_regmult05", classes = "lq", regmult = 0.5),
  mx_case("l_no_samples", classes = "l", add_samples = FALSE),
  mx_case("lqh_duplicate", classes = "lqh", duplicate = TRUE),
  mx_case("absence_flat", formulation = "absence", classes = "lqh"),
  mx_case("absence_weighted", formulation = "absence", classes = "lqh", weighted = TRUE))

mx_rows <- list()
mx_reg <- list()
mx_pred <- list()
for (i in seq_len(nrow(MX_CASES))) {
  row <- MX_CASES[i, ]
  x <- mx_x
  y <- mx_y[[row$response]]
  # The duplicate case appends the first presence's readings once more as an absence, so that
  # presence is not added to the background.
  if (row$duplicate) {
    x <- rbind(x, x[which(y == 1)[1L], , drop = FALSE])
    y <- c(y, 0)
  }
  df <- mx_frame(x)
  classes <- if (nzchar(row$classes)) row$classes else "default"
  f <- maxnet::maxnet.formula(y, df, classes = classes)
  used <- if (nzchar(row$classes)) row$classes else
    as.character(timesift:::ts_maxnet_design_(as.numeric(x), y, nrow(x), ncol(x), "", 50L, 1,
                                              "background", TRUE, 2)$classes)
  if (row$formulation == "background") {
    ref <- maxnet::maxnet(y, df, f, regmult = row$regmult,
                          addsamplestobackground = row$add_samples,
                          control = list(thresh = MX_THRESH, maxit = PEN_MAXIT))
    glmnet::glmnet.control(factory = TRUE)
    if (length(ref$lambda) != 200L) stop("maxnet did not complete its path on ", row$case)
    grown <- if (row$add_samples) {
      keep <- which(y == 1)
      keep <- keep[!vapply(keep, function(k) any(apply(x[y == 0, , drop = FALSE], 1L,
                                                       function(r) identical(r, x[k, ]))),
                           logical(1L))]
      list(x = rbind(x, x[keep, , drop = FALSE]), y = c(y, rep(0, length(keep))))
    } else list(x = x, y = y)
    mm <- stats::model.matrix(f, mx_frame(grown$x))
    reg <- ref$penalty.factor
    w <- grown$y + (1 - grown$y) * 100
    lambda <- ref$lambda[200L]
    a0 <- as.numeric(ref$a0[200L])
    beta <- as.numeric(ref$beta[, 200L])
    y_fit <- grown$y
    out <- mx_frame(x * 1.3)
    mx_pred[[i]] <- data.frame(
      case = row$case, row = seq_len(nrow(x)),
      cloglog = sprintf("%.12g", stats::predict(ref, df, type = "cloglog")),
      logistic = sprintf("%.12g", stats::predict(ref, df, type = "logistic")),
      cloglog_out = sprintf("%.12g", stats::predict(ref, out, type = "cloglog")),
      logistic_out = sprintf("%.12g", stats::predict(ref, out, type = "logistic")),
      stringsAsFactors = FALSE)
    extra <- list(entropy = ref$entropy, alpha = ref$alpha, index = 200L, n_point = 200L)
  } else {
    mm <- stats::model.matrix(f, df)
    reg <- maxnet::maxnet.default.regularization(y, mm) * row$regmult
    w <- if (row$weighted) pen_w else rep(1, length(y))
    cv <- settled(glmnet::cv.glmnet(mm, y, family = "binomial", weights = w, standardize = FALSE,
                                    penalty.factor = reg, foldid = pen_fold + 1L,
                                    type.measure = "deviance",
                                    control = list(thresh = MX_THRESH, maxit = PEN_MAXIT)))
    k <- which(cv$lambda == cv$lambda.min)
    lambda <- cv$lambda.min
    a0 <- as.numeric(cv$glmnet.fit$a0[k])
    beta <- as.numeric(cv$glmnet.fit$beta[, k])
    y_fit <- y
    p <- stats::plogis(a0 + as.numeric(mm %*% beta))
    mx_pred[[i]] <- data.frame(case = row$case, row = seq_len(nrow(x)),
                               cloglog = "NA", logistic = sprintf("%.12g", p),
                               cloglog_out = "NA", logistic_out = "NA", stringsAsFactors = FALSE)
    extra <- list(entropy = NA_real_, alpha = NA_real_, index = k,
                  n_point = length(cv$lambda))
  }
  vp <- reg * length(reg) / sum(reg)
  wn <- w / sum(w)
  eta <- a0 + as.numeric(mm %*% beta)
  objective <- -sum(wn * (y_fit * eta - log1p(exp(eta)))) + lambda * sum(vp * abs(beta))
  mx_rows[[i]] <- data.frame(
    row, classes_used = used, n_row = nrow(mm), n_feature = ncol(mm),
    lambda = sprintf("%.12g", lambda), objective = sprintf("%.15g", objective),
    entropy = if (is.na(extra$entropy)) "NA" else sprintf("%.12g", extra$entropy),
    alpha = if (is.na(extra$alpha)) "NA" else sprintf("%.12g", extra$alpha),
    index = extra$index, n_point = extra$n_point, stringsAsFactors = FALSE)
  mx_reg[[i]] <- data.frame(case = row$case, feature = seq_along(reg),
                            reg = sprintf("%.17g", reg), stringsAsFactors = FALSE)
}
glmnet::glmnet.control(factory = TRUE)
write_fixture(cbind(do.call(rbind, mx_rows), thresh = sprintf("%.12g", MX_THRESH),
                    max_pass = sprintf("%.12g", PEN_MAXIT), reg_tolerance = 1e-12,
                    objective_tolerance = 1e-10, prediction_tolerance = 1e-4),
              "maxnet_cases.csv")
write_fixture(do.call(rbind, mx_reg), "maxnet_regularization.csv")
write_fixture(do.call(rbind, mx_pred), "maxnet_predict.csv")
cat("wrote", nrow(MX_CASES), "maxnet cases\n")
# The envelope.
#
# The reference is biomod2's own `bm_SRE()`, on the weekly columns maxnet's fixtures read and the
# three responses those carry: its bounds, read with `do.extrem = TRUE`, and its projection onto
# the fixture's rows, where a quantile of zero puts a presence exactly on a bound, and onto every
# reading scaled by 1.02. The bounds are arithmetic on the readings and are pinned to rounding.
if (!requireNamespace("biomod2", quietly = TRUE)) {
  stop("the envelope fixtures are the envelope biomod2 draws, so it has to be installed to ",
       "regenerate them.", call. = FALSE)
}
ENV_CASES <- data.frame(
  case = c("edges", "default", "default_12", "default_8", "tenth_12", "median"),
  response = c("y_binomial", "y_binomial", "y_12", "y_8", "y_12", "y_binomial"),
  quantile = c(0, 0.025, 0.025, 0.025, 0.1, 0.5),
  stringsAsFactors = FALSE)
env_rows <- list()
env_bounds <- list()
env_pred <- list()
for (i in seq_len(nrow(ENV_CASES))) {
  row <- ENV_CASES[i, ]
  y <- mx_y[[row$response]]
  df <- mx_frame(mx_x)
  bounds <- biomod2::bm_SRE(resp.var = y, expl.var = df, quant = row$quantile, do.extrem = TRUE)
  inside <- biomod2::bm_SRE(resp.var = y, expl.var = df, new.env = df, quant = row$quantile)
  outside <- biomod2::bm_SRE(resp.var = y, expl.var = df, new.env = mx_frame(mx_x * 1.02),
                             quant = row$quantile)
  env_rows[[i]] <- data.frame(row, n_presence = sum(y == 1), stringsAsFactors = FALSE)
  env_bounds[[i]] <- data.frame(case = row$case, column = seq_len(ncol(mx_x)),
                                lo = sprintf("%.17g", bounds[, 1L]),
                                hi = sprintf("%.17g", bounds[, 2L]), stringsAsFactors = FALSE)
  env_pred[[i]] <- data.frame(case = row$case, row = seq_len(nrow(mx_x)),
                              inside = as.integer(inside[, 1L]),
                              inside_out = as.integer(outside[, 1L]), stringsAsFactors = FALSE)
}
write_fixture(cbind(do.call(rbind, env_rows), bound_tolerance = 1e-12), "envelope_cases.csv")
write_fixture(do.call(rbind, env_bounds), "envelope_bounds.csv")
write_fixture(do.call(rbind, env_pred), "envelope_predict.csv")
cat("wrote", nrow(ENV_CASES), "envelope cases\n")

# The stepwise model.
#
# Three references, each written separately from the core. MASS's `stepAIC()` over biomod2's
# formula, a power of a column per term, for every direction a search runs: the terms in the order
# the final model holds them, its deviance and rank, and its fitted means. `glm()` on every term for
# the unselected fit. And the forward search over column terms in R alone,
# `tests/testthat/helper-oracle-stepwise.R`, which is the arm the published comparison ran, under
# fractional case weights. MASS reads the binomial family, whose criterion rounds fractional weights
# into counts, so its binomial cases are unweighted; its Gaussian criterion differs from the core's
# by a constant and orders the moves the same way under any weights. Two cases carry the first
# column twice, so a term is aliased: the backward search drops it before anything else and the
# two-way search never offers it.
source("tests/testthat/helper-oracle-stepwise.R")
if (!requireNamespace("MASS", quietly = TRUE)) {
  stop("the stepwise fixtures are MASS's own search, so it has to be installed to regenerate ",
       "them.", call. = FALSE)
}
sw_y <- c(mx_y, list(y_gaussian = pen_y_gaussian))
sw_case <- function(case, reference, response, direction, terms = "power", degree = 2L,
                    max_terms = Inf, weighted = FALSE, duplicate = FALSE) {
  data.frame(case = case, reference = reference, response = response, direction = direction,
             terms = terms, degree = degree, max_terms = max_terms, weighted = weighted,
             duplicate = duplicate, stringsAsFactors = FALSE)
}
SW_CASES <- rbind(
  sw_case("mass_both_cubic", "MASS", "y_binomial", "both", degree = 3L),
  sw_case("mass_both_12", "MASS", "y_12", "both"),
  sw_case("mass_forward_gaussian", "MASS", "y_gaussian", "forward"),
  sw_case("mass_both_gaussian_weighted", "MASS", "y_gaussian", "both", weighted = TRUE),
  sw_case("mass_backward", "MASS", "y_binomial", "backward"),
  sw_case("mass_backward_duplicate", "MASS", "y_binomial", "backward", duplicate = TRUE),
  sw_case("mass_both_duplicate", "MASS", "y_binomial", "both", duplicate = TRUE),
  sw_case("glm_none", "glm", "y_binomial", "none"),
  sw_case("glm_none_column_gaussian_weighted", "glm", "y_gaussian", "none", terms = "column",
          weighted = TRUE),
  sw_case("oracle_forward", "oracle", "y_binomial", "forward", terms = "column", max_terms = 3),
  sw_case("oracle_forward_weighted", "oracle", "y_binomial", "forward", terms = "column",
          max_terms = 3, weighted = TRUE),
  sw_case("oracle_forward_12_weighted", "oracle", "y_12", "forward", terms = "column",
          weighted = TRUE),
  sw_case("oracle_forward_gaussian_weighted", "oracle", "y_gaussian", "forward",
          terms = "column", max_terms = 6, weighted = TRUE))

sw_design <- function(duplicate) if (duplicate) cbind(mx_x, mx_x[, 1L]) else mx_x
sw_names <- function(x) sprintf("v%02d", seq_len(ncol(x)))
# A term's label in the order the design lays the terms out: each column's powers, column by
# column, which is the order biomod2's formula names them and the core catalogues them.
sw_upper <- function(x, degree) {
  labels <- unlist(lapply(sw_names(x), function(v) {
    c(v, if (degree >= 2L) sprintf("I(%s^%d)", v, seq(2L, degree)))
  }))
  stats::as.formula(paste("~", paste(labels, collapse = " + ")))
}
sw_term_key <- function(label, x) {
  power <- if (startsWith(label, "I(")) as.integer(sub(".*\\^([0-9]+)\\)$", "\\1", label)) else 1L
  name <- if (startsWith(label, "I(")) sub("^I\\((v[0-9]+)\\^.*$", "\\1", label) else label
  sprintf("%d:%d", match(name, sw_names(x)), power)
}

sw_rows <- list()
sw_pred <- list()
for (i in seq_len(nrow(SW_CASES))) {
  row <- SW_CASES[i, ]
  x <- sw_design(row$duplicate)
  y <- sw_y[[row$response]]
  w <- if (row$weighted) pen_w else rep(1, length(y))
  gaussian <- row$response == "y_gaussian"
  family <- if (gaussian) stats::gaussian() else stats::binomial()
  df <- stats::setNames(as.data.frame(x), sw_names(x))
  out <- stats::setNames(as.data.frame(x * 1.01), sw_names(x))
  df$y <- y
  df$w <- w
  if (row$reference == "oracle") {
    f <- oracle_forward_aic(x, y, row$max_terms, row$degree, if (gaussian) "gaussian" else
                              "binomial", w)
    chosen <- sprintf("%d:0", f$columns)
    fitted <- oracle_predict_forward(f, x)
    fitted_out <- oracle_predict_forward(f, x * 1.01)
    deviance <- f$fit$deviance
    rank <- f$fit$rank
    converged <- f$fit$converged
    steps <- length(f$columns)
  } else {
    upper <- sw_upper(x, row$degree)
    full <- stats::update(upper, y ~ .)
    start <- if (row$direction %in% c("backward", "none")) full else y ~ 1
    g <- stats::glm(start, family = family, data = df, weights = w)
    if (row$reference == "MASS") {
      g <- MASS::stepAIC(g, scope = list(upper = upper, lower = ~1), direction = row$direction,
                         trace = FALSE, k = 2)
    }
    labels <- attr(stats::terms(g), "term.labels")
    chosen <- if (row$terms == "column") {
      sprintf("%d:0", unique(match(sub("^I\\((v[0-9]+)\\^.*$", "\\1", labels), sw_names(x))))
    } else vapply(labels, sw_term_key, character(1L), x = x)
    fitted <- as.numeric(stats::fitted(g))
    fitted_out <- as.numeric(stats::predict(g, out, type = "response"))
    deviance <- g$deviance
    rank <- g$rank
    converged <- g$converged
    steps <- if (row$reference == "MASS") nrow(g$anova) - 1L else 0L
  }
  sw_rows[[i]] <- data.frame(row, chosen = paste(chosen, collapse = " "), rank = rank,
                             deviance = sprintf("%.15g", deviance), converged = converged,
                             steps = steps, stringsAsFactors = FALSE)
  sw_pred[[i]] <- data.frame(case = row$case, row = seq_len(nrow(x)),
                             fitted = sprintf("%.15g", fitted),
                             fitted_out = sprintf("%.15g", fitted_out), stringsAsFactors = FALSE)
}
write_fixture(cbind(do.call(rbind, sw_rows), deviance_tolerance = 1e-9,
                    prediction_tolerance = 1e-8), "stepwise_cases.csv")
write_fixture(do.call(rbind, sw_pred), "stepwise_predict.csv")
cat("wrote", nrow(SW_CASES), "stepwise cases\n")
