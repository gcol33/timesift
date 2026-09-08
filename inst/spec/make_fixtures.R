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
  con <- file(file.path(out_dir, file), open = "wb")
  on.exit(close(con), add = TRUE)
  write.csv(x, con, row.names = FALSE, quote = FALSE, eol = "\n")
}

# Two series, because a record that starts on a bin boundary cannot tell two binning rules apart.
# The first begins at midnight on the default year_start, so every coarse grain is in phase with
# it from the first reading. The second begins at an arbitrary hour of an arbitrary day, which is
# what a logger deployed when someone could walk to it gives, and puts every grain out of phase.
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

# Two calendars that break what a supplied one has to satisfy, each named so both suites build the
# same function. `late` gives every reading the midnight after it, which is what a calendar shifted
# by one boundary returns and what an interval lookup wraps to below its first edge; `alternate`
# sends consecutive readings to two bins an hour apart, which interleaves them.
BAD_CALENDARS <- list(
  late = function(when) {
    as.POSIXct((floor(as.numeric(when) / 86400) + 1) * 86400, origin = "1970-01-01", tz = "UTC")
  },
  alternate = function(when) {
    t0 <- min(as.numeric(when))
    as.POSIXct(t0 + 3600 * (((as.numeric(when) - t0) / 3600) %% 2), origin = "1970-01-01",
               tz = "UTC")
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
  for (stats in list("mean", c("cold_day", "mean", "warm_day"))) {
    add(name, "astronomical", stats)
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
add("aligned", "day", c("min", "mean", "max"), tz = "Europe/Vienna")
add("zoned", "day", "mean")
for (w in c("native", "halfday", "day", "week")) {
  add("zoned", w, "mean", tz = "America/Sao_Paulo")
}
add("zoned", "day", c("min", "mean", "max"), tz = "America/Sao_Paulo")
add("zoned", "day", c("cold_day", "mean", "warm_day"), tz = "America/Sao_Paulo")
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
  list(series = "offset", calendar = "alternate", message = "interleaves two of its bins")
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
  tss = tss, roc_auc = roc_auc,
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
QUANTITIES <- c("diff", "lower", "upper", "n_variable", "n_cell", "n_favour", "p_value")
contrast <- paired_contrast(as_ladder(contrast_cells), "week|a", "week|b")
write_fixture(
  data.frame(quantity = QUANTITIES,
             value = vapply(QUANTITIES, function(nm) sprintf("%.12g", contrast[[nm]]),
                            character(1L)),
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
