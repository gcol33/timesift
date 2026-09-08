# The representation as it was written in R alone, before the two languages shared a core.
#
# Nothing dispatches to this and the package never reaches it at runtime. It is kept because the
# Python side was written from `inst/spec/representation.md` rather than from the R source, and one
# shared binary would otherwise make the agreement between the two languages trivially true. One
# implementation in production, two in evidence.
#
# It reads the calendar by writing local strings and parsing them back, so it answers only for a
# series in UTC; that is what the tests hand it.

oracle_group_sum <- function(values, cell, n_cell) {
  s <- rowsum(values, cell, reorder = TRUE)
  out <- numeric(n_cell)
  out[sort(unique(cell))] <- as.vector(s)
  out
}

oracle_group_edge <- function(values, cell, n_cell, upper) {
  o <- order(cell, values, method = "radix")
  g <- cell[o]
  keep <- !duplicated(g, fromLast = upper)
  out <- rep(NA_real_, n_cell)
  out[g[keep]] <- values[o][keep]
  out
}

# The day-level stage: reduce every (unit, calendar day) to its own mean, minimum and maximum, and
# say which bin cell each of those days belongs to. The four day-level statistics are then a second
# reduction over the days of a bin, which is what keeps an extreme day distinct from an extreme
# reading.
oracle_day_level <- function(reading, unit, when, units, bins, ys, tz, n_cell) {
  day_start <- oracle_bin_start(when, "day", ys, tz)
  n_u <- length(units)
  days <- sort(unique(day_start))
  dcell <- (match(unclass(day_start), unclass(days)) - 1L) * n_u + match(unit, units)
  n_dcell <- n_u * length(days)

  count <- tabulate(dcell, nbins = n_dcell)
  present <- which(count > 0L)

  day_unit <- ((present - 1L) %% n_u) + 1L
  day_of <- ((present - 1L) %/% n_u) + 1L
  bin_of <- findInterval(as.numeric(days[day_of]), as.numeric(bins))
  cell <- (bin_of - 1L) * n_u + day_unit

  list(cell = cell,
       mean = oracle_group_sum(reading, dcell, n_dcell)[present] / count[present],
       min = oracle_group_edge(reading, dcell, n_dcell, FALSE)[present],
       max = oracle_group_edge(reading, dcell, n_dcell, TRUE)[present],
       n_day = tabulate(cell, nbins = n_cell))
}

# A record of many units shares its reading instants across them, and binning is a function of the
# instant alone, so the calendar is read once per distinct instant rather than once per reading.
# On three years of hourly readings from 894 units that is 26,304 calendar lookups instead of
# 23,515,776, and the difference is minutes.
oracle_bin_start <- function(when, grain, ys, tz) {
  u <- unique(when)
  if (length(u) == length(when)) {
    return(oracle_bin_of(when, grain, ys, tz))
  }
  oracle_bin_of(u, grain, ys, tz)[match(unclass(when), unclass(u))]
}

oracle_bin_of <- function(when, grain, ys, tz) {
  if (is.function(grain)) {
    out <- grain(when)
    if (!inherits(out, "POSIXct") || length(out) != length(when)) {
      stop("a `grain` function must return one POSIXct bin start per reading.", call. = FALSE)
    }
    return(out)
  }
  if (grain == "native") {
    return(when)
  }
  if (grain == "halfday") {
    day <- as.POSIXct(trunc(when, units = "days"), tz = tz)
    return(day + 43200 * (as.integer(format(when, "%H", tz = tz)) >= 12L))
  }
  if (grain == "day") {
    return(as.POSIXct(trunc(when, units = "days"), tz = tz))
  }
  if (grain == "week") {
    day <- as.Date(when, tz = tz)
    monday <- day - (as.integer(format(day, "%u")) - 1L)
    return(as.POSIXct(paste0(monday, " 00:00:00"), tz = tz))
  }
  if (grain == "month") {
    return(as.POSIXct(paste0(format(when, "%Y-%m", tz = tz), "-01 00:00:00"), tz = tz))
  }
  step <- if (grain == "season") 3L else 12L
  oracle_anniversary(oracle_offset_months(when, ys, tz) %/% step * step, ys, tz)
}

# Which bins the record does not cover for their whole calendar span. The record covers from its
# first reading to its last plus one sampling interval, and a bin is partial when its own span
# reaches outside that. Only a bin at an end of the record can, because .check_grid() has already
# required every unit to hold readings in every bin between them.
oracle_bin_partial <- function(when, bins, grain, ys, tz) {
  covered <- range(as.numeric(when))
  covered[2L] <- covered[2L] + oracle_sampling_step(when)
  as.numeric(bins) < covered[1L] |
    as.numeric(oracle_bin_next(bins, grain, ys, tz, covered[2L])) > covered[2L]
}

oracle_sampling_step <- function(when) {
  u <- sort(unique(as.numeric(when)))
  if (length(u) < 2L) 0 else min(diff(u))
}

# Where each bin ends, which is where the next one on the same calendar starts. The four coarse
# grains step by the calendar rather than by a count of seconds, so the successor is taken by
# landing well inside the following bin and flooring that, which is exact whatever the month length
# or the daylight-saving offset. A caller-supplied binning declares its own bins, so its successors
# are read off the bins themselves and its last bin is taken to end with the record.
oracle_bin_next <- function(bins, grain, ys, tz, covered_end) {
  if (is.function(grain) || identical(grain, "native")) {
    return(c(bins[-1L], .POSIXct(covered_end, tz = tz)))
  }
  switch(grain,
         halfday = bins + 43200,
         day = oracle_bin_of(bins + 36 * 3600, "day", ys, tz),
         week = oracle_bin_of(bins + 180 * 3600, "week", ys, tz),
         month = oracle_bin_of(bins + 40 * 86400, "month", ys, tz),
         season = oracle_anniversary(oracle_offset_months(bins, ys, tz) + 3L, ys, tz),
         year = oracle_anniversary(oracle_offset_months(bins, ys, tz) + 12L, ys, tz))
}

oracle_offset_months <- function(when, ys, tz) {
  y <- as.integer(format(when, "%Y", tz = tz))
  m <- as.integer(format(when, "%m", tz = tz))
  d <- as.integer(format(when, "%d", tz = tz))
  y * 12L + (m - 1L) - (ys$month - 1L) - as.integer(d < ys$day)
}

oracle_anniversary <- function(offset, ys, tz) {
  absolute <- offset + (ys$month - 1L)
  as.POSIXct(sprintf("%04d-%02d-%02d 00:00:00",
                     absolute %/% 12L, absolute %% 12L + 1L, ys$day),
             tz = tz)
}

# The lookback, from the section of `inst/spec/representation.md` that describes it rather
# than from the C++ underneath. It reads whole seconds and knows no zone, so it answers for a series
# already expressed in the calendar to bin by; that is what the tests hand it.

oracle_duration <- function(x) {
  if (is.numeric(x)) {
    return(as.numeric(x))
  }
  size <- c(second = 1, minute = 60, hour = 3600, day = 86400, week = 604800,
            month = 2592000, year = 31536000)
  parts <- strsplit(trimws(x), " +")[[1L]]
  if (length(parts) == 1L) {
    return(as.numeric(parts))
  }
  as.numeric(parts[1L]) * unname(size[[sub("s$", "", tolower(parts[2L]))]])
}

oracle_bin_offsets <- function(span, lag, bins) {
  step <- span / bins
  size <- c(day = 86400, hour = 3600, minute = 60, second = 1)
  vapply(seq_len(bins) - 1L, function(b) {
    x <- b * step - lag - span
    unit <- names(size)[which(x %% size == 0)[1L]]
    n <- x / size[[unit]]
    paste0(n, " ", unit, if (abs(n) == 1) "" else "s")
  }, character(1L))
}

# The seven statistics over the readings of one cell. The four day-level ones reduce each calendar
# day first and reduce again over the days of the cell, oldest first, which is what keeps an
# extreme day distinct from an extreme reading.
oracle_cell_stat <- function(name, v, t) {
  if (name == "mean") return(mean(v))
  if (name == "min") return(min(v))
  if (name == "max") return(max(v))
  key <- floor(t / 86400)
  day <- split(v, factor(key, levels = unique(key)))
  switch(name,
         cold_day = min(vapply(day, mean, numeric(1L))),
         warm_day = max(vapply(day, mean, numeric(1L))),
         mean_daily_min = mean(vapply(day, min, numeric(1L))),
         mean_daily_max = mean(vapply(day, max, numeric(1L))))
}

oracle_lookback_matrix <- function(data, id, time, value, at, span, lag = "0 days", bins = 1L,
                                 stats = "mean") {
  span <- oracle_duration(span)
  lag <- oracle_duration(lag)
  step <- span / bins
  unit <- as.character(data[[id]])
  when <- as.numeric(data[[time]])
  reading <- as.numeric(data[[value]])
  who <- as.character(at$id)
  anchor <- as.numeric(at$at)
  n_t <- length(anchor)

  # A day-level statistic is defined only where every calendar day lies whole inside one bin,
  # which for a lookback is a step of whole days and a lookback opening on a day boundary.
  if (any(stats %in% c("cold_day", "warm_day", "mean_daily_min", "mean_daily_max"))) {
    if (step %% 86400 != 0) {
      stop("needs bins of a calendar day or coarser", call. = FALSE)
    }
    off <- which((anchor - lag - span) %% 86400 != 0)
    if (length(off)) {
      stop("needs bins that open on a day boundary: target ", off[1L], call. = FALSE)
    }
  }

  out <- array(NA_real_, dim = c(n_t, bins, length(stats)),
               dimnames = list(as.character(seq_len(n_t)),
                               oracle_bin_offsets(span, lag, bins), stats))
  count <- matrix(0L, nrow = n_t, ncol = bins)

  for (i in seq_len(n_t)) {
    open <- anchor[i] - lag - span
    inside <- unit == who[i] & when >= open & when < open + span
    t <- when[inside]
    v <- reading[inside]
    o <- order(t, method = "radix")
    t <- t[o]
    v <- v[o]
    b <- floor((t - open) / step) + 1L
    for (k in seq_len(bins)) {
      take <- b == k
      if (!any(take)) {
        stop("(target, bin) cell holds no readings, first: target ", i, call. = FALSE)
      }
      count[i, k] <- sum(take)
      out[i, k, ] <- vapply(stats, oracle_cell_stat, numeric(1L), v = v[take], t = t[take])
    }
  }
  list(values = out, bin_n = count)
}

oracle_grain_matrix <- function(data, id, time, value, grain = "day", stats = "mean",
                                 year_start = "09-01") {
  unit <- as.character(data[[id]])
  when <- data[[time]]
  reading <- as.numeric(data[[value]])
  tz <- attr(when, "tzone")
  if (is.null(tz) || !nzchar(tz)) tz <- "UTC"
  ys <- list(month = as.integer(substr(year_start, 1L, 2L)),
             day = as.integer(substr(year_start, 4L, 5L)))

  bin_start <- oracle_bin_start(when, grain, ys, tz)
  units <- sort(unique(unit), method = "radix")
  bins <- sort(unique(bin_start))
  n_u <- length(units)
  n_b <- length(bins)
  n_cell <- n_u * n_b

  bin_of <- match(unclass(bin_start), unclass(bins))
  cell <- (bin_of - 1L) * n_u + match(unit, units)
  count <- tabulate(cell, nbins = n_cell)

  out <- array(NA_real_,
               dim = c(n_u, n_b, length(stats)),
               dimnames = list(units, format(bins, "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"), stats))
  day <- if (any(stats %in% c("cold_day", "warm_day", "mean_daily_min", "mean_daily_max"))) {
    oracle_day_level(reading, unit, when, units, bins, ys, tz, n_cell)
  } else {
    NULL
  }

  for (k in seq_along(stats)) {
    out[, , k] <- switch(
      stats[k],
      mean = matrix(oracle_group_sum(reading, cell, n_cell) / count, nrow = n_u, ncol = n_b),
      min = matrix(oracle_group_edge(reading, cell, n_cell, FALSE), nrow = n_u, ncol = n_b),
      max = matrix(oracle_group_edge(reading, cell, n_cell, TRUE), nrow = n_u, ncol = n_b),
      cold_day = matrix(oracle_group_edge(day$mean, day$cell, n_cell, FALSE),
                        nrow = n_u, ncol = n_b),
      warm_day = matrix(oracle_group_edge(day$mean, day$cell, n_cell, TRUE),
                        nrow = n_u, ncol = n_b),
      mean_daily_min = matrix(oracle_group_sum(day$min, day$cell, n_cell) / day$n_day,
                              nrow = n_u, ncol = n_b),
      mean_daily_max = matrix(oracle_group_sum(day$max, day$cell, n_cell) / day$n_day,
                              nrow = n_u, ncol = n_b)
    )
  }

  list(values = out,
       bin_start = bins,
       bin_end = .POSIXct(oracle_group_edge(as.numeric(when), bin_of, n_b, TRUE), tz = tz),
       bin_n = matrix(count, nrow = n_u, ncol = n_b),
       bin_partial = oracle_bin_partial(when, bins, grain, ys, tz))
}
