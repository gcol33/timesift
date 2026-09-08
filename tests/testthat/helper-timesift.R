# The contract's fixtures, read from the installed package where there is one and from the
# source tree otherwise, so the suite asserts them on an installed copy too.
fixture_dir <- function() {
  installed <- system.file("spec", "fixtures", package = "timesift")
  candidates <- c(installed, "../../inst/spec/fixtures", "inst/spec/fixtures")
  for (p in candidates) {
    if (nzchar(p) && file.exists(file.path(p, "digests.csv"))) {
      return(p)
    }
  }
  NULL
}

# A synthetic record whose units differ by a level and by what one stretch of the calendar did,
# so a test can ask a fitted model which of the two it read.
sim_series <- function(n_unit = 40L, days = 120L, sd = 0.5, level = 2, seed = 1L,
                       from = "2021-09-01") {
  set.seed(seed)
  t <- seq(as.POSIXct(from, tz = "UTC"), by = "hour", length.out = 24L * days)
  units <- sprintf("p%03d", seq_len(n_unit))
  warmth <- stats::rnorm(n_unit)
  value <- as.numeric(vapply(warmth, function(w) {
    level * w + 5 * sin(seq_along(t) / (24 * 30)) + stats::rnorm(length(t), sd = sd)
  }, numeric(length(t))))
  list(units = units, warmth = warmth,
       readings = data.frame(plot = rep(units, each = length(t)),
                             t = rep(t, times = n_unit),
                             temp = value, stringsAsFactors = FALSE))
}

sim_response <- function(sim, strength = 3, n_var = 2L, seed = 2L) {
  set.seed(seed)
  y <- vapply(seq_len(n_var), function(j) {
    stats::rbinom(length(sim$warmth), 1L, stats::plogis(strength * sim$warmth * (-1)^(j + 1L)))
  }, numeric(length(sim$warmth)))
  dimnames(y) <- list(sim$units, paste0("sp", seq_len(n_var)))
  y
}

# The statistic every threshold metric is checked against: every cut, written out.
brute_tss <- function(y, p) {
  cuts <- sort(unique(p))
  max(vapply(cuts, function(c) {
    d <- as.integer(p >= c)
    sum(d == 1L & y == 1L) / sum(y == 1L) + sum(d == 0L & y == 0L) / sum(y == 0L) - 1
  }, numeric(1L)))
}

# A directory that lives for one test. withr would do this and is not worth an entry in Suggests
# for three lines; test_that() runs its block in a function, so on.exit() there is the whole
# lifetime.
temp_dir <- function() {
  dir <- tempfile("timesift")
  dir.create(dir)
  dir
}
