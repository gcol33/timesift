#!/usr/bin/env Rscript
# Reproduce the Schrankogel grid with timesift.
#
#   Rscript schrankogel.R <deposit_dir> <out_dir> [options]
#
# <deposit_dir> is the unpacked data directory of the Chytry et al. deposit
# (doi:10.5281/zenodo.17047026), holding logger_data.csv, spe_wide.csv, seasons.csv and
# output_temperature_variables_scaled.csv. Each of the four is checked against the size and the
# MD5 sum of the deposit's copy before it is read. <out_dir> receives one CSV per stage and a
# run.meta beside them saying what every file was computed on.
#
# Options, each --name=value:
#   --stages    which stages to run, comma-separated: contract, representation, baseline,
#               networks, contrasts, grains, inflation. Default all but networks.
#   --grains    which grains the network grid covers. Default day,week,month,season,year, the
#               five coarse grains; native and halfday are 26,304 and 2,192 steps per plot and
#               want a graphics processor, as they had in the study.
#   --learners  which encoders the network grid covers, plus `ensemble` for the eleven-member
#               set, whose members run as arms of their own and whose held-out predictions are
#               averaged into one further arm. Default cnn.
#   --baseline  which aggregated-feature arms to fit: elastic_net, stepwise, or both. Default
#               elastic_net. Forward selection over 188 columns is one glm fit per candidate
#               column per step per species per fold and takes many hours single-threaded.
#   --folds     a CSV of logger_ID and fold. Default folds.csv beside this script, the study's
#               own map, which is what every number below was checked against; `build` draws a
#               map with fold_map() instead, a different partition of the same design.
#   --epochs    epoch budget per network fit. Default 60, the budget the study used.
#
# Nothing here caps the data: a stage either runs over all 894 plots and all 101 species or it
# does not run.

suppressMessages({
  library(timesift)
})

# ---- arguments -------------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) {
  stop("usage: Rscript schrankogel.R <deposit_dir> <out_dir> [--name=value ...]", call. = FALSE)
}
deposit <- args[1L]
out_dir <- args[2L]
here <- grep("^--file=", commandArgs(FALSE), value = TRUE)
here <- if (length(here)) dirname(normalizePath(sub("^--file=", "", here[1L]))) else getwd()
opt <- local({
  named <- grep("^--", args, value = TRUE)
  keys <- sub("^--([^=]+)=.*$", "\\1", named)
  stats::setNames(as.list(sub("^--[^=]+=", "", named)), keys)
})
pick <- function(name, default) if (is.null(opt[[name]])) default else opt[[name]]
split_opt <- function(name, default) {
  strsplit(pick(name, default), ",", fixed = TRUE)[[1L]]
}

stages <- split_opt("stages", "contract,representation,baseline,contrasts,inflation")
grid_grains <- split_opt("grains", "day,week,month,season,year")
grid_learners <- split_opt("learners", "cnn")
baseline_arms <- split_opt("baseline", "elastic_net")
epochs <- as.integer(pick("epochs", "60"))
folds_from <- pick("folds", file.path(here, "folds.csv"))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

say <- function(...) cat(format(Sys.time(), "[%H:%M:%S] "), ..., "\n", sep = "")
assert_equal <- function(what, got, expected) {
  say(sprintf("%-38s %s", paste0(what, ":"), format(got)))
  if (!isTRUE(all.equal(got, expected))) {
    stop(what, " is ", format(got), ", the deposit gives ", format(expected),
         ". The input is not what this script was written against.", call. = FALSE)
  }
}
# Every file written carries its identity beside it: run.meta names the deposit, the package and
# the fold map every number was computed on, and each CSV is listed there with its own sum as it
# is written, so a table on disk can be traced to an input and a build rather than to a name.
meta_path <- file.path(out_dir, "run.meta")
meta <- function(...) cat(..., "\n", sep = "", file = meta_path, append = TRUE)
write_out <- function(x, name) {
  path <- file.path(out_dir, name)
  utils::write.csv(x, path, row.names = FALSE)
  meta("wrote ", name, " md5 ", unname(tools::md5sum(path)), " at ",
       format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"))
  say("wrote ", path)
  invisible(path)
}

# The deposit as this script was written against it: the record's DOI, the sum of the archive
# the four files were unpacked from, and each file's own size and sum, read from a copy whose
# archive sum matched the Zenodo record. A file that differs is refused before it is read, so a
# re-deposited version of the same shape cannot pass as the one the numbers below were checked on.
DEPOSIT <- list(
  doi = "10.5281/zenodo.17047026",
  archive = "data.zip",
  archive_md5 = "777723c82f054d039903f6c3270d8006",
  files = data.frame(
    file = c("spe_wide.csv", "seasons.csv", "output_temperature_variables_scaled.csv",
             "logger_data.csv"),
    bytes = c(459314, 26309, 3303659, 1272670373),
    md5 = c("dfbb2bc3d8c7a8f9e7acecdbdd9e863d", "eac2a6678fa4cbe93c18981332bb469c",
            "2160c2ce2bd01bf375143bb74855dbd2", "626d583d326545e8627afb3a54891ca2"),
    stringsAsFactors = FALSE))

deposit_file <- function(name) {
  path <- file.path(deposit, name)
  expected <- DEPOSIT$files[match(name, DEPOSIT$files$file), ]
  if (!file.exists(path)) {
    stop(name, " is not in ", deposit, ". The deposit's data directory holds it.", call. = FALSE)
  }
  bytes <- file.size(path)
  if (bytes != expected$bytes) {
    stop(name, " is ", bytes, " bytes; the deposit's is ", expected$bytes,
         ". The input is not the deposit this script was written against.", call. = FALSE)
  }
  sum <- unname(tools::md5sum(path))
  if (!identical(sum, expected$md5)) {
    stop(name, " has md5 ", sum, "; the deposit's is ", expected$md5,
         ". The input is not the deposit this script was written against.", call. = FALSE)
  }
  say(sprintf("%-38s %s", paste0(name, ":"), sum))
  meta("input ", name, " md5 ", sum, " bytes ", bytes)
  path
}

cat("", file = meta_path)
meta("timesift reproduction of doi:", DEPOSIT$doi)
meta("deposit archive ", DEPOSIT$archive, " md5 ", DEPOSIT$archive_md5)
meta("deposit_dir ", normalizePath(deposit))
meta("started ", format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"))
meta("timesift ", as.character(utils::packageVersion("timesift")))
meta("R ", paste(R.version$major, R.version$minor, sep = "."), " on ", R.version$platform)
meta("stages ", paste(stages, collapse = ","))
meta("folds ", if (identical(folds_from, "build")) "built by fold_map()" else folds_from)

# The contract of Chytry et al.: species in at least 25 plots, then five aggregate taxa removed.
MIN_OCCURRENCES <- 25L
DROP_TAXA <- c("Alchemilla vulgaris agg.", "Taraxacum sp.", "Festuca halleri agg.",
               "Euphrasia sp.", "Phleum alpinum agg.")
CV_FOLDS <- 10L
CV_SEED <- 1L
REPORTED_STATS <- c("cold_day", "mean", "warm_day")
METRIC_NAME <- "tss"

# ---- the response, the folds and the cells ----------------------------------------------------

say("reading ", file.path(deposit, "spe_wide.csv"))
spe <- utils::read.csv(deposit_file("spe_wide.csv"), check.names = FALSE)
rownames(spe) <- as.character(spe$logger_ID)
counts <- colSums(spe[setdiff(names(spe), "logger_ID")])
keep <- setdiff(names(counts)[counts >= MIN_OCCURRENCES], DROP_TAXA)
y <- as.matrix(spe[, keep, drop = FALSE])

assert_equal("plots", nrow(y), 894L)
assert_equal("species after the contract filter", ncol(y), 101L)
assert_equal("rarest retained species", min(colSums(y)), 26)

folds <- if (!identical(folds_from, "build")) {
  say("reading the fold map from ", folds_from)
  f <- utils::read.csv(folds_from)
  meta("folds md5 ", unname(tools::md5sum(folds_from)))
  stats::setNames(as.integer(f$fold), as.character(f$logger_ID))
} else {
  say("building a fold map: ", CV_FOLDS, " folds, seed ", CV_SEED, ", richness quintiles")
  fold_map(y, v = CV_FOLDS, seed = CV_SEED, strata = 5L)
}

cells <- scorable_cells(y, folds)
assert_equal("cells", nrow(cells), 1010L)
say(sprintf("%-38s %d (%.1f%%)", "scorable cells:", sum(cells$scorable),
            100 * mean(cells$scorable)))
say(sprintf("%-38s %d of %d", "species with a scorable fold:",
            sum(tapply(cells$scorable, cells$variable, any)), ncol(y)))
write_out(cells, "cells.csv")

# The deposit cuts its seasons at the equinoxes and the solstices rather than on the first of a
# month, and labels every date of the record in seasons.csv. Reading that file back as a binning
# function is how the season rung of the ladder is the deposit's season rather than a quarter.
astronomical_seasons <- function(path) {
  labels <- utils::read.csv(path)
  key <- paste(labels$season, format(as.Date(labels$day), "%Y"))
  edges <- as.POSIXct(paste0(labels$day[!duplicated(key)], " 00:00:00"), tz = "UTC")
  edges <- sort(edges)
  function(when) edges[findInterval(as.numeric(when), as.numeric(edges))]
}

EXPECTED_BINS <- c(native = 26304L, halfday = 2192L, day = 1096L, week = 157L, month = 36L,
                   season = 13L, year = 3L)

# The record and its calendar are read only by the stages that bin it, so a run of the contract
# alone needs neither the 1.2 GB file nor seasons.csv.
if (!"representation" %in% stages && !"networks" %in% stages) {
  readings <- NULL
  BINNING <- NULL
} else {
  say("reading ", file.path(deposit, "logger_data.csv"), " (1.2 GB, a few minutes)")
  readings <- utils::read.csv(
    deposit_file("logger_data.csv"),
    colClasses = c(logger_ID = "character", date = "character", logger_serial_number = "NULL",
                   temp = "numeric", day = "NULL", month = "NULL"))
  readings$date <- as.POSIXct(readings$date, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  assert_equal("readings", nrow(readings), 894L * 26304L)
  assert_equal("readings per plot", nrow(readings) / length(unique(readings$logger_ID)), 26304)
  BINNING <- list(native = "native", halfday = "halfday", day = "day", week = "week",
                  month = "month", season = astronomical_seasons(deposit_file("seasons.csv")),
                  year = "year")
}

build <- function(grain, stats) {
  x <- grain_matrix(readings, logger_ID, date, temp, grain = BINNING[[grain]], stats = stats)
  assert_equal(paste(grain, "bins"), dim(x)[2L], EXPECTED_BINS[[grain]])
  x
}

if ("representation" %in% stages) {
  say("building the representation at every grain")
  shape <- lapply(names(BINNING), function(w) {
    x <- build(w, "mean")
    data.frame(grain = w, bins = dim(x)[2L], readings_per_bin = mean(attr(x, "bin_n")),
               numbers_per_plot = dim(x)[2L], stringsAsFactors = FALSE)
  })
  shape <- do.call(rbind, shape)
  # The reported reading is three channels of whole days, so a week of it is three numbers.
  shape$numbers_per_plot_reported <- ifelse(shape$grain %in% c("native", "halfday"), NA_integer_,
                                            3L * shape$bins)
  write_out(shape, "representation.csv")
}

# ---- the aggregated-feature arms --------------------------------------------------------------

if ("baseline" %in% stages) {
  say("reading the deposit's aggregated temperature features")
  agg <- utils::read.csv(deposit_file("output_temperature_variables_scaled.csv"),
                         check.names = FALSE)
  rownames(agg) <- as.character(agg$logger_ID)
  agg <- as.matrix(agg[rownames(y), setdiff(names(agg), "logger_ID"), drop = FALSE])
  assert_equal("aggregated temperature variables", ncol(agg), 188L)

  features <- feature_matrix(agg, label = "aggregates")
  say("fitting the aggregated-feature arms, selection redone inside every fold")
  arms <- list(
    elastic_net = elasticnet(alpha = 0.5, n_inner = 5L, squares = TRUE, seed = CV_SEED),
    stepwise = stepwise(max_terms = 3L, degree = 2L))[baseline_arms]
  baseline <- grain_ladder(features, y, arms, folds = folds, metric = METRIC_NAME)
  write_out(baseline, "baseline.csv")
  print(summary(baseline))
}

# ---- the network grid ---------------------------------------------------------------------

# The member set as arms of the run, and their held-out predictions averaged into one further arm
# per grain. A member's out-of-fold prediction on a fold is its held-out prediction there, so
# averaging the eleven and then choosing a threshold is the set scored as one model rather than as
# a vote between eleven decisions.
ensemble_arm <- function(set, y, folds, members) {
  lad <- grain_ladder(set, y, members, folds = folds, metric = METRIC_NAME)
  oof <- attr(lad, "predictions")
  cells <- attr(lad, "cells")
  combined <- lapply(names(set), function(w) {
    arms <- paste(w, names(members), sep = "|")
    stack <- ensemble_fit(oof[arms], y, cells, folds, spec = ensemble("mean"))
    cbind(grain = w, learner = "ensemble",
          score_predictions(y, ensemble_combine(stack, oof[arms]), folds, cells,
                            METRIC_NAME), stringsAsFactors = FALSE)
  })
  rbind(as.data.frame(lad), do.call(rbind, combined))
}

if ("networks" %in% stages) {
  # The eleven-member set of the study spans two architectures, three widths and three seeds, and
  # is trained with weight averaging. Members were chosen on inner-validation strength and on
  # architectural diversity, never on the held-out folds.
  members <- c(
    lapply(list(c(16L, 32L, 64L, 128L), c(32L, 64L, 128L, 256L), c(16L, 32L, 64L)),
           function(ch) cnn(channels = ch, epochs = epochs, batch_size = 32L, swa = TRUE)),
    lapply(c(5L, 7L, 9L),
           function(k) cnn(kernel = k, epochs = epochs, batch_size = 32L, swa = TRUE)),
    lapply(c(1L, 2L, 3L),
           function(sd) cnn(epochs = epochs, batch_size = 32L, swa = TRUE, seed = sd)),
    lapply(c(1L, 2L),
           function(sd) rescnn(epochs = epochs, batch_size = 32L, swa = TRUE, seed = sd)))
  names(members) <- sprintf("m%02d", seq_along(members))

  encoders <- list(mlp = mlp(epochs = epochs),
                   cnn = cnn(epochs = epochs, batch_size = 32L),
                   rescnn = rescnn(epochs = epochs, batch_size = 32L))[intersect(grid_learners,
                                                               c("mlp", "cnn", "rescnn"))]
  for (statistic in c("mean", "extremeday")) {
    grains <- if (statistic == "mean") grid_grains else
      intersect(grid_grains, c("week", "month", "season", "year"))
    if (!length(grains)) {
      next
    }
    stats_used <- if (statistic == "mean") "mean" else REPORTED_STATS
    say("network grid on the ", statistic, " reading: ", paste(grains, collapse = ", "))
    set <- timesift_set(stats::setNames(
      lapply(grains, function(w) {
        x <- build(w, stats_used)
        bind_channels(x, calendar_channels(x))
      }), grains))
    rows <- list()
    if (length(encoders)) {
      rows$encoders <- as.data.frame(
        grain_ladder(set, y, encoders, folds = folds, metric = METRIC_NAME, keep_fits = FALSE))
    }
    if ("ensemble" %in% grid_learners) {
      rows$ensemble <- ensemble_arm(set, y, folds, members)
    }
    grid <- structure(do.call(rbind, rows), class = c("timesift_ladder", "data.frame"),
                      metric = METRIC_NAME, response = "presence_absence")
    write_out(grid, paste0("networks_", statistic, ".csv"))
    print(summary(grid))
  }
}

# ---- the contrasts every claim is made on -----------------------------------------------------

if ("contrasts" %in% stages) {
  parts <- list.files(out_dir, pattern = "^(baseline|networks_)", full.names = TRUE)
  if (length(parts) < 2L) {
    say("contrasts need at least two result files in ", out_dir, "; skipping")
  } else {
    ladder <- do.call(rbind, lapply(parts, utils::read.csv, stringsAsFactors = FALSE))
    ladder <- structure(ladder, class = c("timesift_ladder", "data.frame"),
                        metric = METRIC_NAME, response = "presence_absence")
    arms <- unique(paste(ladder$grain, ladder$learner, sep = "|"))
    pairs <- utils::combn(arms, 2L, simplify = FALSE)
    out <- do.call(rbind, lapply(pairs, function(p) paired_contrast(ladder, p[1L], p[2L])))
    write_out(out[order(-out$diff), ], "contrasts.csv")
  }
}

# ---- each grain against its architecture's best ----------------------------------------------

if ("grains" %in% stages) {
  parts <- list.files(out_dir, pattern = "^networks_mean", full.names = TRUE)
  if (!length(parts)) {
    say("the grain contrast needs networks_mean.csv in ", out_dir, "; skipping")
  } else {
    ladder <- utils::read.csv(parts[1L], stringsAsFactors = FALSE)
    ladder <- structure(ladder, class = c("timesift_ladder", "data.frame"),
                        metric = METRIC_NAME, response = "presence_absence")
    out <- do.call(rbind, lapply(unique(ladder$learner), function(l)
      grain_contrasts(ladder, learner = l)))
    out$p_bh <- stats::p.adjust(out$p_value, method = "BH")
    write_out(out, "grain_contrasts.csv")
    print(out)
  }
}

# ---- what the reported level is an upper bound on ---------------------------------------------

if ("inflation" %in% stages) {
  say("measuring how much the self-selected threshold inflates a level on this design")
  out <- tss_inflation(y, folds, skill = c(0.6, 0.7, 0.9), replicates = 2000L, seed = CV_SEED)
  write_out(out, "inflation.csv")
  print(out)

  # What a level actually read is consistent with, which is the only reading of a level that is
  # about the population rather than about the scoring rule.
  levels_read <- list.files(out_dir, pattern = "^(baseline|networks_)", full.names = TRUE)
  if (length(levels_read)) {
    ladder <- do.call(rbind, lapply(levels_read, utils::read.csv, stringsAsFactors = FALSE))
    ladder <- structure(ladder, class = c("timesift_ladder", "data.frame"),
                        metric = METRIC_NAME, response = "presence_absence")
    reported <- summary(ladder)
    back <- implied_skill(y, folds, observed = reported$score, replicates = 500L, seed = CV_SEED)
    write_out(cbind(reported[c("learner", "grain")], back), "implied_skill.csv")
  }
}

meta("finished ", format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"))
say("done")
