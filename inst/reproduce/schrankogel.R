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
#               networks, selection, contrasts, grains, inflation. Default all but networks and
#               selection, the two that fit encoders.
#   --grains    which grains the network grid covers. Default day,week,month,season,year, the
#               five coarse grains; native and halfday are 26,304 and 2,192 steps per plot and
#               want a graphics processor, as they had in the study.
#   --learners  which encoders the network grid covers, plus `ensemble` for the eleven-member
#               set, whose members run as arms of their own and whose held-out predictions are
#               averaged into one further arm. Default cnn.
#   --baseline  which arms to fit: elastic_net on the deposit's 188 aggregated variables,
#               stepwise on the same, series for the penalised fit on the weekly coldest-day,
#               mean and warmest-day reading of the record itself, or several. Default
#               elastic_net. Forward selection over 188 columns is one glm fit per candidate
#               column per step per species per fold and takes many hours single-threaded.
#   --folds     a CSV of logger_ID and fold. Default folds.csv beside this script, the study's
#               own map, which is what every number below was checked against; `build` draws a
#               map with fold_map() instead, a different partition of the same design.
#   --inner     a CSV of outer_fold, logger_ID and inner_fold. Default inner_folds.csv beside
#               this script, the study's own inner partition of each outer training set, which
#               the selection stage chooses a candidate on; `build` deals five inner folds per
#               outer training set with fold_map() instead.
#   --epochs    epoch budget per network fit. Default 60, the budget the study used.
#   --smoke     `<outer folds>,<species>`, e.g. `1,4`. A smoke run: it names every file it writes
#               `smoke_`, records itself as a smoke run in run.meta, and makes no comparison
#               against the study's numbers. It exists to see the stages run before an overnight
#               one, and its output is not the reproduction.
#
# Outside a smoke run nothing here caps the data: a stage either runs over all 894 plots and all
# 101 species or it does not run.

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
inner_from <- pick("inner", file.path(here, "inner_folds.csv"))
smoke <- if (is.null(opt$smoke)) NULL else as.integer(split_opt("smoke", ""))
if (!is.null(smoke) && (length(smoke) != 2L || anyNA(smoke) || any(smoke < 1L))) {
  stop("--smoke is two whole numbers, outer folds and species, as --smoke=1,4.", call. = FALSE)
}
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

say <- function(...) cat(format(Sys.time(), "[%H:%M:%S] "), ..., "\n", sep = "")
assert_equal <- function(what, got, expected) {
  say(sprintf("%-38s %s", paste0(what, ":"), format(got)))
  # A smoke run holds a few folds and a few species on purpose, so the counts of the study are not
  # what it is looking at and nothing it produces is compared with anything.
  if (!is.null(smoke)) {
    return(invisible(FALSE))
  }
  if (!isTRUE(all.equal(got, expected))) {
    stop(what, " is ", format(got), ", the deposit gives ", format(expected),
         ". The input is not what this script was written against.", call. = FALSE)
  }
}

# Every comparison with a published number goes through here, at a tolerance TOLERANCE names
# before anything is fitted. Nothing stops the run: a number outside its tolerance is the finding,
# and the table of every comparison is written at the end.
checks <- list()
compare_with <- function(what, got, expected, tolerance) {
  # A smoke run holds a few folds and a few species, so its numbers are of another design and are
  # reported rather than compared with anything the study published.
  if (!is.null(smoke)) {
    say(sprintf("%-38s %.5f, smoke run, not compared", paste0(what, ":"), got))
    return(invisible(NA))
  }
  inside <- is.finite(got) && abs(got - expected) <= tolerance
  say(sprintf("%-38s %.5f against %.5f, %+.5f, tolerance %.5f: %s", paste0(what, ":"), got,
              expected, got - expected, tolerance, if (inside) "inside" else "OUTSIDE"))
  checks[[length(checks) + 1L]] <<- data.frame(
    quantity = what, reproduced = got, reported = expected, difference = got - expected,
    tolerance = tolerance, inside = inside, stringsAsFactors = FALSE)
  invisible(inside)
}
# Every file written carries its identity beside it: run.meta names the deposit, the package and
# the fold map every number was computed on, and each CSV is listed there with its own sum as it
# is written, so a table on disk can be traced to an input and a build rather than to a name.
meta_path <- file.path(out_dir, "run.meta")
meta <- function(...) cat(..., "\n", sep = "", file = meta_path, append = TRUE)
write_out <- function(x, name) {
  name <- if (is.null(smoke)) name else paste0("smoke_", name)
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
meta("inner ", if (identical(inner_from, "build")) "dealt by fold_map()" else inner_from)
if (!is.null(smoke)) {
  meta("SMOKE RUN ", smoke[1L], " outer folds and ", smoke[2L], " species; not the reproduction")
  say("SMOKE RUN: ", smoke[1L], " outer folds, ", smoke[2L],
      " species. Nothing written here is the reproduction.")
}

# The environment the numbers were produced in, which for a fitted encoder is as much a part of
# the input as the deposit is: a network fitted under another libtorch on another device does not
# return the same weights.
environment_note <- function() {
  torch_line <- if (!requireNamespace("torch", quietly = TRUE)) {
    "torch not installed"
  } else {
    installed <- tryCatch(as.character(utils::packageVersion("torch")), error = function(e) NA)
    # The libtorch a build links is reported under one name in some versions of the package and
    # another in others, so both are asked before the field is left unreported.
    lantern <- tryCatch(as.character(torch::torch_config()$libtorch_version),
                        error = function(e) NA_character_)
    if (is.na(lantern)) {
      lantern <- tryCatch(as.character(getFromNamespace("torch_version", "torch")()),
                          error = function(e) NA_character_)
    }
    cuda <- tryCatch(torch::cuda_is_available(), error = function(e) NA)
    device <- if (isTRUE(cuda)) {
      paste0("cuda, ", tryCatch(torch::cuda_device_count(), error = function(e) NA), " device(s)")
    } else {
      "cpu"
    }
    paste0("torch ", installed, ", libtorch ", if (is.na(lantern)) "unreported" else lantern,
           ", device ", device)
  }
  paste0("R ", paste(R.version$major, R.version$minor, sep = "."), " on ", R.version$platform,
         "; timesift ", as.character(utils::packageVersion("timesift")), "; ", torch_line)
}
meta("environment ", environment_note())
say(environment_note())

# The contract of Chytry et al.: species in at least 25 plots, then five aggregate taxa removed.
MIN_OCCURRENCES <- 25L
DROP_TAXA <- c("Alchemilla vulgaris agg.", "Taraxacum sp.", "Festuca halleri agg.",
               "Euphrasia sp.", "Phleum alpinum agg.")
CV_FOLDS <- 10L
INNER_FOLDS <- 5L
CV_SEED <- 1L
REPORTED_STATS <- c("cold_day", "mean", "warm_day")
METRIC_NAME <- "tss"
# The selection is made on the area under the curve, as the study's was, and the arms are reported
# under both that and the true skill statistic.
SELECTION_METRIC <- "roc_auc"

# The study's candidate set: every (window, summary) pair its grid holds, 33 of them. The four
# day-level summaries need whole days, so they are defined from the weekly window up, and the two
# reading-level extremes from the half-daily window up; the mean is defined everywhere. Written as
# a table rather than a list of names, so a pair is added by widening a row.
CANDIDATE_SUMMARIES <- list(
  mean         = list(stats = "mean",
                      grains = c("native", "halfday", "day", "week", "month", "season", "year")),
  min          = list(stats = "min",
                      grains = c("halfday", "day", "week", "month", "season", "year")),
  max          = list(stats = "max",
                      grains = c("halfday", "day", "week", "month", "season", "year")),
  minmeanmax   = list(stats = c("min", "mean", "max"),
                      grains = c("halfday", "day", "week", "month", "season", "year")),
  dailyextreme = list(stats = c("mean_daily_min", "mean", "mean_daily_max"),
                      grains = c("week", "month", "season", "year")),
  extremeday   = list(stats = REPORTED_STATS,
                      grains = c("week", "month", "season", "year")))
EXPECTED_CANDIDATES <- 33L

# What the study reported, and the tolerance each comparison is made at. Stated here, before
# anything is fitted, so a number that lands outside is a finding rather than a tolerance chosen
# after the fact.
#
# `select_window` is exact: the study's five-inner-fold selection chose a weekly candidate in every
# one of the ten outer folds, and a run landing on another window is a different procedure rather
# than a different draw. `select_summary` is not asserted: in five of those folds the best and the
# second-best inner score differ by less than 0.0013 and in one by 0.00001, which is inside the
# seed noise of a single encoder fit (a standard deviation of 0.0017 at the median cell), so which
# summary wins there is not reproducible and the script only reports it.
#
# The level and the margin tolerances are three times that seed noise, taken over 101 species; the
# elastic-net tolerance is the 0.001 the aggregated-feature arm already reproduces to, doubled,
# since its inner cross-validation draws its own folds.
REFERENCE <- list(
  select_window = "week",
  selection_auc = 0.87697, selection_tss = 0.70982,
  series_tss = 0.69621, series_auc = 0.86801,
  margin_auc = 0.00896, margin_auc_lo = 0.00550, margin_auc_hi = 0.01241,
  margin_tss = 0.01361,
  aggregates_tss = 0.687)
TOLERANCE <- list(level = 0.005, margin = 0.005, elastic_net = 0.002)

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

if (!is.null(smoke)) {
  keep_folds <- sort(unique(folds))[seq_len(min(smoke[1L], length(unique(folds))))]
  keep_units <- names(folds)[folds %in% keep_folds]
  keep_species <- names(sort(colSums(y), decreasing = TRUE))[seq_len(min(smoke[2L], ncol(y)))]
  y <- y[keep_units, keep_species, drop = FALSE]
  folds <- folds[keep_units]
  say(sprintf("smoke: %d plots, %d species, outer folds %s", nrow(y), ncol(y),
              paste(keep_folds, collapse = ",")))
}

# The inner partition of each outer training set the selection chooses a candidate on. The study
# wrote one, as it wrote the outer map, so the selection reads it back rather than dealing its
# own; the fit is handed only its training response, so which outer fold it is in is read off the
# units the fit does not hold.
inner_splitter <- function(path, folds) {
  if (identical(path, "build")) {
    say("dealing ", INNER_FOLDS, " inner folds per outer training set with fold_map()")
    return(function(y_train) fold_map(y_train, v = INNER_FOLDS, seed = CV_SEED, strata = 5L))
  }
  say("reading the inner fold map from ", path)
  meta("inner md5 ", unname(tools::md5sum(path)))
  tab <- utils::read.csv(path)
  tab$logger_ID <- as.character(tab$logger_ID)
  function(y_train) {
    units <- rownames(y_train)
    outside <- unique(folds[setdiff(names(folds), units)])
    if (length(outside) != 1L) {
      stop("this fit leaves out ", length(outside), " outer folds, and the study's inner map is ",
           "written one per outer fold. Use --inner=build.", call. = FALSE)
    }
    rows <- tab[tab$outer_fold == outside, , drop = FALSE]
    map <- stats::setNames(as.integer(rows$inner_fold), rows$logger_ID)
    covered <- units %in% names(map)
    if (!all(covered)) {
      stop("the inner map of outer fold ", outside, " covers ", sum(covered), " of ",
           length(units), " training plots.", call. = FALSE)
    }
    map[units]
  }
}
inner_split <- inner_splitter(inner_from, folds)

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
reads_record <- any(c("representation", "networks", "selection") %in% stages) ||
  ("baseline" %in% stages && "series" %in% baseline_arms)
if (!reads_record) {
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
  if (!is.null(smoke)) {
    readings <- readings[readings$logger_ID %in% rownames(y), , drop = FALSE]
    say("smoke: ", nrow(readings), " readings over ", length(unique(readings$logger_ID)), " plots")
  }
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

# One arm's held-out predictions read under a metric other than the one it was scored by. The
# predictions are the arm's own, so this rescores rather than refits.
rescore <- function(ladder, arm, metric) {
  rows <- score_predictions(y, attr(ladder, "predictions")[[arm]], folds, cells, metric)
  mean(tapply(rows$score[!is.na(rows$score)], rows$variable[!is.na(rows$score)], mean))
}

# The series arm as an arm of the metric the selection is made on, so the contrast between the two
# is read on one metric. Its predictions are the ones the arm already made; only the reading of
# them changes.
series_ladder_auc <- function(lad) {
  if (is.null(lad)) {
    return(NULL)
  }
  rows <- score_predictions(y, attr(lad, "predictions")[["series|elastic_net"]], folds, cells,
                            SELECTION_METRIC)
  structure(cbind(grain = "series", learner = "elastic_net", rows, stringsAsFactors = FALSE),
            class = c("timesift_ladder", "data.frame"), metric = SELECTION_METRIC,
            response = "presence_absence")
}

series_ladder <- NULL

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
  aggregated <- intersect(baseline_arms, c("elastic_net", "stepwise"))
  if (length(aggregated)) {
    say("reading the deposit's aggregated temperature features")
    agg <- utils::read.csv(deposit_file("output_temperature_variables_scaled.csv"),
                           check.names = FALSE)
    rownames(agg) <- as.character(agg$logger_ID)
    agg <- as.matrix(agg[rownames(y), setdiff(names(agg), "logger_ID"), drop = FALSE])
    assert_equal("aggregated temperature variables", ncol(agg), 188L)

    features <- feature_matrix(agg, label = "aggregates")
    say("fitting the aggregated-feature arms, selection redone inside every fold")
    arms <- list(
      elastic_net = elasticnet(alpha = 0.5, n_inner = INNER_FOLDS, squares = TRUE,
                               seed = CV_SEED),
      stepwise = stepwise(max_terms = 3L, degree = 2L))[aggregated]
    baseline <- grain_ladder(features, y, arms, folds = folds, metric = METRIC_NAME)
    write_out(baseline, "baseline.csv")
    print(summary(baseline))
    level <- summary(baseline)
    if ("elastic_net" %in% aggregated) {
      compare_with("the 188 aggregates, elastic net, TSS",
                   level$score[level$learner == "elastic_net"], REFERENCE$aggregates_tss,
                   TOLERANCE$elastic_net)
    }
  }

  # The penalised fit on the record itself rather than on summaries of it: the weekly coldest day,
  # the weekly mean and the weekly warmest day, 157 weeks by three channels, which is the 471
  # numbers per plot the paper reports and the same columns the study's own series arm read.
  if ("series" %in% baseline_arms) {
    say("fitting the penalised model on the weekly coldest-day, mean and warmest-day series")
    weekly <- build("week", REPORTED_STATS)
    assert_equal("numbers per plot, weekly three-channel", dim(weekly)[2L] * dim(weekly)[3L],
                 471L)
    series_ladder <- grain_ladder(
      timesift_set(list(series = weekly)), y,
      list(elastic_net = elasticnet(alpha = 0.5, n_inner = INNER_FOLDS, squares = TRUE,
                                    seed = CV_SEED)),
      folds = folds, metric = METRIC_NAME)
    write_out(series_ladder, "baseline_series.csv")
    print(summary(series_ladder))
    compare_with("the weekly series elastic net, TSS", summary(series_ladder)$score,
                 REFERENCE$series_tss, TOLERANCE$level)
    compare_with("the weekly series elastic net, AUC",
                 rescore(series_ladder, "series|elastic_net", SELECTION_METRIC),
                 REFERENCE$series_auc, TOLERANCE$level)
  }
}

# ---- the demonstration's own procedure, through the public interface -------------------------

# The study chose one of 33 candidates inside each outer training set, on five inner folds, by the
# area under the curve, and refitted the winner on the whole training set. That is what
# select_grain() does, so the demonstration is this call rather than a description of one.
if ("selection" %in% stages) {
  say("building the candidate set")
  parts <- list()
  for (summary_name in names(CANDIDATE_SUMMARIES)) {
    spec <- CANDIDATE_SUMMARIES[[summary_name]]
    for (w in intersect(spec$grains, grid_grains)) {
      x <- build(w, spec$stats)
      parts[[paste(w, summary_name, sep = ".")]] <- bind_channels(x, calendar_channels(x))
    }
  }
  if (identical(sort(grid_grains), sort(names(EXPECTED_BINS)))) {
    assert_equal("candidates", length(parts), EXPECTED_CANDIDATES)
  } else {
    say(sprintf("%-38s %d over %s", "candidates:", length(parts),
                paste(grid_grains, collapse = ",")))
  }
  set <- timesift_set(parts)

  say("selecting inside each outer training set: ", length(parts), " candidates, ",
      length(unique(folds)), " outer folds, ", INNER_FOLDS, " inner folds, ", epochs, " epochs")
  selection <- select_grain(set, y, cnn(epochs = epochs, batch_size = 32L), folds = folds,
                            inner = inner_split, metric = SELECTION_METRIC,
                            compare = series_ladder_auc(series_ladder), verbose = TRUE)
  print(selection)
  write_out(selection$selected, "selection.csv")
  write_out(selection$inner, "selection_inner.csv")
  write_out(selection$scores, "selection_cells_auc.csv")
  write_out(score_predictions(y, attr(selection, "predictions")[["selected|selected"]], folds,
                              cells, METRIC_NAME), "selection_cells_tss.csv")

  # Which window each outer fold chose. The study chose a weekly candidate in all ten; which
  # weekly summary won is inside seed noise in half of them, so it is reported and not compared.
  windows <- sub("[.].*$", "", selection$selected$grain)
  say("windows selected: ", paste(sprintf("fold %s %s", selection$selected$fold, windows),
                                  collapse = "; "))
  if (is.null(smoke)) {
    checks[[length(checks) + 1L]] <- data.frame(
      quantity = "outer folds selecting the reported window", reproduced = sum(windows ==
        REFERENCE$select_window), reported = length(windows), difference = sum(windows ==
        REFERENCE$select_window) - length(windows), tolerance = 0, inside = all(windows ==
        REFERENCE$select_window), stringsAsFactors = FALSE)
  }

  est <- selection$estimate
  compare_with("the selected procedure, AUC",
               est$score[est$metric == SELECTION_METRIC & est$interval == "variables"],
               REFERENCE$selection_auc, TOLERANCE$level)
  compare_with("the selected procedure, TSS",
               est$score[est$metric == METRIC_NAME & est$interval == "variables"],
               REFERENCE$selection_tss, TOLERANCE$level)

  if (!is.null(selection$contrast)) {
    write_out(selection$contrast, "selection_contrast.csv")
    print(selection$contrast)
    compare_with("the procedure over the weekly series elastic net, AUC",
                 selection$contrast$diff[1L], REFERENCE$margin_auc, TOLERANCE$margin)
  } else {
    say("no contrast: the weekly series arm is not in this run. Add --baseline=series.")
  }
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

# ---- what was compared, and to what ------------------------------------------------------------

if (length(checks)) {
  table_of <- do.call(rbind, checks)
  write_out(table_of, "checks.csv")
  cat("\n")
  print(table_of, row.names = FALSE, digits = 5)
  outside <- table_of$quantity[!table_of$inside]
  say(if (length(outside)) {
    paste0(length(outside), " of ", nrow(table_of), " outside tolerance: ",
           paste(outside, collapse = "; "))
  } else {
    paste0("all ", nrow(table_of), " comparisons inside the tolerances stated above")
  })
}

meta("finished ", format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"))
say("done")
