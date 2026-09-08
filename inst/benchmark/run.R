#!/usr/bin/env Rscript
# One or more replicates of one benchmark cell. Every replicate is a self-contained unit of work:
# it draws its own units, runs the selection, the single-loop reading and the oracle against an
# independent deployment sample, and writes one tidy file. A run that is killed loses the replicate
# in flight and nothing else, and restarting the same command skips whatever is already on disk.
#
# Usage:
#   Rscript inst/benchmark/run.R --cell=elasticnet-event-n300 --reps=1:200 --out=<dir> [--force]
#   Rscript inst/benchmark/run.R --scale=smoke --list

suppressWarnings(suppressMessages({
  library(timesift)
}))

here <- grep("^--file=", commandArgs(FALSE), value = TRUE)
here <- if (length(here)) dirname(normalizePath(sub("^--file=", "", here[1L]))) else getwd()
source(file.path(here, "design.R"))

opt <- bench_args(list(cell = NA_character_, reps = NA_character_, out = NA_character_,
                       force = FALSE, list = FALSE, pkg = NA_character_, scale = "full"))
bench_scale(opt$scale)
pkg_dir <- if (is.na(opt$pkg)) dirname(dirname(here)) else opt$pkg

cells <- bench_cells()
if (opt$list) {
  print(cells)
  quit(save = "no")
}
if (is.na(opt$cell) || !opt$cell %in% cells$cell_id) {
  stop("--cell must name one of: ", paste(cells$cell_id, collapse = ", "), call. = FALSE)
}
cell <- cells[match(opt$cell, cells$cell_id), ]
reps <- if (is.na(opt$reps)) seq_len(cell$replicates) else eval(parse(text = opt$reps))
if (!is.numeric(reps) || anyNA(reps) || any(reps < 1L)) {
  stop("--reps must evaluate to positive whole numbers, e.g. 1:200 or c(3,7).", call. = FALSE)
}
out_dir <- file.path(if (is.na(opt$out)) file.path(getwd(), BENCH$results) else opt$out,
                     cell$cell_id)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

candidates <- bench_candidates()
learner <- bench_learner(cell$block)


bench_replicate <- function(cell, replicate, candidates, learner, pkg_dir) {
  stamp <- bench_stamp(cell, replicate, candidates, pkg_dir, learner)
  message(sprintf("[%s] replicate %d: n=%d, %s, %d candidates (%s), inner=%d, outer=%d, %s@%s",
                  stamp$cell_id, replicate, stamp$n_unit, stamp$mechanism, stamp$n_candidate,
                  stamp$candidate_digest, stamp$inner, stamp$outer, stamp$pkg_version,
                  substr(stamp$pkg_commit, 1L, 8L)))
  clock <- list()
  tick <- function(name, expr) {
    t0 <- Sys.time()
    out <- force(expr)
    clock[[name]] <<- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    out
  }

  sim <- tick("simulate", .bench_simulate(cell, cell$n_unit, stamp$design_seed, replicate))
  set <- tick("represent", bench_representation(sim$readings, candidates))
  dep <- tick("deployment", .bench_deployment(cell, stamp, candidates))
  folds <- fold_map(sim$y, v = BENCH$outer, seed = replicate)

  sel <- tick("select", select_grain(set, sim$y, learner, folds = folds, inner = cell$inner,
                                     metric = BENCH$metric, verbose = FALSE))
  bench_assert_candidates(sel, stamp)
  lad <- tick("ladder", grain_ladder(set, sim$y, learner, folds = folds, metric = BENCH$metric,
                                      verbose = FALSE))

  f <- attr(sel, "folds")
  levels <- sort(unique(f))
  train1 <- names(f)[f != levels[1L]]

  # The oracle arm: every candidate fitted on one outer training set and scored on units the
  # procedure never saw, which is the only place a regret can be read from.
  truth <- tick("oracle", vapply(candidates$candidate, function(cc) {
    fit <- fit_learner(learner, timesift:::.subset_units(set[[cc]], which(f != levels[1L])),
                       sim$y[train1, , drop = FALSE])
    mean(.bench_deploy_score(fit, dep$set[[cc]], dep$y, BENCH$metric))
  }, numeric(1L)))

  # The same refit the procedure makes at line 12 of its own algorithm, scored on the deployment
  # sample instead of on the outer fold, so the reported number has something to be honest about.
  true_fold <- tick("procedure", vapply(seq_along(levels), function(i) {
    cc <- sel$selected$grain[match(levels[i], sel$selected$fold)]
    idx <- which(f != levels[i])
    fit <- fit_learner(learner, timesift:::.subset_units(set[[cc]], idx),
                       sim$y[names(f)[idx], , drop = FALSE])
    mean(.bench_deploy_score(fit, dep$set[[cc]], dep$y, BENCH$metric))
  }, numeric(1L)))

  grid <- summary(lad)
  single <- grid$grain[which.max(grid$score)]
  est <- sel$estimate
  sel_est <- est[est$metric == BENCH$metric, ]

  rows <- list(
    .bench_row(stamp, "nested", NA_character_, NA_integer_, est$metric, "reported", est$score),
    .bench_row(stamp, "nested", NA_character_, NA_integer_, est$metric, "reported_se", est$se),
    .bench_row(stamp, "nested", NA_character_, NA_integer_, BENCH$metric, "reported_lower",
               sel_est$score - 1.96 * sel_est$se),
    .bench_row(stamp, "nested", NA_character_, NA_integer_, BENCH$metric, "reported_upper",
               sel_est$score + 1.96 * sel_est$se),
    .bench_row(stamp, "nested", NA_character_, NA_integer_, BENCH$metric, "true",
               mean(true_fold)),
    .bench_row(stamp, "nested", sel$selected$grain, sel$selected$fold, BENCH$metric, "true_fold",
               true_fold),
    .bench_row(stamp, "nested", sel$selected$grain, sel$selected$fold, BENCH$metric, "selected",
               sel$selected$inner_score),
    .bench_row(stamp, "single_loop", single, NA_integer_, BENCH$metric, "reported",
               max(grid$score)),
    .bench_row(stamp, "single_loop", single, NA_integer_, BENCH$metric, "true",
               unname(truth[single])),
    .bench_row(stamp, "oracle", names(truth)[which.max(truth)], NA_integer_, BENCH$metric, "true",
               max(truth)),
    .bench_row(stamp, "candidate", names(truth), NA_integer_, BENCH$metric, "true",
               unname(truth)),
    .bench_row(stamp, "candidate", grid$grain, NA_integer_, BENCH$metric, "ladder", grid$score),
    .bench_row(stamp, "candidate", sel$inner$grain, sel$inner$fold, BENCH$metric, "inner",
               sel$inner$score),
    .bench_row(stamp, "stage", names(clock), NA_integer_, NA_character_, "secs",
               unlist(clock))
  )
  do.call(rbind, rows)
}

for (r in reps) {
  target <- file.path(out_dir, sprintf("rep_%05d.csv.gz", r))
  if (file.exists(target) && !opt$force) {
    message("[", cell$cell_id, "] replicate ", r, " already on disk, skipping")
    next
  }
  started <- Sys.time()
  rows <- bench_replicate(cell, r, candidates, learner, pkg_dir)
  tmp <- paste0(target, ".partial")
  con <- gzfile(tmp, open = "wt")
  utils::write.csv(rows, con, row.names = FALSE)
  close(con)
  file.rename(tmp, target)
  message(sprintf("[%s] replicate %d written in %.1f min", cell$cell_id, r,
                  as.numeric(difftime(Sys.time(), started, units = "mins"))))
}
