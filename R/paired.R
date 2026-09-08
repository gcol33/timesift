#' Compare two arms cell by cell
#'
#' Two arms scored on the same held-out units do not necessarily have the same set of defined
#' cells, so a difference of two marginal means is not a difference between the arms. This takes
#' the difference inside each `(variable, fold)` cell both arms scored, averages it within a
#' variable over its folds, and summarises those per-variable means, the variables being the
#' independent replicates.
#'
#' Pairing also cancels what a threshold-selected metric carries in its level. TSS read at the
#' threshold that maximises it is biased upward where presences are thin, both arms carry the same
#' bias on the same cell, and it cancels in the difference. That is why the levels a ladder reports
#' are upper bounds while the differences between arms are read at face value.
#'
#' @param ladder A [grain_ladder()] result.
#' @param a,b The two arms, each given as `"learner"` or `"grain|learner"`. Naming a learner alone
#'   takes its best grain.
#'
#' @return A one-row data frame: the mean per-variable difference, a 95 percent interval from its
#'   standard error across variables, the number of variables the difference favours, the paired
#'   cells and variables it rests on, and a Wilcoxon signed-rank p-value.
#'
#' @examplesIf requireNamespace("glmnet", quietly = TRUE)
#' set.seed(1)
#' t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24 * 200)
#' units <- sprintf("p%02d", 1:60)
#' warmth <- rnorm(60)
#' d <- data.frame(
#'   plot = rep(units, each = length(t)), t = rep(t, length(units)),
#'   temp = as.numeric(vapply(warmth, function(w) w + sin(seq_along(t) / 300) + rnorm(length(t)),
#'                            numeric(length(t)))))
#' y <- matrix(rbinom(120, 1, plogis(c(warmth, -warmth))), nrow = 60,
#'             dimnames = list(units, c("sp1", "sp2")))
#' x <- grain_matrix(d, plot, t, temp, grain = c("week", "month"))
#' lad <- grain_ladder(x, y, elasticnet(), folds = fold_map(y, v = 3), verbose = FALSE)
#' paired_contrast(lad, "week|elasticnet", "month|elasticnet")
#'
#' @export
paired_contrast <- function(ladder, a, b) {
  ra <- .arm_rows(ladder, a)
  rb <- .arm_rows(ladder, b)
  key_a <- paste(ra$variable, ra$fold)
  key_b <- paste(rb$variable, rb$fold)
  shared <- intersect(key_a[!is.na(ra$score)], key_b[!is.na(rb$score)])
  if (!length(shared)) {
    stop("the two arms share no cell both scored.", call. = FALSE)
  }
  va <- ra$score[match(shared, key_a)]
  vb <- rb$score[match(shared, key_b)]
  variable <- ra$variable[match(shared, key_a)]
  per_variable <- tapply(va - vb, variable, mean)

  n <- length(per_variable)
  d <- mean(per_variable)
  se <- stats::sd(per_variable) / sqrt(n)
  p <- if (n > 1L && any(per_variable != 0)) {
    suppressWarnings(stats::wilcox.test(per_variable)$p.value)
  } else {
    NA_real_
  }
  data.frame(a = attr(ra, "label"), b = attr(rb, "label"), diff = d,
             lower = d - 1.96 * se, upper = d + 1.96 * se,
             n_variable = n, n_cell = length(shared), n_favour = sum(per_variable > 0),
             p_value = p, stringsAsFactors = FALSE)
}

.arm_rows <- function(ladder, arm) {
  if (!inherits(ladder, "timesift_ladder")) {
    stop("expected a grain_ladder() result, got ", class(ladder)[1L], ".", call. = FALSE)
  }
  # An arm is matched whole against the labels the ladder carries rather than split at a `|`, so
  # a grain or a learner whose own name holds one is still found.
  label <- paste(ladder$grain, ladder$learner, sep = "|")
  if (arm %in% label) {
    rows <- ladder[label == arm, , drop = FALSE]
  } else {
    s <- summary(ladder)
    s <- s[s$learner == arm & !is.na(s$score), , drop = FALSE]
    if (!nrow(s)) {
      stop("no arm or learner called \"", arm, "\" in this ladder.", call. = FALSE)
    }
    grain <- s$grain[which.max(s$score)]
    rows <- ladder[ladder$grain == grain & ladder$learner == arm, , drop = FALSE]
  }
  structure(rows, label = paste(rows$grain[1L], rows$learner[1L], sep = "|"),
            grain = rows$grain[1L], learner = rows$learner[1L])
}

#' How much a self-selected threshold inflates the reported level
#'
#' The true skill statistic is read at the threshold that maximises it, chosen on the same held-out
#' units the score is then read on. That selection inflates the level, and by more the fewer
#' presences a cell holds. Most code carries the inflation silently; this measures it for the
#' presence counts of a given design.
#'
#' Predictions are simulated under a normal model in which the population skill is exactly `skill`,
#' at the cell sizes and presence counts of the response and fold map supplied, and the level is
#' read back exactly as [grain_ladder()] reports it. The gap between what comes back and the truth
#' planted is the inflation.
#'
#' It cancels in the paired differences [paired_contrast()] takes, since both arms carry it on the
#' same cell. It does not cancel in a level, so a level is an upper bound on the skill a population
#' has.
#'
#' @inheritParams scorable_cells
#' @param skill Population skill values to plant.
#' @param replicates Replicates per value.
#' @param seed Random seed.
#'
#' @return A data frame with one row per planted value: the truth, the mean level read back, the
#'   inflation, and its interval across replicates.
#'
#' @examples
#' set.seed(1)
#' y <- matrix(rbinom(1200, 1, 0.15), nrow = 200,
#'             dimnames = list(sprintf("p%03d", 1:200), paste0("sp", 1:6)))
#' tss_inflation(y, fold_map(y, v = 5), skill = c(0.6, 0.9), replicates = 40)
#'
#' @export
tss_inflation <- function(y, folds, skill = c(0.6, 0.7, 0.9), replicates = 200L, seed = 1L) {
  y <- .as_response(y)
  f <- .as_folds(folds, rownames(y))
  cells <- scorable_cells(y, stats::setNames(f, rownames(y)))
  cells <- cells[cells$scorable, , drop = FALSE]
  if (!nrow(cells)) {
    stop("no cell of this design is scorable, so there is no level to measure.", call. = FALSE)
  }

  old <- .seed_state()
  on.exit(.restore_seed(old), add = TRUE)
  set.seed(seed)

  out <- lapply(skill, function(target) {
    # Under a binormal model with unit variances the maximum of sensitivity + specificity - 1 is
    # 2 * Phi(delta / 2) - 1, so a separation is what plants a population skill exactly.
    delta <- 2 * stats::qnorm((target + 1) / 2)
    # The variables are visited in the order the mask carries them, which is C collation, and not
    # in the order tapply() would impose by turning them into a factor: that follows LC_COLLATE,
    # and the draws are consumed variable by variable, so a session in another locale would read a
    # different number off the same design. Same order as the Python side, for the same reason.
    variables <- unique(cells$variable)
    per_replicate <- vapply(seq_len(replicates), function(r) {
      by_variable <- vapply(variables, function(v) {
        mean(vapply(which(cells$variable == v), function(k) {
          n_pos <- cells$pres_test[k]
          n_neg <- cells$abs_test[k]
          tss(c(rep(1L, n_pos), rep(0L, n_neg)),
              c(stats::rnorm(n_pos, delta), stats::rnorm(n_neg)))
        }, numeric(1L)))
      }, numeric(1L))
      mean(by_variable)
    }, numeric(1L))
    data.frame(skill = target, reported = mean(per_replicate),
               inflation = mean(per_replicate) - target,
               lower = unname(stats::quantile(per_replicate, 0.025)),
               upper = unname(stats::quantile(per_replicate, 0.975)),
               replicates = replicates)
  })
  do.call(rbind, out)
}

#' What population skill a reported level is consistent with
#'
#' [tss_inflation()] maps a population skill to the level a design reports for it. This inverts
#' that map: given a level actually read off a ladder, it solves for the population skill whose
#' expected reported level equals it.
#'
#' It answers the question a level raises once the inflation is known, and it is the only honest
#' way to read a level as a statement about a population rather than about a scoring rule. It says
#' nothing about a difference between two arms, where the inflation cancels and the reported number
#' stands as it is.
#'
#' @inheritParams tss_inflation
#' @param observed Reported levels to invert.
#' @param grid Population skills the forward map is measured on before interpolating between them.
#'
#' @return A data frame of one row per observed level: the level, the population skill it is
#'   consistent with, and whether that sits inside the grid the map was measured on.
#'
#' @examples
#' set.seed(1)
#' y <- matrix(rbinom(1200, 1, 0.15), nrow = 200,
#'             dimnames = list(sprintf("p%03d", 1:200), paste0("sp", 1:6)))
#' implied_skill(y, fold_map(y, v = 5), observed = 0.71, replicates = 40)
#'
#' @export
implied_skill <- function(y, folds, observed, grid = seq(0, 0.95, by = 0.05),
                          replicates = 200L, seed = 1L) {
  map <- tss_inflation(y, folds, skill = grid, replicates = replicates, seed = seed)
  map <- map[order(map$reported), , drop = FALSE]
  inside <- observed >= min(map$reported) & observed <= max(map$reported)
  data.frame(observed = observed,
             skill = stats::approx(map$reported, map$skill, xout = observed, rule = 2,
                                   ties = "ordered")$y,
             within_grid = inside)
}
