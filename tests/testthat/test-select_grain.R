# A candidate grid whose grains differ in what they let a learner see, so the selector has a real
# choice to get right or wrong.
selection_learner <- function(offset = 0) {
  learner(
    "linear",
    fit = function(x, y, ...) {
      m <- apply(x[, , 1, drop = FALSE], 1L, mean)
      list(coef = vapply(seq_len(ncol(y)), function(j) {
        stats::coef(stats::glm(y[, j] ~ m, family = stats::binomial()))
      }, numeric(2L)), offset = offset)
    },
    predict = function(model, x) {
      m <- apply(x[, , 1, drop = FALSE], 1L, mean)
      p <- vapply(seq_len(ncol(model$coef)), function(j) {
        stats::plogis(model$coef[1L, j] + model$coef[2L, j] * m + model$offset)
      }, numeric(length(m)))
      matrix(p, nrow = length(m))
    }
  )
}

selection_fixture <- function(v = 4L) {
  sim <- sim_series(n_unit = 56L, days = 90L, seed = 31L)
  y <- sim_response(sim, n_var = 4L, seed = 32L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = c("day", "week", "month"))
  list(x = x, y = y, folds = fold_map(y, v = v, seed = 6L))
}

test_that("a selection reports one winner per outer fold from the candidate set it searched", {
  f <- selection_fixture()
  sel <- suppressWarnings(select_grain(f$x, f$y, selection_learner(), folds = f$folds, inner = 3L,
                                       verbose = FALSE))
  expect_s3_class(sel, "timesift_selection")
  expect_equal(nrow(sel$selected), 4L)
  expect_setequal(sel$selected$fold, sort(unique(unclass(f$folds))))
  expect_equal(nrow(sel$candidates), 3L)
  expect_true(all(paste(sel$selected$grain, sel$selected$learner) %in%
                    paste(sel$candidates$grain, sel$candidates$learner)))
})

test_that("the estimate is reported under every registered metric on one set of predictions", {
  f <- selection_fixture()
  sel <- suppressWarnings(select_grain(f$x, f$y, selection_learner(), folds = f$folds, inner = 3L,
                                       verbose = FALSE))
  expect_setequal(sel$estimate$metric, metrics())
  expect_true(all(is.finite(sel$estimate$score)))
  expect_true(all(sel$estimate$n_variable <= ncol(f$y)))
  # The selection metric's estimate is the mean of the same per-cell scores the object carries.
  own <- sel$estimate$score[sel$estimate$metric == attr(sel, "metric")]
  cells <- sel$scores[!is.na(sel$scores$score), , drop = FALSE]
  expect_equal(own, mean(tapply(cells$score, cells$variable, mean)))
})

test_that("no outer test unit reaches the selector or the refit of its own fold", {
  seen <- new.env(parent = emptyenv())
  seen$fitted <- list()
  spy <- learner(
    "spy",
    fit = function(x, y, ...) {
      seen$fitted[[length(seen$fitted) + 1L]] <- dimnames(x)[[1L]]
      list(rate = colMeans(y))
    },
    predict = function(model, x) {
      m <- apply(x[, , 1, drop = FALSE], 1L, mean)
      outer(rank(m) / length(m), model$rate, function(a, b) stats::plogis(a - 0.5 + b))
    }
  )
  f <- selection_fixture()
  sel <- suppressWarnings(select_grain(f$x, f$y, spy, folds = f$folds, inner = 3L, verbose = FALSE))
  outer_fold <- stats::setNames(as.integer(f$folds), names(f$folds))
  # Every unit a model was fitted on during outer fold k, across the inner ladder and the refit,
  # must have come from outside fold k.
  per_outer <- length(seen$fitted) / nrow(sel$selected)
  expect_equal(per_outer %% 1, 0)
  for (i in seq_len(nrow(sel$selected))) {
    k <- sel$selected$fold[i]
    held <- names(outer_fold)[outer_fold == k]
    block <- seen$fitted[seq_len(per_outer) + (i - 1L) * per_outer]
    for (fitted_on in block) {
      expect_length(intersect(fitted_on, held), 0L)
    }
  }
})

test_that("the summary counts how often each candidate won and print stays terse", {
  f <- selection_fixture()
  sel <- suppressWarnings(select_grain(f$x, f$y, selection_learner(), folds = f$folds, inner = 3L,
                                       verbose = FALSE))
  s <- summary(sel)
  expect_equal(nrow(s), nrow(sel$candidates))
  expect_named(s, c("grain", "learner", "n_selected", "share", "inner_score"))
  expect_equal(sum(s$n_selected), nrow(sel$selected))
  expect_equal(sum(s$share), 1)
  expect_output(print(sel), "timesift selection")
})

test_that("the plot draws the inner scores and returns them", {
  f <- selection_fixture()
  sel <- suppressWarnings(select_grain(f$x, f$y, selection_learner(), folds = f$folds, inner = 3L,
                                       verbose = FALSE))
  path <- tempfile(fileext = ".png")
  grDevices::png(path)
  drawn <- plot(sel)
  grDevices::dev.off()
  expect_true(file.exists(path))
  expect_equal(nrow(drawn), nrow(sel$candidates) * nrow(sel$selected))
  expect_named(drawn, c("fold", "grain", "learner", "score", "n_variable"))
  unlink(path)
})

test_that("the contrast against a ladder runs through paired_contrast on matched cells", {
  f <- selection_fixture()
  lad <- suppressWarnings(grain_ladder(f$x, f$y, selection_learner(), folds = f$folds,
                                        verbose = FALSE))
  sel <- suppressWarnings(select_grain(f$x, f$y, selection_learner(), folds = f$folds, inner = 3L,
                                       compare = lad, verbose = FALSE))
  expect_equal(nrow(sel$contrast), 3L)
  expect_true(all(sel$contrast$a == "selected|selected"))
  expect_setequal(sel$contrast$b, paste(lad$grain, lad$learner, sep = "|"))
  expect_true(all(sel$contrast$lower <= sel$contrast$diff &
                    sel$contrast$diff <= sel$contrast$upper))
})

test_that("a comparator scored by another metric is refused", {
  f <- selection_fixture()
  lad <- suppressWarnings(grain_ladder(f$x, f$y, selection_learner(), folds = f$folds,
                                        metric = "roc_auc", verbose = FALSE))
  expect_error(select_grain(f$x, f$y, selection_learner(), folds = f$folds, inner = 3L,
                            compare = lad, verbose = FALSE),
               "same metric")
})

test_that("a candidate set with nothing to choose between is refused", {
  f <- selection_fixture()
  expect_error(select_grain(f$x[1L], f$y, selection_learner(), folds = f$folds, inner = 3L,
                            verbose = FALSE),
               "at least two candidates")
  expect_error(select_grain(f$x, f$y, selection_learner(), folds = f$folds, inner = 1L,
                            verbose = FALSE),
               "at least 2")
})

test_that("adding a grain to the set widens the search with no other change", {
  sim <- sim_series(n_unit = 48L, days = 90L, seed = 33L)
  y <- sim_response(sim, n_var = 3L, seed = 34L)
  narrow <- grain_matrix(sim$readings, plot, t, temp, grain = c("week", "month"))
  wide <- timesift_set(c(as.list(narrow),
                          list(week_extreme = grain_matrix(sim$readings, plot, t, temp,
                                                            grain = "week",
                                                            stats = c("cold_day", "mean",
                                                                      "warm_day")))))
  folds <- fold_map(y, v = 3L, seed = 9L)
  a <- suppressWarnings(select_grain(narrow, y, selection_learner(), folds = folds, inner = 3L,
                                     verbose = FALSE))
  b <- suppressWarnings(select_grain(wide, y, selection_learner(), folds = folds, inner = 3L,
                                     verbose = FALSE))
  expect_equal(nrow(a$candidates), 2L)
  expect_equal(nrow(b$candidates), 3L)
  expect_true("week_extreme" %in% b$candidates$grain)
})


# Recovery. The generating grain is planted, so the selector can be asked whether it finds it and
# whether the level it reports is honest about how it was chosen.

# Units differ only in a slow component. Hourly noise buries it, and averaging over a month recovers
# it, so the monthly grain is the grain the response was generated at.
planted_grain <- function(n_unit = 72L, days = 168L, noise = 20, seed = 81L) {
  set.seed(seed)
  t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24L * days)
  units <- sprintf("p%03d", seq_len(n_unit))
  warmth <- stats::rnorm(n_unit)
  value <- as.numeric(vapply(warmth, function(w) {
    w * 1.5 + stats::rnorm(length(t), sd = noise)
  }, numeric(length(t))))
  list(units = units, warmth = warmth,
       readings = data.frame(plot = rep(units, each = length(t)), t = rep(t, times = n_unit),
                             temp = value, stringsAsFactors = FALSE))
}

planted_response <- function(sim, n_var = 8L, strength = 3, seed = 82L) {
  set.seed(seed)
  sign <- rep(c(1, -1), length.out = n_var)
  matrix(stats::rbinom(length(sim$warmth) * n_var, 1L,
                       stats::plogis(strength * as.numeric(outer(sim$warmth, sign)))),
         ncol = n_var, dimnames = list(sim$units, paste0("sp", seq_len(n_var))))
}

test_that("the grain the response was generated at is selected above chance", {
  skip_if_not_installed("glmnet")
  sim <- planted_grain()
  y <- planted_response(sim)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = c("day", "week", "month"))
  sel <- suppressWarnings(select_grain(x, y, elasticnet(),
                                       folds = fold_map(y, v = 5L, seed = 7L),
                                       inner = 4L, seed = 3L, verbose = FALSE))
  picked <- table(factor(sel$selected$grain, levels = names(x)))
  # Chance over three candidates is a third of the five outer folds; the planted grain has to beat
  # that, and the finest grain, where the signal is buried, must not win outright.
  expect_gt(picked[["month"]], nrow(sel$selected) / 3)
  expect_gte(picked[["month"]], max(picked[["day"]], picked[["week"]]))
})

test_that("the nested estimate stays under what choosing on the held-out units would have paid", {
  skip_if_not_installed("glmnet")
  sim <- planted_grain(seed = 83L)
  y <- planted_response(sim, seed = 84L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = c("day", "week", "month"))
  folds <- fold_map(y, v = 5L, seed = 7L)
  lad <- suppressWarnings(grain_ladder(x, y, elasticnet(), folds = folds, verbose = FALSE))
  sel <- suppressWarnings(select_grain(x, y, elasticnet(), folds = folds, inner = 4L,
                                       seed = 3L, compare = lad, verbose = FALSE))

  # The bound the nested estimate must respect is the oracle: the same candidates, the same fits,
  # but the grain for each cell picked with the held-out score itself. The procedure picks one of
  # those candidates without seeing them, so cell for cell it cannot come out above the oracle, and
  # the gap is what selecting on the test units would have bought.
  best_cell <- tapply(lad$score, paste(lad$variable, lad$fold), function(v) {
    if (all(is.na(v))) NA_real_ else max(v, na.rm = TRUE)
  })
  keep <- !is.na(best_cell)
  variable <- sub(" .*$", "", names(best_cell)[keep])
  oracle <- mean(tapply(as.numeric(best_cell[keep]), variable, mean))
  own <- sel$estimate$score[sel$estimate$metric == attr(sel, "metric")]
  expect_lte(own, oracle)
  expect_lt(own, oracle)

  # Against the finest grain, where the planted signal is buried, the procedure must still win.
  against_day <- sel$contrast[sel$contrast$b == "day|elasticnet", ]
  expect_gt(against_day$diff, 0)
  expect_gt(against_day$lower, 0)
})

test_that("a fold's held-out predictions are those of the candidate it selected", {
  skip_if_not_installed("glmnet")
  sim <- planted_grain(n_unit = 48L, days = 90L, seed = 87L)
  y <- planted_response(sim, n_var = 4L, seed = 88L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = c("week", "month"))
  folds <- fold_map(y, v = 3L, seed = 7L)
  lad <- suppressWarnings(grain_ladder(x, y, elasticnet(), folds = folds, verbose = FALSE))
  sel <- suppressWarnings(select_grain(x, y, elasticnet(), folds = folds, inner = 3L,
                                       seed = 3L, verbose = FALSE))
  # The refit is the ladder's own fit on the same units at the same grain, so every cell of the
  # selected procedure is a cell of the ladder rather than a number from a second fitting path. It
  # is what makes the oracle a bound rather than a comparison of two different pipelines.
  f <- stats::setNames(as.integer(folds), names(folds))
  for (i in seq_len(nrow(sel$selected))) {
    k <- sel$selected$fold[i]
    arm <- paste(sel$selected$grain[i], sel$selected$learner[i], sep = "|")
    held <- names(f)[f == k]
    expect_equal(attr(sel$scores, "predictions")[["selected|selected"]][held, ],
                 attr(lad, "predictions")[[arm]][held, ])
  }
})

test_that("with no signal at any grain the procedure scores at the design's own floor", {
  skip_if_not_installed("glmnet")
  sim <- planted_grain(seed = 85L)
  set.seed(86)
  y <- matrix(stats::rbinom(length(sim$units) * 4L, 1L, 0.35), ncol = 4L,
              dimnames = list(sim$units, paste0("sp", 1:4)))
  x <- grain_matrix(sim$readings, plot, t, temp, grain = c("week", "month"))
  folds <- fold_map(y, v = 5L, seed = 7L)
  sel <- suppressWarnings(select_grain(x, y, elasticnet(), folds = folds, inner = 4L,
                                       seed = 3L, verbose = FALSE))
  # TSS read at the cut that maximises it is biased upward on cells this small, so the floor is what
  # a design with no signal reports rather than zero.
  floor <- tss_inflation(y, folds, skill = 0, replicates = 60L, seed = 12L)$reported
  own <- sel$estimate$score[sel$estimate$metric == "tss"]
  expect_lt(own, floor + 0.12)
})

test_that("a selection hands its control to the inner search and to the refit alike", {
  f <- selection_fixture(v = 2L)
  seen <- new.env(parent = emptyenv())
  seen$epochs <- integer()
  trained <- learner(
    "trained",
    fit = function(x, y, control, ...) {
      seen$epochs <- c(seen$epochs, control$epochs)
      list(coef = colMeans(y))
    },
    predict = function(model, x) {
      outer(rank(apply(x[, , 1, drop = FALSE], 1L, mean)) / dim(x)[1L], model$coef,
            function(a, b) a)
    },
    multi = "joint")
  suppressWarnings(select_grain(f$x, f$y, trained, folds = f$folds, inner = 2L,
                                control = train_control(epochs = 7L), verbose = FALSE))
  # Two outer folds, each running an inner ladder over three grains at two inner folds and one
  # refit: nothing in that chain may reach a learner without the control the caller gave.
  expect_equal(length(seen$epochs), 2L * (3L * 2L + 1L))
  expect_true(all(seen$epochs == 7L))
})
