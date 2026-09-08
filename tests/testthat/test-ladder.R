constant_learner <- function(offset = 0) {
  learner(
    "constant",
    fit = function(x, y, ...) list(rate = colMeans(y), offset = offset),
    predict = function(model, x) {
      # ranks the units by their mean reading and by how much it moves across the bins, so the arm
      # has a defined direction to score and the grains do not all give the same answer
      m <- apply(x[, , 1, drop = FALSE], 1L, mean) + apply(x[, , 1, drop = FALSE], 1L, stats::sd)
      outer(rank(m) / length(m), model$rate, function(a, b) stats::plogis(a - 0.5 + model$offset))
    }
  )
}

ladder_fixture <- function(v = 4L) {
  sim <- sim_series(n_unit = 48L, days = 90L, seed = 21L)
  y <- sim_response(sim, n_var = 3L, seed = 22L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = c("day", "week", "month"))
  list(x = x, y = y, folds = fold_map(y, v = v, seed = 5L))
}

test_that("every arm is scored on the same cells", {
  f <- ladder_fixture()
  lad <- grain_ladder(f$x, f$y, list(a = constant_learner(), b = constant_learner(0.2)),
                       folds = f$folds, verbose = FALSE)
  by_arm <- split(lad$scorable, paste(lad$grain, lad$learner))
  expect_true(all(vapply(by_arm, identical, logical(1L), by_arm[[1L]])))
  expect_equal(nrow(lad), 3L * 2L * 3L * 4L)
  expect_true(all(is.na(lad$score[!lad$scorable])))
})

test_that("a ladder carries one held-out prediction per unit and arm", {
  f <- ladder_fixture()
  lad <- grain_ladder(f$x, f$y, constant_learner(), folds = f$folds, verbose = FALSE)
  preds <- attr(lad, "predictions")
  expect_named(preds, c("day|constant", "week|constant", "month|constant"))
  for (p in preds) {
    expect_equal(dim(p), c(48L, 3L))
    expect_false(anyNA(p))
  }
})

test_that("a unit is never fitted on and scored in the same fold", {
  seen <- new.env(parent = emptyenv())
  seen$fitted <- list()
  seen$tested <- list()
  spy <- learner(
    "spy",
    fit = function(x, y, ...) {
      seen$fitted[[length(seen$fitted) + 1L]] <- dimnames(x)[[1L]]
      list(rate = colMeans(y))
    },
    predict = function(model, x) {
      seen$tested[[length(seen$tested) + 1L]] <- dimnames(x)[[1L]]
      matrix(rep(model$rate, each = dim(x)[1L]), nrow = dim(x)[1L])
    }
  )
  f <- ladder_fixture()
  invisible(grain_ladder(f$x, f$y, spy, folds = f$folds, verbose = FALSE))
  # One model is fitted and predicted from per response, so the two records run in step.
  expect_equal(length(seen$fitted), length(seen$tested))
  expect_gt(length(seen$fitted), 0L)
  for (i in seq_along(seen$fitted)) {
    expect_length(intersect(seen$fitted[[i]], seen$tested[[i]]), 0L)
  }
})

test_that("the summary marks each learner's best grain", {
  f <- ladder_fixture()
  lad <- grain_ladder(f$x, f$y, list(a = constant_learner(), b = constant_learner(0.3)),
                       folds = f$folds, verbose = FALSE)
  s <- summary(lad)
  expect_equal(sum(s$best), 2L)
  for (l in unique(s$learner)) {
    rows <- s[s$learner == l, ]
    expect_equal(rows$grain[rows$best], rows$grain[which.max(rows$score)])
  }
})

test_that("a paired contrast runs on cells both arms scored", {
  f <- ladder_fixture()
  lad <- grain_ladder(f$x, f$y, list(a = constant_learner(), b = constant_learner(0.3)),
                       folds = f$folds, verbose = FALSE)
  p <- paired_contrast(lad, "week|a", "week|b")
  expect_equal(p$n_variable, 3L)
  expect_lte(p$n_cell, sum(lad$scorable) / 6L)
  expect_true(p$lower <= p$diff && p$diff <= p$upper)
  expect_equal(paired_contrast(lad, "week|a", "week|b")$diff,
               -paired_contrast(lad, "week|b", "week|a")$diff)
})

test_that("naming a learner alone contrasts it at its own best grain", {
  f <- ladder_fixture()
  lad <- grain_ladder(f$x, f$y, list(a = constant_learner(), b = constant_learner(0.3)),
                       folds = f$folds, verbose = FALSE)
  s <- summary(lad)
  best <- s$grain[s$learner == "a" & s$best]
  expect_equal(paired_contrast(lad, "a", "b")$a, paste(best, "a", sep = "|"))
})

test_that("two learners cannot be reported under one name", {
  f <- ladder_fixture()
  expect_error(grain_ladder(f$x, f$y, list(constant_learner(), constant_learner(0.3)),
                             folds = f$folds, verbose = FALSE),
               "same name")
})

test_that("the plot returns what it drew", {
  f <- ladder_fixture()
  lad <- grain_ladder(f$x, f$y, constant_learner(), folds = f$folds, verbose = FALSE)
  path <- tempfile(fileext = ".png")
  grDevices::png(path)
  drawn <- plot(lad)
  grDevices::dev.off()
  expect_true(file.exists(path))
  expect_equal(nrow(drawn), 3L)
  expect_named(drawn, c("learner", "grain", "score", "se"))
  unlink(path)
})

test_that("a ladder on which nothing was scorable reports no level rather than failing", {
  skip_if_not_installed("glmnet")
  sim <- sim_series(n_unit = 24L, days = 40L)
  y <- sim_response(sim)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = c("week", "month"))
  lad <- suppressWarnings(
    grain_ladder(x, y, elasticnet(), folds = fold_map(y, v = 3L), verbose = FALSE))
  lad$score <- NA_real_
  out <- summary(lad)
  expect_equal(nrow(out), 0L)
  expect_named(out, c("learner", "grain", "score", "n_variable", "best"))
  expect_output(print(lad), "timesift ladder")
})

test_that("a ladder hands its control to every learner that declares one", {
  f <- ladder_fixture(v = 2L)
  seen <- new.env(parent = emptyenv())
  seen$epochs <- integer()
  trained <- learner(
    "trained",
    fit = function(x, y, control, ...) {
      seen$epochs <- c(seen$epochs, control$epochs)
      list(rate = colMeans(y))
    },
    predict = function(model, x) {
      outer(rank(apply(x[, , 1, drop = FALSE], 1L, mean)) / dim(x)[1L], model$rate,
            function(a, b) a)
    },
    control = train_control(batch_size = 8L))
  grain_ladder(f$x, f$y, trained, folds = f$folds, control = train_control(epochs = 3L),
                verbose = FALSE)
  expect_true(length(seen$epochs) > 0L)
  expect_true(all(seen$epochs == 3L))
})

test_that("a learner's own control overrides the ladder's on the settings it names", {
  f <- ladder_fixture(v = 2L)
  seen <- new.env(parent = emptyenv())
  seen$got <- list()
  trained <- learner(
    "trained",
    fit = function(x, y, control, ...) {
      seen$got[[length(seen$got) + 1L]] <- control
      list(rate = colMeans(y))
    },
    predict = function(model, x) {
      outer(rank(apply(x[, , 1, drop = FALSE], 1L, mean)) / dim(x)[1L], model$rate,
            function(a, b) a)
    },
    control = train_control(epochs = 11L))
  grain_ladder(f$x, f$y, trained, folds = f$folds,
                control = train_control(epochs = 3L, batch_size = 8L), verbose = FALSE)
  expect_true(all(vapply(seen$got, function(c) c$epochs, integer(1L)) == 11L))
  expect_true(all(vapply(seen$got, function(c) c$batch_size, integer(1L)) == 8L))
})
