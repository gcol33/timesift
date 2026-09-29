# MARS on the core, against the earth package, and the learner above it.
#
# The reference is earth's own fit on the fixture design, written by `inst/spec/make_fixtures.R`:
# every term of the forward pass, the terms the pruning pass keeps, their coefficients and the
# predictions on the design scaled by 1.01. The terms are asserted exactly, in the order the
# forward pass added them, each cut to the tolerance its case carries; the coefficients and the
# predictions likewise. The Python suite reads the same files.

mars_fixture <- function() {
  dir <- fixture_dir()
  if (is.null(dir)) skip("the fixtures ship with the package and are not installed here")
  d <- utils::read.csv(file.path(dir, "penalised_input.csv"), stringsAsFactors = FALSE,
                       check.names = FALSE)
  held <- c("unit", "y_gaussian", "y_binomial", "w", "fold")
  x <- as.matrix(d[, setdiff(names(d), held), drop = FALSE])
  x <- x[, !endsWith(colnames(x), "^2"), drop = FALSE]
  list(x = x, w = d$w, y = list(y_binomial = d$y_binomial, y_gaussian = d$y_gaussian),
       cases = utils::read.csv(file.path(dir, "mars_cases.csv"), stringsAsFactors = FALSE),
       coef = utils::read.csv(file.path(dir, "mars_coef.csv"), stringsAsFactors = FALSE),
       predict = utils::read.csv(file.path(dir, "mars_predict.csv"), stringsAsFactors = FALSE))
}

mars_case_fit <- function(fx, row) {
  y <- fx$y[[row$response]]
  w <- if (isTRUE(row$weighted)) fx$w else rep(1, length(y))
  family <- if (row$response == "y_gaussian") "gaussian" else "binomial"
  .mars_fit(fx$x, y, w, family, degree = row$degree, minspan = row$minspan,
            endspan = row$endspan, fast_k = row$fast_k, prune = isTRUE(row$prune),
            nprune = if (is.na(row$nprune)) NULL else row$nprune)
}

# One term's factors as the fixture writes them: `column:direction` in column order, and the cuts
# apart so they are compared as numbers.
mars_terms <- function(f) {
  lapply(seq_len(length(f$factor_start) - 1L), function(t) {
    idx <- f$factor_start[t] + seq_len(f$factor_start[t + 1L] - f$factor_start[t])
    list(key = if (length(idx)) paste(sprintf("%d:%d", f$factor_column[idx] + 1L,
                                              f$factor_dir[idx]), collapse = "*") else "1",
         cut = f$factor_cut[idx])
  })
}

mars_fixture_terms <- function(forward) {
  lapply(strsplit(forward, " ", fixed = TRUE)[[1L]], function(term) {
    if (term == "1") return(list(key = "1", cut = numeric()))
    parts <- strsplit(strsplit(term, "*", fixed = TRUE)[[1L]], ":", fixed = TRUE)
    list(key = paste(vapply(parts, function(p) paste(p[1:2], collapse = ":"), character(1L)),
                     collapse = "*"),
         cut = as.numeric(vapply(parts, `[`, character(1L), 3L)))
  })
}

test_that("the core's forward and pruning passes keep the terms earth keeps", {
  fx <- mars_fixture()
  for (i in seq_len(nrow(fx$cases))) {
    row <- fx$cases[i, ]
    f <- mars_case_fit(fx, row)
    got <- mars_terms(f)
    want <- mars_fixture_terms(row$forward)
    expect_identical(vapply(got, `[[`, character(1L), "key"),
                     vapply(want, `[[`, character(1L), "key"), info = row$case)
    expect_equal(unlist(lapply(got, `[[`, "cut")), unlist(lapply(want, `[[`, "cut")),
                 tolerance = row$cut_tolerance, info = row$case)
    expect_identical(f$termcond, row$termcond, info = row$case)
    expect_identical(paste(f$selected + 1L, collapse = " "), row$selected, info = row$case)
    expect_equal(f$gcv, row$gcv, tolerance = row$coef_tolerance, info = row$case)
    coef <- fx$coef[fx$coef$case == row$case, ]
    expect_equal(f$beta, coef$coefficient[match(f$selected + 1L, coef$term)],
                 tolerance = row$coef_tolerance, info = row$case)
    pred <- fx$predict[fx$predict$case == row$case, ]
    expect_equal(.mars_predict(f, fx$x * 1.01), pred$fitted_out,
                 tolerance = row$prediction_tolerance, info = row$case)
  }
})

test_that("the thread count does not change what the passes return", {
  fx <- mars_fixture()
  for (weighted in c(FALSE, TRUE)) {
    w <- if (weighted) fx$w else rep(1, nrow(fx$x))
    one <- .mars_fit(fx$x, fx$y$y_gaussian, w, "gaussian", degree = 2L, threads = 1L)
    four <- .mars_fit(fx$x, fx$y$y_gaussian, w, "gaussian", degree = 2L, threads = 4L)
    expect_identical(four, one)
  }
})

test_that("mars refuses settings it has no pass for", {
  expect_error(mars(degree = 0), "whole number")
  expect_error(mars(degree = 1.5), "whole number")
  expect_error(mars(penalty = -2), "-1")
  expect_error(mars(thresh = 1), "\\[0, 1\\)")
  expect_error(mars(nk = 0), "whole number")
  expect_error(mars(prune = NA), "TRUE or FALSE")
  x <- matrix(c(1, 3, 2, 5, 4, 6), ncol = 1L)
  expect_error(.mars_fit(x, c(0, 1, 0, 1, 0, 1), c(1, 1, -1, 1, 1, 1), "gaussian"),
               "zero or more")
  f <- .mars_fit(x, c(0, 1, 0, 1, 0, 1), rep(1, 6L), "binomial")
  expect_error(.mars_predict(f, cbind(x, x)), "fitted on 1 columns")
})

test_that("mars fits under both heads, round trips, and an intercept alone is the share", {
  sim <- sim_series(n_unit = 60L, days = 120L, seed = 43L)
  y <- sim_response(sim, n_var = 2L, seed = 44L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "month")
  for (l in list(mars(), mars(degree = 2L), mars(prune = FALSE), mars(nprune = 3L))) {
    fit <- fit_learner(l, x, y)
    p <- stats::predict(fit, x)
    expect_equal(dim(p), c(60L, 2L))
    expect_true(all(p >= 0 & p <= 1))
    path <- tempfile(fileext = ".rds")
    saveRDS(fit, path)
    expect_identical(stats::predict(readRDS(path), x), p)
    unlink(path)
  }
  lone <- .mars_fit(.flatten(x), y[, 1], rep(1, 60L), "binomial", nprune = 1L)
  expect_identical(lone$selected, 0L)
  expect_equal(.mars_predict(lone, .flatten(x)), rep(mean(y[, 1]), 60L), tolerance = 1e-10)
})

test_that("mars fits the family the response head's loss names, and a constant is its mean", {
  local_response("continuous_test", list(
    prepare = function(y) .as_response(y), activation = "identity",
    loss = "squared_error", metric = "roc_auc",
    cells = function(y, folds) scorable_cells(y > stats::median(y), folds)))
  sim <- sim_series(n_unit = 60L, days = 60L, seed = 91L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "week")
  level <- 10 + 3 * scale(rowMeans(x[, , 1L]))[, 1L]
  y <- matrix(level, ncol = 1L, dimnames = list(dimnames(x)[[1L]], "height"))
  fit <- fit_learner(mars(), x, y, response = "continuous_test")
  expect_equal(fit$model$family, "gaussian")
  expect_gt(stats::cor(stats::predict(fit, x)[, 1L], y[, 1L]), 0.8)

  flat <- matrix(0, nrow = 60L, ncol = 1L, dimnames = list(dimnames(x)[[1L]], "absent"))
  expect_equal(unique(stats::predict(fit_learner(mars(), x, flat), x)[, 1L]), 0)
})
