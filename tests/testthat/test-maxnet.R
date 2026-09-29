# maxnet's features, regularisation and fit on the core, against the maxnet package's own, and the
# learner above them.
#
# The reference is maxnet's, on the weekly columns the penalised fixtures carry. The features and
# their penalty factors are asserted to rounding. The fit is asserted by the objective at the
# reference's penalty, which two descents run to the same tolerance reach alike, and by the
# predictions, which collinear hinges leave slightly less determined. The Python suite reads the
# same files.

maxnet_fixture <- function() {
  dir <- fixture_dir()
  if (is.null(dir)) skip("the fixtures ship with the package and are not installed here")
  d <- utils::read.csv(file.path(dir, "penalised_input.csv"), stringsAsFactors = FALSE,
                       check.names = FALSE)
  held <- c("unit", "y_gaussian", "y_binomial", "w", "fold")
  x <- as.matrix(d[, setdiff(names(d), held), drop = FALSE])
  x <- x[, !grepl("\\^2$", colnames(x)), drop = FALSE]
  r <- utils::read.csv(file.path(dir, "maxnet_response.csv"), stringsAsFactors = FALSE)
  r <- r[match(d$unit, r$unit), ]
  list(x = x, y = list(y_binomial = d$y_binomial, y_8 = r$y_8, y_12 = r$y_12), w = d$w,
       fold = d$fold,
       cases = utils::read.csv(file.path(dir, "maxnet_cases.csv"), stringsAsFactors = FALSE),
       reg = utils::read.csv(file.path(dir, "maxnet_regularization.csv"),
                             stringsAsFactors = FALSE),
       predict = utils::read.csv(file.path(dir, "maxnet_predict.csv"), stringsAsFactors = FALSE))
}

# The case's input: the fixture's columns and response, with the first presence's readings
# appended once more as an absence where the case asks for it.
maxnet_case_input <- function(fx, row) {
  x <- fx$x
  y <- fx$y[[row$response]]
  if (isTRUE(row$duplicate)) {
    x <- rbind(x, x[which(y == 1)[1L], , drop = FALSE])
    y <- c(y, 0)
  }
  w <- if (isTRUE(row$weighted)) fx$w else rep(1, length(y))
  list(x = x, y = y, w = w)
}

maxnet_case_classes <- function(row) {
  if (is.na(row$classes) || !nzchar(row$classes)) NULL else row$classes
}

maxnet_case_fit <- function(fx, row, input) {
  absence <- row$formulation == "absence"
  .maxnet_fit(input$x, input$y, input$w, classes = maxnet_case_classes(row),
              regmult = row$regmult, formulation = row$formulation,
              add_samples = row$add_samples, thresh = row$thresh, max_pass = row$max_pass,
              fold = if (absence) fx$fold else NULL, n_fold = if (absence) 5L else 0L)
}

maxnet_key <- function(f) {
  paste(f$kind, f$a, f$b, sprintf("%.17g", f$lo), sprintf("%.17g", f$hi))
}

# The objective both implementations minimise, on glmnet's scale, at the fit's own penalty: the
# weighted mean negative log likelihood over the rows fitted, background rows included, plus the
# penalty with the factors rescaled to sum to the feature count.
maxnet_objective <- function(fit, design, input, row) {
  xa <- input$x[design$rows + 1L, , drop = FALSE]
  ya <- design$y
  w <- if (row$formulation == "background") ya + (1 - ya) * 100 else input$w
  wn <- w / sum(w)
  eta <- fit$lasso_intercept + .maxnet_predict(fit, xa, clamp = FALSE, type = "link") -
    fit$intercept
  vp <- design$reg * length(design$reg) / sum(design$reg)
  at <- match(maxnet_key(fit), maxnet_key(design))
  -sum(wn * (ya * eta - log1p(exp(eta)))) + fit$lambda * sum(vp[at] * abs(fit$beta))
}

test_that("the core builds maxnet's features and regularises each as maxnet does", {
  fx <- maxnet_fixture()
  for (i in seq_len(nrow(fx$cases))) {
    row <- fx$cases[i, ]
    input <- maxnet_case_input(fx, row)
    design <- .maxnet_design(input$x, input$y, classes = maxnet_case_classes(row),
                             regmult = row$regmult, formulation = row$formulation,
                             add_samples = row$add_samples)
    ref <- fx$reg[fx$reg$case == row$case, ]
    expect_identical(design$classes, row$classes_used, info = row$case)
    expect_identical(length(design$rows), row$n_row, info = row$case)
    expect_identical(length(design$reg), row$n_feature, info = row$case)
    expect_equal(design$reg, ref$reg, tolerance = row$reg_tolerance, info = row$case)
  }
})

test_that("the core's maxnet reaches the objective maxnet's own glmnet fit does", {
  fx <- maxnet_fixture()
  for (i in seq_len(nrow(fx$cases))) {
    row <- fx$cases[i, ]
    input <- maxnet_case_input(fx, row)
    design <- .maxnet_design(input$x, input$y, classes = maxnet_case_classes(row),
                             regmult = row$regmult, formulation = row$formulation,
                             add_samples = row$add_samples)
    fit <- maxnet_case_fit(fx, row, input)
    expect_identical(fit$stalled, 0L, info = row$case)
    expect_equal(fit$lambda, row$lambda, tolerance = 1e-10, info = row$case)
    expect_equal(maxnet_objective(fit, design, input, row), row$objective,
                 tolerance = row$objective_tolerance, info = row$case)
    if (row$formulation == "background") {
      expect_equal(fit$entropy, row$entropy, tolerance = row$prediction_tolerance,
                   info = row$case)
      expect_equal(fit$intercept, row$alpha, tolerance = row$prediction_tolerance,
                   info = row$case)
    }
  }
})

test_that("the core's maxnet predicts what maxnet's own fit does, clamped outside the range", {
  fx <- maxnet_fixture()
  for (i in seq_len(nrow(fx$cases))) {
    row <- fx$cases[i, ]
    input <- maxnet_case_input(fx, row)
    fit <- maxnet_case_fit(fx, row, input)
    ref <- fx$predict[fx$predict$case == row$case, ]
    tol <- row$prediction_tolerance
    expect_equal(.maxnet_predict(fit, input$x, type = "logistic"), ref$logistic, tolerance = tol,
                 info = row$case)
    if (row$formulation == "background") {
      out <- input$x * 1.3
      expect_equal(.maxnet_predict(fit, input$x, type = "cloglog"), ref$cloglog,
                   tolerance = tol, info = row$case)
      expect_equal(.maxnet_predict(fit, out, type = "cloglog"), ref$cloglog_out,
                   tolerance = tol, info = row$case)
      expect_equal(.maxnet_predict(fit, out, type = "logistic"), ref$logistic_out,
                   tolerance = tol, info = row$case)
    }
  }
})

test_that("maxnet's classes follow the presence count, and its knots its own seq()", {
  expect_identical(ts_maxnet_design_(c(1, 2, 3, 4), c(1, 1, 0, 0), 4L, 1L, "", 50L, 1,
                                     "background", TRUE, 2)$classes, "l")
  x <- matrix(c(0, 1, 2, 3, 5, 5, 5, 5), ncol = 2L)
  design <- .maxnet_design(x, c(1, 0, 1, 0), classes = "lqht")
  # The constant second column takes no feature; the first takes 1 + 1 + 98 + 49.
  expect_length(design$reg, 149L)
  expect_true(all(design$a == 0L))
  hinges <- design$kind == 2L
  expect_equal(unique(design$hi[hinges][1:49]), 3)
  expect_equal(design$lo[hinges][1:49], seq(0, 3, length.out = 50)[1:49])
  expect_equal(design$hi[hinges][50:98], seq(0, 3, length.out = 50)[2:50])
  expect_equal(design$lo[design$kind == 3L], seq(0, 3, length.out = 52)[2:50 + 1])
})

test_that("maxnet refuses what it has no model for", {
  x <- matrix(c(1, 2, 3, 4), ncol = 1L)
  expect_error(.maxnet_fit(x, c(1, 0, 0, 0), rep(1, 4)), "at least two presences")
  expect_error(.maxnet_fit(x, c(1, 0, 2, 0), rep(1, 4)), "zero and one")
  expect_error(.maxnet_fit(x, c(1, 1, 0, 0), rep(1, 4), classes = "lz"), "letters")
  expect_error(.maxnet_fit(x, c(1, 1, 0, 0), rep(1, 4), regmult = 0), "positive")
  expect_error(.maxnet_fit(matrix(1, 4, 1), c(1, 1, 0, 0), rep(1, 4)), "more than one value")
  expect_error(.maxnet_fit(x, c(1, 1, 0, 0), rep(1, 4), formulation = "absence"),
               "two folds")
  fit <- .maxnet_fit(x, c(1, 1, 0, 0), rep(1, 4), formulation = "absence",
                     fold = c(0L, 1L, 0L, 1L), n_fold = 2L)
  expect_error(.maxnet_predict(fit, x, type = "cloglog"), "background formulation")
  expect_error(.maxnet_predict(fit, cbind(x, x)), "columns it was fitted on")
  expect_error(maxnet(classes = "lqx"), "letters")
  expect_error(maxnet(formulation = "absence", type = "cloglog"), "\"logistic\"")
  expect_error(maxnet(type = "exponential"), "\"cloglog\" or \"logistic\"")
})

test_that("a design above the limit is refused with the size it would have taken", {
  set.seed(3)
  x <- matrix(stats::rnorm(200 * 30), 200L, 30L)
  y <- rep(c(1, 0), each = 100L)
  # 30 columns under "lqph" are 30 + 30 + 2940 + 435 features, over 200 units and the 100
  # presences added to the background: 300 x 3435 doubles.
  expect_error(.maxnet_fit(x, y, rep(1, 200L), classes = "lqph", max_design = 0.001),
               "3435 features over 300 rows, a design of 0.0082 GB")
})

test_that("maxnet fits, predicts, survives a round trip and refuses a different representation", {
  sim <- sim_series(n_unit = 60L, days = 60L, seed = 35L)
  y <- sim_response(sim, n_var = 2L, seed = 36L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "week")
  for (l in list(maxnet(), maxnet(formulation = "absence"))) {
    fit <- fit_learner(l, x, y)
    p <- stats::predict(fit, x)
    expect_equal(dim(p), c(60L, 2L))
    expect_true(all(p >= 0 & p <= 1))
    expect_gt(tss(y[, 1], p[, 1]), 0.4)
    path <- tempfile(fileext = ".rds")
    saveRDS(fit, path)
    expect_equal(stats::predict(readRDS(path), x), p)
    unlink(path)
  }
  other <- grain_matrix(sim$readings, plot, t, temp, grain = "month")
  expect_error(stats::predict(fit, other), "different channels or bins")
})

test_that("the background formulation ignores case weights and the absence one reads them", {
  sim <- sim_series(n_unit = 60L, days = 60L, seed = 51L)
  y <- sim_response(sim, n_var = 1L, seed = 52L)
  m <- .flatten(grain_matrix(sim$readings, plot, t, temp, grain = "week"))
  w <- ifelse(y[, 1] == 1, 5, 1)
  a <- .maxnet_fit(m, y[, 1], rep(1, 60L))
  b <- .maxnet_fit(m, y[, 1], w)
  expect_identical(a$beta, b$beta)
  fold <- rep_len(0:4, 60L)
  a <- .maxnet_fit(m, y[, 1], rep(1, 60L), formulation = "absence", fold = fold, n_fold = 5L)
  b <- .maxnet_fit(m, y[, 1], w, formulation = "absence", fold = fold, n_fold = 5L)
  expect_false(identical(a$beta, b$beta))
})

test_that("clamping holds a reading outside the fitted range at the range's edge", {
  sim <- sim_series(n_unit = 60L, days = 60L, seed = 61L)
  y <- sim_response(sim, n_var = 1L, seed = 62L)
  m <- .flatten(grain_matrix(sim$readings, plot, t, temp, grain = "week"))
  fit <- .maxnet_fit(m, y[, 1], rep(1, 60L), classes = "lq")
  top <- matrix(fit$var_max, nrow = 1L)
  expect_equal(.maxnet_predict(fit, top + 10), .maxnet_predict(fit, top))
  expect_false(isTRUE(all.equal(.maxnet_predict(fit, top + 10, clamp = FALSE),
                                .maxnet_predict(fit, top))))
})

test_that("maxnet needs a presence-absence head, and a thin response is its share", {
  local_response("continuous_test", list(
    prepare = function(y) .as_response(y), activation = "identity",
    loss = "squared_error", metric = "roc_auc",
    cells = function(y, folds) scorable_cells(y > stats::median(y), folds)))
  sim <- sim_series(n_unit = 60L, days = 60L, seed = 91L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "week")
  level <- matrix(rowMeans(x[, , 1L]), ncol = 1L, dimnames = list(dimnames(x)[[1L]], "height"))
  expect_error(fit_learner(maxnet(), x, level, response = "continuous_test"),
               "presence-absence")
  one <- matrix(c(1, rep(0, 59)), ncol = 1L, dimnames = list(dimnames(x)[[1L]], "rare"))
  fit <- fit_learner(maxnet(), x, one)
  expect_identical(fit$model$unfitted, "rare")
  expect_equal(unique(stats::predict(fit, x)[, 1L]), 1 / 60)
})
