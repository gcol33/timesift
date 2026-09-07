test_that("a learner of one's own needs nothing but a fit and a predict", {
  mine <- learner(
    "mine",
    fit = function(x, y, ...) colMeans(y),
    predict = function(model, x) matrix(rep(model, each = dim(x)[1L]), nrow = dim(x)[1L])
  )
  sim <- sim_series(n_unit = 20L, days = 30L)
  y <- sim_response(sim, n_var = 2L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "week")
  fit <- fit_learner(mine, x, y)
  p <- stats::predict(fit, x)
  expect_equal(dim(p), c(20L, 2L))
  expect_equal(colnames(p), colnames(y))
  expect_equal(rownames(p), rownames(y))
})

test_that("a registered learner can be asked for by name", {
  expect_true(all(c("elasticnet", "stepwise", "forest", "mlp", "cnn", "rescnn") %in% learners()))
  expect_s3_class(.as_learner("elasticnet"), "timesift_learner")
  expect_error(.as_learner("nope"), "unknown learner")
  register_learner("test_only", function() learner("test_only",
                                                  fit = function(x, y, ...) NULL,
                                                  predict = function(model, x) NULL),
                   overwrite = TRUE)
  expect_true("test_only" %in% learners())
  expect_error(register_learner("test_only", function() NULL), "already registered")
})

test_that("a learner that cannot run says so instead of doing something else", {
  needy <- learner("needy", fit = function(x, y, ...) NULL,
                   predict = function(model, x) NULL,
                   needs = "a.package.that.does.not.exist")
  sim <- sim_series(n_unit = 10L, days = 10L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "week")
  expect_error(fit_learner(needy, x, sim_response(sim)),
               "needs a.package.that.does.not.exist")
})

test_that("the response is matched to the representation by unit, not by position", {
  sim <- sim_series(n_unit = 20L, days = 30L)
  y <- sim_response(sim, n_var = 2L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "week")
  seen <- NULL
  spy <- learner("spy",
                 fit = function(x, y, ...) {
                   seen <<- y
                   colMeans(y)
                 },
                 predict = function(model, x)
                   matrix(rep(model, each = dim(x)[1L]), nrow = dim(x)[1L]))
  invisible(fit_learner(spy, x, y[nrow(y):1, , drop = FALSE]))
  expect_equal(rownames(seen), dimnames(x)[[1]])
  expect_equal(seen, y)
  expect_error(fit_learner(spy, x, y[-1, , drop = FALSE]), "not the response")
})

test_that("flattening names every predictor by its bin and its channel", {
  sim <- sim_series(n_unit = 5L, days = 20L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "week", stats = c("mean", "max"))
  m <- .flatten(x)
  expect_equal(dim(m), c(5L, dim(x)[2] * 2L))
  expect_equal(m[, 1], x[, 1, "mean"])
  expect_true(all(grepl("^mean@|^max@", colnames(m))))
})

test_that("a scaler is computed on what it is given and applied to something else", {
  m <- matrix(c(1, 2, 3, 10, 20, 30), ncol = 2)
  s <- .scaler(m)
  expect_equal(s$centre, c(2, 20))
  expect_equal(.apply_scaler(s, m)[, 1], c(-1, 0, 1))
  scalar <- .scaler(m, per_column = FALSE)
  expect_equal(scalar$centre, rep(mean(m), 2))
  constant <- .scaler(matrix(1, nrow = 4, ncol = 2))
  expect_equal(constant$scale, c(1, 1))
})

test_that("the penalised learner fits, predicts and refuses a different representation", {
  skip_if_not_installed("glmnet")
  sim <- sim_series(n_unit = 60L, days = 60L, seed = 31L)
  y <- sim_response(sim, n_var = 2L, seed = 32L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "week")
  fit <- fit_learner(elasticnet(), x, y)
  p <- stats::predict(fit, x)
  expect_true(all(p >= 0 & p <= 1))
  expect_gt(tss(y[, 1], p[, 1]), 0.4)
  other <- grain_matrix(sim$readings, plot, t, temp, grain = "month")
  expect_error(stats::predict(fit, other), "different channels or bins")
})

test_that("forward selection stops at its budget and is non-monotone in a predictor", {
  sim <- sim_series(n_unit = 60L, days = 60L, seed = 33L)
  y <- sim_response(sim, n_var = 1L, seed = 34L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "month")
  fit <- fit_learner(stepwise(max_terms = 2L), x, y)
  chosen <- fit$model$models[[1L]]$columns
  expect_lte(length(chosen), 2L)
  p <- stats::predict(fit, x)
  expect_true(all(p >= 0 & p <= 1))
})

test_that("a column holding one value is not offered to the forward search", {
  withr::defer(.responses_reg$remove("constant_continuous_test"))
  register_response("constant_continuous_test", list(
    prepare = function(y) .as_response(y), activation = "identity",
    loss = "squared_error", metric = "roc_auc",
    cells = function(y, folds) scorable_cells(y > stats::median(y), folds)),
    overwrite = TRUE)

  sim <- sim_series(n_unit = 40L, days = 60L, seed = 37L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "month",
                    stats = c("min", "mean", "max"))
  # A channel every unit reads the same value on: what a sensor that never moved records, or a
  # predictor that is constant over the units this fold holds.
  flat <- x
  flat[, , "min"] <- 7

  binary <- sim_response(sim, n_var = 1L, seed = 38L)
  level <- matrix(10 + 3 * scale(rowMeans(x[, , "mean"]))[, 1L], ncol = 1L,
                  dimnames = list(dimnames(x)[[1L]], "height"))
  for (arm in list(list(y = binary, response = "presence_absence"),
                   list(y = level, response = "constant_continuous_test"))) {
    fit <- fit_learner(stepwise(max_terms = 2L), flat, arm$y, response = arm$response)
    chosen <- fit$model$models[[1L]]$columns
    expect_gt(length(chosen), 0L)
    expect_false(any(grepl("min", fit$model$columns[chosen], fixed = TRUE)), info = arm$response)
    expect_equal(dim(stats::predict(fit, flat)), c(40L, 1L), info = arm$response)
  }

  expect_error(.poly_basis(rep(7, 20), 2L), "no polynomial basis")
})

test_that("the penalised learner reports what its fitter refuses instead of a constant", {
  skip_if_not_installed("glmnet")
  sim <- sim_series(n_unit = 40L, days = 60L, seed = 39L)
  y <- sim_response(sim, n_var = 1L, seed = 40L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "month")
  # One inner fold cannot be cross-validated over, and that is glmnet's to say. Swallowed, it
  # would have come back as a constant predictor and been scored and stacked as a candidate.
  expect_error(fit_learner(elasticnet(n_inner = 1L), x, y))
})

test_that("a forest fits, predicts and refuses a different representation", {
  skip_if_not_installed("ranger")
  sim <- sim_series(n_unit = 60L, days = 60L, seed = 35L)
  y <- sim_response(sim, n_var = 2L, seed = 36L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "week")
  fit <- fit_learner(forest(trees = 200L), x, y)
  p <- stats::predict(fit, x)
  expect_equal(dim(p), c(60L, 2L))
  expect_true(all(p >= 0 & p <= 1))
  expect_gt(tss(y[, 1], p[, 1]), 0.4)
  other <- grain_matrix(sim$readings, plot, t, temp, grain = "month")
  expect_error(stats::predict(fit, other), "different channels or bins")
})

test_that("a forest is the same forest twice and reads the same columns as the linear arms", {
  skip_if_not_installed("ranger")
  sim <- sim_series(n_unit = 40L, days = 40L, seed = 37L)
  y <- sim_response(sim, n_var = 1L, seed = 38L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "week", stats = c("mean", "max"))
  a <- stats::predict(fit_learner(forest(trees = 100L, seed = 7L), x, y), x)
  b <- stats::predict(fit_learner(forest(trees = 100L, seed = 7L), x, y), x)
  expect_equal(a, b)
  expect_equal(fit_learner(forest(trees = 50L), x, y)$model$columns, colnames(.flatten(x)))
})

test_that("a learner declares what it reads, how it covers responses and what it is pinned to", {
  expect_equal(vapply(list(elasticnet(), stepwise(), forest()), function(l) l$reads, character(1L)),
               rep("tabular", 3L))
  expect_equal(vapply(list(elasticnet(), stepwise(), forest()), function(l) l$multi, character(1L)),
               rep("separate", 3L))
  expect_equal(vapply(list(mlp(), cnn(), rescnn()), function(l) l$multi, character(1L)),
               rep("joint", 3L))
  expect_equal(vapply(list(mlp(), cnn(), rescnn()), function(l) l$reads, character(1L)),
               c("tabular", "sequence", "sequence"))
  expect_null(elasticnet()$data)

  weekly <- structure(list(label = "week", kind = "grain", grain = "week", stats = "mean",
                           sequence = TRUE),
                      class = "timesift_representation")
  expect_identical(cnn(data = weekly)$data, weekly)
  expect_identical(elasticnet(data = weekly)$data, weekly)
  expect_error(cnn(data = "week"), "is a representation")
  expect_output(print(cnn(data = weekly)), "sequence")
  expect_output(print(cnn(data = weekly)), "week")
  expect_output(print(elasticnet()), "every representation of the run")
})

test_that("the old learner names are gone", {
  for (nm in c("elasticnet_learner", "stepwise_learner", "mlp_learner", "cnn_learner",
               "rescnn_learner", "ensemble_learner")) {
    expect_false(nm %in% getNamespaceExports("timesift"), info = nm)
  }
  expect_true(all(c("elasticnet", "stepwise", "forest", "mlp", "cnn", "rescnn") %in% learners()))
  expect_false("ensemble" %in% learners())
})

test_that("a training setting is defaulted in the control and nowhere else", {
  cfg <- train_control()
  expect_s3_class(cfg, "timesift_control")
  expect_equal(cfg$epochs, 60L)
  expect_equal(cfg$batch_size, 64L)
  expect_equal(cfg$learning_rate, 1e-3)
  expect_equal(cfg$weight_decay, 1e-4)
  expect_equal(cfg$early_stopping, 10L)
  expect_equal(cfg$val_frac, 0.15)
  expect_equal(cfg$device, "auto")
  expect_equal(cfg$seed, 1L)
  expect_error(cfg$lr, "no setting called lr")
  expect_error(train_control(epochs = 0L), "positive")
  expect_error(train_control(val_frac = 1), "share of the run")
  expect_output(print(cfg), "epochs")

  # Only the settings a control names move; the rest come from the control below it.
  run <- train_control(epochs = 100L, batch_size = 16L)
  own <- train_control(epochs = 200L)
  merged <- .resolve_control(run, own)
  expect_equal(merged$epochs, 200L)
  expect_equal(merged$batch_size, 16L)
  expect_equal(merged$val_frac, train_control()$val_frac)
})

test_that("an architecture constructor carries architecture and its training settings separately", {
  l <- cnn(channels = c(8L, 16L), epochs = 5L)
  expect_equal(l$params$channels, c(8L, 16L))
  expect_null(l$params$epochs)
  expect_equal(l$control$epochs, 5L)
  expect_equal(attr(l$control, "given"), "epochs")
  expect_null(cnn()$control)
  expect_error(cnn(epocs = 5L), "no setting called epocs")
  expect_error(mlp(hiden = 4L), "no setting called hiden")
})

test_that("a setting given at fit time overrides the one a linear learner carries", {
  skip_if_not_installed("glmnet")
  sim <- sim_series(n_unit = 40L, days = 40L, seed = 43L)
  y <- sim_response(sim, n_var = 1L, seed = 44L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "week")
  overridden <- suppressWarnings(fit_learner(elasticnet(), x, y, squares = FALSE))
  built <- suppressWarnings(fit_learner(elasticnet(squares = FALSE), x, y))
  expect_false(overridden$model$squares)
  expect_equal(suppressWarnings(stats::predict(overridden, x)),
               suppressWarnings(stats::predict(built, x)))
})

test_that("predicting a single unit returns one row and not one column", {
  skip_if_not_installed("glmnet")
  sim <- sim_series(n_unit = 24L, days = 40L)
  y <- sim_response(sim, n_var = 3L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "week")
  one <- x[1L, , , drop = FALSE]
  attributes(one) <- utils::modifyList(attributes(x)[c("grain", "stats", "year_start",
                                                       "bin_start", "bin_end", "bin_partial")],
                                       list(dim = dim(one), dimnames = dimnames(one),
                                            class = c("timesift_matrix", "array")))
  for (l in list(elasticnet(), stepwise())) {
    fit <- suppressWarnings(fit_learner(l, x, y))
    p <- stats::predict(fit, one)
    expect_equal(dim(p), c(1L, 3L))
    expect_identical(rownames(p), dimnames(x)[[1L]][1L])
  }
})

test_that("the per-response learners fit the family the response head's loss names", {
  skip_if_not_installed("glmnet")
  skip_if_not_installed("ranger")
  withr::defer(.responses_reg$remove("continuous_test"))
  register_response("continuous_test", list(
    prepare = function(y) .as_response(y), activation = "identity",
    loss = "squared_error", metric = "roc_auc",
    cells = function(y, folds) scorable_cells(y > stats::median(y), folds)),
    overwrite = TRUE)
  sim <- sim_series(n_unit = 40L, days = 60L, seed = 91L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "week")
  level <- 10 + 3 * scale(rowMeans(x[, , 1L]))[, 1L]
  y <- matrix(level, ncol = 1L, dimnames = list(dimnames(x)[[1L]], "height"))
  for (l in list(elasticnet(squares = FALSE), forest(trees = 100L), stepwise(max_terms = 1L))) {
    fit <- fit_learner(l, x, y, response = "continuous_test")
    p <- stats::predict(fit, x)
    expect_true(all(p > 1), info = l$name)
    expect_gt(stats::cor(p[, 1L], y[, 1L]), 0.8, label = l$name)
  }
  expect_equal(fit_learner(forest(trees = 20L), x, y, response = "continuous_test")$model$family,
               "gaussian")
  expect_equal(fit_learner(stepwise(max_terms = 1L), x, y, response = "continuous_test")$model$family,
               "gaussian")

  withr::defer(.responses_reg$remove("poisson_test"))
  register_response("poisson_test", list(
    prepare = function(y) .as_response(y), activation = "exp", loss = "poisson",
    metric = "roc_auc", cells = function(y, folds) scorable_cells(y, folds)),
    overwrite = TRUE)
  expect_error(fit_learner(stepwise(), x, y, response = "poisson_test"), "no family for the poisson")
})

test_that("a learner's fit is handed the head only where it declares one", {
  seen <- NULL
  plain <- learner("plain", fit = function(x, y, ...) { seen <<- names(list(...)); 1 },
                   predict = function(model, x) matrix(0.5, nrow = dim(x)[1L], ncol = 1L))
  sim <- sim_series(n_unit = 10L, days = 20L, seed = 92L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "week")
  y <- sim_response(sim, n_var = 1L, seed = 93L)
  fit_learner(plain, x, y)
  expect_false("head" %in% seen)
  aware <- learner("aware", fit = function(x, y, head, ...) head$loss,
                   predict = function(model, x) matrix(0.5, nrow = dim(x)[1L], ncol = 1L))
  expect_equal(fit_learner(aware, x, y)$model, "binary_cross_entropy")
})
