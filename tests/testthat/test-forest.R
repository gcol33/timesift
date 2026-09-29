# The forest core, against the forest the spec describes, and the learner above it.
#
# The reference is `helper-oracle-forest.R`, the forest grown from the spec's text in R alone: its
# generator, its bootstrap, its column draws and its split search. The fixtures are written from it
# and the Python suite reads the same files.

forest_fixture <- function() {
  dir <- fixture_dir()
  if (is.null(dir)) skip("the fixtures ship with the package and are not installed here")
  d <- utils::read.csv(file.path(dir, "penalised_input.csv"), stringsAsFactors = FALSE,
                       check.names = FALSE)
  held <- c("unit", "y_gaussian", "y_binomial", "w", "fold")
  counts <- utils::read.csv(file.path(dir, "tree_weights.csv"), stringsAsFactors = FALSE)
  list(x = as.matrix(d[, setdiff(names(d), held), drop = FALSE]), binomial = d$y_binomial,
       gaussian = d$y_gaussian, count = counts$count[match(d$unit, counts$unit)],
       cases = utils::read.csv(file.path(dir, "forest_cases.csv"), stringsAsFactors = FALSE,
                               colClasses = c(seed = "numeric")),
       nodes = utils::read.csv(file.path(dir, "forest_nodes.csv"), stringsAsFactors = FALSE),
       predict = utils::read.csv(file.path(dir, "forest_predict.csv"), stringsAsFactors = FALSE),
       stream = utils::read.csv(file.path(dir, "forest_stream.csv"), stringsAsFactors = FALSE,
                                colClasses = c(seed = "numeric", output = "numeric")))
}

forest_case_fit <- function(fx, row) {
  y <- if (row$family == "binomial") fx$binomial else fx$gaussian
  w <- if (row$weights == "counts") fx$count else rep(1, nrow(fx$x))
  .forest_fit(fx$x, y, w, row$family, row$trees, row$mtry, row$min_leaf, row$balance == 1L,
              row$seed)
}

expect_same_trees <- function(core, reference, info = NULL) {
  expect_length(core, length(reference))
  for (t in seq_along(reference)) {
    for (field in names(reference[[t]])) {
      expect_identical(core[[t]][[field]], reference[[t]][[field]],
                       info = paste(info, "tree", t, field))
    }
  }
}

test_that("the generator is the spec's, output for output", {
  fx <- forest_fixture()
  for (key in unique(paste(fx$stream$seed, fx$stream$tree))) {
    ref <- fx$stream[paste(fx$stream$seed, fx$stream$tree) == key, ]
    expect_identical(.forest_stream(ref$seed[1L], ref$tree[1L], nrow(ref)), ref$output,
                     info = key)
  }
  s <- oracle_stream(7, 12)
  expect_identical(.forest_stream(7, 12, 50L), vapply(1:50, function(i) s$next_word(), 0))
})

test_that("the forest core grows the spec's forest, node for node", {
  fx <- forest_fixture()
  for (i in seq_len(nrow(fx$cases))) {
    row <- fx$cases[i, ]
    trees <- oracle_split_forest(forest_case_fit(fx, row))
    ref <- fx$nodes[fx$nodes$case == row$case, ]
    reference <- lapply(split(ref, ref$tree), function(r) {
      list(column = r$column, threshold = r$threshold, less_left = r$less_left, left = r$left,
           right = r$right, value = r$value)
    })
    expect_same_trees(trees, unname(reference), row$case)
  }
})

test_that("the forest predicts the mean of its trees, as the spec's does", {
  fx <- forest_fixture()
  for (i in seq_len(nrow(fx$cases))) {
    row <- fx$cases[i, ]
    ref <- fx$predict[fx$predict$case == row$case, ]
    expect_identical(.forest_predict(forest_case_fit(fx, row), fx$x), ref$value, info = row$case)
  }
})

test_that("the core and the oracle agree off the fixtures too, ties and weights included", {
  set.seed(83)
  n <- 45L
  x <- cbind(round(stats::rnorm(n), 1), stats::rnorm(n), sample(1:4, n, replace = TRUE),
             stats::runif(n))
  yb <- as.numeric(x[, 1] + x[, 3] / 2 + stats::rnorm(n) > 1)
  yg <- x[, 2] + stats::rnorm(n, sd = 0.3)
  w <- stats::runif(n, 0, 3)
  w[c(3L, 9L)] <- 0
  for (case in list(list(y = yb, f = "binomial", mtry = 2L, leaf = 1L, balance = FALSE),
                    list(y = yb, f = "binomial", mtry = 4L, leaf = 3L, balance = TRUE),
                    list(y = yg, f = "gaussian", mtry = 1L, leaf = 2L, balance = FALSE))) {
    core <- .forest_fit(x, case$y, w, case$f, 4L, case$mtry, case$leaf, case$balance, 99)
    expect_same_trees(oracle_split_forest(core),
                      oracle_forest(x, case$y, w, case$f, 4L, case$mtry, case$leaf,
                                    case$balance, 99),
                      paste(case$f, case$mtry, case$balance))
  }
})

test_that("a forest is the same forest on any number of threads", {
  set.seed(84)
  x <- matrix(stats::rnorm(300), 60L)
  y <- as.numeric(x[, 1] > 0)
  one <- .forest_fit(x, y, rep(1, 60L), "binomial", 40L, 2L, 1L, FALSE, 5)
  expect_identical(.forest_fit(x, y, rep(1, 60L), "binomial", 40L, 2L, 1L, FALSE, 5,
                               threads = 3L), one)
  expect_false(identical(.forest_fit(x, y, rep(1, 60L), "binomial", 40L, 2L, 1L, FALSE, 6), one))
})

test_that("a balanced forest needs a binomial response both classes of which weigh something", {
  set.seed(85)
  x <- matrix(stats::rnorm(80), 40L)
  y <- c(rep(1, 6L), rep(0, 34L))
  w <- rep(1, 40L)
  expect_error(.forest_fit(x, y, c(rep(0, 6L), rep(1, 34L)), "binomial", 2L, 1L, 1L, TRUE, 1),
               "one class weighs nothing")
  expect_error(.forest_fit(x, y, w, "gaussian", 2L, 1L, 1L, TRUE, 1), "binomial response")
})

test_that("the forest core refuses what it cannot grow on", {
  x <- matrix(c(1, 2, 3, 4), ncol = 1L)
  expect_error(.forest_fit(x, c(0, 1, 0, 2), rep(1, 4), "binomial", 2L, 1L, 1L, FALSE, 1),
               "0 and 1 alone")
  expect_error(.forest_fit(x, c(0, 1, NA, 1), rep(1, 4), "binomial", 2L, 1L, 1L, FALSE, 1),
               "finite values")
  expect_error(.forest_fit(x, c(0, 1, 0, 1), c(1, -1, 1, 1), "binomial", 2L, 1L, 1L, FALSE, 1),
               "zero or more")
  expect_error(.forest_fit(x, c(0, 1, 0, 1), rep(1, 4), "binomial", 2L, 2L, 1L, FALSE, 1),
               "between 1 and the 1 columns")
  expect_error(.forest_fit(x, c(0, 1, 0, 1), rep(0, 4), "binomial", 2L, 1L, 1L, FALSE, 1),
               "sum to more than zero")
  fit <- .forest_fit(x, c(0, 0, 1, 1), rep(1, 4), "binomial", 2L, 1L, 1L, FALSE, 1)
  expect_error(.forest_predict(fit, matrix(NA_real_, 1L, 1L)), "finite values")
})

test_that("a preset fills the settings left open, as randomForest and biomod2 have them", {
  expect_identical(.forest_settings("package", "binomial", 471L, NULL, NULL, NULL),
                   list(trees = 500L, mtry = 21L, min_node = 1L))
  expect_identical(.forest_settings("package", "gaussian", 471L, NULL, NULL, NULL),
                   list(trees = 500L, mtry = 157L, min_node = 5L))
  expect_identical(.forest_settings("package", "gaussian", 2L, NULL, NULL, NULL)$mtry, 1L)
  expect_identical(.forest_settings("bigboss", "binomial", 471L, NULL, NULL, NULL),
                   list(trees = 500L, mtry = 2L, min_node = 5L))
  expect_identical(.forest_settings("bigboss", "binomial", 1L, 100L, NULL, 3L),
                   list(trees = 100L, mtry = 1L, min_node = 3L))
})

test_that("a forest fits, predicts, survives a round trip and refuses a different representation", {
  sim <- sim_series(n_unit = 60L, days = 60L, seed = 35L)
  y <- sim_response(sim, n_var = 2L, seed = 36L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "week")
  fit <- fit_learner(forest(trees = 200L), x, y)
  p <- stats::predict(fit, x)
  expect_equal(dim(p), c(60L, 2L))
  expect_true(all(p >= 0 & p <= 1))
  expect_gt(tss(y[, 1], p[, 1]), 0.4)
  expect_equal(fit$model$columns, colnames(.flatten(x)))
  path <- tempfile(fileext = ".rds")
  on.exit(unlink(path), add = TRUE)
  saveRDS(fit, path)
  expect_identical(stats::predict(readRDS(path), x), p)
  other <- grain_matrix(sim$readings, plot, t, temp, grain = "month")
  expect_error(stats::predict(fit, other), "different channels or bins")
  balanced <- stats::predict(fit_learner(forest(trees = 100L, balance = TRUE), x, y), x)
  expect_gt(tss(y[, 1], balanced[, 1]), 0.4)
})

test_that("a forest fits the family the response head's loss names, and a constant is its mean", {
  local_response("continuous_test", list(
    prepare = function(y) .as_response(y), activation = "identity",
    loss = "squared_error", metric = "roc_auc",
    cells = function(y, folds) scorable_cells(y > stats::median(y), folds)))
  sim <- sim_series(n_unit = 60L, days = 60L, seed = 91L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "week")
  level <- 10 + 3 * scale(rowMeans(x[, , 1L]))[, 1L]
  y <- matrix(level, ncol = 1L, dimnames = list(dimnames(x)[[1L]], "height"))
  fit <- fit_learner(forest(trees = 100L), x, y, response = "continuous_test")
  expect_equal(fit$model$family, "gaussian")
  expect_gt(stats::cor(stats::predict(fit, x)[, 1L], y[, 1L]), 0.8)
  expect_error(fit_learner(forest(trees = 10L, balance = TRUE), x, y,
                           response = "continuous_test"), "binomial response")

  flat <- matrix(0, nrow = 60L, ncol = 1L, dimnames = list(dimnames(x)[[1L]], "absent"))
  expect_equal(unique(stats::predict(fit_learner(forest(trees = 10L), x, flat), x)[, 1L]), 0)
})
