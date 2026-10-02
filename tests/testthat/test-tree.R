# The tree core, against the reference the fixtures carry, and the learner above it.
#
# The reference is rpart's, grown on the design the penalised fixtures carry under integer case
# weights, where the core's class priors and rpart's are the same numbers. The node table, the
# complexity table and its cross-validated error are asserted to rounding, and the Python suite
# reads the same files.

tree_fixture <- function() {
  dir <- fixture_dir()
  if (is.null(dir)) skip("the fixtures ship with the package and are not installed here")
  d <- utils::read.csv(file.path(dir, "penalised_input.csv"), stringsAsFactors = FALSE,
                       check.names = FALSE)
  held <- c("unit", "y_gaussian", "y_binomial", "w", "fold")
  x <- as.matrix(d[, setdiff(names(d), held), drop = FALSE])
  counts <- utils::read.csv(file.path(dir, "tree_weights.csv"), stringsAsFactors = FALSE)
  list(x = x, binomial = d$y_binomial, gaussian = d$y_gaussian,
       poisson = fixture_counts(dir, d$unit), fold = d$fold,
       count = counts$count[match(d$unit, counts$unit)],
       cases = utils::read.csv(file.path(dir, "tree_cases.csv"), stringsAsFactors = FALSE),
       nodes = utils::read.csv(file.path(dir, "tree_nodes.csv"), stringsAsFactors = FALSE),
       table = utils::read.csv(file.path(dir, "tree_cptable.csv"), stringsAsFactors = FALSE),
       predict = utils::read.csv(file.path(dir, "tree_predict.csv"), stringsAsFactors = FALSE))
}

tree_case_fit <- function(fx, row) {
  y <- fx[[row$family]]
  w <- if (row$weights == "counts") fx$count else rep(1, nrow(fx$x))
  .tree_fit(fx$x, y, w, row$family, row$min_split, row$min_leaf, row$cp, row$max_depth,
            fx$fold, 5L, row$shrink)
}

test_that("the tree core grows rpart's tree node for node", {
  fx <- tree_fixture()
  for (i in seq_len(nrow(fx$cases))) {
    row <- fx$cases[i, ]
    fit <- tree_case_fit(fx, row)
    ref <- fx$nodes[fx$nodes$case == row$case, ]
    expect_identical(fit$number, ref$number, info = row$case)
    expect_identical(fit$column, ref$column, info = row$case)
    expect_identical(fit$less_left, ref$less_left, info = row$case)
    expect_identical(fit$n, ref$n, info = row$case)
    for (field in c("threshold", "weight", "risk", "complexity", "value")) {
      expect_equal(fit[[field]], ref[[field]], tolerance = 1e-12, info = paste(row$case, field))
    }
  }
})

test_that("the tree core's complexity table and its cross-validated error are rpart's", {
  fx <- tree_fixture()
  for (i in seq_len(nrow(fx$cases))) {
    row <- fx$cases[i, ]
    fit <- tree_case_fit(fx, row)
    ref <- fx$table[fx$table$case == row$case, ]
    expect_identical(fit$nsplit, ref$nsplit, info = row$case)
    for (field in c("cp", "rel_error", "xerror", "xstd")) {
      expect_equal(fit[[field]], ref[[field]], tolerance = 1e-12, info = paste(row$case, field))
    }
  }
})

test_that("pruned by biomod2's rule, the tree predicts what rpart's pruned tree does", {
  fx <- tree_fixture()
  for (i in seq_len(nrow(fx$cases))) {
    row <- fx$cases[i, ]
    fit <- tree_case_fit(fx, row)
    at <- .tree_prune_cp(fit, "se_sum")
    pruned <- if (is.null(at)) fit else .tree_prune(fit, at)
    ref <- fx$predict[fx$predict$case == row$case, ]
    expect_equal(.tree_predict(pruned, fx$x), ref$se_sum, tolerance = 1e-12, info = row$case)
  }
})

test_that("pruning collapses every split at or below the complexity and keeps the table", {
  fx <- tree_fixture()
  row <- fx$cases[fx$cases$case == "gaussian_counts_bigboss", ]
  fit <- tree_case_fit(fx, row)
  at <- fit$cp[3L]
  pruned <- .tree_prune(fit, at)
  expect_true(all(pruned$complexity[pruned$column >= 0L] > at))
  expect_identical(sum(pruned$column >= 0L), fit$nsplit[3L])
  expect_identical(pruned$cp, fit$cp)
  expect_identical(.tree_prune(fit, 0)$number, fit$number)
})

test_that("each pruning rule reads the complexity table the way it says", {
  tree <- list(cp = c(0.5, 0.2, 0.1, 0.05, 0.01), nsplit = c(0L, 1L, 2L, 4L, 7L),
               xerror = c(1, 0.58, 0.5, 0.55, 0.5), xstd = c(0.1, 0.08, 0.09, 0.07, 0.09))
  expect_equal(.tree_prune_cp(tree, "min"), 0.1)
  expect_equal(.tree_prune_cp(tree, "one_se"), 0.2)
  # 0.58 + 0.08, 0.5 + 0.09, 0.55 + 0.07, 0.5 + 0.09: the last of the two least.
  expect_equal(.tree_prune_cp(tree, "se_sum"), 0.01)
  expect_null(.tree_prune_cp(tree, "none"))
  expect_null(.tree_prune_cp(list(cp = 1, nsplit = 0L, xerror = 1, xstd = 0), "se_sum"))
  expect_null(.tree_prune_cp(list(cp = 1, nsplit = 0L, xerror = numeric(0), xstd = numeric(0)),
                             "min"))
})

test_that("the tree core refuses what it cannot grow on", {
  x <- matrix(c(1, 2, 3, 4), ncol = 1L)
  expect_error(.tree_fit(x, c(0, 1, 0, 2), rep(1, 4), "binomial", 2L, 1L, 0, 30L),
               "0 and 1 alone")
  expect_error(.tree_fit(x, c(0, 1, -2, 2), rep(1, 4), "poisson", 2L, 1L, 0, 30L),
               "zero or more")
  expect_error(.tree_fit(x, c(0, 0, 0, 0), rep(1, 4), "poisson", 2L, 1L, 0, 30L),
               "one count above zero")
  expect_error(.tree_fit(x, c(0, 1, NA, 1), rep(1, 4), "binomial", 2L, 1L, 0, 30L),
               "finite values")
  expect_error(.tree_fit(x, c(0, 1, 0, 1), c(1, -1, 1, 1), "binomial", 2L, 1L, 0, 30L),
               "zero or more")
  expect_error(.tree_fit(x, c(0, 1, 0, 1), rep(1, 4), "binomial", 2L, 1L, 0, 31L),
               "between 0 and 30")
  fit <- .tree_fit(x, c(0, 0, 1, 1), rep(1, 4), "binomial", 2L, 1L, 0, 30L)
  expect_error(.tree_predict(fit, matrix(NA_real_, 1L, 1L)), "finite values")
})

test_that("a preset fills the settings left open, as rpart and biomod2 have them", {
  p <- tree()$params
  expect_identical(p[c("min_split", "min_leaf", "max_depth", "n_inner")],
                   list(min_split = 20L, min_leaf = 7L, max_depth = 30L, n_inner = 10L))
  expect_equal(p$cp, 0.01)
  expect_identical(tree(min_split = 30L)$params$min_leaf, 10L)
  expect_identical(tree(min_leaf = 4L)$params$min_split, 12L)
  b <- tree(preset = "bigboss")$params
  expect_identical(b[c("min_split", "min_leaf", "max_depth", "n_inner")],
                   list(min_split = 5L, min_leaf = 5L, max_depth = 10L, n_inner = 5L))
  expect_equal(b$cp, 0.001)
  expect_identical(tree(preset = "bigboss", min_split = 12L)$params$min_leaf, 5L)
})

test_that("a tree fits, predicts, survives a round trip and refuses a different representation", {
  sim <- sim_series(n_unit = 60L, days = 60L, seed = 35L)
  y <- sim_response(sim, n_var = 2L, seed = 36L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "week")
  fit <- fit_learner(tree(), x, y)
  p <- stats::predict(fit, x)
  expect_equal(dim(p), c(60L, 2L))
  expect_true(all(p >= 0 & p <= 1))
  expect_gt(tss(y[, 1], p[, 1]), 0.4)
  expect_equal(fit$model$columns, colnames(.flatten(x)))
  path <- tempfile(fileext = ".rds")
  on.exit(unlink(path), add = TRUE)
  saveRDS(fit, path)
  expect_equal(stats::predict(readRDS(path), x), p)
  other <- grain_matrix(sim$readings, plot, t, temp, grain = "month")
  expect_error(stats::predict(fit, other), "different channels or bins")
})

test_that("an unpruned tree keeps at least the leaves a pruned one does", {
  sim <- sim_series(n_unit = 80L, days = 60L, seed = 41L)
  y <- sim_response(sim, n_var = 1L, seed = 42L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "week")
  grown <- fit_learner(tree(prune = "none", cp = 0), x, y)$model$models[[1L]]
  pruned <- fit_learner(tree(cp = 0), x, y)$model$models[[1L]]
  expect_gte(sum(grown$column < 0L), sum(pruned$column < 0L))
  expect_length(grown$xerror, 0L)
})

test_that("a tree fits the family the response head's loss names, and a constant is its mean", {
  local_response("continuous_test", list(
    prepare = function(y) .as_response(y), activation = "identity",
    loss = "squared_error", metric = "roc_auc",
    cells = function(y, folds) scorable_cells(y > stats::median(y), folds)))
  sim <- sim_series(n_unit = 60L, days = 60L, seed = 91L)
  x <- grain_matrix(sim$readings, plot, t, temp, grain = "week")
  level <- 10 + 3 * scale(rowMeans(x[, , 1L]))[, 1L]
  y <- matrix(level, ncol = 1L, dimnames = list(dimnames(x)[[1L]], "height"))
  fit <- fit_learner(tree(min_split = 6L), x, y, response = "continuous_test")
  expect_equal(fit$model$family, "gaussian")
  expect_gt(stats::cor(stats::predict(fit, x)[, 1L], y[, 1L]), 0.8)

  flat <- matrix(0, nrow = 60L, ncol = 1L, dimnames = list(dimnames(x)[[1L]], "absent"))
  expect_equal(unique(stats::predict(fit_learner(tree(), x, flat), x)[, 1L]), 0)
})
