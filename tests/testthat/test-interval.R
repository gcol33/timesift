# The interval for the procedure's risk: the pieces of the estimator, and its coverage on a design
# whose truth is measured rather than assumed.

test_that("the jackknife variance of a mean of per-unit losses is the variance over the count", {
  set.seed(4)
  e <- rnorm(60)
  got <- .jackknife_var(length(e), function(i) mean(e[-i]))
  expect_equal(got, stats::var(e) / length(e))
})

test_that("the interval rescales, bounds and bias-corrects as the paper states", {
  terms <- list(
    terms = data.frame(`repeat` = 1L, fold = 1:5, s_in = c(0.70, 0.72, 0.68, 0.71, 0.69),
                       s_out = c(0.74, 0.70, 0.72, 0.75, 0.69), b = rep(0.0004, 5L),
                       check.names = FALSE),
    estimate = 0.72, se_naive = 0.01, folds = 5L, n_variable = 10L)
  got <- .ncv_interval(terms)
  t <- terms$terms
  mse_ncv <- mean((t$s_in - t$s_out)^2) - mean(t$b)
  expect_equal(got$mse_ncv, mse_ncv)
  # Rescaled by (K - 1) / K, then held between the naive standard error and sqrt(K) times it.
  expect_equal(got$se, min(sqrt(4 / 5 * mse_ncv), sqrt(5) * 0.01))
  expect_equal(got$err_ncv, mean(t$s_in))
  expect_equal(got$bias, (1 + 3 / 5) * (mean(t$s_in) - 0.72))
  expect_equal(got$center, mean(t$s_in) - got$bias)
  expect_equal(c(got$lower, got$upper),
               got$center + c(-1, 1) * stats::qnorm(0.975) * got$se)

  # A mean squared error estimated below the naive standard error is raised to it, and one above
  # sqrt(K) times it is cut back, so neither bound can be crossed.
  tiny <- terms
  tiny$terms$s_out <- tiny$terms$s_in
  expect_equal(.ncv_interval(tiny)$se, 0.01)
  wide <- terms
  wide$terms$s_out <- wide$terms$s_in + 1
  expect_equal(.ncv_interval(wide)$se, sqrt(5) * 0.01)
})

# A cheap design whose truth is known by measurement: each variable's driver is standard normal,
# the response is Bernoulli on it, and the representation the procedure chooses between is that
# driver read through a little noise or through a lot. Twenty thousand further units drawn from
# the same design are what the procedure fitted on the whole sample is scored on, so the interval
# has a number to cover rather than an assumption.
ncv_design <- function(n, seed, variables = 3L) {
  set.seed(seed)
  d <- matrix(stats::rnorm(n * variables), n, variables)
  y <- matrix(stats::rbinom(n * variables, 1L, stats::plogis(-1 + 1.2 * d)), n, variables)
  dimnames(y) <- list(sprintf("s%d_%05d", seed, seq_len(n)), sprintf("v%d", seq_len(variables)))
  good <- d + matrix(stats::rnorm(n * variables, sd = 0.7), n, variables)
  noisy <- d + matrix(stats::rnorm(n * variables, sd = 2), n, variables)
  dimnames(good) <- dimnames(noisy) <- dimnames(y)
  list(y = y, x = timesift_set(list(good = feature_matrix(good, "good"),
                                    noisy = feature_matrix(noisy, "noisy"))))
}

ncv_learner <- function() {
  learner("centroid",
          fit = function(x, y, ...) {
            m <- matrix(x[, , 1], nrow = dim(x)[1L])
            list(w = vapply(seq_len(ncol(y)), function(j) {
              colMeans(m[y[, j] == 1, , drop = FALSE]) - colMeans(m[y[, j] == 0, , drop = FALSE])
            }, numeric(ncol(m))))
          },
          predict = function(model, x) matrix(x[, , 1], nrow = dim(x)[1L]) %*% model$w)
}

test_that("the nested cross-validation interval covers the procedure's risk at the nominal rate", {
  skip_on_cran()
  replicates <- 150L
  deployment <- ncv_design(20000L, 999999L)
  rows <- lapply(seq_len(replicates), function(r) {
    s <- ncv_design(150L, r)
    sel <- select_grain(s$x, s$y, ncv_learner(), folds = fold_map(s$y, v = 5L, seed = r),
                        inner = 3L, metric = "roc_auc", interval = "nested_cv", repeats = 1L,
                        seed = r, verbose = FALSE)
    # The truth: the procedure fitted on every unit of the sample, scored on units it never saw.
    p <- stats::predict(sel$final$fit, deployment$x[[sel$final$grain]])
    truth <- mean(vapply(colnames(deployment$y),
                         function(v) roc_auc(deployment$y[, v], p[, v]), numeric(1L)))
    est <- sel$estimate[sel$estimate$metric == "roc_auc", ]
    data.frame(truth = truth, est[c("interval", "center", "se", "lower", "upper")])
  })
  d <- do.call(rbind, rows)
  d$cover <- d$lower <= d$truth & d$truth <= d$upper
  nested <- d[d$interval == "nested_cv", ]
  across <- d[d$interval == "variables", ]

  # Coverage inside its own Monte Carlo margin of the nominal rate.
  margin <- stats::qnorm(0.975) * sqrt(0.95 * 0.05 / replicates)
  expect_equal(mean(nested$cover), 0.95, tolerance = margin, scale = 1)
  # The across-variable interval is the narrower one, which is what it is not an interval for the
  # procedure's risk: its width is the spread between variables.
  expect_gt(mean(nested$se), mean(across$se))
  expect_lt(abs(mean(nested$center - nested$truth)), 0.02)
})
