# The hierarchical model on the core, against tulpa's own fits and against a dense implementation of
# the same Laplace marginal, and the learner above it.
#
# The tulpa reference is written by `inst/spec/make_fixtures.R` at the levels where its answer is
# deterministic: the posterior mode, the empirical-Bayes fit, and each node of the nested grid. The
# Python suite reads the same files.

hier_fixture <- function() {
  dir <- fixture_dir()
  if (is.null(dir)) skip("the fixtures ship with the package and are not installed here")
  read <- function(f) utils::read.csv(file.path(dir, f), stringsAsFactors = FALSE,
                                      check.names = FALSE)
  d <- read("hierarchical_input.csv")
  list(d = d, X = cbind(1, d$x1, d$x2), coords = cbind(d$lon, d$lat),
       units = sort(unique(d$unit)), cases = read("hierarchical_cases.csv"),
       ranef = read("hierarchical_ranef.csv"), hsgp = read("hierarchical_hsgp_nodes.csv"),
       nngp = read("hierarchical_nngp_nodes.csv"))
}

hier_unit_index <- function(fx) match(fx$d$unit, fx$units) - 1L

test_that("a hierarchical learner declares its settings", {
  l <- hierarchical(spatial = "hsgp", m = 8L)
  expect_equal(l$params$spatial, "hsgp")
  expect_equal(l$params$m, 8L)
  expect_length(l$needs, 0L)
  expect_error(hierarchical(spatial = "gp"), "should be one of")
  expect_error(hierarchical(random = NA), "TRUE or FALSE")
  expect_error(hierarchical(neighbours = 0), "`neighbours` is one whole number")
  expect_error(hierarchical(m = 2), "`m` is one whole number")
  expect_error(hierarchical(boundary = 0.5), "`boundary` is one number")
  expect_error(hierarchical(cov = "spherical"), "should be one of")
})

test_that("coords names two numeric columns and reaches the array, split with its units", {
  expect_error(.check_coords(data.frame(a = 1, b = 2, c = 3), c("a", "b", "c")), "two columns")
  expect_error(.check_coords(data.frame(a = 1, b = "x"), c("a", "b")), "must be numeric")
  expect_error(.check_coords(data.frame(a = 1, b = NA_real_), c("a", "b")), "must be numeric")
  x <- feature_matrix(matrix(seq_len(8), 4L, 2L, dimnames = list(letters[1:4], c("u", "v"))))
  attr(x, "coords") <- matrix(1:8, 4L, 2L, dimnames = list(letters[1:4], c("lon", "lat")))
  sub <- .subset_units(x, c(3L, 1L))
  expect_equal(attr(sub, "coords"), attr(x, "coords")[c(3L, 1L), ])
})

test_that("the posterior mode of the coefficients is tulpa's, weighted and not", {
  fx <- hier_fixture()
  for (i in which(fx$cases$case %in% c("map", "map_weighted"))) {
    row <- fx$cases[i, ]
    w <- if (row$weighted) fx$d$weight else rep(1, nrow(fx$X))
    fit <- .hier_fit(fx$X, fx$d$y, w)
    expect_equal(fit$beta, unlist(row[c("b0", "b1", "b2")], use.names = FALSE),
                 tolerance = row$tolerance, info = row$case)
    expect_equal(fit$log_marginal, row$log_marginal, tolerance = row$tolerance, info = row$case)
  }
})

test_that("the empirical-Bayes fit with an intercept for each unit is tulpa's", {
  fx <- hier_fixture()
  row <- fx$cases[fx$cases$case == "eb", ]
  fit <- .hier_fit(fx$X, fx$d$y, rep(1, nrow(fx$X)), unit = hier_unit_index(fx),
                   n_unit = length(fx$units))
  expect_equal(exp(fit$theta_hat), row$sd_unit, tolerance = 1e-5)
  expect_equal(fit$beta, unlist(row[c("b0", "b1", "b2")], use.names = FALSE), tolerance = 1e-6)
  expect_equal(fit$log_marginal, row$log_marginal, tolerance = 1e-6)
  expect_equal(fit$unit_effect, fx$ranef$effect[match(fx$units, fx$ranef$unit)], tolerance = 1e-5)
})

test_that("each node of a field's grid is tulpa's conditional fit, and the weights follow", {
  fx <- hier_fixture()
  n <- nrow(fx$X)
  place <- oracle_locations(oracle_standardise(fx$coords)$xy)$of_target
  for (field in c("hsgp", "nngp")) {
    nodes <- fx[[field]]
    gap <- numeric(nrow(nodes))
    for (k in seq_len(nrow(nodes))) {
      row <- nodes[k, ]
      fit <- .hier_fit(fx$X, fx$d$y, rep(1, n), coords = fx$coords, field = field,
                       beta_sd = 100, theta = log(c(sqrt(row$sigma2), row$range)))
      coef <- unlist(row[grep("^c[0-9]+$", names(row))], use.names = FALSE)
      mine <- as.numeric(fit$node_field)
      if (identical(field, "nngp")) mine <- mine[place]
      expect_equal(fit$beta, unlist(row[c("b0", "b1", "b2")], use.names = FALSE),
                   tolerance = row$tolerance, info = paste(field, k))
      expect_equal(mine, coef, tolerance = row$tolerance, info = paste(field, k))
      gap[k] <- fit$node_log_post - row$log_weight
    }
    expect_lt(diff(range(gap)), nodes$weight_tolerance[1L])
  }
})

test_that("a model with unit intercepts and a nearest-neighbour field is its dense Laplace marginal", {
  fx <- hier_fixture()
  n <- nrow(fx$X)
  st <- oracle_standardise(fx$coords)
  place <- oracle_locations(st$xy)
  index <- hier_unit_index(fx)
  w <- fx$d$weight
  for (cov in c("exponential", "matern32", "matern52", "gaussian")) {
    sigma_u <- 0.8
    sigma <- 1.3
    range <- 0.45
    fit <- .hier_fit(fx$X, fx$d$y, w, unit = index, n_unit = length(fx$units), coords = fx$coords,
                     field = "nngp", neighbours = 5L, cov = match(cov, .hier_covs) - 1L,
                     theta = log(c(sigma_u, sigma, range)))
    v <- oracle_vecchia(place$loc, sigma, range, cov, 5L)
    S <- matrix(0, n, nrow(place$loc))
    S[cbind(seq_len(n), place$of_target)] <- 1
    ref <- oracle_laplace(fx$X, fx$d$y, w, S, v$precision, v$log_det, index, sigma_u, 2.5)
    expect_equal(fit$log_marginal, ref$log_marginal, tolerance = 1e-8, info = cov)
    expect_equal(c(fit$beta, as.numeric(fit$node_field), fit$unit_effect), ref$latent,
                 tolerance = 1e-7, info = cov)
  }
})

test_that("a model with unit intercepts and a Hilbert-space field is its dense Laplace marginal", {
  fx <- hier_fixture()
  n <- nrow(fx$X)
  st <- oracle_standardise(fx$coords)
  sigma_u <- 0.8
  sigma <- 1.3
  range <- 0.45
  fit <- .hier_fit(fx$X, fx$d$y, fx$d$weight, unit = hier_unit_index(fx), n_unit = length(fx$units),
                   coords = fx$coords, field = "hsgp", theta = log(c(sigma_u, sigma, range)))
  basis <- oracle_hsgp(st$xy, sigma, range)
  ref <- oracle_laplace(fx$X, fx$d$y, fx$d$weight, basis$design, diag(ncol(basis$design)), 0,
                        hier_unit_index(fx), sigma_u, 2.5)
  expect_equal(fit$log_marginal, ref$log_marginal, tolerance = 1e-8)
  expect_equal(c(fit$beta, as.numeric(fit$node_field), fit$unit_effect), ref$latent,
               tolerance = 1e-7)
})

test_that("a nearest-neighbour field interpolates to new places as its conditional mean", {
  fx <- hier_fixture()
  n <- nrow(fx$X)
  st <- oracle_standardise(fx$coords)
  place <- oracle_locations(st$xy)
  sigma <- 1.3
  range <- 0.45
  fit <- .hier_fit(fx$X, fx$d$y, rep(1, n), coords = fx$coords, field = "nngp", neighbours = 6L,
                   theta = log(c(sigma, range)))
  set.seed(5)
  newx <- cbind(1, stats::rnorm(8), stats::rnorm(8))
  new_coords <- cbind(stats::runif(8), stats::runif(8))
  got <- .hier_predict(fit, newx, coords = new_coords)
  xy <- sweep(new_coords, 2L, st$centre) / st$scale
  field <- as.numeric(fit$node_field)
  for (i in 1:8) {
    d <- sqrt(rowSums((place$loc - rep(xy[i, ], each = nrow(place$loc)))^2))
    nb <- order(d)[1:6]
    C <- oracle_kernel("exponential", as.matrix(stats::dist(place$loc[nb, ])), sigma^2, range)
    diag(C) <- diag(C) + 1e-6
    cv <- oracle_kernel("exponential", d[nb], sigma^2, range)
    expect_equal(got[i], sum(newx[i, ] * fit$beta) + sum(cv * solve(C, field[nb])),
                 tolerance = 1e-10)
  }
  at_training <- .hier_predict(fit, fx$X, coords = fx$coords)
  expect_equal(at_training, as.numeric(fx$X %*% fit$beta) + field[place$of_target], tolerance = 1e-5)
})

test_that("a fit refuses what it cannot read", {
  fx <- hier_fixture()
  w <- rep(1, nrow(fx$X))
  expect_error(.hier_fit(fx$X, fx$d$y * 2, w), "zero and one")
  expect_error(.hier_fit(fx$X, fx$d$y, -w), "zero or more")
  expect_error(.hier_fit(fx$X, fx$d$y, w, field = "nngp"), "coordinates")
  expect_error(.hier_fit(fx$X, fx$d$y, w, coords = fx$coords, field = "nngp", cov = 9L),
               "covariance")
  expect_error(.hier_fit(fx$X, fx$d$y, w, coords = fx$coords, field = "hsgp", theta = 0),
               "hyperparameters")
})

hier_run <- function(spatial, ...) {
  set.seed(2)
  n <- 60L
  targets <- data.frame(plot = sprintf("p%03d", seq_len(n)), lon = stats::runif(n),
                        lat = stats::runif(n))
  targets$s1 <- stats::rbinom(n, 1L, stats::plogis(-0.3 + 2 * sin(5 * targets$lon)))
  targets$s2 <- stats::rbinom(n, 1L, stats::plogis(1.5 * (targets$lat - 0.5)))
  days <- seq(as.POSIXct("2020-01-01", tz = "UTC"), by = "12 hours", length.out = 240L)
  series <- do.call(rbind, lapply(targets$plot, function(p) {
    data.frame(plot = p, t = days, v = stats::rnorm(length(days)))
  }))
  list(targets = targets, series = series,
       fit = timesift(targets, series, y = tidyselect::starts_with("s"), id = plot, time = t, ...,
                      models = hierarchical(spatial = spatial), sift = grains("month"),
                      resampling = cv(v = 3L), inner = NULL, ensemble = FALSE, verbose = FALSE))
}

test_that("a field needs the coordinates, and a fit without one does not", {
  expect_error(hier_run("hsgp"), "coords = ")
  run <- hier_run("none")
  p <- stats::predict(run$fit, run$targets[1:5, ], run$series, candidate = "selected")
  expect_equal(dim(p), c(5L, 2L))
  expect_true(all(p > 0 & p < 1))
})

test_that("a field predicts from the coordinates of the new units, and survives saveRDS", {
  for (spatial in c("hsgp", "nngp")) {
    run <- hier_run(spatial, coords = c(lon, lat))
    new <- run$targets[1:6, ]
    p <- stats::predict(run$fit, new, run$series, candidate = "selected")
    expect_equal(dim(p), c(6L, 2L))
    moved <- new
    moved$lon <- 1 - moved$lon
    expect_false(isTRUE(all.equal(p, stats::predict(run$fit, moved, run$series,
                                                    candidate = "selected"))), info = spatial)
    path <- tempfile(fileext = ".rds")
    saveRDS(run$fit, path)
    expect_equal(stats::predict(readRDS(path), new, run$series, candidate = "selected"), p)
  }
})

test_that("an intercept for each unit needs the unit, and a held-out unit is predicted at the population level", {
  set.seed(3)
  n <- 40L
  targets <- data.frame(plot = rep(sprintf("p%02d", 1:20), each = 2L), when = rep(0:1, 20L))
  targets$t0 <- as.POSIXct("2020-06-01", tz = "UTC") + targets$when * 86400
  shift <- stats::rnorm(20L)[rep(1:20, each = 2L)]
  targets$s1 <- stats::rbinom(n, 1L, stats::plogis(shift))
  series <- do.call(rbind, lapply(unique(targets$plot), function(p) {
    data.frame(plot = p, t = as.POSIXct("2020-01-01", tz = "UTC") + 3600 * (0:3000),
               v = stats::rnorm(3001L))
  }))
  fit <- timesift(targets, series, y = s1, id = plot, time = t, target_time = t0,
                  models = hierarchical(random = TRUE), sift = lookbacks("30 days"),
                  resampling = grouped_cv("plot", v = 2L), inner = NULL, ensemble = FALSE,
                  verbose = FALSE)
  p <- stats::predict(fit, targets, series, candidate = "selected")
  expect_equal(dim(p), c(n, 1L))
  expect_true(all(p > 0 & p < 1))
  none <- timesift(targets, series, y = s1, id = plot, time = t, target_time = t0,
                   models = hierarchical(), sift = lookbacks("30 days"),
                   resampling = grouped_cv("plot", v = 2L), inner = NULL, ensemble = FALSE,
                   verbose = FALSE)
  expect_false(isTRUE(all.equal(p, stats::predict(none, targets, series, candidate = "selected"))))
})
