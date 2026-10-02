# The hierarchical model's Laplace marginal written out densely, for the tests to hold the core to:
# the latent vector of coefficients, field values and unit intercepts is solved by Newton's method
# on full matrices, and the marginal is the penalised log likelihood at the mode plus half the log
# determinant of the prior's precision less half that of the information.

oracle_kernel <- function(cov, d, sigma2, range) {
  switch(cov,
         exponential = sigma2 * exp(-d / range),
         matern32 = sigma2 * (1 + sqrt(3) * d / range) * exp(-sqrt(3) * d / range),
         matern52 = sigma2 * (1 + sqrt(5) * d / range + 5 * d^2 / (3 * range^2)) *
           exp(-sqrt(5) * d / range),
         gaussian = sigma2 * exp(-(d / range)^2))
}

# The coordinates centred and divided by the root of the mean of the columns' variances.
oracle_standardise <- function(coords) {
  centre <- colMeans(coords)
  scale <- sqrt(mean(apply(coords, 2L, stats::var)))
  list(xy = sweep(coords, 2L, centre) / scale, centre = centre, scale = scale)
}

# The distinct locations in lexicographic order, and each target's place among them.
oracle_locations <- function(xy) {
  loc <- unique(xy)
  loc <- loc[order(loc[, 1L], loc[, 2L]), , drop = FALSE]
  list(loc = loc, of_target = match(paste(xy[, 1L], xy[, 2L]), paste(loc[, 1L], loc[, 2L])))
}

# `(I - A)' D^-1 (I - A)` over the locations, each conditioned on its `nn` nearest earlier ones, and
# its log determinant.
oracle_vecchia <- function(loc, sigma, range, cov, nn) {
  L <- nrow(loc)
  A <- matrix(0, L, L)
  D <- numeric(L)
  D[1L] <- sigma^2
  for (i in seq_len(L)[-1L]) {
    earlier <- loc[seq_len(i - 1L), , drop = FALSE]
    d <- sqrt(rowSums((earlier - rep(loc[i, ], each = i - 1L))^2))
    nb <- order(d)[seq_len(min(nn, i - 1L))]
    C <- oracle_kernel(cov, as.matrix(stats::dist(earlier[nb, , drop = FALSE])), sigma^2, range)
    diag(C) <- diag(C) + 1e-8
    cv <- oracle_kernel(cov, d[nb], sigma^2, range)
    a <- solve(C, cv)
    A[i, nb] <- a
    D[i] <- max(sigma^2 - sum(cv * a), 1e-10)
  }
  IA <- diag(L) - A
  list(precision = crossprod(IA, IA / D), log_det = -sum(log(D)))
}

# The Laplace marginal and the latent mode at a given precision of the field.
oracle_laplace <- function(X, y, w, field_design, field_precision, field_log_det, unit, sigma_u,
                           beta_sd) {
  n <- nrow(X)
  p <- ncol(X)
  m <- ncol(field_design)
  g <- if (is.null(unit)) 0L else max(unit) + 1L
  U <- matrix(0, n, g)
  if (g > 0L) U[cbind(seq_len(n), unit + 1L)] <- 1
  Z <- cbind(X, field_design, U)
  tau_u <- if (g > 0L) 1 / sigma_u^2 else 1
  Q <- matrix(0, p + m + g, p + m + g)
  Q[seq_len(p), seq_len(p)] <- diag(1 / beta_sd^2, p)
  if (m > 0L) Q[p + seq_len(m), p + seq_len(m)] <- field_precision
  if (g > 0L) Q[p + m + seq_len(g), p + m + seq_len(g)] <- diag(tau_u, g)
  value <- function(th) {
    eta <- as.numeric(Z %*% th)
    sum(w * (y * eta - log1p(exp(eta)))) - 0.5 * sum(th * (Q %*% th))
  }
  th <- numeric(p + m + g)
  for (it in 1:200) {
    pr <- stats::plogis(as.numeric(Z %*% th))
    grad <- crossprod(Z, w * (y - pr)) - Q %*% th
    H <- crossprod(Z, (w * pr * (1 - pr)) * Z) + Q
    step <- as.numeric(solve(H, grad))
    t <- 1
    while (value(th + t * step) < value(th) - 1e-12 && t > 1e-8) t <- t / 2
    th <- th + t * step
    if (max(abs(t * step)) < 1e-13) break
  }
  pr <- stats::plogis(as.numeric(Z %*% th))
  H <- crossprod(Z, (w * pr * (1 - pr)) * Z) + Q
  log_prior_det <- p * log(1 / beta_sd^2) + g * log(tau_u) + field_log_det
  list(latent = th,
       log_marginal = value(th) + 0.5 * log_prior_det -
         0.5 * as.numeric(determinant(H, logarithm = TRUE)$modulus))
}

# The Hilbert-space basis at standardised coordinates, scaled by the spectral weights at `sigma` and
# `range`.
oracle_hsgp <- function(xy, sigma, range, m = 6L, boundary = 1.5) {
  centre <- (apply(xy, 2L, max) + apply(xy, 2L, min)) / 2
  half <- pmax(boundary * (apply(xy, 2L, max) - apply(xy, 2L, min)) / 2, 0.1)
  out <- matrix(0, nrow(xy), m * m)
  lambda <- numeric(m * m)
  for (j1 in 1:m) for (j2 in 1:m) {
    col <- (j1 - 1L) * m + j2
    out[, col] <- sin(pi * j1 * (xy[, 1L] - centre[1L] + half[1L]) / (2 * half[1L])) / sqrt(half[1L]) *
      sin(pi * j2 * (xy[, 2L] - centre[2L] + half[2L]) / (2 * half[2L])) / sqrt(half[2L])
    lambda[col] <- (pi * j1 / (2 * half[1L]))^2 + (pi * j2 / (2 * half[2L]))^2
  }
  s <- sqrt(sigma^2 * 2 * pi * range^2 * exp(-0.5 * range^2 * lambda))
  list(design = sweep(out, 2L, s, "*"), basis = out, scale = s)
}
