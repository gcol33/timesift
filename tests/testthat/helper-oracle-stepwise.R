# The forward search over column terms in R alone: `stats::glm()` for every fit, R's own `poly()`
# for the basis, and a candidate refused where the fitter warns. Nothing dispatches to this and the
# package never reaches it at runtime. The core is asserted against it on the forward search over
# column terms, which is the arm the published comparison ran, and against MASS's `stepAIC()` in
# the fixtures for the rest.

# The family the forward search fits under. The binomial one is the quasi-binomial: the
# coefficients and the deviance are the same, and it takes a case weight that is not a whole
# number without complaint, where the binomial family reads one as a fractional count of
# successes. The criterion is read off the deviance, see `oracle_glm_aic()`.
oracle_glm_family <- function(family) {
  switch(family, binomial = stats::quasibinomial(), gaussian = stats::gaussian())
}

# Akaike's criterion of a forward-search fit. For a 0/1 response the saturated log-likelihood is
# zero, so the criterion is the weighted deviance plus twice the rank, which is what the binomial
# family reports for unweighted data; the Gaussian family reports its own.
oracle_glm_aic <- function(fit, family) {
  if (identical(family, "binomial")) fit$deviance + 2 * fit$rank else fit$aic
}

# Forward selection by AIC, one column admitted at a time. The polynomial basis is stored with the
# fit rather than rebuilt, because an orthogonal basis refitted on new units is a different basis.
oracle_forward_aic <- function(m, y, max_terms, degree, family, w = rep(1, length(y))) {
  if (length(unique(y)) < 2L) {
    return(list(constant = mean(y)))
  }
  link <- oracle_glm_family(family)
  chosen <- integer(0)
  bases <- list()
  best_aic <- oracle_glm_aic(stats::glm(y ~ 1, family = link, weights = w), family)
  # A column holding one value has no polynomial basis to enter as, so it is not a candidate. It
  # is the intercept the search already starts from, and offering it is what makes an orthogonal
  # basis divide by a norm of zero.
  offered <- which(vapply(seq_len(ncol(m)), function(j) length(unique(m[, j])) > 1L, logical(1L)))
  repeat {
    if (length(chosen) >= max_terms) {
      break
    }
    gains <- rep(NA_real_, ncol(m))
    fits <- vector("list", ncol(m))
    for (j in setdiff(offered, chosen)) {
      b <- oracle_poly_basis(m[, j], degree)
      d <- oracle_design_frame(c(bases, list(b)))
      # A candidate whose fit separates the response, or does not settle, is refused rather than
      # admitted: those are the states the criterion cannot be read off, and admitting one would
      # let the search prefer a column for having no answer. Anything the fitter raises as an
      # error is a fault rather than a verdict on the candidate, and propagates.
      fit <- tryCatch(stats::glm(y ~ ., data = d, family = link, weights = w),
                      warning = function(cond) NULL)
      if (!is.null(fit) && is.finite(oracle_glm_aic(fit, family))) {
        gains[j] <- oracle_glm_aic(fit, family)
        fits[[j]] <- list(fit = fit, basis = b)
      }
    }
    if (!any(is.finite(gains)) || min(gains, na.rm = TRUE) >= best_aic) {
      break
    }
    j <- which.min(gains)
    best_aic <- gains[j]
    chosen <- c(chosen, j)
    bases <- c(bases, list(fits[[j]]$basis))
    current <- fits[[j]]$fit
  }
  if (!length(chosen)) {
    return(list(constant = mean(y)))
  }
  list(columns = chosen, bases = bases, fit = current)
}

oracle_predict_forward <- function(f, m) {
  if (!is.null(f$constant)) {
    return(rep(f$constant, nrow(m)))
  }
  b <- lapply(seq_along(f$columns), function(k) oracle_apply_basis(f$bases[[k]], m[, f$columns[k]]))
  as.numeric(stats::predict(f$fit, oracle_design_frame(b), type = "response"))
}

# An orthogonal polynomial basis, kept with the coefficients it was fitted beside so that new units
# are mapped through the same basis rather than through one re-derived from themselves.
oracle_poly_basis <- function(v, degree) {
  degree <- min(degree, length(unique(v)) - 1L)
  if (degree < 1L) {
    stop("a column holding one value has no polynomial basis to enter as.", call. = FALSE)
  }
  b <- stats::poly(v, degree = degree)
  list(degree = degree, coefs = attr(b, "coefs"), values = b)
}

oracle_apply_basis <- function(basis, v) {
  list(degree = basis$degree, coefs = basis$coefs,
       values = stats::poly(v, degree = basis$degree, coefs = basis$coefs))
}

oracle_design_frame <- function(bases) {
  cols <- list()
  for (k in seq_along(bases)) {
    b <- bases[[k]]$values
    for (p in seq_len(ncol(b))) {
      cols[[sprintf("t%d_%d", k, p)]] <- as.numeric(b[, p])
    }
  }
  as.data.frame(cols)
}
