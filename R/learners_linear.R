#' Penalised regression on the flattened representation
#'
#' One elastic net per variable, over every bin-by-channel column of the representation and, by
#' default, their squares. There is no discrete selection step: the penalty path uses every column
#' and shrinks, and the penalty itself is chosen by an inner cross-validation on the fitting units,
#' so nothing about the model is decided outside the fold it is fitted in.
#'
#' The family is the response head's: a binary cross-entropy loss fits a logistic model and a
#' squared-error loss a linear one, so the learner is the same under a presence-absence head and
#' under a continuous one.
#'
#' This is the aggregate-feature side of the comparison the package was built for, and it is the
#' fair opponent for a network: a per-fold discrete selector pays selection variance a network
#' never pays, so beating that one is not a matched result.
#'
#' @param data A representation the learner is pinned to, or `NULL` to run across every
#'   representation of the run.
#' @param alpha Elastic-net mixing, `1` lasso and `0` ridge.
#' @param n_inner Folds of the inner cross-validation that chooses the penalty.
#' @param squares Add the square of every column, giving the same quadratic capacity a
#'   second-order polynomial term would.
#' @param s Which penalty of the inner path to predict at.
#' @param weight_positives Weight presences by the ratio of absences to presences among the
#'   fitting units, so a rare variable is not fitted away. Read under a presence-absence head
#'   only.
#' @param seed Seed for the inner cross-validation's fold draw, which is random and would otherwise
#'   make the fit irreproducible.
#'
#' @return A [learner()].
#'
#' @examples
#' elasticnet(alpha = 0.5)
#'
#' @export
elasticnet <- function(data = NULL, alpha = 0.5, n_inner = 5L, squares = TRUE, s = "lambda.min",
                       weight_positives = TRUE, seed = 1L) {
  learner(
    name = "elasticnet",
    data = data, reads = "tabular", multi = "separate",
    needs = "glmnet",
    params = list(alpha = alpha, n_inner = n_inner, squares = squares, s = s,
                  weight_positives = weight_positives, seed = seed),
    fit = function(x, y, alpha, n_inner, squares, s, weight_positives, seed, head, ...) {
      family <- .head_family(head)
      m <- .design(x, squares)
      old <- .seed_state()
      on.exit(.restore_seed(old), add = TRUE)
      seeds <- .variable_seeds(seed, y)
      models <- lapply(seq_len(ncol(y)), function(j) {
        yj <- y[, j]
        if (length(unique(yj)) < 2L) {
          return(mean(yj))
        }
        set.seed(seeds[j])
        w <- if (weight_positives && family == "binomial") .imbalance_weights(yj)
             else rep(1, length(yj))
        tryCatch(
          glmnet::cv.glmnet(m, yj, family = family, alpha = alpha, weights = w,
                            nfolds = n_inner, type.measure = "deviance"),
          error = function(e) mean(yj)
        )
      })
      list(models = models, squares = squares, s = s, columns = colnames(m))
    },
    predict = function(model, x) {
      m <- .design(x, model$squares)
      if (!identical(colnames(m), model$columns)) {
        stop("the representation predicted on has different channels or bins from the fitted one.",
             call. = FALSE)
      }
      .as_predictions(vapply(model$models, function(f) {
        if (is.numeric(f)) rep(f, nrow(m))
        else as.numeric(stats::predict(f, m, s = model$s, type = "response"))
      }, numeric(nrow(m))), nrow(m))
    }
  )
}

#' Forward selection by AIC on the flattened representation
#'
#' One generalised linear model per variable, its predictors chosen by forward selection over every
#' bin-by-channel column, admitting a column while it lowers AIC and stopping at a fixed budget.
#' Each candidate enters as an orthogonal polynomial, so a term can be non-monotone in the reading
#' the way a niche optimum is. The family is the response head's: logistic under a binary
#' cross-entropy loss, Gaussian under a squared-error one.
#'
#' Selection happens inside whichever units the learner is handed, so under [grain_ladder()] it is
#' redone in every fold. That is the footing the other learners are fitted on. Reported beside a
#' penalised fit it also prices discrete selection: choosing a handful of columns out of hundreds
#' is high variance, and that variance is a cost of the selector rather than of the features.
#'
#' @inheritParams elasticnet
#' @param max_terms Predictors admitted before selection stops.
#' @param degree Polynomial degree each admitted column enters at.
#'
#' @return A [learner()].
#'
#' @examples
#' stepwise(max_terms = 3)
#'
#' @export
stepwise <- function(data = NULL, max_terms = 3L, degree = 2L) {
  learner(
    name = "stepwise",
    data = data, reads = "tabular", multi = "separate",
    params = list(max_terms = max_terms, degree = degree),
    fit = function(x, y, max_terms, degree, head, ...) {
      family <- .head_family(head)
      m <- .flatten(x)
      models <- lapply(seq_len(ncol(y)), function(j) {
        .forward_aic(m, y[, j], max_terms, degree, family)
      })
      list(models = models, columns = colnames(m), degree = degree, family = family)
    },
    predict = function(model, x) {
      m <- .flatten(x)
      if (!identical(colnames(m), model$columns)) {
        stop("the representation predicted on has different channels or bins from the fitted one.",
             call. = FALSE)
      }
      .as_predictions(vapply(model$models, function(f) .predict_forward(f, m), numeric(nrow(m))),
                      nrow(m))
    }
  )
}

# Forward selection by AIC, one column admitted at a time. The polynomial basis is stored with the
# fit rather than rebuilt, because an orthogonal basis refitted on new units is a different basis.
.forward_aic <- function(m, y, max_terms, degree, family) {
  if (length(unique(y)) < 2L) {
    return(list(constant = mean(y)))
  }
  link <- .glm_family(family)
  chosen <- integer(0)
  bases <- list()
  best_aic <- stats::glm(y ~ 1, family = link)$aic
  repeat {
    if (length(chosen) >= max_terms) {
      break
    }
    gains <- rep(NA_real_, ncol(m))
    fits <- vector("list", ncol(m))
    for (j in setdiff(seq_len(ncol(m)), chosen)) {
      b <- .poly_basis(m[, j], degree)
      d <- .design_frame(c(bases, list(b)))
      fit <- tryCatch(stats::glm(y ~ ., data = d, family = link),
                      error = function(e) NULL, warning = function(w) NULL)
      if (!is.null(fit) && is.finite(fit$aic)) {
        gains[j] <- fit$aic
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

.predict_forward <- function(f, m) {
  if (!is.null(f$constant)) {
    return(rep(f$constant, nrow(m)))
  }
  b <- lapply(seq_along(f$columns), function(k) .apply_basis(f$bases[[k]], m[, f$columns[k]]))
  as.numeric(stats::predict(f$fit, .design_frame(b), type = "response"))
}

# An orthogonal polynomial basis, kept with the coefficients it was fitted beside so that new units
# are mapped through the same basis rather than through one re-derived from themselves.
.poly_basis <- function(v, degree) {
  degree <- min(degree, max(1L, length(unique(v)) - 1L))
  b <- stats::poly(v, degree = degree)
  list(degree = degree, coefs = attr(b, "coefs"), values = b)
}

.apply_basis <- function(basis, v) {
  list(degree = basis$degree, coefs = basis$coefs,
       values = stats::poly(v, degree = basis$degree, coefs = basis$coefs))
}

.design_frame <- function(bases) {
  cols <- list()
  for (k in seq_along(bases)) {
    b <- bases[[k]]$values
    for (p in seq_len(ncol(b))) {
      cols[[sprintf("t%d_%d", k, p)]] <- as.numeric(b[, p])
    }
  }
  as.data.frame(cols)
}

.design <- function(x, squares) {
  m <- .flatten(x)
  if (!squares) {
    return(m)
  }
  out <- cbind(m, m^2)
  colnames(out) <- c(colnames(m), paste0(colnames(m), "^2"))
  out
}

# vapply drops to a vector when there is one unit to predict, which would reach the caller as one
# row per variable instead of one column. The shape is restored here so a single new site predicts
# the same way a thousand do.
.as_predictions <- function(p, units) {
  matrix(p, nrow = units)
}

.imbalance_weights <- function(y, cap = Inf) {
  pos <- sum(y == 1)
  neg <- sum(y == 0)
  w <- if (pos > 0) min(neg / pos, cap) else 1
  ifelse(y == 1, w, 1)
}
