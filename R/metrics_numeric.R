# Each metric of a numeric response and its predictions, as a function of both.
.regression_metrics <- list(
  r_squared = function(y, p) {
    sst <- sum((y - mean(y))^2)
    if (sst > 0) 1 - sum((y - p)^2) / sst else NA_real_
  },
  pearson = function(y, p) {
    if (stats::sd(y) > 0 && stats::sd(p) > 0) stats::cor(y, p) else NA_real_
  },
  rmse = function(y, p) sqrt(mean((y - p)^2)),
  mse = function(y, p) mean((y - p)^2),
  mae = function(y, p) mean(abs(y - p)),
  max_error = function(y, p) max(abs(y - p))
)

#' A metric of a numeric response and its predictions
#'
#' The statistics biomod2 reads an abundance model by. With `e = y - p`:
#'
#' | `metric` | reads |
#' |---|---|
#' | `"r_squared"` | `1 - sum(e^2) / sum((y - mean(y))^2)`, `NA` where `y` is constant |
#' | `"pearson"` | the correlation of `y` and `p`, `NA` where either is constant |
#' | `"rmse"` | `sqrt(mean(e^2))` |
#' | `"mse"` | `mean(e^2)` |
#' | `"mae"` | `mean(abs(e))` |
#' | `"max_error"` | `max(abs(e))` |
#'
#' A comparison across candidates reads the highest score as the best, so the four errors are
#' registered under the names `neg_rmse`, `neg_mse`, `neg_mae` and `neg_max_error` with their sign
#' reversed, and `r_squared` and `pearson` under their own names. A cell holding a prediction that
#' is not a number scores `NA`.
#'
#' @param y Observed values.
#' @param p Predictions for the same units, in the same order.
#' @param metric One of the names in the table.
#'
#' @return One number, or `NA` where the cell defines none.
#'
#' @examples
#' y <- c(1, 2, 3, 4)
#' p <- c(1.5, 2, 2.5, 5)
#' regression_metric(y, p, "rmse")
#' regression_metric(y, p, "r_squared")
#'
#' @export
regression_metric <- function(y, p, metric = names(.regression_metrics)) {
  metric <- match.arg(metric)
  y <- as.numeric(y)
  p <- as.numeric(p)
  if (length(y) != length(p)) {
    stop("`y` and `p` must be the same length, got ", length(y), " and ", length(p), ".",
         call. = FALSE)
  }
  if (!length(y) || anyNA(y) || anyNA(p) || !all(is.finite(p))) {
    return(NA_real_)
  }
  value <- .regression_metrics[[metric]](y, p)
  if (is.finite(value)) value else NA_real_
}

# Each metric of ordinal classes, from the confusion table of the observed classes against the
# class each prediction is nearest to. With `m[i, j]` the units of observed class `j` predicted as
# class `i` and `k` the classes observed: recall averages `m[j, j] / sum_i m[i, j]` over the classes
# and precision `m[i, i] / sum_j m[i, j]`, a class nothing is predicted in or no unit holds counting
# zero; the mean is over all `k` classes.
.ordinal_metrics <- list(
  accuracy = function(m) sum(diag(m)) / sum(m),
  recall = function(m) sum(diag(m) / colSums(m), na.rm = TRUE) / nrow(m),
  precision = function(m) sum(diag(m) / rowSums(m), na.rm = TRUE) / nrow(m),
  f1 = function(m) {
    r <- sum(diag(m) / colSums(m), na.rm = TRUE) / nrow(m)
    q <- sum(diag(m) / rowSums(m), na.rm = TRUE) / nrow(m)
    if (r + q > 0) 2 * q * r / (q + r) else NA_real_
  }
)

#' A metric of ordinal classes
#'
#' The statistics biomod2 reads a model of an ordinal response by. The response is a column of
#' whole-number classes, the model predicts a number on the same scale, and each prediction is read
#' as the observed class nearest to it, the lower class on a tie. With `m[i, j]` the units of
#' observed class `j` predicted as class `i`, over the `k` classes observed in the cell:
#'
#' | `metric` | reads |
#' |---|---|
#' | `"accuracy"` | `sum(diag(m)) / sum(m)` |
#' | `"recall"` | the mean over the `k` classes of `m[j, j]` over the units of class `j` |
#' | `"precision"` | the mean over the `k` classes of `m[i, i]` over the units predicted as `i` |
#' | `"f1"` | `2 P R / (P + R)` of the two means, `NA` where both are zero |
#'
#' A class with no unit, or in which nothing is predicted, adds zero to its mean. The four are
#' registered as `ordinal_accuracy`, `ordinal_recall`, `ordinal_precision` and `ordinal_f1`.
#'
#' @inheritParams regression_metric
#' @param metric One of the names in the table.
#'
#' @return One number, or `NA` where the cell defines none.
#'
#' @examples
#' ordinal_metric(c(1, 1, 2, 3, 3), c(1.2, 2.4, 2, 2.8, 3.4), "accuracy")
#'
#' @export
ordinal_metric <- function(y, p, metric = names(.ordinal_metrics)) {
  metric <- match.arg(metric)
  y <- as.numeric(y)
  p <- as.numeric(p)
  if (length(y) != length(p)) {
    stop("`y` and `p` must be the same length, got ", length(y), " and ", length(p), ".",
         call. = FALSE)
  }
  if (!length(y) || anyNA(y) || anyNA(p) || !all(is.finite(p))) {
    return(NA_real_)
  }
  classes <- sort(unique(y))
  predicted <- vapply(p, function(v) which.min(abs(classes - v)), integer(1L))
  m <- table(factor(predicted, levels = seq_along(classes)),
             factor(match(y, classes), levels = seq_along(classes)))
  value <- .ordinal_metrics[[metric]](unclass(m))
  if (is.finite(value)) value else NA_real_
}

# The cells of a numeric response. A model needs two distinct values to be fitted on and a score
# two to be read at, so a cell is scorable where each side of its split holds at least two. The
# counts the cells table carries are of units, `pres_*`, and of distinct values, `abs_*`.
.numeric_cells <- function(y, folds) {
  y <- .as_response(y)
  f <- .as_folds(folds, rownames(y))
  levels <- sort(unique(f))
  grid <- expand.grid(variable = colnames(y), fold = levels, KEEP.OUT.ATTRS = FALSE,
                      stringsAsFactors = FALSE)
  side <- function(v, rows) c(units = sum(rows), distinct = length(unique(y[rows, v])))
  counts <- t(vapply(seq_len(nrow(grid)), function(i) {
    v <- grid$variable[i]
    test <- f == grid$fold[i]
    c(side(v, !test), side(v, test))
  }, numeric(4L)))
  out <- data.frame(variable = grid$variable, fold = grid$fold,
                    n_occ = vapply(grid$variable, function(v) length(unique(y[, v])), numeric(1L)),
                    pres_train = counts[, 1L], abs_train = counts[, 2L],
                    pres_test = counts[, 3L], abs_test = counts[, 4L], stringsAsFactors = FALSE)
  for (nm in c("n_occ", "pres_train", "abs_train", "pres_test", "abs_test")) {
    out[[nm]] <- as.integer(out[[nm]])
  }
  out$scorable <- out$abs_train >= 2L & out$abs_test >= 2L
  out <- out[order(out$variable, out$fold, method = "radix"), ]
  rownames(out) <- NULL
  structure(out, class = c("timesift_cells", "data.frame"))
}

.numeric_head <- function(check, metric) {
  list(
    prepare = function(y) {
      y <- .as_response(y)
      if (!is.numeric(y) || !all(is.finite(y))) {
        stop("a numeric response is finite numbers, with no missing value.", call. = FALSE)
      }
      check(y)
      y
    },
    activation = "identity",
    loss = "squared_error",
    metric = metric,
    cells = .numeric_cells
  )
}

# The heads for a numeric response. All three are fitted under squared error through an identity
# output; they differ in what the response may hold and in the metric a comparison reads.
.numeric_heads <- function() {
  list(
    continuous = .numeric_head(function(y) invisible(NULL), "r_squared"),
    abundance = .numeric_head(function(y) {
      if (any(y < 0)) {
        stop("an abundance response is not negative.", call. = FALSE)
      }
    }, "r_squared"),
    ordinal = .numeric_head(function(y) {
      if (any(y != round(y))) {
        stop("an ordinal response holds whole-number classes.", call. = FALSE)
      }
    }, "ordinal_f1")
  )
}
