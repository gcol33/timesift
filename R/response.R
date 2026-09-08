#' Which cells a score is defined on
#'
#' A per-variable score needs both classes among the held-out units, and a per-variable model needs
#' both classes among the units it was fitted on, so a `(variable, fold)` cell where either side of
#' the split is one-class carries no score. The mask says which cells those are.
#'
#' It is computed from the response and the fold map alone, with no model involved. Every learner
#' in a ladder is then restricted to the same cells, so their means share one denominator and every
#' paired difference runs on matched cells. Computing it from a model instead would let a joint
#' multi-label learner, which emits a number for every cell whether or not it could be fitted per
#' variable, be scored on cells its opponents were never fitted on.
#'
#' @inheritParams fold_map
#' @param folds A fold map from [fold_map()], or any named integer vector of the same shape.
#'
#' @return A data frame of one row per `(variable, fold)` cell, of class `timesift_cells`, with
#'   the counts on each side of the split and a `scorable` flag.
#'
#' @examples
#' set.seed(1)
#' y <- matrix(rbinom(600, 1, 0.2), nrow = 100,
#'             dimnames = list(sprintf("p%03d", 1:100), paste0("sp", 1:6)))
#' cells <- scorable_cells(y, fold_map(y, v = 5))
#' cells
#'
#' @export
scorable_cells <- function(y, folds) {
  y <- .as_response(y)
  f <- .as_folds(folds, rownames(y))
  levels <- sort(unique(f))
  vars <- colnames(y)

  pres_test <- vapply(levels, function(k) colSums(y[f == k, , drop = FALSE]), numeric(ncol(y)))
  n_test <- vapply(levels, function(k) sum(f == k), numeric(1L))
  pres_test <- matrix(pres_test, nrow = ncol(y), dimnames = list(vars, NULL))
  n_occ <- colSums(y)

  out <- data.frame(
    variable = rep(vars, times = length(levels)),
    fold = rep(levels, each = length(vars)),
    n_occ = rep(n_occ, times = length(levels)),
    pres_test = as.vector(pres_test),
    abs_test = rep(n_test, each = length(vars)) - as.vector(pres_test),
    stringsAsFactors = FALSE
  )
  out$pres_train <- out$n_occ - out$pres_test
  out$abs_train <- (nrow(y) - out$n_occ) - out$abs_test
  # Counts, so integers: a mask read back from a file carries integers, and a mask that compared
  # unequal to it on storage mode alone would report a difference there is none of.
  for (nm in c("n_occ", "pres_train", "abs_train", "pres_test", "abs_test")) {
    out[[nm]] <- as.integer(out[[nm]])
  }
  out$scorable <- out$pres_train >= 1L & out$abs_train >= 1L &
    out$pres_test >= 1L & out$abs_test >= 1L
  # C collation for the variable names, so the cell order is the same on every machine and the
  # same as the one the Python side builds.
  keep <- c("variable", "fold", "n_occ", "pres_train", "abs_train", "pres_test", "abs_test",
            "scorable")
  out <- out[order(out$variable, out$fold, method = "radix"), keep]
  rownames(out) <- NULL
  structure(out, class = c("timesift_cells", "data.frame"))
}

#' @export
print.timesift_cells <- function(x, ...) {
  keep <- tapply(x$scorable, x$variable, any)
  cat("<timesift cells>", .plural(nrow(x), "cell"), "over",
      .plural(length(keep), "variable"), "\n")
  cat(sprintf("scorable: %d (%.1f%%); variables with at least one scorable fold: %d of %d\n",
              sum(x$scorable), 100 * mean(x$scorable), sum(keep), length(keep)))
  invisible(x)
}

# The response reaches everything downstream as a numeric matrix with unit identifiers in its row
# names, whether it arrived as a matrix, as a data frame with an identifier column, or as a bare
# vector for a single variable.
.as_response <- function(y) {
  if (is.vector(y) && !is.list(y)) {
    y <- matrix(y, ncol = 1L, dimnames = list(names(y), "y"))
  }
  if (is.data.frame(y)) {
    id <- vapply(y, function(col) is.character(col) || is.factor(col), logical(1L))
    if (any(id)) {
      first <- which(id)[1L]
      rn <- as.character(y[[first]])
      y <- y[, -which(id), drop = FALSE]
      rownames(y) <- rn
    }
    rn <- rownames(y)
    y <- as.matrix(y)
    rownames(y) <- rn
  }
  if (!is.matrix(y)) {
    stop("the response must be a matrix, a data frame or a vector, got ", class(y)[1L], ".",
         call. = FALSE)
  }
  if (is.null(rownames(y))) {
    rownames(y) <- as.character(seq_len(nrow(y)))
  }
  if (is.null(colnames(y))) {
    colnames(y) <- if (ncol(y) == 1L) "y" else paste0("v", seq_len(ncol(y)))
  }
  storage.mode(y) <- "double"
  if (anyNA(y)) {
    stop("the response holds missing values. Fill or drop them before fitting.", call. = FALSE)
  }
  y
}

.presence_absence <- list(
  prepare = function(y) {
    y <- .as_response(y)
    if (!all(y %in% c(0, 1))) {
      stop("a presence-absence response must be 0/1 or logical.", call. = FALSE)
    }
    y
  },
  activation = "sigmoid",
  loss = "binary_cross_entropy",
  metric = "tss",
  cells = function(y, folds) scorable_cells(y, folds)
)
