#' Random forest on the flattened representation
#'
#' One forest per response, over every bin-by-channel column of the representation: a probability
#' forest under a presence-absence head and a regression forest under a head with a squared-error
#' loss or a count head, a count being cut on its variance and a leaf reporting its mean. Trees
#' split on one column at a time and pay nothing for columns that carry nothing, so a forest reads
#' a wide tabular representation without a penalty path and without a selection step, and it finds
#' an interaction between two bins that a linear model would need the product term for.
#'
#' Each tree is grown on a bootstrap draw of the units, as many draws as there are units, and each
#' node is split on the best of `mtry` columns drawn for it, by the Gini index or the sum of squares
#' [tree()] splits by. A tree is grown out: a node is split while it holds at least twice
#' `min_node` units and its responses differ. A leaf reports the share of presences among the draws
#' it holds, or their mean, and the forest the mean over its trees. The forest is grown by the core
#' the Python package calls, and every draw comes from one generator seeded per tree, so the two
#' languages grow the same forest on any number of threads.
#'
#' `balance = TRUE` is the down-sampled forest biomod2 fits as `RFd`: each tree draws as many units
#' from each class as the smaller class holds, so a rare response's presences are half of every
#' tree's draw.
#'
#' `preset` says whose defaults the settings left `NULL` take. `"default"` is randomForest's own,
#' which is what biomod2's default option set fits: 500 trees, `mtry` the square root of the column
#' count under presence-absence and a third of it under a squared-error loss, and `min_node` 1 and
#' 5 under the two. `"bigboss"` is biomod2's tuned option set: 500 trees, `mtry = 2` and
#' `min_node = 5`. A setting given explicitly beats either, and an `mtry` above the column count is
#' the column count.
#'
#' The case weights are the response head's, [positive_weights()] under presence-absence, and
#' weight the bootstrap draw: a unit is drawn in proportion to its weight, and within its class
#' under `balance`.
#' Under the shipped presence-absence head those weights are on, so a default `forest()` is
#' randomForest's specification fitted under them; a head registered without `weights` fits it
#' unweighted.
#'
#' @inheritParams elasticnet
#' @param trees Trees in the forest.
#' @param mtry Columns drawn for each node, the best of which it is split on.
#' @param min_node Units each side of a split keeps.
#' @param balance Whether each tree draws as many units from each class as the smaller holds.
#' @param preset Whose defaults the settings left `NULL` take: `"default"` or `"bigboss"`.
#' @param seed Seed for the bootstrap draws and the column draws.
#' @param threads Trees grown at once. The forest is the same on any number.
#'
#' @return A [learner()].
#'
#' @examples
#' forest(trees = 200L)
#' forest(preset = "bigboss", balance = TRUE)
#'
#' @export
forest <- function(data = NULL, trees = NULL, mtry = NULL, min_node = NULL, balance = FALSE,
                   preset = c("default", "bigboss"), seed = 1L, threads = 1L) {
  preset <- match.arg(preset)
  learner(
    name = "forest",
    data = data, reads = "tabular", multi = "separate",
    params = list(trees = trees, mtry = mtry, min_node = min_node, balance = isTRUE(balance),
                  preset = preset, seed = as.integer(seed), threads = as.integer(threads)),
    fit = function(x, y, trees, mtry, min_node, balance, preset, seed, threads, head, weights,
                   ...) {
      family <- .head_family(head)
      m <- .flatten(x)
      settings <- .forest_settings(preset, family, ncol(m), trees, mtry, min_node)
      seeds <- .variable_seeds(seed, y)
      models <- lapply(seq_len(ncol(y)), function(j) {
        yj <- y[, j]
        if (length(unique(yj)) < 2L) {
          return(mean(yj))
        }
        .forest_fit(m, yj, weights[, j], family, settings$trees, settings$mtry,
                    settings$min_node, balance, seeds[j], threads)
      })
      list(models = models, columns = colnames(m), family = family)
    },
    predict = function(model, x) {
      m <- .flatten(x)
      .as_predictions(vapply(model$models, function(f) {
        if (is.numeric(f)) rep(f, nrow(m)) else .forest_predict(f, m)
      }, numeric(nrow(m))), nrow(m))
    }
  )
}

# The settings a forest is grown under: those given, and the preset's for the rest. randomForest's
# own defaults depend on the family and on how many columns there are, so they are settled at the
# fit.
.forest_settings <- function(preset, family, n_column, trees, mtry, min_node) {
  binomial <- identical(family, "binomial")
  base <- if (identical(preset, "bigboss")) {
    list(trees = 500L, mtry = 2L, min_node = 5L)
  } else {
    list(trees = 500L,
         mtry = max(1, if (binomial) floor(sqrt(n_column)) else floor(n_column / 3)),
         min_node = if (binomial) 1L else 5L)
  }
  out <- list(trees = trees %||% base$trees, mtry = mtry %||% base$mtry,
              min_node = min_node %||% base$min_node)
  out$trees <- as.integer(out$trees)
  out$mtry <- as.integer(min(out$mtry, n_column))
  out$min_node <- as.integer(out$min_node)
  out
}
