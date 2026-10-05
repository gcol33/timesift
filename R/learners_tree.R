#' Classification and regression tree on the flattened representation
#'
#' One tree per response, over every bin-by-channel column of the representation, grown under
#' rpart's rules: the Gini index under a presence-absence head, the sum of squares under a head
#' with a squared-error loss and the Poisson deviance under a count head, a split only between two
#' distinct values of a column, and the
#' cost-complexity bookkeeping that keeps a split only where it lowers the risk by at least `cp` of
#' the root's. On the same columns, weights and folds the tree is the one rpart grows, split for
#' split, and its complexity table the one rpart reports; the tree is grown by the core the Python
#' package calls, so the two languages grow it identically.
#'
#' The grown tree is pruned back by an inner cross-validation. Its folds are dealt for each
#' response and stratified on it, as the elastic net's are, and `prune` names the rule that reads
#' the complexity table: `"se_sum"` takes the row of least cross-validated error plus its standard
#' error among the rows that keep a split, the last of them where several tie, which is how
#' biomod2 prunes its classification tree; `"one_se"` takes the smallest tree within one standard
#' error of the least cross-validated error; `"min"` the first row reaching the least error; and
#' `"none"` keeps the tree as grown.
#'
#' `preset` says whose defaults the settings left `NULL` take. `"default"` is rpart's own, which is
#' what biomod2's default option set fits: `min_split = 20`, `min_leaf = round(min_split / 3)`
#' (or `min_split = 3 * min_leaf` where only `min_leaf` is given), `cp = 0.01`, `max_depth = 30`
#' and ten inner folds. `"bigboss"` is biomod2's tuned option set: `min_split = 5`, `min_leaf = 5`,
#' `cp = 0.001`, `max_depth = 10` and five inner folds. A setting given explicitly beats either.
#'
#' The case weights are the response head's, [positive_weights()] under presence-absence. They
#' weigh every class count, sum of squares and event count the tree is grown on; `min_split` and
#' `min_leaf` count observations, as rpart's do.
#' Under the shipped presence-absence head those weights are on, so a default `tree()` is
#' rpart's specification fitted under them; a head registered without `weights` fits it unweighted.
#'
#' Under a count head a leaf predicts a rate, and the rate is shrunk towards the rate of the units
#' the tree is grown on, as rpart's `method = "poisson"` shrinks it: the posterior mean of a gamma
#' prior whose coefficient of variation is `shrink`. A split is chosen on the deviance of the
#' unshrunk rates, and a subtree's risk, its complexity and the pruning's cross-validated error are
#' read on the shrunk ones.
#'
#' @inheritParams elasticnet
#' @param min_split Observations a node needs before a split of it is tried.
#' @param min_leaf Observations each child of a split keeps.
#' @param cp The share of the root's risk a split has to remove to be kept.
#' @param max_depth Depth of the deepest node, the root at depth 0; 30 at most.
#' @param prune How the grown tree is pruned: `"se_sum"`, `"one_se"`, `"min"` or `"none"`.
#' @param n_inner Folds of the inner cross-validation the pruning reads.
#' @param preset Whose defaults the settings left `NULL` take: `"default"` or `"bigboss"`.
#' @param shrink Under a count head, the coefficient of variation of the gamma prior a leaf's rate
#'   is shrunk by; `0` for no shrinkage. rpart's default is `1`.
#' @param seed Seed for the inner cross-validation's fold draw.
#' @param threads Responses grown at once. What comes back does not depend on it.
#'
#' @return A [learner()].
#'
#' @examples
#' tree()
#' tree(preset = "bigboss", prune = "one_se")
#'
#' @export
tree <- function(data = NULL, min_split = NULL, min_leaf = NULL, cp = NULL, max_depth = NULL,
                 prune = c("se_sum", "one_se", "min", "none"), n_inner = NULL,
                 preset = c("default", "bigboss"), shrink = 1, seed = 1L, threads = 1L) {
  prune <- match.arg(prune)
  preset <- match.arg(preset)
  if (!is.numeric(shrink) || length(shrink) != 1L || is.na(shrink) || shrink < 0) {
    stop("`shrink` is one number of zero or more, got ", .describe(shrink), ".", call. = FALSE)
  }
  .check_count(threads, "threads", 1)
  settings <- .tree_settings(preset, min_split, min_leaf, cp, max_depth, n_inner)
  learner(
    name = "tree",
    data = data, reads = "tabular", multi = "separate",
    params = c(settings, list(prune = prune, shrink = as.numeric(shrink), seed = as.integer(seed),
                              threads = as.integer(threads))),
    fit = function(x, y, min_split, min_leaf, cp, max_depth, n_inner, prune, shrink, seed,
                   threads, head, weights, group = NULL, ...) {
      family <- .head_family(head)
      m <- .flatten(x)
      fit <- .varies(y)
      models <- as.list(unname(colMeans(y)))
      if (any(fit)) {
        folds <- if (identical(prune, "none")) NULL else
          .response_folds(y[, fit, drop = FALSE], n_inner, .variable_seeds(seed, y)[fit], group)
        grown <- .tree_fits(m, y[, fit, drop = FALSE], weights[, fit, drop = FALSE], family,
                            min_split, min_leaf, cp, max_depth, fold = folds$fold,
                            n_fold = folds$n_fold %||% integer(0), shrink = shrink,
                            threads = threads)
        models[fit] <- lapply(grown, function(g) {
          at <- .tree_prune_cp(g, prune)
          if (is.null(at)) g else .tree_prune(g, at)
        })
      }
      list(models = models, columns = colnames(m), family = family)
    },
    predict = function(model, x) {
      m <- .flatten(x)
      .as_predictions(vapply(model$models, function(f) {
        if (is.numeric(f)) rep(f, nrow(m)) else .tree_predict(f, m)
      }, numeric(nrow(m))), nrow(m))
    }
  )
}

# The settings a tree is grown under: those given, and the preset's for the rest. Under rpart's
# own defaults a `min_leaf` left open follows `min_split` and a `min_split` left open follows a
# given `min_leaf`, as `rpart.control()` has them.
.tree_settings <- function(preset, min_split, min_leaf, cp, max_depth, n_inner) {
  if (identical(preset, "bigboss")) {
    base <- list(min_split = 5L, min_leaf = 5L, cp = 0.001, max_depth = 10L, n_inner = 5L)
  } else {
    split <- min_split %||% if (is.null(min_leaf)) 20L else 3L * as.integer(min_leaf)
    base <- list(min_split = split, min_leaf = round(split / 3), cp = 0.01, max_depth = 30L,
                 n_inner = 10L)
  }
  out <- list(min_split = min_split %||% base$min_split, min_leaf = min_leaf %||% base$min_leaf,
              cp = cp %||% base$cp, max_depth = max_depth %||% base$max_depth,
              n_inner = n_inner %||% base$n_inner)
  out$min_split <- as.integer(out$min_split)
  out$min_leaf <- as.integer(out$min_leaf)
  out$max_depth <- as.integer(out$max_depth)
  out$n_inner <- as.integer(out$n_inner)
  out
}
