# The tree, over the core `src/ts_tree.cpp` compiles into both languages. Nothing here grows
# anything: the design, the family, the case weights and the inner folds are settled above, and
# what is left is to hand them over column-major and to read the complexity table that comes back.
#
# A fitted tree is a plain list of numbers, so it round trips through `saveRDS()` and predicts on
# another machine, which is what every other fitted object in the package is.

.tree_fit <- function(x, y, w, family, min_split, min_leaf, cp, max_depth, fold = NULL,
                      n_fold = 0L) {
  ts_tree_fit_(as.numeric(x), as.numeric(y), as.numeric(w), nrow(x), ncol(x), family,
               as.integer(min_split), as.integer(min_leaf), as.numeric(cp),
               as.integer(max_depth), if (is.null(fold)) NULL else as.integer(fold),
               as.integer(n_fold))
}

.tree_prune <- function(tree, cp) {
  ts_tree_prune_(tree, as.numeric(cp))
}

.tree_predict <- function(tree, newx) {
  ts_tree_predict_(tree, as.numeric(newx), nrow(newx), ncol(newx))
}

# The forest, over the same core. A seed is a number modulo 2^32 there, and crosses as a double
# since an R integer cannot hold every one.
.forest_fit <- function(x, y, w, family, trees, mtry, min_leaf, balance, seed, threads = 1L) {
  ts_forest_fit_(as.numeric(x), as.numeric(y), as.numeric(w), nrow(x), ncol(x), family,
                 as.integer(trees), as.integer(mtry), as.integer(min_leaf), isTRUE(balance),
                 as.numeric(seed) %% 2^32, as.integer(threads))
}

.forest_predict <- function(forest, newx) {
  ts_forest_predict_(forest, as.numeric(newx), nrow(newx), ncol(newx))
}

# The first `n` outputs of the generator tree `tree` of a forest seeded `seed` draws from.
.forest_stream <- function(seed, tree, n) {
  ts_forest_stream_(as.numeric(seed) %% 2^32, as.numeric(tree), as.integer(n))
}

# The complexity a tree is pruned back to, read off its cross-validated complexity table.
# `"se_sum"` is the row of least cross-validated error plus its standard error among the rows
# that keep a split, the last of them where several tie, which is how biomod2 prunes its
# classification tree. `"one_se"` is the smallest tree within one standard error of the least
# cross-validated error, and `"min"` the first row reaching the least error. `NULL` is no
# pruning: a table without a cross-validation, or a rule that finds no row, leaves the tree as it
# was grown.
.tree_prune_cp <- function(tree, rule) {
  if (identical(rule, "none") || !length(tree$xerror)) {
    return(NULL)
  }
  switch(
    rule,
    se_sum = {
      keep <- which(tree$nsplit > 0L)
      if (!length(keep)) {
        return(NULL)
      }
      xsum <- tree$xerror[keep] + tree$xstd[keep]
      tree$cp[keep[max(which(xsum == min(xsum)))]]
    },
    one_se = {
      best <- which.min(tree$xerror)
      tree$cp[which(tree$xerror <= tree$xerror[best] + tree$xstd[best])[1L]]
    },
    min = tree$cp[which.min(tree$xerror)]
  )
}
