# Gradient boosted trees on the flattened representation

One boosted model per response, over every bin-by-channel column of the
representation: a logistic model under a presence-absence head and a
squared-error one under a head with a squared-error loss. The score
starts at the log-odds of the weighted share of presences, or the
weighted mean, and each tree is fitted to the loss's gradient at the
current score and added to it scaled by `shrinkage`. Each tree is grown
on a subsample of the units drawn without replacement, and reads a
subsample of the columns.

## Usage

``` r
boosting(
  data = NULL,
  trees = NULL,
  depth = NULL,
  shrinkage = NULL,
  min_leaf = NULL,
  subsample = NULL,
  colsample = NULL,
  newton = FALSE,
  lambda = NULL,
  gamma = NULL,
  n_inner = NULL,
  preset = c("package", "bigboss"),
  seed = 1L,
  threads = 1L
)
```

## Arguments

- data:

  A representation the learner is pinned to, or `NULL` to run across
  every representation of the run.

- trees:

  Trees fitted.

- depth:

  Splits in a tree under gbm, its depth under xgboost.

- shrinkage:

  The scale each tree is added to the score at.

- min_leaf:

  Units each side of a split keeps under gbm, hessian under xgboost.

- subsample:

  Share of the units each tree is grown on.

- colsample:

  Share of the columns each tree reads.

- newton:

  Whether the trees are xgboost's second-order ones rather than gbm's.

- lambda, gamma:

  xgboost's L2 penalty on a leaf and least gain of a split; zero under
  gbm.

- n_inner:

  Folds of the inner cross-validation choosing how many trees are kept,
  or 0 to keep them all.

- preset:

  Whose defaults the settings left `NULL` take: `"package"` or
  `"bigboss"`.

- seed:

  Seed for the subsamples and the inner folds.

- threads:

  Fits of one response's inner cross-validation run at once. The model
  is the same on any number.

## Value

A
[`learner()`](https://gillescolling.com/timesift/reference/learner.md).

## Details

`newton` picks the trees. Off, they are gbm's, which is what biomod2
fits as `GBM`: `depth` splits grown best first, each the one that most
reduces the weighted squared error of the working response with at least
`min_leaf` units on each side, and a leaf that takes one Newton step on
the loss. On, they are xgboost's exact greedy trees, which biomod2 fits
as `XGBOOST`: grown level by level to `depth`, each split chosen by the
second-order gain under the L2 penalty `lambda` with at least `min_leaf`
of hessian on each side, pruned where a split gains less than `gamma`,
and a leaf the step `-G / (H + lambda)`. Either way `depth` is the order
of interaction a tree can hold. With `subsample = 1` the first-order fit
is gbm's own to rounding, and the second-order one xgboost's to its
single-precision storage; the model is grown by the core the Python
package calls, so the two languages fit the same model.

`n_inner` folds, when above zero, choose how many trees are kept: the
fit is repeated on each fold's complement, and the number of trees of
least held-out deviance, summed over the folds and weighted by how many
units each holds, is kept, as gbm's `cv.folds` chooses it. The folds are
dealt for each response and stratified on it, as the elastic net's are.

`preset` says whose defaults the settings left `NULL` take. `"package"`
is the fitting package's own, which is what biomod2's default option set
fits: under gbm 100 trees of one split, `shrinkage = 0.1`,
`min_leaf = 10` and `subsample = 0.5`; under xgboost 100 trees of depth
6, `shrinkage = 0.3`, `min_leaf = 1`, `lambda = 1` and every unit and
column. `"bigboss"` is biomod2's tuned option set: under gbm 2500 trees
of seven splits, `shrinkage = 0.001`, `min_leaf = 5`, `subsample = 0.5`
and three inner folds; under xgboost four trees of depth 2 at
`shrinkage = 1`. A setting given explicitly beats either.

The case weights are the response head's,
[`positive_weights()`](https://gillescolling.com/timesift/reference/positive_weights.md)
under presence-absence, and weigh the gradient and every sum a tree is
grown on; `min_leaf` counts units under gbm, as `n.minobsinnode` does.

## Examples

``` r
boosting()
boosting(preset = "bigboss")
boosting(newton = TRUE, depth = 3L)
```
