# Tune a learner's settings on the training units

Returns a learner that, whenever it is fitted, searches `grid` on the
units it is handed and fits the setting that scored best. The search is
a cross-validation inside those units, so in a run the outer folds never
see it: each fold chooses from its own training units, and the score it
is then read at is not selected on. biomod2's `BIOMOD_Tuning()` searches
a grid per algorithm by the same device.

## Usage

``` r
tune(learner, grid = NULL, metric = NULL, n_inner = 5L, seed = 1L)
```

## Arguments

- learner:

  A
  [`learner()`](https://gillescolling.com/timesift/reference/learner.md),
  or the name of a registered one.

- grid:

  A named list of values to try, one element per setting, or `NULL` for
  the grid registered for the learner.

- metric:

  The registered metric, or a function of `(y, p)`, a setting is scored
  by. Left unset it is the response head's own.

- n_inner:

  Number of inner folds.

- seed:

  Random seed of the inner folds.

## Value

A [`learner()`](https://gillescolling.com/timesift/reference/learner.md)
reporting under the name of the one it wraps.

## Details

A learner's settings are the ones it carries as `params`: the arguments
of its constructor, as `mtry` and `trees` are for
[`forest()`](https://gillescolling.com/timesift/reference/forest.md).
`grid` names some of them and gives the values to try; the grid is every
combination. A value that is itself a vector, as the layer widths of
[`mlp()`](https://gillescolling.com/timesift/reference/torch_learners.md)
are, is given as an element of a list. The inner folds are dealt by
[`fold_map()`](https://gillescolling.com/timesift/reference/fold_map.md)
and keep the grouping the outer fold map keeps whole. A setting is
scored by the mean over the responses of the mean over inner folds of
`metric` on the cells a score is defined on, and ties go to the first
combination in the grid.

What was chosen is recorded: on the fitted model as `$model$chosen` and
`$model$table`, and in the `settings` column of the candidate table of a
run, which reads the model fitted on all targets.

With `grid` left unset the learner is searched over the grid registered
under its name by
[`register_tuning()`](https://gillescolling.com/timesift/reference/register_tuning.md),
which for the learners that ship is the one `BIOMOD_Tuning()` searches:
`mtry` of a
[`forest()`](https://gillescolling.com/timesift/reference/forest.md)
from 1 to the smaller of 10 and the number of columns; `trees`, `depth`
and `shrinkage` of a gbm-style
[`boosting()`](https://gillescolling.com/timesift/reference/boosting.md),
`shrinkage` and `colsample` of the second-order one; `degree` and
`nprune` of
[`mars()`](https://gillescolling.com/timesift/reference/mars.md);
`degree` of
[`discriminant()`](https://gillescolling.com/timesift/reference/discriminant.md);
`regmult` of
[`maxent()`](https://gillescolling.com/timesift/reference/maxent.md);
`quantile` of
[`envelope()`](https://gillescolling.com/timesift/reference/envelope.md);
and the layer width of
[`mlp()`](https://gillescolling.com/timesift/reference/torch_learners.md)
at 2, 4, 6 and 8. biomod2's weight decay is a training setting here,
which
[`train_control()`](https://gillescolling.com/timesift/reference/train_control.md)
holds and a grid does not reach.

## Examples

``` r
tuned <- tune(forest(), list(mtry = c(2, 4), min_node = c(1, 5)), n_inner = 3L)
tuned
```
