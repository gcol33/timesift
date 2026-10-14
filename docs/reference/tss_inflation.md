# How much a self-selected threshold inflates the reported level

The true skill statistic (TSS) comes out too high when its best
threshold is picked on the same held-out units the score is read from,
and the more so the fewer presences a cell holds. This measures that
excess for the presence counts of a given design;
[`implied_skill()`](https://gillescolling.com/timesift/reference/implied_skill.md)
goes the other way, from a reported level to the skill it is consistent
with.

## Usage

``` r
tss_inflation(y, folds, skill = c(0.6, 0.7, 0.9), replicates = 200L, seed = 1L)
```

## Arguments

- y:

  The response: a matrix or data frame of units by variables, with unit
  identifiers in the row names or in a leading character or factor
  column.

- folds:

  A fold map from
  [`fold_map()`](https://gillescolling.com/timesift/reference/fold_map.md),
  or any named integer vector of the same shape.

- skill:

  Population skill values to simulate.

- replicates:

  Replicates per value.

- seed:

  Random seed.

## Value

A data frame with one row per `skill` value: the skill, the mean level
read back, the inflation, and its interval across replicates.

## Details

Predictions are simulated for each value of `skill` as normal scores of
equal variance, at the cell sizes and presence counts of the response
and fold map supplied. The TSS is read at its best threshold exactly as
[`grain_ladder()`](https://gillescolling.com/timesift/reference/grain_ladder.md)
reports it, and the mean reading minus `skill` is the inflation.

The inflation is an expectation. A level read on one design is
optimistic on average, not a bound every reading sits above: a single
level can fall below the population skill. Predictions distributed
otherwise inflate by a different amount at the same skill, which is also
why a paired difference
[`paired_contrast()`](https://gillescolling.com/timesift/reference/paired_contrast.md)
takes is not free of it.

## Examples

``` r
set.seed(1)
y <- matrix(rbinom(1200, 1, 0.15), nrow = 200,
            dimnames = list(sprintf("p%03d", 1:200), paste0("sp", 1:6)))
tss_inflation(y, fold_map(y, v = 5), skill = c(0.6, 0.9), replicates = 40)
```
