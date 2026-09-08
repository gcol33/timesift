# How much a self-selected threshold inflates the reported level

The true skill statistic is read at the threshold that maximises it,
chosen on the same held-out units the score is then read on. That
selection inflates the level, and by more the fewer presences a cell
holds. Most code carries the inflation silently; this measures it for
the presence counts of a given design.

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

  Population skill values to plant.

- replicates:

  Replicates per value.

- seed:

  Random seed.

## Value

A data frame with one row per planted value: the truth, the mean level
read back, the inflation, and its interval across replicates.

## Details

Predictions are simulated under a normal model in which the population
skill is exactly `skill`, at the cell sizes and presence counts of the
response and fold map supplied, and the level is read back exactly as
[`grain_ladder()`](https://gillescolling.com/timesift/reference/grain_ladder.md)
reports it. The gap between what comes back and the truth planted is the
inflation.

It cancels in the paired differences
[`paired_contrast()`](https://gillescolling.com/timesift/reference/paired_contrast.md)
takes, since both arms carry it on the same cell. It does not cancel in
a level, so a level is an upper bound on the skill a population has.

## Examples

``` r
set.seed(1)
y <- matrix(rbinom(1200, 1, 0.15), nrow = 200,
            dimnames = list(sprintf("p%03d", 1:200), paste0("sp", 1:6)))
tss_inflation(y, fold_map(y, v = 5), skill = c(0.6, 0.9), replicates = 40)
```
