# What population skill a reported level is consistent with

[`tss_inflation()`](https://gillescolling.com/timesift/reference/tss_inflation.md)
maps a population skill to the level a design reports for it. This
inverts that map: given a level actually read off a ladder, it solves
for the population skill whose expected reported level equals it.

## Usage

``` r
implied_skill(
  y,
  folds,
  observed,
  grid = seq(0, 0.95, by = 0.05),
  replicates = 200L,
  seed = 1L
)
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

- observed:

  Reported levels to invert.

- grid:

  Population skills the forward map is measured on before interpolating
  between them.

- replicates:

  Replicates per value.

- seed:

  Random seed.

## Value

A data frame of one row per observed level: the level, the population
skill it is consistent with, and whether that sits inside the grid the
map was measured on.

## Details

It answers the question a level raises once the inflation is known, and
it is the only honest way to read a level as a statement about a
population rather than about a scoring rule. It says nothing about a
difference between two arms, where the inflation cancels and the
reported number stands as it is.

## Examples

``` r
set.seed(1)
y <- matrix(rbinom(1200, 1, 0.15), nrow = 200,
            dimnames = list(sprintf("p%03d", 1:200), paste0("sp", 1:6)))
implied_skill(y, fold_map(y, v = 5), observed = 0.71, replicates = 40)
```
