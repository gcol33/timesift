# Which cells a score is defined on

A per-variable score needs both classes among the held-out units, and a
per-variable model needs both classes among the units it was fitted on,
so a `(variable, fold)` cell where either side of the split is one-class
carries no score. The mask says which cells those are.

## Usage

``` r
scorable_cells(y, folds)
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

## Value

A data frame of one row per `(variable, fold)` cell, of class
`timesift_cells`, with the counts on each side of the split and a
`scorable` flag.

## Details

It is computed from the response and the fold map alone, with no model
involved. Every learner in a ladder is then restricted to the same
cells, so their means share one denominator and every paired difference
runs on matched cells. Computing it from a model instead would let a
joint multi-label learner, which emits a number for every cell whether
or not it could be fitted per variable, be scored on cells its opponents
were never fitted on.

## Examples

``` r
set.seed(1)
y <- matrix(rbinom(600, 1, 0.2), nrow = 100,
            dimnames = list(sprintf("p%03d", 1:100), paste0("sp", 1:6)))
cells <- scorable_cells(y, fold_map(y, v = 5))
cells
```
