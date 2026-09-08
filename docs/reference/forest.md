# Random forest on the flattened representation

One forest per response, over every bin-by-channel column of the
representation: a probability forest under a presence-absence head and a
regression forest under a head with a squared-error loss. Trees split on
one column at a time and pay nothing for columns that carry nothing, so
a forest reads a wide tabular representation without a penalty path and
without a selection step, and it finds an interaction between two bins
that a linear model would need the product term for.

## Usage

``` r
forest(data = NULL, trees = 500L, mtry = NULL, min_node = 1L, seed = 1L)
```

## Arguments

- data:

  A representation the learner is pinned to, or `NULL` to run across
  every representation of the run.

- trees:

  Trees in the forest.

- mtry:

  Columns tried at each split, or `NULL` for the square root of the
  column count.

- min_node:

  Smallest node a split is made on.

- seed:

  Seed for the bootstrap draw and the split sampling, which are random
  and would otherwise make the fit irreproducible.

## Value

A
[`learner()`](https://gillescolling.com/timesift/reference/learner.md).

## Details

Under a presence-absence head, presences are up-weighted in the
bootstrap draw by the ratio of absences to presences among the fitting
targets, the same weighting the penalised fit uses, so a rare response
is not fitted away by either of them for a reason the other does not
share.

## Examples

``` r
forest(trees = 200L)
```
