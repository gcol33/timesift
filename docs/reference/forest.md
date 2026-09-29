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
forest(
  data = NULL,
  trees = NULL,
  mtry = NULL,
  min_node = NULL,
  balance = FALSE,
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

  Trees in the forest.

- mtry:

  Columns drawn for each node, the best of which it is split on.

- min_node:

  Units each side of a split keeps.

- balance:

  Whether each tree draws as many units from each class as the smaller
  holds.

- preset:

  Whose defaults the settings left `NULL` take: `"package"` or
  `"bigboss"`.

- seed:

  Seed for the bootstrap draws and the column draws.

- threads:

  Trees grown at once. The forest is the same on any number.

## Value

A
[`learner()`](https://gillescolling.com/timesift/reference/learner.md).

## Details

Each tree is grown on a bootstrap draw of the units, as many draws as
there are units, and each node is split on the best of `mtry` columns
drawn for it, by the Gini index or the sum of squares
[`tree()`](https://gillescolling.com/timesift/reference/tree.md) splits
by. A tree is grown out: a node is split while it holds at least twice
`min_node` units and its responses differ. A leaf reports the share of
presences among the draws it holds, or their mean, and the forest the
mean over its trees. The forest is grown by the core the Python package
calls, and every draw comes from one generator seeded per tree, so the
two languages grow the same forest on any number of threads.

`balance = TRUE` is the down-sampled forest biomod2 fits as `RFd`: each
tree draws as many units from each class as the smaller class holds, so
a rare response's presences are half of every tree's draw.

`preset` says whose defaults the settings left `NULL` take. `"package"`
is randomForest's own, which is what biomod2's default option set fits:
500 trees, `mtry` the square root of the column count under
presence-absence and a third of it under a squared-error loss, and
`min_node` 1 and 5 under the two. `"bigboss"` is biomod2's tuned option
set: 500 trees, `mtry = 2` and `min_node = 5`. A setting given
explicitly beats either, and an `mtry` above the column count is the
column count.

The case weights are the response head's,
[`positive_weights()`](https://gillescolling.com/timesift/reference/positive_weights.md)
under presence-absence, and weight the bootstrap draw: a unit is drawn
in proportion to its weight, and within its class under `balance`.

## Examples

``` r
forest(trees = 200L)
forest(preset = "bigboss", balance = TRUE)
```
