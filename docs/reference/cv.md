# How the folds are drawn

The resampling
[`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
scores on. `cv()` deals units into folds balanced on the stratifying
value; `grouped_cv()` deals whole groups, keeping every target sharing a
group value on one side of each split. `block_cv()` and `env_cv()` hold
out a block of units whole: a region of space for the first, a region of
predictor space for the second.

## Usage

``` r
cv(v = 10L, seed = 1L, strata = 5L, by = NULL, repeats = 1L)

block_cv(by, v = 4L)

env_cv(by, v = 4L)

grouped_cv(group, v = 10L, seed = 1L, repeats = 1L)
```

## Arguments

- v:

  Number of folds.

- seed:

  Random seed, fixed so the map is reproducible.

- strata:

  Number of strata, or `1` for no stratification.

- by:

  What `cv()` stratifies on instead of the richness of the response: the
  name of a numeric column of `targets`, or a vector with one value per
  target. For `block_cv()` and `env_cv()`, the columns the blocks are
  cut on: names of columns of `targets`, or a numeric matrix with one
  row per target.

- repeats:

  Number of times the split is drawn, each with its own seed (`seed`,
  `seed + 1`, and so on). In
  [`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
  every repeat is a full run on its own fold map: a score stays one
  fit's held-out score, the report averages a response's scores over
  folds and repeats, and the combiner is fitted on the out-of-fold
  predictions of all the repeats together. `block_cv()` and `env_cv()`
  draw nothing, so repeat nothing.

- group:

  The grouping: the name of a column of `targets`, or a vector with one
  value per target.

## Value

A `timesift_resampling`.

## Details

The blocks are cut by halving. The units are split at the median of the
column with the widest range (the first on a tie), the lower share going
to the left and the units tied on that column ordered as they arrived,
and each part is cut again until there are `v` of them, so a block holds
as many units as another to within one and a map of `v` blocks has `v`
folds. Nothing is drawn, so both languages return the same map.
`env_cv()` centres and scales every column first, which a coordinate
system does not need. Under the nested selection the inner folds keep
blocks whole as well, so `inner` is at most `v - 1`.

`resampling` also accepts a fold vector or a
[`fold_map()`](https://gillescolling.com/timesift/reference/fold_map.md)
result directly, which is how a split the package has no constructor for
– a spatial block, a season held out whole – reaches the same fitting
path.

## Examples

``` r
cv(v = 5L)
cv(v = 5L, by = "elevation")
grouped_cv("site")
block_cv(c("x", "y"), v = 4L)
env_cv(c("elevation", "slope"), v = 5L)
```
