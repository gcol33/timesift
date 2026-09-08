# How the folds are drawn

The resampling
[`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
scores on. `cv()` deals units into folds balanced on the stratifying
value; `grouped_cv()` deals whole groups, keeping every target sharing a
group value on one side of each split.

## Usage

``` r
cv(v = 10L, seed = 1L, strata = 5L)

grouped_cv(group, v = 10L, seed = 1L)
```

## Arguments

- v:

  Number of folds.

- seed:

  Random seed, fixed so the map is reproducible.

- strata:

  Number of strata, or `1` for no stratification.

- group:

  The grouping: the name of a column of `targets`, or a vector with one
  value per target.

## Value

A `timesift_resampling`.

## Details

`resampling` also accepts a fold vector or a
[`fold_map()`](https://gillescolling.com/timesift/reference/fold_map.md)
result directly, which is how a split the package has no constructor for
– a spatial block, a season held out whole – reaches the same fitting
path.

## Examples

``` r
cv(v = 5L)
grouped_cv("site")
```
