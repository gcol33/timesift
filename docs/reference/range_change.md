# How the range of each response changes between two projections

Counts the cells a response is lost from, kept in and gained, between a
map for the present and a map for another period or scenario, as
`BIOMOD_RangeSize()` does. A cell is in the range where the response is
predicted present.

## Usage

``` r
range_change(now, later, threshold = NULL)
```

## Arguments

- now, later:

  The two maps: a
  [terra::SpatRaster](https://rspatial.github.io/terra/reference/SpatRaster-class.html)
  with a layer per response, such as
  [`project()`](https://gillescolling.com/timesift/reference/project.md)
  with `type = "binary"` returns, or a matrix of cells by responses.
  Layers are matched by position and must be as many.

- threshold:

  `NULL` where the maps are already 0 and 1. Otherwise one cut per
  response, or one for all, at or above which a cell is present; the
  cuts
  [`decision_threshold()`](https://gillescolling.com/timesift/reference/kappa_score.md)
  learns for a fit are what to give.

## Value

A `timesift_range_change`: a list with the `table`, one row per
response, and the `map`, a raster or a matrix of the codes above, of the
shape of the inputs.

## Details

With `L` the cells lost, `K` the cells kept, `G` the cells gained and
`A` the cells absent in both: the current range is `L + K`, the later
one `K + G`, `percent_loss` is `100 L / (L + K)`, `percent_gain` is
`100 G / (L + K)` and `change` is `percent_gain - percent_loss`, the
change in range size as a share of the current one. A cell that is `NA`
in either map is left out of every count.

The `map` codes a cell as biomod2 does: `-2` lost, `-1` kept, `0` absent
in both and `1` gained.

## Examples

``` r
now <- cbind(sp1 = c(1, 1, 1, 0, 0, NA), sp2 = c(0, 0, 1, 1, 1, 1))
later <- cbind(sp1 = c(1, 0, 0, 1, 1, 1), sp2 = c(0, 0, 1, 1, 0, 0))
range_change(now, later)$table
```
