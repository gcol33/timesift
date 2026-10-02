# Draw the change in range

The map of cells lost, kept, absent and gained, for one response.

## Usage

``` r
# S3 method for class 'timesift_range_change'
plot(x, variable = NULL, ...)
```

## Arguments

- x:

  A
  [`range_change()`](https://gillescolling.com/timesift/reference/range_change.md)
  result made from rasters.

- variable:

  The response to draw; the first by default.

- ...:

  Passed to
  [`terra::plot()`](https://rspatial.github.io/terra/reference/plot.html).

## Value

`x`, invisibly.
