# A representation as a block of predictors

Lays a `[unit, bin, channel]` array out as one row per unit and one
column per bin of each channel, which is what a model reading a table of
predictors takes. It is what the learners that read a tabular block fit
on, so a learner of your own that calls it reads the same columns they
do.

## Usage

``` r
# S3 method for class 'timesift_matrix'
as.matrix(x, ...)
```

## Arguments

- x:

  A
  [`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md),
  [`lookback_matrix()`](https://gillescolling.com/timesift/reference/lookback_matrix.md),
  [`feature_matrix()`](https://gillescolling.com/timesift/reference/feature_matrix.md)
  or
  [`build_representation()`](https://gillescolling.com/timesift/reference/build_representation.md)
  result.

- ...:

  Ignored.

## Value

A numeric `[unit, predictor]` matrix, the units as row names.

## Details

Columns are named `channel@bin`, channel by channel and bin by bin
within each. A channel that holds the same number in every bin, as a
`static` predictor does, is one column named by the channel alone.

## Examples

``` r
t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24 * 60)
d <- data.frame(plot = rep(c("a", "b"), each = length(t)), t = rep(t, 2),
                temp = sin(seq_len(2 * length(t)) / 24))
x <- grain_matrix(d, plot, t, temp, grain = "month", stats = c("min", "max"))
as.matrix(x)
```
