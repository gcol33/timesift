# Several built representations of the same targets

A named list of arrays covering the same targets and differing only in
how the record was reduced: several calendar grains, several lookback
spans, a block of features beside a sequence. It is keyed by the label
each representation is reported under, which is what
[`grain_ladder()`](https://gillescolling.com/timesift/reference/grain_ladder.md)
fits across and what a candidate in a
[`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
fit is named by.

## Usage

``` r
timesift_set(x)
```

## Arguments

- x:

  A named list of built representations –
  [`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md),
  [`lookback_matrix()`](https://gillescolling.com/timesift/reference/lookback_matrix.md)
  or
  [`feature_matrix()`](https://gillescolling.com/timesift/reference/feature_matrix.md)
  results – or a single one.

## Value

A `timesift_set`: the list, keyed by representation label.

## Examples

``` r
t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24 * 60)
d <- data.frame(plot = rep(c("a", "b"), each = length(t)), t = rep(t, 2),
                temp = rnorm(2 * length(t)))
s <- grain_matrix(d, plot, t, temp, grain = c("day", "week", "month"))
s
names(s)
```
