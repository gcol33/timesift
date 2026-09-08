# Where in the year each bin sits

An encoder that ends in global pooling discards when a thermal event
happened, so the position of a bin in the year has to be given to it as
input if it is to be used at all. These two channels carry that position
as the sine and cosine of the bin's fractional place in the year, which
is continuous across the turn of the year where the fraction itself is
not.

## Usage

``` r
calendar_channels(x)
```

## Arguments

- x:

  A
  [`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md)
  result.

## Value

An array of the same units and bins with two channels, `year_sin` and
`year_cos`, identical across units. Combine it with the readings using
[`bind_channels()`](https://gillescolling.com/timesift/reference/bind_channels.md).

## Details

They are the time index of each bin, not a summary of the readings, so
adding them introduces no hand-built thermal feature: whatever a model
does with them it could have done with a calendar.

## Examples

``` r
t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24 * 400)
d <- data.frame(plot = "a", t = t, temp = sin(seq_along(t) / 500))
x <- grain_matrix(d, plot, t, temp, grain = "month")
round(calendar_channels(x)[1, 1:4, ], 3)
```
