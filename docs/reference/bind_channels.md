# Put channels side by side

Joins representations of the same units and bins into one array, in the
order given. It is how a temperature reading, an external product such
as snow cover, and the calendar position of each bin reach a model as
one input.

## Usage

``` r
bind_channels(...)
```

## Arguments

- ...:

  Two or more arrays of shape `[unit, bin, channel]`, agreeing on their
  units and bins.

## Value

One array carrying every channel, with the attributes of the first
argument.

## Examples

``` r
t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24 * 60)
d <- data.frame(plot = rep(c("a", "b"), each = length(t)), t = rep(t, 2),
                temp = rnorm(2 * length(t)))
x <- grain_matrix(d, plot, t, temp, grain = "week")
dimnames(bind_channels(x, calendar_channels(x)))[[3]]
```
