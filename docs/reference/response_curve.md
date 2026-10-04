# How a prediction responds to one predictor

Varies one predictor of a fitted candidate across the range it takes
while every other predictor is held at a reference value, and records
the prediction, as biomod2's response curves do. A predictor is a cell
of the representation the candidate reads: one statistic in one bin.
Given as the name of a channel, `"warm_day"`, it is that statistic moved
together in every bin, which is the way a column named in `static` is
moved and the way a statistic of the whole record is asked about. Given
as `list(bin = "2021-11", channel = "mean")` it is one cell, and a bare
bin name is enough where the representation has a single channel.

## Usage

``` r
response_curve(x, ...)

# Default S3 method
response_curve(x, ...)

# S3 method for class 'timesift'
response_curve(
  x,
  candidate = "ensemble",
  predictor,
  with = NULL,
  fixed = c("mean", "median", "min", "max"),
  n = 50L,
  spread = FALSE,
  ...
)
```

## Arguments

- x:

  A
  [`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
  result.

- ...:

  Passed to the method.

- candidate:

  The candidate to read, as
  [`summary()`](https://rdrr.io/r/base/summary.html) names it, or
  `"ensemble"`.

- predictor:

  The predictor to vary: a channel, a bin of a one-channel
  representation, or `list(bin = , channel = )`.

- with:

  A second predictor, given the same way. The prediction is then read
  over the grid of the two.

- fixed:

  What the other predictors are held at: `"mean"`, `"median"`, `"min"`
  or `"max"` of the cell over the targets.

- n:

  Number of values of a predictor, equally spaced over the range it
  takes.

- spread:

  For the ensemble, also return the members' standard deviation and
  interval at every value, as `predict(type = "spread")` reads them.

## Value

A data frame of one row per value and response, of class
`timesift_response_curve`: the value of the predictor (`value`, and
`value_with` for a second one), the response (`variable`) and the
`prediction`, with `sd`, `lower` and `upper` under `spread`.

## Details

The reference is one made-up unit whose every cell holds the `fixed`
summary of that cell over the targets. A cell that is the same for every
target, as a calendar channel is, keeps the value it has. The curve is
read off the model fitted on all targets, not off the per-fold models.

For the ensemble, every member is moved the same way: a member that does
not carry the predictor is held at the reference, and the members'
predictions are combined by the stack. A member reading another grain
carries a channel by its name, so an ensemble is asked about by channel.

## Examples

``` r
set.seed(1)
t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24 * 120)
units <- sprintf("p%02d", 1:40)
warmth <- rnorm(40)
d <- data.frame(
  plot = rep(units, each = length(t)), t = rep(t, length(units)),
  temp = as.numeric(vapply(warmth, function(w) w + sin(seq_along(t) / 300) + rnorm(length(t)),
                           numeric(length(t)))))
targets <- data.frame(plot = units, sp1 = rbinom(40, 1, plogis(warmth)),
                      sp2 = rbinom(40, 1, plogis(-warmth)))
fit <- timesift(targets, d, y = starts_with("sp"), id = plot, time = t, x = temp,
                learners = elasticnet(),
                sift = grains("month", stats = c("cold_day", "warm_day")),
                ensemble = FALSE, n_inner = NULL, resampling = cv(v = 3), verbose = FALSE)
head(response_curve(fit, "elasticnet / month", "warm_day", n = 5))
```
