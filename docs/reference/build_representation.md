# Build one representation for a set of targets

Turns a representation and the two tables into the
`[target, bin, channel]` array a learner is fitted on. It is the one
place the fitting layer builds an array, so
[`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
and
[`predict.timesift()`](https://gillescolling.com/timesift/reference/predict.timesift.md)
reach a record the same way and a candidate refitted on new targets is
built from the settings its own arm was.

## Usage

``` r
build_representation(rep, series, targets, spec)
```

## Arguments

- rep:

  A representation from
  [`native()`](https://gillescolling.com/timesift/reference/native.md),
  [`grain()`](https://gillescolling.com/timesift/reference/native.md),
  [`multigrain()`](https://gillescolling.com/timesift/reference/native.md)
  or
  [`lookback()`](https://gillescolling.com/timesift/reference/native.md).

- series:

  The long table of readings, or `NULL` where the targets carry the
  whole predictor block.

- targets:

  The table of prediction targets.

- spec:

  The resolved settings
  [`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
  carries: the identifier, time, value, anchor and static columns it
  settled on.

## Value

A `timesift_matrix`.

## Details

The rows are the targets, in the order the fitting layer keeps them:
sorted by identifier where one target row belongs to each unit, and in
the targets' own order where `target_time` anchors them. Columns named
in `static` are appended as channels holding one value per target,
constant across the bins.

## Examples

``` r
set.seed(1)
t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24 * 90)
units <- sprintf("p%02d", 1:30)
warmth <- rnorm(30)
logger <- data.frame(
  plot = rep(units, each = length(t)), datetime = rep(t, 30),
  temp = as.numeric(vapply(warmth, function(w) w + sin(seq_along(t) / 300), numeric(length(t)))))
plots <- data.frame(plot = units, sp_a = rbinom(30, 1, plogis(2 * warmth)))
# \donttest{
fit <- timesift(plots, logger, y = sp_a, id = plot, time = datetime,
                sift = grains("month"), resampling = cv(v = 3L), ensemble = FALSE,
                verbose = FALSE)
x <- build_representation(grain("week"), logger, plots, fit$spec)
dim(x)
# }
```
