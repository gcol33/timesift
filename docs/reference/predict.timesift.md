# Predict from a fitted timesift

Rebuilds every member's representation for the new targets from the
settings its own arm was built with, predicts with the model refitted on
all targets, and combines them where the ensemble is asked for.

## Usage

``` r
# S3 method for class 'timesift'
predict(object, targets, series = NULL, candidate = "ensemble", ...)
```

## Arguments

- object:

  A
  [`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
  fit.

- targets:

  A data frame of targets, carrying the identifier, the anchor and the
  static columns the fit was given.

- series:

  The long table of readings for those targets, or `NULL` for a
  targets-only fit.

- candidate:

  `"ensemble"`, or the name of one candidate.

- ...:

  Ignored.

## Value

A `[target, response]` matrix of predictions, named by target and in the
order the fit carries its own targets: sorted by identifier, or the
targets' own order where `target_time` anchors them.
