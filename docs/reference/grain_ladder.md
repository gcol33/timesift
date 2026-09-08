# Fit at every grain and see where skill saturates

Cross-validates every learner at every grain of a representation set, on
one fold map and one mask of scorable cells, and returns the score of
each `(grain, learner, variable, fold)` cell. It is the measurement the
package exists for: how much of a record a model needs, read off the
point where making the record finer stops paying.

## Usage

``` r
grain_ladder(
  x,
  y,
  learners,
  folds = NULL,
  response = "presence_absence",
  metric = NULL,
  keep_fits = FALSE,
  verbose = TRUE
)

# S3 method for class 'timesift_ladder'
summary(object, ...)
```

## Arguments

- x:

  A
  [`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md)
  result, a
  [`timesift_set()`](https://gillescolling.com/timesift/reference/timesift_set.md),
  or a named list of representations.

- y:

  The response for the same units.

- learners:

  A learner, a set of them from [`c()`](https://rdrr.io/r/base/c.html),
  a list, or names of registered ones. An unnamed set is labelled by
  each learner's own name.

- folds:

  A fold map from
  [`fold_map()`](https://gillescolling.com/timesift/reference/fold_map.md),
  or any named integer vector. Built with the defaults of
  [`fold_map()`](https://gillescolling.com/timesift/reference/fold_map.md)
  when not given.

- response:

  Name of the registered response head.

- metric:

  Name of the registered metric, or `NULL` for the response's own.

- keep_fits:

  Keep every per-fold fitted model, which is what lets
  [`occlusion()`](https://gillescolling.com/timesift/reference/occlusion.md)
  read a fitted model without refitting it.

- verbose:

  Report each arm and each fold as it runs.

- object:

  A ladder.

- ...:

  Ignored, so that [`summary()`](https://rdrr.io/r/base/summary.html)
  takes the arguments its generic declares.

## Value

A data frame of one row per scored cell, of class `timesift_ladder`,
carrying the grain, the learner, the variable, the fold and the score.
The held-out prediction of every unit is kept in the `predictions`
attribute, and the scorable-cell mask in `cells`.

## Details

Every arm sees identical splits and is restricted to identical cells, so
the arms' means share a denominator and any two of them can be compared
cell by cell with
[`paired_contrast()`](https://gillescolling.com/timesift/reference/paired_contrast.md).

## Examples

``` r
set.seed(1)
t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24 * 200)
units <- sprintf("p%02d", 1:60)
warmth <- rnorm(60)
d <- data.frame(
  plot = rep(units, each = length(t)), t = rep(t, length(units)),
  temp = as.numeric(vapply(warmth, function(w) w + sin(seq_along(t) / 300) + rnorm(length(t)),
                           numeric(length(t)))))
y <- matrix(rbinom(120, 1, plogis(c(warmth, -warmth))), nrow = 60,
            dimnames = list(units, c("sp1", "sp2")))
x <- grain_matrix(d, plot, t, temp, grain = c("week", "month"))
lad <- grain_ladder(x, y, elasticnet(), folds = fold_map(y, v = 3), verbose = FALSE)
summary(lad)
```
