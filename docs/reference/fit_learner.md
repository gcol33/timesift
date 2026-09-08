# Fit one learner at one grain

Fit one learner at one grain

## Usage

``` r
fit_learner(learner, x, y, response = "presence_absence", control = NULL, ...)

# S3 method for class 'timesift_fit'
predict(object, newdata, ...)
```

## Arguments

- learner:

  A
  [`learner()`](https://gillescolling.com/timesift/reference/learner.md),
  or the name of a registered one.

- x:

  A
  [`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md)
  result.

- y:

  The response for the same units.

- response:

  Name of the registered response head. `"presence_absence"` ships.

- control:

  The run's
  [`train_control()`](https://gillescolling.com/timesift/reference/train_control.md).
  The learner's own control overrides it on the settings that control
  names, and a setting given in `...` overrides both.

- ...:

  Passed to the learner's `fit`.

- object:

  A `timesift_fit`.

- newdata:

  A representation of the same channels for the units to predict.

## Value

A `timesift_fit`, which
[`stats::predict()`](https://rdrr.io/r/stats/predict.html) takes a new
representation.

## Examples

``` r
set.seed(1)
t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24 * 120)
units <- sprintf("p%02d", 1:40)
d <- data.frame(plot = rep(units, each = length(t)), t = rep(t, length(units)),
                temp = as.numeric(replicate(length(units), rnorm(length(t)))))
x <- grain_matrix(d, plot, t, temp, grain = "month")
y <- matrix(rbinom(80, 1, 0.4), nrow = 40, dimnames = list(units, c("sp1", "sp2")))
fit <- fit_learner(elasticnet(), x, y)
dim(stats::predict(fit, x))
```
