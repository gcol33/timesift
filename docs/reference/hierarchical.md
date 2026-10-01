# Bayesian logistic model with a spatial field, through tulpa

One model per response, fitted by the tulpa package: a logistic
regression on every bin-by-channel column of the representation,
standardised so the prior on a coefficient means the same thing for
each, and optionally a Gaussian-process field over the targets'
coordinates. A unit then carries what its own record says and what its
neighbours' presences say, under the folds every other learner is scored
on.

## Usage

``` r
hierarchical(
  data = NULL,
  spatial = c("none", "nngp", "hsgp"),
  random = FALSE,
  inference = NULL
)
```

## Arguments

- data:

  A representation the learner is pinned to, or `NULL` to run across
  every representation of the run.

- spatial:

  `"none"` for no field, `"nngp"` or `"hsgp"` for a Gaussian-process
  field over the coordinates.

- random:

  A random intercept for each unit.

- inference:

  tulpa's `mode`. `NULL` takes `"laplace"` without a field or an
  intercept, `"eb"` with an intercept alone, which estimates its
  standard deviation by empirical Bayes, and `"auto"` with a field,
  which integrates the field's hyperparameters.

## Value

A
[`learner()`](https://gillescolling.com/timesift/reference/learner.md).

## Details

The field is a Gaussian process on the two columns named by `coords` in
[`timesift()`](https://gillescolling.com/timesift/reference/timesift.md):
`"nngp"` is the nearest-neighbour approximation and `"hsgp"` the
Hilbert-space one, and the field's hyperparameters are integrated by
nested Laplace. A prediction at new units interpolates the field to
their coordinates, so
[`predict.timesift()`](https://gillescolling.com/timesift/reference/predict.timesift.md)
is given targets that carry the same coordinate columns. Without a field
no coordinates are needed and the model is a Bayesian logistic
regression, conditioned on the posterior mode by Laplace's method.

tulpa's nested Laplace and its random-effect integrators carry no case
weights on every route, so a fit with a field or an intercept is
unweighted whatever the response head weighs; without either, the head's
weights enter the likelihood. A response holding one value is predicted
its mean and named in `unfitted`. The learner fits a presence-absence
head: tulpa holds a Gaussian response's dispersion fixed unless it is
estimated by empirical Bayes, which a field does not allow.

With `random = TRUE` each unit, as named by `id` in
[`timesift()`](https://gillescolling.com/timesift/reference/timesift.md),
gets a random intercept, which absorbs what its several targets share
beyond the record: it is identified where a unit carries more than one
target, as an anchored fit's units do. Its standard deviation is
estimated, not conditioned on. A prediction adds a unit's intercept
where the unit was in the fit and leaves it at zero, the population
level, where it was not, so a unit held out whole is predicted from its
record and its place alone.

This learner is in the R package only, because tulpa is an R package.

## Examples

``` r
hierarchical()
hierarchical(data = grain("month"), spatial = "hsgp")
```
