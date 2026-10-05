# Bayesian logistic model with a spatial field

One model per response: a logistic regression on every bin-by-channel
column of the representation, standardised so the prior on a coefficient
means the same thing for each, optionally with an intercept for each
unit and a Gaussian-process field over the targets' coordinates. A unit
then carries what its own record says and what its neighbours' presences
say, under the folds every other learner is scored on.

## Usage

``` r
hierarchical(
  data = NULL,
  spatial = c("none", "nngp", "hsgp"),
  random = FALSE,
  cov = c("exponential", "matern32", "matern52", "gaussian"),
  neighbours = 15L,
  m = 6L,
  boundary = 1.5,
  nodes = 5L,
  threads = 1L
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

  An intercept for each unit.

- cov:

  The covariance of the `"nngp"` field: `"exponential"`, `"matern32"`,
  `"matern52"` or `"gaussian"`, `sigma^2 exp(-d / range)` for the first.

- neighbours:

  The neighbours each location of an `"nngp"` field is conditioned on.

- m:

  Eigenfunctions per axis of an `"hsgp"` field.

- boundary:

  The factor by which an `"hsgp"` field's box is wider than the
  coordinates.

- nodes:

  Grid points per hyperparameter.

- threads:

  Conditional fits over the hyperparameter grid run at once. What comes
  back does not depend on it.

## Value

A
[`learner()`](https://gillescolling.com/timesift/reference/learner.md).

## Details

The coefficients, the intercept included, have a `N(0, 2.5^2)` prior. A
unit's intercept is `N(0, sd^2)` and the field has a marginal standard
deviation and a range, each standard deviation under a
penalised-complexity prior (Simpson et al. 2017) with
`P(sd > 3) = 0.01`, and the range under one anchored at a fifth of the
coordinates' extent with `P(range < anchor) = 0.5` (Fuglstad et al.
2019). Coordinates are centred and divided by one factor, so distances
keep their proportions.

The field is a Gaussian process on the two columns named by `coords` in
[`timesift()`](https://gillescolling.com/timesift/reference/timesift.md).
`"hsgp"` is the Hilbert-space approximation (Solin and Sarkka 2020) with
`m` Laplacian eigenfunctions per axis, and `"nngp"` the
nearest-neighbour process (Datta et al. 2016) over the distinct
locations, each conditioned on its `neighbours` nearest among those
before it in lexicographic order of the coordinates, with the covariance
`cov`, whose range is the field's. Its sparse precision is factored once
per set of hyperparameters, so the fit scales with the number of
locations rather than their square.

Inference is Laplace's method over the coefficients, the intercepts and
the field together, and the hyperparameters are integrated over on a
grid of `nodes` points per hyperparameter, centred on the mode of their
posterior and weighted by it (Rue, Martino and Chopin 2009). With an
intercept alone its standard deviation is set at the mode of its
posterior; with neither the fit is the posterior mode of the
coefficients, so at `spatial = "none"` and `random = FALSE` the learner
is a penalised logistic model, every coefficient shrunk towards zero by
its prior. A prediction at new units interpolates the field to their
coordinates, so
[`predict.timesift()`](https://gillescolling.com/timesift/reference/predict.timesift.md)
is given targets that carry the same coordinate columns. The head's case
weights enter the likelihood in every configuration. A response holding
one value is predicted its mean and named in `unfitted`. The learner
fits a presence-absence head.

With `random = TRUE` each unit, as named by `id` in
[`timesift()`](https://gillescolling.com/timesift/reference/timesift.md),
gets an intercept, which absorbs what its several targets share beyond
the record: it is identified where a unit carries more than one target,
as an anchored fit's units do. A prediction adds a unit's intercept
where the unit was in the fit and leaves it at zero, the population
level, where it was not, so a unit held out whole is predicted from its
record and its place alone.

Both languages call one C++ core, `src/ts_hierarchical.cpp`.

## Examples

``` r
hierarchical()
hierarchical(data = grain("month"), spatial = "hsgp")
```
