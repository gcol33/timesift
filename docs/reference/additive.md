# Generalised additive model on the flattened representation

One additive model per response, a smooth function of every
bin-by-channel column of the representation, fitted as mgcv's
`gam(y ~ s(x1) + s(x2) + ..., method = "GCV.Cp")` fits it and as biomod2
fits `GAM`. Each column enters as a thin plate regression spline of `k`
basis functions (Wood 2003): a cubic radial function centred on each of
the column's distinct values, reduced to its `k - 2` directions of
greatest eigenvalue, together with the linear function, which the
penalty on the spline's squared second derivative leaves free. Each
smooth sums to zero over the units, beside one intercept.

## Usage

``` r
additive(data = NULL, k = 10L, gamma = 1, max_knots = 2000L, threads = 1L)
```

## Arguments

- data:

  A representation the learner is pinned to, or `NULL` to run across
  every representation of the run.

- k:

  The basis dimension of a column's smooth, the constant included, at
  least 3.

- gamma:

  The criterion's charge per effective degree of freedom.

- max_knots:

  The most distinct values of a column the radial functions are centred
  on.

- threads:

  Columns and responses worked at once. The model is the same on any
  number.

## Value

A
[`learner()`](https://gillescolling.com/timesift/reference/learner.md).

## Details

The coefficients maximise the penalised likelihood, by penalised
iteratively reweighted least squares, and the smoothing parameters, one
per column, minimise the unbiased risk estimator under presence-absence
and a count head and the generalised cross-validation score under a
squared-error head, by Newton's method with the exact derivatives (Wood
2008). `gamma` multiplies the charge each effective degree of freedom
adds to that criterion, as mgcv's `gamma` does; above one it gives
smoother fits. Over columns as alike as neighbouring weeks the criterion
can have more than one local minimum; the one the search settles in then
depends on where it starts, and mgcv, starting from a rule of its own
parametrisation, can settle in another.

The defaults are mgcv's, which biomod2 passes unchanged: `k = 10` and a
criterion charge of one. Above `max_knots` distinct values, the radial
functions are centred on that many of them, drawn as mgcv draws them, so
that the basis is the one mgcv fits. A column of fewer than `k` distinct
values takes as many basis functions as it holds values, a column of two
enters linearly, and a column of one is left out; mgcv refuses the first
and the last of those. A column whose linear part the columns before it
already span keeps its penalised part and loses the linear one.

The case weights are the response head's,
[`positive_weights()`](https://gillescolling.com/timesift/reference/positive_weights.md)
under presence-absence, and enter the likelihood as mgcv's prior
weights. The model holds at most as many coefficients as there are
units, as mgcv's does: one for the intercept and `k - 1` per column.
Under the shipped presence-absence head those weights are on, so a
default `additive()` is mgcv's specification fitted under them; a head
registered without `weights` fits it unweighted.

The basis, the fit and the choice of the smoothing parameters live in
the core the Python package calls, pinned against mgcv in the fixtures,
so the two languages fit the same model. A response holding one value is
predicted its mean and named in `unfitted`; one whose smoothing
parameter search stopped short of its tolerance is named in `stopped`.

## Examples

``` r
additive()
additive(data = grain("season"), k = 5)
```
