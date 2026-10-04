# Single-hidden-layer network on the flattened representation

One network per response, over every bin-by-channel column of the
representation, fitted as the nnet package fits it and as biomod2 fits
`ANN`. Each of `hidden` logistic units takes a bias and every column,
and the output takes a bias, every hidden unit and, with `skip`, every
column again. The weights start uniform on `[-range, range]` and are
fitted by the variable metric (BFGS) method of Nash (1990), the
minimiser nnet uses, on the response head's loss plus `decay` times the
sum of the squared weights, biases included. The fit stops after
`max_iter` iterations, when the objective falls below `abs_tol`, or when
an iteration lowers it by no more than `rel_tol` of itself.

## Usage

``` r
perceptron(
  data = NULL,
  hidden = NULL,
  decay = NULL,
  range = NULL,
  max_iter = NULL,
  skip = FALSE,
  standardise = FALSE,
  abs_tol = 1e-04,
  rel_tol = 1e-08,
  preset = c("default", "bigboss"),
  max_hessian = 2,
  threads = 1L,
  seed = 1L
)
```

## Arguments

- data:

  A representation the learner is pinned to, or `NULL` to run across
  every representation of the run.

- hidden:

  Hidden units.

- decay:

  The weight of the squared weights' sum in the objective.

- range:

  The starting weights are uniform on `[-range, range]`.

- max_iter:

  The most iterations of the minimiser.

- skip:

  Whether the output also takes every column directly.

- standardise:

  Whether each column is centred and scaled before the fit.

- abs_tol, rel_tol:

  The fit stops when the objective falls below `abs_tol`, or when an
  iteration lowers it by no more than `rel_tol` of itself.

- preset:

  Whose defaults the settings left `NULL` take: `"default"` or
  `"bigboss"`.

- max_hessian:

  Gigabytes the minimisers' approximate inverse Hessians may take
  together.

- threads:

  Responses fitted at once.

- seed:

  Seed for the starting weights.

## Value

A
[`learner()`](https://gillescolling.com/timesift/reference/learner.md).

## Details

Under a presence-absence head the output is the logistic function of its
sum and the loss the cross-entropy, nnet's `entropy = TRUE`; under a
head with a squared-error loss the output is the sum itself and the loss
the sum of squares, nnet's `linout = TRUE`; under a count head the
output is the exponential of the sum and the loss the Poisson deviance,
which nnet does not offer. biomod2 leaves nnet's own `entropy = FALSE`,
fitting a presence-absence response by least squares on the logistic
output; the learner fits the head's loss, as every learner does.

`preset` says whose defaults the settings left `NULL` take. `"default"`
is what biomod2's default option set fits: two hidden units, as biomod2
sets them, and nnet's own `decay = 0`, `range = 0.7` and
`max_iter = 100`. `"bigboss"` is biomod2's tuned option set: five hidden
units, `decay = 0.1`, `range = 0.1` and `max_iter = 200`. A setting
given explicitly beats either.

nnet reads the columns as given, and so does the default: a record in
its own units saturates the hidden units sooner the wider its range.
With `standardise = TRUE` each column is centred on its mean and divided
by its sample standard deviation over the fitting units, and a
prediction centres and scales by the fit's own.

The network, its objective and the minimiser live in the core the Python
package calls, pinned against nnet in the fixtures from the same
starting weights, so the two languages fit the same network. The
starting weights are drawn from the core's own generator, so a fit does
not repeat nnet's from the same R seed. `threads` fit that many
responses at once, each network the same as when fitted alone. The
minimiser holds an approximate inverse Hessian of one number per pair of
weights for every network fitted at once; a fit that would need more
than `max_hessian` gigabytes for them is refused with the size, and a
coarser grain, fewer hidden units or fewer threads shrinks it.

The case weights are the response head's,
[`positive_weights()`](https://gillescolling.com/timesift/reference/positive_weights.md)
under presence-absence, and weigh each unit's term of the loss as nnet's
`weights` do. Under the shipped presence-absence head those weights are
on, so a default `perceptron()` is biomod2's `ANN` specification fitted
under them; a head registered without `weights` fits it unweighted.

A response holding one value is predicted its mean.

## Examples

``` r
perceptron()
perceptron(preset = "bigboss")
perceptron(hidden = 4L, decay = 0.01)
```
