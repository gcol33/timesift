# Multivariate adaptive regression splines on the flattened representation

One MARS model per response, over every bin-by-channel column of the
representation, fitted as the earth package fits it and as biomod2 fits
`MARS`. The forward pass starts from the intercept and at each step
multiplies a term already in the model by a pair of hinges on one
column, `max(0, x - t)` and `max(0, t - x)`, taking the parent, the
column and the knot `t` that most reduce the residual sum of squares of
a least-squares fit to the response. A knot at a column's least value
enters the column linearly. `degree` bounds how many hinges a term
multiplies, so `degree = 1` is an additive model and `2` admits pairwise
interactions. The pass stops at `max_terms` terms, when a step raises
the R-squared by less than `min_gain`, or when no term reduces the
residuals.

## Usage

``` r
mars(
  data = NULL,
  degree = 1L,
  penalty = NULL,
  max_terms = NULL,
  min_gain = 0.001,
  minspan = 0L,
  endspan = 0L,
  fast_k = 20L,
  fast_beta = 1,
  prune = TRUE,
  nprune = NULL,
  threads = 1L
)
```

## Arguments

- data:

  A representation the learner is pinned to, or `NULL` to run across
  every representation of the run.

- degree:

  The most hinges a term multiplies.

- penalty:

  The generalised cross-validation's charge per knot. `NULL` is earth's,
  2 at degree one and 3 above; -1 charges nothing.

- max_terms:

  The most terms the forward pass reaches, the intercept included.
  `NULL` is `min(200, max(20, 2 p)) + 1` for `p` columns.

- min_gain:

  The least rise in R-squared a forward step is kept for.

- minspan, endspan:

  Units between knots, and units at either end of a column no knot is
  placed among. 0 is Friedman's rule; a negative `minspan` asks for that
  many knots per column.

- fast_k, fast_beta:

  Fast MARS: parents tried at each step, and how fast an untried parent
  ages up the queue. `fast_k = 0` tries every term.

- prune:

  Whether the pruning pass runs. Without it every term of the forward
  pass is kept.

- nprune:

  The most terms kept, the intercept included; `NULL` for no bound.

- threads:

  Responses fitted at once, or, with one response to fit, columns
  searched at once. The model is the same on any number.

## Value

A
[`learner()`](https://gillescolling.com/timesift/reference/learner.md).

## Details

The pruning pass then removes terms one at a time, each time the one
whose loss raises the residuals least, and keeps the subset of least
generalised cross-validation, which charges `penalty` for each knot.
Under a presence-absence head the kept terms are refitted as a logistic
model, as earth's `glm = list(family = binomial)` refits them and
biomod2 asks it to, and the prediction is that model's probability;
under a count head they are refitted as a Poisson model with a log link,
as earth's `glm = list(family = poisson)` refits them, and the
prediction is its mean count; under a squared-error head they are
refitted by least squares.

The defaults are earth's own, which biomod2 uses under its default
option set and under `"bigboss"` alike: degree one, `penalty = 2`,
`min_gain = 0.001`, `max_terms = min(200, max(20, 2 p)) + 1` for `p`
columns, Friedman's rules for the spans between knots, and Fast MARS
over the 20 best parents. The forward pass, the pruning pass and the
refit live in the core the Python package calls, pinned against earth in
the fixtures, so the two languages keep the same terms and return the
same coefficients.

The case weights are the response head's,
[`positive_weights()`](https://gillescolling.com/timesift/reference/positive_weights.md)
under presence-absence, and weigh both passes and the refit as earth
weighs them. earth refits the whole basis by QR at every candidate knot
of a weighted fit; the core reaches the same residual sums with
Friedman's running updates, which is what makes a weighted fit over
hundreds of columns affordable. Under the shipped presence-absence head
those weights are on, so a default `mars()` is earth's specification
fitted under them; a head registered without `weights` fits it
unweighted.

A response holding one value is predicted its mean.

## Examples

``` r
mars()
mars(degree = 2L)
mars(data = grain("season"), nprune = 10L)
```
