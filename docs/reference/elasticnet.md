# Penalised regression on the flattened representation

One elastic net per variable, over every bin-by-channel column of the
representation and, by default, their squares. There is no discrete
selection step: the penalty path uses every column and shrinks, and the
penalty itself is chosen by an inner cross-validation on the fitting
units, so nothing about the model is decided outside the fold it is
fitted in.

## Usage

``` r
elasticnet(
  data = NULL,
  alpha = 0.5,
  n_inner = 5L,
  squares = TRUE,
  s = "lambda.min",
  n_lambda = 100L,
  thresh = 1e-08,
  seed = 1L
)
```

## Arguments

- data:

  A representation the learner is pinned to, or `NULL` to run across
  every representation of the run.

- alpha:

  Elastic-net mixing, `1` lasso and `0` ridge.

- n_inner:

  Folds of the inner cross-validation that chooses the penalty.

- squares:

  Add the square of every column, giving the same quadratic capacity a
  second-order polynomial term would.

- s:

  Which penalty of the inner path to predict at: `"lambda.min"`,
  `"lambda.1se"`, or a penalty of its own, which is interpolated between
  the two points of the path around it.

- n_lambda:

  Points of the penalty path.

- thresh:

  Where the coordinate descent stops, read off the largest coefficient
  move of a pass. The default leaves the fit as close to the optimum as
  glmnet's own default does; a looser one is faster and a tighter one
  costs time roughly in proportion.

- seed:

  Seed for the inner cross-validation's fold draw, which is random and
  would otherwise make the fit irreproducible.

## Value

A
[`learner()`](https://gillescolling.com/timesift/reference/learner.md).

## Details

The family is the response head's: a binary cross-entropy loss fits a
logistic model and a squared-error loss a linear one, so the learner is
the same under a presence-absence head and under a continuous one. So
are the case weights: the head's `weights`,
[`positive_weights()`](https://gillescolling.com/timesift/reference/positive_weights.md)
for presence-absence, are what every learner that ships fits under.

The inner folds are dealt for each response and stratified on it, so a
rare outcome is spread over them as evenly as its count allows. A
presence-absence response whose inner training sets cannot each hold two
of each outcome, the fewest a logistic path is fitted to, has too few of
one outcome to choose a penalty on. It is predicted its share among the
fitting units, as a response holding one outcome is, and the fit names
every such response in `unfitted`.

This is the aggregate-feature side of the comparison the package was
built for, and it is the fair opponent for a network: a per-fold
discrete selector pays selection variance a network never pays, so
beating that one is not a matched result.

The path is fitted by the same core the Python package calls, so the two
return the same coefficients for the same input. Its conventions are
glmnet's, which is what the arm is measured against: weights normalised
to sum to one, columns centred and scaled by their weighted mean and
weighted standard deviation, a hundred penalties down from the smallest
that leaves every coefficient at zero, and the held-out deviance read
fold by fold.

## Examples

``` r
elasticnet(alpha = 0.5)
```
