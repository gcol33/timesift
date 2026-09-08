# Penalised logistic regression on the flattened representation

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
  weight_positives = TRUE,
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

  Which penalty of the inner path to predict at.

- weight_positives:

  Weight presences by the ratio of absences to presences among the
  fitting units, so a rare variable is not fitted away.

- seed:

  Seed for the inner cross-validation's fold draw, which is random and
  would otherwise make the fit irreproducible.

## Value

A
[`learner()`](https://gillescolling.com/timesift/reference/learner.md).

## Details

This is the aggregate-feature side of the comparison the package was
built for, and it is the fair opponent for a network: a per-fold
discrete selector pays selection variance a network never pays, so
beating that one is not a matched result.

## Examples

``` r
elasticnet(alpha = 0.5)
```
