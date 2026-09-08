# Forward selection by AIC on the flattened representation

One generalised linear model per variable, its predictors chosen by
forward selection over every bin-by-channel column, admitting a column
while it lowers AIC and stopping at a fixed budget. Each candidate
enters as an orthogonal polynomial, so a term can be non-monotone in the
reading the way a niche optimum is. The family is the response head's:
logistic under a binary cross-entropy loss, Gaussian under a
squared-error one.

## Usage

``` r
stepwise(data = NULL, max_terms = 3L, degree = 2L)
```

## Arguments

- data:

  A representation the learner is pinned to, or `NULL` to run across
  every representation of the run.

- max_terms:

  Predictors admitted before selection stops.

- degree:

  Polynomial degree each admitted column enters at.

## Value

A
[`learner()`](https://gillescolling.com/timesift/reference/learner.md).

## Details

Selection happens inside whichever units the learner is handed, so under
[`grain_ladder()`](https://gillescolling.com/timesift/reference/grain_ladder.md)
it is redone in every fold. That is the footing the other learners are
fitted on. Reported beside a penalised fit it also prices discrete
selection: choosing a handful of columns out of hundreds is high
variance, and that variance is a cost of the selector rather than of the
features.

## Examples

``` r
stepwise(max_terms = 3)
```
