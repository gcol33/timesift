# How far the members of an ensemble disagree

biomod2's `EMcv` and `EMci`, read on the members' predictions under the
weights the stack carries. For each target and response: the weighted
mean `m` of the members' predictions; their weighted standard deviation
`s`, the square root of `sum(w * (p - m)^2) / (1 - sum(w^2))`, which is
the sample standard deviation when the weights are equal; the
coefficient of variation `s / m`; and the interval
`m -+ qt(1 - alpha / 2, n - 1) * s * sqrt(sum(w^2))`, `n` the number of
members carrying weight, which is the t interval of a mean of `n`
members when the weights are equal. Under a response head whose
predictions are probabilities, the interval is held inside zero and one.
An uncertainty map is this on one target per map cell.

## Usage

``` r
ensemble_spread(stack, preds, alpha = 0.05)
```

## Arguments

- stack:

  A
  [`ensemble_fit()`](https://gillescolling.com/timesift/reference/ensemble_fit.md)
  result.

- preds:

  Named list of `[target, response]` matrices, one per member of the
  stack.

- alpha:

  One minus the interval's coverage.

## Value

A `[target, response, statistic]` array, the statistics being `mean`,
`sd`, `cv`, `lower` and `upper`. `sd`, `cv` and the interval are `NA`
where fewer than two members carry weight.

## Details

A committee's and a median's members are read at equal weight, and a
committee's spread is that of the members' predictions rather than of
their votes.

## Examples

``` r
set.seed(1)
y <- matrix(rbinom(200, 1, 0.4), nrow = 50,
            dimnames = list(sprintf("p%02d", 1:50), paste0("sp", 1:4)))
folds <- fold_map(y, v = 5)
truth <- matrix(runif(200), nrow = 50, dimnames = dimnames(y))
oof <- list(good = 0.8 * y + 0.2 * truth, fair = 0.6 * y + 0.4 * truth, noise = truth)
st <- ensemble_fit(oof, y, scorable_cells(y, folds), folds, ensemble("mean"))
ensemble_spread(st, oof)[1:3, "sp1", ]
```
