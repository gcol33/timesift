# Combine one prediction per member into one prediction

Combine one prediction per member into one prediction

## Usage

``` r
ensemble_combine(stack, preds)
```

## Arguments

- stack:

  A
  [`ensemble_fit()`](https://gillescolling.com/timesift/reference/ensemble_fit.md)
  result.

- preds:

  Named list of `[target, response]` matrices, one per member of the
  stack.

## Value

One `[target, response]` matrix.

## Examples

``` r
set.seed(1)
y <- matrix(rbinom(200, 1, 0.4), nrow = 50,
            dimnames = list(sprintf("p%02d", 1:50), paste0("sp", 1:4)))
folds <- fold_map(y, v = 5)
truth <- matrix(runif(200), nrow = 50, dimnames = dimnames(y))
oof <- list(good = 0.8 * y + 0.2 * truth, noise = truth)
st <- ensemble_fit(oof, y, scorable_cells(y, folds), folds)
dim(ensemble_combine(st, oof))
```
