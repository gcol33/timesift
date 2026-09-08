# The weights the combiner fitted

The weights the combiner fitted

## Usage

``` r
ensemble_weights(fit)
```

## Arguments

- fit:

  A `timesift` result, or the stack itself.

## Value

A named numeric vector, or `NULL` where the run fitted no combiner.

## Examples

``` r
set.seed(1)
y <- matrix(rbinom(200, 1, 0.4), nrow = 50,
            dimnames = list(sprintf("p%02d", 1:50), paste0("sp", 1:4)))
folds <- fold_map(y, v = 5)
truth <- matrix(runif(200), nrow = 50, dimnames = dimnames(y))
oof <- list(good = 0.8 * y + 0.2 * truth, noise = truth)
ensemble_weights(ensemble_fit(oof, y, scorable_cells(y, folds), folds))
```
