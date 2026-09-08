# Fit the combiner on the out-of-fold predictions

The combiner sees the out-of-fold predictions, the response, the mask of
scorable cells and the fold map, and never a model. That is what keeps
it honest: there is no way for it to read anything a candidate fitted
in-sample, because it is not handed one.

## Usage

``` r
ensemble_fit(oof, y, cells, folds, spec = ensemble(), scores = NULL)
```

## Arguments

- oof:

  Named list of `[target, response]` matrices, one per candidate.

- y:

  The response matrix.

- cells:

  The scorable-cell mask from
  [`scorable_cells()`](https://gillescolling.com/timesift/reference/scorable_cells.md).

- folds:

  The fold map.

- spec:

  An
  [`ensemble()`](https://gillescolling.com/timesift/reference/ensemble.md)
  specification.

- scores:

  The per-cell scores of the run, a data frame carrying `candidate`,
  `variable`, `fold`, `score` and `scorable`. Read only where `spec`
  names no metric of its own.

## Value

A `timesift_stack`: `method`, the named `weights`, and what they were
fitted on.

## Examples

``` r
set.seed(1)
y <- matrix(rbinom(200, 1, 0.4), nrow = 50,
            dimnames = list(sprintf("p%02d", 1:50), paste0("sp", 1:4)))
folds <- fold_map(y, v = 5)
truth <- matrix(runif(200), nrow = 50, dimnames = dimnames(y))
oof <- list(good = 0.8 * y + 0.2 * truth, noise = truth)
ensemble_fit(oof, y, scorable_cells(y, folds), folds)
```
