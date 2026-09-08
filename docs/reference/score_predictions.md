# Score held-out predictions on the cells the mask allows

The scoring every arm of a ladder and every candidate of a run goes
through, reachable on its own for a prediction matrix that came from
somewhere else: a combination of arms, a model fitted outside the
package, predictions read back from a file.

## Usage

``` r
score_predictions(y, p, folds, cells = NULL, metric = "tss")
```

## Arguments

- y:

  The response, as a matrix of units by variables.

- p:

  Held-out predictions for the same units and variables.

- folds:

  A
  [`fold_map()`](https://gillescolling.com/timesift/reference/fold_map.md)
  result, or one fold per unit.

- cells:

  A
  [`scorable_cells()`](https://gillescolling.com/timesift/reference/scorable_cells.md)
  mask. Computed from `y` and `folds` when left unset.

- metric:

  Name of the registered metric to read the cells by.

## Value

A data frame of one row per variable and fold, carrying the score and
whether the cell was scorable.

## Details

A cell is one response in one fold. Scoring only the cells the mask
allows is what keeps two arms comparable, so the mask is computed from
the response and the fold map alone and never from a model.

## Examples

``` r
set.seed(1)
y <- matrix(rbinom(120, 1, 0.4), nrow = 30,
            dimnames = list(sprintf("p%02d", 1:30), paste0("sp", 1:4)))
p <- matrix(runif(120), nrow = 30, dimnames = dimnames(y))
head(score_predictions(y, p, fold_map(y, v = 3)))
```
