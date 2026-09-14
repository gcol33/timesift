# Average precision

The area under the precision-recall curve, as the step sum over the
distinct predictions: at each, the precision of calling every unit at or
above it a presence, weighted by the share of presences that cut adds.
Units sharing a prediction enter together, so the order they arrived in
does not move the score. Its floor is the prevalence rather than one
half, which makes it the reading of how well presences are ranked above
absences where presences are rare, and it is reported beside
[`roc_auc()`](https://gillescolling.com/timesift/reference/roc_auc.md)
for that reason.

## Usage

``` r
average_precision(y, p)
```

## Arguments

- y:

  Observed presence-absence, `0`/`1` or logical.

- p:

  Predicted scores for the same units, in the same order. Higher means
  presence.

## Value

One number, or `NA` where the cell defines none.

## Examples

``` r
average_precision(c(0, 0, 1, 1), c(0.1, 0.2, 0.8, 0.9))
average_precision(c(0, 1, 0, 1), c(0.1, 0.2, 0.8, 0.9))
```
