# The area under the ROC curve

The whole curve rather than its best point. TSS is a maximum over cuts,
so a small change in a prediction often moves it not at all and then
moves it a long way; the area responds to every reordering, which is
what makes it the steadier reading when many rescorings are compared, as
in
[`occlusion()`](https://gillescolling.com/timesift/reference/occlusion.md).
Tied predictions take the average rank.

## Usage

``` r
roc_auc(y, p)
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
roc_auc(c(0, 0, 1, 1), c(0.1, 0.2, 0.8, 0.9))
```
