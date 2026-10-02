# A metric of ordinal classes

The statistics biomod2 reads a model of an ordinal response by. The
response is a column of whole-number classes, the model predicts a
number on the same scale, and each prediction is read as the observed
class nearest to it, the lower class on a tie. With `m[i, j]` the units
of observed class `j` predicted as class `i`, over the `k` classes
observed in the cell:

## Usage

``` r
ordinal_metric(y, p, metric = names(.ordinal_metrics))
```

## Arguments

- y:

  Observed values.

- p:

  Predictions for the same units, in the same order.

- metric:

  One of the names in the table.

## Value

One number, or `NA` where the cell defines none.

## Details

|  |  |
|----|----|
| `metric` | reads |
| `"accuracy"` | `sum(diag(m)) / sum(m)` |
| `"recall"` | the mean over the `k` classes of `m[j, j]` over the units of class `j` |
| `"precision"` | the mean over the `k` classes of `m[i, i]` over the units predicted as `i` |
| `"f1"` | `2 P R / (P + R)` of the two means, `NA` where both are zero |

A class with no unit, or in which nothing is predicted, adds zero to its
mean. The four are registered as `ordinal_accuracy`, `ordinal_recall`,
`ordinal_precision` and `ordinal_f1`.

## Examples

``` r
ordinal_metric(c(1, 1, 2, 3, 3), c(1.2, 2.4, 2, 2.8, 3.4), "accuracy")
```
