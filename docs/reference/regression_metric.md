# A metric of a numeric response and its predictions

The statistics biomod2 reads an abundance model by. With `e = y - p`:

## Usage

``` r
regression_metric(y, p, metric = names(.regression_metrics))
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
| `"r_squared"` | `1 - sum(e^2) / sum((y - mean(y))^2)`, `NA` where `y` is constant |
| `"pearson"` | the correlation of `y` and `p`, `NA` where either is constant |
| `"rmse"` | `sqrt(mean(e^2))` |
| `"mse"` | `mean(e^2)` |
| `"mae"` | `mean(abs(e))` |
| `"max_error"` | `max(abs(e))` |

A comparison across candidates reads the highest score as the best, so
the four errors are registered under the names `neg_rmse`, `neg_mse`,
`neg_mae` and `neg_max_error` with their sign reversed, and `r_squared`
and `pearson` under their own names. A cell holding a prediction that is
not a number scores `NA`.

## Examples

``` r
y <- c(1, 2, 3, 4)
p <- c(1.5, 2, 2.5, 5)
regression_metric(y, p, "rmse")
regression_metric(y, p, "r_squared")
```
