# Draw how stable the choice of grain was

Every candidate's inner score in every outer fold, one line per outer
fold, with an open circle on the candidate that fold selected. A
selection that lands on the same candidate each time draws its circles
in one column; one that wanders says the grid is flat enough that the
choice is arbitrary, which is worth seeing beside the estimate rather
than after it.

## Usage

``` r
# S3 method for class 'timesift_selection'
plot(x, col = NULL, ...)
```

## Arguments

- x:

  A
  [`select_grain()`](https://gillescolling.com/timesift/reference/select_grain.md)
  result.

- col:

  One colour per outer fold, recycled.

- ...:

  Passed to
  [`graphics::plot()`](https://rdrr.io/r/graphics/plot.default.html).

## Value

The table of inner scores the plot is drawn from, invisibly.

## Examples

``` r
set.seed(1)
t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24 * 200)
units <- sprintf("p%02d", 1:60)
warmth <- rnorm(60)
d <- data.frame(
  plot = rep(units, each = length(t)), t = rep(t, length(units)),
  temp = as.numeric(vapply(warmth, function(w) w + sin(seq_along(t) / 300) + rnorm(length(t)),
                           numeric(length(t)))))
y <- matrix(rbinom(120, 1, plogis(c(warmth, -warmth))), nrow = 60,
            dimnames = list(units, c("sp1", "sp2")))
x <- grain_matrix(d, plot, t, temp, grain = c("week", "month"))
sel <- select_grain(x, y, elasticnet(), folds = fold_map(y, v = 3), inner = 3,
                    verbose = FALSE)
plot(sel)
```
