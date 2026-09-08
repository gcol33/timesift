# Draw a ladder

One line per learner across the grains, at the across-variable mean of
the per-variable score, with an interval from its standard error across
variables. An open circle marks each learner's best grain, which is
where the curve says the record stops paying for being read more finely.

## Usage

``` r
# S3 method for class 'timesift_ladder'
plot(x, col = NULL, interval = TRUE, ...)
```

## Arguments

- x:

  A
  [`grain_ladder()`](https://gillescolling.com/timesift/reference/grain_ladder.md)
  result.

- col:

  One colour per learner, recycled.

- interval:

  Draw the interval across variables.

- ...:

  Passed to
  [`graphics::plot()`](https://rdrr.io/r/graphics/plot.default.html).

## Value

The summary table the plot is drawn from, invisibly.

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
x <- grain_matrix(d, plot, t, temp, grain = c("day", "week", "month"))
lad <- grain_ladder(x, y, elasticnet(), folds = fold_map(y, v = 3), verbose = FALSE)
plot(lad)
```
