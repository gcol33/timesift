# Draw a run

One line per learner across the representations it ran on, read the way
a ladder is read, and the stack's held-out score drawn across them, its
weights fitted inside each outer training fold. The curves are scored on
the folds a choice among them would be judged on, so the best of them
sits a little high; the ensemble line does not. Where the ensemble line
sits above every curve the candidates are carrying different parts of
the signal, and where it sits on or below the best curve they are not.

## Usage

``` r
# S3 method for class 'timesift'
plot(x, col = NULL, interval = TRUE, ...)
```

## Arguments

- x:

  A `timesift` result.

- col:

  One colour per learner, recycled.

- interval:

  Draw the interval across responses.

- ...:

  Passed to
  [`graphics::plot()`](https://rdrr.io/r/graphics/plot.default.html).

## Value

The table the plot is drawn from, invisibly.

## Examples

``` r
set.seed(1)
t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24 * 90)
units <- sprintf("p%02d", 1:30)
warmth <- rnorm(30)
logger <- data.frame(
  plot = rep(units, each = length(t)), datetime = rep(t, 30),
  temp = as.numeric(vapply(warmth, function(w) w + sin(seq_along(t) / 300), numeric(length(t)))))
plots <- data.frame(plot = units,
                    sp_a = rbinom(30, 1, plogis(2 * warmth)),
                    sp_b = rbinom(30, 1, plogis(-2 * warmth)))
# \donttest{
fit <- timesift(plots, logger, y = starts_with("sp_"), id = plot, time = datetime,
                sift = grains("week", "month"), resampling = cv(v = 3L), verbose = FALSE)
plot(fit)
# }
```
