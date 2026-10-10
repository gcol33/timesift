# Draw a response curve

One line per response against the value of the predictor. For two
predictors, the prediction of one response as a filled image over the
grid of the two.

## Usage

``` r
# S3 method for class 'timesift_response_curve'
plot(x, variable = NULL, col = NULL, legend = "topright", ...)
```

## Arguments

- x:

  A
  [`response_curve()`](https://gillescolling.com/timesift/reference/response_curve.md)
  result.

- variable:

  For two predictors, the response to draw; the first by default.

- col:

  One colour per response, recycled.

- legend:

  Where the legend of the responses goes, a position
  [`graphics::legend()`](https://rdrr.io/r/graphics/legend.html) takes
  such as `"bottomright"`, or `NULL` for none. Four responses go in each
  column of it.

- ...:

  Passed to
  [`graphics::plot()`](https://rdrr.io/r/graphics/plot.default.html) or
  [`graphics::image()`](https://rdrr.io/r/graphics/image.html).

## Value

The data frame drawn, invisibly.

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
                sift = grains("month"), resampling = cv(v = 3L), ensemble = FALSE,
                n_inner = NULL, verbose = FALSE)
plot(response_curve(fit, "elasticnet / month", "mean", n = 20))
# }
```
