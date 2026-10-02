# Draw a response curve

One line per response against the value of the predictor. For two
predictors, the prediction of one response as a filled image over the
grid of the two.

## Usage

``` r
# S3 method for class 'timesift_response_curve'
plot(x, variable = NULL, col = NULL, ...)
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

- ...:

  Passed to
  [`graphics::plot()`](https://rdrr.io/r/graphics/plot.default.html) or
  [`graphics::image()`](https://rdrr.io/r/graphics/image.html).

## Value

The data frame drawn, invisibly.
