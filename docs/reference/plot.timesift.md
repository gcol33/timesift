# Draw a run

One line per learner across the representations it ran on, read the way
a ladder is read, and the level the combined prediction reaches drawn
across them. Where the ensemble line sits above every curve the
candidates are carrying different parts of the signal, and where it sits
on the best curve they are not.

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
