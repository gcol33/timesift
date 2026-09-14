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
