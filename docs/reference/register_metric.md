# Register a metric

A metric scores one held-out cell: the observed response of the units in
a fold and a model's predictions for them. Registering one makes it
available to
[`grain_ladder()`](https://gillescolling.com/timesift/reference/grain_ladder.md)
by name, with no change to the fitting code.

## Usage

``` r
register_metric(name, fn, overwrite = FALSE)

metrics()
```

## Arguments

- name:

  Name the metric is asked for by.

- fn:

  A function of `(y, p)`, the observed values and the predictions for
  one cell, both vectors of the same length, returning one number or
  `NA` where the cell defines none.

- overwrite:

  Replace an existing registration.

## Value

The registered function, invisibly.

## Examples

``` r
register_metric("hit_rate", function(y, p) mean((p >= 0.5) == (y == 1)), overwrite = TRUE)
"hit_rate" %in% metrics()
```
