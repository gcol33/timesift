# Bring an already-reduced feature table into a ladder

A representation the package did not build, such as a published set of
hand-aggregated climate summaries, enters here. It becomes a one-channel
`[unit, feature, 1]` array, which is what a learner reads, so a feature
table and a temporal grain can be arms of the same
[`grain_ladder()`](https://gillescolling.com/timesift/reference/grain_ladder.md)
and be scored on the same cells by the same rule.

## Usage

``` r
feature_matrix(m, label = "features")
```

## Arguments

- m:

  A matrix or data frame of units by features, with unit identifiers in
  the row names or in a leading character or factor column.

- label:

  The name the arm is reported under.

## Value

A `timesift_matrix` of shape `[unit, feature, 1]`.

## Details

It carries no time axis, because it has none: the reduction already
happened, elsewhere, and what reaches the model is a list of numbers per
unit. That is the whole point of comparing against it.

## Examples

``` r
m <- matrix(rnorm(30), nrow = 10,
            dimnames = list(sprintf("p%02d", 1:10), paste0("bio", 1:3)))
feature_matrix(m)
```
