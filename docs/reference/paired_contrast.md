# Compare two arms cell by cell

Two arms scored on the same held-out units do not necessarily have the
same set of defined cells, so a difference of two marginal means is not
a difference between the arms. This takes the difference inside each
`(variable, fold)` cell both arms scored, averages it within a variable
over its folds, and summarises those per-variable means, the variables
being the independent replicates.

## Usage

``` r
paired_contrast(ladder, a, b)
```

## Arguments

- ladder:

  A
  [`grain_ladder()`](https://gillescolling.com/timesift/reference/grain_ladder.md)
  result.

- a, b:

  The two arms, each given as `"learner"` or `"grain|learner"`. Naming a
  learner alone takes its best grain.

## Value

A one-row data frame: the mean per-variable difference, a 95 percent
interval from its standard error across variables, the number of
variables the difference favours, the paired cells and variables it
rests on, and a Wilcoxon signed-rank p-value.

## Details

Pairing also cancels what a threshold-selected metric carries in its
level. TSS read at the threshold that maximises it is biased upward
where presences are thin, both arms carry the same bias on the same
cell, and it cancels in the difference. That is why the levels a ladder
reports are upper bounds while the differences between arms are read at
face value.

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
lad <- grain_ladder(x, y, elasticnet(), folds = fold_map(y, v = 3), verbose = FALSE)
paired_contrast(lad, "week|elasticnet", "month|elasticnet")
```
