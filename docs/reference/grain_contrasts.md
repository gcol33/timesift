# Compare every grain against a learner's best one

A mixed model on the per-cell scores of one learner,
`score ~ grain + (1 | variable) + (1 | fold)`, fitted by restricted
maximum likelihood. Every grain is then compared against the reference
by Dunnett's many-to-one procedure, which corrects for the comparisons
made without correcting for pairs nobody asked about.

## Usage

``` r
grain_contrasts(ladder, learner = NULL, reference = NULL, adjust = "mvt")
```

## Arguments

- ladder:

  A
  [`grain_ladder()`](https://gillescolling.com/timesift/reference/grain_ladder.md)
  result.

- learner:

  Which learner's grid to fit. The only one in the ladder by default.

- reference:

  The grain every other is compared against. The learner's best by
  default.

- adjust:

  Multiplicity adjustment passed to
  [`emmeans::contrast()`](https://rvlenth.github.io/emmeans/reference/contrast.html).

## Value

A data frame of one row per grain: the estimated marginal mean
difference from the reference, its interval, and the adjusted p-value.
Needs `lme4`, `lmerTest` and `emmeans`.

## Details

The design is balanced across variables, folds and grains, so each grain
is compared within a variable and within a fold and the variation
between variables cancels from the comparison. That is what lets a
difference of 0.015 hold up where absolute skill ranges across variables
by ten times as much.

Taking the best-observed grain as the reference is a choice that favours
the reference, so read this beside the paired differences
[`paired_contrast()`](https://gillescolling.com/timesift/reference/paired_contrast.md)
gives, which single out no grain.

## Examples

``` r
# \donttest{
set.seed(1)
t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24 * 200)
units <- sprintf("p%03d", 1:60)
warmth <- rnorm(60)
d <- data.frame(
  plot = rep(units, each = length(t)), t = rep(t, length(units)),
  temp = as.numeric(vapply(warmth, function(w) 1.5 * w + rnorm(length(t), sd = 20),
                           numeric(length(t)))))
y <- matrix(rbinom(60 * 6, 1, plogis(3 * as.numeric(outer(warmth, rep(c(1, -1), 3))))),
            ncol = 6, dimnames = list(units, paste0("sp", 1:6)))
x <- grain_matrix(d, plot, t, temp, grain = c("day", "week", "month"))
lad <- grain_ladder(x, y, elasticnet(), folds = fold_map(y, v = 5), verbose = FALSE)
grain_contrasts(lad)
# }
```
