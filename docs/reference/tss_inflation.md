# How much a self-selected threshold inflates the reported level

The true skill statistic (TSS) is overestimated when the threshold is
chosen to maximise TSS on the same held-out observations used to report
the score. The fewer presences there are in the held-out data, the more
TSS tends to overestimate population skill. `tss_inflation()` estimates
the TSS that a given sampling design would report for a known population
skill.
[`implied_skill()`](https://gillescolling.com/timesift/reference/implied_skill.md)
reverses this calculation: given a reported TSS, it estimates the
population skill that would yield that score on average under the same
sampling design and assumed prediction distribution. This
simulation-based matching follows the principle of indirect inference
([Gouriéroux et al.,
1993](https://stat.cmu.edu/~cshalizi/462/readings/Indirect%20Inference%20%28Gourieroux%2C%20Monfort%20and%20Renault%29.pdf)).

## Usage

``` r
tss_inflation(y, folds, skill = c(0.6, 0.7, 0.9), replicates = 200L, seed = 1L)
```

## Arguments

- y:

  The response: a matrix or data frame of units by variables, with unit
  identifiers in the row names or in a leading character or factor
  column.

- folds:

  A fold map from
  [`fold_map()`](https://gillescolling.com/timesift/reference/fold_map.md),
  or any named integer vector of the same shape.

- skill:

  Population skill values to simulate.

- replicates:

  Replicates per value.

- seed:

  Random seed.

## Value

A data frame with one row per `skill` value: the skill, the mean level
read back, the inflation, and its interval across replicates.

## Details

Predictions are simulated for each value of `skill` as normal scores of
equal variance, at the cell sizes and presence counts of the response
and fold map supplied. The TSS is read at its best threshold exactly as
[`grain_ladder()`](https://gillescolling.com/timesift/reference/grain_ladder.md)
reports it, and the mean reading minus `skill` is the inflation.

The inflation is an expectation. A level read on one design is
optimistic on average, not a bound every reading sits above: a single
level can fall below the population skill. Predictions distributed
otherwise inflate by a different amount at the same skill, which is also
why a paired difference
[`paired_contrast()`](https://gillescolling.com/timesift/reference/paired_contrast.md)
takes is not free of it.

## Examples

``` r
set.seed(1)
y <- matrix(rbinom(1200, 1, 0.15), nrow = 200,
            dimnames = list(sprintf("p%03d", 1:200), paste0("sp", 1:6)))
tss_inflation(y, fold_map(y, v = 5), skill = c(0.6, 0.9), replicates = 40)
```
