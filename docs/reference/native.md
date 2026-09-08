# How a series becomes an array a learner reads

A representation is the reduction the package exists to make explicit:
the record unreduced, the record at a calendar grain, several grains
bound into one block of features, or a lookback of fixed length ending
at each target's own instant. It carries the settings and nothing else,
so the same object describes a representation before any record has been
seen, names the arm it produced in a fitted object, and rebuilds itself
for new targets in
[`predict.timesift()`](https://gillescolling.com/timesift/reference/predict.timesift.md).

## Usage

``` r
native(stats = "mean", year_start = "09-01")

grain(grain, stats = "mean", year_start = "09-01")

multigrain(grains = NULL, stats = "mean", year_start = "09-01")

lookback(span, lag = "0 days", bins = 1L, stats = "mean")
```

## Arguments

- stats:

  Statistics computed per bin, one channel each, in the order given. See
  [`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md)
  for the seven and for what separates an extreme reading from an
  extreme day.

- year_start:

  `"MM-DD"` boundary of the hydrological year, used by `"season"` and
  `"year"`.

- grain:

  One of `"native"`, `"halfday"`, `"day"`, `"week"`, `"month"`,
  `"season"`, `"year"`, or a function of the reading instants returning
  each reading's bin start. See
  [`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md).

- grains:

  Grains bound side by side into one block, or `NULL` for the automatic
  set
  [`grains()`](https://gillescolling.com/timesift/reference/grains.md)
  describes.

- span:

  The lookback's length, as a duration such as `"30 days"` or a number
  of seconds. See
  [`lookback_matrix()`](https://gillescolling.com/timesift/reference/lookback_matrix.md)
  for how a duration is read.

- lag:

  The gap between a target's instant and the end of its lookback.

- bins:

  Sub-bins the lookback is cut into, oldest first. One gives a block of
  features, several give a sequence.

## Value

A `timesift_representation`.

## Details

Every representation carries `label`, the name it is reported under;
`kind`, one of `"grain"`, `"multigrain"` and `"lookback"`; the settings
its kind uses; and `sequence`, which says whether its bins are ordered
in time and so mean something to a convolution. `native()`, `grain()`
and a `lookback()` of more than one bin are sequences; `multigrain()`
and a one-bin `lookback()` are blocks of features.

`multigrain()` flattens each of its grains to one row per target and
puts them side by side, so a column of the block names the grain, the
statistic and the bin it came from. It is the tabular representation a
penalised regression or a random forest reads.

## Examples

``` r
native()
grain("week", stats = c("cold_day", "mean", "warm_day"))
multigrain(c("month", "season"))
lookback("30 days", bins = 3L)
```
