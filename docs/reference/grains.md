# Several representations to run the same learners across

The set a learner without a `data =` of its own is fitted at every
member of. `grains()` names calendar grains, `lookbacks()` names
lookback spans, and either takes its arguments as separate names or as
one vector.

## Usage

``` r
grains(..., stats = "mean", year_start = "09-01")

lookbacks(..., lag = "0 days", bins = 1L, stats = "mean")

timesift_sift(x)
```

## Arguments

- ...:

  Grain names for `grains()`, lookback spans for `lookbacks()`, given as
  separate arguments or as one vector.

- stats:

  Statistics computed per bin, one channel each, in the order given. See
  [`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md)
  for the seven and for what separates an extreme reading from an
  extreme day.

- year_start:

  `"MM-DD"` boundary of the hydrological year, used by `"season"` and
  `"year"`.

- lag:

  The gap between a target's instant and the end of its lookback.

- bins:

  Sub-bins the lookback is cut into, oldest first. One gives a block of
  features, several give a sequence.

- x:

  A named list of representations, a single representation, or a
  character vector of grain names.

## Value

A `timesift_sift`: a named list of representations, labelled by each
one's own label where the list was not named.

## Details

`grains("auto")` is the set of named grains the record gives at least
two bins, in the order `native`, `halfday`, `day`, `week`, `month`,
`season`, `year`, leaving out any grain the requested statistics are not
defined at. There is no cap on it: over three years of hourly readings
`native` is 26304 bins, and a caller who does not want that names the
grains instead.

## Examples

``` r
grains("day", "week", "month")
grains(c("month", "year"), stats = c("cold_day", "mean", "warm_day"))
lookbacks("30 days", "90 days", bins = 3L)
```
