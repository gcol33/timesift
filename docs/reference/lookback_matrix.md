# Reduce sensor series to a lookback anchored on each target

Reads, for every target, a fixed length of record ending a fixed lag
before that target's own instant, and summarises it by one or more
statistics. It is the reduction a calendar cannot express: two targets
on the same unit a fortnight apart read two different stretches of the
same series, so the bins are relative to the target rather than to a
month or a week.

## Usage

``` r
lookback_matrix(
  data,
  id,
  time,
  value,
  at,
  span,
  lag = "0 days",
  bins = 1L,
  stats = "mean"
)
```

## Arguments

- data:

  A data frame of readings in long form, one row per reading.

- id:

  Column identifying the unit carrying the sensor. A bare column name or
  a string.

- time:

  Column of reading instants, `POSIXct`. A bare column name or a string.

- value:

  Column of readings, numeric. A bare column name or a string.

- at:

  A data frame of targets, whose first column is the unit and whose
  second is the anchor instant, `POSIXct`. One row per target; a unit
  may carry any number of them.

- span:

  The lookback's length, as a duration. See Durations.

- lag:

  The gap between the anchor and the end of the lookback, as a duration.
  Defaults to `"0 days"`, which ends the lookback at the anchor itself.

- bins:

  How many sub-bins the lookback is cut into, oldest first. `span` must
  divide by it exactly. One bin gives a block of features; several give
  a sequence a convolution can read.

- stats:

  Statistics to compute per bin, one channel each, in the order given.
  The same seven
  [`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md)
  carries.

## Value

A numeric array of shape `[target, bin, channel]`, of class
`timesift_matrix`. Its rows are the rows of `at`, in `at`'s own order,
named by `at`'s row names where it carries them and by position where it
does not. Its bins are named by where each one opens relative to the
anchor, oldest first, and its channels by the statistic. Attributes:

- `grain`: `"lookback"`.

- `span`, `lag`: the durations, resolved to seconds.

- `bins`: how many bins the lookback was cut into.

- `stats`: the statistic names in channel order.

- `bin_n`: a `[target, bin]` matrix of how many readings fell in each
  cell.

## Details

Bin `b` of a target anchored at `a` covers
`[a - lag - span + b * step, a - lag - span + (b + 1) * step)`, with
`step` the lookback's length divided by `bins` and `b` counted from
zero. The interval is closed at the left and open at the right, so a
reading on a boundary belongs to the later bin, and only the readings of
the target's own unit are read.

Every `(target, bin)` cell must hold at least one reading. A lookback
reaching past either end of the record is an error naming the target and
the interval, never a padded row: an invented value in front of a model
is worse than a target the record cannot answer for.

The four day-level statistics reduce each calendar day first, so they
are defined only where every day lies whole inside one bin. For a
lookback that is two conditions rather than one: `step` must be a whole
number of days, and `a - lag - span` must fall on a day boundary. Either
failing is an error naming the target.

## Durations

`span` and `lag` are read from a count and a unit – `"30 days"`,
`"12 hours"`, `"1 year"` – or from a bare number of seconds. A **year is
365 days and a month is 30 days** here. A lookback of a fixed length is
a fixed length, not a calendar step: the point of anchoring on the
target is that every target reads the same amount of record, which a
February and a leap year would take away. Where the calendar is what
matters,
[`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md)
is the call that follows it.

The units are `seconds`, `minutes`, `hours`, `days`, `weeks`, `months`
and `years`, singular or plural.

## Time zone

The calendar is the series', taken from the `tzone` attribute of `time`
as
[`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md)
takes it; a column with none is read as UTC. The anchors are instants
and are read as a clock in that same calendar, whatever zone `at`
carries, so one record is binned by one calendar.

## See also

[`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md),
the reduction that follows the calendar instead.

## Examples

``` r
t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24 * 60)
d <- data.frame(plot = rep(c("a", "b"), each = length(t)),
                t = rep(t, 2),
                temp = c(sin(seq_along(t) / 24), cos(seq_along(t) / 24)))
at <- data.frame(plot = c("a", "b"),
                 when = as.POSIXct(c("2021-10-20", "2021-10-25"), tz = "UTC"))
x <- lookback_matrix(d, plot, t, temp, at = at, span = "30 days", bins = 3L,
                   stats = c("cold_day", "mean", "warm_day"))
dim(x)
dimnames(x)[[2]]
```
