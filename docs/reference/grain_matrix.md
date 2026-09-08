# Reduce sensor series to a temporal grain

Bins each unit's readings by the calendar and summarises every bin by
one or more statistics, returning the array a model is fitted on. The
reduction is the choice the package exists to make explicit: `grain`
sets how coarse the record becomes, `stats` sets what survives the
reduction, and the two are not interchangeable.

## Usage

``` r
grain_matrix(
  data,
  id,
  time,
  value,
  grain = "day",
  stats = "mean",
  year_start = "09-01",
  partial = c("keep", "drop")
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

- grain:

  One of `"native"`, `"halfday"`, `"day"`, `"week"`, `"month"`,
  `"season"`, `"year"`. The four coarse grains follow the calendar, so a
  bin is a real week or month rather than a fixed block of hours. Naming
  several grains returns one representation per grain, a
  [`timesift_set()`](https://gillescolling.com/timesift/reference/timesift_set.md).
  A function is called on the reading instants and must return the
  `POSIXct` start of each reading's bin, which is how a calendar the
  package does not carry, such as astronomical seasons, is binned.

- stats:

  Statistics to compute per bin, one channel each, in the order given.
  See Details.

- year_start:

  `"MM-DD"` boundary of the hydrological year, used by `"season"` and
  `"year"`. Defaults to `"09-01"`.

- partial:

  What to do with a bin the record does not cover for its whole calendar
  span, which is what a record beginning or ending away from a bin
  boundary produces. `"keep"`, the default, returns it alongside the
  full bins; `"drop"` removes it. See Partial bins.

## Value

A numeric array of shape `[unit, bin, channel]`, with dimnames giving
the sorted unit identifiers, the ISO-8601 start of each bin, and the
statistic names. Attributes:

- `grain`: the grain name.

- `stats`: the statistic names in channel order.

- `year_start`: the boundary used.

- `bin_start`: the instant each bin begins on the calendar, which is
  earlier than the bin's first reading wherever the record does not
  reach the boundary.

- `bin_end`: the last reading instant assigned to each bin.

- `bin_n`: a `[unit, bin]` matrix of how many readings fell in each bin.

- `bin_partial`: a logical vector marking the bins the record does not
  cover for their whole calendar span.

Naming more than one grain returns a
[`timesift_set()`](https://gillescolling.com/timesift/reference/timesift_set.md)
of those arrays.

## Details

Seven statistics are available, and the distinction between an extreme
reading, an extreme day and a typical day is deliberate rather than
pedantic:

- `mean`: arithmetic mean of the readings in the bin.

- `min`, `max`: coldest and warmest single reading in the bin.

- `cold_day`, `warm_day`: coldest and warmest day, each day first
  reduced to its own mean. Defined for `"day"` and coarser.

- `mean_daily_min`, `mean_daily_max`: the bin's average daily minimum
  and average daily maximum, each day first reduced to its own extreme.
  Defined for `"day"` and coarser.

An extreme day is a state the unit was in; an extreme reading can be one
hour; an average daily extreme is the exposure a typical day of the bin
brought. On alpine soil temperature the day-level pair carries more
predictive signal than the bin mean, and by more the coarser the bin.

Whether a grain is a day or coarser is decided from the bins rather than
from the grain's name, so a supplied calendar that cuts inside a day is
refused for these four as well, naming the day it splits.

Nothing is standardised here. Scaling belongs to the fold it is computed
on, never to the representation, because computing it over all units
would leak held-out units into the input.

## Time zone

Bins follow the calendar the series is carried in, which is the `tzone`
attribute of `time`; a column with none is read as UTC. The zone is
resolved once, at the edge: below it the binning works in local time,
where a day is 86400 seconds whatever the night did, so a zone that
moves its clock at midnight has no midnight to lose. A bin start is a
local time, so reporting it back as an instant needs a rule: one the
clock skipped resolves to the instant the clock jumped to, one the clock
repeated to the first of the two. Instants are read at whole seconds.

## Bins that do not tile the record

Every unit must reach every bin, and consecutive bins must be one bin
apart on the grain's own calendar. A bin no unit reaches is never built,
so a month missing from the whole record would otherwise pass as four
adjacent monthly bins with one simply gone. Neither the `"native"`
grain, whose bin is the reading itself, nor a supplied calendar, which
declares its own bin lengths, is held to the second rule.
[`coverage()`](https://gillescolling.com/timesift/reference/coverage.md)
lays the same binning out as a count of readings per unit and bin, which
is where a refused record's gaps are read off.

## Partial bins

A bin is partial when the record does not cover its whole calendar span.
Which bins those are follows from where the record starts and stops
against the calendar, not from the grain alone: three years of hourly
readings from 1 September carry no partial month and no partial season
on a `"09-01"` boundary, but the same record carries a partial week at
each end, because 1 September is a Wednesday. A record from an arbitrary
deployment date carries one at each end of almost every grain.

A bin is partial if its start precedes the first reading of the record,
or if its calendar span runs past the last reading plus the record's own
sampling interval, taken as the smallest gap between consecutive
distinct reading instants. Only a bin at an end of the record can
satisfy either, because every unit is required to span every bin in
between. The verdict is returned as the `bin_partial` attribute
whichever way `partial` is set, so a kept partial bin is labelled rather
than silent.

Keeping partial bins is the default because dropping them discards the
record's ends: on a seasonal grain that is up to three months of
readings at each end. The cost of keeping them is that such a bin's mean
is taken over fewer readings and its extremes over fewer days, so
`cold_day` and `warm_day` there are drawn from a shorter draw and sit
closer to the bin mean than a full bin's would. `bin_n` gives the count
the bin was actually reduced from.

A caller-supplied binning declares its own bins, so the package cannot
know where the last one was meant to end and takes the record's end as
its end. Such a final bin is never reported partial; its leading bin is
judged as any other.

## Examples

``` r
t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24 * 40)
d <- data.frame(plot = rep(c("a", "b"), each = length(t)),
                t = rep(t, 2),
                temp = c(sin(seq_along(t) / 24), cos(seq_along(t) / 24)))
x <- grain_matrix(d, plot, t, temp, grain = "week",
                   stats = c("cold_day", "mean", "warm_day"))
dim(x)
dimnames(x)[[3]]

ladder <- grain_matrix(d, plot, t, temp, grain = c("day", "week"))
names(ladder)
```
