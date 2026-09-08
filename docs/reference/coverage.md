# Which units reach which bins

A representation needs every unit in every bin, and
[`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md)
refuses a record where one is missing rather than pad it. This is the
same binning laid out so the gaps can be read: how many readings each
unit has in each bin, over every bin the calendar tiles the record with
from the first bin any unit touches to the last. A logger that started
late, stopped early or lost a month is a row with zeros in it; a bin the
whole record skips is a column of zeros.

## Usage

``` r
coverage(data, id, time, grain = "day", year_start = "09-01")
```

## Arguments

- data:

  A data frame of readings in long form, one row per reading.

- id:

  Column identifying the unit carrying the sensor. A bare column name or
  a string.

- time:

  Column of reading instants, `POSIXct`. A bare column name or a string.

- grain:

  One of `"native"`, `"halfday"`, `"day"`, `"week"`, `"month"`,
  `"season"`, `"year"`. The four coarse grains follow the calendar, so a
  bin is a real week or month rather than a fixed block of hours. Naming
  several grains returns one representation per grain, a
  [`timesift_set()`](https://gillescolling.com/timesift/reference/timesift_set.md).
  A function is called on the reading instants and must return the
  `POSIXct` start of each reading's bin, which is how a calendar the
  package does not carry, such as astronomical seasons, is binned.

- year_start:

  `"MM-DD"` boundary of the hydrological year, used by `"season"` and
  `"year"`. Defaults to `"09-01"`.

## Value

An integer matrix of reading counts, one row per unit and one column per
bin, of class `timesift_coverage`, with the units and the ISO-8601 bin
starts as dimnames and the `grain` and the `bin_start` instants as
attributes.

## Details

What to do about a gap is the analyst's decision, and this is the table
it is made on: drop the units that do not span the record, cut the
record to the span every unit covers, or move to a grain the gap does
not reach. Nothing here fills a cell.

## Examples

``` r
t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24 * 40)
d <- data.frame(plot = rep(c("a", "b"), each = length(t)), t = rep(t, 2),
                temp = rnorm(2 * length(t)))
# Unit b loses the calendar week beginning Monday 6 September.
lost <- d$plot == "b" & d$t >= as.POSIXct("2021-09-06", tz = "UTC") &
  d$t < as.POSIXct("2021-09-13", tz = "UTC")
coverage(d[!lost, ], plot, t, grain = "week")
```
