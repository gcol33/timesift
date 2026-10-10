# Troubleshooting

`timesift` refuses a record it cannot represent faithfully, and says
why. This article collects the refusals and surprises met most often,
each reproduced on a small simulated record: the message as it appears,
what produced it, and the call that works.

Every example below reads the same two tables: 24 plots, each with a
logger reading soil temperature every hour for 60 days from 1 September
2021, and two species whose presence follows each plot’s own level of
warmth.

## Installation

``` r

# Install from CRAN
install.packages("timesift")

# Or install the development version from GitHub
# install.packages("pak")
pak::pak("gcol33/timesift")
```

``` bash
# Install from PyPI
pip install timesift

# with the torch encoders, the contrasts and the plots
pip install "timesift[torch,contrasts,plot]"

# Or install the development version from GitHub
pip install git+https://github.com/gcol33/timesift
```

``` r

hours <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24 * 60)
ids <- sprintf("p%02d", 1:24)
warmth <- rnorm(24)

series <- data.frame(
  plot = rep(ids, each = length(hours)),
  t = rep(hours, times = 24),
  temp = as.numeric(vapply(warmth, function(w) {
    w + 5 * sin(seq_along(hours) / 200) + rnorm(length(hours), sd = 3)
  }, numeric(length(hours))))
)
targets <- data.frame(plot = ids,
                      sp1 = rbinom(24, 1, plogis(2 * warmth)),
                      sp2 = rbinom(24, 1, plogis(-2 * warmth)))
colSums(targets[c("sp1", "sp2")])
#> sp1 sp2 
#>  14  10
```

## A unit has no readings in a bin

A logger that was stolen for a week, or one that started late, leaves a
hole in the record. Here plot `p03` loses the calendar week beginning
Monday 13 September.

``` r

lost <- series$plot == "p03" &
  series$t >= as.POSIXct("2021-09-13", tz = "UTC") &
  series$t < as.POSIXct("2021-09-20", tz = "UTC")
gappy <- series[!lost, ]

grain_matrix(gappy, plot, t, temp, grain = "week")
#> Error:
#> ! 1 (unit, bin) cell holds no readings, first: unit p03 at 2021-09-13T00:00:00. Every unit must span every bin; gaps are not padded. coverage() lists them.
```

Every unit must reach every bin of the representation. A cell with no
reading has no mean, and filling it, with an interpolation or the mean
of the other plots, would put a number in front of the model that no
logger recorded.
[`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
raises the same error, because it builds its arrays the same way.

[`coverage()`](https://gillescolling.com/timesift/reference/coverage.md)
lays out the same binning as a count of readings per unit and bin, so
the gaps can be found before deciding what to do about them.

``` r

cov <- coverage(gappy, plot, t, grain = "week")
cov
#> <timesift coverage> 24 units x 9 bins at the week grain
#> 1 empty (unit, bin) cell in 1 unit: p03
cov["p03", ]
#> 2021-08-30T00:00:00Z 2021-09-06T00:00:00Z 2021-09-13T00:00:00Z 
#>                  120                  168                    0 
#> 2021-09-20T00:00:00Z 2021-09-27T00:00:00Z 2021-10-04T00:00:00Z 
#>                  168                  168                  168 
#> 2021-10-11T00:00:00Z 2021-10-18T00:00:00Z 2021-10-25T00:00:00Z 
#>                  168                  168                  144
```

There are three ways forward, and the choice belongs to the analysis:
drop the units that do not span the record, cut the record to the span
every unit covers, or move to a grain the gap does not reach. Here a
month is coarse enough, since `p03` still has readings in September.

``` r

dim(grain_matrix(gappy[gappy$plot != "p03", ], plot, t, temp, grain = "week"))
#> [1] 23  9  1
dim(grain_matrix(gappy, plot, t, temp, grain = "month"))
#> [1] 24  2  1
```

A week missing from every unit at once is refused under a different
message, because the gap is then a bin that no unit reaches. Dropping
that bin silently would make the weeks either side of it look adjacent.

``` r

everyone <- series$t >= as.POSIXct("2021-09-13", tz = "UTC") &
  series$t < as.POSIXct("2021-09-20", tz = "UTC")
grain_matrix(series[!everyone, ], plot, t, temp, grain = "week")
#> Error:
#> ! the week bins are not contiguous: nothing falls in the one beginning 2021-09-13T00:00:00, between 2021-09-06T00:00:00 and 2021-09-20T00:00:00. Bins must tile the record; a gap is not closed up.
```

## The first and last bins are partial

1 September 2021 was a Wednesday, so the first weekly bin begins on
Monday 30 August, two days before the record does, and the last week
runs past its end.

``` r

x <- grain_matrix(series, plot, t, temp, grain = "week")
dimnames(x)[[2]]
#> [1] "2021-08-30T00:00:00Z" "2021-09-06T00:00:00Z" "2021-09-13T00:00:00Z"
#> [4] "2021-09-20T00:00:00Z" "2021-09-27T00:00:00Z" "2021-10-04T00:00:00Z"
#> [7] "2021-10-11T00:00:00Z" "2021-10-18T00:00:00Z" "2021-10-25T00:00:00Z"
attr(x, "bin_partial")
#> [1]  TRUE FALSE FALSE FALSE FALSE FALSE FALSE FALSE  TRUE
attr(x, "bin_n")[1, ]
#> 2021-08-30T00:00:00Z 2021-09-06T00:00:00Z 2021-09-13T00:00:00Z 
#>                  120                  168                  168 
#> 2021-09-20T00:00:00Z 2021-09-27T00:00:00Z 2021-10-04T00:00:00Z 
#>                  168                  168                  168 
#> 2021-10-11T00:00:00Z 2021-10-18T00:00:00Z 2021-10-25T00:00:00Z 
#>                  168                  168                  144
```

Nothing is wrong: the calendar tiles the record, and a record that does
not start on a Monday has a partial week at its start. `bin_partial`
marks those bins and `bin_n` counts the readings each bin was reduced
from, 120 in the first week against 168 in a full one. A partial bin’s
mean is taken over fewer readings, and its `cold_day` and `warm_day` are
drawn from fewer days, so they sit closer to the mean than a full bin’s
would.

`partial = "drop"` removes them.

``` r

dim(grain_matrix(series, plot, t, temp, grain = "week", partial = "drop"))
#> [1] 24  7  1
```

On a grain coarser than the record, dropping can leave nothing, and that
is an error.

``` r

grain_matrix(series, plot, t, temp, grain = "season", partial = "drop")
#> Error:
#> ! dropping the partial bins leaves no bin: the record covers no whole season. Use `partial = "keep"` or a finer grain.
```

[`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
keeps partial bins, because dropping them throws away the ends of the
record: on a seasonal grain that can be three months at each end. To fit
on whole bins alone, cut the series to the bin boundaries before the
call. Monday 6 September to Sunday 24 October is seven whole weeks.

``` r

whole <- series[series$t >= as.POSIXct("2021-09-06", tz = "UTC") &
                  series$t < as.POSIXct("2021-10-25", tz = "UTC"), ]
attr(grain_matrix(whole, plot, t, temp, grain = "week"), "bin_partial")
#> [1] FALSE FALSE FALSE FALSE FALSE FALSE FALSE
```

## The time column is not POSIXct, or sits in an unexpected zone

A time column read from a CSV file arrives as text, and one built with
[`as.Date()`](https://rdrr.io/r/base/as.Date.html) has lost its hours.
Both are refused.

``` r

as_text <- transform(series, t = format(t))
grain_matrix(as_text, plot, t, temp, grain = "day")
#> Error:
#> ! `t` must be POSIXct, not character.

as_date <- transform(series, t = as.Date(t))
grain_matrix(as_date, plot, t, temp, grain = "day")
#> Error:
#> ! `t` must be POSIXct, not Date.
```

The fix is to parse the column with its zone stated, for example
`as.POSIXct(x, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")`. A zone name
the zone database does not know is refused too, since reading a clock in
it would fall back to UTC with only a warning.

``` r

odd <- series
attr(odd$t, "tzone") <- "CET+1"
grain_matrix(odd, plot, t, temp, grain = "day")
#> Error:
#> ! `t` is carried in the time zone "CET+1", which the zone database does not know. Name one of `OlsonNames()`.
```

The surprise that raises no error is the zone itself. Bins follow the
calendar of the `tzone` attribute, so the same instants carried in
`"Europe/Vienna"` are cut into days at Vienna’s midnight, which in
September is 22:00 UTC.

``` r

vienna <- series
attr(vienna$t, "tzone") <- "Europe/Vienna"
xv <- grain_matrix(vienna, plot, t, temp, grain = "day")
dimnames(xv)[[2]][1:2]
#> [1] "2021-08-31T22:00:00Z" "2021-09-01T22:00:00Z"
attr(xv, "bin_partial")[1:2]
#> [1]  TRUE FALSE
attr(xv, "bin_n")[1, 1:2]
#> 2021-08-31T22:00:00Z 2021-09-01T22:00:00Z 
#>                   22                   24
```

The first Vienna day began two hours before the first reading, so it is
partial and holds 22 readings. In a zone that keeps daylight saving time
the day the clock goes forward holds 23 hours and the day it goes back
25. A record kept on a fixed offset bins into days of 24 hours once its
`tzone` names that offset, such as `"Etc/GMT-1"` for Central European
standard time. The zone is a decision about what a day means for the
question, and it is worth setting explicitly.

``` r

fixed <- series
attr(fixed$t, "tzone") <- "Etc/GMT-1"
dimnames(grain_matrix(fixed, plot, t, temp, grain = "day"))[[2]][1:2]
#> [1] "2021-08-31T23:00:00Z" "2021-09-01T23:00:00Z"
```

On that offset every day begins at 23:00 UTC, in summer and in winter
alike.

## Missing readings and repeated instants

A reading that is `NA` and an instant recorded twice for the same unit
are both refused where the record is read.

``` r

holed <- series
holed$temp[5] <- NA
grain_matrix(holed, plot, t, temp, grain = "day")
#> Error:
#> ! 1 reading is not a finite number, first: unit p01 at 2021-09-01T04:00:00. Fill or drop it before building a representation.

grain_matrix(rbind(series, series[1, ]), plot, t, temp, grain = "day")
#> Error:
#> ! 1 duplicated (unit, time) pair, first: p01 at 2021-09-01T00:00:00Z.
```

A logger that wrote the same hour twice usually did so around a download
or a clock change. Removing the repeated rows, or keeping one per
instant on a rule that suits the logger, comes before any
representation.

## Targets and series carry different units

`targets` and `series` are linked by the column `id` names. A target
whose unit has no record is refused, naming the units.

``` r

extra <- rbind(targets, data.frame(plot = c("p98", "p99"), sp1 = 1, sp2 = 0))
timesift(extra, series, y = starts_with("sp"), id = plot, time = t, verbose = FALSE)
#> Error:
#> ! 2 targets name a unit `series` does not carry: p98, p99.
```

The other direction is not an error. A unit in `series` that no target
names is read by nobody, so it does not enter the fit.

``` r

fewer <- targets[-(1:2), ]
fit_fewer <- timesift(fewer, series, y = starts_with("sp"), id = plot, time = t,
                      sift = grains("week"), resampling = cv(v = 3), n_inner = NULL,
                      ensemble = FALSE, verbose = FALSE)
dim(fit_fewer$y)
#> [1] 22  2
```

The fit holds 22 targets, although the series carries 24 plots. When a
plot seems to be missing from the results, `rownames(fit$y)` lists the
targets that were fitted.

Two rows of `targets` with one identifier are refused unless
`target_time` says where in time each row sits, because otherwise both
rows would read the same record.

``` r

timesift(rbind(targets, targets[1, ]), series, y = starts_with("sp"), id = plot, time = t,
         verbose = FALSE)
#> Error:
#> ! `targets` holds more than one row for 1 identifier: p01. Give `target_time` to say where in time each row sits.
```

## A learner cannot read the representation

A learner declares what it reads: a tabular block of features, or a
sequence whose bins are ordered in time.
[`elasticnet()`](https://gillescolling.com/timesift/reference/elasticnet.md)
is tabular, and
[`native()`](https://gillescolling.com/timesift/reference/native.md) is
the record unreduced, one column per hourly reading. Pinned to it
through `data =`, it is refused before any array is built.

``` r

timesift(targets, series, y = starts_with("sp"), id = plot, time = t,
         learners = elasticnet(data = native()), verbose = FALSE)
#> Error:
#> ! `elasticnet()` reads a tabular representation; `native()` gives it one column per reading. Use `grain()`, `multigrain()` or `lookback()`.
```

When the same pair arises from a `sift` expansion, the run goes ahead
without it, says so once, and records why in `fit$candidates`.

``` r

fit_skip <- timesift(targets, series, y = starts_with("sp"), id = plot, time = t,
                     sift = c(native(), grain("week")), resampling = cv(v = 3),
                     n_inner = NULL, ensemble = FALSE)
#> skipping 1 candidate no learner can read: elasticnet / native
#> fold 1 of 3, 0 s
#> fold 2 of 3, 0 s
#> fold 3 of 3, 0 s
#> refitting every candidate on all 24 targets
fit_skip$candidates[c("candidate", "status", "note")]
#>             candidate         status
#> 1 elasticnet / native not applicable
#> 2   elasticnet / week         fitted
#>                                                                                                                                        note
#> 1 `elasticnet()` reads a tabular representation; `native()` gives it one column per reading. Use `grain()`, `multigrain()` or `lookback()`.
#> 2                                                                                                                                      <NA>
```

If no pair in the run can be fitted, the run stops and lists every
reason.

``` r

timesift(targets, series, y = starts_with("sp"), id = plot, time = t, sift = native(),
         verbose = FALSE)
#> Error:
#> ! no learner can read any representation in the sift:
#>   `elasticnet()` reads a tabular representation; `native()` gives it one column per reading. Use `grain()`, `multigrain()` or `lookback()`.
```

The sequence learners,
[`cnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
and
[`rescnn()`](https://gillescolling.com/timesift/reference/torch_learners.md),
are refused the opposite case: a
[`multigrain()`](https://gillescolling.com/timesift/reference/native.md)
is a flat block of features with no order to convolve along, and a grain
the record holds only one bin of is a sequence of length one. The first
is refused before the record is read, the second once the array says how
many bins it has. Neither needs torch to be refused.

``` r

timesift(targets, series, y = starts_with("sp"), id = plot, time = t,
         learners = cnn(data = multigrain(c("week", "month"))), verbose = FALSE)
#> Error:
#> ! `cnn()` reads a sequence; `multigrain()` gives one row of features.

timesift(targets, series, y = starts_with("sp"), id = plot, time = t,
         learners = cnn(data = grain("year")), verbose = FALSE)
#> Error:
#> ! `cnn()` reads a sequence; `grain("year")` gives one row of features.
```

## A learner’s `data =` names a representation outside the sift

A learner given `data =` is fitted on that representation alone, and the
run builds it whether or not the `sift` holds it. Here the forest reads
months while the elastic net runs across the sift.

``` r

fit_pin <- timesift(targets, series, y = starts_with("sp"), id = plot, time = t,
                    learners = c(elasticnet(), forest(data = grain("month"))),
                    sift = grains("week"), resampling = cv(v = 3), n_inner = NULL,
                    ensemble = FALSE, verbose = FALSE)
fit_pin$candidates[c("candidate", "representation", "bins")]
#>           candidate representation bins
#> 1 elasticnet / week           week    9
#> 2    forest / month          month    2
```

The error comes when the pinned representation and one in the sift would
be reported under the same name but differ in their settings, such as
the weekly mean in the sift and the weekly extremes for the forest.

``` r

timesift(targets, series, y = starts_with("sp"), id = plot, time = t,
         learners = c(elasticnet(),
                      forest(data = grain("week", stats = c("cold_day", "warm_day")))),
         sift = grains("week"), verbose = FALSE)
#> Error:
#> ! two representations are reported under the name "week": the one the forest learner is pinned to and the one in the sift. Name the sift to tell them apart.
```

Naming the sift gives its representation a name of its own, and the two
are then reported apart.

``` r

fit_named <- timesift(targets, series, y = starts_with("sp"), id = plot, time = t,
                      learners = c(elasticnet(),
                                   forest(data = grain("week", stats = c("cold_day", "warm_day")))),
                      sift = list(week_mean = grain("week")), resampling = cv(v = 3),
                      n_inner = NULL, ensemble = FALSE, verbose = FALSE)
fit_named$candidates[c("candidate", "channels")]
#>                candidate channels
#> 1 elasticnet / week_mean        1
#> 2          forest / week        2
```

## Day-level statistics at a sub-daily grain

`cold_day`, `warm_day`, `mean_daily_min` and `mean_daily_max` first
reduce each day to one number, so they need bins of a day or longer. The
refusal comes from the constructor, before any data are read.

``` r

grain("halfday", stats = c("mean", "cold_day"))
#> Error:
#> ! `halfday` bins are shorter than a day, so cold_day is not defined there. Use a grain of `day` or coarser.
```

The coldest single reading of a half-day bin is `min`, and the coldest
day is defined from the `"day"` grain upward.

``` r

grain("halfday", stats = c("mean", "min"))
#> <timesift representation> halfday 
#> kind    : grain (a sequence) 
#> grain   : halfday 
#> stats   : mean, min
grain("week", stats = c("cold_day", "mean", "warm_day"))
#> <timesift representation> week 
#> kind    : grain (a sequence) 
#> grain   : week 
#> stats   : cold_day, mean, warm_day
```

## A response has too few presences to score

A presence-absence score needs both classes among the held-out units,
and a model per response needs both among the units it is fitted on. A
cell, one response in one fold, where either side of the split holds a
single class carries no score. Below, `sp3` is present in two plots and
`sp4` in none.

``` r

rare <- targets
rare$sp3 <- c(1, 1, rep(0, 22))
rare$sp4 <- 0
fit_rare <- timesift(rare, series, y = starts_with("sp"), id = plot, time = t,
                     sift = grains("week"), resampling = cv(v = 4), n_inner = NULL,
                     ensemble = FALSE, verbose = FALSE)
fit_rare$cells
#> <timesift cells> 16 cells over 4 variables 
#> scorable: 8 (50.0%); variables with at least one scorable fold: 2 of 4
cells <- as.data.frame(fit_rare$cells)
cells[cells$variable == "sp3", ]
#>    variable fold n_occ pres_train abs_train pres_test abs_test scorable
#> 9       sp3    1     2          2        16         0        6    FALSE
#> 10      sp3    2     2          2        16         0        6    FALSE
#> 11      sp3    3     2          0        18         2        4    FALSE
#> 12      sp3    4     2          2        16         0        6    FALSE
```

In folds 1, 2 and 4 both presences of `sp3` sit on the training side,
leaving no presence to score, and in fold 3 both are held out, leaving
none to fit on. The run does not stop for this. `fit$cells` is the mask,
computed from the response and the fold map alone, and every candidate
is scored on the same scorable cells, so `sp3` and `sp4` contribute
nothing to any mean.
[`scorable_cells()`](https://gillescolling.com/timesift/reference/scorable_cells.md)
gives the same table before a fit, from a response matrix and a fold
map.

``` r

y <- as.matrix(rare[c("sp1", "sp2", "sp3", "sp4")])
rownames(y) <- rare$plot
scorable_cells(y, fold_map(y, v = 4))
#> <timesift cells> 16 cells over 4 variables 
#> scorable: 8 (50.0%); variables with at least one scorable fold: 2 of 4
```

Fewer folds give a rare response more chance of landing on both sides of
each split. Only when no cell at all is scorable does the run stop, as
it does for a response absent everywhere.

``` r

absent <- data.frame(plot = ids, sp0 = 0)
timesift(absent, series, y = sp0, id = plot, time = t, verbose = FALSE)
#> Error:
#> ! no (response, fold) cell is scorable under this fold map, so no candidate can be scored: every cell needs both classes on each side of its split. See scorable_cells() for the counts; a rarer response needs fewer folds, or a response present somewhere.
```

The response itself must be 0 and 1, with no missing value.

``` r

counted <- targets
counted$sp1[1] <- 2
timesift(counted, series, y = starts_with("sp"), id = plot, time = t, verbose = FALSE)
#> Error:
#> ! a presence-absence response must be 0/1 or logical.

unknown <- targets
unknown$sp1[1] <- NA
timesift(unknown, series, y = starts_with("sp"), id = plot, time = t, verbose = FALSE)
#> Error:
#> ! the response holds missing values. Fill or drop them before fitting.
```

## A torch learner without torch

[`mlp()`](https://gillescolling.com/timesift/reference/torch_learners.md),
[`cnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
and
[`rescnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
declare that they need torch. Constructing one does not load it, so a
learner can be written into a script on a machine without torch; the
check runs when the learner is fitted, and stops with

    the cnn learner needs torch. Install it with install.packages("torch").

naming the learner. That message is quoted from the source here, since
the machine this article was built on has torch installed. There is no
fallback to a learner that runs without torch, because two code paths
would drift apart. The torch package also installs its own libraries the
first time it is used, which is torch’s step and not `timesift`’s.

## `predict()` on a new record

A calendar grain reads a record by the instants of its bins. A fit made
on weeks from 30 August to 25 October expects a new record over the same
weeks, and a shorter or later record is refused.

``` r

fit <- timesift(targets, series, y = starts_with("sp"), id = plot, time = t,
                sift = grains("week", "month"), resampling = cv(v = 3), verbose = FALSE)
```

``` r

shorter <- series[series$t < as.POSIXct("2021-10-15", tz = "UTC"), ]
predict(fit, targets, shorter)
#> Error:
#> ! the representation predicted on has different channels or bins from the fitted one: 7 bins here and 9 bins in the fit. A calendar grain is read by its bins' instants, so a fit predicts a record over the same period; a lookback reads a span relative to each target and predicts any period.

next_year <- transform(series, t = t + 365 * 86400)
predict(fit, targets, next_year)
#> Error:
#> ! the representation predicted on has different channels or bins from the fitted one: bin 1 is 2022-08-29T00:00:00Z here and 2021-08-30T00:00:00Z in the fit. A calendar grain is read by its bins' instants, so a fit predicts a record over the same period; a lookback reads a span relative to each target and predicts any period.

predict(fit, targets, gappy)
#> Error:
#> ! 1 (unit, bin) cell holds no readings, first: unit p03 at 2021-09-13T00:00:00. Every unit must span every bin; gaps are not padded. coverage() lists them.
```

The first two are the same mismatch: the bins of the new record are not
those the fit was made on. The third is the coverage refusal again,
raised while building the new array. A record over the same period
predicts, for any subset of the units.

``` r

predict(fit, targets[1:3, ], series[series$plot %in% ids[1:3], ])
#>           sp1       sp2
#> p01 0.1745044 0.6128003
#> p02 0.5458671 0.4945805
#> p03 0.1465358 0.6298695
```

To predict another period, the representation has to be anchored on each
target’s own instant rather than on the calendar. That is a
[`lookback()`](https://gillescolling.com/timesift/reference/native.md),
with `target_time` naming the column of `targets` holding the instant.

``` r

anchored <- transform(targets, at = as.POSIXct("2021-10-30", tz = "UTC"))
fit_lb <- timesift(anchored, series, y = starts_with("sp"), id = plot, time = t,
                   target_time = at, sift = lookbacks("28 days", "56 days"),
                   resampling = cv(v = 3), verbose = FALSE)

later <- transform(anchored, at = at + 365 * 86400)
predict(fit_lb, later[1:3, ], next_year[next_year$plot %in% ids[1:3], ])
#>         sp1       sp2
#> 1 0.1886727 0.6033119
#> 2 0.5668084 0.4835910
#> 3 0.1495701 0.6241430
```

## Runs take longer than expected

The cost of a run is a count of fits. With `n_inner` set, every
candidate is cross-validated inside every outer training fold and then
refitted on that fold once, which is
`v_outer * (n_inner + 1) * candidates` fits, plus one refit per
candidate on every target. With `n_inner = NULL` the inner search is
skipped and the count drops to `v_outer * candidates`, at the price of
the held-out `estimate`.

``` r

v_outer <- 3
candidates <- 2
c(nested = v_outer * (5 + 1) * candidates + candidates,
  outer_only = v_outer * candidates + candidates)
#>     nested outer_only 
#>         38          8

nested <- system.time(timesift(targets, series, y = starts_with("sp"), id = plot, time = t,
                               sift = grains("week", "month"), resampling = cv(v = 3),
                               verbose = FALSE))[["elapsed"]]
outer_only <- system.time(timesift(targets, series, y = starts_with("sp"), id = plot, time = t,
                                   sift = grains("week", "month"), resampling = cv(v = 3),
                                   n_inner = NULL, verbose = FALSE))[["elapsed"]]
c(nested = nested, outer_only = outer_only)
#>     nested outer_only 
#>       0.98       0.25
```

On this record the nested run took 0.98 seconds and the run on the outer
folds alone 0.25.

Each fit in turn costs what its array costs. A calendar grain has as
many columns as bins times statistics, and
[`native()`](https://gillescolling.com/timesift/reference/native.md) has
one bin per reading, so its size follows the record directly.

``` r

xn <- grain_matrix(series, plot, t, temp, grain = "native")
dim(xn)
#> [1]   24 1440    1
format(object.size(xn), units = "MB")
#> [1] "0.7 Mb"

# Three years of hourly readings on 894 plots, one channel, in megabytes
894 * (3 * 365 + 1) * 24 * 8 / 2^20
#> [1] 179.4111
```

The weekly grain of the same record holds 9 bins against the 1440 of the
native array, and the Schrankogel record of the package’s origin, 894
plots read hourly for three years, would be about 179 MB for a single
channel. A run on a long hourly record is quickest to iterate on at a
few coarse grains with `n_inner = NULL`, reaching for
[`native()`](https://gillescolling.com/timesift/reference/native.md) and
the inner search once the comparison is settled.
[`elasticnet()`](https://gillescolling.com/timesift/reference/elasticnet.md)
and [`forest()`](https://gillescolling.com/timesift/reference/forest.md)
take `threads`, and what they return does not depend on it.

## Two runs give different numbers

The random draws in a run come from seeds the call carries:
`cv(seed = )` for the outer folds, the `seed` of
[`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
for the inner splits, and the `seed` of each learner that draws, such as
[`elasticnet()`](https://gillescolling.com/timesift/reference/elasticnet.md)’s
inner cross-validation and
[`forest()`](https://gillescolling.com/timesift/reference/forest.md)’s
bootstrap. With the same seeds the same call gives the same numbers.

``` r

run <- function(seed) {
  timesift(targets, series, y = starts_with("sp"), id = plot, time = t,
           sift = grains("week", "month"), resampling = cv(v = 3, seed = seed),
           n_inner = NULL, ensemble = FALSE, verbose = FALSE)
}
a <- run(1)
b <- run(1)
c2 <- run(2)
identical(a$scores, b$scores)
#> [1] TRUE
identical(a$scores, c2$scores)
#> [1] FALSE
```

The fold map draws under its own seed and puts the session’s random
number stream back as it found it, so a call does not shift the draws
that follow it.

``` r

y2 <- as.matrix(targets[c("sp1", "sp2")])
rownames(y2) <- targets$plot
set.seed(3); first <- runif(1)
set.seed(3); invisible(fold_map(y2, v = 3)); second <- runif(1)
identical(first, second)
#> [1] TRUE
```

When a rerun differs, the usual cause is a changed input: a row order, a
time zone or a filter upstream. The digest of the array is a quick test
of whether two runs read the same representation.

``` r

digest_array(grain_matrix(series, plot, t, temp, grain = "week"))
#> [1] "61296ebef3384a8fde101fc1540d2ae8"
digest_array(grain_matrix(series[sample(nrow(series)), ], plot, t, temp, grain = "week"))
#> [1] "61296ebef3384a8fde101fc1540d2ae8"
```

The two digests agree because the rows of a representation are sorted by
unit, whatever order the series arrives in. The neural learners are
seeded through `train_control(seed = )`, but a network trained by torch
is not promised to give the same weights to the last bit on another
machine or device, and the package does not test that it does.

## R and Python give different numbers

The R and the Python package call one compiled core, and the contract
between them is written down in [The representation
contract](https://gillescolling.com/timesift/articles/contract.md).
[`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md)
and
[`lookback_matrix()`](https://gillescolling.com/timesift/reference/lookback_matrix.md)
must give the same array in both, which is checked as a digest against
fixtures shipped with the package. The check can be run on any
installation.

``` r

fx <- system.file("spec", "fixtures", package = "timesift")
rec <- read.csv(file.path(fx, "series.csv"))
rec$time <- as.POSIXct(rec$time, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
expected <- read.csv(file.path(fx, "digests.csv"))
row <- expected[expected$series == "aligned" & expected$grain == "week" & expected$tz == "UTC" &
                  expected$year_start == "09-01" & expected$partial == "keep" &
                  expected$stat == "mean", ]
got <- grain_matrix(rec, id, time, value, grain = "week", year_start = "09-01")
c(expected = row$digest, got = digest_array(got))
#>                           expected                                got 
#> "23015e94770e53ec80212f6e756f96bf" "23015e94770e53ec80212f6e756f96bf"
```

The elastic net is pinned as a distance rather than a digest: two
coordinate descents stop at the same point to the tolerance they are run
at and no closer, so its coefficients agree to that tolerance. Neural
networks trained by R’s torch and Python’s are not required to agree
beyond what two seeds of one network would. A difference in a
representation digest, or in a penalised fit beyond its tolerance, is a
bug in whichever language changed, and worth reporting.

The more common source of disagreement is the input. R sorts units in C
collation, as NumPy does, but a time column parsed in different zones in
the two languages bins on different calendars, and a `year_start` set on
one side only moves every season and year.

## Questions that come up

### Does `timesift` standardise the readings?

Not in the representation.
[`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md)
returns the readings in their own units, because a scale computed over
all units would carry the held-out units into the input. The neural
encoders standardise each channel over the units each network is fitted
on.

### Can units cover different periods?

Not on a calendar grain, where every unit must reach every bin. A target
with an instant of its own, such as a survey date, is read with a
[`lookback()`](https://gillescolling.com/timesift/reference/native.md)
and `target_time`, and then only the span before each target has to be
covered.

### Are columns of `targets` used as predictors automatically?

No. A column that is neither the response, the identifier nor
`target_time` is ignored unless `static` names it.

### Which number goes in a paper?

`fit$estimate`, the held-out score of the selection and the stack. The
highest candidate mean in `fit$candidates` was picked on the same folds
it is scored on.

### Why is a response missing from the scores?

Its cells are not scorable under the fold map, and `fit$cells` says
which and why.

### Which candidates were skipped?

Those with `status` “not applicable” in `fit$candidates`, with the
reason in `note`.

### Can a fit be saved and used in another session?

Yes. A fit is a plain list, and a fitted encoder holds its weights as
arrays, so [`saveRDS()`](https://rdrr.io/r/base/readRDS.html) and
[`readRDS()`](https://rdrr.io/r/base/readRDS.html) round trip it and it
predicts on another machine. A learner registered with
[`register_learner()`](https://gillescolling.com/timesift/reference/register_learner.md)
has to be registered again before predicting.

### What does `grains("auto")` choose?

Every named grain the record gives at least two bins.

## Reporting a bug

A refusal that names the wrong cause, a digest that disagrees with the
fixtures, or a result that changes between two runs with the same seeds
is worth an issue at <https://github.com/gcol33/timesift/issues>. The
most useful report holds a small record that reproduces it, built as the
tables in this article are, the output of `packageVersion("timesift")`
and [`sessionInfo()`](https://rdrr.io/r/utils/sessionInfo.html), and the
message as printed.
