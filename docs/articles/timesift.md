# Choosing how a record is read

A sensor records every hour for years. Before any model sees it, the
record is reduced: to monthly means, to growing-degree-days, to whatever
the analyst settled on once. `timesift` makes that reduction an
argument, fits at each setting of it, and shows how much of the record
the prediction actually needed.

This vignette runs the whole path on a small simulated record, from two
tables to a scored comparison and the prediction that follows from it.

## Two tables

`targets` is one row per thing to predict. `series` is the long,
time-stamped record belonging to those rows, and the two are linked by
an identifier both carry.

``` r

hours <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24 * 120)
ids <- sprintf("p%03d", 1:60)
warmth <- rnorm(60)

series <- data.frame(
  plot = rep(ids, each = length(hours)),
  t = rep(hours, times = 60),
  temp = as.numeric(vapply(warmth, function(w) {
    1.5 * w + 6 * sin(seq_along(hours) / (24 * 40)) + rnorm(length(hours), sd = 12)
  }, numeric(length(hours))))
)

sign <- rep(c(1, -1), length.out = 6)
targets <- data.frame(plot = ids, elevation = 2000 + 300 * rnorm(60))
targets[paste0("sp", 1:6)] <- lapply(sign, function(s) rbinom(60, 1, plogis(3 * s * warmth)))
str(targets[1:4], give.attr = FALSE)
#> 'data.frame':    60 obs. of  4 variables:
#>  $ plot     : chr  "p001" "p002" "p003" "p004" ...
#>  $ elevation: num  2644 1978 2365 2160 1899 ...
#>  $ sp1      : int  1 0 0 1 0 0 0 1 1 0 ...
#>  $ sp2      : int  1 0 1 0 1 1 0 0 0 0 ...
```

Each plot has a level of its own, and that level is buried in
hour-to-hour noise an order of magnitude larger.

## One call

``` r

fit <- timesift(
  targets, series,
  y = starts_with("sp"),
  id = plot,
  time = t,
  static = elevation,
  sift = grains("day", "week", "month"),
  resampling = cv(v = 5),
  verbose = FALSE
)
fit
#> timesift  60 targets, 6 responses, 5-fold random CV, tss
#> 
#> candidate                    mean    won  responses
#> elasticnet / day            0.721      1  separate
#> elasticnet / week           0.758      2  separate
#> elasticnet / month          0.787      3  separate
#> ensemble                    0.799      -
#> 
#> weights  elasticnet / month 0.86   elasticnet / day 0.10   elasticnet / week 0.04
```

Every representation named in `sift` was built, and `models` defaulting
to `list(elasticnet())`, a penalised logistic regression was fitted on
each of them over the same five folds and the out-of-fold predictions
were stacked. `won` is how many responses a candidate scored highest on,
and `responses` says whether one fitted model covered them all or one
was fitted per response.

A column of `targets` reaches the model only where `static` names it:
`elevation` is a predictor here because it was asked for, and a column
of notes sitting beside it would not be.

``` r

fit$candidates[c("candidate", "representation", "bins", "channels", "status")]
#>            candidate representation bins channels status
#> 1   elasticnet / day            day  120        2 fitted
#> 2  elasticnet / week           week   18        2 fitted
#> 3 elasticnet / month          month    4        2 fitted
```

Two channels at every grain: the binned temperature, and elevation held
constant across the bins.

``` r

plot(fit)
```

![Mean true skill statistic against representation, with the level the
combination reached drawn across
it.](timesift_files/figure-html/plot-run-1.svg)

[`predict()`](https://rdrr.io/r/stats/predict.html) rebuilds each
member’s representation for new rows from the settings its own arm was
built with, and combines them through the ensemble.

``` r

p <- predict(fit, targets, series)
round(p[1:3, 1:4], 3)
#>        sp1   sp2   sp3   sp4
#> p001 0.165 0.920 0.244 0.851
#> p002 0.544 0.582 0.568 0.700
#> p003 0.031 0.943 0.167 0.932
```

## Representations

A representation carries the settings and nothing else, so the same
object describes an arm before any record has been read and rebuilds
itself for new targets afterwards.

``` r

native()
#> <timesift representation> native 
#> kind    : grain (a sequence) 
#> grain   : native 
#> stats   : mean
grain("week", stats = c("cold_day", "mean", "warm_day"))
#> <timesift representation> week 
#> kind    : grain (a sequence) 
#> grain   : week 
#> stats   : cold_day, mean, warm_day
multigrain(c("week", "month"))
#> <timesift representation> multigrain 
#> kind    : multigrain (a block of features) 
#> grains  : week, month 
#> stats   : mean
lookback("30 days", bins = 3)
#> <timesift representation> 30 days x3 
#> kind    : lookback (a sequence) 
#> span    : 30 days in 3 bins ending 0 days before the target
#> stats   : mean
```

[`grains()`](https://gillescolling.com/timesift/reference/grains.md) and
[`lookbacks()`](https://gillescolling.com/timesift/reference/grains.md)
are sets of them, and `grains("auto")` reads off the record every named
grain it gives at least two bins.
[`multigrain()`](https://gillescolling.com/timesift/reference/native.md)
flattens several grains side by side into one block of features;
[`lookback()`](https://gillescolling.com/timesift/reference/native.md)
is a fixed span ending at each target’s own instant, which is what a
plot carrying several targets through time needs, given as
`target_time`.

A learner runs across the whole set, or at one representation it is
pinned to:

``` r

elasticnet(data = grain("month"))
#> <timesift learner> elasticnet 
#> reads   : tabular ; one model per response: yes, separate 
#> data    : month 
#> settings: alpha = 0.5, n_inner = 5, squares = TRUE, s = lambda.min, weight_positives = TRUE, seed = 1 
#> needs   : glmnet
```

## What a learner may be handed

A learner declares whether the bins reach it as a block of predictors or
as a sequence whose order in time is what it reads. A tabular learner
given
[`native()`](https://gillescolling.com/timesift/reference/native.md) is
refused before any record is touched, since building the array it would
have been handed is the expensive half of the call; a sequence learner
is refused a representation that turns out to hold one bin, once the
array says how many it has.

``` r

timesift(targets, series, y = starts_with("sp"), id = plot, time = t,
         models = elasticnet(), sift = native())
#> Error:
#> ! no learner can read any representation in the sift:
#>   `elasticnet()` reads a tabular representation; `native()` gives it one column per reading. Use `grain()`, `multigrain()` or `lookback()`.
```

Inside a set such a pair is skipped and listed as `not applicable` by
[`summary()`](https://rdrr.io/r/base/summary.html); named through a
learner’s own `data =` it is an error, because a representation named by
hand is a decision.

## The split, and the cells a score is defined on

One fold map is read by everything that scores, so every candidate is
fitted and scored on identical splits.
[`cv()`](https://gillescolling.com/timesift/reference/cv.md) deals units
into folds balanced on a stratifying value and
[`grouped_cv()`](https://gillescolling.com/timesift/reference/cv.md)
keeps every target sharing a group value on one side of each split;
`resampling` also takes a fold vector or a
[`fold_map()`](https://gillescolling.com/timesift/reference/fold_map.md)
result directly.

``` r

fit$folds
#> <timesift folds> 60 units in 5 folds 
#> fold
#>  1  2  3  4  5 
#> 14 11 12 12 11
```

A per-response score needs both classes among the held-out units, and a
per-response model needs both classes among the units it was fitted on.
The mask says which cells those are, from the response and the fold map
alone.

``` r

fit$cells
#> <timesift cells> 30 cells over 6 variables 
#> scorable: 30 (100.0%); variables with at least one scorable fold: 6 of 6
```

Because it involves no model, every candidate is restricted to the same
cells: their means share one denominator and every paired difference
runs on matched cells.

## Learners, and how they are trained

[`elasticnet()`](https://gillescolling.com/timesift/reference/elasticnet.md)
and
[`stepwise()`](https://gillescolling.com/timesift/reference/stepwise.md)
read a block of features,
[`forest()`](https://gillescolling.com/timesift/reference/forest.md)
grows a probability forest over one, and the `torch` encoders
[`mlp()`](https://gillescolling.com/timesift/reference/torch_learners.md),
[`cnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
and
[`rescnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
read a sequence with a joint multi-label head, so every response is
predicted together from a shared embedding. Pooling strength across
responses is what makes the rarer ones learnable at these sample sizes.

Architecture belongs to the constructor and training belongs to
[`train_control()`](https://gillescolling.com/timesift/reference/train_control.md),
which every neural learner of a run reads. A learner given a control of
its own overrides the run’s on the settings it names and takes the rest
from it.

``` r

train_control(epochs = 200L, device = "cpu")
#> <timesift control>
#>   epochs          200
#>   batch_size      64   (default)
#>   learning_rate   0.001   (default)
#>   weight_decay    1e-04   (default)
#>   early_stopping  10   (default)
#>   val_frac        0.15   (default)
#>   device          cpu
#>   seed            1   (default)
#>   pos_weight_cap  50   (default)
#>   swa             FALSE   (default)
#>   swa_start       0.7   (default)
cnn(channels = c(16L, 32L), epochs = 300L)
#> <timesift learner> cnn 
#> reads   : sequence ; one model per response: no, joint 
#> data    : every representation of the run 
#> settings: channels = 16/32, kernel = 7, dropout = 0.3 
#> training: epochs = 300 
#> needs   : torch
```

A learner of your own is a fit and a predict pair, and it goes through
the same folds, the same cells and the same scoring as the ones that
ship.

``` r

flat <- function(x) matrix(as.numeric(x), nrow = dim(x)[1])

nearest_neighbour <- learner(
  "1nn",
  fit = function(x, y, ...) list(x = flat(x), y = y),
  predict = function(model, x) {
    d <- as.matrix(dist(rbind(flat(x), model$x)))[seq_len(dim(x)[1]), -seq_len(dim(x)[1])]
    model$y[apply(d, 1, which.min), , drop = FALSE]
  }
)

both <- timesift(targets, series, y = starts_with("sp"), id = plot, time = t,
                 models = c(elasticnet(), nearest_neighbour),
                 sift = grains("week", "month"), resampling = cv(v = 5), verbose = FALSE)
summary(both)
#> timesift  60 targets, 6 responses, 5-fold random CV, tss
#> 
#> candidate                    mean    won  responses
#> 1nn / week                  0.432      0  separate
#> 1nn / month                 0.489      0  separate
#> elasticnet / week           0.758      1  separate
#> elasticnet / month          0.802      5  separate
#> ensemble                    0.797      -
#> 
#> weights  elasticnet / month 0.89   elasticnet / week 0.11
```

## The combination

The combiner is handed the out-of-fold predictions, the response, the
mask and the fold map, and never a model. `"stack"` fits non-negative
weights summing to one by minimising the response head’s own loss over
the scorable cells; `"mean"`, `"median"` and `"weighted"` combine
without fitting anything.

``` r

ensemble_weights(fit)
#>   elasticnet / day  elasticnet / week elasticnet / month 
#>         0.09995481         0.03840539         0.86163979
```

The weights say how much of the combination each candidate carries, and
a weight at zero is one the combination reached past. Where the ensemble
line of the plot above sits over every curve the candidates are carrying
different parts of the signal, and where it sits on the best curve they
are not.

## Reading a level honestly

The true skill statistic is read at the threshold that maximises it,
chosen on the same units the score is then read on. That inflates the
level, and by more the fewer presences a cell holds.
[`tss_inflation()`](https://gillescolling.com/timesift/reference/tss_inflation.md)
measures the inflation for the presence counts of the design in hand.

``` r

tss_inflation(fit$y, fit$folds, skill = c(0.6, 0.9), replicates = 100)
#>   skill  reported  inflation     lower     upper replicates
#> 1   0.6 0.7595084 0.15950839 0.6885376 0.8208576        100
#> 2   0.9 0.9673158 0.06731575 0.9425344 0.9863952        100
```

The inflation is common to every candidate scored the same way, so it
cancels in a paired difference. It does not cancel in a level, so a
level is an upper bound on the skill a population has, and
[`implied_skill()`](https://gillescolling.com/timesift/reference/implied_skill.md)
inverts the map to say which population skills a level read is
consistent with.

## The arrays on their own

[`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md)
is the representation without the fitting layer around it. It bins the
readings by the calendar and summarises every bin.

``` r

x <- grain_matrix(series, plot, t, temp, grain = "week",
                  stats = c("cold_day", "mean", "warm_day"))
x
#> <timesift matrix> 60 units x 18 bins x 3 channels 
#> grain: week   stats: cold_day, mean, warm_day 
#> from  : 2021-08-30 to 2021-12-27
```

Bins follow the calendar. A month is 28, 30 or 31 days, and a week
starts on a Monday, so a bin is a real month or a real week and not a
drifting block of 730 or 168 hours.

``` r

attr(grain_matrix(series, plot, t, temp, grain = "month"), "bin_n")[1, ]
#> 2021-09-01T00:00:00Z 2021-10-01T00:00:00Z 2021-11-01T00:00:00Z 
#>                  720                  744                  720 
#> 2021-12-01T00:00:00Z 
#>                  696
```

### An extreme day is not an extreme reading

`min` and `max` take the coldest and warmest single reading of a bin.
`cold_day` and `warm_day` reduce each day to its own mean first and then
take the extreme over days. `mean_daily_min` and `mean_daily_max` take
the mean of the daily extremes, which is the exposure a typical day of
the bin brought. One hour at -50 sets `min` to -50 outright; it reaches
the day-level statistics only through its twenty-fourth of that day’s
mean.

``` r

week <- grain_matrix(series, plot, t, temp, grain = "week",
                     stats = c("min", "mean_daily_min", "cold_day", "mean",
                               "warm_day", "mean_daily_max", "max"))
round(week[1, 1, ], 2)
#>            min mean_daily_min       cold_day           mean       warm_day 
#>         -23.41         -17.61          -6.76          -0.17           4.98 
#> mean_daily_max            max 
#>          24.02          27.89
```

An encoder that ends in global pooling discards when a thermal event
happened, so the position of a bin in the year is given to it as input.

``` r

week_mean <- grain_matrix(series, plot, t, temp, grain = "week")
dimnames(bind_channels(week_mean, calendar_channels(week_mean)))[[3]]
#> [1] "mean"     "year_sin" "year_cos"
```

## One grain at a time

Where the arrays are already built,
[`grain_ladder()`](https://gillescolling.com/timesift/reference/grain_ladder.md)
fits every learner at every grain of a set on one split and one mask. It
is the ladder a run reports as a curve, reachable on its own.

``` r

set <- grain_matrix(series, plot, t, temp, grain = c("day", "week", "month"))
lad <- grain_ladder(set, fit$y, elasticnet(), folds = fit$folds, verbose = FALSE)
summary(lad)
#>      learner grain     score n_variable  best
#> 1 elasticnet   day 0.7205556          6 FALSE
#> 2 elasticnet  week 0.7581349          6 FALSE
#> 3 elasticnet month 0.8020346          6  TRUE
```

A claim about one step of that curve rests on the paired contrast. The
difference is taken inside each cell both arms scored, averaged within a
response, and summarised across the responses, which are the independent
replicates. Six responses is few, so the interval is wide and the rank
test behind `p_value` has few values to work with.

``` r

paired_contrast(lad, "month|elasticnet", "day|elasticnet")
#>                  a              b       diff      lower     upper n_variable
#> 1 month|elasticnet day|elasticnet 0.08147908 0.02078153 0.1421766          6
#>   n_cell n_favour p_value
#> 1     30        5  0.0625
```

Where the whole curve is the question rather than one step of it,
[`grain_contrasts()`](https://gillescolling.com/timesift/reference/grain_contrasts.md)
fits a mixed model on the per-cell scores and compares every grain
against the best one, correcting for the comparisons made and no others.

``` r

grain_contrasts(lad)
#>      learner grain reference        diff      lower       upper    p_value
#> 1 elasticnet   day     month -0.08147908 -0.1676444 0.004686274 0.06638815
#> 2 elasticnet  week     month -0.04389971 -0.1300651 0.042265639 0.41335058
```

## What was read

With the per-fold fits kept,
[`occlusion()`](https://gillescolling.com/timesift/reference/occlusion.md)
holds each bin of the record back in turn, rescores the held-out units,
and records the fall in score as that bin’s weight. Nothing is refitted.

``` r

kept <- timesift(targets, series, y = starts_with("sp"), id = plot, time = t,
                 sift = grains("month"), resampling = cv(v = 5), ensemble = FALSE,
                 keep_fits = TRUE, verbose = FALSE)
weight <- occlusion(kept, "elasticnet / month", permutations = 5)
head(aggregate(weight ~ part, weight, mean), 4)
#>                   part     weight
#> 1 2021-09-01T00:00:00Z 0.11109740
#> 2 2021-10-01T00:00:00Z 0.09029750
#> 3 2021-11-01T00:00:00Z 0.10322054
#> 4 2021-12-01T00:00:00Z 0.05507654
```

Holding a channel back instead asks what each statistic of a grain
carries, which is the question behind keeping a bin’s extremes at all.
