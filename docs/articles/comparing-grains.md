# Finding the grain

The question `timesift` is built around is at what temporal grain a
record should be read for a given prediction. On field data the answer
is unknown, so a method that claims to find it is hard to check there.
[`simulate_records()`](https://gillescolling.com/timesift/reference/simulate_records.md)
draws a record whose response depends on it at one named grain, which
turns the comparison into a test: the answer is fixed before anything is
fitted, and each tool of the package can be read against it.

This article plants a response at the weekly grain, runs the comparison
over four grains, and reads the result with the ladder, the paired
contrast, the mixed-model contrast, the nested selection and the
occlusion profile. Two further records, planted at the season and at the
day, show the ladder moving with the truth, and a record with no signal
at all shows what the same tools report when there is nothing to find.

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

## A record with a known answer

[`simulate_records()`](https://gillescolling.com/timesift/reference/simulate_records.md)
draws, for each unit, a year of readings made of three parts: a seasonal
cycle every unit shares, a constant offset of the unit’s own, and an
autocorrelated anomaly. Measurement noise is added on top. Each response
variable is driven by a weighted mean of the unit’s anomaly over a short
stretch of the year, and the shape of the weights sets the grain. Under
`mechanism = "lag"` the weights decay geometrically over four
consecutive weeks from an anchor week, one anchor per variable, so the
driver is an exact linear combination of weekly means.

``` r

sim <- simulate_records(n = 200L, mechanism = "lag", variables = 8L, days = 365L,
                        step_hours = 6, prevalence = 0.3, auc = 0.9, seed = 2L)
sim
#> <timesift simulation> lag over 200 units x 1460 readings 
#> true grain: week (53 bins), stat mean 
#> 8 variables at prevalence 0.300 (drawn 0.296), population AUC 0.90
sim$design$anchor
#> [1]  2  9 15 22 29 36 42 49
```

The true grain of a mechanism is defined as the coarsest grain at which
the driver is still an exact linear function of the binned record. Day
bins nest inside weeks, so a daily representation also holds everything
the driver reads, spread over seven times as many columns. Month bins
straddle week boundaries, and a season bin averages thirteen weeks
together, so both lose part of the driver. The anchors are the positions
of each variable’s first week among the 53 weekly bins of this record.

`auc = 0.9` is the area under the ROC curve of the driver itself against
the response. No fitted model reaches it: the readings carry noise the
driver does not, and the weights have to be estimated from 200 units.
The prevalence is shared by every variable.

The first variable reads four weeks starting at its anchor:

``` r

when <- unique(sim$readings$time)
range(when[sim$weights[, 1] > 0])
#> [1] "2021-09-06 00:00:00 UTC" "2021-10-03 18:00:00 UTC"
```

`targets` carries one row per unit with the eight responses, and
`sim$readings` is the long table of readings that belongs to them.

``` r

targets <- data.frame(unit = rownames(sim$y), sim$y)
str(sim$readings)
#> 'data.frame':    292000 obs. of  3 variables:
#>  $ unit   : chr  "d001u00001" "d001u00002" "d001u00003" "d001u00004" ...
#>  $ time   : POSIXct, format: "2021-09-01 00:00:00" "2021-09-01 00:00:00" ...
#>  $ reading: num  8.4 5.34 8.2 7.32 5.7 ...
```

## One run over four grains

[`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
builds each representation in `sift`, fits every learner on each, and
scores all of them on one set of outer folds. The learner here is the
elastic net with a shorter penalty path and a looser stopping tolerance
than its defaults, to keep this article quick to build. `year_start`
matches the boundary the simulation located its bins with, which matters
for the seasonal grain. `keep_fits = TRUE` keeps the per-fold models,
which the occlusion profile later reads.

``` r

en <- elasticnet(n_inner = 5L, n_lambda = 50L, tol = 1e-6)

fit <- timesift(
  targets, sim$readings,
  y = starts_with("v"), id = unit, time = time,
  learners = en,
  sift = grains("day", "week", "month", "season", year_start = "09-01"),
  resampling = cv(v = 5),
  n_inner = 3L,
  keep_fits = TRUE,
  verbose = FALSE
)
fit
#> timesift  200 targets, 8 responses, 5-fold random CV, roc_auc
#> 
#> candidates, scored on the outer folds
#> candidate                     mean    won  responses
#> elasticnet / season          0.550      0  separate
#> elasticnet / month           0.704      0  separate
#> elasticnet / day             0.784      0  separate
#> elasticnet / week            0.838      8  separate
#> 
#> procedure, chosen and weighted inside each outer training fold
#> selected                     0.838  se 0.012
#> ensemble                     0.838  se 0.012
#> selected elasticnet / week in 5 of 5 folds
#> 
#> choice on every target  elasticnet / week
#> weights on every target  elasticnet / week 1.00
```

The weekly candidate has the highest mean AUC, 0.838, and scored highest
on all eight responses. The daily candidate, which holds the same
information over 365 columns, sits at 0.784; the monthly and seasonal
candidates fall further, to 0.704 and 0.550, the last close to what a
model with no information would score. The planted grain is recovered,
and the level reached is about 0.06 below the population ceiling of
0.90.

The procedure rows were computed differently. In each outer training
fold the four candidates were cross-validated again on three inner
folds, one was chosen and the stack’s weights were fitted, and only then
was the outer test fold predicted. Every one of the five folds chose the
weekly candidate, and the stack put all its weight on it, so the
procedure’s estimate equals the weekly candidate’s outer mean here. That
equality is a property of this run, where the choice never wavered; on a
flatter profile the two numbers separate, as the record with no signal
below shows.

``` r

plot(fit)
```

![Mean AUC of the elastic net at the day, week, month and season grains,
peaking at week, with the ensemble level drawn
across.](comparing-grains_files/figure-html/plot-fit-1.svg)

The estimate is reported under every registered metric. Four of them are
the ones a presence-absence study usually reads:

``` r

est <- fit$estimate
est[est$arm == "selected" & est$metric %in% c("roc_auc", "tss", "average_precision", "boyce"),
    c("metric", "score", "lower", "upper", "n_variable")]
#>               metric     score     lower     upper n_variable
#> 2  average_precision 0.7213009 0.6733501 0.7692517          8
#> 4              boyce 0.7519169 0.7225914 0.7812423          8
#> 25           roc_auc 0.8380377 0.8093854 0.8666901          8
#> 27               tss 0.6253196 0.5825838 0.6680553          8
```

The interval is across the eight responses, which are fitted and scored
on the same units and the same folds, so it carries the spread between
responses and not the error they share.

`resampling = cv(v = 5)` dealt the 200 units into five folds of equal
size, balanced on richness, the number of responses present at a unit,
because one fold map serves all eight responses at once. The mask of
scorable cells is computed from the responses and that map alone, before
any model is fitted.

``` r

table(fit$folds)
#> 
#>  1  2  3  4  5 
#> 40 40 40 40 40
fit$cells
#> <timesift cells> 40 cells over 8 variables 
#> scorable: 40 (100.0%); variables with at least one scorable fold: 8 of 8
```

All 40 cells are scorable: at a prevalence near 0.3 every fold holds
both presences and absences of every response. Rarer responses change
that, as the last section shows.

## The ladder on prebuilt arrays

Where the arrays already exist,
[`grain_ladder()`](https://gillescolling.com/timesift/reference/grain_ladder.md)
runs the same comparison without the run around it: every learner at
every grain, on one fold map and one mask of scorable cells.
[`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md)
builds the four arrays from the readings, and passing the run’s own
folds makes the ladder reproduce the run’s candidate scores.

``` r

x <- grain_matrix(sim$readings, unit, time, reading,
                  grain = c("day", "week", "month", "season"), year_start = "09-01")
vapply(x, function(m) dim(m)[2L], numeric(1L))
#>    day   week  month season 
#>    365     53     12      4

lad <- grain_ladder(x, sim$y, en, folds = fit$folds, verbose = FALSE)
lad
#> <timesift ladder> 4 grains x 1 learner 
#> metric: roc_auc on 160 scorable cells 
#>      learner  grain     score n_variable  best
#> 1 elasticnet    day 0.7837905          8 FALSE
#> 2 elasticnet   week 0.8380377          8  TRUE
#> 3 elasticnet  month 0.7038627          8 FALSE
#> 4 elasticnet season 0.5499666          8 FALSE
```

The 160 scorable cells are eight responses in five folds at four grains.
The means match the candidate table above to every printed digit,
because the arrays, the learner, the folds and the mask are the same.

The plot draws each grain’s mean with a 95% interval across responses
and circles the best grain.

``` r

plot(lad)
```

![Ladder of mean AUC across day, week, month and season, with intervals
across responses and an open circle at
week.](comparing-grains_files/figure-html/plot-ladder-1.svg)

The curve rises from day to week and falls from week on. The rise is the
cost of variance: the daily array holds the driver exactly, but the
penalised fit has to find it among 365 columns with 160 training units.
The fall is the cost of bias, where the bins average away part of what
the driver reads.

## Two arms on matched cells

A step of the ladder is read with
[`paired_contrast()`](https://gillescolling.com/timesift/reference/paired_contrast.md).
It takes the difference between two arms inside each cell both scored,
averages it within a response over the folds, and summarises the
per-response means. The responses are the replicates: the interval is a
Student’s t interval across them, and `p_value` is a Wilcoxon
signed-rank test of the per-response differences against zero.

``` r

rbind(paired_contrast(lad, "week|elasticnet", "month|elasticnet"),
      paired_contrast(lad, "week|elasticnet", "day|elasticnet"))
#>                 a                b       diff     center      lower      upper
#> 1 week|elasticnet month|elasticnet 0.13417500 0.13417500 0.06109161 0.20725839
#> 2 week|elasticnet   day|elasticnet 0.05424728 0.05424728 0.02962598 0.07886857
#>   n_variable n_cell n_favour   p_value p_method  interval
#> 1          8     40        8 0.0078125    exact variables
#> 2          8     40        8 0.0078125    exact variables
```

Week leads month by 0.134 and day by 0.054 in AUC, and both intervals
exclude zero. Each contrast rests on 40 paired cells, and every one of
the eight responses favours the weekly arm (`n_favour`). Both p-values
are 0.0078, and they are equal because with eight responses all on one
side, the exact signed-rank test has reached the smallest value it can
give. That floor depends only on the number of responses:

``` r

n_responses <- 3:10
data.frame(n_responses, smallest_p = 2 / 2^n_responses)
#>   n_responses  smallest_p
#> 1           3 0.250000000
#> 2           4 0.125000000
#> 3           5 0.062500000
#> 4           6 0.031250000
#> 5           7 0.015625000
#> 6           8 0.007812500
#> 7           9 0.003906250
#> 8          10 0.001953125
```

With five responses or fewer, no contrast can reach 0.05 however large
the difference. The interval behaves the same way. The same contrast on
three of the eight responses, from a ladder at two grains:

``` r

lad3 <- grain_ladder(x[c("week", "month")], sim$y[, 1:3], en, folds = fit$folds,
                     verbose = FALSE)
paired_contrast(lad3, "week|elasticnet", "month|elasticnet")
#>                 a                b      diff    center      lower     upper
#> 1 week|elasticnet month|elasticnet 0.1167193 0.1167193 -0.2371545 0.4705932
#>   n_variable n_cell n_favour p_value p_method  interval
#> 1          3     15        3    0.25    exact variables
```

The difference is of the same sign, but the interval is several times
wider, because a t interval on two degrees of freedom has a large
multiplier, and the p-value cannot fall below 0.25.

An arm is named whole, as `"grain|learner"`. A learner named alone would
have to take its best grain, chosen on the held-out scores the contrast
is read off, and
[`paired_contrast()`](https://gillescolling.com/timesift/reference/paired_contrast.md)
refuses it.

## The whole curve at once

[`grain_contrasts()`](https://gillescolling.com/timesift/reference/grain_contrasts.md)
fits a mixed model to the per-cell scores of one learner,
`score ~ grain + (1 | variable) + (1 | fold)`, and compares every grain
against the best one by Dunnett’s many-to-one procedure. The random
effects absorb the differences in level between responses and between
folds, so each comparison is made within a response and within a fold.

``` r

grain_contrasts(lad)
#> boundary (singular) fit: see help('isSingular')
#>      learner  grain reference        diff       lower       upper      p_value
#> 1 elasticnet    day      week -0.05424728 -0.09933213 -0.00916242 1.355957e-02
#> 2 elasticnet  month      week -0.13417500 -0.17925986 -0.08909015 1.543908e-10
#> 3 elasticnet season      week -0.28807111 -0.33315597 -0.24298626 0.000000e+00
```

The singular-fit message from lme4 means that one of the two variance
components was estimated at zero; the fixed-effect contrasts are still
read in the usual way. Every grain falls below week, and the day
contrast, the closest, has an adjusted interval that excludes zero. The
three intervals have the same width, about 0.09, because the model
estimates one residual variance from all the cells and the design is
balanced. The paired intervals each carry their own spread across
responses instead: the week-to-day one is narrower than its mixed-model
counterpart, and the week-to-month one wider. The reference was itself
chosen as the best observed grain, which favours it, so this table is
best read beside the paired contrasts, which single out no grain.

## Choosing inside the training data

The best grain on the ladder was chosen on the same held-out folds its
score is read from.
[`select_grain()`](https://gillescolling.com/timesift/reference/select_grain.md)
moves the choice inside the training data. In each outer fold the
training units are split again, every candidate is fitted and scored on
the inner folds, the best one is refitted on the whole outer training
set, and the outer test fold is predicted once. The estimate that comes
back is of the procedure, choice included.

``` r

sel <- select_grain(x, sim$y, en, folds = fit$folds, n_inner = 3L, verbose = FALSE)
sel
#> <timesift selection> 5 outer folds over 4 candidates by argmax 
#> roc_auc: 0.838 for the procedure, selection included
#>   95% interval 0.809 to 0.867 (se 0.012), for the spread across the variables of this dataset, fitted and scored on these units and folds
#>    grain    learner n_selected share inner_score
#> 1   week elasticnet          5     1   0.7851108
#> 2    day elasticnet          0     0   0.7222283
#> 3  month elasticnet          0     0   0.6526898
#> 4 season elasticnet          0     0   0.5246855
sel$selected[c("fold", "grain", "inner_score", "inner_se")]
#>   fold grain inner_score    inner_se
#> 1    1  week   0.7784789 0.011835732
#> 2    2  week   0.7847287 0.018373944
#> 3    3  week   0.7665858 0.006686333
#> 4    4  week   0.7791096 0.025934327
#> 5    5  week   0.8166510 0.017529066
```

All five outer folds chose week, and the procedure scores 0.838, the
same as the run’s estimate, which made the same choice on the same
folds. The inner scores, between 0.767 and 0.817, are lower than the
outer ones because each inner model was fitted on two thirds of 160
units. They are used only to rank the candidates within a fold.

The plot draws every candidate’s inner score in every outer fold, one
line per fold, and circles the candidate each fold chose.

``` r

plot(sel, ylab = "inner AUC")
```

![Inner AUC of the four candidates in each of five outer folds, every
fold circling the weekly
candidate.](comparing-grains_files/figure-html/plot-select-1.svg)

The five lines have the same shape and every circle sits on the weekly
candidate. A selection whose circles wander between candidates is one
whose grid is flat enough for the choice to be arbitrary.

## A record with nothing to find

Under `mechanism = "none"` the driver is drawn independently of the
record, so no grain carries information and every AUC should sit near
0.5.

``` r

null <- simulate_records(n = 200L, mechanism = "none", variables = 8L, days = 365L,
                         step_hours = 6, prevalence = 0.3, auc = 0.9, seed = 2L)
x_null <- grain_matrix(null$readings, unit, time, reading,
                       grain = c("day", "week", "month", "season"), year_start = "09-01")
folds_null <- fold_map(null$y, v = 5)
summary(grain_ladder(x_null, null$y, en, folds = folds_null, verbose = FALSE))
#>      learner  grain     score n_variable  best
#> 1 elasticnet    day 0.5243166          8  TRUE
#> 2 elasticnet   week 0.5050516          8 FALSE
#> 3 elasticnet  month 0.5025545          8 FALSE
#> 4 elasticnet season 0.5001511          8 FALSE
```

The ladder has a best grain all the same, the daily one at 0.524. The
highest of four means read on the folds it was chosen on is the number
most open to chance, and a reader who took it as the level would report
skill that is not there.

``` r

select_grain(x_null, null$y, en, folds = folds_null, n_inner = 3L, verbose = FALSE)
#> <timesift selection> 5 outer folds over 4 candidates by argmax 
#> roc_auc: 0.520 for the procedure, selection included
#>   95% interval 0.480 to 0.560 (se 0.017), for the spread across the variables of this dataset, fitted and scored on these units and folds
#>    grain    learner n_selected share inner_score
#> 1    day elasticnet          4   0.8   0.5102334
#> 2  month elasticnet          1   0.2   0.5029502
#> 3   week elasticnet          0   0.0   0.5036710
#> 4 season elasticnet          0   0.0   0.4984559
```

The procedure scores 0.520 with an interval from 0.480 to 0.560, which
contains 0.5. The folds chose day four times and month once, and the
inner scores of all four candidates lie between 0.498 and 0.510. On this
record the gap between the best candidate and the procedure is small,
0.004; it grows with the number of candidates searched and with how much
they differ by chance.

`choose = "coarsest_adequate"` replaces the highest inner score by the
coarsest candidate within one standard error of it. On a flat profile
that is the least storage the record can be kept at without a measured
loss inside the training data:

``` r

coarse <- select_grain(x_null, null$y, en, folds = folds_null, n_inner = 3L,
                       choose = "coarsest_adequate", verbose = FALSE)
coarse$selected[c("fold", "grain", "inner_score", "inner_best", "inner_se")]
#>   fold  grain inner_score inner_best    inner_se
#> 1    1 season   0.5026841  0.5160886 0.021869436
#> 2    2    day   0.5136709  0.5136709 0.009197280
#> 3    3  month   0.5081641  0.5139814 0.008927186
#> 4    4   week   0.5000884  0.5043865 0.008318459
#> 5    5  month   0.5078216  0.5078216 0.002637874
```

The five folds chose all four grains between them. In folds 2 and 5 the
coarsest adequate candidate was the best one itself; in the others a
coarser grain lay within one standard error of the best. A choice that
changes from fold to fold is what a flat profile looks like, and it is
the profile to expect when the record carries nothing the response
reads.

## The ladder moves with the truth

Two more records share the design but plant the driver elsewhere. Under
`"season"` the weights are uniform over one season bin, so the true
grain is the season and every finer grain nests inside it. Under
`"event"` they are uniform over three consecutive days, so the true
grain is the day, and a week mixes the three days with four others.

``` r

ladder_for <- function(mechanism) {
  s <- simulate_records(n = 200L, mechanism = mechanism, variables = 8L, days = 365L,
                        step_hours = 6, prevalence = 0.3, auc = 0.9, seed = 2L)
  arrays <- grain_matrix(s$readings, unit, time, reading,
                         grain = c("day", "week", "month", "season"), year_start = "09-01")
  grain_ladder(arrays, s$y, en, folds = fold_map(s$y, v = 5), verbose = FALSE)
}
lad_season <- ladder_for("season")
lad_event <- ladder_for("event")
summary(lad_season)[c("grain", "score", "best")]
#>    grain     score  best
#> 1    day 0.5877275 FALSE
#> 2   week 0.6339183 FALSE
#> 3  month 0.8085942 FALSE
#> 4 season 0.8346128  TRUE
summary(lad_event)[c("grain", "score", "best")]
#>    grain     score  best
#> 1    day 0.8640497  TRUE
#> 2   week 0.7807978 FALSE
#> 3  month 0.5635600 FALSE
#> 4 season 0.5196670 FALSE
```

Each ladder peaks at its planted grain: the seasonal record at season,
0.835, and the event record at day, 0.864. The seasonal ladder falls
gently from season to month, 0.809, where the month array holds the
driver exactly over three times as many columns, and steeply below that.
The event ladder drops by 0.083 from day to week, the information the
weekly mean averages away.

``` r

plot(lad_event)
```

![Ladder of mean AUC for the event record, highest at day and falling
through week, month and
season.](comparing-grains_files/figure-html/plot-event-1.svg)

## What the weekly model read

[`occlusion()`](https://gillescolling.com/timesift/reference/occlusion.md)
holds one bin back at a time from the per-fold models the run kept,
rescores the held-out units, and records the fall in score as that bin’s
weight. Nothing is refitted. Here each weekly bin is permuted across the
held-out units five times.

``` r

weight <- occlusion(fit, "elasticnet / week", permutations = 5)
weight
#> <timesift occlusion> week|elasticnet read by roc_auc 
#> held back: bin ; substitute: permute 
#> heaviest: 2022-01-24T00:00:00Z, 2022-08-01T00:00:00Z, 2022-03-14T00:00:00Z, 2022-06-13T00:00:00Z, 2022-05-02T00:00:00Z 
#>                   part variable score_full score_held_back       weight
#> 1 2021-08-30T00:00:00Z      v01  0.8126711       0.8150954 -0.002424242
#> 2 2021-08-30T00:00:00Z      v02  0.7804821       0.7804821  0.000000000
#> 3 2021-08-30T00:00:00Z      v03  0.8136333       0.8136333  0.000000000
#> 4 2021-08-30T00:00:00Z      v04  0.8546297       0.8546297  0.000000000
#> 5 2021-08-30T00:00:00Z      v05  0.8411787       0.8444288 -0.003250083
#> 6 2021-08-30T00:00:00Z      v06  0.8818554       0.8741113  0.007744102
#>     importance
#> 1 2.000123e-04
#> 2 2.220446e-17
#> 3 4.440892e-17
#> 4 8.881784e-17
#> 5 7.285231e-03
#> 6 4.333633e-03
```

The planted weights are known, so the profile can be checked response by
response. The first variable’s anchor is the second weekly bin, the week
of 6 September 2021; the second variable’s is the ninth, the week of 25
October.

``` r

anchors <- dimnames(x$week)[[2L]][sim$design$anchor[1:2]]
anchors
#> [1] "2021-09-06T00:00:00Z" "2021-10-25T00:00:00Z"
heaviest <- function(v) {
  w <- as.data.frame(weight)
  w <- w[w$variable == v, c("part", "weight", "importance")]
  head(w[order(-w$weight), ], 4)
}
heaviest("v01")
#>                     part     weight importance
#> 9   2021-09-06T00:00:00Z 0.28650026  0.9091535
#> 169 2022-01-24T00:00:00Z 0.11151412  0.2306684
#> 17  2021-09-13T00:00:00Z 0.05880619  0.1544646
#> 65  2021-10-25T00:00:00Z 0.01701299  0.0312855
heaviest("v02")
#>                     part     weight importance
#> 66  2021-10-25T00:00:00Z 0.24679058 0.91812116
#> 194 2022-02-14T00:00:00Z 0.03154324 0.11461641
#> 202 2022-02-21T00:00:00Z 0.02564648 0.03822851
#> 18  2021-09-13T00:00:00Z 0.01928881 0.04161467
```

For both variables the heaviest bin is the anchor week, with a fall in
AUC of 0.287 and 0.247. The planted weights decay by a factor of e from
one week to the next, and for the first variable the following week is
also among the heaviest. Bins outside the planted stretch carry smaller
weights, such as 0.112 for a January week in the first variable, which
suggests the penalised fit picks up some structure the mechanism does
not contain when it has 160 units to fit 53 columns.

`over = "channel"` holds back a whole statistic instead. The simulated
response reads the weekly mean, so a run that also offers each week’s
coldest and warmest day should lean on the mean.

``` r

fit_ch <- timesift(targets, sim$readings, y = starts_with("v"), id = unit, time = time,
                   learners = en,
                   sift = grain("week", stats = c("cold_day", "mean", "warm_day"),
                                year_start = "09-01"),
                   resampling = cv(v = 5), n_inner = NULL, ensemble = FALSE,
                   keep_fits = TRUE, verbose = FALSE)
by_channel <- occlusion(fit_ch, "elasticnet / week", over = "channel", permutations = 5)
aggregate(weight ~ part, by_channel, mean)
#>       part     weight
#> 1 cold_day 0.07661365
#> 2     mean 0.15731098
#> 3 warm_day 0.04626612
```

The mean carries the largest fall, 0.157, about twice the coldest day’s.
The extreme days are not independent of the mean, which is why they
carry some weight too.

## How the prediction responds

[`response_curve()`](https://gillescolling.com/timesift/reference/response_curve.md)
moves one predictor across its observed range while every other is held
at its mean, and reads the prediction off the model fitted on all
targets. A predictor here is one cell of the representation: the weekly
mean in the first variable’s anchor week.

``` r

rc <- response_curve(fit, "elasticnet / week", list(bin = anchors[1], channel = "mean"), n = 20)
plot(rc, col = c("black", rep("grey70", 7)), legend = "topleft")
```

![Predicted probability against the mean reading in the week of 6
September 2021, rising steeply for v01 and nearly flat for the other
responses.](comparing-grains_files/figure-html/response-curve-1.svg)

``` r

tapply(rc$prediction, rc$variable, function(p) round(diff(range(p)), 3))
#>   v01   v02   v03   v04   v05   v06   v07   v08 
#> 0.926 0.000 0.001 0.000 0.000 0.000 0.000 0.097
```

The first variable’s prediction rises with that week’s mean, as the
positive sign of its planted driver says it should, and spans 0.926 of
the probability scale. The other variables read different weeks. Five of
them do not move at the third decimal, a sixth moves by 0.001, and the
eighth by 0.097, a coefficient the penalised fit gave that cell without
the mechanism putting weight there.

## Scores, thresholds and their inflation

A run scores presence-absence responses by
[`roc_auc()`](https://gillescolling.com/timesift/reference/roc_auc.md)
unless told otherwise. TSS, the statistic species distribution studies
usually report, is the maximum of sensitivity plus specificity minus one
over every cut, and the cut is chosen on the units the score is then
read on. That choice inflates the level, by more the fewer presences a
cell holds.
[`score_predictions()`](https://gillescolling.com/timesift/reference/score_predictions.md)
scores any matrix of held-out predictions on the scorable cells, here
the weekly candidate’s out-of-fold predictions under four metrics:

``` r

p_week <- fit$oof[["elasticnet / week"]]
per_metric <- vapply(c("roc_auc", "tss", "average_precision", "boyce"), function(m) {
  s <- score_predictions(sim$y, p_week, fit$folds, metric = m)
  mean(tapply(s$score, s$variable, mean, na.rm = TRUE))
}, numeric(1L))
round(per_metric, 3)
#>           roc_auc               tss average_precision             boyce 
#>             0.838             0.625             0.721             0.752
```

Each metric reads the predictions differently.
[`average_precision()`](https://gillescolling.com/timesift/reference/average_precision.md)
has the prevalence as its floor, here about 0.3, where AUC has 0.5.
[`boyce_index()`](https://gillescolling.com/timesift/reference/boyce_index.md)
uses no absences and correlates the ratio of presences to all units with
the prediction across its range.
[`tss()`](https://gillescolling.com/timesift/reference/tss.md) is the
one with a cut to choose.
[`tss_inflation()`](https://gillescolling.com/timesift/reference/tss_inflation.md)
plants predictions of known skill at this design’s cell sizes and reads
them back the way a ladder does:

``` r

tss_inflation(sim$y, fit$folds, skill = c(0.6, 0.9), replicates = 50)
#>   skill  reported  inflation    lower     upper replicates
#> 1   0.6 0.6988735 0.09887349 0.667724 0.7313339         50
#> 2   0.9 0.9461458 0.04614581 0.928901 0.9611459         50
```

At a true skill of 0.6, the level read back is about 0.1 higher; at 0.9
the inflation is smaller, 0.046, nearer the upper bound of 1.
[`implied_skill()`](https://gillescolling.com/timesift/reference/implied_skill.md)
inverts the map: given the TSS the weekly candidate reached, it returns
the population skill whose expected reported level that is.

``` r

implied_skill(sim$y, fit$folds, observed = per_metric[["tss"]], replicates = 50)
#>    observed     skill within_grid
#> 1 0.6253196 0.5135347        TRUE
```

A TSS of 0.625 on this design is consistent with a population skill near
0.51. The inversion holds under the distribution of predictions
[`tss_inflation()`](https://gillescolling.com/timesift/reference/tss_inflation.md)
plants, and two models of equal skill whose predictions are distributed
differently can be inflated by different amounts. A paired difference in
TSS can therefore favour one arm with no difference in skill behind it,
which is why a run is scored by AUC, which has no cut, and reports TSS
beside it.

### Rare responses and the scorable-cell mask

A score needs both classes among the held-out units of a cell. With rare
responses some cells hold no presence, and the mask, computed from the
response and the fold map alone, leaves them out for every arm alike.

``` r

rare <- simulate_records(n = 200L, mechanism = "lag", variables = 8L, days = 365L,
                         step_hours = 6, prevalence = 0.03, auc = 0.9, seed = 2L)
colSums(rare$y)
#> v01 v02 v03 v04 v05 v06 v07 v08 
#>   7   9   8   7  11   5   4   4
folds_rare <- fold_map(rare$y, v = 5)
scorable_cells(rare$y, folds_rare)
#> <timesift cells> 40 cells over 8 variables 
#> scorable: 32 (80.0%); variables with at least one scorable fold: 8 of 8
tss_inflation(rare$y, folds_rare, skill = c(0.6, 0.9), replicates = 50)
#>   skill  reported inflation     lower     upper replicates
#> 1   0.6 0.8349672 0.2349672 0.7655255 0.8997266         50
#> 2   0.9 0.9843859 0.0843859 0.9616244 0.9954212         50
```

At a prevalence of 0.03 the eight responses hold between 4 and 11
presences in 200 units, eight of the forty cells are unscorable, and the
inflation of TSS at a true skill of 0.6 rises to 0.235. A response with
four presences spread over five folds is scored on at most four of them,
each holding one presence.

## Practical guidance

These points come from the simulations above. They describe these
designs, and a study with other sizes can run the same checks on its own
response and fold map.

- The paired contrast treats responses as replicates. With eight
  responses all favouring one arm, its exact p-value was 0.0078, the
  smallest it can be; with five or fewer no contrast can reach 0.05, and
  with three the interval on the week-to-month step was several times
  wider than with eight. A comparison of grains on a handful of species
  is best reported as differences with their intervals, without a test.
- The mixed-model contrast uses every cell and pools one residual
  variance across grains, so its intervals all had the same width,
  narrower than the paired interval for one step and wider for another.
  It takes the best observed grain as its reference, so it is read
  beside the paired contrasts.
- With 200 units and five outer folds, 40 units are held out per fold.
  For the lag record that was enough to separate week from day by 0.054
  AUC with all eight responses agreeing. The selection’s inner folds see
  two thirds of each outer training set, so their scores sit lower than
  the outer ones and serve only to rank.
- Rare responses shrink the scorable cells and widen the TSS inflation.
  [`scorable_cells()`](https://gillescolling.com/timesift/reference/scorable_cells.md)
  and
  [`tss_inflation()`](https://gillescolling.com/timesift/reference/tss_inflation.md)
  run in seconds on the response and the fold map before any model is
  fitted, and fewer folds leave more presences in each held-out cell.
- The best candidate’s outer mean is chosen on the folds it is scored
  on. On the record with no signal it read 0.524 while the procedure’s
  interval contained 0.5. The level to report is the procedure’s
  estimate, from
  [`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
  with `n_inner` set or from
  [`select_grain()`](https://gillescolling.com/timesift/reference/select_grain.md).
- Every run in this article used a single fold map. `cv(repeats = )`
  runs the comparison again on new maps and reads the estimate off all
  of them, which shows how much of a difference between grains belongs
  to one particular split.
- A selected grain says which representation predicted best on these
  units. On the lag record the daily array holds the driver exactly and
  still lost to the weekly one, because its information is spread over
  more columns; the grain a model prefers depends on the sample size as
  well as on the mechanism.
