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
#> timesift  60 targets, 6 responses, 5-fold random CV, roc_auc
#> 
#> candidates, scored on the outer folds
#> candidate                    mean    won  responses
#> elasticnet / day            0.866      1  separate
#> elasticnet / week           0.897      3  separate
#> elasticnet / month          0.907      2  separate
#> 
#> procedure, chosen and weighted inside each outer training fold
#> selected                    0.908  se 0.022
#> ensemble                    0.907  se 0.016
#> selected elasticnet / month in 4, elasticnet / week in 1 of 5 folds
#> 
#> choice on every target  elasticnet / month
#> weights on every target  elasticnet / month 1.00
```

Every representation named in `sift` was built, and `learners`
defaulting to
[`elasticnet()`](https://gillescolling.com/timesift/reference/elasticnet.md),
a penalised logistic regression was fitted on each of them over the same
five folds. The report has two parts.

The candidates are the comparison. Each is scored on the five outer
folds, on the same cells, so their means can be read against each other
across grains. `won` is how many responses a candidate scored highest
on, and `responses` says whether one fitted model covered them all or
one was fitted per response. The highest of those means was picked out
on the folds it is scored on, so it is not the number to report.

The procedure rows are. Inside each outer training fold the candidates
were cross-validated again on five inner folds, one was chosen on its
inner score and the stack’s weights were fitted on the inner out-of-fold
predictions; the choice and the weights then predicted the outer test
fold once. `selected` and `ensemble` are those held-out scores, choosing
and weighting included.

``` r

fit$estimate[c("arm", "metric", "score", "se", "lower", "upper")]
#>         arm               metric      score          se       lower      upper
#> 1  selected             accuracy  0.8777778 0.015908690  0.83688319  0.9186724
#> 2  selected    average_precision  0.9134774 0.018894456  0.86490764  0.9620471
#> 3  selected                 bias  1.0727116 0.056937161  0.92635001  1.2190733
#> 4  selected                boyce  0.6824181 0.055424952  0.53994375  0.8248925
#> 5  selected                  csi  0.7860714 0.024178549  0.72391849  0.8482244
#> 6  selected                  ets  0.6348278 0.041399419  0.52840719  0.7412484
#> 7  selected                  far  0.1187302 0.028711737  0.04492429  0.1925360
#> 8  selected                kappa  0.6367593 0.034685563  0.54759718  0.7259213
#> 9  selected         kappa_youden  0.7543294 0.032708510  0.67024955  0.8384093
#> 10 selected              neg_mae -0.3557994 0.019010008 -0.40466614 -0.3069326
#> 11 selected        neg_max_error -0.6121056 0.011515318 -0.64170664 -0.5825045
#> 12 selected              neg_mse -0.1542971 0.013224621 -0.18829204 -0.1203021
#> 13 selected neg_poisson_deviance -0.4858941 0.024935228 -0.54999210 -0.4217960
#> 14 selected             neg_rmse -0.3900576 0.015724765 -0.43047939 -0.3496358
#> 15 selected                   or 23.5000000 0.500000000 17.14689763 29.8531024
#> 16 selected     ordinal_accuracy  0.8138889 0.017435950  0.76906835  0.8587094
#> 17 selected           ordinal_f1  0.8197278 0.017391577  0.77502138  0.8644343
#> 18 selected    ordinal_precision  0.8219907 0.016217367  0.78030267  0.8636788
#> 19 selected       ordinal_recall  0.8180159 0.018711488  0.76991646  0.8661153
#> 20 selected                 orss  0.9890723 0.006914267  0.97129857  1.0068459
#> 21 selected              pearson  0.7009506 0.032456953  0.61751740  0.7843839
#> 22 selected                  pod  0.8964683 0.025170501  0.83176542  0.9611711
#> 23 selected                 pofd  0.1257143 0.039611921  0.02388860  0.2275400
#> 24 selected            r_squared  0.3437177 0.059344801  0.19116699  0.4962683
#> 25 selected              roc_auc  0.9084160 0.021620212  0.85283948  0.9639925
#> 26 selected                   sr  0.8812698 0.028711737  0.80746397  0.9550757
#> 27 selected                  tss  0.7707540 0.035825510  0.67866156  0.8628464
#> 28 ensemble             accuracy  0.8750000 0.011180340  0.84626002  0.9037400
#> 29 ensemble    average_precision  0.9143980 0.013535606  0.87960363  0.9491924
#> 30 ensemble                 bias  1.0679497 0.048673935  0.94282940  1.1930701
#> 31 ensemble                boyce  0.6897743 0.037156752  0.59425981  0.7852888
#> 32 ensemble                  csi  0.7776058 0.018333372  0.73047839  0.8247333
#> 33 ensemble                  ets  0.6189548 0.030714246  0.54000128  0.6979082
#> 34 ensemble                  far  0.1224339 0.023392168  0.06230238  0.1825653
#> 35 ensemble                kappa  0.6253307 0.016040524  0.58409721  0.6665642
#> 36 ensemble         kappa_youden  0.7477055 0.024184193  0.68553807  0.8098730
#> 37 ensemble              neg_mae -0.3530131 0.018303754 -0.40006442 -0.3059618
#> 38 ensemble        neg_max_error -0.6163724 0.011503915 -0.64594412 -0.5868006
#> 39 ensemble              neg_mse -0.1521954 0.011427574 -0.18157095 -0.1228199
#> 40 ensemble neg_poisson_deviance -0.4823667 0.022655663 -0.54060494 -0.4241285
#> 41 ensemble             neg_rmse -0.3879671 0.013953420 -0.42383548 -0.3520987
#> 42 ensemble                   or 23.5000000 0.500000000 17.14689763 29.8531024
#> 43 ensemble     ordinal_accuracy  0.8138889 0.017435950  0.76906835  0.8587094
#> 44 ensemble           ordinal_f1  0.8197278 0.017391577  0.77502138  0.8644343
#> 45 ensemble    ordinal_precision  0.8219907 0.016217367  0.78030267  0.8636788
#> 46 ensemble       ordinal_recall  0.8180159 0.018711488  0.76991646  0.8661153
#> 47 ensemble                 orss  0.9890723 0.006914267  0.97129857  1.0068459
#> 48 ensemble              pearson  0.7050684 0.027212866  0.63511551  0.7750213
#> 49 ensemble                  pod  0.8917063 0.022858665  0.83294628  0.9504664
#> 50 ensemble                 pofd  0.1257143 0.033072963  0.04069753  0.2107310
#> 51 ensemble            r_squared  0.3522563 0.051730730  0.21927827  0.4852344
#> 52 ensemble              roc_auc  0.9074901 0.016384905  0.86537134  0.9496088
#> 53 ensemble                   sr  0.8775661 0.023392168  0.81743466  0.9376976
#> 54 ensemble                  tss  0.7659921 0.027637201  0.69494838  0.8370358
fit$selected[c("fold", "candidate", "inner_score")]
#>   fold          candidate inner_score
#> 1    1 elasticnet / month   0.9035741
#> 2    2 elasticnet / month   0.8993175
#> 3    3  elasticnet / week   0.9297381
#> 4    4 elasticnet / month   0.9089008
#> 5    5 elasticnet / month   0.9144762
```

The interval is across the six responses of this dataset, all fitted and
scored on the same plots and folds, so it does not carry the error those
share.

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

![Mean AUC against representation, with the level the combination
reached drawn across it.](timesift_files/figure-html/plot-run-1.svg)

[`predict()`](https://rdrr.io/r/stats/predict.html) rebuilds each
member’s representation for new rows from the settings its own arm was
built with, and combines them through the ensemble refitted on every
target; `candidate = "selected"` predicts with the candidate the rule
chose on every target instead.

``` r

p <- predict(fit, targets, series)
round(p[1:3, 1:4], 3)
#>        sp1   sp2   sp3   sp4
#> p001 0.290 0.803 0.396 0.741
#> p002 0.583 0.534 0.576 0.548
#> p003 0.170 0.841 0.334 0.774
```

`type = "binary"` cuts each response into presence and absence. The cut
is learned from the same candidate’s out-of-fold predictions of the
fit’s own targets, so it comes from predictions of plots the model had
not been fitted on.
[`decision_threshold()`](https://gillescolling.com/timesift/reference/kappa_score.md)
on the fit returns the cuts themselves, and `rule` picks how they are
learned: `"youden"` maximises sensitivity plus specificity, `"kappa"`
maximises Cohen’s kappa, and `"prevalence"` predicts as many presences
as were observed.

``` r

decision_threshold(fit, rule = "prevalence")
#>       sp1       sp2       sp3       sp4       sp5       sp6 
#> 0.5454921 0.5204788 0.5309027 0.5044327 0.5217123 0.5156719
predict(fit, targets, series, type = "binary", rule = "prevalence")[1:3, 1:4]
#>      sp1 sp2 sp3 sp4
#> p001   0   1   0   1
#> p002   1   1   1   1
#> p003   0   1   0   1
```

A binary map is this prediction with one target per map cell, each
carrying a series of its own at the grain the fit reads.

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
#> settings: alpha = 0.5, n_inner = 10, squares = TRUE, s = lambda.1se, n_lambda = 100, tol = 1e-08, threads = 1, seed = 1
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
         learners = elasticnet(), sift = native())
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
#> 12 12 12 12 12
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
and [`linear()`](https://gillescolling.com/timesift/reference/linear.md)
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

[`elasticnet()`](https://gillescolling.com/timesift/reference/elasticnet.md)
is the arm a network is measured against, so it is fitted by the same
compiled core the Python package calls rather than by a fitter on either
side: the penalty path, the standardisation and the cross-validated
choice of penalty are one implementation, and the two languages return
the same coefficients for the same design. `s` reads that fit at
`"lambda.1se"`, glmnet’s default, at `"lambda.min"`, or at a penalty of
your own, and `tol` trades how close the descent settles to the optimum
against what it costs.

Architecture belongs to the constructor and training belongs to
[`train_control()`](https://gillescolling.com/timesift/reference/train_control.md),
which every neural learner of a run reads. A learner given a control of
its own overrides the run’s on the settings it names and takes the rest
from it.

``` r

train_control(epochs = 200L, device = "cpu")
#> <timesift control>
#>   epochs           200
#>   batch_size       64   (default)
#>   learning_rate    0.001   (default)
#>   weight_decay     1e-04   (default)
#>   optimizer        adamw   (default)
#>   penalty          0   (default)
#>   alpha            0.5   (default)
#>   schedule         cosine   (default)
#>   plateau_factor   0.1   (default)
#>   plateau_patience 10   (default)
#>   early_stopping   Inf   (default)
#>   val_frac         0   (default)
#>   device           cpu
#>   seed             1   (default)
#>   swa              FALSE   (default)
#>   swa_start        0.7   (default)
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
ship. [`as.matrix()`](https://rdrr.io/r/base/matrix.html) lays a
representation out as the block of predictors the tabular learners read,
one column per bin of each channel. A fit that declares a `weights`
argument is handed the case weights every shipped learner fits under,
which weight a presence by how rare it is.

``` r

nearest_neighbour <- learner(
  "1nn",
  fit = function(x, y, ...) list(x = as.matrix(x), y = y),
  predict = function(model, x) {
    m <- as.matrix(x)
    d <- as.matrix(dist(rbind(m, model$x)))[seq_len(nrow(m)), -seq_len(nrow(m))]
    model$y[apply(d, 1, which.min), , drop = FALSE]
  }
)

weighted_glm <- learner(
  "weighted_glm",
  fit = function(x, y, weights, ...) {
    f <- as.data.frame(as.matrix(x))
    lapply(seq_len(ncol(y)), function(j) {
      glm(y[, j] ~ ., data = f, weights = weights[, j], family = quasibinomial(),
          control = glm.control(maxit = 100))
    })
  },
  predict = function(model, x) {
    f <- as.data.frame(as.matrix(x))
    vapply(model, function(m) predict(m, f, type = "response"), numeric(nrow(f)))
  },
  data = grain("month")
)

both <- timesift(targets, series, y = starts_with("sp"), id = plot, time = t,
                 learners = c(elasticnet(), nearest_neighbour, weighted_glm),
                 sift = grains("week", "month"), resampling = cv(v = 5), verbose = FALSE)
summary(both)
#> timesift  60 targets, 6 responses, 5-fold random CV, roc_auc
#> 
#> candidates, scored on the outer folds
#> candidate                      mean    won  responses
#> 1nn / week                    0.720      0  separate
#> 1nn / month                   0.772      0  separate
#> elasticnet / week             0.897      2  separate
#> weighted_glm / month          0.904      3  separate
#> elasticnet / month            0.907      1  separate
#> 
#> procedure, chosen and weighted inside each outer training fold
#> selected                      0.908  se 0.022
#> ensemble                      0.896  se 0.016
#> selected elasticnet / month in 4, elasticnet / week in 1 of 5 folds
#> 
#> choice on every target  elasticnet / month
#> weights on every target  weighted_glm / month 0.66   elasticnet / month 0.30   1nn / month 0.04
```

## The combination

The combiner is handed out-of-fold predictions, the response, the mask
and the fold map, and never a model. `"stack"` fits non-negative weights
summing to one by minimising the response head’s own loss over the
scorable cells; `"mean"`, `"median"` and `"weighted"` combine without
fitting anything, and `"committee"` is the share of members voting
presence, each at the cut its own out-of-fold predictions give. The
weights below are fitted on the outer out-of-fold predictions of every
target and are what [`predict()`](https://rdrr.io/r/stats/predict.html)
uses. The ensemble’s score is not read with them, because they were
fitted to the responses it would be scored against; each outer fold’s
weights are in `fit$fold_weights`.

``` r

ensemble_weights(fit)
#>   elasticnet / day  elasticnet / week elasticnet / month 
#>      3.375410e-139       2.476064e-23       1.000000e+00
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
#> 1   0.6 0.7581421 0.15814206 0.6931759 0.8198188        100
#> 2   0.9 0.9664636 0.06646362 0.9361726 0.9855159        100
```

The inflation is an average over the planted model’s predictions: a
level is optimistic in expectation, and
[`implied_skill()`](https://gillescolling.com/timesift/reference/implied_skill.md)
inverts the map to say which population skills a level read is
consistent with. Its size depends on how a model’s predictions are
distributed as well as on the presence counts, so two candidates of
equal skill scored on the same cells can be inflated by different
amounts, and a paired difference in TSS is not free of it. That is why a
run is scored by AUC unless told otherwise.

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
#> 1 elasticnet   day 0.8663062          6 FALSE
#> 2 elasticnet  week 0.8970888          6 FALSE
#> 3 elasticnet month 0.9074901          6  TRUE
```

A claim about one step of that curve rests on the paired contrast. The
difference is taken inside each cell both arms scored, averaged within a
response, and summarised across the responses, which are the independent
replicates. Six responses is few, so the interval is wide and the rank
test behind `p_value` has few values to work with.

``` r

paired_contrast(lad, "month|elasticnet", "day|elasticnet")
#>                  a              b       diff     center       lower      upper
#> 1 month|elasticnet day|elasticnet 0.04118386 0.04118386 -0.01256729 0.09493501
#>   n_variable n_cell n_favour p_value p_method  interval
#> 1          6     30        5 0.09375    exact variables
```

Where the whole curve is the question rather than one step of it,
[`grain_contrasts()`](https://gillescolling.com/timesift/reference/grain_contrasts.md)
fits a mixed model on the per-cell scores and compares every grain
against the best one, correcting for the comparisons made and no others.

``` r

grain_contrasts(lad)
#>      learner grain reference        diff       lower       upper    p_value
#> 1 elasticnet   day     month -0.04118386 -0.08790089 0.005533166 0.09132994
#> 2 elasticnet  week     month -0.01040123 -0.05711826 0.036315794 0.83437774
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
#> 1 2021-09-01T00:00:00Z 0.05928263
#> 2 2021-10-01T00:00:00Z 0.09155445
#> 3 2021-11-01T00:00:00Z 0.04998280
#> 4 2021-12-01T00:00:00Z 0.01739462
```

Holding a channel back instead asks what each statistic of a grain
carries, which is the question behind keeping a bin’s extremes at all.
