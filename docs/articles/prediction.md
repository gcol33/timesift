# Ensembles, thresholds and maps

A scored run is a comparison: every candidate fitted on the same folds
and read on the same cells. This article follows what comes after it. It
starts with the split, because every number downstream is read off it,
then turns to how the candidates are combined, how a combined prediction
becomes presence and absence, and how a fit is carried onto a map for
the present and for a warmer record. The last section collects what the
numbers below say about when each choice pays. Everything runs on one
small simulated grid.

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

## A record on a grid

The grid has 12 by 10 cells. Each cell carries a daily temperature
record for one year, a value of elevation, its coordinates, and the
presence or absence of four species. Two properties of a cell drive the
record: `warmth` shifts the whole year up or down and runs along a
gradient across the grid, and `frost` sets how deep a cold snap in early
February goes, deeper towards the north. The first two species follow
warmth in opposite directions, the third follows the depth of the snap,
and the fourth both.

``` r

nx <- 12L
ny <- 10L
cell <- seq_len(nx * ny)
x <- (cell - 1L) %% nx + 0.5
y <- ny - (cell - 1L) %/% nx - 0.5
warmth <- as.numeric(scale(0.35 * x + 0.2 * y + rnorm(length(cell), sd = 0.6)))
frost <- as.numeric(scale(-0.3 * y + rnorm(length(cell), sd = 0.8)))

days <- seq(as.POSIXct("2021-01-01", tz = "UTC"), by = "day", length.out = 365)
doy <- seq_along(days)
snap <- exp(-((doy - 40) / 5)^2)
temp <- t(vapply(cell, function(i) {
  6 + 1.5 * warmth[i] - 10 * cos(2 * pi * doy / 365) - 6 * (1 + frost[i]) * snap +
    rnorm(365, sd = 3)
}, numeric(365)))

series <- data.frame(cell = rep(cell, times = 365), day = rep(days, each = length(cell)),
                     temp = as.numeric(temp))
targets <- data.frame(cell = cell, x = x, y = y,
                      elevation = round(2200 - 150 * warmth + rnorm(length(cell), sd = 40)))
signal <- cbind(2.2 * warmth, -2.2 * warmth, 2.2 * frost, 1.5 * warmth - 1.5 * frost)
targets[paste0("sp", 1:4)] <- lapply(1:4, function(j) {
  rbinom(length(cell), 1, plogis(signal[, j] - 0.5))
})
targets$transect <- paste0("t", ny - floor(y))
colSums(targets[paste0("sp", 1:4)])
#> sp1 sp2 sp3 sp4 
#>  52  59  51  47
```

Cells are numbered row by row from the top left, which is the order
`terra` numbers the cells of a raster in. The `temp` matrix, one row per
cell and one column per day, is therefore also the content of a raster
with 365 layers, and the maps at the end are built from it directly.
`transect` names the row of the grid a cell sits in.

## The split, drawn once

A run reads one fold map. Every candidate is fitted and scored on it,
the combiner is fitted on the predictions it produces, and the
thresholds are learned from those predictions, so the split decides what
every later number means. Four constructors describe how it is drawn,
and each returns a specification that the run turns into a map once it
has the targets in hand.

``` r

cv(v = 5)
#> <timesift resampling> cv in 5 folds 
#> strata    : 5
grouped_cv("transect", v = 5)
#> <timesift resampling> grouped_cv in 5 folds 
#> grouped by: transect
block_cv(c("x", "y"), v = 5)
#> <timesift resampling> block_cv in 5 folds 
#> cut on    : x, y
env_cv("elevation", v = 5)
#> <timesift resampling> env_cv in 5 folds 
#> cut on    : elevation
```

[`cv()`](https://gillescolling.com/timesift/reference/cv.md) deals units
into folds within strata of the number of responses present at a unit,
so each fold carries a similar mix.
[`grouped_cv()`](https://gillescolling.com/timesift/reference/cv.md)
deals whole groups, so every target sharing a value of `transect` lands
in one fold.
[`block_cv()`](https://gillescolling.com/timesift/reference/cv.md) cuts
the units into compact blocks of the columns it is given, here the
coordinates: it halves the set at the median of the widest column and
repeats until there are `v` blocks, and nothing about it is random.
[`env_cv()`](https://gillescolling.com/timesift/reference/cv.md) does
the same on predictor columns, each centred and scaled first, so that a
fold holds out a region of the environment, here a band of elevation.
Both [`cv()`](https://gillescolling.com/timesift/reference/cv.md) and
[`grouped_cv()`](https://gillescolling.com/timesift/reference/cv.md)
take `repeats`, which draws the split again under the next seed and
makes a full run on each.

[`fold_map()`](https://gillescolling.com/timesift/reference/fold_map.md)
is the map itself, drawn from a response matrix. With `group` it deals
groups, as
[`grouped_cv()`](https://gillescolling.com/timesift/reference/cv.md)
does inside a run.

``` r

resp <- as.matrix(targets[paste0("sp", 1:4)])
rownames(resp) <- targets$cell
random <- fold_map(resp, v = 5)
table(random)
#> random
#>  1  2  3  4  5 
#> 24 24 24 24 24
by_transect <- fold_map(resp, v = 5, group = targets$transect)
table(transect = targets$transect, fold = by_transect)[1:4, ]
#>         fold
#> transect  1  2  3  4  5
#>      t1  12  0  0  0  0
#>      t10  0  0 12  0  0
#>      t2  12  0  0  0  0
#>      t3   0  0  0  0 12
```

The plain map puts 24 cells in each fold. The grouped one puts each
transect of twelve cells in a single fold, so two transects make one
fold of 24.

A score for one response needs both presences and absences among the
held-out units, and a model of one response needs both among the units
it is fitted on.
[`scorable_cells()`](https://gillescolling.com/timesift/reference/scorable_cells.md)
reads which (response, fold) cells meet both conditions, from the
response and the fold map alone.

``` r

cells <- scorable_cells(resp, random)
cells
#> <timesift cells> 20 cells over 4 variables 
#> scorable: 20 (100.0%); variables with at least one scorable fold: 4 of 4
head(as.data.frame(cells), 4)
#>   variable fold n_occ pres_train abs_train pres_test abs_test scorable
#> 1      sp1    1    52         42        54        10       14     TRUE
#> 2      sp1    2    52         39        57        13       11     TRUE
#> 3      sp1    3    52         40        56        12       12     TRUE
#> 4      sp1    4    52         43        53         9       15     TRUE
```

Every one of the 20 cells is scorable here, because each species is
present at between 47 and 59 of the 120 cells. With a rare species some
folds would hold no presence, and those cells would drop out of every
candidate’s score at once. Because no model enters the mask, all
candidates share one denominator.

## One run under two splits

The run fits two learners at two grains, an elastic net and a random
forest at the week and at the month, with elevation carried as a static
predictor. It is made twice: once under a random five-fold split and
once under five spatial blocks.

``` r

fit <- timesift(targets, series, y = starts_with("sp"), id = cell, time = day, x = temp,
                static = elevation, learners = c(elasticnet(), forest()),
                sift = grains("week", "month"), resampling = cv(v = 5), verbose = FALSE)
fit
#> timesift  120 targets, 4 responses, 5-fold random CV, roc_auc
#> 
#> candidates, scored on the outer folds
#> candidate                    mean    won  responses
#> forest / month              0.840      0  separate
#> forest / week               0.855      0  separate
#> elasticnet / week           0.865      1  separate
#> elasticnet / month          0.870      3  separate
#> 
#> procedure, chosen and weighted inside each outer training fold
#> selected                    0.866  se 0.014
#> ensemble                    0.861  se 0.019
#> selected elasticnet / month in 3, elasticnet / week in 2 of 5 folds
#> 
#> choice on every target  elasticnet / month
#> weights on every target  forest / week 0.52   elasticnet / month 0.26   forest / month 0.22
```

The candidate rows are the comparison on the outer folds, and the two
procedure rows are the held-out level. Inside each outer training fold
the candidates were cross-validated again on five inner folds; one was
chosen on its inner score and the stack’s weights were fitted on the
inner out-of-fold predictions, and both then predicted the outer test
fold once. Under the random split the selected candidate scores an AUC
of 0.866 and the stack 0.861.

The block run differs in two settings, and both follow from the blocks.
The inner split of a nested run deals whole blocks as the outer one
does, so with five blocks an outer training set holds four and `n_inner`
can be at most 4. The elastic net chooses its penalty on folds of its
own, which keep blocks whole as well; an inner training set then holds
three blocks, and `elasticnet(n_inner = 3L)` asks for no more folds than
that. Its default of ten would ask for more folds than there are blocks,
which is an error.

``` r

blocked <- timesift(targets, series, y = starts_with("sp"), id = cell, time = day, x = temp,
                    static = elevation, learners = c(elasticnet(n_inner = 3L), forest()),
                    sift = grains("week", "month"), resampling = block_cv(c("x", "y"), v = 5),
                    n_inner = 4L, verbose = FALSE)
blocked
#> timesift  120 targets, 4 responses, 5-fold grouped CV, roc_auc
#> 
#> candidates, scored on the outer folds
#> candidate                    mean    won  responses
#> forest / month              0.682      0  separate
#> forest / week               0.708      0  separate
#> elasticnet / week           0.782      1  separate
#> elasticnet / month          0.784      3  separate
#> 
#> procedure, chosen and weighted inside each outer training fold
#> selected                    0.784  se 0.023
#> ensemble                    0.791  se 0.030
#> selected elasticnet / month in 5 of 5 folds
#> 
#> choice on every target  elasticnet / month
#> weights on every target  forest / week 0.62   elasticnet / month 0.38
```

Each run keeps the map it read in `$folds`. Drawn on the grid, the
random map scatters every fold over the grid and the block map gives
each fold one compact region.

``` r

op <- par(mfrow = c(1, 2), mar = c(1, 1, 2, 1))
show_folds <- function(f, main) {
  m <- matrix(f[as.character(cell)], nrow = ny, ncol = nx, byrow = TRUE)
  image(seq_len(nx), seq_len(ny), t(m[ny:1, ]), col = hcl.colors(5, "Set 2"),
        axes = FALSE, xlab = "", ylab = "", main = main)
}
show_folds(fit$folds, "cv(v = 5)")
show_folds(blocked$folds, "block_cv(c(\"x\", \"y\"), v = 5)")
```

![Two maps of the 12 by 10 grid coloured by fold: scattered under cv(),
five compact regions under
block_cv().](prediction_files/figure-html/fold-maps-1.svg)

``` r

par(op)
```

``` r

est <- function(run) run$estimate[run$estimate$metric == "roc_auc", c("arm", "score", "se")]
rbind(random = est(fit), block = est(blocked))
#>                arm     score         se
#> random.25 selected 0.8655684 0.01351887
#> random.52 ensemble 0.8611598 0.01913396
#> block.25  selected 0.7835526 0.02274001
#> block.52  ensemble 0.7911384 0.02979666
```

Under blocks the selected candidate falls by 0.082, to 0.784, and the
stack by 0.070, to 0.791. The likely reason is the gradient: in the
random split a held-out cell nearly always has neighbours on the
training side at the same position on the warmth gradient, while under
blocks a whole region is held out and the model has to reach a stretch
of the gradient it saw less of. The block number is the one that
describes prediction into a part of the grid the fit was not trained in.
The ranking of the candidates keeps its order between the two runs, and
the forests lose more than the elastic nets: 0.840 to 0.682 at the
month, against 0.870 to 0.784.

## The combination

The combiner is handed the out-of-fold predictions, the response, the
mask and the fold map, and never a model. The default,
`ensemble("stack")`, fits non-negative weights summing to one that
minimise the binomial deviance over the scorable cells, one weight
vector across all responses.

``` r

round(ensemble_weights(fit), 3)
#>  elasticnet / week elasticnet / month      forest / week     forest / month 
#>              0.000              0.264              0.516              0.220
fold_w <- fit$fold_weights
fold_w[-1] <- round(fold_w[-1], 3)
fold_w
#>   fold elasticnet / week elasticnet / month forest / week forest / month
#> 1    1             0.000              0.032         0.968          0.000
#> 2    2             0.000              0.373         0.392          0.234
#> 3    3             0.000              0.389         0.318          0.293
#> 4    4             0.257              0.000         0.372          0.372
#> 5    5             0.000              0.154         0.846          0.000
fit$selected[c("fold", "candidate", "inner_score")]
#>   fold          candidate inner_score
#> 1    1 elasticnet / month   0.8692383
#> 2    2  elasticnet / week   0.8928760
#> 3    3 elasticnet / month   0.8704109
#> 4    4  elasticnet / week   0.8867160
#> 5    5 elasticnet / month   0.8935208
```

[`ensemble_weights()`](https://gillescolling.com/timesift/reference/ensemble_weights.md)
returns the weights fitted on the outer out-of-fold predictions of every
target, which are the weights
[`predict()`](https://rdrr.io/r/stats/predict.html) uses. They put 0.516
on the weekly forest, 0.264 on the monthly elastic net and 0.220 on the
monthly forest, and nothing on the weekly elastic net. Those weights
were fitted to the responses the run is scored against, so the
ensemble’s score is never read with them. `fit$fold_weights` holds the
weights each outer fold fitted on its own training targets, and these
are the ones behind the held-out `ensemble` row. They move between
folds: the weekly forest takes 0.968 in fold 1 and 0.318 in fold 3.
`fit$selected` shows the same instability for the single choice, which
alternates between the two elastic nets.

[`ensemble()`](https://gillescolling.com/timesift/reference/ensemble.md)
offers five methods. `"stack"` fits the weights; `"mean"` and `"median"`
combine without fitting anything; `"weighted"` weights each candidate by
its mean score, with `decay` to weight by rank as biomod2’s
`EMwmean.decay` does; and `"committee"` cuts each member at its own
threshold and averages the votes. `min_score` drops any candidate whose
mean score is below it, and `scope` narrows the members to the learners
at the best representation or to the best learner across
representations. A run takes one of these as `ensemble =`.

[`ensemble_fit()`](https://gillescolling.com/timesift/reference/ensemble_fit.md)
is the combiner on its own. Given the run’s out-of-fold predictions it
fits any of the five, and
[`ensemble_combine()`](https://gillescolling.com/timesift/reference/ensemble_combine.md)
applies the result to a set of member predictions.

``` r

methods <- c("stack", "mean", "median", "weighted", "committee")
stacks <- lapply(setNames(methods, methods), function(m) {
  ensemble_fit(fit$oof, fit$y, fit$cells, fit$folds, ensemble(m), scores = fit$scores)
})
stacks$stack
#> <timesift stack> stack over 4 candidates 
#> fitted on 480 scorable cells , binary_cross_entropy 0.4862 
#>   forest / week                0.516
#>   elasticnet / month           0.264
#>   forest / month               0.220
#>   elasticnet / week            0.000
round(stacks$committee$thresholds, 3)
#>                      sp1   sp2   sp3   sp4
#> elasticnet / week  0.508 0.473 0.456 0.531
#> elasticnet / month 0.556 0.524 0.460 0.535
#> forest / week      0.556 0.530 0.322 0.486
#> forest / month     0.522 0.550 0.444 0.660
```

The stack refitted here carries the same weights as
`ensemble_weights(fit)`, because it was fitted on the same predictions.
The committee holds one cut per member and response, learned by the
`"youden"` rule from that member’s out-of-fold predictions.

Each combination can be scored on the outer folds with
[`score_predictions()`](https://gillescolling.com/timesift/reference/score_predictions.md),
which reads any prediction matrix on the cells the mask allows. The mean
and the median fit nothing to the response, so their scores are held-out
scores. The stack, the weighted mean and the committee were fitted on
the same predictions they are scored on here, so their scores are
optimistic, and the nested `estimate` is the one to quote for them.

``` r

auc <- vapply(stacks, function(st) {
  s <- score_predictions(fit$y, ensemble_combine(st, fit$oof), fit$folds, fit$cells)
  mean(tapply(s$score[s$scorable], s$variable[s$scorable], mean))
}, numeric(1))
round(auc, 3)
#>     stack      mean    median  weighted committee 
#>     0.868     0.870     0.868     0.870     0.834
```

The plain mean of the four candidates scores 0.870, level with the best
single candidate’s 0.870 on the same folds, and the median 0.868. The
committee scores 0.834. With four members, a committee’s prediction is a
share of four votes and takes only five values, so it ranks the units
coarsely, and AUC reads the ranking.

[`ensemble_spread()`](https://gillescolling.com/timesift/reference/ensemble_spread.md)
reads the members side by side instead of combining them: for each
target and response the weighted mean, standard deviation, coefficient
of variation and an interval. It is the counterpart of biomod2’s `EMcv`
and `EMci`.

``` r

sp <- ensemble_spread(stacks$mean, fit$oof)
round(sp[1:4, "sp3", ], 3)
#>      mean    sd    cv lower upper
#> 1   0.318 0.051 0.160 0.237 0.399
#> 10  0.254 0.019 0.076 0.224 0.285
#> 100 0.621 0.031 0.050 0.572 0.671
#> 101 0.524 0.052 0.099 0.442 0.606
```

## What the report carries

Printing a run prints its
[`summary()`](https://rdrr.io/r/base/summary.html), documented under the
topic timesift_report. The summary is a data frame with one row per
candidate and one per procedure arm, so it can be filtered and joined
like any other table; the weights fitted on every target and the
candidate chosen on every target travel as attributes.

``` r

report <- as.data.frame(summary(fit))
report[c("candidate", "mean", "se", "won", "scored")]
#>            candidate      mean         se won      scored
#> 1     forest / month 0.8397208         NA   0 outer folds
#> 2      forest / week 0.8548242         NA   0 outer folds
#> 3  elasticnet / week 0.8652522         NA   1 outer folds
#> 4 elasticnet / month 0.8696235         NA   3 outer folds
#> 5           selected 0.8655684 0.01351887  NA      nested
#> 6           ensemble 0.8611598 0.01913396  NA      nested
attr(summary(fit), "choice")
#> [1] "elasticnet / month"
```

`won` counts the responses a candidate scored highest on: the monthly
elastic net wins three, the weekly one wins the fourth, and the weekly
forest, which carries the largest weight in the stack, wins none. A
candidate can carry the combination without being best on any single
response, which the mean alone would not show. The `se` of the two
procedure rows is across the four responses, all fitted and scored on
the same cells, so it does not include the error those share.

## Predicting new targets

[`predict()`](https://rdrr.io/r/stats/predict.html) rebuilds each
member’s representation for new targets from the settings its own arm
was built with and predicts with the models refitted on every target.
New targets need the identifier and the static columns the fit was
given, and the series needs readings for each of them.

``` r

new <- targets[1:6, c("cell", "elevation")]
new_series <- series[series$cell %in% new$cell, ]
round(predict(fit, new, new_series), 3)
#>     sp1   sp2   sp3   sp4
#> 1 0.086 0.843 0.134 0.694
#> 2 0.147 0.817 0.331 0.691
#> 3 0.129 0.845 0.199 0.176
#> 4 0.387 0.182 0.128 0.866
#> 5 0.798 0.237 0.141 0.427
#> 6 0.297 0.260 0.168 0.345
round(predict(fit, new, new_series, candidate = "selected"), 3)
#>     sp1   sp2   sp3   sp4
#> 1 0.206 0.699 0.152 0.485
#> 2 0.349 0.603 0.578 0.376
#> 3 0.334 0.682 0.433 0.395
#> 4 0.620 0.393 0.247 0.704
#> 5 0.612 0.444 0.242 0.650
#> 6 0.469 0.459 0.345 0.599
round(predict(fit, new, new_series, candidate = "forest / week"), 3)
#>     sp1   sp2   sp3   sp4
#> 1 0.038 0.892 0.130 0.770
#> 2 0.050 0.894 0.232 0.804
#> 3 0.048 0.902 0.076 0.098
#> 4 0.290 0.082 0.092 0.904
#> 5 0.870 0.150 0.120 0.332
#> 6 0.222 0.176 0.132 0.262
```

`candidate = "ensemble"` is the default and combines the members under
the refitted stack; `"selected"` is the candidate the rule chose on
every target, here `elasticnet / month`; any candidate’s name predicts
with that candidate alone. The selected elastic net’s predictions lie
closer to one half than the ensemble’s, which carries half its weight on
a forest whose predictions reach further towards zero and one.

## Thresholds

`type = "binary"` cuts each response at a threshold learned from the
same candidate’s out-of-fold predictions of the fit’s own targets, so
the cut comes from predictions of units the model had not been fitted
on.
[`decision_threshold()`](https://gillescolling.com/timesift/reference/kappa_score.md)
on the fit returns the cuts, one per response, and `rule` picks how they
are learned.

``` r

rules <- c("youden", "kappa", "prevalence", "mpa")
round(sapply(rules, function(r) decision_threshold(fit, rule = r)), 3)
#>     youden kappa prevalence   mpa
#> sp1  0.568 0.568      0.534 0.268
#> sp2  0.576 0.576      0.585 0.297
#> sp3  0.401 0.401      0.508 0.368
#> sp4  0.467 0.640      0.518 0.218
```

`"youden"` maximises sensitivity plus specificity, the cut TSS is read
at. `"kappa"` maximises Cohen’s kappa. `"prevalence"` predicts as many
presences as were observed, and reads nothing from which units held
them. `"mpa"`, the minimum predicted area, takes the highest cut that
keeps a share `perc` of the observed presences, 0.9 by default, so it
reads the presences alone. Here youden and kappa agree on three species
and part on `sp4`, 0.467 against 0.640, and mpa cuts lowest on every
species. The number of cells each rule calls present, against the
observed count:

``` r

called <- sapply(rules, function(r) colSums(predict(fit, targets, series, type = "binary", rule = r)))
cbind(observed = colSums(resp), called)
#>     observed youden kappa prevalence mpa
#> sp1       52     52    52         52  61
#> sp2       59     58    58         58  74
#> sp3       51     52    52         51  53
#> sp4       47     48    46         47  70
predict(fit, new, new_series, type = "binary", rule = "prevalence")
#>   sp1 sp2 sp3 sp4
#> 1   0   1   0   1
#> 2   0   1   0   1
#> 3   0   1   0   0
#> 4   0   0   0   1
#> 5   1   0   0   0
#> 6   0   0   0   0
```

The prevalence rule reproduces the observed counts to within one cell,
as it is built to. Youden and kappa land within two cells of them on
this grid, and mpa calls between 53 and 74 cells present against 47 to
59 observed, because it gives up specificity to keep nine presences in
ten. The binary matrix is integer, 1 for presence, and a response whose
held-out predictions give no cut would be `NA`.

## How far the members disagree

`type = "spread"` returns
[`ensemble_spread()`](https://gillescolling.com/timesift/reference/ensemble_spread.md)
on the members’ predictions for new targets, a
`[target, response, statistic]` array.

``` r

s <- predict(fit, new, new_series, type = "spread")
round(s[, "sp1", ], 3)
#>    mean    sd    cv lower upper
#> 1 0.086 0.092 1.064 0.000 0.268
#> 2 0.147 0.159 1.085 0.000 0.461
#> 3 0.129 0.157 1.220 0.000 0.439
#> 4 0.387 0.179 0.462 0.034 0.741
#> 5 0.798 0.142 0.178 0.518 1.000
#> 6 0.297 0.133 0.448 0.035 0.560
```

The interval is held inside zero and one, since the predictions are
probabilities. Where the mean is small the coefficient of variation is
large: the first three cells sit below 0.15 with a coefficient above
one, while cell 5 has a mean of 0.798 and a coefficient of 0.178. On one
target per map cell this is an uncertainty map, and the sections on maps
below draw one.

## Presence-only records

Presence-only data give a response of ones.
[`pseudo_absences()`](https://gillescolling.com/timesift/reference/pseudo_absences.md)
draws units to stand as absences from a pool of background units, by
biomod2’s three strategies: `"random"` takes any unit that is not a
presence, `"sre"` a unit outside the envelope of the presences in the
`env` columns, and `"disk"` a unit whose distance to the nearest
presence lies between `dist_min` and `dist_max`. A unit of the pool
whose `id` is a presence is never drawn.

``` r

presences <- targets[targets$sp3 == 1, c("cell", "x", "y", "elevation")]
pool <- targets[c("cell", "x", "y", "elevation")]
pseudo_absences(pool, presences, n = 10, strategy = "disk", id = "cell",
                coords = c("x", "y"), dist_min = 1.5)
#> <timesift pseudo-absences> disk strategy, 18 candidates in the pool, 1 draw of 10 
#>   cell    x   y elevation set pseudo
#> 1    4  3.5 9.5      2148   1   TRUE
#> 2   10  9.5 9.5      1947   1   TRUE
#> 3    1  0.5 9.5      2402   1   TRUE
#> 4    2  1.5 9.5      2141   1   TRUE
#> 5   22  9.5 8.5      1904   1   TRUE
#> 6   36 11.5 7.5      2036   1   TRUE
pseudo_absences(pool, presences, n = 10, strategy = "sre", id = "cell", env = "elevation",
                quantile = 0.1)
#> <timesift pseudo-absences> sre strategy, 15 candidates in the pool, 1 draw of 10 
#>   cell    x   y elevation set pseudo
#> 1   47 10.5 6.5      1988   1   TRUE
#> 2   11 10.5 9.5      1863   1   TRUE
#> 3   22  9.5 8.5      1904   1   TRUE
#> 4    1  0.5 9.5      2402   1   TRUE
#> 5    9  8.5 9.5      1960   1   TRUE
#> 6   74  1.5 3.5      2453   1   TRUE
```

The header says how many units of the pool a strategy admits. On this
grid the disk strategy admits 18 cells at least 1.5 cells from any
presence, and the envelope strategy 15 cells outside the band holding
the middle 80% of the presences’ elevations; asking either for more than
it admits is an error that names both counts. `repeats` makes several
independent draws, numbered in `set`.

``` r

drawn <- pseudo_absences(pool, presences, n = 40, strategy = "random", id = "cell", repeats = 3)
table(drawn$set)
#> 
#>  1  2  3 
#> 40 40 40
one <- drawn[drawn$set == 1, c("cell", "x", "y", "elevation")]
po <- rbind(cbind(presences, sp3 = 1), cbind(one, sp3 = 0))
po_fit <- timesift(po, series[series$cell %in% po$cell, ], y = sp3, id = cell, time = day,
                   x = temp, learners = elasticnet(), sift = grains("week"),
                   ensemble = FALSE, n_inner = NULL, resampling = cv(v = 5), verbose = FALSE)
po_fit
#> timesift  91 targets, 1 response, 5-fold random CV, roc_auc
#> 
#> candidates, scored on the outer folds
#> candidate                   mean    won  responses
#> elasticnet / week          0.772      1  separate
#> 
#> choice on every target  elasticnet / week
```

The drawn units are ordinary rows once bound to the presences with a
response of zero, and the fit does not mark them. Its AUC of 0.772
separates presences from background cells, some of which hold the
species, so it is a different quantity from the AUC of the
presence-absence run above. A fit per draw, each scored the same way,
shows how much the result depends on which cells were drawn.

## Maps

[`project()`](https://gillescolling.com/timesift/reference/project.md)
predicts one target per raster cell, each carrying the record of its own
cell, and returns a raster with one layer per response. The record is a
`SpatRaster` whose layers are instants, set with
[`terra::time()`](https://rspatial.github.io/terra/reference/time.html),
and the static predictors are a raster whose layers are named as the
fit’s `static` columns. The fit must have been made with `id`, because a
cell is named by its number.

``` r

grid <- terra::rast(nrows = ny, ncols = nx, xmin = 0, xmax = nx, ymin = 0, ymax = ny,
                    nlyrs = 365)
terra::values(grid) <- temp
terra::time(grid) <- days
elev <- terra::rast(grid, nlyrs = 1)
terra::values(elev) <- targets$elevation
names(elev) <- "elevation"
grid
#> class       : SpatRaster
#> size        : 10, 12, 365  (nrow, ncol, nlyr)
#> resolution  : 1, 1  (x, y)
#> extent      : 0, 12, 0, 10  (xmin, xmax, ymin, ymax)
#> coord. ref. : lon/lat WGS 84 (CRS84) (OGC:CRS84)
#> source(s)   : memory
#> names       :      lyr.1,      lyr.2,      lyr.3,      lyr.4,      lyr.5,      lyr.6, ...
#> min values  : -12.110636, -14.466712, -12.721483, -13.719059, -13.856304, -13.491216, ...
#> max values  :    4.73181,    3.38011,   3.467108,   3.979679,   2.536751,   5.488145, ...
#> time        : 2021-01-01 00:00:00zUTC to 2021-12-31 00:00:00zUTC (365 steps)
```

``` r

took <- system.time(now <- project(fit, series = grid, static = elev))
took[["elapsed"]]
#> [1] 0.28
now
#> class       : SpatRaster
#> size        : 10, 12, 4  (nrow, ncol, nlyr)
#> resolution  : 1, 1  (x, y)
#> extent      : 0, 12, 0, 10  (xmin, xmax, ymin, ymax)
#> coord. ref. : lon/lat WGS 84 (CRS84) (OGC:CRS84)
#> source(s)   : memory
#> names       :      sp1,      sp2,      sp3,      sp4
#> min values  : 0.021942, 0.041537, 0.102653, 0.030131
#> max values  : 0.960931, 0.939719, 0.912621, 0.943166
all.equal(as.numeric(terra::values(now[["sp1"]])),
          as.numeric(predict(fit, targets, series)[as.character(cell), "sp1"]))
#> [1] TRUE
terra::plot(now)
```

![Four maps of the predicted probability of presence, one per
species.](prediction_files/figure-html/project-1.svg)

The projection of the ensemble onto the 120 cells took 0.28 seconds. It
is the same prediction
[`predict()`](https://rdrr.io/r/stats/predict.html) gives for the same
cells as targets, which
[`all.equal()`](https://rdrr.io/r/base/all.equal.html) confirms for
`sp1`. A raster too large for memory is read `chunk` cells at a time,
5000 by default. A cell missing a reading anywhere in its record is `NA`
in every layer.

A map for another period is the same call with that period’s record.
Here the later record is the present one warmed by 1.5 degrees
throughout, with elevation unchanged. `type = "binary"` cuts each layer
at the threshold
[`decision_threshold()`](https://gillescolling.com/timesift/reference/kappa_score.md)
learns for the fit, and
[`range_change()`](https://gillescolling.com/timesift/reference/range_change.md)
counts the cells each species loses, keeps and gains between the two
binary maps.

``` r

warmer <- grid + 1.5
terra::time(warmer) <- days
now_bin <- project(fit, series = grid, static = elev, type = "binary")
later_bin <- project(fit, series = warmer, static = elev, type = "binary")
change <- range_change(now_bin, later_bin)
change
#> <timesift range change> 4 responses 
#>  variable lost kept gained absent current later percent_loss percent_gain
#>       sp1    2   50     30     38      52    80        3.846       57.692
#>       sp2   42   16      1     61      58    17       72.414        1.724
#>       sp3   14   38     14     54      52    52       26.923       26.923
#>       sp4    0   48     38     34      48    86        0.000       79.167
#>  change
#>   53.85
#>  -70.69
#>    0.00
#>   79.17
plot(change, "sp1")
```

![Map of the cells sp1 loses, keeps, gains, or is absent from in both
periods.](prediction_files/figure-html/range-1.svg)

The warm-favoured `sp1` and `sp4` gain 30 and 38 cells, the
cold-favoured `sp2` loses 42 of its 58, and `sp3`, which the simulation
ties to the depth of the snap alone, loses and gains 14 cells each and
keeps its size. `change` is the gain minus the loss as a percentage of
the present range. The map codes a cell as biomod2 does, and the plot
colours the codes: red for lost (-2), grey for kept (-1), pale for
absent in both (0) and green for gained (1). Given maps on the
probability scale instead,
[`range_change()`](https://gillescolling.com/timesift/reference/range_change.md)
takes `threshold`, one cut per response.

`type = "spread"` projects the disagreement among the members, one layer
per response and statistic, named `response.statistic`.

``` r

unc <- project(fit, series = grid, static = elev, type = "spread")
names(unc)[1:5]
#> [1] "sp1.mean" "sp2.mean" "sp3.mean" "sp4.mean" "sp1.sd"
terra::plot(unc[["sp3.sd"]])
```

![Map of the standard deviation of the ensemble members' predictions for
sp3.](prediction_files/figure-html/spread-map-1.svg)

## Response curves

[`response_curve()`](https://gillescolling.com/timesift/reference/response_curve.md)
varies one predictor across the range it takes over the targets, holds
every other predictor at its `fixed` summary over the targets, and
records the prediction. A predictor is a cell of the representation, one
statistic in one bin. Named by its channel, it is that statistic moved
in every bin at once; given as `list(bin = , channel = )`, it is a
single cell.

``` r

names(fit$representations)
#> [1] "week"  "month"
dimnames(fit$representations[["week"]])[[3]]
#> [1] "mean"      "elevation"
rc <- response_curve(fit, "ensemble", "mean")
rc
#> <timesift response curve> ensemble over mean with the rest at their mean 
#>       value variable prediction
#> 1 -20.16849      sp1 0.08532928
#> 2 -19.33751      sp1 0.08521417
#> 3 -18.50653      sp1 0.08512623
#> 4 -17.67555      sp1 0.08505960
#> 5 -16.84457      sp1 0.08501007
#> 6 -16.01358      sp1 0.08497465
plot(rc, ylim = c(0, 1.3), legend = "top")
```

![Ensemble prediction of each species against the weekly mean
temperature moved in every week at
once.](prediction_files/figure-html/curve-1.svg)

Moved in every week at once, the mean runs from the coldest weekly mean
any cell has to the warmest, so the curve sweeps a whole year from deep
winter to high summer. `sp1` rises and `sp2` falls along it, as the
simulation has them, and `sp3`, which depends on one cold week, changes
little. For the ensemble every member is moved the same way and the
combination is taken through the stack; `spread = TRUE` adds the
members’ standard deviation and interval.

The signal for `sp3` sits in a single week, and a curve over that bin
alone shows it. The coldest week of the mean record is the week of the
snap.

``` r

week <- fit$representations[["week"]]
cold_week <- dimnames(week)[[2]][which.min(colMeans(week[, , "mean"]))]
cold_week
#> [1] "2021-02-08T00:00:00Z"
rc_bin <- response_curve(fit, "elasticnet / week", list(bin = cold_week, channel = "mean"))
plot(rc_bin, ylim = c(0, 1.3))
```

![Prediction of sp3 by the weekly elastic net against the mean
temperature of the week of the cold
snap.](prediction_files/figure-html/curve-bin-1.svg)

Moving that one week, `sp3` falls as the week warms, so the deeper the
snap the more likely the species, which is how the simulation made it.
`sp1` and `sp2` do not move: one week shifts the level of the year too
little for the model of either.

Given a second predictor in `with`, the curve becomes a surface over the
grid of the two, drawn as an image for one response.

``` r

rc2 <- response_curve(fit, "elasticnet / week", "mean", with = "elevation", n = 25)
plot(rc2, "sp1")
```

![Prediction of sp1 by the weekly elastic net over the grid of the
weekly mean and elevation.](prediction_files/figure-html/curve-2d-1.svg)

The prediction for `sp1` changes along the weekly mean and hardly along
elevation, darker colours being higher.

## Practical guidance

The stack and the selected candidate are both estimated by the nested
procedure, so their held-out scores can be read against each other.
Under the random split the selected candidate scored 0.866 and the stack
0.861, a difference well inside the standard errors of 0.014 and 0.019.
One elastic net led on three of the four responses, and the weight the
stack moved onto the forests did not raise the held-out score. Under
spatial blocks the order reversed, 0.784 for the selected candidate
against 0.791 for the stack, again within the standard errors. A stack
pays when the members carry different parts of the signal and none leads
everywhere; when one candidate wins most responses, as the monthly
elastic net does here, the stack mostly reproduces it at the cost of
more models to fit and store. The weights in `fit$fold_weights` say
which case a run is in: weights that swing between folds, as the weekly
forest’s do here, mean the stack has no stable composition to report,
and the selected candidate is the simpler model to describe.

Among the methods that fit nothing, the plain mean was level with the
best candidate on the outer folds. It is a reasonable choice where the
stack’s weights are unstable. The committee ranks coarsely when there
are few members, which costs AUC; it is suited to a binary map, where
only the votes matter.

For the threshold rule:

- `"prevalence"` where the map is to carry the observed number of
  presences, and where an absolute level such as kappa is to be read,
  since it selects nothing from which units held them.
- `"youden"` where omission and commission errors cost the same; it is
  the cut TSS is read at.
- `"kappa"` where agreement beyond chance is the target, which here
  moved only the cut of `sp4`.
- `"mpa"` where missing a presence costs more than predicting a false
  one, as for a conservation area that should contain nearly all known
  occurrences. On this grid it called up to 23 more cells present than
  were observed.

Spatial folds belong wherever units are clustered in space or along the
predictor gradients the model reads, and wherever the map will be read
in places the training data did not cover. The two runs above differ by
0.070 to 0.082 AUC, the random split being the higher.
[`env_cv()`](https://gillescolling.com/timesift/reference/cv.md) is the
analogue for environmental extrapolation, and
[`grouped_cv()`](https://gillescolling.com/timesift/reference/cv.md) for
units that share a site, a transect or a logger and would otherwise be
split across training and test. Under any blocked split the inner split
keeps blocks whole too, which bounds `n_inner` and the elastic net’s own
`n_inner`.

A projection costs one prediction per cell. The ensemble of four members
took 0.28 seconds for 120 cells with a year of daily readings, and the
time grows with the number of cells and the length of the record. A
large raster is better projected once from a script, with `chunk` set to
what memory allows. The two runs themselves cost far more than the
projection: each fitted every candidate on every inner and outer fold.
