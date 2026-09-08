# Reproducing the Schrankogel grid

The measurement `timesift` exists to make was first run on 894 alpine
plots on Schrankogel, in the central Austrian Alps: three hydrological
years of hourly soil temperature, one logger per plot, predicting
presence and absence of 101 vascular plant species. The record was read
at seven grains, from every hour to a single value per year, by three
architectures and by logistic models on 188 hand-built summaries of the
same loggers. Skill peaked at the weekly average and fell by 0.08 toward
yearly, and the coldest and warmest day of a grain carried more than its
mean.

This vignette says which settings of this package correspond to which
part of that grid, and points at the script that runs it.

## What the reproduction needs

The soil-temperature series and the species records are the Zenodo
deposit of Chytrý et al. (<doi:10.5281/zenodo.17047026>, CC BY 4.0).
Four files of it are read:

| file | what it carries |
|----|----|
| `spe_wide.csv` | presence and absence, 894 plots by 252 species |
| `logger_data.csv` | 23,515,776 hourly readings, 26,304 per plot |
| `seasons.csv` | the season each date of the record belongs to |
| `output_temperature_variables_scaled.csv` | the 188 hand-aggregated temperature summaries |

The snow arm of the paper reads a daily snow-cover series that is an
in-house product of the study group and is not in the deposit, so it is
not reproduced here. Everything else is.

## The contract

Species in at least 25 plots, then five aggregate taxa removed. That is
inherited from the published baseline rather than chosen, and it is what
fixes the species set at 101.

``` r

spe <- read.csv(file.path(deposit, "spe_wide.csv"), check.names = FALSE)
rownames(spe) <- as.character(spe$logger_ID)
counts <- colSums(spe[setdiff(names(spe), "logger_ID")])
keep <- setdiff(names(counts)[counts >= 25],
                c("Alchemilla vulgaris agg.", "Taraxacum sp.", "Festuca halleri agg.",
                  "Euphrasia sp.", "Phleum alpinum agg."))
y <- as.matrix(spe[, keep, drop = FALSE])
dim(y)
#> [1] 894 101
```

## The split and the cells

The fold map is an artifact rather than an algorithm: it is built once
and read by everything that scores, so no arm can regenerate its own.
The study wrote one to `folds.csv`, the package ships it beside the
reproduction script, and reading that file back reproduces its splits
exactly.

``` r

f <- read.csv(system.file("reproduce", "folds.csv", package = "timesift"))
folds <- setNames(as.integer(f$fold), as.character(f$logger_ID))

cells <- scorable_cells(y, folds)
sum(cells$scorable)
#> [1] 1003
```

1003 of 1010 cells, 99.3 percent, and all 101 species keep at least one
scorable fold. Building a map here instead gives a different partition
of the same design, since
[`fold_map()`](https://gillescolling.com/timesift/reference/fold_map.md)
draws on R’s random stream and `rsample` drew on a different one:

``` r

folds <- fold_map(y, v = 10, seed = 1, strata = 5)
```

## The seven grains

Six of the seven grains are named ones. The seventh is not: the deposit
cuts its seasons at the equinoxes and the solstices rather than on the
first of a month, and labels every date of the record accordingly, which
gives 13 bins over the three years rather than 12. Reading that file
back as a binning function is how the season rung is the deposit’s
season.

``` r

astronomical_seasons <- function(path) {
  labels <- read.csv(path)
  key <- paste(labels$season, format(as.Date(labels$day), "%Y"))
  edges <- sort(as.POSIXct(paste0(labels$day[!duplicated(key)], " 00:00:00"), tz = "UTC"))
  function(when) edges[findInterval(as.numeric(when), as.numeric(edges))]
}

binning <- list(native = "native", halfday = "halfday", day = "day", week = "week",
                month = "month",
                season = astronomical_seasons(file.path(deposit, "seasons.csv")),
                year = "year")
```

Each gives the bin count the paper reports:

| grain   | bins  |
|---------|-------|
| hour    | 26304 |
| halfday | 2192  |
| day     | 1096  |
| week    | 157   |
| month   | 36    |
| season  | 13    |
| year    | 3     |

The script asserts every one of them before fitting anything.

## The two readings of a grain

The resolution ladder is fitted on the grain mean, the only summary
defined at all seven grains. The models themselves are reported on the
grain’s coldest day, its mean and its warmest day, which is defined from
the weekly grain up because it reduces to whole days first.

``` r

mean_reading <- grain_matrix(readings, logger_ID, date, temp, grain = "week")
reported <- grain_matrix(readings, logger_ID, date, temp, grain = "week",
                          stats = c("cold_day", "mean", "warm_day"))
```

The paper’s other three schemes are the same call with different
`stats`: `"min"` and `"max"` alone, `c("min", "mean", "max")` for the
grain’s own extremes, and
`c("mean_daily_min", "mean", "mean_daily_max")` for the average daily
minimum and maximum.

## The channels the encoders are given

An encoder that ends in global pooling discards when a thermal event
happened, so the position of each bin in the year is supplied as two
channels. They are the time index of each bin, not a summary of the
readings, so they add no hand-built thermal feature to the network arm.

``` r

input <- bind_channels(reported, calendar_channels(reported))
```

## The arms

The aggregated-feature arms read the deposit’s 188 variables. A feature
table has no time axis, so it enters through
[`feature_matrix()`](https://gillescolling.com/timesift/reference/feature_matrix.md)
and is then an arm of the same ladder, scored on the same cells by the
same rule.

``` r

agg <- read.csv(file.path(deposit, "output_temperature_variables_scaled.csv"), check.names = FALSE)
rownames(agg) <- as.character(agg$logger_ID)
features <- feature_matrix(as.matrix(agg[rownames(y), setdiff(names(agg), "logger_ID")]),
                           label = "aggregates")

baseline <- grain_ladder(
  features, y,
  list(elastic_net = elasticnet(alpha = 0.5, n_inner = 5, squares = TRUE),
       stepwise = stepwise(max_terms = 3, degree = 2)),
  folds = folds)
```

Both redo their predictor selection inside every fold, on the training
plots only, which is the footing the encoders are fitted on. The
encoders are the three the paper reports, at the settings it reports
them at, which are the defaults here:

``` r

encoders <- list(mlp = mlp(), cnn = cnn(), rescnn = rescnn())
```

[`cnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
is four blocks of a one-dimensional convolution of kernel width 7, batch
normalisation, a rectified linear activation and max pooling by two, at
16, 32, 64 and 128 channels, then global average pooling and dropout at
0.3.
[`rescnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
is a convolutional stem and four stages of 32, 64, 128 and 256 channels
holding two dilated residual blocks each, dilations 1, 2, 4 and 8, with
squeeze-excitation gates, pooling average and maximum together.
[`mlp()`](https://gillescolling.com/timesift/reference/torch_learners.md)
flattens the channels through two hidden layers of 512 and 256 units.

How all three are trained is
[`train_control()`](https://gillescolling.com/timesift/reference/train_control.md).
Its defaults are AdamW at a learning rate of 1e-3 with weight decay
1e-4, cosine annealing over 60 epochs, early stopping after ten epochs
without an improvement on an inner validation split of 15 percent of the
fitting plots, and a per-species positive-class weight capped at 50. The
encoders of the study read batches of 32 plots, which the grid asks for;
the default is 64.

The eleven members of the paper’s ensemble are the same architectures at
three widths, three kernel widths and three seeds, trained with weight
averaging over the tail of the epoch budget:

``` r

members <- c(
  lapply(list(c(16L, 32L, 64L, 128L), c(32L, 64L, 128L, 256L), c(16L, 32L, 64L)),
         function(ch) cnn(channels = ch, swa = TRUE)),
  lapply(c(5L, 7L, 9L), function(k) cnn(kernel = k, swa = TRUE)),
  lapply(c(1L, 2L, 3L), function(s) cnn(swa = TRUE, seed = s)),
  lapply(c(1L, 2L), function(s) rescnn(swa = TRUE, seed = s)))
names(members) <- sprintf("m%02d", seq_along(members))
```

## The grid

The ladder is one call. Each grain is built once, the calendar channels
are joined to it, and every encoder is fitted at every rung on the one
fold map and the one mask.

``` r

ladder_input <- timesift_set(lapply(binning, function(w) {
  x <- grain_matrix(readings, logger_ID, date, temp, grain = w)
  bind_channels(x, calendar_channels(x))
}))

grid <- grain_ladder(ladder_input, y, encoders, folds = folds)
summary(grid)
```

The ensemble is one further arm on the same rungs. Each member runs as
an arm of its own, and their held-out predictions are averaged: a
member’s out-of-fold prediction on a fold is its held-out prediction
there, so averaging the eleven and choosing a threshold afterwards is
the set scored as one model rather than as a vote between eleven
decisions.

``` r

lad <- grain_ladder(ladder_input, y, members, folds = folds)
oof <- attr(lad, "predictions")
arms <- paste("week", names(members), sep = "|")

stack <- ensemble_fit(oof[arms], y, attr(lad, "cells"), folds, spec = ensemble("mean"))
combined <- ensemble_combine(stack, oof[arms])
```

`combined` is then scored on the same cells and by the same metric as
every other arm, which is what the driver appends to the grid under the
name `ensemble`.

## The grain contrast

The table of every grain against its architecture’s best is a mixed
model on the per-cell scores,
`score ~ grain + (1 | species) + (1 | fold)` by restricted maximum
likelihood, with Dunnett’s many-to-one procedure against the reference:

``` r

grain_contrasts(grid, learner = "cnn")
```

The design is balanced at 101 species by ten folds by seven grains, so
each grain is compared within a species and within a fold.
Benjamini-Hochberg across all eighteen comparisons is
`p.adjust(out$p_value, "BH")` over the three architectures’ tables
stacked.

## Running it

The driver is installed with the package:

``` r

system.file("reproduce", "schrankogel.R", package = "timesift")
```

    Rscript schrankogel.R <deposit_dir> <out_dir> \
      --stages=contract,representation,baseline,networks,contrasts,inflation \
      --grains=day,week,month,season,year --learners=cnn,rescnn,mlp,ensemble --folds=folds.csv

It writes one CSV per stage and asserts the input at every step: the
plot count, the species count, the rarest retained species, the cell
count, and the bin count of every grain. A stage runs over all 894 plots
and all 101 species or it does not run, so there is no setting that
quietly shrinks what a number was computed on.

The coarse grains are affordable on a processor. The hourly rung is
26,304 steps per plot and wants a graphics processor, as it had in the
study; the whole grid there was three architectures by seven grains by
ten folds, refitted under four seeds.

## What the reproduction is checked against

The paper’s own numbers, arm by arm. The verified ones, in the order the
script produces them:

| quantity | reported | reproduced |
|----|----|----|
| plots, species, rarest species | 894, 101, 26 | 894, 101, 26 |
| scorable cells | 1003 of 1010 | 1003 of 1010 |
| bins per grain | 26304, 2192, 1096, 157, 36, 13, 3 | same |
| numbers per plot, weekly three-channel | 471 | 471 |
| inflation of a level whose truth is 0.60 | +0.110 | +0.110 |
| elastic net on the 188 aggregates | 0.687 | 0.686 |
| stepwise AIC on the 188 aggregates | 0.662 | not rerun |
| convolutional network, weekly, coldest and warmest day | 0.712 | not rerun |
| the same network on the full hourly record | 0.658 | not rerun |

A model fitted twice on different hardware does not give the same
weights, so the network cells are reproduced to the seed noise the paper
measures, a standard deviation of 0.0017 across seeds at the median
cell. The representation, the fold map, the mask and the
aggregated-feature arms carry no such noise and reproduce exactly.
