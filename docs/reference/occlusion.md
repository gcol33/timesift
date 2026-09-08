# What part of the record a fitted model reads

Holds one bin of the record back at a time, rescores the held-out units,
and records the fall in score as that bin's weight. Nothing is refitted:
the models kept by `grain_ladder(keep_fits = TRUE)` are the ones read,
so the profile describes the models that produced the reported scores
rather than a fresh set of them.

## Usage

``` r
occlusion(x, ...)

# Default S3 method
occlusion(x, ...)

# S3 method for class 'timesift_ladder'
occlusion(
  x,
  data,
  y,
  arm,
  over = c("bin", "channel"),
  substitute = c("permute", "fold_mean", "unit_mean"),
  metric = "roc_auc",
  permutations = 20L,
  seed = 1L,
  ...
)

# S3 method for class 'timesift'
occlusion(x, candidate, over = c("bin", "channel"), ...)
```

## Arguments

- x:

  A
  [`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
  fit, or a
  [`grain_ladder()`](https://gillescolling.com/timesift/reference/grain_ladder.md)
  result, in either case fitted with `keep_fits = TRUE`.

- ...:

  Passed to the method.

- data:

  The representation set the ladder was fitted on.

- y:

  The response it was fitted to.

- arm:

  The arm to read, as `"grain|learner"` or `"learner"`.

- over:

  `"bin"` to hold each bin back in turn, `"channel"` for each channel.

- substitute:

  What a held-back part is replaced by: `"permute"`, `"fold_mean"` or
  `"unit_mean"`.

- metric:

  Name of the registered metric the rescoring is read by. The area under
  the ROC curve responds to every reordering of the units, where a
  maximum over thresholds frequently does not move at all, which is why
  it is the default here and not for the ladder.

- permutations:

  Draws averaged over, for `substitute = "permute"`.

- seed:

  Random seed.

- candidate:

  Name of the candidate to read, for a run.

## Value

A data frame of one row per held-back part and variable, carrying the
mean weight over folds and the score with and without the part.

## Details

A model has to be shown something in place of a held-back bin, and what
it is shown decides what the weight means. Permuting the bin's values
across units keeps the observed readings exactly and cuts only the link
between a reading and its unit. Replacing every unit by the fitting-fold
mean removes all between-unit variation while keeping the shape of the
year. Replacing the bin by each unit's own mean over the record keeps
how warm a unit is and removes only that bin's departure from it.

Read with `over = "channel"` the same machinery asks what each statistic
of a grain carries, holding one channel back across the whole record
instead of one bin across all channels.

Reached through a
[`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
run rather than through a ladder, the profile reads the per-fold models
the run was told to keep, so every bin is held back from a model that
never saw the units it is rescored on. The candidate is named as
[`summary()`](https://rdrr.io/r/base/summary.html) reports it.

## Examples

``` r
set.seed(1)
t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24 * 120)
units <- sprintf("p%02d", 1:40)
warmth <- rnorm(40)
d <- data.frame(
  plot = rep(units, each = length(t)), t = rep(t, length(units)),
  temp = as.numeric(vapply(warmth, function(w) w + sin(seq_along(t) / 300) + rnorm(length(t)),
                           numeric(length(t)))))
y <- matrix(rbinom(80, 1, plogis(c(warmth, -warmth))), nrow = 40,
            dimnames = list(units, c("sp1", "sp2")))
x <- grain_matrix(d, plot, t, temp, grain = "month")
lad <- grain_ladder(x, y, elasticnet(), folds = fold_map(y, v = 3),
                     keep_fits = TRUE, verbose = FALSE)
head(occlusion(lad, x, y, "month|elasticnet", permutations = 3))
```
