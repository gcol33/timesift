# Fit and compare representations of time-varying data

One call from two tables to a scored comparison. `targets` is one row
per thing to predict and `series` is the long, time-stamped record
belonging to those rows. Every representation in `sift` is built, every
learner in `models` is fitted on the ones it can read, each on the same
folds and restricted to the same scorable cells, and the out-of-fold
predictions are stacked into an ensemble. What comes back says where
predictive skill saturates as the record is read more coarsely, which is
the measurement the package exists for.

## Usage

``` r
timesift(
  targets,
  series = NULL,
  y,
  x = NULL,
  id = NULL,
  time = NULL,
  target_time = NULL,
  static = NULL,
  models = NULL,
  sift = NULL,
  ensemble = TRUE,
  resampling = cv(),
  response = "presence_absence",
  metric = NULL,
  control = train_control(),
  keep_fits = FALSE,
  verbose = TRUE
)
```

## Arguments

- targets:

  A data frame, one row per prediction target.

- series:

  A long data frame of readings, or `NULL` to fit on `static` alone.

- y:

  Columns of `targets` holding the response, as a tidyselect expression
  such as `starts_with("sp_")`.

- x:

  Columns of `series` holding the readings, as a tidyselect expression.
  Defaults to every numeric column but `id` and `time`.

- id:

  Column naming the unit, present in both tables. A bare column name or
  a string.

- time:

  Column of reading instants in `series`, `POSIXct`.

- target_time:

  Column of `targets` anchoring each row in time, `POSIXct`. Optional,
  and what a unit carrying several targets through time needs.

- static:

  Columns of `targets` carried alongside the representation, as a
  tidyselect expression. None by default.

- models:

  A learner, a set of them from [`c()`](https://rdrr.io/r/base/c.html),
  or a list. Defaults to
  [`elasticnet()`](https://gillescolling.com/timesift/reference/elasticnet.md).

- sift:

  The representations a learner without a `data =` of its own is run
  across. A
  [`grains()`](https://gillescolling.com/timesift/reference/grains.md)
  or
  [`lookbacks()`](https://gillescolling.com/timesift/reference/grains.md)
  set, a set from [`c()`](https://rdrr.io/r/base/c.html), a bare vector
  of grain names, a single representation, or a list of them. Defaults
  to `grains("auto")`.

- ensemble:

  `TRUE` for the default stack, `FALSE` for none, or an
  [`ensemble()`](https://gillescolling.com/timesift/reference/ensemble.md)
  spec.

- resampling:

  [`cv()`](https://gillescolling.com/timesift/reference/cv.md),
  [`grouped_cv()`](https://gillescolling.com/timesift/reference/cv.md),
  a fold vector, or a
  [`fold_map()`](https://gillescolling.com/timesift/reference/fold_map.md)
  result.

- response:

  Name of the registered response head.

- metric:

  Name of a registered metric, or a function of `(y, p)`, or `NULL` for
  the response head's own. Whichever it is, it travels with the fit and
  is what every later rescoring reads; a function is reported as
  `<function>`.

- control:

  [`train_control()`](https://gillescolling.com/timesift/reference/train_control.md),
  the training settings every neural learner reads.

- keep_fits:

  Keep every per-fold fitted candidate beside the refits.

- verbose:

  Report each candidate as it runs.

## Value

A `timesift` object: a list carrying `candidates`, `scores`, `oof`,
`representations`, `stack`, `weights`, `models`, `folds`, `cells`, `y`,
and the `metric`, `response`, `spec` and `call` it was asked for.

## Rules the entry point enforces

`static` is never implicit: a column of `targets` that is neither the
response, the identifier nor the anchor is ignored unless `static` names
it, because a predictor nobody asked for is worse than one that is
missing.

One target row per `id`, unless `target_time` says where in time each
row sits. Repeated identifiers without an anchor are an error naming
them.

With `target_time`, every representation has to be anchored on the
target, so
[`native()`](https://gillescolling.com/timesift/reference/native.md),
[`grain()`](https://gillescolling.com/timesift/reference/native.md) and
[`multigrain()`](https://gillescolling.com/timesift/reference/native.md)
are refused and `sift` must be given as
[`lookbacks()`](https://gillescolling.com/timesift/reference/grains.md).
There is no default set of spans, because there is no defensible one.

Without `series`, `static` is the whole predictor block and `sift` is
ignored.

## What a learner may be handed

A learner declares whether it reads a tabular block or a sequence. A
tabular learner given
[`native()`](https://gillescolling.com/timesift/reference/native.md) is
refused before anything is built, and a sequence learner given a
representation of one bin is refused once the array says how many bins
it has. Inside a `sift` expansion such a pair is skipped and reported
once by name; named explicitly through a learner's `data =` it is an
error.

## See also

[`build_representation()`](https://gillescolling.com/timesift/reference/build_representation.md)
for the array a candidate reads,
[`fold_map()`](https://gillescolling.com/timesift/reference/fold_map.md)
for the splits.

## Examples

``` r
set.seed(1)
t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24 * 90)
units <- sprintf("p%02d", 1:30)
warmth <- rnorm(30)
logger <- data.frame(
  plot = rep(units, each = length(t)), datetime = rep(t, 30),
  temp = as.numeric(vapply(warmth, function(w) w + sin(seq_along(t) / 300), numeric(length(t)))))
plots <- data.frame(plot = units,
                    sp_a = rbinom(30, 1, plogis(2 * warmth)),
                    sp_b = rbinom(30, 1, plogis(-2 * warmth)))
# \donttest{
fit <- timesift(plots, logger, y = starts_with("sp_"), id = plot, time = datetime,
                sift = grains("week", "month"), resampling = cv(v = 3L),
                ensemble = FALSE, verbose = FALSE)
fit
# }
```
