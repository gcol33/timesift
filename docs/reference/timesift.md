# Fit and compare representations of time-varying data

One call from two tables to a scored comparison and a held-out estimate
of choosing among it. `targets` is one row per thing to predict and
`series` is the long, time-stamped record belonging to those rows. Every
representation in `sift` is built and every learner in `models` is
paired with the ones it can read; each pair is a candidate.

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
  inner = 5L,
  rule = c("argmax", "coarsest_adequate"),
  response = "presence_absence",
  metric = NULL,
  control = train_control(),
  keep_fits = FALSE,
  seed = 1L,
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

  The outer split:
  [`cv()`](https://gillescolling.com/timesift/reference/cv.md),
  [`grouped_cv()`](https://gillescolling.com/timesift/reference/cv.md),
  a fold vector, or a
  [`fold_map()`](https://gillescolling.com/timesift/reference/fold_map.md)
  result.

- inner:

  Number of inner folds the choice and the stack's weights are made on
  inside each outer training set, a function of the outer training
  response returning a fold map for those targets, or `NULL` to compare
  the candidates on the outer folds without an estimate. A count deals
  the inner folds by the grouping the outer split carries, as
  [`select_grain()`](https://gillescolling.com/timesift/reference/select_grain.md)
  does.

- rule:

  How a candidate is chosen from its inner scores, `"argmax"` or
  `"coarsest_adequate"`, as in
  [`select_grain()`](https://gillescolling.com/timesift/reference/select_grain.md).

- response:

  Name of the registered response head.

- metric:

  Name of a registered metric, or a function of `(y, p)`, or `NULL` for
  the response head's own. It is what the candidates are chosen on and
  what the report reads. Whichever it is, it travels with the fit and is
  what every later rescoring reads; a function is reported as
  `<function>`. The estimate is also reported under every registered
  metric.

- control:

  [`train_control()`](https://gillescolling.com/timesift/reference/train_control.md),
  the training settings every neural learner reads.

- keep_fits:

  Keep every per-fold fitted candidate beside the refits.

- seed:

  Seed for the inner splits. Each outer fold splits under `seed` plus
  its position.

- verbose:

  Report each outer fold as it runs.

## Value

A `timesift` object, a list carrying:

- `estimate`: the held-out score of the selected candidate
  (`arm = "selected"`) and of the stack (`arm = "ensemble"`), one row
  per metric, with the standard error and the 95% interval across
  response variables. An interval across the variables of this dataset,
  not one for a new sample: every variable is fitted and scored on the
  same targets and folds, so the error they share is not in it. `NULL`
  with `inner = NULL`.

- `selected`: one row per outer fold, the candidate it chose, the inner
  score it chose on, the highest inner score and that score's standard
  error; `inner`: every candidate's inner score in every outer fold;
  `fold_weights`: the stack's weights in every outer fold, one row per
  fold. All `NULL` with `inner = NULL`.

- `predictions`: the held-out prediction of every target under the
  selected candidate and under the stack.

- `candidates`, `scores` and `oof`: every candidate, its per-cell scores
  on the outer folds and its outer out-of-fold predictions.

- `choice`, `models`, `stack` and `weights`: the procedure applied to
  every target, which is what
  [`predict()`](https://rdrr.io/r/stats/predict.html) uses. Every
  candidate is refitted on all of them; `choice` is the candidate the
  rule takes on the outer scores, with the outer folds as the split it
  chooses on, and `stack` holds weights fitted on the outer out-of-fold
  predictions.

- `representations`, `fits`, `folds`, `cells`, `y`, and the `metric`,
  `response`, `spec` and `call` it was asked for.

## Details

Within each outer fold of `resampling` the training targets are split
again into `inner` folds. Every candidate is cross-validated on that
inner split, the rule picks one on its inner score, and the stack's
weights are fitted on the inner out-of-fold predictions. Every candidate
is then refitted on the whole outer training set and predicts the outer
test fold, and the selected candidate's prediction and the prediction
combined under that fold's weights are kept. Nothing the outer test fold
holds enters the choice or the weights it is scored under, so `estimate`
is of the procedure, selection and stacking included, which is what an
ecologist applying it to a new site would run.

The same refits give every candidate an out-of-fold prediction on the
outer folds, and `scores` holds those. They say where predictive skill
saturates as the record is read more coarsely, which is the measurement
the package exists for, but the highest of them is a number the held-out
targets helped choose: read the candidates for the shape of the
comparison and `estimate` for the level. With `inner = NULL` no inner
search is run, the candidates are compared on the outer folds alone and
no estimate is made.

The cost is `v_outer * (v_inner + 1) * candidates` fits for the
evaluation and one refit per candidate on every target, against
`v_outer * candidates` for the comparison alone.

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
