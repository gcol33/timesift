# Define a learner

A learner is a pair of functions: one that fits a model to a
representation and a response, and one that predicts from it. Everything
the package fits goes through this pair, so a learner of your own sits
beside the ones that ship and needs no change to the ladder, the folds
or the scoring.

## Usage

``` r
learner(
  name,
  fit,
  predict,
  data = NULL,
  reads = c("tabular", "sequence"),
  multi = c("separate", "joint"),
  control = NULL,
  needs = character(),
  params = list()
)
```

## Arguments

- name:

  Name the learner is reported under.

- fit:

  A function of `(x, y, ...)`, where `x` is a `[unit, bin, channel]`
  array and `y` the response matrix for the same units, returning a
  fitted object. A `fit` that declares a `control` argument is handed
  the resolved
  [`train_control()`](https://gillescolling.com/timesift/reference/train_control.md),
  and one that declares a `head` argument is handed the registered
  response head, whose `loss` and `activation` say what it is fitting
  toward. The learners that ship read both from there and hold no
  response of their own. One that declares a `group` argument is handed
  the grouping the outer fold map keeps whole, one value per unit of `x`
  or `NULL`, so a split it draws inside the fit keeps the same groups
  whole.

- predict:

  A function of `(model, x)` returning a `[unit, variable]` matrix of
  predictions for the units of `x`, in that order.

- data:

  A representation the learner is pinned to, or `NULL` to run across
  every representation of the run.

- reads:

  `"tabular"` or `"sequence"`.

- multi:

  `"separate"` where the learner fits one model per response, `"joint"`
  where one model covers them all. It is handed the whole response
  matrix either way; this is what the learner says it does with it, and
  what a report says of the candidate.

- control:

  A
  [`train_control()`](https://gillescolling.com/timesift/reference/train_control.md)
  for this learner alone. The settings it names override the control the
  run was given; everything else is taken from that one.

- needs:

  Packages the learner requires. A learner that cannot run says so at
  once rather than falling back to something else.

- params:

  Settings carried with the learner and passed to `fit`.

## Value

A `timesift_learner`.

## Details

A learner declares what it can be handed and how it covers several
responses. `reads` is `"tabular"` where the bins reach it as a block of
predictors and `"sequence"` where their order in time is what it reads,
and `multi` is `"joint"` where one fitted model covers every response
and `"separate"` where the learner fits one per response. Either way it
is handed the whole response matrix and returns one column per response,
so nothing above the learner layer has to know which it was, and the
block of predictors is built once for the fit rather than once for every
response of it.

`data` pins a learner to one representation. Left `NULL` the learner
runs across every representation of the run.

## Examples

``` r
# The bin means of a unit, fed to one logistic regression per variable.
flat_glm <- learner(
  "flat_glm",
  fit = function(x, y, ...) {
    f <- as.data.frame(apply(x, c(1, 3), mean))
    lapply(seq_len(ncol(y)), function(j)
      stats::glm(y[, j] ~ ., data = f, family = stats::binomial()))
  },
  predict = function(model, x) {
    f <- as.data.frame(apply(x, c(1, 3), mean))
    vapply(model, function(m) stats::predict(m, f, type = "response"), numeric(nrow(f)))
  }
)
flat_glm
```
