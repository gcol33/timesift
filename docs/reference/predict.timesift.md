# Predict from a fitted timesift

Rebuilds every member's representation for the new targets from the
settings its own arm was built with, predicts with the model refitted on
all targets, and combines them where the ensemble is asked for.

## Usage

``` r
# S3 method for class 'timesift'
predict(
  object,
  targets,
  series = NULL,
  candidate = "ensemble",
  type = c("response", "binary", "spread"),
  rule = c("youden", "kappa", "prevalence"),
  alpha = 0.05,
  ...
)
```

## Arguments

- object:

  A
  [`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
  fit.

- targets:

  A data frame of targets, carrying the identifier, the anchor and the
  static columns the fit was given.

- series:

  The long table of readings for those targets, or `NULL` for a
  targets-only fit.

- candidate:

  `"ensemble"`, the stack refitted on every target; `"selected"`, the
  candidate the rule chose on every target (`object$choice`); or the
  name of one candidate.

- type:

  `"response"` for the prediction on the scale of the response head, a
  probability of presence under the shipped one, `"binary"` for presence
  and absence, or `"spread"` for how far the ensemble's members
  disagree.

- rule:

  With `type = "binary"`, the rule of
  [`decision_threshold()`](https://gillescolling.com/timesift/reference/kappa_score.md)
  the cut is learned by.

- alpha:

  With `type = "spread"`, one minus the coverage of the interval.

- ...:

  Ignored.

## Value

A `[target, response]` matrix of predictions, named by target and in the
order the fit carries its own targets: sorted by identifier, or the
targets' own order where `target_time` anchors them. Under
`type = "binary"` an integer matrix of 0 and 1, `NA` for a response
whose held-out predictions give no cut. Under `type = "spread"` the
`[target, response, statistic]` array of
[`ensemble_spread()`](https://gillescolling.com/timesift/reference/ensemble_spread.md).

## Details

With `type = "binary"` each response is cut into presence and absence at
the threshold
[`decision_threshold()`](https://gillescolling.com/timesift/reference/kappa_score.md)
learns from the same candidate's out-of-fold predictions of the fit's
own targets: never from predictions of units the model was fitted on,
and never from the new targets, which carry no response. `rule` is the
rule that picks it. A binary map of a species is this prediction on one
target per map cell.

With `type = "spread"` the ensemble's members are read side by side
rather than combined:
[`ensemble_spread()`](https://gillescolling.com/timesift/reference/ensemble_spread.md)
gives their weighted mean, standard deviation, coefficient of variation
and interval at `alpha`, which is biomod2's `EMcv` and `EMci` and, on
one target per map cell, an uncertainty map.

## See also

[`decision_threshold()`](https://gillescolling.com/timesift/reference/kappa_score.md)
for the cuts themselves.
