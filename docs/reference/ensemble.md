# How the candidates are combined

Every candidate emits an out-of-fold prediction for every scorable cell
over the same folds, so the combination is arithmetic on those
predictions and nothing else. `ensemble()` says which arithmetic.

## Usage

``` r
ensemble(
  method = c("stack", "mean", "median", "weighted", "committee"),
  scope = c("all", "learners", "representations"),
  metric = NULL,
  response = NULL,
  min_score = NULL,
  decay = NULL,
  rule = NULL
)
```

## Arguments

- method:

  How the members are combined.

- scope:

  Which candidates are eligible.

- metric:

  Name of the registered metric the eligibility, `min_score` and the
  `"weighted"` weights are read by, or `NULL` for the score the run
  already carries.

- response:

  Name of the registered response head whose loss `"stack"` minimises,
  or `NULL` for the head the run was fitted under. Naming a head the run
  does not fit toward is an error rather than an override, and
  [`ensemble_fit()`](https://gillescolling.com/timesift/reference/ensemble_fit.md)
  called on its own reads `NULL` as `"presence_absence"`.

- min_score:

  `NULL`, or the mean score a candidate needs to be eligible.

- decay:

  For `"weighted"`, `"proportional"` (the default) or a number of at
  least one, the ratio of one rank's weight to the next one's.

- rule:

  For `"committee"`, the rule of
  [`decision_threshold()`](https://gillescolling.com/timesift/reference/kappa_score.md)
  each member's cut is learned by; `"youden"` by default.

## Value

A `timesift_ensemble`.

## Details

`"stack"` fits non-negative weights summing to one on the out-of-fold
predictions alone, never on in-sample ones, minimising the response
head's loss over the scorable cells: binomial deviance for
presence-absence. One weight vector covers every response, because
per-response weights would be fitted on the handful of cells a rare
response has. `"mean"` and `"median"` combine without fitting anything.
`"weighted"` takes each candidate's own mean score, keeps its
non-negative part and rescales those to sum to one, so a candidate
scoring at or below zero is left out and the rest are weighted by how
well they scored; where no candidate scores above zero the scores order
nothing worth weighting by, and every candidate weighs the same.

`decay` changes how `"weighted"` turns scores into weights, as biomod2's
`EMwmean.decay` does. Given a number `d`, the candidates scoring above
zero are ranked from the best down, the one ranked `r` of `K` takes
`d^(K - r + 1)`, candidates on the same score share the mean of their
ranks' weights, and the weights are rescaled to sum to one. Each rank
down weighs `1 / d` of the one above it, whatever the gap in score
between them.

`"committee"` is biomod2's committee averaging. Each member cuts each
response at its own threshold, learned by
[`decision_threshold()`](https://gillescolling.com/timesift/reference/kappa_score.md)
under `rule` from that member's out-of-fold predictions of every target,
and the combined prediction is the share of members voting presence. A
member whose predictions of a response give no cut does not vote on it.
Under
[`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
the thresholds of each outer fold are learned from the inner out-of-fold
predictions of its training targets, as a stack's weights are, so the
estimate never reads a cut chosen on the targets it scores.

`min_score` is biomod2's `metric.select.thresh`: a candidate whose mean
score is below it is not eligible, and the filter is applied before
`scope` picks among what is left.

`scope` says which candidates are eligible. `"all"` is every candidate.
`"learners"` keeps the several learners that read the representation of
the best-scoring candidate, and `"representations"` keeps the one
learner of the best-scoring candidate across the representations it ran
on; both are read off the same mean scores the report shows.

## Examples

``` r
ensemble()
ensemble("weighted", scope = "learners")
ensemble("weighted", decay = 1.6, min_score = 0.4)
ensemble("committee", rule = "kappa")
```
