# How the candidates are combined

Every candidate emits an out-of-fold prediction for every scorable cell
over the same folds, so the combination is arithmetic on those
predictions and nothing else. `ensemble()` says which arithmetic.

## Usage

``` r
ensemble(
  method = c("stack", "mean", "median", "weighted"),
  scope = c("all", "learners", "representations"),
  metric = NULL,
  response = NULL
)
```

## Arguments

- method:

  How the members are combined.

- scope:

  Which candidates are eligible.

- metric:

  Name of the registered metric the eligibility and the `"weighted"`
  weights are read by, or `NULL` for the score the run already carries.

- response:

  Name of the registered response head whose loss `"stack"` minimises,
  or `NULL` for the head the run was fitted under. Naming a head the run
  does not fit toward is an error rather than an override, and
  [`ensemble_fit()`](https://gillescolling.com/timesift/reference/ensemble_fit.md)
  called on its own reads `NULL` as `"presence_absence"`.

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
well they scored.

`scope` says which candidates are eligible. `"all"` is every candidate.
`"learners"` keeps the several learners that read the representation of
the best-scoring candidate, and `"representations"` keeps the one
learner of the best-scoring candidate across the representations it ran
on; both are read off the same mean scores the report shows.

## Examples

``` r
ensemble()
ensemble("weighted", scope = "learners")
```
