# What a run found

Two kinds of row. A candidate row is the candidate's mean score on the
outer folds, how many responses it scored highest on, and whether one
fitted model covered those responses or one was fitted per response.
These rows are the comparison: they share folds and cells, so their
shape is read across grains and learners, but the highest of them was
picked out on the folds it is scored on. The `selected` and `ensemble`
rows are the procedure's held-out score, the choice and the weights made
inside every outer training fold, and they are the level to quote.

## Usage

``` r
# S3 method for class 'timesift'
summary(object, ...)

# S3 method for class 'timesift'
print(x, ...)

# S3 method for class 'timesift_summary'
print(x, ...)
```

## Arguments

- object:

  A `timesift` result.

- ...:

  Ignored, so that the methods take the arguments their generics
  declare.

- x:

  A `timesift` result, or the table this returns.

## Value

A data frame of class `timesift_summary`, one row per candidate and,
where the run made an estimate, one for the selected candidate and one
for the stack. It carries the mean score, its standard error across
responses for the two procedure rows, the responses won, how the
responses were covered, and `scored`: `"outer folds"` for a candidate,
`"nested"` for the procedure. The weights fitted on every target are in
the `weights` attribute and the candidate chosen on every target in
`choice`.

## Details

Both columns beside a candidate's mean are worth reading. A candidate
can carry the ensemble without winning a single response, which is what
`won` shows and a mean alone hides; and a joint model and a per-response
one reach the same `[target, response]` matrix by different routes,
which is what `responses` records.

A candidate the run built no representation for, because its learner
cannot read the representation it was paired with, is listed with no
mean rather than dropped, so the report says what was asked for as well
as what ran.
