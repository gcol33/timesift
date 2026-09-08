# What a run found

The mean score of every candidate, how many responses each of them
scored highest on, whether one fitted model covered those responses or
one was fitted per response, and the level the combined prediction
reached with the weights it reached it under.

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

A data frame of one row per candidate and one for the ensemble, of class
`timesift_summary`, carrying the mean score, the responses won and how
the responses were covered. The weights are in the `weights` attribute.

## Details

Both columns beside the mean are worth reading. A candidate can carry
the ensemble without winning a single response, which is what `won`
shows and a mean alone hides; and a joint model and a per-response one
reach the same `[target, response]` matrix by different routes, which is
what `responses` records.

A candidate the run built no representation for, because its learner
cannot read the representation it was paired with, is listed with no
mean rather than dropped, so the report says what was asked for as well
as what ran.
