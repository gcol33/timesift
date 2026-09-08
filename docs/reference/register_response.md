# Register a response head

A response head says what the values being predicted are, how they reach
a learner, and which cells of the (variable, fold) grid a score is
defined on. Presence-absence with a joint multi-label head is what
ships; an abundance or phenology response is a registration rather than
a second fitting path.

## Usage

``` r
register_response(name, spec, overwrite = FALSE)

responses()
```

## Arguments

- name:

  Name the response is asked for by.

- spec:

  A list with elements `prepare(y)`, returning the numeric matrix a
  learner is fitted on; `activation`, the name of the output transform
  (`"sigmoid"` or `"identity"`); `loss`, the name of the training
  objective (`"binary_cross_entropy"` or `"squared_error"`); `metric`,
  the default metric name; and `cells(y, folds)`, returning the mask of
  scorable cells. Every learner that ships reads `loss` and `activation`
  from here: the encoders train under the loss and predict through the
  activation, and the learners fitting one model per response take the
  family the loss names, logistic or Gaussian. The combiner minimises
  the same loss. An optional `weights(y)` returns a `[unit, variable]`
  matrix of case weights every learner fits under; the shipped head's is
  [`positive_weights()`](https://gillescolling.com/timesift/reference/positive_weights.md),
  and a head without one fits unweighted.

- overwrite:

  Replace an existing registration.

## Value

The registered specification, invisibly.

## Examples

``` r
responses()
```
