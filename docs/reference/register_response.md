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
  objective; `metric`, the default metric name; and `cells(y, folds)`,
  returning the mask of scorable cells.

- overwrite:

  Replace an existing registration.

## Value

The registered specification, invisibly.

## Examples

``` r
responses()
```
