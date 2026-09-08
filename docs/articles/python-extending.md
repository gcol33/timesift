# Python: extending

The response head and the metric are registrations, never a fork of the
fitting code.

## `register_learner()`

``` python
register_learner(name: str, constructor: Callable, overwrite: bool = False)
```

Make a learner available by name. The learners that ship are registered
the same way.

`constructor` is called with no arguments and returns a `Learner`, so a
learner asked for by name is built with its own defaults.

## `register_metric()`

``` python
register_metric(name: str, fn: Callable, overwrite: bool = False)
```

Register a metric: a function of `(y, p)` on one held-out cell.

`y` and `p` are the observed values of the units in one fold and a
model’s predictions for them, both the same length. It returns one
number, or a value that is not finite where the cell defines none.
Registering one makes it available to `grain_ladder` by name, with no
change to the fitting code.

## `register_response()`

``` python
register_response(name: str, spec: dict, overwrite: bool = False)
```

Register a response head: what the values being predicted are and where
a score is defined.

`spec` is a mapping with `prepare(y)`, returning the response a learner
is fitted on; `activation`, the name of the output transform
(`"sigmoid"` or `"identity"`); `loss`, the name of the training
objective (`"binary_cross_entropy"` or `"squared_error"`); `metric`, the
default metric name; and `cells(y, folds)`, returning the mask of
scorable cells. Presence-absence with a joint multi-label head is what
ships; an abundance or phenology response is a registration rather than
a second fitting path. Every learner that ships reads `loss` and
`activation` from here: the encoders train under the loss and predict
through the activation, and the learners fitting one model per response
take the family the loss names, logistic or Gaussian. The combiner
minimises the same loss.

## `learners()`

``` python
learners()
```

The learners registered under this session.

## `metrics()`

``` python
metrics()
```

The metrics registered under this session.

## `responses()`

``` python
responses()
```

The response heads registered under this session.

## `get_learner()`

``` python
get_learner(learner)
```

A `Learner`, whether it arrived as one or as the name of a registered
one.
