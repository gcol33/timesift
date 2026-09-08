# Python: fitting

The run from targets and series, the combiner over its candidates, and
fitting across a set of grains on its own.

## `timesift()`

``` python
timesift(
    targets,
    series=None,
    *,
    y,
    x=None,
    id=None,
    time=None,
    target_time=None,
    static=None,
    models=None,
    sift=None,
    ensemble=True,
    resampling=None,
    response: str = 'presence_absence',
    metric=None,
    control=None,
    keep_fits: bool = False,
    verbose: bool = True,
)
```

Fit every learner across every representation, on one fold map, and
combine them.

`targets` is one row per thing to predict and `series` is the long,
time-stamped record belonging to it; both are mappings of column name to
array, which a data frame satisfies. `y`, `x` and `static` are
selections over their own table: a name, a list of names, a glob such as
`"sp_*"`, or a function of a name.

Columns of `targets` that are neither the response nor the identifier
nor the anchor are ignored unless `static` names them: a predictor is
never picked up because it happened to be in the table.

## `Timesift`

``` python
Timesift(
    candidates,
    scores,
    oof,
    representations,
    sift,
    stack,
    weights,
    models,
    folds,
    cells,
    y,
    metric,
    scorer,
    response,
    spec,
    fits,
    control,
)
```

A fitted sift: every candidate’s out-of-fold predictions, their scores,
and the combiner.

`candidates` and `scores` are tables held as columns; `oof`, `models`
and `fits` are keyed by candidate name. What reads them – the summary,
the weights, the occlusion profile – reads them and never a model, so a
number reported here was read where the score was.

Attributes:

- `candidates` - dict
- `scores` - dict
- `oof` - dict
- `representations` - dict
- `sift` - Sift
- `stack` - object
- `weights` - object
- `models` - dict
- `folds` - Folds
- `cells` - object
- `y` - Response
- `metric` - str
- `scorer` - object
- `response` - str
- `spec` - TimesiftSpec
- `fits` - dict
- `control` - object

### `representation_of()`

``` python
representation_of(self, candidate: str)
```

Which representation a candidate reads.

### `predict()`

``` python
predict(self, targets, series=None, candidate: str = 'ensemble')
```

Predict new targets, rebuilding each member’s representation from the
stored settings.

Every candidate is refitted on all the targets at the end of a fit, so
what predicts here is one model per candidate rather than a fold’s worth
of them.

## `TimesiftSpec`

``` python
TimesiftSpec(y, x, id, time, target_time, static, tz, response, metric)
```

How a fit was asked for: the columns each table plays, and the calendar
they are read in.

Attributes:

- `y` - tuple\[str, …\]
- `x` - tuple\[str, …\]
- `id` - str \| None
- `time` - str \| None
- `target_time` - str \| None
- `static` - tuple\[str, …\]
- `tz` - object
- `response` - str
- `metric` - object

## `CandidateFit`

``` python
CandidateFit(learner, fits, variables)
```

One candidate’s fitted models, whichever way its learner covers the
responses.

A joint learner is one model over the whole response matrix and a
separate one is a model per response; either way this predicts the same
`[target, response]` matrix, in the response order the candidate was
fitted on.

Attributes:

- `learner` - object
- `fits` - tuple\[Fit, …\]
- `variables` - tuple\[str, …\]

### `predict()`

``` python
predict(self, x: TimesiftMatrix)
```

One `[row, response]` matrix, whatever the candidate is made of.

A joint learner contributes one fit covering every response and a
separate one contributes a fit per response; the columns are assembled
by name either way, so nothing above a candidate can tell which it was.

## `n_targets()`

``` python
n_targets(targets, spec: TimesiftSpec)
```

How many rows of targets there are, read off a column the spec is sure
of.

## `target_labels()`

``` python
target_labels(targets, spec: TimesiftSpec)
```

What names the rows of every array in one fit.

The unit identifier names them where a unit carries one target. Where
`target_time` lets a unit carry several, no identifier tells them apart,
so the row’s own position does, which is what
`timesift.representation.lookback_matrix` already names its targets by.

## `select_columns()`

``` python
select_columns(columns, spec, arg: str, exclude=())
```

The columns a selection names, in the order the selection names them.

An explicit list is taken in the caller’s order, because naming the
response columns in an order is a decision; a glob and a predicate are
taken in the table’s own order, because matching is not an ordering.

`exclude` names columns that exist but cannot be selected here, so
naming one is refused with the reason rather than reported as a column
that does not exist.

## `column_names()`

``` python
column_names(data)
```

The column names of a table, whether it is a mapping of arrays or a data
frame.

## `summary()`

``` python
summary(fit)
```

The fit as text: one row per candidate, the ensemble under them, and the
weights.

## `candidate_table()`

``` python
candidate_table(fit)
```

One row per candidate: its level, how many responses it was best on, and
how it covered them.

A candidate whose learner and representation could not be paired carries
no level, and is listed under the ones that do.

## `ensemble_row()`

``` python
ensemble_row(fit)
```

The ensemble’s level, read on the same cells and by the same metric as
its members.

## `ensemble()`

``` python
ensemble(
    method: str = 'stack',
    scope: str = 'all',
    metric=None,
    response: str | None = None,
)
```

Ask for an ensemble of the candidates a fit produced.

`stack` fits non-negative weights summing to one on the out-of-fold
predictions, `mean` and `median` combine without fitting, and `weighted`
uses each candidate’s own mean score rescaled to sum to one. `scope` is
which candidates are eligible: every one of them, only the several
learners sharing the best candidate’s representation, or only its
learner across the representations. `metric` names the metric the
ensemble is reported in, or `None` for the fit’s own, and `response` is
the registered head whose loss the weights minimise, or `None` for the
head the run was fitted under. Naming a head the run does not fit toward
is an error rather than an override, and `ensemble_fit` called on its
own reads `None` as `"presence_absence"`.

## `ensemble_fit()`

``` python
ensemble_fit(oof: dict, y, cells, folds, spec=None, scores=None)
```

Fit the combiner on the out-of-fold predictions and nothing else.

`oof` is one `[target, response]` matrix per candidate, in the
response’s own row order. Only the cells the mask admits are read, so
every candidate is weighted on the same cells its score was read on.

## `ensemble_combine()`

``` python
ensemble_combine(stack: Stack, preds: dict)
```

One `[n, response]` matrix from each member’s `[n, response]` matrix.

## `ensemble_weights()`

``` python
ensemble_weights(fit)
```

The weight the combiner gave each of its members, or nothing where a run
combined none.

## `EnsembleSpec`

``` python
EnsembleSpec(method, scope, metric, response)
```

How the candidates are to be combined, which of them are eligible, and
under which head.

Attributes:

- `method` - str
- `scope` - str
- `metric` - object
- `response` - str \| None

## `Stack`

``` python
Stack(method, weights, members)
```

A fitted combiner: what it does and what weight it gave each of its
members.

Attributes:

- `method` - str
- `weights` - dict
- `members` - tuple\[str, …\]

## `grain_ladder()`

``` python
grain_ladder(
    x,
    y,
    learners,
    folds=None,
    response: str = 'presence_absence',
    metric=None,
    keep_fits: bool = False,
    verbose: bool = True,
)
```

Cross-validate every learner at every grain, on one fold map and one
mask of cells.

Every arm sees identical splits and is restricted to identical cells, so
the arms’ means share a denominator and any two of them can be compared
with `paired_contrast`.

`folds` left at `None` builds one with the defaults of `fold_map`. Where
both languages must see the same splits, build it once and read it in
the other with `read_folds`.

## `fit_learner()`

``` python
fit_learner(
    learner,
    x: TimesiftMatrix,
    y,
    response: str = 'presence_absence',
    control=None,
    **kwargs,
)
```

Fit one learner at one grain, under one registered response head.

A `fit` that declares a `head` argument is handed the registered head,
whose `loss` and `activation` say what it is fitting toward; the
learners that ship read both from there and hold no response of their
own. A `fit` that declares a `control` is handed the run’s training
settings the same way.

## `select_grain()`

``` python
select_grain(
    x,
    y,
    learners,
    folds=None,
    inner=5,
    response: str = 'presence_absence',
    metric=None,
    compare: Ladder | None = None,
    seed: int = 1,
    verbose: bool = True,
)
```

Choose the grain inside each outer fold’s training units, then score the
whole procedure.

Within each outer fold the training units are split again, every
candidate is fitted on part of them and scored on the rest, the best is
refitted on the whole outer training set, and the outer test fold is
predicted once. The estimate that comes back is therefore of the
procedure including its choice of grain, which is what an ecologist
applying it to a new site would run.

What the estimate is of: the expected held-out score of the whole
pipeline, selection included, on units drawn as these were. What it is
not: the score of the winning grain. That is higher, by the amount
selection buys itself, and the difference between the two is the
quantity this function exists to keep out of a reported number.

The cost is the ladder’s, multiplied by the number of inner folds:
`v_outer * (v_inner * candidates + 1)` fits.

## `Ladder`

``` python
Ladder(
    grain,
    learner,
    variable,
    fold,
    score,
    scorable,
    predictions,
    cells,
    folds,
    metric,
    scorer,
    response,
    fits,
)
```

One score per `(grain, learner, variable, fold)` cell, and what produced
it.

Attributes:

- `grain` - np.ndarray
- `learner` - np.ndarray
- `variable` - np.ndarray
- `fold` - np.ndarray
- `score` - np.ndarray
- `scorable` - np.ndarray
- `predictions` - dict
- `cells` - object
- `folds` - Folds
- `metric` - str
- `scorer` - object
- `response` - str
- `fits` - dict

### `arm()`

``` python
arm(self, name: str)
```

A mask over the rows of one grain-and-learner arm, named
`grain/learner`.

### `summary()`

``` python
summary(self)
```

The across-variable mean of the per-variable score, one row per arm.

## `Fit`

``` python
Fit(learner, model, variables, response)
```

A fitted learner and the variables it was fitted on.

Attributes:

- `learner` - Learner
- `model` - object
- `variables` - tuple\[str, …\]
- `response` - str

### `predict()`

``` python
predict(self, x: TimesiftMatrix)
```

Predictions for a representation, as a `[unit, variable]` matrix.

## `Selection`

``` python
Selection(selected, estimate, contrast, candidates, scores, inner, metric, response)
```

What a nested selection chose, what it scores, and what it was searched
over.

Attributes:

- `selected` - list\[dict\]
- `estimate` - list\[dict\]
- `contrast` - list\[dict\] \| None
- `candidates` - list\[dict\]
- `scores` - Ladder
- `inner` - list\[dict\]
- `metric` - str
- `response` - str
