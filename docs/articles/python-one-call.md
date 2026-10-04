# Python: the one call

Two tables to a scored comparison of representations, and the prediction
that follows from it.

[All of the Python
reference](https://gillescolling.com/timesift/articles/python-reference.md)

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
    coords=None,
    models=None,
    sift=None,
    ensemble=True,
    resampling=None,
    inner=5,
    rule: str = 'argmax',
    response: str = 'presence_absence',
    metric=None,
    control=None,
    keep_fits: bool = False,
    seed: int = 1,
    verbose: bool = True,
    refit: bool = True,
)
```

Compare every learner across every representation, and estimate choosing
among them.

`targets` is one row per thing to predict and `series` is the long,
time-stamped record belonging to it; both are mappings of column name to
array, which a data frame satisfies. `y`, `x`, `static` and `coords` are
selections over their own table: a name, a list of names, a glob such as
`"sp_*"`, or a function of a name. `coords` names the two columns
holding each target’s coordinates; they place a target and are not
predictors, and a learner that places a spatial field by them,
`timesift.hierarchical`, reads them off the array.

Within each outer fold of `resampling` the training targets are split
again into `inner` folds. Every candidate is cross-validated on that
inner split, `rule` picks one on its inner score (`"argmax"` or
`"coarsest_adequate"`, as in `select_grain`), and the stack’s weights
are fitted on the inner out-of-fold predictions. Every candidate is then
refitted on the whole outer training set and predicts the outer test
fold, and the selected candidate’s prediction and the prediction
combined under that fold’s weights are kept. Nothing the outer test fold
holds enters the choice or the weights it is scored under, so `estimate`
is of the procedure, selection and stacking included. Its interval is
across the response variables of this dataset, all fitted and scored on
the same targets and folds, and not one for a new sample.

The same refits give every candidate an out-of-fold prediction on the
outer folds, which `scores` holds: the comparison, whose highest level
was picked out on the folds it is scored on. `inner=None` runs no inner
search and makes no estimate. `choice`, `models` and `stack` are the
procedure applied to every target, for prediction: the rule read on the
outer scores and weights fitted on the outer out-of-fold predictions.

Columns of `targets` that are neither the response nor the identifier
nor the anchor are ignored unless `static` names them: a predictor is
never picked up because it happened to be in the table.

With `repeats` above one in `resampling` the run is made once per
repeat: `scores` carries a `repeat` array and every repeat’s folds under
numbers of their own, `estimate` is read off all the repeats’ held-out
predictions, `oof` is a target’s mean prediction over the repeats, and
the stack is fitted on the repeats’ out-of-fold predictions together.
The models, the per-fold fits and `folds` are the first repeat’s.
`refit` says whether the candidates are refitted on all targets at the
end, which is what `predict` reads; a repeated resampling asks it of its
first run alone.

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
    estimate,
    selected,
    inner,
    fold_weights,
    predictions,
    choice,
    repeats,
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
- `estimate` - list \| None
- `selected` - list \| None
- `inner` - list \| None
- `fold_weights` - list \| None
- `predictions` - dict \| None
- `choice` - str \| None
- `repeats` - int

### `representation_of()`

``` python
representation_of(self, candidate: str)
```

Which representation a candidate reads.

### `predict()`

``` python
predict(
    self,
    targets,
    series=None,
    candidate: str = 'ensemble',
    type: str = 'response',
    rule: str = 'youden',
    alpha: float = 0.05,
)
```

Predict new targets, rebuilding each member’s representation from the
stored settings.

Every candidate is refitted on all the targets at the end of a fit, so
what predicts here is one model per candidate rather than a fold’s worth
of them. `"ensemble"` combines them under the weights fitted on every
target, `"selected"` predicts with the candidate the rule chose on every
target (`choice`), and any other value names one candidate.

`type="binary"` cuts each response into presence and absence at the
threshold
[`decision_threshold()`](https://gillescolling.com/timesift/reference/kappa_score.md)
learns by `rule` from the same candidate’s out-of-fold predictions of
the fit’s own targets, and returns 0.0 and 1.0, NaN for a response whose
held-out predictions give no cut. A binary map of a species is this
prediction on one target per map cell.

`type="spread"` reads the ensemble’s members side by side rather than
combining them:
[`ensemble_spread()`](https://gillescolling.com/timesift/reference/ensemble_spread.md)
gives their weighted mean, standard deviation, coefficient of variation
and interval at `alpha` as an `[n, response, statistic]` array,
biomod2’s `EMcv` and `EMci`, and on one target per map cell an
uncertainty map.

## `TimesiftSpec`

``` python
TimesiftSpec(y, x, id, time, target_time, static, response, metric, coords)
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
- `response` - str
- `metric` - object
- `coords` - tuple\[str, …\]

## `summary()`

``` python
summary(fit)
```

The fit as text: the candidates on the outer folds, the procedure, and
the weights.

## `candidate_table()`

``` python
candidate_table(fit)
```

One row per candidate: its level, how many responses it was best on, and
how it covered them.

A candidate whose learner and representation could not be paired carries
no level, and is listed under the ones that do.

## `procedure_table()`

``` python
procedure_table(fit)
```

The selected candidate’s and the stack’s held-out level under the run’s
own metric, with the standard error across responses: one row each, or
none where the run made no estimate.

## `plot()`

``` python
plot(x, col=None, interval: bool = True, ax=None, **kwargs)
```

Draw a ladder, a run or a selection, and return the table the drawing
was made from.

A ladder is drawn as one line per learner across the grains, at the
across-variable mean of the per-variable score, with a 95 percent
interval from its standard error across variables on Student’s t with
one degree of freedom fewer than there are variables; an open circle
marks each learner’s best grain. A run is drawn the same way across the
representations, with the stack’s held-out score across them as a dashed
line: the curves are scored on the folds a choice among them would be
judged on, so the best of them sits a little high, and the stack’s line
does not. A selection is drawn as every candidate’s inner score in every
outer fold, one line per fold, an open circle on the candidate the fold
chose.

`col` is one colour per line, recycled. `ax` is the matplotlib axes to
draw on, a new figure’s where it is left unset, and `kwargs` reach
`ax.set()`, so `title=` or `ylim=` are set as they would be there.
`interval` draws the interval on a ladder or a run and is ignored on a
selection, which draws none. Needs matplotlib.

## `project()`

``` python
project(
    fit,
    series=None,
    static=None,
    candidate: str = 'ensemble',
    type: str = 'response',
    chunk: int = 5000,
    **kwargs,
)
```

Predict one target per cell of a grid, each carrying the record of its
own cell.

This is `BIOMOD_Projection()` and `BIOMOD_EnsembleForecasting()`: a map
is the fit applied to one target per cell, at the grain the fit reads.
`series` is an `xarray.DataArray` with a `time` dimension and two
spatial ones, or a mapping of the fit’s `x` column names to such;
`static` is an `xarray.Dataset`, or a mapping of the fit’s `static`
column names to `DataArray` over the two spatial dimensions. Every input
shares one grid, and one set of instants. Cells are predicted in chunks
of `chunk`, and a cell is predicted where every input holds a value at
that cell: a cell with a missing reading anywhere is NaN in every layer.
`candidate`, `type` and the other keywords are those of
`Timesift.predict`.

Returns an `xarray.DataArray` over `response` and the two spatial
dimensions of the inputs; for `type="spread"` a `statistic` dimension as
well.

## `range_change()`

``` python
range_change(now, later, threshold=None)
```

How the range of each response changes between two maps.

Counts the cells a response is lost from, kept in and gained, as
`BIOMOD_RangeSize()` does, a cell being in the range where the response
is predicted present. With `L` the cells lost, `K` kept, `G` gained and
`A` absent in both: the current range is `L + K`, the later one `K + G`,
`percent_loss` is `100 L / (L + K)`, `percent_gain` is `100 G / (L + K)`
and `change` is `percent_gain - percent_loss`. A cell that is NaN in
either map is left out of every count. `now` and `later` are
`xarray.DataArray` with a `response` dimension, such as
[`project()`](https://gillescolling.com/timesift/reference/project.md)
returns, or arrays of cells by responses. `threshold` is None where the
maps are already 0 and 1, and otherwise one cut per response, or one for
all.

## `RangeChange`

``` python
RangeChange(table, map, variables)
```

The change in range of each response: a `table` of counts, and the `map`
of codes (-2 lost, -1 kept, 0 absent in both, 1 gained) of the shape of
the inputs.

Attributes:

- `table` - list
- `map` - np.ndarray
- `variables` - tuple

## `pseudo_absences()`

``` python
pseudo_absences(
    pool,
    presences,
    n: int,
    strategy: str = 'random',
    id=None,
    env=None,
    quantile: float = 0.025,
    coords=None,
    dist_min: float = 0.0,
    dist_max: float = np.inf,
    lonlat: bool = False,
    repeats: int = 1,
    seed: int = 1,
)
```

Draw pseudo-absences from a pool of background units.

Presence-only records give a response of ones. A model needs units where
the response is zero, and these are drawn from `pool`, a mapping of
column to array of background units that carry the same columns as the
targets. The strategies are `bm_PseudoAbsences()`’s: `"random"` admits
any unit of the pool that is not a presence; `"sre"` a unit outside the
envelope of the presences, the band between the `quantile` and
`1 - quantile` quantiles of each of the `env` columns over the
presences, a unit being outside where it leaves the band in at least one
column; `"disk"` a unit whose distance to the nearest presence lies
between `dist_min` and `dist_max`, both included, on the two `coords`
columns, planar in the units of the coordinates or in metres from
longitude and latitude in degrees where `lonlat` is true, on a sphere of
radius 6371008.8 m.

A presence is never its own absence: a unit of the pool whose `id` is
one of the presences’ is not a candidate. The units a strategy admits
are the same in both languages; which `n` of them are drawn depends on
the language’s generator, as the folds of `fold_map` do. The result is a
mapping of column to array holding the drawn rows of the pool, with
`set` (the draw, from 1) and `pseudo` (true) added; draw `r` uses
`seed + r - 1`.

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
