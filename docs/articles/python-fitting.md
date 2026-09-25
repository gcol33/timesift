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
    inner=5,
    rule: str = 'argmax',
    response: str = 'presence_absence',
    metric=None,
    control=None,
    keep_fits: bool = False,
    seed: int = 1,
    verbose: bool = True,
)
```

Compare every learner across every representation, and estimate choosing
among them.

`targets` is one row per thing to predict and `series` is the long,
time-stamped record belonging to it; both are mappings of column name to
array, which a data frame satisfies. `y`, `x` and `static` are
selections over their own table: a name, a list of names, a glob such as
`"sp_*"`, or a function of a name.

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
of them. `"ensemble"` combines them under the weights fitted on every
target, `"selected"` predicts with the candidate the rule chose on every
target (`choice`), and any other value names one candidate.

## `TimesiftSpec`

``` python
TimesiftSpec(y, x, id, time, target_time, static, response, metric)
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

The weights are fitted to the response on those predictions, so the
combination scored against the same response is scored on the data its
weights were fitted to, and that score is optimistic. `timesift`
evaluates the stack the other way: each outer fold’s weights are fitted
on inner out-of-fold predictions of its training targets and applied to
the outer test fold.

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
    control=None,
    keep_fits: bool = False,
    interval: str = 'variables',
    repeats: int = 1,
    seed: int = 1,
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

`control` is the `train_control` every neural learner of the ladder
trains under; a learner carrying settings of its own overrides it on the
ones it names.

`interval="nested_cv"` refits every arm inside every outer training set
of every repetition, which is what
`paired_contrast(interval="nested_cv")` reads an interval for the
difference in risk off. Two tables whose contrast is to be read take the
same `folds`, `repeats` and `seed`.

## `fit_learner()`

``` python
fit_learner(
    learner,
    x: TimesiftMatrix,
    y,
    response: str = 'presence_absence',
    control=None,
    group=None,
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
    rule: str = 'argmax',
    threshold: str | None = None,
    interval: str = 'variables',
    repeats: int = 1,
    response: str = 'presence_absence',
    metric=None,
    compare: Ladder | None = None,
    control=None,
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

`control` is the `train_control` every neural learner trains under, in
the inner search and in the refit alike; a learner carrying settings of
its own overrides it on the ones it names.

Inside each outer fold every candidate carries an inner score, the mean
over variables of its per-variable mean over the inner folds, and a
standard error, the standard deviation over the inner folds of the
fold’s own score divided by the square root of their number. `rule`
chooses among them. `"argmax"` takes the highest score, and on an exact
tie the candidate declared first. `"coarsest_adequate"` is the
one-standard-error rule (Breiman, Friedman, Olshen and Stone 1984;
Hastie, Tibshirani and Friedman 2009, section 7.10) with coarseness in
place of complexity: every candidate scoring at least the highest minus
its standard error is adequate, and the one with the fewest bins wins,
then the fewest channels, then the higher score, then the one declared
first. A standard error that cannot be computed is taken as zero.

`threshold` names a rule of `decision_threshold` (`"youden"`, the cut
that maximises TSS, `"kappa"` or `"prevalence"`). With it set, each
outer fold learns one cut per variable on the inner out-of-fold
predictions of the candidate it selected, which cover the outer training
units and nothing else, freezes it, and reads the outer test fold’s
predictions at it with `tss`. The estimate then carries a row
`tss_inner_cut`, `thresholds` the cut of every outer fold and variable,
and `cut_scores` the per-cell rows.

The estimate carries the interval across the response variables, which
is the spread between the variables of this dataset rather than an
interval for what the procedure would score on a new sample.
`interval="nested_cv"` adds one that is meant to be, by the nested
cross-validation of Bates, Hastie and Tibshirani (2024): each outer
training set is cross-validated again over the remaining folds of the
same map, over `repeats` fold maps, which gives the mean squared error
of a cross-validation estimate, and the centre carries the paper’s bias
correction, so `final` holds the procedure fitted on every unit, whose
risk the interval is for. The width departs from the paper in one
respect: the paper’s is the mean squared error of the plain estimate and
takes the correction as a shift with no spread, and in the package’s
benchmark that spread exceeded the estimate’s where the sample was small
or the signal absent; the width here is the same identity applied to the
corrected estimator, from a cross-validation one level further down,
held above the corrected centre’s own naive standard error, and the
paper’s is kept beside it as `se_bates`. One repetition costs one fit of
the procedure per unordered pair and per unordered triple of outer
folds.

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
    ncv,
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
- `ncv` - dict \| None

### `arm()`

``` python
arm(self, name: str)
```

A mask over the rows of one grain-and-learner arm, named
`grain|learner`.

### `summary()`

``` python
summary(self)
```

The across-variable mean of the per-variable score, one row per arm.

## `Fit`

``` python
Fit(learner, model, variables, response, bins, channels)
```

A fitted learner, the variables it was fitted on, and the bins and
channels of the representation it was made on.

A representation asked to predict is checked against those once, before
any learner sees it: a calendar grain’s bins are named by their starts,
so a record from another period is refused by the first bin that differs
rather than read by position.

Attributes:

- `learner` - Learner
- `model` - object
- `variables` - tuple\[str, …\]
- `response` - str
- `bins` - tuple\[str, …\]
- `channels` - tuple\[str, …\]

### `predict()`

``` python
predict(self, x: TimesiftMatrix)
```

Predictions for a representation, as a `[unit, variable]` matrix.

## `Selection`

``` python
Selection(
    selected,
    estimate,
    contrast,
    candidates,
    scores,
    inner,
    metric,
    response,
    rule,
    threshold,
    thresholds,
    cut_scores,
    interval,
    nested_cv,
    final,
)
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
- `rule` - str
- `threshold` - str \| None
- `thresholds` - list\[dict\] \| None
- `cut_scores` - Ladder \| None
- `interval` - str
- `nested_cv` - list\[dict\] \| None
- `final` - dict \| None

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
