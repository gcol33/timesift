# Python: learners

The arms that ship, how they are trained, and the interface a learner of
your own goes through.

## `elasticnet()`

``` python
elasticnet(data=None, alpha=0.5, n_inner=5, squares=True, weight_positives=True, seed=1)
```

One penalised regression per variable, over every bin-by-channel column
and, by default, their squares, with the penalty chosen by an inner
cross-validation on the fitting units.

There is no discrete selection step: the penalty path uses every column
and shrinks, and nothing about the model is decided outside the fold it
is fitted in. The family is the response head’s: logistic under a binary
cross-entropy loss, linear under a squared-error one, and
`weight_positives` is read under the first alone.

The design is standardised before it is penalised, as it is on the R
side, so a column is not penalised for the scale it was recorded on. The
penalty itself is the one the inner cross-validation refits at; where R
takes a named point of the path through `s`, scikit-learn keeps only
that one, so there is nothing to name here.

## `stepwise()`

``` python
stepwise(data=None, max_terms=3, degree=2)
```

One generalised linear model per variable, its predictors chosen by
forward selection over every bin-by-channel column, admitting a column
while it lowers Akaike’s criterion and stopping at a fixed budget.

Each candidate enters as an orthogonal polynomial, so a term can be
non-monotone in the reading the way a niche optimum is. Selection
happens inside whichever units the learner is handed, so under
`grain_ladder` it is redone in every fold. Reported beside a penalised
fit it also prices discrete selection: choosing a handful of columns out
of hundreds is high variance, and that variance is a cost of the
selector rather than of the features. The family is the response head’s:
logistic under a binary cross-entropy loss, Gaussian under a
squared-error one.

## `forest()`

``` python
forest(data=None, trees=500, mtry=None, min_node=1, seed=1)
```

One random forest per variable, over every bin-by-channel column: a
classifier under a presence-absence head and a regressor under a head
with a squared-error loss.

`mtry` is how many columns are offered at a split, defaulting to the
square root of how many there are, and `min_node` is the smallest leaf a
split may produce. The forest reads the columns one at a time and
carries no order between them, so it is the arm that asks what the
features hold once nothing about the record’s shape is available to the
model.

## `mlp()`

``` python
mlp(data=None, hidden=(512, 256), dropout=0.3, **settings)
```

Flattens the channels and builds in no temporal geometry.

`hidden` and `dropout` are the architecture; anything else named is a
training setting applied on top of the
[`train_control()`](https://gillescolling.com/timesift/reference/train_control.md)
the learner is fitted under.

## `cnn()`

``` python
cnn(data=None, channels=(16, 32, 64, 128), kernel=7, dropout=0.3, **settings)
```

Convolution, batch normalisation, activation and pooling, then global
average pooling.

## `rescnn()`

``` python
rescnn(
    data=None,
    channels=(32, 64, 128, 256),
    blocks_per_stage=2,
    kernel=7,
    dilations=(1, 2, 4, 8),
    dropout=0.3,
    **settings,
)
```

Dilated residual blocks with channel gates, pooling average and maximum
together.

## `Learner`

``` python
Learner(name, fit, predict, needs, params, data, reads, multi)
```

A name, a fit and a predict, what has to be installed for them to run,
and what the learner reads.

The one interface every arm goes through, the ones that ship and a pair
of your own alike. `data` pins the learner to one representation, or is
`None` to run it across every representation offered. `reads` is whether
it takes a tabular block or an ordered sequence of bins, and `multi` is
whether one fitted model covers every response or one is fitted per
response and the matrix assembled from them.

Attributes:

- `name` - str
- `fit` - Callable
- `predict` - Callable
- `needs` - tuple\[str, …\]
- `params` - dict
- `data` - object
- `reads` - str
- `multi` - str

### `require()`

``` python
require(self)
```

Error, naming the install, unless what the learner needs is importable.

## `train_control()`

``` python
train_control(**settings)
```

The settings every neural learner reads, with anything named here
replacing its default.

`epochs`, `batch_size`, `learning_rate`, `weight_decay`,
`early_stopping`, `val_frac`, `device` and `seed` are the settings a run
is described by. `pos_weight_cap` bounds the weight a rare response’s
presences are given against its absences, at least one, and `swa` with
`swa_start` average the weights over the tail of the schedule rather
than keeping one epoch out of it.

`batch_size` is the most targets an optimiser step reads: the fitting
targets are cut into as few batches of at most that many as they divide
into, of as equal a length as they can be. `val_frac` is held back from
every fit alike, the fit on all targets a run ends with included, one
target from each of as many equal-count strata of the response total as
the set holds.

`device` is `"auto"` for the graphics processor where there is one,
NVIDIA’s or Apple’s, or the name of a device to train on. A fitted
encoder carries the setting rather than the device it resolved to, so a
fit made on one machine predicts on another.

## `TrainControl`

``` python
TrainControl(
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    early_stopping,
    val_frac,
    device,
    seed,
    pos_weight_cap,
    swa,
    swa_start,
)
```

How long to train, on what, and when to stop.

Attributes:

- `epochs` - int
- `batch_size` - int
- `learning_rate` - float
- `weight_decay` - float
- `early_stopping` - int
- `val_frac` - float
- `device` - str
- `seed` - int
- `pos_weight_cap` - float
- `swa` - bool
- `swa_start` - float

### `override()`

``` python
override(self, settings: dict)
```

The control with the settings a learner or a call gave applied on top of
it.

## `flatten()`

``` python
flatten(x: TimesiftMatrix)
```

`[unit, bin, channel]` to `[unit, bin * channel]` in the array’s own
order.

A channel that holds the same number in every bin carries no bin of its
own and is one predictor, read once: repeating it would put the same
column in front of a penalised fit as often as the grain has bins, and
give it that many chances of being drawn by a forest or picked by a
forward search.
