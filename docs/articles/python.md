# Choosing how a record is read, in Python

A sensor records every hour for years. Before any model sees it, the
record is reduced: to monthly means, to growing-degree-days, to whatever
the analyst settled on once. `timesift` makes that reduction an
argument, fits at each setting of it, and shows how much of the record
the prediction actually needed.

This article runs the whole path on a small simulated record, from two
tables to a scored comparison and the prediction that follows from it.
It follows [the R
walkthrough](https://gillescolling.com/timesift/articles/timesift.md)
section by section. The binning, the statistics and the penalised fit
under both languages are one compiled core, held to the same numbers by
[the contract](https://gillescolling.com/timesift/articles/contract.md).

``` bash
pip install timesift
```

## Two tables

`targets` is one row per thing to predict. `series` is the long,
time-stamped record belonging to those rows, and the two are linked by
an identifier both carry. Either is a mapping of column name to array,
which a pandas or polars data frame is.

``` python
import numpy as np
import pandas as pd
import timesift as ts

rng = np.random.default_rng(1)
hours = pd.date_range("2021-09-01", periods=24 * 120, freq="h")
ids = [f"p{i:03d}" for i in range(1, 61)]
warmth = rng.normal(size=60)
season = 6 * np.sin(np.arange(1, len(hours) + 1) / (24 * 40))

series = pd.DataFrame({
    "plot": np.repeat(ids, len(hours)),
    "t": np.tile(hours, 60),
    "temp": np.concatenate([1.5 * w + season + rng.normal(scale=12, size=len(hours))
                            for w in warmth]),
})

targets = pd.DataFrame({"plot": ids, "elevation": 2000 + 300 * rng.normal(size=60)})
for k, sign in enumerate([1, -1, 1, -1, 1, -1], start=1):
    targets[f"sp{k}"] = rng.binomial(1, 1 / (1 + np.exp(-3 * sign * warmth)))
targets.iloc[:, :4].head()
#>    plot    elevation  sp1  sp2
#> 0  p001  2159.960722    1    0
#> 1  p002  1516.179723    1    0
#> 2  p003  1818.289129    1    0
#> 3  p004  2233.373010    0    1
#> 4  p005  1779.544877    1    0
```

Each plot has a level of its own, and that level is buried in
hour-to-hour noise an order of magnitude larger.

## One call

``` python
fit = ts.timesift(
    targets, series,
    y="sp*",
    id="plot",
    time="t",
    static="elevation",
    sift=ts.grains("day", "week", "month"),
    resampling=ts.cv(v=5),
    verbose=False,
)
fit
#> timesift  60 targets, 6 responses, 5-fold random CV, roc_auc
#>
#> candidates, scored on the outer folds
#> candidate                    mean    won  responses
#> elasticnet / day            0.879      1  separate
#> elasticnet / week           0.906      1  separate
#> elasticnet / month          0.910      4  separate
#>
#> procedure, chosen and weighted inside each outer training fold
#> selected                    0.912  se 0.014
#> ensemble                    0.916  se 0.012
#> selected elasticnet / month in 4, elasticnet / week in 1 of 5 folds
#>
#> choice on every target  elasticnet / month
#> weights on every target  elasticnet / month 0.72   elasticnet / week 0.28
```

Every representation named in `sift` was built, and with `models` left
at its default of `[ts.elasticnet()]`, a penalised logistic regression
was fitted on each of them over the same five folds. The report has two
parts.

The candidates are the comparison. Each is scored on the five outer
folds, on the same cells, so their means can be read against each other
across grains. `won` is how many responses a candidate scored highest
on, and `responses` says whether one fitted model covered them all or
one was fitted per response. The highest of those means was picked out
on the folds it is scored on, so it is not the number to report.

The procedure rows are. Inside each outer training fold the candidates
were cross-validated again on five inner folds, one was chosen on its
inner score and the stack’s weights were fitted on the inner out-of-fold
predictions; the choice and the weights then predicted the outer test
fold once. `selected` and `ensemble` are those held-out scores, choosing
and weighting included.

``` python
pd.DataFrame(fit.estimate)[["arm", "metric", "score", "se", "lower", "upper"]]
#>          arm                metric      score        se      lower      upper
#> 0   selected              accuracy   0.902778  0.011719   0.872652   0.932904
#> 1   selected     average_precision   0.912736  0.023887   0.851332   0.974140
#> 2   selected                  bias   0.987235  0.046206   0.868460   1.106011
#> 3   selected                 boyce   0.661197  0.028310   0.588424   0.733970
#> 4   selected                   csi   0.817593  0.025308   0.752536   0.882649
#> 5   selected                   ets   0.682243  0.036524   0.588354   0.776131
#> 6   selected                   far   0.081706  0.027808   0.010224   0.153189
#> 7   selected                 kappa   0.686667  0.036028   0.594054   0.779279
#> 8   selected          kappa_youden   0.797754  0.024534   0.734687   0.860820
#> 9   selected               neg_mae  -0.280351  0.015292  -0.319660  -0.241042
#> 10  selected         neg_max_error  -0.735800  0.012070  -0.766827  -0.704773
#> 11  selected               neg_mse  -0.131604  0.008301  -0.152942  -0.110266
#> 12  selected  neg_poisson_deviance  -0.443547  0.028867  -0.517753  -0.369342
#> 13  selected              neg_rmse  -0.358039  0.012266  -0.389570  -0.326508
#> 14  selected                    or  24.000000  0.577350  21.515862  26.484138
#> 15  selected      ordinal_accuracy   0.816667  0.012172   0.785379   0.847955
#> 16  selected            ordinal_f1   0.826092  0.010342   0.799507   0.852677
#> 17  selected     ordinal_precision   0.820331  0.010924   0.792248   0.848413
#> 18  selected        ordinal_recall   0.832712  0.010828   0.804879   0.860545
#> 19  selected                  orss   0.986508  0.006571   0.969618   1.003399
#> 20  selected               pearson   0.707152  0.023807   0.645953   0.768350
#> 21  selected                   pod   0.889378  0.019559   0.839101   0.939655
#> 22  selected                  pofd   0.069616  0.025054   0.005214   0.134019
#> 23  selected             r_squared   0.419123  0.034854   0.329528   0.508718
#> 24  selected               roc_auc   0.911593  0.014499   0.874321   0.948865
#> 25  selected                    sr   0.918294  0.027808   0.846811   0.989776
#> 26  selected                   tss   0.819762  0.023356   0.759723   0.879801
#> 27  ensemble              accuracy   0.897222  0.009044   0.873974   0.920470
#> 28  ensemble     average_precision   0.918701  0.018726   0.870564   0.966838
#> 29  ensemble                  bias   1.044537  0.044212   0.930887   1.158188
#> 30  ensemble                 boyce   0.687740  0.029985   0.610661   0.764820
#> 31  ensemble                   csi   0.813479  0.018509   0.765900   0.861057
#> 32  ensemble                   ets   0.666883  0.028104   0.594640   0.739126
#> 33  ensemble                   far   0.109749  0.027043   0.040232   0.179266
#> 34  ensemble                 kappa   0.663810  0.037109   0.568417   0.759202
#> 35  ensemble          kappa_youden   0.786277  0.019087   0.737211   0.835343
#> 36  ensemble               neg_mae  -0.282437  0.014369  -0.319374  -0.245499
#> 37  ensemble         neg_max_error  -0.739320  0.008787  -0.761909  -0.716732
#> 38  ensemble               neg_mse  -0.131707  0.006787  -0.149153  -0.114261
#> 39  ensemble  neg_poisson_deviance  -0.444870  0.024573  -0.508036  -0.381704
#> 40  ensemble              neg_rmse  -0.359284  0.009496  -0.383695  -0.334874
#> 41  ensemble                    or  21.000000  2.846050  13.098099  28.901901
#> 42  ensemble      ordinal_accuracy   0.822222  0.014055   0.786094   0.858351
#> 43  ensemble            ordinal_f1   0.826303  0.015320   0.786922   0.865684
#> 44  ensemble     ordinal_precision   0.819517  0.014959   0.781065   0.857970
#> 45  ensemble        ordinal_recall   0.833929  0.016250   0.792157   0.875700
#> 46  ensemble                  orss   0.983114  0.004731   0.970953   0.995276
#> 47  ensemble               pearson   0.709169  0.020091   0.657523   0.760815
#> 48  ensemble                   pod   0.911918  0.012180   0.880608   0.943228
#> 49  ensemble                  pofd   0.102474  0.025343   0.037327   0.167620
#> 50  ensemble             r_squared   0.419408  0.025940   0.352727   0.486090
#> 51  ensemble               roc_auc   0.915733  0.011500   0.886170   0.945296
#> 52  ensemble                    sr   0.890251  0.027043   0.820734   0.959768
#> 53  ensemble                   tss   0.809444  0.017458   0.764568   0.854321
```

``` python
pd.DataFrame(fit.selected)[["fold", "candidate", "inner_score"]]
#>    fold           candidate  inner_score
#> 0     1  elasticnet / month     0.882041
#> 1     2  elasticnet / month     0.897021
#> 2     3  elasticnet / month     0.887111
#> 3     4  elasticnet / month     0.902577
#> 4     5   elasticnet / week     0.888307
```

The interval is across the six responses of this dataset, all fitted and
scored on the same plots and folds, so it does not carry the error those
share.

A column of `targets` reaches the model only where `static` names it:
`elevation` is a predictor here because it was asked for, and a column
of notes sitting beside it would not be.

``` python
pd.DataFrame(fit.candidates)[["candidate", "representation", "bins", "channels"]]
#>             candidate representation  bins  channels
#> 0    elasticnet / day            day   120         2
#> 1   elasticnet / week           week    18         2
#> 2  elasticnet / month          month     4         2
```

Two channels at every grain: the binned temperature, and elevation held
constant across the bins.

``` python
ts.plot(fit);
```

![Mean AUC against representation, with the level the combination
reached drawn across it.](python_files/figure-1.svg)

[`predict()`](https://rdrr.io/r/stats/predict.html) rebuilds each
member’s representation for new rows from the settings its own arm was
built with, and combines them through the ensemble refitted on every
target; `candidate="selected"` predicts with the candidate the rule
chose on every target instead.

``` python
p = fit.predict(targets, series)
p[:3, :4].round(3)
#> array([[0.588, 0.362, 0.638, 0.16 ],
#>        [0.841, 0.063, 0.818, 0.062],
#>        [0.731, 0.153, 0.701, 0.313]])
```

`type="binary"` cuts each response into presence and absence. The cut is
learned from the same candidate’s out-of-fold predictions of the fit’s
own targets, so it comes from predictions of plots the model had not
been fitted on. `ts.decision_threshold()` on the fit returns the cuts
themselves, and `rule` picks how they are learned: `"youden"` maximises
sensitivity plus specificity, `"kappa"` maximises Cohen’s kappa, and
`"prevalence"` predicts as many presences as were observed.

``` python
pd.Series(ts.decision_threshold(fit, rule="prevalence"))
#> sp1    0.511295
#> sp2    0.498009
#> sp3    0.543913
#> sp4    0.448653
#> sp5    0.494650
#> sp6    0.465196
#> dtype: float64
```

``` python
fit.predict(targets, series, type="binary", rule="prevalence")[:3, :4]
#> array([[1., 0., 1., 0.],
#>        [1., 0., 1., 0.],
#>        [1., 0., 1., 0.]])
```

A binary map is this prediction with one target per map cell, each
carrying a series of its own at the grain the fit reads.

## Representations

A representation carries the settings and nothing else, so the same
object describes an arm before any record has been read and rebuilds
itself for new targets afterwards.

``` python
ts.native()
#> <timesift representation> native
#> kind    : grain (a sequence)
#> grain   : native
#> stats   : mean
ts.grain("week", stats=("cold_day", "mean", "warm_day"))
#> <timesift representation> week
#> kind    : grain (a sequence)
#> grain   : week
#> stats   : cold_day, mean, warm_day
ts.multigrain(("week", "month"))
#> <timesift representation> multigrain(week+month)
#> kind    : multigrain (a block of features)
#> grains  : week, month
#> stats   : mean
ts.lookback("30 days", bins=3)
#> <timesift representation> 30 days x3
#> kind    : lookback (a sequence)
#> span    : 30 days in 3 bins ending 0 days before the target
#> stats   : mean
```

`ts.grains()` and `ts.lookbacks()` are sets of them, and
`ts.grains("auto")` reads off the record every named grain it gives at
least two bins. `ts.multigrain()` flattens several grains side by side
into one block of features; `ts.lookback()` is a fixed span ending at
each target’s own instant, which is what a plot carrying several targets
through time needs, given as `target_time`.

A learner runs across the whole set, or at one representation it is
pinned to:

``` python
ts.elasticnet(data=ts.grain("month"))
#> <timesift learner> elasticnet
#> reads   : tabular ; one model per response: yes, separate
#> data    : month
#> settings: alpha = 0.5, n_inner = 5, squares = True, s = lambda.min, n_lambda = 100, thresh = 1e-08, threads = 1, seed = 1
```

## What a learner may be handed

A learner declares whether the bins reach it as a block of predictors or
as a sequence whose order in time is what it reads. A tabular learner
given `ts.native()` is refused before any record is touched, since
building the array it would have been handed is the expensive half of
the call; a sequence learner is refused a representation that turns out
to hold one bin, once the array says how many it has.

``` python
ts.timesift(targets, series, y="sp*", id="plot", time="t",
            models=[ts.elasticnet()], sift=ts.native(), verbose=False)
#> ValueError: no learner can read any of the representations:
#>   elasticnet() reads a tabular representation; native gives it one column per reading. Use grain(), multigrain() or lookback().
```

Inside a set such a pair is skipped and listed as `not applicable` by
`ts.summary()`; named through a learner’s own `data=` it is an error,
because a representation named by hand is a decision.

## The split, and the cells a score is defined on

One fold map is read by everything that scores, so every candidate is
fitted and scored on identical splits. `ts.cv()` deals units into folds
balanced on a stratifying value and `ts.grouped_cv()` keeps every target
sharing a group value on one side of each split; `resampling` also takes
a fold vector or a `ts.fold_map()` result directly.

``` python
fit.folds
#> <timesift folds> 60 units in 5 folds
#> 1: 12  2: 12  3: 12  4: 12  5: 12
```

A per-response score needs both classes among the held-out units, and a
per-response model needs both classes among the units it was fitted on.
The mask says which cells those are, from the response and the fold map
alone.

``` python
fit.cells
#> <timesift cells> 30 cells over 6 variables
#> scorable: 30 (100.0%); variables with at least one scorable fold: 6 of 6
```

Because it involves no model, every candidate is restricted to the same
cells: their means share one denominator and every paired difference
runs on matched cells.

## Learners, and how they are trained

`ts.elasticnet()` and `ts.linear()` read a block of features,
`ts.forest()` grows a probability forest over one, and the `torch`
encoders `ts.mlp()`, `ts.cnn()` and `ts.rescnn()` read a sequence with a
joint multi-label head, so every response is predicted together from a
shared embedding. Pooling strength across responses is what makes the
rarer ones learnable at these sample sizes.

`ts.elasticnet()` is the arm a network is measured against, so it is
fitted by the same compiled core the R package calls rather than by a
fitter on either side: the penalty path, the standardisation and the
cross-validated choice of penalty are one implementation, and the two
languages return the same coefficients for the same design. `s` reads
that fit at `"lambda.min"`, at `"lambda.1se"`, or at a penalty of your
own, and `thresh` trades how close the descent settles to the optimum
against what it costs.

Architecture belongs to the constructor and training belongs to
`ts.train_control()`, which every neural learner of a run reads. A
learner given a control of its own overrides the run’s on the settings
it names and takes the rest from it.

``` python
ts.train_control(epochs=200, device="cpu")
#> <timesift control>
#>   epochs          200
#>   batch_size      64   (default)
#>   learning_rate   0.001   (default)
#>   weight_decay    0.0001   (default)
#>   early_stopping  10   (default)
#>   val_frac        0.0   (default)
#>   device          cpu
#>   seed            1   (default)
#>   swa             False   (default)
#>   swa_start       0.7   (default)
ts.cnn(channels=(16, 32), epochs=300)
#> <timesift learner> cnn
#> reads   : sequence ; one model per response: no, joint
#> data    : every representation of the run
#> settings: channels = 16/32, kernel = 7, dropout = 0.3
#> training: epochs = 300
#> needs   : torch
```

A learner of your own is a fit and a predict pair, and it goes through
the same folds, the same cells and the same scoring as the ones that
ship. `ts.flatten()` lays a representation out as the block of
predictors the tabular learners read, one column per bin of each
channel. A fit that declares a `weights` argument is handed the case
weights every shipped learner fits under, which weight a presence by how
rare it is.

``` python
def nn_fit(x, y, **_):
    return {"x": ts.flatten(x), "y": y}

def nn_predict(model, x):
    m = ts.flatten(x)
    d = ((m[:, None, :] - model["x"][None, :, :]) ** 2).sum(axis=2)
    return model["y"][d.argmin(axis=1)].astype(float)

nearest_neighbour = ts.Learner("1nn", fit=nn_fit, predict=nn_predict, multi="joint")


def irls(design, y, w):
    beta = np.zeros(design.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-design @ beta))
        step = np.linalg.solve(design.T @ ((w * p * (1 - p))[:, None] * design),
                               design.T @ (w * (y - p)))
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return beta

def logistic_fit(x, y, weights, **_):
    m = ts.flatten(x)
    centre, scale = m.mean(axis=0), m.std(axis=0)
    design = np.column_stack([np.ones(len(m)), (m - centre) / scale])
    betas = np.column_stack([irls(design, y[:, j], weights[:, j]) for j in range(y.shape[1])])
    return {"centre": centre, "scale": scale, "betas": betas}

def logistic_predict(model, x):
    m = (ts.flatten(x) - model["centre"]) / model["scale"]
    design = np.column_stack([np.ones(len(m)), m])
    return 1 / (1 + np.exp(-design @ model["betas"]))

weighted_logistic = ts.Learner("weighted_logistic", fit=logistic_fit,
                               predict=logistic_predict, data=ts.grain("month"))

both = ts.timesift(targets, series, y="sp*", id="plot", time="t",
                   models=[ts.elasticnet(), nearest_neighbour, weighted_logistic],
                   sift=ts.grains("week", "month"), resampling=ts.cv(v=5), verbose=False)
print(ts.summary(both))
#> timesift  60 targets, 6 responses, 5-fold random CV, roc_auc
#>
#> candidates, scored on the outer folds
#> candidate                           mean    won  responses
#> 1nn / week                         0.744      0  joint
#> 1nn / month                        0.757      0  joint
#> weighted_logistic / month          0.900      1  separate
#> elasticnet / month                 0.903      2  separate
#> elasticnet / week                  0.906      3  separate
#>
#> procedure, chosen and weighted inside each outer training fold
#> selected                           0.904  se 0.015
#> ensemble                           0.917  se 0.012
#> selected elasticnet / month in 3, elasticnet / week in 2 of 5 folds
#>
#> choice on every target  elasticnet / week
#> weights on every target  elasticnet / month 0.48   elasticnet / week 0.34   weighted_logistic / month 0.18
```

Either way a fit is handed every response as the columns of `y` and
returns a model that predicts all of them, one column each.
`multi="joint"` says that one model covers them together, as the nearest
neighbour does; the default, `"separate"`, says the fit holds one model
per response, as the logistic regression does, and it is what the
report’s `responses` column reads.

## The combination

The combiner is handed out-of-fold predictions, the response, the mask
and the fold map, and never a model. `"stack"` fits non-negative weights
summing to one by minimising the response head’s own loss over the
scorable cells; `"mean"`, `"median"` and `"weighted"` combine without
fitting anything, and `"committee"` is the share of members voting
presence, each at the cut its own out-of-fold predictions give. The
weights below are fitted on the outer out-of-fold predictions of every
target and are what [`predict()`](https://rdrr.io/r/stats/predict.html)
uses. The ensemble’s score is not read with them, because they were
fitted to the responses it would be scored against; each outer fold’s
weights are in `fit.fold_weights`.

``` python
pd.Series(ts.ensemble_weights(fit))
#> elasticnet / day      3.833293e-14
#> elasticnet / week     2.785812e-01
#> elasticnet / month    7.214188e-01
#> dtype: float64
```

The weights say how much of the combination each candidate carries, and
a weight at zero is one the combination reached past. Where the ensemble
line of the plot above sits over every curve the candidates are carrying
different parts of the signal, and where it sits on the best curve they
are not.

## Reading a level honestly

The true skill statistic is read at the threshold that maximises it,
chosen on the same units the score is then read on. That inflates the
level, and by more the fewer presences a cell holds.
`ts.tss_inflation()` measures the inflation for the presence counts of
the design in hand.

``` python
pd.DataFrame(ts.tss_inflation(fit.y, fit.folds, skill=(0.6, 0.9), replicates=100))
#>    skill  reported  inflation     lower     upper  replicates
#> 0    0.6  0.769709   0.169709  0.718532  0.835235         100
#> 1    0.9  0.966596   0.066596  0.942541  0.987280         100
```

The inflation is an average over the planted model’s predictions: a
level is optimistic in expectation, and `ts.implied_skill()` inverts the
map to say which population skills a level read is consistent with. Its
size depends on how a model’s predictions are distributed as well as on
the presence counts, so two candidates of equal skill scored on the same
cells can be inflated by different amounts, and a paired difference in
TSS is not free of it. That is why a run is scored by AUC unless told
otherwise.

## The arrays on their own

`ts.grain_matrix()` is the representation without the fitting layer
around it. It bins the readings by the calendar and summarises every
bin.

``` python
x = ts.grain_matrix(series, "plot", "t", "temp", grain="week",
                    stats=("cold_day", "mean", "warm_day"))
x
#> <timesift matrix> 60 units x 18 bins x 3 channels
#> grain: week  stats: cold_day, mean, warm_day
#> from  : 2021-08-30T00:00:00Z to 2021-12-27T00:00:00Z
```

Bins follow the calendar. A month is 28, 30 or 31 days, and a week
starts on a Monday, so a bin is a real month or a real week and not a
drifting block of 730 or 168 hours.

``` python
ts.grain_matrix(series, "plot", "t", "temp", grain="month").bin_n[0]
#> array([720, 744, 720, 696], dtype=int32)
```

### An extreme day is not an extreme reading

`min` and `max` take the coldest and warmest single reading of a bin.
`cold_day` and `warm_day` reduce each day to its own mean first and then
take the extreme over days. `mean_daily_min` and `mean_daily_max` take
the mean of the daily extremes, which is the exposure a typical day of
the bin brought. One hour at -50 sets `min` to -50 outright; it reaches
the day-level statistics only through its twenty-fourth of that day’s
mean.

``` python
week = ts.grain_matrix(series, "plot", "t", "temp", grain="week",
                       stats=("min", "mean_daily_min", "cold_day", "mean",
                              "warm_day", "mean_daily_max", "max"))
pd.Series(week.values[0, 0], index=week.stats).round(2)
#> min              -28.68
#> mean_daily_min   -22.30
#> cold_day          -1.73
#> mean              -0.13
#> warm_day           0.94
#> mean_daily_max    21.54
#> max               31.50
#> dtype: float64
```

An encoder that ends in global pooling discards when a thermal event
happened, so the position of a bin in the year is given to it as input.

``` python
week_mean = ts.grain_matrix(series, "plot", "t", "temp", grain="week")
ts.bind_channels(week_mean, ts.calendar_channels(week_mean)).stats
#> ('mean', 'year_sin', 'year_cos')
```

## One grain at a time

Where the arrays are already built, `ts.grain_ladder()` fits every
learner at every grain of a set on one split and one mask. It is the
ladder a run reports as a curve, reachable on its own.

``` python
grain_set = ts.grain_matrix(series, "plot", "t", "temp", grain=("day", "week", "month"))
lad = ts.grain_ladder(grain_set, fit.y, [ts.elasticnet()], folds=fit.folds, verbose=False)
pd.DataFrame(lad.summary())
#>       learner  grain     score  n_variable   best
#> 0  elasticnet    day  0.882663           6  False
#> 1  elasticnet   week  0.905958           6   True
#> 2  elasticnet  month  0.902628           6  False
```

A claim about one step of that curve rests on the paired contrast. The
difference is taken inside each cell both arms scored, averaged within a
response, and summarised across the responses, which are the independent
replicates. Six responses is few, so the interval is wide and the rank
test behind `p_value` has few values to work with.

``` python
pd.Series(ts.paired_contrast(lad, "month|elasticnet", "day|elasticnet"))
#> a             month|elasticnet
#> b               day|elasticnet
#> diff                  0.019965
#> center                0.019965
#> lower                -0.011869
#> upper                 0.051799
#> n_variable                   6
#> n_cell                      30
#> n_favour                     4
#> p_value                0.21875
#> p_method                 exact
#> interval             variables
#> dtype: object
```

Where the whole curve is the question rather than one step of it,
`ts.grain_contrasts()` fits a mixed model on the per-cell scores and
compares every grain against the best one, correcting for the
comparisons made and no others. It needs scipy, the `contrasts` extra of
the wheel.

``` python
pd.DataFrame(ts.grain_contrasts(lad))
#>       learner  grain reference      diff     lower     upper   p_value
#> 0  elasticnet    day      week -0.023295 -0.066035  0.019446  0.367568
#> 1  elasticnet  month      week -0.003330 -0.046071  0.039411  0.977668
```

## What was read

With the per-fold fits kept, `ts.occlusion()` holds each bin of the
record back in turn, rescores the held-out units, and records the fall
in score as that bin’s weight, one row per bin and one column per
response. Nothing is refitted.

``` python
kept = ts.timesift(targets, series, y="sp*", id="plot", time="t",
                   sift=ts.grains("month"), resampling=ts.cv(v=5), ensemble=False,
                   keep_fits=True, verbose=False)
profile = ts.occlusion(kept, "elasticnet / month", permutations=5)
weight = pd.DataFrame(profile["weight"], index=profile["part"], columns=profile["variable"])
weight.mean(axis=1).head(4)
#> 2021-09-01T00:00:00Z    0.124983
#> 2021-10-01T00:00:00Z    0.093020
#> 2021-11-01T00:00:00Z    0.023396
#> 2021-12-01T00:00:00Z    0.028906
#> dtype: float64
```

Holding a channel back instead asks what each statistic of a grain
carries, which is the question behind keeping a bin’s extremes at all.
