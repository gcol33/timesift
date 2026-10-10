---
title: "Choosing how a record is read, in Python"
---

A sensor records every hour for years. Before any model sees it, the record is reduced: to monthly
means, to growing-degree-days, to whatever the analyst settled on once. `timesift` makes that
reduction an argument, fits at each setting of it, and shows how much of the record the prediction
actually needed.

This article runs the whole path on a small simulated record, from two tables to a scored
comparison and the prediction that follows from it. It follows [the R walkthrough](timesift.html)
section by section. The binning, the statistics and the penalised fit under both languages are one
compiled core, held to the same numbers by [the contract](contract.html).

## Installation

```{r, eval = FALSE}
# Install from CRAN
install.packages("timesift")

# Or install the development version from GitHub
# install.packages("pak")
pak::pak("gcol33/timesift")
```

```bash
# Install from PyPI
pip install timesift

# with the torch encoders, the contrasts and the plots
pip install "timesift[torch,contrasts,plot]"

# Or install the development version from GitHub
pip install git+https://github.com/gcol33/timesift
```

## Two tables

`targets` is one row per thing to predict. `series` is the long, time-stamped record belonging to
those rows, and the two are linked by an identifier both carry. Either is a mapping of column name
to array, which a pandas or polars data frame is.

```python
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
```

Each plot has a level of its own, and that level is buried in hour-to-hour noise an order of
magnitude larger.

## One call

```python
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
```

Every representation named in `sift` was built, and with `learners` left at its default of
`[ts.elasticnet()]`, a penalised logistic regression was fitted on each of them over the same five
folds. The report has two parts.

The candidates are the comparison. Each is scored on the five outer folds, on the same cells, so
their means can be read against each other across grains. `won` is how many responses a candidate
scored highest on, and `responses` says whether one fitted model covered them all or one was fitted
per response. The highest of those means was picked out on the folds it is scored on, so it is not
the number to report.

The procedure rows are. Inside each outer training fold the candidates were cross-validated again
on five inner folds, one was chosen on its inner score and the stack's weights were fitted on the
inner out-of-fold predictions; the choice and the weights then predicted the outer test fold once.
`selected` and `ensemble` are those held-out scores, choosing and weighting included.

```python
pd.DataFrame(fit.estimate)[["arm", "metric", "score", "se", "lower", "upper"]]
```

```python
pd.DataFrame(fit.selected)[["fold", "candidate", "inner_score"]]
```

The interval is across the six responses of this dataset, all fitted and scored on the same plots
and folds, so it does not carry the error those share.

A column of `targets` reaches the model only where `static` names it: `elevation` is a predictor
here because it was asked for, and a column of notes sitting beside it would not be.

```python
pd.DataFrame(fit.candidates)[["candidate", "representation", "bins", "channels"]]
```

Two channels at every grain: the binned temperature, and elevation held constant across the bins.

```python fig-alt="Mean AUC against representation, with the level the combination reached drawn across it."
ts.plot(fit);
```

`predict()` rebuilds each member's representation for new rows from the settings its own arm was
built with, and combines them through the ensemble refitted on every target;
`candidate="selected"` predicts with the candidate the rule chose on every target instead.

```python
p = fit.predict(targets, series)
p[:3, :4].round(3)
```

`type="binary"` cuts each response into presence and absence. The cut is learned from the same
candidate's out-of-fold predictions of the fit's own targets, so it comes from predictions of plots
the model had not been fitted on. `ts.decision_threshold()` on the fit returns the cuts
themselves, and `rule` picks how they are learned: `"youden"` maximises sensitivity plus
specificity, `"kappa"` maximises Cohen's kappa, and `"prevalence"` predicts as many presences as
were observed.

```python
pd.Series(ts.decision_threshold(fit, rule="prevalence"))
```

```python
fit.predict(targets, series, type="binary", rule="prevalence")[:3, :4]
```

A binary map is this prediction with one target per map cell, each carrying a series of its own at
the grain the fit reads.

## Representations

A representation carries the settings and nothing else, so the same object describes an arm before
any record has been read and rebuilds itself for new targets afterwards.

```python
ts.native()
ts.grain("week", stats=("cold_day", "mean", "warm_day"))
ts.multigrain(("week", "month"))
ts.lookback("30 days", bins=3)
```

`ts.grains()` and `ts.lookbacks()` are sets of them, and `ts.grains("auto")` reads off the record
every named grain it gives at least two bins. `ts.multigrain()` flattens several grains side by
side into one block of features; `ts.lookback()` is a fixed span ending at each target's own
instant, which is what a plot carrying several targets through time needs, given as
`target_time`.

A learner runs across the whole set, or at one representation it is pinned to:

```python
ts.elasticnet(data=ts.grain("month"))
```

## What a learner may be handed

A learner declares whether the bins reach it as a block of predictors or as a sequence whose order
in time is what it reads. A tabular learner given `ts.native()` is refused before any record is
touched, since building the array it would have been handed is the expensive half of the call; a
sequence learner is refused a representation that turns out to hold one bin, once the array says
how many it has.

```python error=true
ts.timesift(targets, series, y="sp*", id="plot", time="t",
            learners=[ts.elasticnet()], sift=ts.native(), verbose=False)
```

Inside a set such a pair is skipped and listed as `not applicable` by `ts.summary()`; named
through a learner's own `data=` it is an error, because a representation named by hand is a
decision.

## The split, and the cells a score is defined on

One fold map is read by everything that scores, so every candidate is fitted and scored on
identical splits. `ts.cv()` deals units into folds balanced on a stratifying value and
`ts.grouped_cv()` keeps every target sharing a group value on one side of each split;
`resampling` also takes a fold vector or a `ts.fold_map()` result directly.

```python
fit.folds
```

A per-response score needs both classes among the held-out units, and a per-response model needs
both classes among the units it was fitted on. The mask says which cells those are, from the
response and the fold map alone.

```python
fit.cells
```

Because it involves no model, every candidate is restricted to the same cells: their means share
one denominator and every paired difference runs on matched cells.

## Learners, and how they are trained

`ts.elasticnet()` and `ts.linear()` read a block of features, `ts.forest()` grows a probability
forest over one, and the `torch` encoders `ts.mlp()`, `ts.cnn()` and `ts.rescnn()` read a sequence
with a joint multi-label head, so every response is predicted together from a shared embedding.
Pooling strength across responses is what makes the rarer ones learnable at these sample sizes.

`ts.elasticnet()` is the arm a network is measured against, so it is fitted by the same compiled
core the R package calls rather than by a fitter on either side: the penalty path, the
standardisation and the cross-validated choice of penalty are one implementation, and the two
languages return the same coefficients for the same design. `s` reads that fit at
`"lambda.1se"`, glmnet's default, at `"lambda.min"`, or at a penalty of your own, and `tol` trades
how close the descent settles to the optimum against what it costs.

Architecture belongs to the constructor and training belongs to `ts.train_control()`, which every
neural learner of a run reads. A learner given a control of its own overrides the run's on the
settings it names and takes the rest from it.

```python
ts.train_control(epochs=200, device="cpu")
ts.cnn(channels=(16, 32), epochs=300)
```

A learner of your own is a fit and a predict pair, and it goes through the same folds, the same
cells and the same scoring as the ones that ship. `ts.flatten()` lays a representation out as the
block of predictors the tabular learners read, one column per bin of each channel. A fit that
declares a `weights` argument is handed the case weights every shipped learner fits under, which
weight a presence by how rare it is.

```python
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
                   learners=[ts.elasticnet(), nearest_neighbour, weighted_logistic],
                   sift=ts.grains("week", "month"), resampling=ts.cv(v=5), verbose=False)
print(ts.summary(both))
```

Either way a fit is handed every response as the columns of `y` and returns a model that predicts
all of them, one column each. `multi="joint"` says that one model covers them together, as the
nearest neighbour does; the default, `"separate"`, says the fit holds one model per response, as the
logistic regression does, and it is what the report's `responses` column reads.

## The combination

The combiner is handed out-of-fold predictions, the response, the mask and the fold map, and never
a model. `"stack"` fits non-negative weights summing to one by minimising the response head's own
loss over the scorable cells; `"mean"`, `"median"` and `"weighted"` combine without fitting
anything, and `"committee"` is the share of members voting presence, each at the cut its own
out-of-fold predictions give. The weights below are fitted on the outer out-of-fold predictions of
every target and are what `predict()` uses. The ensemble's score is not read with them, because
they were fitted to the responses it would be scored against; each outer fold's weights are in
`fit.fold_weights`.

```python
pd.Series(ts.ensemble_weights(fit))
```

The weights say how much of the combination each candidate carries, and a weight at zero is one
the combination reached past. Where the ensemble line of the plot above sits over every curve the
candidates are carrying different parts of the signal, and where it sits on the best curve they are
not.

## Reading a level honestly

The true skill statistic is read at the threshold that maximises it, chosen on the same units the
score is then read on. That inflates the level, and by more the fewer presences a cell holds.
`ts.tss_inflation()` measures the inflation for the presence counts of the design in hand.

```python
pd.DataFrame(ts.tss_inflation(fit.y, fit.folds, skill=(0.6, 0.9), replicates=100))
```

The inflation is an average over the planted model's predictions: a level is optimistic in
expectation, and `ts.implied_skill()` inverts the map to say which population skills a level read
is consistent with. Its size depends on how a model's predictions are distributed as well as on the
presence counts, so two candidates of equal skill scored on the same cells can be inflated by
different amounts, and a paired difference in TSS is not free of it. That is why a run is scored by
AUC unless told otherwise.

## The arrays on their own

`ts.grain_matrix()` is the representation without the fitting layer around it. It bins the
readings by the calendar and summarises every bin.

```python
x = ts.grain_matrix(series, "plot", "t", "temp", grain="week",
                    stats=("cold_day", "mean", "warm_day"))
x
```

Bins follow the calendar. A month is 28, 30 or 31 days, and a week starts on a Monday, so a bin is
a real month or a real week and not a drifting block of 730 or 168 hours.

```python
ts.grain_matrix(series, "plot", "t", "temp", grain="month").bin_n[0]
```

### An extreme day is not an extreme reading

`min` and `max` take the coldest and warmest single reading of a bin. `cold_day` and `warm_day`
reduce each day to its own mean first and then take the extreme over days. `mean_daily_min` and
`mean_daily_max` take the mean of the daily extremes, which is the exposure a typical day of the
bin brought. One hour at -50 sets `min` to -50 outright; it reaches the day-level statistics only
through its twenty-fourth of that day's mean.

```python
week = ts.grain_matrix(series, "plot", "t", "temp", grain="week",
                       stats=("min", "mean_daily_min", "cold_day", "mean",
                              "warm_day", "mean_daily_max", "max"))
pd.Series(week.values[0, 0], index=week.stats).round(2)
```

An encoder that ends in global pooling discards when a thermal event happened, so the position of a
bin in the year is given to it as input.

```python
week_mean = ts.grain_matrix(series, "plot", "t", "temp", grain="week")
ts.bind_channels(week_mean, ts.calendar_channels(week_mean)).stats
```

## One grain at a time

Where the arrays are already built, `ts.grain_ladder()` fits every learner at every grain of a set
on one split and one mask. It is the ladder a run reports as a curve, reachable on its own.

```python
grain_set = ts.grain_matrix(series, "plot", "t", "temp", grain=("day", "week", "month"))
lad = ts.grain_ladder(grain_set, fit.y, [ts.elasticnet()], folds=fit.folds, verbose=False)
pd.DataFrame(lad.summary())
```

A claim about one step of that curve rests on the paired contrast. The difference is taken inside
each cell both arms scored, averaged within a response, and summarised across the responses, which
are the independent replicates. Six responses is few, so the interval is wide and the rank test
behind `p_value` has few values to work with.

```python
pd.Series(ts.paired_contrast(lad, "month|elasticnet", "day|elasticnet"))
```

Where the whole curve is the question rather than one step of it, `ts.grain_contrasts()` fits a
mixed model on the per-cell scores and compares every grain against the best one, correcting for
the comparisons made and no others. It needs scipy, the `contrasts` extra of the wheel.

```python
pd.DataFrame(ts.grain_contrasts(lad))
```

## What was read

With the per-fold fits kept, `ts.occlusion()` holds each bin of the record back in turn, rescores
the held-out units, and records the fall in score as that bin's weight, one row per bin and one
column per response. Nothing is refitted.

```python
kept = ts.timesift(targets, series, y="sp*", id="plot", time="t",
                   sift=ts.grains("month"), resampling=ts.cv(v=5), ensemble=False,
                   keep_fits=True, verbose=False)
profile = ts.occlusion(kept, "elasticnet / month", permutations=5)
weight = pd.DataFrame(profile["weight"], index=profile["part"], columns=profile["variable"])
weight.mean(axis=1).head(4)
```

Holding a channel back instead asks what each statistic of a grain carries, which is the question
behind keeping a bin's extremes at all.
