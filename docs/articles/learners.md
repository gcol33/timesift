# Learners

A *learner* specifies how to fit a model and make predictions.
`timesift` includes fifteen learners, including regression models,
decision trees and neural networks. Each learner can be fitted using
different summaries of a sensor record, such as weekly or monthly means.
A model paired with one such summary is a *candidate*.

This article covers the available learners, compatible inputs, model
settings, neural-network training and tuning. It also shows how to add a
custom learner, response type or metric, and compares fitting times on a
small example.

The examples use 150 sampling units, each measured every six hours for a
year, and four responses recording presence or absence. The simulated
responses depend on changes in each unit’s readings across several
weeks. Summarising over longer periods can remove those changes. The
time scale of each summary is called its *grain*.

## Installation

``` r

# Install from CRAN
install.packages("timesift")

# Or install the development version from GitHub
# install.packages("pak")
pak::pak("gcol33/timesift")
```

``` bash
# Install from PyPI
pip install timesift

# with the torch encoders, the contrasts and the plots
pip install "timesift[torch,contrasts,plot]"

# Or install the development version from GitHub
pip install git+https://github.com/gcol33/timesift
```

``` r

sim <- simulate_records(n = 150L, mechanism = "lag", variables = 4L, days = 365L,
                        step_hours = 6, prevalence = 0.3, auc = 0.85, seed = 4L)
targets <- data.frame(unit = rownames(sim$y), sim$y)
colSums(sim$y)
#> v01 v02 v03 v04 
#>  46  55  40  51
```

``` python
import numpy as np
import pandas as pd
import timesift as ts

sim = ts.simulate_records(n=150, mechanism="lag", variables=4, days=365, step_hours=6,
                          prevalence=0.3, auc=0.85, seed=4)
y = sim.y
targets = pd.DataFrame({"unit": y.units, **dict(zip(y.variables, y.values.T.astype(int)))})
targets.drop(columns="unit").sum()
#> v01    48
#> v02    50
#> v03    44
#> v04    48
#> dtype: int64
```

## One interface

A learner is a list containing `fit` and `predict` functions, its
supported input types, whether it models responses jointly or
separately, and its package requirements. Printing it shows these
declarations and its settings. A setting shown as `NULL` takes its
default when the model is fitted.

A learner is an object holding `fit` and `predict` functions, its
supported input types, whether it models responses jointly or
separately, and its package requirements. Printing it shows these
declarations and its settings. A setting shown as `None` takes its
default when the model is fitted.

``` r

elasticnet()
#> <timesift learner> elasticnet 
#> reads   : tabular ; one model per response: yes, separate 
#> data    : every representation of the run 
#> settings: alpha = 0.5, n_inner = 10, squares = TRUE, s = lambda.1se, n_lambda = 100, tol = 1e-08, threads = 1, seed = 1
forest(balance = TRUE)
#> <timesift learner> forest 
#> reads   : tabular ; one model per response: yes, separate 
#> data    : every representation of the run 
#> settings: trees = NULL, mtry = NULL, min_node = NULL, balance = TRUE, preset = default, seed = 1, threads = 1
```

``` python
ts.elasticnet()
#> <timesift learner> elasticnet
#> reads   : tabular ; one model per response: yes, separate
#> data    : every representation of the run
#> settings: alpha = 0.5, n_inner = 10, squares = True, s = lambda.1se, n_lambda = 100, tol = 1e-08, threads = 1, seed = 1
ts.forest(balance=True)
#> <timesift learner> forest
#> reads   : tabular ; one model per response: yes, separate
#> data    : every representation of the run
#> settings: trees = None, mtry = None, min_node = None, balance = True, preset = default, seed = 1, threads = 1
```

`fit` receives a numeric array indexed by `[unit, bin, channel]` and a
response matrix with the same units. A bin is one time interval; a
channel contains one variable and summary statistic, such as mean
temperature. `predict` returns one column per response. Each learner
receives the full response matrix, including learners that fit a
separate model for each response.

[`c()`](https://rdrr.io/r/base/c.html) binds learners into a set, which
[`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
takes as `learners`.

A list binds learners into a set, which
[`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
takes as `learners`.

``` r

c(elasticnet(), forest(), cnn())
#> <timesift models> 3 learners 
#>   elasticnet     reads tabular, separate
#>   forest         reads tabular, separate
#>   cnn            reads sequence, joint
```

``` python
print(*[ts.elasticnet(), ts.forest(), ts.cnn()], sep="\n\n")
#> <timesift learner> elasticnet
#> reads   : tabular ; one model per response: yes, separate
#> data    : every representation of the run
#> settings: alpha = 0.5, n_inner = 10, squares = True, s = lambda.1se, n_lambda = 100, tol = 1e-08, threads = 1, seed = 1
#>
#> <timesift learner> forest
#> reads   : tabular ; one model per response: yes, separate
#> data    : every representation of the run
#> settings: trees = None, mtry = None, min_node = None, balance = False, preset = default, seed = 1, threads = 1
#>
#> <timesift learner> cnn
#> reads   : sequence ; one model per response: no, joint
#> data    : every representation of the run
#> settings: channels = 16/32/64/128, kernel = 7, dropout = 0.3
#> needs   : torch
```

## The learners that ship

Every shipped learner is registered under its own name, and its
declarations can be read off the constructor:

``` r

shipped <- learners()
built <- lapply(shipped, function(nm) getExportedValue("timesift", nm)())
data.frame(learner = shipped,
           reads = vapply(built, `[[`, "", "reads"),
           multi = vapply(built, `[[`, "", "multi"),
           needs = vapply(built, function(l) paste(l$needs, collapse = ", "), ""))
#>         learner    reads    multi needs
#> 1      additive  tabular separate      
#> 2      boosting  tabular separate      
#> 3           cnn sequence    joint torch
#> 4  discriminant  tabular separate      
#> 5    elasticnet  tabular separate      
#> 6      envelope  tabular separate      
#> 7        forest  tabular separate      
#> 8  hierarchical  tabular separate      
#> 9        linear  tabular separate      
#> 10         mars  tabular separate      
#> 11       maxent  tabular separate      
#> 12          mlp  tabular    joint torch
#> 13   perceptron  tabular separate      
#> 14       rescnn sequence    joint torch
#> 15         tree  tabular separate
```

``` python
built = [ts.get_learner(name) for name in ts.learners()]
pd.DataFrame({"learner": ts.learners(),
              "reads": [l.reads for l in built],
              "multi": [l.multi for l in built],
              "needs": [", ".join(l.needs) for l in built]})
#>          learner     reads     multi  needs
#> 0       additive   tabular  separate
#> 1       boosting   tabular  separate
#> 2            cnn  sequence     joint  torch
#> 3   discriminant   tabular  separate
#> 4     elasticnet   tabular  separate
#> 5       envelope   tabular  separate
#> 6         forest   tabular  separate
#> 7   hierarchical   tabular  separate
#> 8         linear   tabular  separate
#> 9           mars   tabular  separate
#> 10        maxent   tabular  separate
#> 11           mlp   tabular     joint  torch
#> 12    perceptron   tabular  separate
#> 13        rescnn  sequence     joint  torch
#> 14          tree   tabular  separate
```

Twelve learners use a table of predictors and fit one model per
response. The neural networks
[`mlp()`](https://gillescolling.com/timesift/reference/torch_learners.md),
[`cnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
and
[`rescnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
fit all responses together, sharing internal features between responses.
[`mlp()`](https://gillescolling.com/timesift/reference/torch_learners.md)
uses a table;
[`cnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
and
[`rescnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
use the time order of the bins. These three networks require torch. The
other learners use the compiled implementation shared by the R and
Python packages.

Each compiled learner is written from the published method and is
checked in the test fixtures against the package whose numbers it
reproduces. The [biomod2
article](https://gillescolling.com/timesift/articles/biomod2.md) maps
each of them to the biomod2 algorithm it corresponds to, with the option
sets biomod2 uses.

| learner | model | numbers checked against | responses |
|----|----|----|----|
| [`elasticnet()`](https://gillescolling.com/timesift/reference/elasticnet.md) | elastic net over every column and its square, penalty chosen by inner cross-validation | glmnet | every head |
| [`linear()`](https://gillescolling.com/timesift/reference/linear.md) | generalised linear model, terms searched by AIC | MASS `stepAIC()`, [`glm()`](https://rdrr.io/r/stats/glm.html) | every head |
| [`tree()`](https://gillescolling.com/timesift/reference/tree.md) | classification and regression tree, pruned by inner cross-validation | rpart | every head |
| [`forest()`](https://gillescolling.com/timesift/reference/forest.md) | random forest | randomForest | every head |
| [`boosting()`](https://gillescolling.com/timesift/reference/boosting.md) | gradient boosted trees, first or second order | gbm, xgboost | every head |
| [`maxent()`](https://gillescolling.com/timesift/reference/maxent.md) | maximum entropy over maxnet’s feature classes | maxnet | presence-absence |
| [`envelope()`](https://gillescolling.com/timesift/reference/envelope.md) | surface range envelope | biomod2 `bm_SRE()` | presence-absence |
| [`mars()`](https://gillescolling.com/timesift/reference/mars.md) | multivariate adaptive regression splines | earth | every head |
| [`discriminant()`](https://gillescolling.com/timesift/reference/discriminant.md) | flexible discriminant analysis on a MARS basis | mda `fda(method = mars)` | presence-absence |
| [`additive()`](https://gillescolling.com/timesift/reference/additive.md) | additive model of thin plate regression splines | mgcv `gam(method = "GCV.Cp")` | every head |
| [`perceptron()`](https://gillescolling.com/timesift/reference/perceptron.md) | network of one hidden layer, fitted by BFGS | nnet | every head |
| [`hierarchical()`](https://gillescolling.com/timesift/reference/hierarchical.md) | Bayesian logistic model, optional unit intercepts and spatial field | none | presence-absence |
| [`mlp()`](https://gillescolling.com/timesift/reference/torch_learners.md) | fully connected encoder, joint head | torch | every head |
| [`cnn()`](https://gillescolling.com/timesift/reference/torch_learners.md) | convolutional encoder, joint head | torch | every head |
| [`rescnn()`](https://gillescolling.com/timesift/reference/torch_learners.md) | dilated residual convolutional encoder, joint head | torch | every head |

A *response head* specifies the type of values being predicted and the
loss function used to fit them. “All response types” in the table means
logistic fitting for presence-absence, squared-error fitting for
continuous, abundance and ordinal responses, and Poisson fitting for
counts.
[`hierarchical()`](https://gillescolling.com/timesift/reference/hierarchical.md)
supports presence-absence. It approximates the posterior distribution
using Laplace’s method and integrates over a grid of hyperparameter
values (Rue, Martino and Chopin 2009). Its `"nngp"` and `"hsgp"` options
model spatial variation using coordinates supplied through
`timesift(coords = )`.

## What a learner may be handed

A tabular learner reads the bins as a block of predictors, one column
per bin of each channel. Given
[`native()`](https://gillescolling.com/timesift/reference/native.md),
the record unreduced, it would be handed one column per reading, so the
pair is refused. When the learner is pinned to
[`native()`](https://gillescolling.com/timesift/reference/native.md)
through `data =`, the refusal comes before the record is read:

``` r

timesift(targets, sim$readings, y = starts_with("v"), id = unit, time = time,
         learners = forest(data = native()), verbose = FALSE)
#> Error:
#> ! `forest()` reads a tabular representation; `native()` gives it one column per reading. Use `grain()`, `multigrain()` or `lookback()`.
```

``` python
ts.timesift(targets, sim.readings, y="v*", id="unit", time="time",
            learners=ts.forest(data=ts.native()), verbose=False)
#> ValueError: forest() reads a tabular representation; native gives it one column per reading. Use grain(), multigrain() or lookback().
```

Inside a `sift` the same pair is skipped and the run goes on. The
candidate table keeps the skipped pair, with the reason in `note`.

``` r

mixed <- timesift(targets, sim$readings, y = starts_with("v"), id = unit, time = time,
                  learners = c(elasticnet(), forest()),
                  sift = c(native(), grain("month")),
                  resampling = cv(v = 5), n_inner = NULL, ensemble = FALSE, verbose = FALSE)
mixed$candidates[c("candidate", "bins", "status")]
#>             candidate bins         status
#> 1 elasticnet / native 1460 not applicable
#> 2  elasticnet / month   12         fitted
#> 3     forest / native 1460 not applicable
#> 4      forest / month   12         fitted
```

The native representation of this record has 1460 bins, one per reading.
A sequence learner is refused the opposite case: a representation that
gives one row of features, such as
[`multigrain()`](https://gillescolling.com/timesift/reference/native.md),
or one whose record turns out to hold a single bin. That refusal is
shown with the encoders below.

## Several learners in one run

`learners` takes a set or a named list. Names distinguish candidates
that use different settings of the same learner. Each learner is fitted
at every compatible grain in `sift`, unless its `data` argument
specifies one representation. Here
[`linear()`](https://gillescolling.com/timesift/reference/linear.md)
uses monthly summaries. The elastic net uses a looser stopping tolerance
to shorten the example; the settings section explains that choice.

Cross-validation divides the units into groups called *folds*. Each fold
is predicted using models trained on the other folds. An additional
split within the training data, called *inner cross-validation*, chooses
candidates and estimates ensemble weights before the outer test fold is
predicted. A response in one fold is called a *cell* in the scoring
output. The score used here is the area under the receiver operating
characteristic curve (AUC), which measures how well predictions rank
presences above absences.

In Python `learners` takes a list, and a candidate reports under its
learner’s `name`, which `dataclasses.replace(learner, name="...")` sets.

``` r

models <- list(
  elasticnet = elasticnet(tol = 1e-4),
  linear = linear(data = grain("month")),
  tree = tree(),
  forest = forest(),
  forest_balanced = forest(balance = TRUE),
  gbm = boosting(),
  xgboost = boosting(method = "xgboost"),
  maxent = maxent()
)
run_time <- system.time(
  fit <- timesift(targets, sim$readings, y = starts_with("v"), id = unit, time = time,
                  learners = models, sift = grains("week", "month"),
                  resampling = cv(v = 5), n_inner = 3L, verbose = FALSE)
)
run_time[["elapsed"]]
#> [1] 30.61
```

``` python
import time
from dataclasses import replace

models = [
    ts.elasticnet(tol=1e-4),
    ts.linear(data=ts.grain("month")),
    ts.tree(),
    ts.forest(),
    replace(ts.forest(balance=True), name="forest_balanced"),
    replace(ts.boosting(), name="gbm"),
    replace(ts.boosting(method="xgboost"), name="xgboost"),
    ts.maxent(),
]
start = time.perf_counter()
fit = ts.timesift(targets, sim.readings, y="v*", id="unit", time="time",
                  learners=models, sift=ts.grains("week", "month"),
                  resampling=ts.cv(v=5), n_inner=3, verbose=False)
elapsed = time.perf_counter() - start
round(elapsed)
#> 27
```

The run took 3127 seconds for 15 candidates. With three inner folds in
each of five outer folds, every candidate is fitted 5 x (3 + 1) = 20
times, three inner fits and one outer fit in each outer fold, and once
more on every target.

[`summary()`](https://rdrr.io/r/base/summary.html) lists the candidates
in order of their mean score on the outer folds, with the number of
responses on which each scored highest, and then the held-out score of
the procedure that chooses and weights them.

``` r

summary(fit)
#> timesift  150 targets, 4 responses, 5-fold random CV, roc_auc
#> 
#> candidates, scored on the outer folds
#> candidate                         mean    won  responses
#> forest_balanced / month          0.459      0  separate
#> forest / month                   0.461      0  separate
#> tree / month                     0.480      0  separate
#> gbm / month                      0.483      0  separate
#> xgboost / month                  0.503      0  separate
#> elasticnet / month               0.523      0  separate
#> tree / week                      0.537      0  separate
#> forest_balanced / week           0.593      0  separate
#> forest / week                    0.595      0  separate
#> xgboost / week                   0.611      0  separate
#> maxent / month                   0.626      0  separate
#> linear / month                   0.634      0  separate
#> gbm / week                       0.637      0  separate
#> elasticnet / week                0.700      0  separate
#> maxent / week                    0.750      4  separate
#> 
#> procedure, chosen and weighted inside each outer training fold
#> selected                         0.741  se 0.048
#> ensemble                         0.740  se 0.040
#> selected maxent / week in 4, elasticnet / week in 1 of 5 folds
#> 
#> choice on every target  maxent / week
#> weights on every target  maxent / week 0.80   gbm / week 0.10   linear / month 0.09   elasticnet / week 0.02
```

``` python
print(ts.summary(fit))
#> timesift  150 targets, 4 responses, 5-fold random CV, roc_auc
#>
#> candidates, scored on the outer folds
#> candidate                         mean    won  responses
#> tree / month                     0.513      0  separate
#> xgboost / month                  0.549      0  separate
#> forest_balanced / month          0.560      0  separate
#> forest / month                   0.561      0  separate
#> elasticnet / month               0.583      0  separate
#> gbm / month                      0.589      0  separate
#> linear / month                   0.603      0  separate
#> tree / week                      0.623      0  separate
#> maxent / month                   0.624      0  separate
#> forest_balanced / week           0.638      0  separate
#> forest / week                    0.649      0  separate
#> gbm / week                       0.677      0  separate
#> xgboost / week                   0.697      0  separate
#> maxent / week                    0.779      2  separate
#> elasticnet / week                0.791      2  separate
#>
#> procedure, chosen and weighted inside each outer training fold
#> selected                         0.777  se 0.030
#> ensemble                         0.775  se 0.027
#> selected maxent / week in 4, elasticnet / week in 1 of 5 folds
#>
#> choice on every target  elasticnet / week
#> weights on every target  maxent / week 0.87   xgboost / week 0.05   elasticnet / week 0.04   linear / month 0.03   xgboost / month 0.01
```

Each learner run at both grains scores higher at the weekly one, which
is the grain the simulation acts at. At the monthly grain the tree, both
forests and gbm score below 0.5, so they find nothing
therebetween0.513and0.589. `maxent / week``elasticnet / week` has the
highest mean, 0.7500.791, and scored highest on all four2 of the four
responses. The procedure rows are the scores to report: inside each
outer training fold the inner folds chose a candidate and fitted the
stack’s weights, and the outer test fold then scored that choice once,
0.7410.777 for the selected candidate and 0.7400.775 for the stack. The
highest candidate mean was picked out on the same folds it is scored on.

The candidate table records how each candidate was built: its learner,
its grain, the number of bins and channels, whether it was fitted, and,
for a tuned learner, the settings it chose.

``` r

head(fit$candidates[c("candidate", "learner", "grain", "bins", "status")], 4)
#>            candidate    learner grain bins status
#> 1  elasticnet / week elasticnet  week   53 fitted
#> 2 elasticnet / month elasticnet month   12 fitted
#> 3     linear / month     linear month   12 fitted
#> 4        tree / week       tree  week   53 fitted
```

``` python
pd.DataFrame(fit.candidates)[["candidate", "learner", "grain", "bins"]].head(4)
#>             candidate     learner  grain  bins
#> 0   elasticnet / week  elasticnet   week    53
#> 1  elasticnet / month  elasticnet  month    12
#> 2      linear / month      linear  month    12
#> 3         tree / week        tree   week    53
```

## One learner on one array

[`fit_learner()`](https://gillescolling.com/timesift/reference/fit_learner.md)
fits a learner to an array from
[`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md)
without the folds and the scoring around it, and
[`predict()`](https://rdrr.io/r/stats/predict.html) on the result takes
an array of the same bins and channels. The units below are split in
half once. The two arrays are built from the two halves of the record,
over the same calendar weeks.

``` r

train <- rownames(sim$y)[1:75]
test <- rownames(sim$y)[76:150]
in_train <- sim$readings$unit %in% train
week_train <- grain_matrix(sim$readings[in_train, ], unit, time, reading, grain = "week")
week_test <- grain_matrix(sim$readings[!in_train, ], unit, time, reading, grain = "week")
f <- fit_learner(elasticnet(), week_train, sim$y[train, ])
f
#> <timesift fit> elasticnet at the week grain
#> channels: mean 
#> response: presence_absence on 4 variables
f$model[c("unfitted", "stopped")]
#> $unfitted
#> character(0)
#> 
#> $stopped
#> character(0)
```

``` python
train, test = y.take_units(np.arange(75)), y.take_units(np.arange(75, 150))
in_train = np.isin(sim.readings["unit"], train.units)

def rows(mask):
    return {k: v[mask] for k, v in sim.readings.items()}

week_train = ts.grain_matrix(rows(in_train), "unit", "time", "reading", grain="week")
week_test = ts.grain_matrix(rows(~in_train), "unit", "time", "reading", grain="week")
f = ts.fit_learner(ts.elasticnet(), week_train, train)
f.learner
#> <timesift learner> elasticnet
#> reads   : tabular ; one model per response: yes, separate
#> data    : every representation of the run
#> settings: alpha = 0.5, n_inner = 10, squares = True, s = lambda.1se, n_lambda = 100, tol = 1e-08, threads = 1, seed = 1
{k: f.model[k] for k in ("unfitted", "stopped")}
#> {'unfitted': [], 'stopped': []}
```

`unfitted` lists responses with too few presences for penalty selection;
their predictions use the prevalence in the training data. `stopped`
lists responses whose penalty path ended early. Neither occurs here.
Prediction requires the same bins used during fitting:

``` r

month_train <- grain_matrix(sim$readings[in_train, ], unit, time, reading, grain = "month")
month_test <- grain_matrix(sim$readings[!in_train, ], unit, time, reading, grain = "month")
predict(f, month_test)
#> Error:
#> ! the representation predicted on has different channels or bins from the fitted one: 12 bins here and 53 bins in the fit. A calendar grain is read by its bins' instants, so a fit predicts a record over the same period; a lookback reads a span relative to each target and predicts any period.
```

``` python
month_train = ts.grain_matrix(rows(in_train), "unit", "time", "reading", grain="month")
month_test = ts.grain_matrix(rows(~in_train), "unit", "time", "reading", grain="month")
f.predict(month_test)
#> ValueError: the representation predicted on has different channels or bins from the fitted one: 12 bins here and 53 bins in the fit. A calendar grain is read by its bins' instants, so a fit predicts a record over the same period; a lookback reads a span relative to each target and predicts any period.
```

## Settings of the main learners

The settings below are the ones most often changed; each constructor’s
documentation lists the rest.

[`elasticnet()`](https://gillescolling.com/timesift/reference/elasticnet.md)
fits one penalised model per response over every column and, with
`squares = TRUE`, its square. `alpha` mixes the two penalties, `1` being
the lasso and `0` ridge. `s` is where the penalty path is read:
`"lambda.1se"`, the largest penalty within one standard error of the
least held-out deviance, is glmnet’s default, `"lambda.min"` is the
penalty of least deviance, and a number is a penalty given directly.
`tol` is where the coordinate descent stops; the default leaves the fit
as close to the optimum as glmnet’s own default does, and a looser one
is faster.

[`forest()`](https://gillescolling.com/timesift/reference/forest.md)
grows 500 trees by default, with `mtry` the square root of the column
count under presence-absence. `balance = TRUE` draws as many units of
each class into every tree as the smaller class holds, so a rare
response’s presences are half of each tree’s draw.
[`boosting()`](https://gillescolling.com/timesift/reference/boosting.md)
fits gbm’s first-order trees by default and XGBoost’s second-order trees
with `method = "xgboost"`, whose `lambda` and `gamma` are the penalty on
a leaf and the least gain a split must make.
[`maxent()`](https://gillescolling.com/timesift/reference/maxent.md)
builds maxnet’s feature classes, the letters of `classes`: `l` linear,
`q` quadratic, `p` pairwise products, `h` hinges and `t` thresholds.
Left `NULL`, the classes follow each response’s presence count, as
maxnet chooses them.

The function below fits a learner on the training half and returns its
AUC on the other half, averaged over the four responses.

``` r

held_out_auc <- function(l) {
  f <- fit_learner(l, week_train, sim$y[train, ])
  p <- predict(f, week_test)
  mean(vapply(colnames(p), function(v) roc_auc(sim$y[test, v], p[, v]), numeric(1)))
}
variants <- list(
  `elasticnet()` = elasticnet(),
  `elasticnet(s = "lambda.min")` = elasticnet(s = "lambda.min"),
  `elasticnet(alpha = 1)` = elasticnet(alpha = 1),
  `elasticnet(tol = 1e-4)` = elasticnet(tol = 1e-4),
  `forest()` = forest(),
  `forest(balance = TRUE)` = forest(balance = TRUE),
  `boosting()` = boosting(),
  `boosting(method = "xgboost")` = boosting(method = "xgboost"),
  `maxent()` = maxent(),
  `maxent(classes = "lq")` = maxent(classes = "lq")
)
round(vapply(variants, held_out_auc, numeric(1)), 3)
#>                 elasticnet() elasticnet(s = "lambda.min") 
#>                        0.631                        0.680 
#>        elasticnet(alpha = 1)       elasticnet(tol = 1e-4) 
#>                        0.608                        0.631 
#>                     forest()       forest(balance = TRUE) 
#>                        0.516                        0.514 
#>                   boosting() boosting(method = "xgboost") 
#>                        0.606                        0.555 
#>                     maxent()       maxent(classes = "lq") 
#>                        0.665                        0.693
```

``` python
def held_out_auc(learner):
    p = ts.fit_learner(learner, week_train, train).predict(week_test)
    return np.mean([ts.roc_auc(test.values[:, j], p[:, j]) for j in range(p.shape[1])])

variants = {
    "elasticnet()": ts.elasticnet(),
    'elasticnet(s="lambda.min")': ts.elasticnet(s="lambda.min"),
    "elasticnet(alpha=1)": ts.elasticnet(alpha=1),
    "elasticnet(tol=1e-4)": ts.elasticnet(tol=1e-4),
    "forest()": ts.forest(),
    "forest(balance=True)": ts.forest(balance=True),
    "boosting()": ts.boosting(),
    'boosting(method="xgboost")': ts.boosting(method="xgboost"),
    "maxent()": ts.maxent(),
    'maxent(classes="lq")': ts.maxent(classes="lq"),
}
held_out = pd.Series({name: held_out_auc(l) for name, l in variants.items()})
held_out.round(3)
#> elasticnet()                  0.729
#> elasticnet(s="lambda.min")    0.726
#> elasticnet(alpha=1)           0.727
#> elasticnet(tol=1e-4)          0.727
#> forest()                      0.626
#> forest(balance=True)          0.605
#> boosting()                    0.615
#> boosting(method="xgboost")    0.638
#> maxent()                      0.710
#> maxent(classes="lq")          0.736
#> dtype: float64
```

On this split the elastic net read at `"lambda.min"` scores 0.6800.726
against 0.6310.729 at the default `"lambda.1se"`, and the lasso scores
0.6080.727. The looser tolerance gives the same AUC as the default to
three decimals0.727against0.729at the default, and the timings at the
end of the article show what it saves. `maxent(classes = "lq")` scores
0.6930.736 against 0.6650.710 for the classes maxnet chooses, and both
forests score close to 0.50.626and0.605 at this grain on 75 units. A
single split of 75 units is a noisy comparison, so a difference of this
size is worth checking under
[`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)’s
folds, with the two settings as two named candidates, before acting on
it.

## Neural encoders

The neural-network constructors
[`mlp()`](https://gillescolling.com/timesift/reference/torch_learners.md),
[`cnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
and
[`rescnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
set the network structure, including layer widths and the number of
neighbouring bins each convolution uses.
[`train_control()`](https://gillescolling.com/timesift/reference/train_control.md)
sets how the networks are trained: passes through the data (epochs),
observations per update (batch size), learning rate, optimiser, stopping
rule and computing device. Settings on an individual learner override
the corresponding settings for the run.

``` r

train_control(epochs = 5L, device = "cpu")
#> <timesift control>
#>   epochs           5
#>   batch_size       64   (default)
#>   learning_rate    0.001   (default)
#>   weight_decay     1e-04   (default)
#>   optimizer        adamw   (default)
#>   penalty          0   (default)
#>   alpha            0.5   (default)
#>   schedule         cosine   (default)
#>   plateau_factor   0.1   (default)
#>   plateau_patience 10   (default)
#>   early_stopping   Inf   (default)
#>   val_frac         0   (default)
#>   device           cpu
#>   seed             1   (default)
#>   swa              FALSE   (default)
#>   swa_start        0.7   (default)
cnn(channels = c(8L, 16L), kernel = 3L, epochs = 10L)
#> <timesift learner> cnn 
#> reads   : sequence ; one model per response: no, joint 
#> data    : every representation of the run 
#> settings: channels =  8/16, kernel = 3, dropout = 0.3 
#> training: epochs = 10 
#> needs   : torch
```

``` python
ts.train_control(epochs=5, device="cpu")
#> <timesift control>
#>   epochs           5
#>   batch_size       64   (default)
#>   learning_rate    0.001   (default)
#>   weight_decay     0.0001   (default)
#>   optimizer        adamw   (default)
#>   penalty          0.0   (default)
#>   alpha            0.5   (default)
#>   schedule         cosine   (default)
#>   plateau_factor   0.1   (default)
#>   plateau_patience 10   (default)
#>   early_stopping   inf   (default)
#>   val_frac         0.0   (default)
#>   device           cpu
#>   seed             1   (default)
#>   swa              False   (default)
#>   swa_start        0.7   (default)
ts.cnn(channels=(8, 16), kernel=3, epochs=10)
#> <timesift learner> cnn
#> reads   : sequence ; one model per response: no, joint
#> data    : every representation of the run
#> settings: channels = 8/16, kernel = 3, dropout = 0.3
#> training: epochs = 10
#> needs   : torch
```

The control marks the settings that were not named as `(default)`, and
the learner lists under `training` the one setting it overrides. The run
below trains both encoders at the weekly grain for a handful of epochs
on the processor. The run’s control sets five epochs and the
convolutional encoder overrides it with ten. This keeps the example
quick and leaves both networks undertrained.

``` r

nets <- timesift(targets, sim$readings, y = starts_with("v"), id = unit, time = time,
                 learners = c(mlp(hidden = c(32L, 16L)),
                              cnn(channels = c(8L, 16L), kernel = 3L, epochs = 10L)),
                 sift = grains("week"), control = train_control(epochs = 5L, device = "cpu"),
                 resampling = cv(v = 3), n_inner = NULL, ensemble = FALSE, verbose = FALSE)
summary(nets)
#> timesift  150 targets, 4 responses, 3-fold random CV, roc_auc
#> 
#> candidates, scored on the outer folds
#> candidate            mean    won  responses
#> mlp / week          0.495      2  joint
#> cnn / week          0.508      2  joint
#> 
#> choice on every target  cnn / week
```

``` python
nets = ts.timesift(targets, sim.readings, y="v*", id="unit", time="time",
                   learners=[ts.mlp(hidden=(32, 16)), ts.cnn(channels=(8, 16), kernel=3, epochs=10)],
                   sift=ts.grains("week"), control=ts.train_control(epochs=5, device="cpu"),
                   resampling=ts.cv(v=3), n_inner=None, ensemble=False, verbose=False)
print(ts.summary(nets))
#> timesift  150 targets, 4 responses, 3-fold random CV, roc_auc
#>
#> candidates, scored on the outer folds
#> candidate            mean    won  responses
#> cnn / week          0.446      1  joint
#> mlp / week          0.463      3  joint
#>
#> choice on every target  mlp / week
```

After so few epochs both encoders score close to 0.5, as an undertrained
network does. The `responses` column reads `joint`: one network covers
all four responses, and each candidate is still reported once.

A sequence learner pinned to a representation that gives one row of
features is refused before the record is read:

``` r

timesift(targets, sim$readings, y = starts_with("v"), id = unit, time = time,
         learners = cnn(data = multigrain(c("week", "month"))), verbose = FALSE)
#> Error:
#> ! `cnn()` reads a sequence; `multigrain()` gives one row of features.
```

``` python
ts.timesift(targets, sim.readings, y="v*", id="unit", time="time",
            learners=ts.cnn(data=ts.multigrain(("week", "month"))), verbose=False)
#> ValueError: no learner can read any of the representations:
#>   cnn() reads a sequence; multigrain(week+month) gives one row of 65 features
```

A fitted encoder holds its weights as plain arrays and the device as the
setting it was given, so a fit saved with
[`saveRDS()`](https://rdrr.io/r/base/readRDS.html) on a machine with a
graphics card predicts after
[`readRDS()`](https://rdrr.io/r/base/readRDS.html) on one without.

## Tuning

[`tune()`](https://gillescolling.com/timesift/reference/tune.md) wraps a
learner so that, each time it is fitted, it cross-validates a grid of
its own settings on the units it is handed and fits the setting that
scored best. In a run each outer fold tunes on its own training units,
so the setting a fold is scored with was never chosen on that fold’s
held-out units. A grid names settings the learner carries, and every
combination is tried.

``` r

tuned <- tune(forest(trees = 200L), list(mtry = c(2L, 8L), min_node = c(1L, 5L)), n_inner = 3L)
tf <- fit_learner(tuned, week_train, sim$y[train, ])
tf$model$chosen
#> $mtry
#> [1] 8
#> 
#> $min_node
#> [1] 5
tf$model$table
#>                 settings     score
#> 1 mtry = 2, min_node = 1 0.5355025
#> 2 mtry = 8, min_node = 1 0.5438940
#> 3 mtry = 2, min_node = 5 0.5415086
#> 4 mtry = 8, min_node = 5 0.5666238
```

``` python
tuned = ts.tune(ts.forest(trees=200), {"mtry": (2, 8), "min_node": (1, 5)}, n_inner=3)
tf = ts.fit_learner(tuned, week_train, train)
tf.model.chosen
#> {'mtry': 8, 'min_node': 5}
pd.DataFrame(tf.model.table)
#>                  settings     score
#> 0  mtry = 2, min_node = 1  0.530352
#> 1  mtry = 8, min_node = 1  0.553803
#> 2  mtry = 2, min_node = 5  0.520598
#> 3  mtry = 8, min_node = 5  0.566769
```

The search chose `mtry = 8` and `min_node = 5``mtry = 8` and
`min_node = 5`, the combination with the highest mean inner AUC. Under
[`timesift()`](https://gillescolling.com/timesift/reference/timesift.md),
the settings each tuned candidate chose on all targets are in the
`settings` column of the candidate table.

Given no `grid`,
[`tune()`](https://gillescolling.com/timesift/reference/tune.md)
searches the grid registered under the learner’s name. Eight shipped
learners carry the grids that biomod2’s `BIOMOD_Tuning()` searches:

``` r

tunings()
#> [1] "boosting"     "discriminant" "envelope"     "forest"       "mars"        
#> [6] "maxent"       "mlp"          "perceptron"
```

``` python
ts.tunings()
#> ['boosting', 'discriminant', 'envelope', 'forest', 'mars', 'maxent', 'mlp', 'perceptron']
```

[`register_tuning()`](https://gillescolling.com/timesift/reference/register_tuning.md)
adds a grid, as a named list or as a function of the learner and the
array for a grid that depends on the data, such as one bounded by the
number of columns. The elastic net has no grid registered; the one below
searches its mixing.

``` r

register_tuning("elasticnet", list(alpha = c(0.25, 1)))
te <- fit_learner(tune(elasticnet(), n_inner = 3L), month_train, sim$y[train, ])
te$model$table
#>       settings     score
#> 1 alpha = 0.25 0.5088889
#> 2    alpha = 1 0.5158333
```

``` python
ts.register_tuning("elasticnet", {"alpha": (0.25, 1)});
te = ts.fit_learner(ts.tune(ts.elasticnet(), n_inner=3), month_train, train)
pd.DataFrame(te.model.table)
#>        settings     score
#> 0  alpha = 0.25  0.578892
#> 1     alpha = 1  0.598877
```

`alpha = 1`, the lasso, scored higher on the three inner folds, and the
tuned learner was fitted with it. The registration lasts for the
session, and replacing a grid already registered takes
`overwrite = TRUE`.

## A learner of one’s own

[`learner()`](https://gillescolling.com/timesift/reference/learner.md)
takes a `fit` and a `predict` function, and `fit` is handed what it
declares. A `fit` with an argument called `weights` receives the
response head’s case weights, one per cell of the response. One with
`head` receives the head itself, whose `loss` says what to fit toward.
One with `control` receives the resolved
[`train_control()`](https://gillescolling.com/timesift/reference/train_control.md),
and one with `group` receives the grouping the outer folds keep whole,
so that a split drawn inside the fit keeps it too. A `fit` that declares
none of these is fitted unweighted, with nothing but the array and the
response.

The learner below reduces every channel to its mean over the bins and
fits one generalised linear model per response, in the family the head’s
loss names.

``` r

channel_means <- function(data = NULL) {
  learner(
    "channel_means",
    fit = function(x, y, head, weights, ...) {
      f <- data.frame(apply(unclass(x), c(1L, 3L), mean))
      family <- switch(head$loss,
                       binary_cross_entropy = stats::quasibinomial(),
                       squared_error = stats::gaussian(),
                       poisson_deviance = stats::quasipoisson())
      lapply(seq_len(ncol(y)), function(j) {
        stats::glm(y[, j] ~ ., data = f, weights = weights[, j], family = family)
      })
    },
    predict = function(model, x) {
      f <- data.frame(apply(unclass(x), c(1L, 3L), mean))
      vapply(model, function(m) stats::predict(m, f, type = "response"), numeric(nrow(f)))
    },
    data = data
  )
}
channel_means()
#> <timesift learner> channel_means 
#> reads   : tabular ; one model per response: yes, separate 
#> data    : every representation of the run
```

``` python
families = {
    "binary_cross_entropy": (lambda eta: 1 / (1 + np.exp(-eta)), lambda mu: mu * (1 - mu)),
    "squared_error": (lambda eta: eta, np.ones_like),
    "poisson_deviance": (np.exp, lambda mu: mu),
}

def glm(design, y, w, loss):
    inverse, variance = families[loss]
    beta = np.zeros(design.shape[1])
    for _ in range(50):
        mu = inverse(design @ beta)
        step = np.linalg.solve(design.T @ ((w * variance(mu))[:, None] * design),
                               design.T @ (w * (y - mu)))
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return beta

def channel_means(data=None):
    def design(x):
        return np.column_stack([np.ones(x.values.shape[0]), x.values.mean(axis=1)])

    def fit(x, y, head, weights, **_):
        return [glm(design(x), y[:, j], weights[:, j], head["loss"]) for j in range(y.shape[1])], head

    def predict(model, x):
        betas, head = model
        inverse = families[head["loss"]][0]
        return np.column_stack([inverse(design(x) @ b) for b in betas])

    return ts.Learner("channel_means", fit=fit, predict=predict, data=data)

channel_means()
#> <timesift learner> channel_means
#> reads   : tabular ; one model per response: yes, separate
#> data    : every representation of the run
```

[`register_learner()`](https://gillescolling.com/timesift/reference/register_learner.md)
makes the constructor available by name, beside the shipped learners,
and a registered learner can then be given to
[`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
or
[`fit_learner()`](https://gillescolling.com/timesift/reference/fit_learner.md)
as its name alone. A fit made by a registered learner is saved as the
learner’s name and settings, so a session that reads the fit back with
[`readRDS()`](https://rdrr.io/r/base/readRDS.html) registers the learner
again before predicting from it.

``` r

register_learner("channel_means", channel_means)
"channel_means" %in% learners()
#> [1] TRUE
own <- timesift(targets, sim$readings, y = starts_with("v"), id = unit, time = time,
                learners = c(elasticnet(), channel_means()),
                sift = grains("week", "month"),
                resampling = cv(v = 5), n_inner = NULL, ensemble = FALSE, verbose = FALSE)
summary(own)
#> timesift  150 targets, 4 responses, 5-fold random CV, roc_auc
#> 
#> candidates, scored on the outer folds
#> candidate                       mean    won  responses
#> channel_means / month          0.461      0  separate
#> channel_means / week           0.461      0  separate
#> elasticnet / month             0.524      0  separate
#> elasticnet / week              0.701      4  separate
#> 
#> choice on every target  elasticnet / week
```

``` python
ts.register_learner("channel_means", channel_means);
"channel_means" in ts.learners()
#> True
own = ts.timesift(targets, sim.readings, y="v*", id="unit", time="time",
                  learners=[ts.elasticnet(), channel_means()],
                  sift=ts.grains("week", "month"),
                  resampling=ts.cv(v=5), n_inner=None, ensemble=False, verbose=False)
print(ts.summary(own))
#> timesift  150 targets, 4 responses, 5-fold random CV, roc_auc
#>
#> candidates, scored on the outer folds
#> candidate                       mean    won  responses
#> channel_means / month          0.470      0  separate
#> channel_means / week           0.472      0  separate
#> elasticnet / month             0.571      0  separate
#> elasticnet / week              0.789      4  separate
#>
#> choice on every target  elasticnet / week
```

`channel_means` scores 0.461 at both grains0.470 at the monthly grain
and 0.472 at the weekly. The mean of a channel over the whole year is
nearly the same number whichever bins it is averaged from, and in this
simulation the responses read each unit’s departure from its own level,
which a yearly mean averages away. The weekly elastic net, which reads
every bin, scores 0.7010.789.

## Other responses, and other metrics

A response head defines the response type, fitting loss and conditions
under which a cell can be scored. In addition to `"presence_absence"`,
four response heads are available:

``` r

responses()
#> [1] "abundance"        "continuous"       "count"            "ordinal"         
#> [5] "presence_absence"
```

``` python
ts.responses()
#> ['abundance', 'continuous', 'count', 'ordinal', 'presence_absence']
```

A count is fitted under the Poisson deviance. The counts below are drawn
from each unit’s mean reading, and the forest fits them with no change
to the learner.

``` r

week <- grain_matrix(sim$readings, unit, time, reading, grain = "week")
rate <- exp(0.5 + 0.8 * as.numeric(scale(apply(week, 1L, mean))))
counts <- matrix(rpois(150, rate), ncol = 1L, dimnames = list(rownames(sim$y), "count"))
cf <- fit_learner(forest(), week_train, counts[train, , drop = FALSE], response = "count")
round(head(predict(cf, week_test), 3), 2)
#>            count
#> d001u00076  2.26
#> d001u00077  1.23
#> d001u00078  1.29
```

``` python
week = ts.grain_matrix(sim.readings, "unit", "time", "reading", grain="week")
level = week.values.mean(axis=(1, 2))
rate = np.exp(0.5 + 0.8 * (level - level.mean()) / level.std(ddof=1))
rng = np.random.default_rng(1)
counts = ts.Response(rng.poisson(rate).reshape(-1, 1).astype(float), y.units, ("count",))
counts_train = counts.take_units(np.arange(75))
cf = ts.fit_learner(ts.forest(), week_train, counts_train, response="count")
cf.predict(week_test)[:3].round(2)
#> array([[0.65],
#>        [2.66],
#>        [1.43]])
```

[`maxent()`](https://gillescolling.com/timesift/reference/maxent.md),
[`envelope()`](https://gillescolling.com/timesift/reference/envelope.md),
[`discriminant()`](https://gillescolling.com/timesift/reference/discriminant.md)
and
[`hierarchical()`](https://gillescolling.com/timesift/reference/hierarchical.md)
need a presence-absence response and refuse any other head:

``` r

fit_learner(maxent(), week_train, counts[train, , drop = FALSE], response = "count")
#> Error:
#> ! maxnet fits a presence-absence response, under a head whose loss is the binary cross-entropy; this head's loss is poisson_deviance.
```

``` python
ts.fit_learner(ts.maxent(), week_train, counts_train, response="count")
#> ValueError: maxnet fits a presence-absence response, under a head whose loss is the binary cross-entropy; this head's loss is 'poisson_deviance'.
```

[`register_response()`](https://gillescolling.com/timesift/reference/register_response.md)
adds a head. Its specification names `prepare`, `activation`, `loss`,
`metric` and `cells`, and optionally `weights`. The activation and the
loss are each one of the three the learners know: `"sigmoid"`,
`"identity"` or `"exp"`, and `"binary_cross_entropy"`, `"squared_error"`
or `"poisson_deviance"`. The shipped presence-absence head weights each
presence by the ratio of absences to presences among the fitting units,
capped at 50 by default, through
[`positive_weights()`](https://gillescolling.com/timesift/reference/positive_weights.md).
The weight of a presence in each response of this record:

``` r

w <- positive_weights(sim$y)
apply(w, 2L, max)
#>      v01      v02      v03      v04 
#> 2.260870 1.727273 2.750000 1.941176
```

``` python
w = ts.positive_weights(y)
w.max(axis=0)
#> array([2.125     , 2.        , 2.40909091, 2.125     ])
```

A head with a different cap reaches every learner through one
registration. With a cap of 2, the presences of `v01` and `v03` weigh 2
and those of `v02` and `v04` keep the weights above.

``` r

register_response("presence_absence_cap2", list(
  prepare = function(y) {
    y <- as.matrix(y)
    storage.mode(y) <- "double"
    if (!all(y %in% c(0, 1))) stop("a presence-absence response is 0/1.", call. = FALSE)
    y
  },
  activation = "sigmoid",
  loss = "binary_cross_entropy",
  metric = "roc_auc",
  cells = function(y, folds) scorable_cells(y, folds),
  weights = function(y, fitting) positive_weights(y, cap = 2, fitting = fitting)
))
```

``` python
ts.register_response("presence_absence_cap2", {
    "prepare": lambda y: ts.as_response(y).check_presence_absence(),
    "activation": "sigmoid",
    "loss": "binary_cross_entropy",
    "metric": "roc_auc",
    "cells": lambda y, folds: ts.scorable_cells(y, folds),
    "weights": lambda y, fitting: ts.positive_weights(y, cap=2, fitting=fitting),
});
```

A metric is a function of the observed values and the predictions of one
held-out cell.
[`register_metric()`](https://gillescolling.com/timesift/reference/register_metric.md)
adds one by name, and `timesift(metric = )` then chooses and reports by
it. The Brier score below has its sign reversed, so that a higher score
is better, as the choice of a candidate assumes of every metric.

``` r

register_metric("neg_brier", function(y, p) -mean((y - p)^2))
capped <- timesift(targets, sim$readings, y = starts_with("v"), id = unit, time = time,
                   learners = c(elasticnet(), forest()), sift = grains("month"),
                   response = "presence_absence_cap2", metric = "neg_brier",
                   resampling = cv(v = 5), n_inner = NULL, ensemble = FALSE, verbose = FALSE)
summary(capped)
#> timesift  150 targets, 4 responses, 5-fold random CV, neg_brier
#> 
#> candidates, scored on the outer folds
#> candidate                    mean    won  responses
#> forest / month             -0.259      0  separate
#> elasticnet / month         -0.239      4  separate
#> 
#> choice on every target  elasticnet / month
```

``` python
ts.register_metric("neg_brier", lambda y, p: -np.mean((y - p) ** 2));
capped = ts.timesift(targets, sim.readings, y="v*", id="unit", time="time",
                     learners=[ts.elasticnet(), ts.forest()], sift=ts.grains("month"),
                     response="presence_absence_cap2", metric="neg_brier",
                     resampling=ts.cv(v=5), n_inner=None, ensemble=False, verbose=False)
print(ts.summary(capped))
#> timesift  150 targets, 4 responses, 5-fold random CV, neg_brier
#>
#> candidates, scored on the outer folds
#> candidate                    mean    won  responses
#> forest / month             -0.240      2  separate
#> elasticnet / month         -0.238      2  separate
#>
#> choice on every target  elasticnet / month
```

The run is chosen and reported by the Brier score under the capped
weights. The elastic net, at -0.239-0.238, scored above the forest, at
-0.259-0.240, and scored highest on all four2 of the four responses.

## Choosing learners

The cost of a learner depends on how many columns its representation
flattens to, and on how much of its fit is an inner search. The timings
below fit each tabular learner once on all 150 units and four responses
at the weekly grain, which flattens to 53 columns.

``` r

timed <- list(elasticnet = elasticnet(), `elasticnet(tol = 1e-4)` = elasticnet(tol = 1e-4),
              linear = linear(), tree = tree(), forest = forest(), boosting = boosting(),
              xgboost = boosting(method = "xgboost"), maxent = maxent(),
              envelope = envelope(), mars = mars(), discriminant = discriminant(),
              perceptron = perceptron(), hierarchical = hierarchical())
seconds <- vapply(timed, function(l) {
  system.time(fit_learner(l, week, sim$y))[["elapsed"]]
}, numeric(1))
sort(round(seconds, 2), decreasing = TRUE)
#>                 linear             elasticnet                 forest 
#>                  10.08                   3.91                   0.46 
#>                 maxent elasticnet(tol = 1e-4)           discriminant 
#>                   0.43                   0.31                   0.18 
#>                xgboost             perceptron               boosting 
#>                   0.09                   0.06                   0.03 
#>                   tree                   mars           hierarchical 
#>                   0.02                   0.02                   0.02 
#>               envelope 
#>                   0.00
```

``` python
timed = {
    "elasticnet": ts.elasticnet(), "elasticnet(tol=1e-4)": ts.elasticnet(tol=1e-4),
    "linear": ts.linear(), "tree": ts.tree(), "forest": ts.forest(), "boosting": ts.boosting(),
    "xgboost": ts.boosting(method="xgboost"), "maxent": ts.maxent(),
    "envelope": ts.envelope(), "mars": ts.mars(), "discriminant": ts.discriminant(),
    "perceptron": ts.perceptron(), "hierarchical": ts.hierarchical(),
}

def seconds_to_fit(learner, x=week):
    start = time.perf_counter()
    ts.fit_learner(learner, x, y)
    return time.perf_counter() - start

seconds = pd.Series({name: seconds_to_fit(l) for name, l in timed.items()})
seconds.round(2).sort_values(ascending=False)
#> linear                  4.89
#> elasticnet              3.85
#> maxent                  0.57
#> forest                  0.46
#> discriminant            0.19
#> elasticnet(tol=1e-4)    0.18
#> xgboost                 0.09
#> perceptron              0.03
#> boosting                0.03
#> tree                    0.02
#> mars                    0.01
#> hierarchical            0.01
#> envelope                0.00
#> dtype: float64
```

[`additive()`](https://gillescolling.com/timesift/reference/additive.md)
is not in that list, because at its default `k = 10` an additive model
of 53 columns needs more coefficients than there are units:

``` r

fit_learner(additive(), week, sim$y)
#> Error:
#> ! an additive model of 478 coefficients has more than the 150 units it is fitted to; fewer columns or a smaller `k` fit.
```

``` python
ts.fit_learner(ts.additive(), week, y)
#> ValueError: an additive model of 478 coefficients has more than the 150 units it is fitted to; fewer columns or a smaller `k` fit.
```

At the monthly grain, 12 columns, it fits, and its cost depends on `k`:

``` r

month <- grain_matrix(sim$readings, unit, time, reading, grain = "month")
additive_seconds <- c(
  `k = 10` = system.time(fit_learner(additive(), month, sim$y))[["elapsed"]],
  `k = 5` = system.time(fit_learner(additive(k = 5L), month, sim$y))[["elapsed"]])
round(additive_seconds, 2)
#> k = 10  k = 5 
#>  21.86   0.54
```

``` python
month = ts.grain_matrix(sim.readings, "unit", "time", "reading", grain="month")
additive_seconds = pd.Series({"k = 10": seconds_to_fit(ts.additive(), month),
                              "k = 5": seconds_to_fit(ts.additive(k=5), month)})
additive_seconds.round(2)
#> k = 10    12.11
#> k = 5      0.70
#> dtype: float64
```

With `k = 10` the fit took 21.912.1 seconds, and with `k = 5` it took
0.50.7 seconds.

The elastic net includes every predictor and uses a penalty to limit
coefficient size. It chooses that penalty within the training data. At
its default tolerance it took 3.93.9 seconds here; `tol = 1e-4` took
0.30.2 seconds and gave the same held-out AUC on the split abovea
held-out AUC of0.727on the split above, against0.729at the default.
[`linear()`](https://gillescolling.com/timesift/reference/linear.md)
took 10.14.9 seconds. Its search refits the model for each term it could
add or remove, with 106 possible terms at the weekly grain. A coarser
grain or a bounded search, such as
`linear(select = "forward", terms = "column", max_terms = 3)`, reduces
the number of fits. Every other tabular learner took at most 0.460.57
seconds.

What a learner can be fitted on depends on the number of units against
the number of columns its representation flattens to:

- The additive model holds one coefficient for the intercept and `k - 1`
  for each column, and refuses a fit with more coefficients than units.
  On a small sample it needs a coarse grain or a small `k`, and its cost
  grows with `k`, as the timings above show.
- `linear(select = "none")` fits every term without selecting, and over
  more columns than units it separates any response it is given. It is
  meant for a coarse grain such as `data = grain("season")`.
- The envelope keeps a unit inside only where every column agrees, so
  each added column leaves fewer presences inside. It too is meant for a
  coarse grain.
- The elastic net, the forest and boosting keep a wide block fittable on
  few units through the penalty, the column draw and the shrinkage,
  which makes them the first learners to run at a fine grain on a small
  sample.
- [`maxent()`](https://gillescolling.com/timesift/reference/maxent.md)
  builds a hinge per column per knot, so its design grows with the
  grain. A fit whose design would pass `max_design` gigabytes is refused
  with the size it would have taken, and a coarser grain or fewer
  classes brings it within the limit.
- [`perceptron()`](https://gillescolling.com/timesift/reference/perceptron.md)
  reads the columns in their own units unless `standardise = TRUE`, and
  a record in degrees saturates its hidden units sooner the wider its
  range.

The encoders fit every response through one network, so a rarer response
borrows from the common ones. That pooling is what they are for, and it
needs many responses and enough units to train a network on. In the
Schrankogel study, 894 plots and 101 species read hourly for three
years, a fully connected network on the series was level with a
penalised logistic model on 188 hand-built features, and the gain came
from convolution reading the series at a coarse grain. A study with few
units and few responses may gain little from the encoders over the
elastic net, and pays for the training.

A learner is the wrong choice where it cannot read the response or the
representation.
[`maxent()`](https://gillescolling.com/timesift/reference/maxent.md),
[`envelope()`](https://gillescolling.com/timesift/reference/envelope.md),
[`discriminant()`](https://gillescolling.com/timesift/reference/discriminant.md)
and
[`hierarchical()`](https://gillescolling.com/timesift/reference/hierarchical.md)
fit presence-absence alone. A tabular learner cannot read
[`native()`](https://gillescolling.com/timesift/reference/native.md),
and a sequence learner cannot read a
[`multigrain()`](https://gillescolling.com/timesift/reference/native.md)
block or a representation of one bin. A learner can also be handed a
reduction that has already discarded the signal: `channel_means` above
was fitted correctly and scored below 0.5, because a yearly mean keeps
nothing of a weekly departure. No learner recovers what the
representation dropped, which is why
[`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
compares learners across grains.
