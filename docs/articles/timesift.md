# Choosing how a record is read

Sensor measurements can be summarised as daily, weekly or monthly
values. `timesift` fits models using each of these summaries and
compares how well they predict observations excluded from fitting. The
time scale of a summary is called its *grain*.

This vignette uses a small simulated dataset to show how to prepare the
two input tables, compare grains and predict new observations.

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

## Two tables

`targets` contains one row per sampling unit, such as a study plot, and
the values to predict, called *responses*. Here the responses record
whether each species occurs on a plot. `series` contains the
time-stamped sensor readings for those plots. A shared plot identifier
links the two tables.

In Python either table can be a mapping of column names to arrays,
including a pandas or polars data frame. The Python block draws its own
record, so the values it prints differ from the ones quoted in the text,
which come from the R run.

``` r

hours <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24 * 120)
ids <- sprintf("p%03d", 1:60)
warmth <- rnorm(60)

series <- data.frame(
  plot = rep(ids, each = length(hours)),
  t = rep(hours, times = 60),
  temp = as.numeric(vapply(warmth, function(w) {
    1.5 * w + 6 * sin(seq_along(hours) / (24 * 40)) + rnorm(length(hours), sd = 12)
  }, numeric(length(hours))))
)

sign <- rep(c(1, -1), length.out = 6)
targets <- data.frame(plot = ids, elevation = 2000 + 300 * rnorm(60))
targets[paste0("sp", 1:6)] <- lapply(sign, function(s) rbinom(60, 1, plogis(3 * s * warmth)))
str(targets[1:4], give.attr = FALSE)
#> 'data.frame':    60 obs. of  4 variables:
#>  $ plot     : chr  "p001" "p002" "p003" "p004" ...
#>  $ elevation: num  2644 1978 2365 2160 1899 ...
#>  $ sp1      : int  1 0 0 1 0 0 0 1 1 0 ...
#>  $ sp2      : int  1 0 1 0 1 1 0 0 0 0 ...
```

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

Each plot has its own temperature offset. The hourly noise is an order
of magnitude larger than the differences between these offsets.

## One call

``` r

fit <- timesift(
  targets, series,
  y = starts_with("sp"),
  id = plot,
  time = t,
  static = elevation,
  sift = grains("day", "week", "month"),
  resampling = cv(v = 5),
  verbose = FALSE
)
fit
#> timesift  60 targets, 6 responses, 5-fold random CV, roc_auc
#> 
#> candidates, scored on the outer folds
#> candidate                    mean    won  responses
#> elasticnet / day            0.866      1  separate
#> elasticnet / week           0.897      3  separate
#> elasticnet / month          0.907      2  separate
#> 
#> procedure, chosen and weighted inside each outer training fold
#> selected                    0.908  se 0.022
#> ensemble                    0.907  se 0.016
#> selected elasticnet / month in 4, elasticnet / week in 1 of 5 folds
#> 
#> choice on every target  elasticnet / month
#> weights on every target  elasticnet / month 1.00
```

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
#> elasticnet / day            0.870      0  separate
#> elasticnet / week           0.899      3  separate
#> elasticnet / month          0.911      3  separate
#>
#> procedure, chosen and weighted inside each outer training fold
#> selected                    0.911  se 0.011
#> ensemble                    0.907  se 0.012
#> selected elasticnet / month in 5 of 5 folds
#>
#> choice on every target  elasticnet / month
#> weights on every target  elasticnet / month 0.96   elasticnet / week 0.04
```

The call builds a table of predictors at each time scale listed in
`sift`. By default, it fits a penalised logistic regression, which
estimates presence probabilities while limiting the size of its
coefficients. To evaluate predictions, the plots are divided into five
groups, called *folds*. Each fold is predicted by a model fitted on the
other four. This is five-fold cross-validation.

Each model and time-scale combination is a *candidate*. The candidate
rows compare scores on the same folds for the same responses. A response
in one fold is called a *cell* in the scoring output. `won` counts the
responses on which a candidate scored highest. `responses` indicates
whether the model fits responses jointly or separately. The default
score is the area under the receiver operating characteristic curve
(AUC), which measures how well predictions rank presences above
absences.

The `selected` row evaluates the process of choosing a candidate. The
`ensemble` row evaluates a weighted combination of candidates. Both use
an additional round of cross-validation within each training set: these
*inner folds* choose the candidate and estimate the weights, and the
*outer test fold* evaluates their predictions. Use these rows to report
predictive performance after selection or combination.

``` r

fit$estimate[c("arm", "metric", "score", "se", "lower", "upper")]
#>         arm               metric      score          se       lower      upper
#> 1  selected             accuracy  0.8777778 0.015908690  0.83688319  0.9186724
#> 2  selected    average_precision  0.9134774 0.018894456  0.86490764  0.9620471
#> 3  selected                 bias  1.0727116 0.056937161  0.92635001  1.2190733
#> 4  selected                boyce  0.6824181 0.055424952  0.53994375  0.8248925
#> 5  selected                  csi  0.7860714 0.024178549  0.72391849  0.8482244
#> 6  selected                  ets  0.6348278 0.041399419  0.52840719  0.7412484
#> 7  selected                  far  0.1187302 0.028711737  0.04492429  0.1925360
#> 8  selected                kappa  0.6367593 0.034685563  0.54759718  0.7259213
#> 9  selected         kappa_youden  0.7543294 0.032708510  0.67024955  0.8384093
#> 10 selected              neg_mae -0.3557994 0.019010008 -0.40466614 -0.3069326
#> 11 selected        neg_max_error -0.6121056 0.011515318 -0.64170664 -0.5825045
#> 12 selected              neg_mse -0.1542971 0.013224621 -0.18829204 -0.1203021
#> 13 selected neg_poisson_deviance -0.4858941 0.024935228 -0.54999210 -0.4217960
#> 14 selected             neg_rmse -0.3900576 0.015724765 -0.43047939 -0.3496358
#> 15 selected                   or 23.5000000 0.500000000 17.14689763 29.8531024
#> 16 selected     ordinal_accuracy  0.8138889 0.017435950  0.76906835  0.8587094
#> 17 selected           ordinal_f1  0.8197278 0.017391577  0.77502138  0.8644343
#> 18 selected    ordinal_precision  0.8219907 0.016217367  0.78030267  0.8636788
#> 19 selected       ordinal_recall  0.8180159 0.018711488  0.76991646  0.8661153
#> 20 selected                 orss  0.9890723 0.006914267  0.97129857  1.0068459
#> 21 selected              pearson  0.7009506 0.032456953  0.61751740  0.7843839
#> 22 selected                  pod  0.8964683 0.025170501  0.83176542  0.9611711
#> 23 selected                 pofd  0.1257143 0.039611921  0.02388860  0.2275400
#> 24 selected            r_squared  0.3437177 0.059344801  0.19116699  0.4962683
#> 25 selected              roc_auc  0.9084160 0.021620212  0.85283948  0.9639925
#> 26 selected                   sr  0.8812698 0.028711737  0.80746397  0.9550757
#> 27 selected                  tss  0.7707540 0.035825510  0.67866156  0.8628464
#> 28 ensemble             accuracy  0.8750000 0.011180340  0.84626002  0.9037400
#> 29 ensemble    average_precision  0.9143980 0.013535606  0.87960363  0.9491924
#> 30 ensemble                 bias  1.0679497 0.048673935  0.94282940  1.1930701
#> 31 ensemble                boyce  0.6897743 0.037156752  0.59425981  0.7852888
#> 32 ensemble                  csi  0.7776058 0.018333372  0.73047839  0.8247333
#> 33 ensemble                  ets  0.6189548 0.030714246  0.54000128  0.6979082
#> 34 ensemble                  far  0.1224339 0.023392168  0.06230238  0.1825653
#> 35 ensemble                kappa  0.6253307 0.016040524  0.58409721  0.6665642
#> 36 ensemble         kappa_youden  0.7477055 0.024184193  0.68553807  0.8098730
#> 37 ensemble              neg_mae -0.3530131 0.018303754 -0.40006442 -0.3059618
#> 38 ensemble        neg_max_error -0.6163724 0.011503915 -0.64594412 -0.5868006
#> 39 ensemble              neg_mse -0.1521954 0.011427574 -0.18157095 -0.1228199
#> 40 ensemble neg_poisson_deviance -0.4823667 0.022655663 -0.54060494 -0.4241285
#> 41 ensemble             neg_rmse -0.3879671 0.013953420 -0.42383548 -0.3520987
#> 42 ensemble                   or 23.5000000 0.500000000 17.14689763 29.8531024
#> 43 ensemble     ordinal_accuracy  0.8138889 0.017435950  0.76906835  0.8587094
#> 44 ensemble           ordinal_f1  0.8197278 0.017391577  0.77502138  0.8644343
#> 45 ensemble    ordinal_precision  0.8219907 0.016217367  0.78030267  0.8636788
#> 46 ensemble       ordinal_recall  0.8180159 0.018711488  0.76991646  0.8661153
#> 47 ensemble                 orss  0.9890723 0.006914267  0.97129857  1.0068459
#> 48 ensemble              pearson  0.7050684 0.027212866  0.63511551  0.7750213
#> 49 ensemble                  pod  0.8917063 0.022858665  0.83294628  0.9504664
#> 50 ensemble                 pofd  0.1257143 0.033072963  0.04069753  0.2107310
#> 51 ensemble            r_squared  0.3522563 0.051730730  0.21927827  0.4852344
#> 52 ensemble              roc_auc  0.9074901 0.016384905  0.86537134  0.9496088
#> 53 ensemble                   sr  0.8775661 0.023392168  0.81743466  0.9376976
#> 54 ensemble                  tss  0.7659921 0.027637201  0.69494838  0.8370358
fit$selected[c("fold", "candidate", "inner_score")]
#>   fold          candidate inner_score
#> 1    1 elasticnet / month   0.9035741
#> 2    2 elasticnet / month   0.8993175
#> 3    3  elasticnet / week   0.9297381
#> 4    4 elasticnet / month   0.9089008
#> 5    5 elasticnet / month   0.9144762
```

``` python
pd.DataFrame(fit.estimate)[["arm", "metric", "score", "se", "lower", "upper"]]
#>          arm                metric      score        se      lower      upper
#> 0   selected              accuracy   0.902778  0.011719   0.872652   0.932904
#> 1   selected     average_precision   0.908116  0.023171   0.848552   0.967679
#> 2   selected                  bias   1.032778  0.038854   0.932901   1.132654
#> 3   selected                 boyce   0.701781  0.015503   0.661930   0.741633
#> 4   selected                   csi   0.819987  0.026700   0.751353   0.888621
#> 5   selected                   ets   0.678483  0.034219   0.590519   0.766447
#> 6   selected                   far   0.102672  0.026525   0.034488   0.170856
#> 7   selected                 kappa   0.660423  0.024039   0.598629   0.722217
#> 8   selected          kappa_youden   0.796165  0.023657   0.735353   0.856978
#> 9   selected               neg_mae  -0.371196  0.013208  -0.405149  -0.337243
#> 10  selected         neg_max_error  -0.635774  0.007559  -0.655206  -0.616342
#> 11  selected               neg_mse  -0.162203  0.006823  -0.179742  -0.144664
#> 12  selected  neg_poisson_deviance  -0.513075  0.018045  -0.559461  -0.466689
#> 13  selected              neg_rmse  -0.401536  0.008560  -0.423540  -0.379532
#> 14  selected                    or  22.750000  1.030776  19.469609  26.030391
#> 15  selected      ordinal_accuracy   0.808333  0.009379   0.784224   0.832442
#> 16  selected            ordinal_f1   0.814673  0.010448   0.787816   0.841529
#> 17  selected     ordinal_precision   0.807500  0.009108   0.784086   0.830914
#> 18  selected        ordinal_recall   0.822665  0.012320   0.790995   0.854336
#> 19  selected                  orss   0.986042  0.005010   0.973164   0.998920
#> 20  selected               pearson   0.675293  0.014535   0.637930   0.712656
#> 21  selected                   pod   0.910992  0.015011   0.872405   0.949579
#> 22  selected                  pofd   0.093545  0.019192   0.044211   0.142879
#> 23  selected             r_squared   0.285434  0.029940   0.208471   0.362397
#> 24  selected               roc_auc   0.910976  0.011209   0.882161   0.939790
#> 25  selected                    sr   0.897328  0.026525   0.829144   0.965512
#> 26  selected                   tss   0.817447  0.019358   0.767686   0.867208
#> 27  ensemble              accuracy   0.897222  0.012485   0.865130   0.929315
#> 28  ensemble     average_precision   0.905069  0.021508   0.849781   0.960356
#> 29  ensemble                  bias   1.013267  0.045904   0.895268   1.131267
#> 30  ensemble                 boyce   0.692614  0.027483   0.621968   0.763260
#> 31  ensemble                   csi   0.810000  0.026807   0.741092   0.878908
#> 32  ensemble                   ets   0.665135  0.036505   0.571295   0.758975
#> 33  ensemble                   far   0.099101  0.028427   0.026027   0.172174
#> 34  ensemble                 kappa   0.663810  0.029921   0.586894   0.740725
#> 35  ensemble          kappa_youden   0.785917  0.025575   0.720174   0.851659
#> 36  ensemble               neg_mae  -0.370921  0.013551  -0.405754  -0.336088
#> 37  ensemble         neg_max_error  -0.647332  0.004716  -0.659455  -0.635209
#> 38  ensemble               neg_mse  -0.163398  0.007436  -0.182512  -0.144283
#> 39  ensemble  neg_poisson_deviance  -0.516055  0.018174  -0.562774  -0.469336
#> 40  ensemble              neg_rmse  -0.403008  0.009305  -0.426928  -0.379088
#> 41  ensemble                    or  23.125000  0.875000  20.340359  25.909641
#> 42  ensemble      ordinal_accuracy   0.811111  0.015316   0.771741   0.850481
#> 43  ensemble            ordinal_f1   0.817597  0.016364   0.775531   0.859662
#> 44  ensemble     ordinal_precision   0.810853  0.015969   0.769804   0.851902
#> 45  ensemble        ordinal_recall   0.825112  0.017178   0.780955   0.869270
#> 46  ensemble                  orss   0.983375  0.006059   0.967800   0.998951
#> 47  ensemble               pearson   0.672901  0.015095   0.634098   0.711704
#> 48  ensemble                   pod   0.896243  0.019159   0.846994   0.945492
#> 49  ensemble                  pofd   0.086878  0.020548   0.034058   0.139699
#> 50  ensemble             r_squared   0.281085  0.030892   0.201674   0.360497
#> 51  ensemble               roc_auc   0.906884  0.011914   0.876257   0.937510
#> 52  ensemble                    sr   0.900899  0.028427   0.827826   0.973973
#> 53  ensemble                   tss   0.809365  0.022716   0.750971   0.867759
pd.DataFrame(fit.selected)[["fold", "candidate", "inner_score"]]
#>    fold           candidate  inner_score
#> 0     1  elasticnet / month     0.883939
#> 1     2  elasticnet / month     0.899061
#> 2     3  elasticnet / month     0.897351
#> 3     4  elasticnet / month     0.908608
#> 4     5  elasticnet / month     0.896122
```

The interval describes variation across the six responses. All responses
use the same plots and folds, so the interval excludes uncertainty
shared through that sampling design.

Use `static` to name columns of `targets` that should enter the model as
predictors. Here it selects `elevation`; additional columns, such as
notes, are ignored.

``` r

fit$candidates[c("candidate", "representation", "bins", "channels", "status")]
#>            candidate representation bins channels status
#> 1   elasticnet / day            day  120        2 fitted
#> 2  elasticnet / week           week   18        2 fitted
#> 3 elasticnet / month          month    4        2 fitted
```

``` python
pd.DataFrame(fit.candidates)[["candidate", "representation", "bins", "channels"]]
#>             candidate representation  bins  channels
#> 0    elasticnet / day            day   120         2
#> 1   elasticnet / week           week    18         2
#> 2  elasticnet / month          month     4         2
```

The model now receives temperature and elevation. Each is stored as a
separate *channel*, a set of values over the time bins. A bin is one
interval, such as a calendar week. Elevation has the same value in every
bin for a given plot.

``` r

plot(fit)
```

![Mean AUC against representation, with the level the combination
reached drawn across it.](timesift_files/figure-html/plot-run-1.svg)

``` python
ts.plot(fit);
```

![Mean AUC against representation, with the level the combination
reached drawn across it.](timesift_files/py-figure-1.svg)

[`predict()`](https://rdrr.io/r/stats/predict.html) prepares the new
data using each candidate’s saved settings, applies models refitted on
all training plots, and combines their predictions using the fitted
ensemble weights. Set `candidate = "selected"` to use the candidate
chosen from all training plots.

``` r

p <- predict(fit, targets, series)
round(p[1:3, 1:4], 3)
#>        sp1   sp2   sp3   sp4
#> p001 0.290 0.803 0.396 0.741
#> p002 0.583 0.534 0.576 0.548
#> p003 0.170 0.841 0.334 0.774
```

``` python
p = fit.predict(targets, series)
p[:3, :4].round(3)
#> array([[0.535, 0.429, 0.629, 0.337],
#>        [0.644, 0.213, 0.772, 0.221],
#>        [0.588, 0.312, 0.63 , 0.381]])
```

`type = "binary"` cuts each response into presence and absence. The cut
is learned from the same candidate’s out-of-fold predictions of the
fit’s own targets, so it comes from predictions of plots the model had
not been fitted on.
[`decision_threshold()`](https://gillescolling.com/timesift/reference/kappa_score.md)
on the fit returns the cuts themselves, and `rule` picks how they are
learned: `"youden"` maximises sensitivity plus specificity, `"kappa"`
maximises Cohen’s kappa, and `"prevalence"` predicts as many presences
as were observed.

``` r

decision_threshold(fit, rule = "prevalence")
#>       sp1       sp2       sp3       sp4       sp5       sp6 
#> 0.5454921 0.5204788 0.5309027 0.5044327 0.5217123 0.5156719
predict(fit, targets, series, type = "binary", rule = "prevalence")[1:3, 1:4]
#>      sp1 sp2 sp3 sp4
#> p001   0   1   0   1
#> p002   1   1   1   1
#> p003   0   1   0   1
```

``` python
pd.Series(ts.decision_threshold(fit, rule="prevalence"))
#> sp1    0.520268
#> sp2    0.502498
#> sp3    0.523192
#> sp4    0.485687
#> sp5    0.513939
#> sp6    0.507207
#> dtype: float64
fit.predict(targets, series, type="binary", rule="prevalence")[:3, :4]
#> array([[1., 0., 1., 0.],
#>        [1., 0., 1., 0.],
#>        [1., 0., 1., 0.]])
```

A binary map is this prediction with one target per map cell, each
carrying a series of its own at the grain the fit reads.

## Representations

A representation stores the settings used to summarise a record. The
same settings are used to build model inputs during fitting and
prediction.

``` r

native()
#> <timesift representation> native 
#> kind    : grain (a sequence) 
#> grain   : native 
#> stats   : mean
grain("week", stats = c("cold_day", "mean", "warm_day"))
#> <timesift representation> week 
#> kind    : grain (a sequence) 
#> grain   : week 
#> stats   : cold_day, mean, warm_day
multigrain(c("week", "month"))
#> <timesift representation> multigrain 
#> kind    : multigrain (a block of features) 
#> grains  : week, month 
#> stats   : mean
lookback("30 days", bins = 3)
#> <timesift representation> 30 days x3 
#> kind    : lookback (a sequence) 
#> span    : 30 days in 3 bins ending 0 days before the target
#> stats   : mean
```

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

[`grains()`](https://gillescolling.com/timesift/reference/grains.md)
defines several time scales to compare. `grains("auto")` includes each
named scale that gives at least two time bins.
[`multigrain()`](https://gillescolling.com/timesift/reference/native.md)
puts summaries at several scales into one table. A *lookback window*
covers a fixed period before a prediction date, such as the four weeks
before a survey.
[`lookback()`](https://gillescolling.com/timesift/reference/native.md)
defines one window and
[`lookbacks()`](https://gillescolling.com/timesift/reference/grains.md)
defines several to compare; `target_time` identifies the date column.

By default, each learner runs on all compatible representations in the
set. Use `data` to assign a specific representation:

``` r

elasticnet(data = grain("month"))
#> <timesift learner> elasticnet 
#> reads   : tabular ; one model per response: yes, separate 
#> data    : month 
#> settings: alpha = 0.5, n_inner = 10, squares = TRUE, s = lambda.1se, n_lambda = 100, tol = 1e-08, threads = 1, seed = 1
```

``` python
ts.elasticnet(data=ts.grain("month"))
#> <timesift learner> elasticnet
#> reads   : tabular ; one model per response: yes, separate
#> data    : month
#> settings: alpha = 0.5, n_inner = 10, squares = True, s = lambda.1se, n_lambda = 100, tol = 1e-08, threads = 1, seed = 1
```

## Compatible model inputs

Learners accept either a table of predictors or a sequence of time bins.
Tabular learners reject
[`native()`](https://gillescolling.com/timesift/reference/native.md)
before the array is built. Sequence learners require more than one bin;
this is checked after building the array.

``` r

timesift(targets, series, y = starts_with("sp"), id = plot, time = t,
         learners = elasticnet(), sift = native())
#> Error:
#> ! no learner can read any representation in the sift:
#>   `elasticnet()` reads a tabular representation; `native()` gives it one column per reading. Use `grain()`, `multigrain()` or `lookback()`.
```

``` python
ts.timesift(targets, series, y="sp*", id="plot", time="t",
            learners=[ts.elasticnet()], sift=ts.native(), verbose=False)
#> ValueError: no learner can read any of the representations:
#>   elasticnet() reads a tabular representation; native gives it one column per reading. Use grain(), multigrain() or lookback().
```

In a set of representations, incompatible learner-representation pairs
are skipped and reported as `not applicable`. An incompatible
representation assigned explicitly through a learner’s `data` argument
raises an error.

## The split, and the cells a score is defined on

A *fold map* records which test fold each plot belongs to. Every
candidate uses the same map.
[`cv()`](https://gillescolling.com/timesift/reference/cv.md) distributes
plots across folds with similar values of a chosen variable.
[`grouped_cv()`](https://gillescolling.com/timesift/reference/cv.md)
keeps observations from the same group together, such as repeated
surveys of one plot. You can also supply fold assignments directly
through `resampling`.

``` r

fit$folds
#> <timesift folds> 60 units in 5 folds 
#> fold
#>  1  2  3  4  5 
#> 12 12 12 12 12
```

``` python
fit.folds
#> <timesift folds> 60 units in 5 folds
#> 1: 12  2: 12  3: 12  4: 12  5: 12
```

For a response to be scored in a fold, both presences and absences must
occur in the held-out data. A model fitted separately for that response
also needs both classes in its training data. The scorable-cell mask
identifies the response-fold pairs that meet these conditions.

``` r

fit$cells
#> <timesift cells> 30 cells over 6 variables 
#> scorable: 30 (100.0%); variables with at least one scorable fold: 6 of 6
```

``` python
fit.cells
#> <timesift cells> 30 cells over 6 variables
#> scorable: 30 (100.0%); variables with at least one scorable fold: 6 of 6
```

The mask is calculated from the responses and folds before fitting. All
candidates are evaluated on the same cells, so mean scores and paired
differences use matching observations.

## Learners, and how they are trained

A *learner* specifies how to fit and predict with a model.
[`elasticnet()`](https://gillescolling.com/timesift/reference/elasticnet.md),
[`linear()`](https://gillescolling.com/timesift/reference/linear.md) and
[`forest()`](https://gillescolling.com/timesift/reference/forest.md) use
a table of predictors. The neural networks
[`mlp()`](https://gillescolling.com/timesift/reference/torch_learners.md),
[`cnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
and
[`rescnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
model all responses together, sharing information between them.
[`mlp()`](https://gillescolling.com/timesift/reference/torch_learners.md)
uses a table;
[`cnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
and
[`rescnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
use the order of the time bins to detect patterns across neighbouring
periods. These three networks require the torch library.

The R and Python versions of
[`elasticnet()`](https://gillescolling.com/timesift/reference/elasticnet.md)
use the same compiled implementation to standardise predictors, fit the
models and choose the penalty by cross-validation. The `s` argument
selects the penalty: `"lambda.min"` gives the lowest cross-validation
error; `"lambda.1se"` chooses a stronger penalty within one standard
error of that minimum. A numeric value specifies the penalty directly.
`tol` controls when the fitting algorithm stops.

Set the network architecture in the learner constructor and shared
training settings in
[`train_control()`](https://gillescolling.com/timesift/reference/train_control.md).
Settings supplied to an individual learner override the corresponding
run-level settings.

``` r

train_control(epochs = 200L, device = "cpu")
#> <timesift control>
#>   epochs           200
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
cnn(channels = c(16L, 32L), epochs = 300L)
#> <timesift learner> cnn 
#> reads   : sequence ; one model per response: no, joint 
#> data    : every representation of the run 
#> settings: channels = 16/32, kernel = 7, dropout = 0.3 
#> training: epochs = 300 
#> needs   : torch
```

``` python
ts.train_control(epochs=200, device="cpu")
#> <timesift control>
#>   epochs           200
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
ts.cnn(channels=(16, 32), epochs=300)
#> <timesift learner> cnn
#> reads   : sequence ; one model per response: no, joint
#> data    : every representation of the run
#> settings: channels = 16/32, kernel = 7, dropout = 0.3
#> training: epochs = 300
#> needs   : torch
```

To add a model, supply functions for fitting and prediction. It will use
the same folds and scoring rules as the built-in models.
[`as.matrix()`](https://rdrr.io/r/base/matrix.html)`flatten()` converts
the input to a table with one predictor per bin and channel. A fitting
function with a `weights` argument receives weights for the
observations; the default presence-absence weights give more weight to
rare presences.

``` r

nearest_neighbour <- learner(
  "1nn",
  fit = function(x, y, ...) list(x = as.matrix(x), y = y),
  predict = function(model, x) {
    m <- as.matrix(x)
    d <- as.matrix(dist(rbind(m, model$x)))[seq_len(nrow(m)), -seq_len(nrow(m))]
    model$y[apply(d, 1, which.min), , drop = FALSE]
  }
)

weighted_glm <- learner(
  "weighted_glm",
  fit = function(x, y, weights, ...) {
    f <- as.data.frame(as.matrix(x))
    lapply(seq_len(ncol(y)), function(j) {
      glm(y[, j] ~ ., data = f, weights = weights[, j], family = quasibinomial(),
          control = glm.control(maxit = 100))
    })
  },
  predict = function(model, x) {
    f <- as.data.frame(as.matrix(x))
    vapply(model, function(m) predict(m, f, type = "response"), numeric(nrow(f)))
  },
  data = grain("month")
)

both <- timesift(targets, series, y = starts_with("sp"), id = plot, time = t,
                 learners = c(elasticnet(), nearest_neighbour, weighted_glm),
                 sift = grains("week", "month"), resampling = cv(v = 5), verbose = FALSE)
summary(both)
#> timesift  60 targets, 6 responses, 5-fold random CV, roc_auc
#> 
#> candidates, scored on the outer folds
#> candidate                      mean    won  responses
#> 1nn / week                    0.720      0  separate
#> 1nn / month                   0.772      0  separate
#> elasticnet / week             0.897      2  separate
#> weighted_glm / month          0.904      3  separate
#> elasticnet / month            0.907      1  separate
#> 
#> procedure, chosen and weighted inside each outer training fold
#> selected                      0.908  se 0.022
#> ensemble                      0.896  se 0.016
#> selected elasticnet / month in 4, elasticnet / week in 1 of 5 folds
#> 
#> choice on every target  elasticnet / month
#> weights on every target  weighted_glm / month 0.66   elasticnet / month 0.30   1nn / month 0.04
```

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
                   learners=[ts.elasticnet(), nearest_neighbour, weighted_logistic],
                   sift=ts.grains("week", "month"), resampling=ts.cv(v=5), verbose=False)
print(ts.summary(both))
#> timesift  60 targets, 6 responses, 5-fold random CV, roc_auc
#>
#> candidates, scored on the outer folds
#> candidate                           mean    won  responses
#> 1nn / week                         0.744      0  joint
#> 1nn / month                        0.757      0  joint
#> elasticnet / week                  0.899      3  separate
#> weighted_logistic / month          0.900      0  separate
#> elasticnet / month                 0.912      3  separate
#>
#> procedure, chosen and weighted inside each outer training fold
#> selected                           0.912  se 0.011
#> ensemble                           0.901  se 0.009
#> selected elasticnet / month in 5 of 5 folds
#>
#> choice on every target  elasticnet / month
#> weights on every target  weighted_logistic / month 0.71   elasticnet / week 0.24   1nn / week 0.05
```

## The combination

The ensemble combines *out-of-fold predictions*: predictions for
observations excluded from the model’s training data. The `"stack"`
method chooses non-negative weights that sum to one to minimise
prediction error, using the loss function defined for the response type.
Other methods take the mean, median, score-weighted mean or the share of
models predicting presence (`"committee"`). The weights below use
out-of-fold predictions for all targets and are used by
[`predict()`](https://rdrr.io/r/stats/predict.html). The reported
ensemble score uses weights estimated separately within each outer
training fold, stored in `fit$fold_weights``fit.fold_weights`.

``` r

ensemble_weights(fit)
#>   elasticnet / day  elasticnet / week elasticnet / month 
#>      3.375410e-139       2.476064e-23       1.000000e+00
```

``` python
pd.Series(ts.ensemble_weights(fit))
#> elasticnet / day      5.967019e-269
#> elasticnet / week      3.517667e-02
#> elasticnet / month     9.648233e-01
#> dtype: float64
```

The weights determine each candidate’s contribution to the ensemble
prediction. A candidate with zero weight contributes nothing. Compare
the ensemble score with the individual candidate scores to see whether
combining their predictions improved performance in this run.

## TSS and threshold selection

The true skill statistic (TSS) is overestimated when the threshold that
maximises it is chosen on the same held-out units the score is then read
from. The overestimate grows as the number of presences shrinks.
[`tss_inflation()`](https://gillescolling.com/timesift/reference/tss_inflation.md)
estimates the TSS a given sampling design would report for a known
population skill, here the presence counts in your response-fold cells.

``` r

tss_inflation(fit$y, fit$folds, skill = c(0.6, 0.9), replicates = 100)
#>   skill  reported  inflation     lower     upper replicates
#> 1   0.6 0.7581421 0.15814206 0.6931759 0.8198188        100
#> 2   0.9 0.9664636 0.06646362 0.9361726 0.9855159        100
```

``` python
pd.DataFrame(ts.tss_inflation(fit.y, fit.folds, skill=(0.6, 0.9), replicates=100))
#>    skill  reported  inflation     lower     upper  replicates
#> 0    0.6  0.769709   0.169709  0.718532  0.835235         100
#> 1    0.9  0.966596   0.066596  0.942541  0.987280         100
```

[`tss_inflation()`](https://gillescolling.com/timesift/reference/tss_inflation.md)
averages the bias over simulated predictions, and
[`implied_skill()`](https://gillescolling.com/timesift/reference/implied_skill.md)
estimates the population skill consistent with an observed score under
that simulation. The bias depends on the distribution of predictions as
well as the presence counts. It can therefore differ between equally
skilled candidates evaluated on the same cells and affect their paired
TSS difference. The default scoring metric is area under the receiver
operating characteristic curve (AUC).

## The arrays on their own

[`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md)
groups readings into calendar intervals and summarises each interval. It
returns the model inputs so you can inspect them or fit a model
separately.

``` r

x <- grain_matrix(series, plot, t, temp, grain = "week",
                  stats = c("cold_day", "mean", "warm_day"))
x
#> <timesift matrix> 60 units x 18 bins x 3 channels 
#> grain: week   stats: cold_day, mean, warm_day 
#> from  : 2021-08-30 to 2021-12-27
```

``` python
x = ts.grain_matrix(series, "plot", "t", "temp", grain="week",
                    stats=("cold_day", "mean", "warm_day"))
x
#> <timesift matrix> 60 units x 18 bins x 3 channels
#> grain: week  stats: cold_day, mean, warm_day
#> from  : 2021-08-30T00:00:00Z to 2021-12-27T00:00:00Z
```

Bins follow the calendar: months have 28 to 31 days, and weeks begin on
Monday.

``` r

attr(grain_matrix(series, plot, t, temp, grain = "month"), "bin_n")[1, ]
#> 2021-09-01T00:00:00Z 2021-10-01T00:00:00Z 2021-11-01T00:00:00Z 
#>                  720                  744                  720 
#> 2021-12-01T00:00:00Z 
#>                  696
```

``` python
ts.grain_matrix(series, "plot", "t", "temp", grain="month").bin_n[0]
#> array([720, 744, 720, 696], dtype=int32)
```

### Daily and individual extremes

`min` and `max` return the lowest and highest individual reading in each
bin. `cold_day` and `warm_day` return the lowest and highest daily mean.
`mean_daily_min` and `mean_daily_max` average the daily minima and
maxima within the bin.

``` r

week <- grain_matrix(series, plot, t, temp, grain = "week",
                     stats = c("min", "mean_daily_min", "cold_day", "mean",
                               "warm_day", "mean_daily_max", "max"))
round(week[1, 1, ], 2)
#>            min mean_daily_min       cold_day           mean       warm_day 
#>         -23.41         -17.61          -6.76          -0.17           4.98 
#> mean_daily_max            max 
#>          24.02          27.89
```

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

[`calendar_channels()`](https://gillescolling.com/timesift/reference/calendar_channels.md)
adds values describing when each bin occurs within the year. This lets a
neural network use the season of an event even when it averages
information across time later in the network.

``` r

week_mean <- grain_matrix(series, plot, t, temp, grain = "week")
dimnames(bind_channels(week_mean, calendar_channels(week_mean)))[[3]]
#> [1] "mean"     "year_sin" "year_cos"
```

``` python
week_mean = ts.grain_matrix(series, "plot", "t", "temp", grain="week")
ts.bind_channels(week_mean, ts.calendar_channels(week_mean)).stats
#> ('mean', 'year_sin', 'year_cos')
```

## One grain at a time

[`grain_ladder()`](https://gillescolling.com/timesift/reference/grain_ladder.md)
fits and evaluates each learner at each grain using arrays you have
already built. All combinations use the same folds and eligible
response-fold pairs.

``` r

set <- grain_matrix(series, plot, t, temp, grain = c("day", "week", "month"))
lad <- grain_ladder(set, fit$y, elasticnet(), folds = fit$folds, verbose = FALSE)
summary(lad)
#>      learner grain     score n_variable  best
#> 1 elasticnet   day 0.8663062          6 FALSE
#> 2 elasticnet  week 0.8970888          6 FALSE
#> 3 elasticnet month 0.9074901          6  TRUE
```

``` python
grain_set = ts.grain_matrix(series, "plot", "t", "temp", grain=("day", "week", "month"))
lad = ts.grain_ladder(grain_set, fit.y, [ts.elasticnet()], folds=fit.folds, verbose=False)
pd.DataFrame(lad.summary())
#>       learner  grain     score  n_variable   best
#> 0  elasticnet    day  0.869715           6  False
#> 1  elasticnet   week  0.899052           6  False
#> 2  elasticnet  month  0.912017           6   True
```

[`paired_contrast()`](https://gillescolling.com/timesift/reference/paired_contrast.md)
calculates the score difference between two candidates in each shared
cell, averages across folds within each response, and summarises those
response-level differences. The interval and rank test use responses as
replicates. With six responses, the interval is wide and the test has
limited p-value resolution.

``` r

paired_contrast(lad, "month|elasticnet", "day|elasticnet")
#>                  a              b       diff     center       lower      upper
#> 1 month|elasticnet day|elasticnet 0.04118386 0.04118386 -0.01256729 0.09493501
#>   n_variable n_cell n_favour p_value p_method  interval
#> 1          6     30        5 0.09375    exact variables
```

``` python
pd.Series(ts.paired_contrast(lad, "month|elasticnet", "day|elasticnet"))
#> a             month|elasticnet
#> b               day|elasticnet
#> diff                  0.042303
#> center                0.042303
#> lower                 0.003589
#> upper                 0.081016
#> n_variable                   6
#> n_cell                      30
#> n_favour                     6
#> p_value                0.03125
#> p_method                 exact
#> interval             variables
#> dtype: object
```

[`grain_contrasts()`](https://gillescolling.com/timesift/reference/grain_contrasts.md)
compares each grain with the best-scoring grain using a mixed model,
which accounts for differences among responses and folds. It adjusts the
comparisons for testing several grains.It needs scipy, available through
the `contrasts` extra.

``` r

grain_contrasts(lad)
#>      learner grain reference        diff       lower       upper    p_value
#> 1 elasticnet   day     month -0.04118386 -0.08790089 0.005533166 0.09132994
#> 2 elasticnet  week     month -0.01040123 -0.05711826 0.036315794 0.83437774
```

``` python
pd.DataFrame(ts.grain_contrasts(lad))
#>       learner grain reference      diff     lower     upper   p_value
#> 0  elasticnet   day     month -0.042303 -0.088449  0.003844  0.076834
#> 1  elasticnet  week     month -0.012965 -0.059111  0.033181  0.752021
```

## Identifying influential bins

[`occlusion()`](https://gillescolling.com/timesift/reference/occlusion.md)
measures how much a score changes when the information in one time bin
is disrupted. It uses the saved models for each fold and evaluates the
modified inputs on the held-out observations.

``` r

kept <- timesift(targets, series, y = starts_with("sp"), id = plot, time = t,
                 sift = grains("month"), resampling = cv(v = 5), ensemble = FALSE,
                 keep_fits = TRUE, verbose = FALSE)
weight <- occlusion(kept, "elasticnet / month", permutations = 5)
head(aggregate(weight ~ part, weight, mean), 4)
#>                   part     weight
#> 1 2021-09-01T00:00:00Z 0.05928263
#> 2 2021-10-01T00:00:00Z 0.09155445
#> 3 2021-11-01T00:00:00Z 0.04998280
#> 4 2021-12-01T00:00:00Z 0.01739462
```

``` python
kept = ts.timesift(targets, series, y="sp*", id="plot", time="t",
                   sift=ts.grains("month"), resampling=ts.cv(v=5), ensemble=False,
                   keep_fits=True, verbose=False)
profile = ts.occlusion(kept, "elasticnet / month", permutations=5)
weight = pd.DataFrame(profile["weight"], index=profile["part"], columns=profile["variable"])
weight.mean(axis=1).head(4)
#> 2021-09-01T00:00:00Z    0.092622
#> 2021-10-01T00:00:00Z    0.078312
#> 2021-11-01T00:00:00Z    0.022786
#> 2021-12-01T00:00:00Z    0.032392
#> dtype: float64
```

Occluding a whole channel measures how predictions depend on a
particular statistic, such as the mean or the coldest day.

## Where next

Each part of this path has an article of its own on the package site:

- [Representing a
  record](https://gillescolling.com/timesift/articles/representations.html):
  every grain and statistic, calendars of one’s own, partial bins,
  coverage and lookbacks.
- [Learners](https://gillescolling.com/timesift/articles/learners.html):
  the shipped learners, training the encoders, tuning, and registering a
  learner, a response or a metric.
- [Finding the
  grain](https://gillescolling.com/timesift/articles/comparing-grains.html):
  the ladder, the contrasts and the nested selection on a simulated
  record whose grain is known.
- [Ensembles, thresholds and
  maps](https://gillescolling.com/timesift/articles/prediction.html):
  spatial and environmental folds, the combiners, thresholds and
  projection onto rasters.
- [Coming from
  biomod2](https://gillescolling.com/timesift/articles/biomod2.html) and
  [Troubleshooting](https://gillescolling.com/timesift/articles/troubleshooting.html).
