# Coming from biomod2

biomod2 fits several species distribution algorithms to one table of
predictors and combines their predictions. When the predictors come from
a sensor record, such as hourly soil temperature or daily air
temperature, that table is already a summary of the record, for example
monthly means or growing-degree-days. `timesift` fits the same
algorithms to the record summarised at every temporal grain it supports,
scores all of them on the same held-out folds, and then combines them.

Each algorithm is implemented once, in a compiled core that the R and
the Python interface both call, so a model fitted from either language
gives the same predictions. Each core reproduces the package biomod2
itself calls (rpart, randomForest, gbm, xgboost, maxnet, MASS, earth,
mda, mgcv), and the test fixtures check it against that package’s
output. `timesift` calls neither biomod2 nor these packages when it fits
a model.

## The mapping

Each biomod2 algorithm corresponds to a learner constructor, and a
variant of an algorithm is an argument of that constructor.

| biomod2 | `timesift` | notes |
|----|----|----|
| `CTA` | [`tree()`](https://gillescolling.com/timesift/reference/tree.md) | rpart’s rules; `prune = "se_sum"` is biomod2’s pruning |
| `RF` | [`forest()`](https://gillescolling.com/timesift/reference/forest.md) | randomForest’s defaults |
| `RFd` | `forest(balance = TRUE)` | each tree draws as many units from each class as the smaller class holds |
| `GBM` | [`boosting()`](https://gillescolling.com/timesift/reference/boosting.md) | gbm’s trees and Newton step |
| `XGBOOST` | `boosting(method = "xgboost")` | xgboost’s exact greedy trees; `lambda` and `gamma` are its penalties |
| `MAXNET`, `MAXENT` | [`maxent()`](https://gillescolling.com/timesift/reference/maxent.md) | `classes`, `regmult` and `knots` are maxnet’s; the model is the background formulation by default |
| `GLM` | [`linear()`](https://gillescolling.com/timesift/reference/linear.md) | `y ~ x + I(x^2)` over every column, searched by AIC as [`MASS::stepAIC()`](https://rdrr.io/pkg/MASS/man/stepAIC.html) does |
| `GAM` | [`additive()`](https://gillescolling.com/timesift/reference/additive.md) | mgcv’s `gam(method = "GCV.Cp")`, `k = 10` |
| `MARS` | [`mars()`](https://gillescolling.com/timesift/reference/mars.md) | earth’s forward and pruning passes, refitted as a logistic model |
| `FDA` | [`discriminant()`](https://gillescolling.com/timesift/reference/discriminant.md) | mda’s `fda(method = mars)` with the probit recalibration biomod2 applies |
| `SRE` | [`envelope()`](https://gillescolling.com/timesift/reference/envelope.md) | `quantile = 0.025`, as `bm_SRE()` |
| `ANN`, `DNN` | [`mlp()`](https://gillescolling.com/timesift/reference/torch_learners.md) | a torch encoder with a joint head over all responses |

For `ANN` and `DNN` the table matches the network architecture, and the
fitted networks differ. biomod2 fits `ANN` with nnet, which minimises an
L2-penalised loss by BFGS, and `DNN` with cito, which uses a stochastic
optimiser.
[`mlp()`](https://gillescolling.com/timesift/reference/torch_learners.md)
trains with AdamW, whose weight decay penalises the weights differently,
so the same architecture gives a comparable network but not the same
one. In
[`mlp()`](https://gillescolling.com/timesift/reference/torch_learners.md),
`hidden` sets the layer widths, `activation` the nonlinearity and
`dropout` the regularisation. The training settings are arguments of
[`train_control()`](https://gillescolling.com/timesift/reference/train_control.md),
which
[`mlp()`](https://gillescolling.com/timesift/reference/torch_learners.md)
also accepts directly. biomod2’s tuned `ANN` (`size = 5`, `decay = 0.1`,
`maxit = 200`) corresponds to
`mlp(hidden = 5, epochs = 200, weight_decay = 0.1)`, and its tuned `DNN`
(`hidden = c(100, 100)`, `activation = "selu"`, 150 epochs, batch size
100) to
`mlp(hidden = c(100, 100), activation = "selu", epochs = 150, batch_size = 100)`.

biomod2 ships two option sets: its defaults and a tuned set called
`"bigboss"`. Where the two differ, the learners offer both, as
`preset = "default"` (the defaults of the fitting package, which
biomod2’s default set uses) and `preset = "bigboss"`. For example,
`tree(preset = "bigboss")` sets `min_split = 5`, `min_leaf = 5`,
`cp = 0.001`, `max_depth = 10` and five inner folds. A setting given
explicitly overrides either preset.
[`mars()`](https://gillescolling.com/timesift/reference/mars.md),
[`discriminant()`](https://gillescolling.com/timesift/reference/discriminant.md)
and
[`additive()`](https://gillescolling.com/timesift/reference/additive.md)
have a single set, because biomod2 uses the same settings for them in
both.

## One run

The example uses a simulated record: 200 units with readings every six
hours for a year, and three responses that depend on the record through
a lagged mechanism acting at weekly grain. Because the simulation puts
the signal at the week, the comparison below can be checked against it.

``` r

sim <- simulate_records(n = 200L, mechanism = "lag", variables = 3L, days = 365L,
                        step_hours = 6, prevalence = 0.25, auc = 0.85, seed = 4L)
targets <- data.frame(unit = rownames(sim$y), sim$y)
str(sim$readings)
#> 'data.frame':    292000 obs. of  3 variables:
#>  $ unit   : chr  "d001u00001" "d001u00002" "d001u00003" "d001u00004" ...
#>  $ time   : POSIXct, format: "2021-09-01 00:00:00" "2021-09-01 00:00:00" ...
#>  $ reading: num  9.09 11.63 7.07 8.92 9.2 ...
```

`models` is a named list, so each candidate is reported under its
biomod2 name. A learner without a `data =` argument runs at every grain
listed in `sift`; a learner with `data =` runs only at that
representation.

``` r

models <- list(
  CTA = tree(),
  RF = forest(),
  RFd = forest(balance = TRUE),
  GBM = boosting(),
  XGBOOST = boosting(method = "xgboost"),
  MAXNET = maxent(),
  GLM = linear(),
  GAM = additive(data = grain("season"), k = 5L),
  MARS = mars(),
  FDA = discriminant(),
  SRE = envelope(data = grain("season"))
)
```

Two learners are fixed to the seasonal grain. An additive model fits
`k - 1` coefficients per column. At weekly grain this record has 53
columns, which would need 478 coefficients for 160 training units, and
[`additive()`](https://gillescolling.com/timesift/reference/additive.md)
refuses such a fit; the four seasonal columns need 17. The envelope
excludes one presence in twenty at each end of every column, so each
added column leaves fewer units inside all the bands, and the envelope
is suited to a coarse grain.

``` r

fit <- timesift(
  targets, sim$readings,
  y = starts_with("v"), id = unit, time = time,
  learners = models,
  sift = grains("week", "month", "season"),
  ensemble = ensemble("weighted", decay = 1.6, min_score = 0.5),
  resampling = cv(v = 5),
  verbose = FALSE
)
fit$candidates[c("candidate", "grain", "bins", "status")][1:6, ]
#>      candidate  grain bins status
#> 1   CTA / week   week   53 fitted
#> 2  CTA / month  month   12 fitted
#> 3 CTA / season season    5 fitted
#> 4    RF / week   week   53 fitted
#> 5   RF / month  month   12 fitted
#> 6  RF / season season    5 fitted
```

The run fitted 29 candidates: nine learners at three grains each and the
two fixed learners at one. All of them were cross-validated on the same
five outer folds and scored on the same cells, so their mean scores are
directly comparable.

``` r

fit
#> timesift  200 targets, 3 responses, 5-fold random CV, roc_auc
#> 
#> candidates, scored on the outer folds
#> candidate                  mean    won  responses
#> SRE / season              0.532      0  separate
#> CTA / month               0.540      0  separate
#> XGBOOST / month           0.569      0  separate
#> XGBOOST / season          0.597      0  separate
#> RF / month                0.600      0  separate
#> CTA / season              0.602      0  separate
#> GBM / month               0.613      0  separate
#> RFd / month               0.613      0  separate
#> RF / season               0.616      0  separate
#> RFd / season              0.621      0  separate
#> GBM / season              0.624      0  separate
#> MARS / season             0.636      0  separate
#> CTA / week                0.636      0  separate
#> FDA / season              0.639      0  separate
#> FDA / month               0.641      0  separate
#> GAM / season              0.645      0  separate
#> GLM / week                0.655      0  separate
#> MARS / month              0.666      0  separate
#> RFd / week                0.670      0  separate
#> MAXNET / season           0.678      0  separate
#> GLM / month               0.681      0  separate
#> XGBOOST / week            0.683      0  separate
#> RF / week                 0.683      0  separate
#> GLM / season              0.687      0  separate
#> MAXNET / month            0.696      0  separate
#> FDA / week                0.700      1  separate
#> GBM / week                0.701      0  separate
#> MARS / week               0.723      0  separate
#> MAXNET / week             0.780      2  separate
#> 
#> procedure, chosen and weighted inside each outer training fold
#> selected                  0.780  se 0.012
#> ensemble                  0.755  se 0.014
#> selected MAXNET / week in 5 of 5 folds
#> 
#> choice on every target  MAXNET / week
#> weights on every target  MAXNET / week 0.38   MARS / week 0.23   GBM / week 0.15   FDA / week 0.09   MAXNET / month 0.06   GLM / season 0.04   RF / week 0.02   XGBOOST / week 0.01   GLM / month 0.01   MAXNET / season 0.01
```

The report gives the AUC of each candidate, the number of responses on
which it scored highest, and the held-out scores of the selection
procedure and of the weighted ensemble. The four highest mean AUCs
belong to weekly candidates, the grain at which the simulation places
the signal. A biomod2 run shows which algorithm performs best on one
table. Here the run also gives the grain at which each algorithm does
best, and the ranking of grains within one algorithm need not match the
ranking of algorithms.

``` r

plot(fit)
```

![Mean AUC of each candidate against the representation it
read.](biomod2_files/figure-html/plot-fit-1.svg)

## Side by side

A biomod2 run formats the data, fits the algorithms under a
cross-validation strategy and then builds the ensembles. The two calls
below set up the same run in each package: biomod2 on a table of
predictors, `timesift` on a record. Neither chunk is evaluated.

``` r

fmt <- BIOMOD_FormatingData(resp.var = y, expl.var = predictors,
                            resp.xy = xy, resp.name = "sp1")
mod <- BIOMOD_Modeling(fmt, modeling.id = "all",
                       models = c("CTA", "RF", "GBM", "XGBOOST", "MAXNET", "GLM",
                                  "GAM", "MARS", "FDA", "SRE"),
                       CV.strategy = "kfold", CV.nb.rep = 1, CV.k = 5,
                       metric.eval = c("TSS", "ROC"))
ens <- BIOMOD_EnsembleModeling(mod, models.chosen = "all",
                               em.algo = c("EMmean", "EMwmean", "EMca"),
                               metric.select = "TSS", metric.select.thresh = 0.5,
                               metric.eval = c("TSS", "ROC"))
```

``` r

fit <- timesift(targets, series, y = starts_with("sp"), id = plot_id, time = datetime,
                learners = list(CTA = tree(), RF = forest(), GBM = boosting(),
                              XGBOOST = boosting(method = "xgboost"), MAXNET = maxent(),
                              GLM = linear(),
                              MARS = mars(), FDA = discriminant()),
                sift = grains("week", "month", "season"),
                ensemble = ensemble("weighted", min_score = 0.5),
                resampling = cv(v = 5))
```

The Python interface takes the same constructors and arguments:

``` python
import timesift as ts

fit = ts.timesift(targets, series, y="sp_*", id="plot_id", time="datetime",
                  learners=[ts.tree(), ts.forest(), ts.boosting(method="xgboost"), ts.maxent(),
                            ts.mars(), ts.discriminant()],
                  sift=ts.grains("week", "month", "season"),
                  ensemble=ts.ensemble("weighted", min_score=0.5),
                  resampling=ts.cv(v=5))
```

biomod2 fits one species per call. In `timesift`, `y` selects all
response columns at once, and the responses share one fold map, one set
of scorable cells and one set of candidates. Each response is a column
of the result.

## Ensembling

biomod2’s ensemble algorithms are options of
[`ensemble()`](https://gillescolling.com/timesift/reference/ensemble.md):

| biomod2 | `timesift` |
|----|----|
| `EMmean` | `ensemble("mean")` |
| `EMmedian` | `ensemble("median")` |
| `EMwmean` | `ensemble("weighted")`; `decay =` is `EMwmean.decay` |
| `EMca` | `ensemble("committee")`; `rule =` picks how each member’s cut is learned |
| `EMcv`, `EMci` | `predict(type = "spread")` |
| `metric.select`, `metric.select.thresh` | `metric =`, `min_score =` |
| none | `ensemble("stack")` |

The stack is `timesift`’s default. It fits non-negative weights that sum
to one, using only the out-of-fold predictions, and minimises the loss
of the response head over the scorable cells. No weight is therefore
fitted on a prediction a member made for a unit it was trained on.

`min_score = 0.5` excluded any candidate whose mean AUC was below 0.5,
the role `metric.select.thresh` plays in biomod2. The six largest
weights in this run:

``` r

round(sort(fit$weights, decreasing = TRUE)[1:6], 3)
#>  MAXNET / week    MARS / week     GBM / week     FDA / week MAXNET / month 
#>          0.375          0.234          0.146          0.092          0.057 
#>   GLM / season 
#>          0.036
```

Committee averaging converts each member’s predictions to presence or
absence at a threshold learned from that member’s out-of-fold
predictions, then averages these votes. A smaller run at one grain:

``` r

fit_ca <- timesift(
  targets, sim$readings,
  y = starts_with("v"), id = unit, time = time,
  learners = models[c("RF", "GBM", "MAXNET", "MARS", "FDA")],
  sift = grains("week"),
  ensemble = ensemble("committee", rule = "kappa"),
  resampling = cv(v = 5),
  verbose = FALSE
)
subset(fit_ca$estimate, metric %in% c("roc_auc", "tss"), c(arm, metric, score, se))
#>         arm  metric     score         se
#> 25 selected roc_auc 0.7804232 0.01208863
#> 27 selected     tss 0.5599815 0.01711471
#> 52 ensemble roc_auc 0.7236305 0.03402524
#> 54 ensemble     tss 0.4526556 0.05909692
```

`EMcv` and `EMci` summarise how much the ensemble members disagree.
`type = "spread"` returns, for every target and response, the weighted
mean, the standard deviation, the coefficient of variation and an
interval:

``` r

sp <- predict(fit, targets[1:5, ], sim$readings, type = "spread")
dimnames(sp)[[3]]
#> [1] "mean"  "sd"    "cv"    "lower" "upper"
round(sp[1:3, 1, ], 3)
#>             mean    sd    cv lower upper
#> d001u00001 0.250 0.135 0.541 0.117 0.382
#> d001u00002 0.220 0.216 0.984 0.007 0.432
#> d001u00003 0.215 0.085 0.395 0.132 0.299
```

With one target per map cell, the coefficient of variation gives an
uncertainty map like the one `EMcv` produces.

## Scores and thresholds

biomod2, like most species distribution code, reports TSS at a threshold
chosen on the same predictions it scores. When presences are few, this
choice inflates TSS. `timesift` chooses each candidate’s threshold from
its out-of-fold predictions, and
[`tss_inflation()`](https://gillescolling.com/timesift/reference/tss_inflation.md)
estimates how large the inflation would be for a given set of presence
counts. For the three responses above, five folds and two levels of true
skill:

``` r

tss_inflation(sim$y, fold_map(sim$y, v = 5), skill = c(0.6, 0.9), replicates = 40)
#>   skill  reported  inflation     lower     upper replicates
#> 1   0.6 0.7042067 0.10420666 0.6510336 0.7413138         40
#> 2   0.9 0.9520334 0.05203343 0.9296009 0.9743530         40
```

[`decision_threshold()`](https://gillescolling.com/timesift/reference/kappa_score.md)
returns the cut that turns a prediction into presence or absence, and
`rule` selects the criterion: `"youden"` (the cut that maximises TSS),
`"kappa"`, `"prevalence"`, or `"mpa"`, the minimum predicted area cut
that keeps `perc` of the presences.

biomod2’s other evaluation statistics are computed by
[`table_metric()`](https://gillescolling.com/timesift/reference/table_metric.md),
which cuts the predictions by a rule and compares the resulting
presences and absences with the observations:

| biomod2 | `timesift` |
|----|----|
| `TSS`, `ROC`, `KAPPA` | [`tss()`](https://gillescolling.com/timesift/reference/tss.md), [`roc_auc()`](https://gillescolling.com/timesift/reference/roc_auc.md), [`kappa_score()`](https://gillescolling.com/timesift/reference/kappa_score.md) |
| `POD`, `POFD`, `FAR`, `SR`, `ACCURACY`, `BIAS`, `OR`, `ORSS`, `CSI`, `ETS` | `table_metric(y, p, "pod")` and the same lower-case names |
| `BOYCE` | [`boyce_index()`](https://gillescolling.com/timesift/reference/boyce_index.md) |
| `MPA` | `decision_threshold(rule = "mpa")` |

biomod2 evaluates each statistic at the cut, out of a grid of 100, that
brings that statistic closest to its optimum, so each statistic uses its
own cut.
[`table_metric()`](https://gillescolling.com/timesift/reference/table_metric.md)
evaluates every statistic at the cut given by `rule`, `"youden"` by
default, so all statistics refer to the same threshold. Every name is a
registered metric, so `grain_ladder(metric = "csi")` can use it.

## Abundance, ordinal, continuous and count responses

biomod2 4.3 also models abundances, ordinal classes and counts. In
`timesift` the response head defines the type of response, and four
heads are available besides presence-absence: `response = "continuous"`
for any real number, `"abundance"` for a non-negative number,
`"ordinal"` for whole-number classes and `"count"` for non-negative
whole numbers. The first three are fitted under squared error with an
identity output, so every learner except
[`maxent()`](https://gillescolling.com/timesift/reference/maxent.md),
[`envelope()`](https://gillescolling.com/timesift/reference/envelope.md)
and
[`discriminant()`](https://gillescolling.com/timesift/reference/discriminant.md),
as well as the ensemble, fits them without change. Continuous and
abundance responses are compared by `r_squared` by default.
[`regression_metric()`](https://gillescolling.com/timesift/reference/regression_metric.md)
provides biomod2’s `RMSE`, `MSE`, `MAE` and `Max_error`, registered as
`neg_rmse`, `neg_mse`, `neg_mae` and `neg_max_error` with the sign
reversed so that the highest score is the best.
[`ordinal_metric()`](https://gillescolling.com/timesift/reference/ordinal_metric.md)
provides `Accuracy`, `Recall`, `Precision` and `F1`, reading each
prediction as the observed class nearest to it. A multiclass response is
fitted as one presence-absence column per class.

A count is fitted under the Poisson deviance with a log link and an
exponential output, and is compared by `neg_poisson_deviance`, the mean
deviance with the sign reversed. The elastic net, the generalised linear
model, MARS, the additive model, the tree and the boosted trees each fit
the Poisson family in their own core, checked against glmnet, MASS and
[`glm()`](https://rdrr.io/r/stats/glm.html), earth, mgcv, rpart, and gbm
or xgboost’s `count:poisson`, respectively. The forest splits a count on
its variance, the networks train under the deviance, and the ensemble
minimises it. As in rpart, a tree shrinks the rate in each leaf towards
the rate of the units the tree was grown on; `tree(shrink = 0)` turns
this shrinkage off.

## Maps

`BIOMOD_Projection()` and `BIOMOD_EnsembleForecasting()` apply a fit to
a stack of rasters.
[`project()`](https://gillescolling.com/timesift/reference/project.md)
applies it to one target per raster cell, each carrying the record for
its own cell, and returns a raster with one layer per response. A map
for a later period uses the same call with the later record, and
[`range_change()`](https://gillescolling.com/timesift/reference/range_change.md)
on the two binary maps replaces `BIOMOD_RangeSize()`.

``` r

now <- project(fit, series = temperature_now, static = terrain, type = "binary")
later <- project(fit, series = temperature_2050, static = terrain, type = "binary")
range_change(now, later)$table
```

## Tuning

`BIOMOD_Tuning()` searches a grid of settings for each algorithm.
[`tune()`](https://gillescolling.com/timesift/reference/tune.md) wraps a
learner so that, on the units it is fitted to, it cross-validates a grid
of its own settings and fits the best one. In a run, each outer fold
tunes on its own training units only, so the settings are never chosen
on the units that score the tuned candidate. The chosen settings are
listed in the candidate table.

``` r

timesift(targets, series, y = starts_with("sp"), id = plot, time = t, x = temp,
         learners = list(RF = tune(forest(), list(mtry = c(2, 5, 10), min_node = c(1, 5)))))
```

## Variable importance

biomod2’s `bm_VariablesImportance()` permutes one predictor and reports
one minus the correlation between the predictions with and without the
permutation.
[`occlusion()`](https://gillescolling.com/timesift/reference/occlusion.md)
computes the same quantity on the models a run kept for each fold,
evaluated on that fold’s held-out units, and reports it as `importance`
alongside the drop in score. A predictor is one bin of the
representation, or, with `over = "channel"`, one statistic of the record
across all bins. When the run has a record, a column named in `static`
is treated as a channel. `occlusion(fit, "ensemble", over = "channel")`
computes importance for the ensemble.

## Response curves

[`response_curve()`](https://gillescolling.com/timesift/reference/response_curve.md)
corresponds to `bm_PlotResponseCurves()`. One predictor varies across
its observed range while the others are held at the summary given by
`fixed` (`"mean"`, `"median"`, `"min"` or `"max"`, as `fixed.var`), and
a second predictor given in `with` produces the bivariate surface
(`do.bivariate`). A predictor is a statistic of the representation, so
the curve for `"warm_day"` shows the prediction as the warmest day
changes in all bins at once.

``` r

rc <- response_curve(fit, "ensemble", "warm_day", spread = TRUE)
plot(rc)
```

## Differences from a biomod2 run

- **Absences.** The learners use the absences they are given.
  [`pseudo_absences()`](https://gillescolling.com/timesift/reference/pseudo_absences.md)
  draws absences from a pool of background units before the fit, using
  biomod2’s `random`, `sre` and `disk` strategies, with repeated draws.
  A drawn unit is an ordinary row of the targets and is not marked in
  the scores; to use a set of one’s own, name those rows of the pool.
  [`maxent()`](https://gillescolling.com/timesift/reference/maxent.md)
  offers both formulations. By default it treats every unit as
  background, as biomod2’s `MAXNET` does; `formulation = "absence"`
  treats absences as absences, weighted by the response head’s case
  weights.
- **Folds.** One fold map is drawn once and used by every candidate and
  by the ensemble.
  [`cv()`](https://gillescolling.com/timesift/reference/cv.md) balances
  units on a stratifying variable,
  [`grouped_cv()`](https://gillescolling.com/timesift/reference/cv.md)
  keeps units that share a group on the same side of each split, and
  [`block_cv()`](https://gillescolling.com/timesift/reference/cv.md) and
  [`env_cv()`](https://gillescolling.com/timesift/reference/cv.md) hold
  out a block of geographic or predictor space. biomod2’s `nb.rep` is
  `repeats =` in
  [`cv()`](https://gillescolling.com/timesift/reference/cv.md) and
  [`grouped_cv()`](https://gillescolling.com/timesift/reference/cv.md):
  each repeat is a full run on its own fold map, and each response is
  averaged over its folds and repeats. biomod2’s `kfold` is
  [`cv()`](https://gillescolling.com/timesift/reference/cv.md), `strat`
  is `cv(by = )`, `block` is
  [`block_cv()`](https://gillescolling.com/timesift/reference/cv.md),
  `env` is
  [`env_cv()`](https://gillescolling.com/timesift/reference/cv.md), and
  `user.defined` is a fold map passed as `resampling`.
- **Scorable cells.** A response is scored only on folds whose held-out
  units contain both classes, and fitted only on training sets that
  contain both. This mask depends only on the response and the fold map,
  not on any model, so every paired comparison uses the same cells.
- **Case weights.** The response head sets them, and
  [`positive_weights()`](https://gillescolling.com/timesift/reference/positive_weights.md)
  returns those of the presence-absence head. The documentation of each
  learner that cannot use case weights (the background
  [`maxent()`](https://gillescolling.com/timesift/reference/maxent.md),
  [`envelope()`](https://gillescolling.com/timesift/reference/envelope.md))
  says so.
- **Presence-absence only for three algorithms.**
  [`maxent()`](https://gillescolling.com/timesift/reference/maxent.md),
  [`envelope()`](https://gillescolling.com/timesift/reference/envelope.md)
  and
  [`discriminant()`](https://gillescolling.com/timesift/reference/discriminant.md)
  require a binary response and refuse a head fitted under squared
  error. The other learners, including
  [`mars()`](https://gillescolling.com/timesift/reference/mars.md), fit
  under whatever loss the head uses.
- **Predictors.** The columns of a representation are the predictors,
  one per bin and channel. A column of `targets` such as elevation
  enters the model when `static` names it.

## When biomod2 is the better tool

When the predictors are a stack of environmental rasters with no record
behind them, there is no grain to compare, and biomod2’s raster
projection, pseudo-absence strategies and response curves are designed
for that case. `timesift` predicts rows of a target table, so a map is a
prediction with one target per cell, each with its own record at the
grain the fit uses. The package is built to find the grain at which a
record should be read; a study that does not need that comparison gains
little from switching.
