# Coming from biomod2

biomod2 fits a set of species distribution algorithms on one table of
predictors and combines them. `timesift` fits the same algorithms, and
the representation of the predictors becomes a second axis of the
comparison. When the predictors are a sensor record (hourly soil
temperature, daily air temperature), the table biomod2 starts from is
already a reduction of it, to monthly means or growing-degree-days. Here
the algorithms run at every temporal grain the record supports, are
scored on one set of held-out folds, and are combined afterwards.

Each algorithm is one implementation in the compiled core that R and
Python both call, so a model fitted from either language returns the
same numbers. The cores reproduce the reference packages biomod2 itself
calls (rpart, randomForest, gbm, xgboost, maxnet, MASS, earth, mda,
mgcv), and each is pinned against the reference’s own output in the test
fixtures. Nothing is fitted through biomod2 or through those packages.

## The mapping

Every biomod2 algorithm is a learner constructor. A variant of an
algorithm is an argument of its constructor.

| biomod2 | `timesift` | notes |
|----|----|----|
| `CTA` | [`tree()`](https://gillescolling.com/timesift/reference/tree.md) | rpart’s rules; `prune = "se_sum"` is biomod2’s pruning |
| `RF` | [`forest()`](https://gillescolling.com/timesift/reference/forest.md) | randomForest’s defaults |
| `RFd` | `forest(balance = TRUE)` | each tree draws as many units from each class as the smaller class holds |
| `GBM` | [`boosting()`](https://gillescolling.com/timesift/reference/boosting.md) | gbm’s trees and Newton step |
| `XGBOOST` | `boosting(newton = TRUE)` | xgboost’s exact greedy trees; `lambda` and `gamma` are its penalties |
| `MAXNET`, `MAXENT` | [`maxnet()`](https://gillescolling.com/timesift/reference/maxnet.md) | `classes`, `regmult` and `knots` are maxnet’s; the model is the background formulation by default |
| `GLM` | `stepwise(direction = "both", terms = "power", max_terms = Inf)` | `y ~ x + I(x^2)` over every column, searched by AIC as [`MASS::stepAIC()`](https://rdrr.io/pkg/MASS/man/stepAIC.html) does |
| `GAM` | [`additive()`](https://gillescolling.com/timesift/reference/additive.md) | mgcv’s `gam(method = "GCV.Cp")`, `k = 10` |
| `MARS` | [`mars()`](https://gillescolling.com/timesift/reference/mars.md) | earth’s forward and pruning passes, refitted as a logistic model |
| `FDA` | [`discriminant()`](https://gillescolling.com/timesift/reference/discriminant.md) | mda’s `fda(method = mars)` with the probit recalibration biomod2 applies |
| `SRE` | [`envelope()`](https://gillescolling.com/timesift/reference/envelope.md) | `quantile = 0.025`, as `bm_SRE()` |
| `ANN`, `DNN` | [`mlp()`](https://gillescolling.com/timesift/reference/torch_learners.md) | a torch encoder with a joint head over all responses |

`ANN` and `DNN` map to a learner by architecture, and not to numbers:
[`mlp()`](https://gillescolling.com/timesift/reference/torch_learners.md)
is a different network from the one biomod2 fits, `hidden` sets the
layer widths, `activation` the nonlinearity and `dropout` the
regularisation, and the training settings live in
[`train_control()`](https://gillescolling.com/timesift/reference/train_control.md).
nnet minimises an L2-penalised loss by BFGS and cito by a stochastic
optimiser, where
[`mlp()`](https://gillescolling.com/timesift/reference/torch_learners.md)
uses AdamW, whose weight decay acts differently, so a fit under the same
architecture is a comparable network and not the same one. biomod2’s
tuned `ANN` (`size = 5`, `decay = 0.1`, `maxit = 200`) is
`mlp(hidden = 5, epochs = 200, weight_decay = 0.1)`, and its tuned `DNN`
(`hidden = c(100, 100)`, `activation = "selu"`, 150 epochs, batch size
100) is
`mlp(hidden = c(100, 100), activation = "selu", epochs = 150, batch_size = 100)`.

biomod2 ships two option sets, its defaults and a tuned set called
`"bigboss"`. Where the two differ, the learners carry both as
`preset = "package"` (the fitting package’s own defaults, which
biomod2’s default set uses) and `preset = "bigboss"`, so
`tree(preset = "bigboss")` is `min_split = 5`, `min_leaf = 5`,
`cp = 0.001`, `max_depth = 10` and five inner folds. A setting given
explicitly overrides either preset.
[`mars()`](https://gillescolling.com/timesift/reference/mars.md),
[`discriminant()`](https://gillescolling.com/timesift/reference/discriminant.md)
and
[`additive()`](https://gillescolling.com/timesift/reference/additive.md)
have one set, which both of biomod2’s share.

## One run

The record is simulated: 200 units, a year of readings every six hours,
three responses that read the record through a lagged, weekly-grain
mechanism. The true grain is known to be the week, which makes the
comparison below checkable.

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

`models` is a named list, so each candidate reports under the biomod2
name. A learner left open runs at every grain of `sift`; one given
`data =` runs at that representation alone.

``` r

models <- list(
  CTA = tree(),
  RF = forest(),
  RFd = forest(balance = TRUE),
  GBM = boosting(),
  XGBOOST = boosting(newton = TRUE),
  MAXNET = maxnet(),
  GLM = stepwise(direction = "both", terms = "power", max_terms = Inf),
  GAM = additive(data = grain("season"), k = 5L),
  MARS = mars(),
  FDA = discriminant(),
  SRE = envelope(data = grain("season"))
)
```

Two of them are pinned to the seasonal grain. An additive model holds
`k - 1` coefficients per column, and the 53 weekly columns of this
record would need 478 coefficients for 160 training units, which
[`additive()`](https://gillescolling.com/timesift/reference/additive.md)
refuses; four seasonal columns need 17. The envelope shuts out one
presence in twenty at each end of every column, so the more columns, the
fewer units fall inside every band, and a coarse grain is what it is
meant for.

``` r

fit <- timesift(
  targets, sim$readings,
  y = starts_with("v"), id = unit, time = time,
  models = models,
  sift = grains("week", "month", "season"),
  ensemble = ensemble("weighted", decay = 1.6, min_score = 0.5),
  resampling = cv(v = 5),
  verbose = FALSE
)
fit$candidates[c("candidate", "grain", "bins", "status")][1:6, ]
#>      candidate  grain bins status
#> 1   CTA / week   week   53 fitted
#> 2  CTA / month  month   12 fitted
#> 3 CTA / season season    4 fitted
#> 4    RF / week   week   53 fitted
#> 5   RF / month  month   12 fitted
#> 6  RF / season season    4 fitted
```

29 candidates were fitted: nine learners at three grains and the two
pinned ones at one. All of them were cross-validated on the same five
outer folds and scored on the same cells, so their means share a
denominator.

``` r

fit
#> timesift  200 targets, 3 responses, 5-fold random CV, roc_auc
#> 
#> candidates, scored on the outer folds
#> candidate                  mean    won  responses
#> CTA / month               0.540      0  separate
#> SRE / season              0.540      0  separate
#> CTA / season              0.542      0  separate
#> XGBOOST / season          0.560      0  separate
#> XGBOOST / month           0.569      0  separate
#> RF / season               0.577      0  separate
#> RFd / season              0.578      0  separate
#> GBM / season              0.583      0  separate
#> RF / month                0.600      0  separate
#> GBM / month               0.613      0  separate
#> RFd / month               0.613      0  separate
#> GLM / season              0.615      0  separate
#> FDA / season              0.617      0  separate
#> MAXNET / season           0.626      0  separate
#> MARS / season             0.634      0  separate
#> CTA / week                0.636      0  separate
#> GAM / season              0.639      0  separate
#> FDA / month               0.641      0  separate
#> GLM / week                0.655      0  separate
#> MARS / month              0.666      0  separate
#> RFd / week                0.670      0  separate
#> GLM / month               0.681      0  separate
#> XGBOOST / week            0.683      0  separate
#> RF / week                 0.683      0  separate
#> MAXNET / month            0.696      0  separate
#> FDA / week                0.700      1  separate
#> GBM / week                0.701      0  separate
#> MARS / week               0.723      0  separate
#> MAXNET / week             0.780      2  separate
#> 
#> procedure, chosen and weighted inside each outer training fold
#> selected                  0.780  se 0.012
#> ensemble                  0.758  se 0.014
#> selected MAXNET / week in 5 of 5 folds
#> 
#> choice on every target  MAXNET / week
#> weights on every target  MAXNET / week 0.38   MARS / week 0.23   GBM / week 0.15   FDA / week 0.09   MAXNET / month 0.06   RF / week 0.04   XGBOOST / week 0.02   GLM / month 0.01   RFd / week 0.01   MARS / month 0.01
```

The AUC of each candidate, the number of responses it scored highest on,
and the held-out score of the procedure that chose among them and of the
weighted combination are in the report above. The four highest means
belong to weekly candidates, the grain the simulation puts the signal
at. Where a biomod2 run reads off which algorithm wins on one table,
this run also reads off the grain at which each algorithm does best, and
the ordering across grains for one algorithm does not have to follow the
ordering across algorithms.

``` r

plot(fit)
```

![Mean AUC of each candidate against the representation it
read.](biomod2_files/figure-html/plot-fit-1.svg)

## Side by side

A biomod2 run formats the data, fits the algorithms under a
cross-validation strategy and then builds the ensembles. The calls below
are the same run in each package, over one table of predictors for
biomod2 and over a record for `timesift`. The biomod2 chunk is not
evaluated.

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
                models = list(CTA = tree(), RF = forest(), GBM = boosting(),
                              XGBOOST = boosting(newton = TRUE), MAXNET = maxnet(),
                              GLM = stepwise(direction = "both", terms = "power",
                                             max_terms = Inf),
                              MARS = mars(), FDA = discriminant()),
                sift = grains("week", "month", "season"),
                ensemble = ensemble("weighted", min_score = 0.5),
                resampling = cv(v = 5))
```

The Python call takes the same constructors and the same arguments:

``` python
import timesift as ts

fit = ts.timesift(targets, series, y="sp_*", id="plot_id", time="datetime",
                  models=[ts.tree(), ts.forest(), ts.boosting(newton=True), ts.maxnet(),
                          ts.mars(), ts.discriminant()],
                  sift=ts.grains("week", "month", "season"),
                  ensemble=ts.ensemble("weighted", min_score=0.5),
                  resampling=ts.cv(v=5))
```

`y` takes every response column at once. biomod2 fits one species per
call; here the responses share one fold map, one set of scorable cells
and one set of candidates, and a response is a column of the result.

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

The stack, `timesift`’s default, fits non-negative weights summing to
one on the out-of-fold predictions alone, minimising the response head’s
loss over the scorable cells. The weights are therefore never fitted on
a prediction the member made for a unit it had been trained on.

`min_score = 0.5` made a candidate ineligible when its mean AUC was
below 0.5, which is the role `metric.select.thresh` plays. The weights
the run settled on:

``` r

round(sort(fit$weights, decreasing = TRUE)[1:6], 3)
#>  MAXNET / week    MARS / week     GBM / week     FDA / week MAXNET / month 
#>          0.375          0.234          0.146          0.092          0.057 
#>      RF / week 
#>          0.036
```

Committee averaging binarises each member at its own threshold, learned
from that member’s out-of-fold predictions, and averages the votes. A
smaller run on one grain shows it:

``` r

fit_ca <- timesift(
  targets, sim$readings,
  y = starts_with("v"), id = unit, time = time,
  models = models[c("RF", "GBM", "MAXNET", "MARS", "FDA")],
  sift = grains("week"),
  ensemble = ensemble("committee", rule = "kappa"),
  resampling = cv(v = 5),
  verbose = FALSE
)
subset(fit_ca$estimate, metric %in% c("roc_auc", "tss"), c(arm, metric, score, se))
#>         arm  metric     score         se
#> 24 selected roc_auc 0.7800964 0.01178032
#> 26 selected     tss 0.5580208 0.01515674
#> 50 ensemble roc_auc 0.7236305 0.03402524
#> 52 ensemble     tss 0.4526556 0.05909692
```

`EMcv` and `EMci` read the members of the ensemble side by side.
`type = "spread"` returns the weighted mean, the standard deviation, the
coefficient of variation and an interval for every target and response:

``` r

sp <- predict(fit, targets[1:5, ], sim$readings, type = "spread")
dimnames(sp)[[3]]
#> [1] "mean"  "sd"    "cv"    "lower" "upper"
round(sp[1:3, 1, ], 3)
#>             mean    sd    cv lower upper
#> d001u00001 0.245 0.140 0.571 0.107 0.382
#> d001u00002 0.215 0.219 1.015 0.000 0.431
#> d001u00003 0.211 0.090 0.429 0.122 0.299
```

On one target per map cell this is an uncertainty map in the way `EMcv`
is.

## Scores and thresholds

biomod2 reports TSS at a threshold chosen on the same predictions it
scores, and so does most species distribution code. Where presences are
thin that choice inflates TSS. `timesift` reads the threshold of a
candidate from its out-of-fold predictions and reports the expected
inflation for a user’s own presence counts with
[`tss_inflation()`](https://gillescolling.com/timesift/reference/tss_inflation.md).
For the three responses above and five folds the inflation at two
population skills reads:

``` r

tss_inflation(sim$y, fold_map(sim$y, v = 5), skill = c(0.6, 0.9), replicates = 40)
#>   skill  reported  inflation     lower     upper replicates
#> 1   0.6 0.7042067 0.10420666 0.6510336 0.7413138         40
#> 2   0.9 0.9520334 0.05203343 0.9296009 0.9743530         40
```

[`decision_threshold()`](https://gillescolling.com/timesift/reference/kappa_score.md)
returns the cuts a binary prediction uses, and `rule` picks the
criterion: `"youden"` for TSS, `"kappa"`, `"prevalence"` or `"mpa"`, the
minimum predicted area cut that keeps `perc` of the presences.

biomod2’s other evaluation statistics come from
[`table_metric()`](https://gillescolling.com/timesift/reference/table_metric.md),
which cuts the predictions by a rule and reads the table of decisions
against observations:

| biomod2 | `timesift` |
|----|----|
| `TSS`, `ROC`, `KAPPA` | [`tss()`](https://gillescolling.com/timesift/reference/tss.md), [`roc_auc()`](https://gillescolling.com/timesift/reference/roc_auc.md), [`kappa_score()`](https://gillescolling.com/timesift/reference/kappa_score.md) |
| `POD`, `POFD`, `FAR`, `SR`, `ACCURACY`, `BIAS`, `OR`, `ORSS`, `CSI`, `ETS` | `table_metric(y, p, "pod")` and the same lower-case names |
| `BOYCE` | [`boyce_index()`](https://gillescolling.com/timesift/reference/boyce_index.md) |
| `MPA` | `decision_threshold(rule = "mpa")` |

biomod2 reads each statistic at the cut that brings that statistic
closest to its own optimum on a grid of 100 cuts.
[`table_metric()`](https://gillescolling.com/timesift/reference/table_metric.md)
reads it at the cut of the `rule` given, `"youden"` by default, so the
statistics are comparable at one operating point. Every name is a
registered metric, so `grain_ladder(metric = "csi")` reads it.

## Abundance, ordinal and continuous responses

biomod2 4.3 also models abundances and ordinal classes. The response
head decides what a response is, and three ship beside presence-absence:
`response = "continuous"` for any real number, `"abundance"` for a
non-negative one, and `"ordinal"` for whole-number classes. All three
are fitted under squared error through an identity output, so every
learner but the three above and the combiner read them unchanged.
`r_squared` is what a continuous or abundance comparison reads by
default, and
[`regression_metric()`](https://gillescolling.com/timesift/reference/regression_metric.md)
carries biomod2’s `RMSE`, `MSE`, `MAE` and `Max_error` (registered as
`neg_rmse`, `neg_mse`, `neg_mae` and `neg_max_error`, since the highest
score is the best).
[`ordinal_metric()`](https://gillescolling.com/timesift/reference/ordinal_metric.md)
carries `Accuracy`, `Recall`, `Precision` and `F1`, each prediction
being read as the observed class nearest to it. A multiclass response is
one presence-absence column per class. A count under a Poisson loss is
not shipped: the penalised, tree, boosting and stepwise cores would each
need that family.

## Maps

`BIOMOD_Projection()` and `BIOMOD_EnsembleForecasting()` apply the fit
to a stack of rasters.
[`project()`](https://gillescolling.com/timesift/reference/project.md)
applies it to one target per cell, each carrying the record of its own
cell, and returns a raster with a layer per response. A map for a later
period is the same call with the later record, and `BIOMOD_RangeSize()`
is
[`range_change()`](https://gillescolling.com/timesift/reference/range_change.md)
on the two binary maps.

``` r

now <- project(fit, series = temperature_now, static = terrain, type = "binary")
later <- project(fit, series = temperature_2050, static = terrain, type = "binary")
range_change(now, later)$table
```

## Tuning

`BIOMOD_Tuning()` searches a grid per algorithm.
[`tune()`](https://gillescolling.com/timesift/reference/tune.md) wraps a
learner so that it searches a grid of its own settings on the units it
is fitted on, by cross-validation inside them, and fits the best. Inside
a run each outer fold chooses from its own training units, so the score
a tuned candidate is read at is not selected on. The settings it chose
are in the candidate table.

``` r

timesift(targets, series, y = starts_with("sp"), id = plot, time = t, x = temp,
         models = list(RF = tune(forest(), list(mtry = c(2, 5, 10), min_node = c(1, 5)))))
```

## Variable importance

biomod2’s `bm_VariablesImportance()` permutes one predictor and reports
one minus its correlation with the prediction made without the
permutation.
[`occlusion()`](https://gillescolling.com/timesift/reference/occlusion.md)
does the same on the models a run kept per fold, on the units each held
out, and reports it as `importance` beside the fall in score. A
predictor is a bin of the representation, or with `over = "channel"` one
statistic of it across the record, which is where a column named in
`static` sits when the run has a record.
`occlusion(fit, "ensemble", over = "channel")` reads the combination.

## Response curves

[`response_curve()`](https://gillescolling.com/timesift/reference/response_curve.md)
is `bm_PlotResponseCurves()`: one predictor is moved across the range it
takes while the others sit at their `fixed` summary (`"mean"`,
`"median"`, `"min"` or `"max"`, as `fixed.var`), and a second predictor
`with` gives the bivariate surface (`do.bivariate`). A predictor is a
statistic of the representation, so the curve of `"warm_day"` is the
prediction as the warmest day moves in every bin together.

``` r

rc <- response_curve(fit, "ensemble", "warm_day", spread = TRUE)
plot(rc)
```

## What differs from a biomod2 run

- **Absences.** The learners read the absences they are given, and
  [`pseudo_absences()`](https://gillescolling.com/timesift/reference/pseudo_absences.md)
  draws them from a pool of background units by biomod2’s `random`,
  `sre` and `disk` strategies, with repeated draws, before the fit. A
  drawn unit is a row of the targets like any other: it is not flagged
  in the score, and a set of one’s own is the rows of the pool it names.
  [`maxnet()`](https://gillescolling.com/timesift/reference/maxnet.md)
  offers both formulations: the default treats every unit as background,
  as biomod2’s `MAXNET` does, and `formulation = "absence"` reads the
  absences as absences under the response head’s case weights.
- **Folds.** One fold map is dealt once and read by every candidate and
  the ensemble.
  [`cv()`](https://gillescolling.com/timesift/reference/cv.md) deals
  units balanced on a stratifying value,
  [`grouped_cv()`](https://gillescolling.com/timesift/reference/cv.md)
  keeps units that share a group on one side of each split, and
  [`block_cv()`](https://gillescolling.com/timesift/reference/cv.md) and
  [`env_cv()`](https://gillescolling.com/timesift/reference/cv.md) hold
  out a block of space or of predictor space. biomod2’s `nb.rep` is
  `repeats =` on
  [`cv()`](https://gillescolling.com/timesift/reference/cv.md) and
  [`grouped_cv()`](https://gillescolling.com/timesift/reference/cv.md):
  every repeat is a full run on its own fold map, and a response is
  averaged over its folds and its repeats. biomod2’s `kfold` is
  [`cv()`](https://gillescolling.com/timesift/reference/cv.md), `strat`
  is `cv(by = )`, `block` is
  [`block_cv()`](https://gillescolling.com/timesift/reference/cv.md),
  `env` is
  [`env_cv()`](https://gillescolling.com/timesift/reference/cv.md) and
  `user.defined` is a fold map passed as `resampling`.
- **Scorable cells.** A response is scored only on folds whose held-out
  units hold both classes, and only fitted on training units that hold
  both. The mask follows from the response and the fold map and involves
  no model, so every paired comparison runs on matched cells.
- **Case weights.** The response head sets them, and
  [`positive_weights()`](https://gillescolling.com/timesift/reference/positive_weights.md)
  gives the presence-absence head’s. A learner that cannot use them (the
  background
  [`maxnet()`](https://gillescolling.com/timesift/reference/maxnet.md),
  [`envelope()`](https://gillescolling.com/timesift/reference/envelope.md))
  says so in its documentation.
- **Presence-absence only for three algorithms.**
  [`maxnet()`](https://gillescolling.com/timesift/reference/maxnet.md),
  [`envelope()`](https://gillescolling.com/timesift/reference/envelope.md)
  and
  [`discriminant()`](https://gillescolling.com/timesift/reference/discriminant.md)
  need a binary response and refuse a head fitted under squared error.
  The other learners follow the head’s loss, and
  [`mars()`](https://gillescolling.com/timesift/reference/mars.md) is
  among them.
- **Predictors.** The columns of a representation are the predictors,
  one per bin and channel, and a column of `targets` such as elevation
  reaches the model where `static` names it.

## When biomod2 is the better tool

A study whose predictors are a stack of environmental rasters, with no
record behind them, has nothing for the representation axis to vary, and
biomod2’s projection onto rasters, its pseudo-absence sampling
strategies and its response curves are built for that setting.
`timesift` predicts rows of a target table, so a map is a prediction
with one target per cell and each cell carrying its own series at the
grain the fit reads. Which grain a record should be read at is the
question the package answers, and a study that does not ask it gains
little from switching.
