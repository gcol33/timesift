# Learners

A learner is the model half of a candidate: a representation says how
the record is reduced, and a learner says what is fitted to the
reduction. `timesift` ships fifteen learners, from a penalised logistic
regression to a residual convolutional network. Every one of them goes
through the same pair of functions, the same folds and the same scoring
as a learner written for a single study, so adding a model never means
changing the code that fits or scores.

This article lists what ships, what each learner can be handed, how the
main ones are set, how the neural encoders are trained and tuned, and
how a learner, a response type or a metric of one’s own is added. It
ends with what each learner costs on a small record and where each is
the wrong choice.

The examples use one simulated record throughout: 150 units read every
six hours for a year, and four presence-absence responses that depend on
the record through a lagged mechanism acting at the weekly grain. In
this simulation the response reads each unit’s departure from its own
mean, so a learner can find the signal only at a grain fine enough to
keep that departure.

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

## One interface

A learner is a list carrying a `fit` function, a `predict` function and
three declarations: what it reads, whether it covers the responses
jointly or one at a time, and which packages it needs. Printing one
shows the declarations and the settings it carries, with `NULL` where a
setting takes its default at the fit.

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

`fit` is handed a `[unit, bin, channel]` array and the response matrix
of the same units, and `predict` returns one column per response. A
learner that fits one model per response is still handed the whole
response matrix, so nothing above the learner layer needs to know which
kind it was. [`c()`](https://rdrr.io/r/base/c.html) binds learners into
a set, which
[`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
takes as `learners`.

``` r

c(elasticnet(), forest(), cnn())
#> <timesift models> 3 learners 
#>   elasticnet     reads tabular, separate
#>   forest         reads tabular, separate
#>   cnn            reads sequence, joint
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

Twelve learners read a tabular block and fit one model per response.
[`mlp()`](https://gillescolling.com/timesift/reference/torch_learners.md)
also reads a tabular block, flattening the channels, but fits every
response jointly through one shared embedding, as the two convolutional
encoders do. Only
[`cnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
and
[`rescnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
read the bins as a sequence. The three encoders need `torch`. Every
other learner runs on the compiled core the package ships, which the
Python package calls as well, so the two languages fit the same model.

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

“Every head” means the learner fits whatever loss the response head
names: a logistic model under presence-absence, a Gaussian one under the
continuous, abundance and ordinal heads, and a Poisson one under the
count head.
[`hierarchical()`](https://gillescolling.com/timesift/reference/hierarchical.md)
has no reference package. It finds the posterior by Laplace’s method and
integrates its hyperparameters over a grid around their mode (Rue,
Martino and Chopin 2009), and with `spatial = "nngp"` or `"hsgp"` it
adds a Gaussian-process field over the coordinates that
`timesift(coords = )` names.

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

`learners` takes a set or a named list. A named list reports each
candidate under its own name, which is how two settings of one learner
sit side by side. Every learner without a `data =` runs at every grain
of `sift`;
[`linear()`](https://gillescolling.com/timesift/reference/linear.md) is
pinned to the monthly grain here, and the elastic net is given a looser
stopping tolerance, both to keep the run short. The section on settings
shows what the tolerance changes.

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
#> [1] 30.54
```

The run took 31 seconds for 15 candidates. With three inner folds in
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

Each learner run at both grains scores higher at the weekly one, which
is the grain the simulation acts at. At the monthly grain the tree, both
forests and gbm score below 0.5, so they find nothing there.
`maxent / week` has the highest mean, 0.750, and scored highest on all
four responses. The procedure rows are the scores to report: inside each
outer training fold the inner folds chose a candidate and fitted the
stack’s weights, and the outer test fold then scored that choice once,
0.741 for the selected candidate and 0.740 for the stack. The highest
candidate mean was picked out on the same folds it is scored on.

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

`unfitted` names any response with too few presences to choose a penalty
on, which is then predicted its share among the fitting units, and
`stopped` names any response whose penalty path ended early. Neither
applies here. The fit carries the bins it was made on, and an array with
other bins is refused:

``` r

month_train <- grain_matrix(sim$readings[in_train, ], unit, time, reading, grain = "month")
month_test <- grain_matrix(sim$readings[!in_train, ], unit, time, reading, grain = "month")
predict(f, month_test)
#> Error:
#> ! the representation predicted on has different channels or bins from the fitted one: 12 bins here and 53 bins in the fit. A calendar grain is read by its bins' instants, so a fit predicts a record over the same period; a lookback reads a span relative to each target and predicts any period.
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

On this split the elastic net read at `"lambda.min"` scores 0.680
against 0.631 at the default `"lambda.1se"`, and the lasso scores 0.608.
The looser tolerance gives the same AUC as the default to three
decimals, and the timings at the end of the article show what it saves.
`maxent(classes = "lq")` scores 0.693 against 0.665 for the classes
maxnet chooses, and both forests score close to 0.5 at this grain on 75
units. A single split of 75 units is a noisy comparison, so a difference
of this size is worth checking under
[`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)’s
folds, with the two settings as two named candidates, before acting on
it.

## Neural encoders

The encoders separate architecture from training.
[`mlp()`](https://gillescolling.com/timesift/reference/torch_learners.md),
[`cnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
and
[`rescnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
take the layer widths, the kernel and the dropout, and
[`train_control()`](https://gillescolling.com/timesift/reference/train_control.md)
holds every training setting: epochs, batch size, learning rate,
optimiser, schedule, early stopping and device.
[`train_control()`](https://gillescolling.com/timesift/reference/train_control.md)
is the one place a training setting is defaulted. A run hands its
control to every neural learner, and a training setting given to one
learner overrides the run’s control for that learner alone, on the
settings it names.

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

The search chose `mtry = 8` and `min_node = 5`, the combination with the
highest mean inner AUC. Under
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

`channel_means` scores 0.461 at both grains. The mean of a channel over
the whole year is nearly the same number whichever bins it is averaged
from, and in this simulation the responses read each unit’s departure
from its own level, which a yearly mean averages away. The weekly
elastic net, which reads every bin, scores 0.701.

## Other responses, and other metrics

The response head says what the values are, how each learner fits them
and which cells a score is defined on. Besides `"presence_absence"`,
four heads ship:

``` r

responses()
#> [1] "abundance"        "continuous"       "count"            "ordinal"         
#> [5] "presence_absence"
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

The run is chosen and reported by the Brier score under the capped
weights. The elastic net, at -0.239, scored above the forest, at -0.259,
and scored highest on all four responses.

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
#>                  10.42                   3.96                   0.47 
#>                 maxent elasticnet(tol = 1e-4)           discriminant 
#>                   0.40                   0.31                   0.17 
#>                xgboost             perceptron                   tree 
#>                   0.08                   0.07                   0.03 
#>               boosting                   mars           hierarchical 
#>                   0.03                   0.02                   0.01 
#>               envelope 
#>                   0.00
```

[`additive()`](https://gillescolling.com/timesift/reference/additive.md)
is not in that list, because at its default `k = 10` an additive model
of 53 columns needs more coefficients than there are units:

``` r

fit_learner(additive(), week, sim$y)
#> Error:
#> ! an additive model of 478 coefficients has more than the 150 units it is fitted to; fewer columns or a smaller `k` fit.
```

At the monthly grain, 12 columns, it fits, and its cost depends on `k`:

``` r

month <- grain_matrix(sim$readings, unit, time, reading, grain = "month")
additive_seconds <- c(
  `k = 10` = system.time(fit_learner(additive(), month, sim$y))[["elapsed"]],
  `k = 5` = system.time(fit_learner(additive(k = 5L), month, sim$y))[["elapsed"]])
round(additive_seconds, 2)
#> k = 10  k = 5 
#>  21.94   0.55
```

With `k = 10` the fit took 21.9 seconds, and with `k = 5` it took 0.5
seconds.

The elastic net is the learner to start with. It reads every column
without a selection step, chooses its penalty inside the units it is
fitted on, and is the arm the package measures a network against. At its
default tolerance it took 4 seconds here, and `tol = 1e-4` took 0.3, at
the same held-out AUC on the split above.
[`linear()`](https://gillescolling.com/timesift/reference/linear.md) was
the slowest at 10.4 seconds: its two-way search refits the model for
every term it could add or drop at every step, and the weekly grain
offers it 106 terms, each column and its square. Pinned to a coarse
grain, or searched forward under a bound as
`linear(select = "forward", terms = "column", max_terms = 3)` does, the
search has fewer steps to take. Every other tabular learner took at most
0.47 seconds.

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
