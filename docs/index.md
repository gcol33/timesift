# timesift

*sifting a record for the grain that carries the signal*

[![CRAN
status](https://www.r-pkg.org/badges/version/timesift)](https://CRAN.R-project.org/package=timesift)
[![CRAN
downloads](https://cranlogs.r-pkg.org/badges/grand-total/timesift)](https://cran.r-project.org/package=timesift)
[![Monthly
downloads](https://cranlogs.r-pkg.org/badges/timesift)](https://cran.r-project.org/package=timesift)
[![PyPI](https://img.shields.io/pypi/v/timesift)](https://pypi.org/project/timesift/)
[![R-CMD-check](https://github.com/gcol33/timesift/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/gcol33/timesift/actions/workflows/R-CMD-check.yaml)
[![pytest](https://github.com/gcol33/timesift/actions/workflows/pytest.yaml/badge.svg)](https://github.com/gcol33/timesift/actions/workflows/pytest.yaml)
[![contract](https://github.com/gcol33/timesift/actions/workflows/contract.yaml/badge.svg)](https://github.com/gcol33/timesift/actions/workflows/contract.yaml)
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Learn predictive representations of time-varying data, in R and Python
over one C++ core.**

Hand `timesift` one row per thing to predict and a long table of
time-stamped readings belonging to those rows. It builds each candidate
representation of the record, from the readings as recorded through a
calendar week or month to a span anchored on each target, fits the
learners you name on every one they can read, and scores them all on one
set of held-out folds. What comes back says at what grain the prediction
needed the record, and what the whole procedure, choosing included,
scores on targets it did not see.

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

## Quick start

[TABLE]

`fit` prints every candidate it fitted and the score each reached on the
held-out folds, then the held-out score of the procedure that chooses
among them and of the stack, both chosen and weighted inside each
training fold:

``` r

fit
#> timesift  80 targets, 4 responses, 10-fold random CV, roc_auc
#>
#> candidates, scored on the outer folds
#> candidate                    mean    won  responses
#> elasticnet / day            0.873      0  separate
#> forest / month              0.904      1  separate
#> forest / day                0.924      0  separate
#> forest / week               0.929      1  separate
#> elasticnet / week           0.930      1  separate
#> elasticnet / month          0.938      1  separate
#>
#> procedure, chosen and weighted inside each outer training fold
#> selected                    0.934  se 0.015
#> ensemble                    0.919  se 0.020
#> selected elasticnet / month in 8, forest / day in 2 of 10 folds
#>
#> choice on every target  elasticnet / month
#> weights on every target  elasticnet / month 0.98   forest / month 0.02
```

`predict(fit, new_plots, new_logger)` rebuilds every member’s
representation for the new rows and predicts through the ensemble fitted
on every target; `candidate = "selected"` predicts with the chosen
candidate. `type = "binary"` cuts each species into presence and absence
at a threshold learned from the fit’s own out-of-fold predictions, and
`type = "spread"` gives how far the ensemble’s members disagree, which
on the same cells is an uncertainty map.

## Statement of need

A sensor records every hour for years. Before any model is fitted, that
record is reduced: to monthly means, to growing-degree-days, to whatever
the analyst decides, and the reduction is rarely revisited. `timesift`
makes it an explicit, testable choice, with every candidate scored on
the same folds and the same cells so the comparison between them is
paired.

Species distribution modelling from microclimate loggers is the
application the package was built for and the setting its defaults
serve: presence-absence, a joint multi-label head, and AUC with the true
skill statistic beside it. On 894 alpine plots, 101 species and three
years of hourly soil temperature:

- The full hourly series was the best input for none of three
  architectures. Reading every hour cost the convolutional network 0.048
  TSS against its own best grain.
- Skill peaked at the weekly average and fell from monthly on, by 0.080
  at yearly.
- A window’s coldest and warmest **day** carried more than its mean, and
  by more as the window widened: 0.006 weekly to 0.046 yearly.
- A fully connected network on the same series was level with a
  penalised logistic model on 188 hand-built features (-0.002 TSS, p =
  0.63), which suggests the gain came from convolution reading the
  series at a coarse grain.

The package is what lets the same test run on other records and other
responses.

## Targets, series, representations and learners

**targets** is one row per thing to predict, carrying the response and
optionally predictors that do not move in time. **series** is the long
record: an identifier, an instant, and one or more value columns. A
**representation** is how that record becomes the array a model reads. A
**learner** is a fit and a predict pair that declares what it can be
handed.

A candidate is one representation paired with one learner, and every
candidate emits an out-of-fold prediction for every scorable cell over
the same folds. Comparison, ensembling and importance read those
predictions and nothing else, which is what lets a penalised regression
on monthly features and a convolution on the unreduced record be
compared and then combined.

The candidates’ scores on the outer folds are the comparison, and the
best of them was picked out on the folds it is scored on. The `selected`
and `ensemble` rows are chosen and weighted inside each outer training
fold, on an inner split of it, and scored on the outer fold once, so
they are the numbers to report.

## Features

### Representations

``` r

native()                        # the record as it was recorded
grain("week")                   # one calendar grain
multigrain(c("month", "year"))  # several grains side by side as one block of features
lookback("30 days", bins = 3)   # a fixed span ending at each target's own instant
```

`grains("day", "week", "month")` and `lookbacks("30 days", "90 days")`
are sets of them, and `grains("auto")` reads off the record every named
grain it gives at least two bins. A learner runs across the whole set,
or across one representation it is pinned to:

``` r

cnn()                      # every representation of the run
cnn(data = native())       # the record unreduced only
cnn(data = grain("week"))  # weekly only
```

[`lookback()`](https://gillescolling.com/timesift/reference/native.md)
serves a unit carrying several targets through time: two targets a
fortnight apart on one sensor read two different stretches of the same
series, anchored by `target_time`.
[`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md)
and
[`lookback_matrix()`](https://gillescolling.com/timesift/reference/lookback_matrix.md)
return the arrays themselves, `[unit, bin, channel]`, for use outside
the fitting layer.

### Learners

- **Penalised and linear**:
  [`elasticnet()`](https://gillescolling.com/timesift/reference/elasticnet.md),
  [`linear()`](https://gillescolling.com/timesift/reference/linear.md)
  (stepwise GLM)
- **Trees**:
  [`tree()`](https://gillescolling.com/timesift/reference/tree.md)
  (CART),
  [`forest()`](https://gillescolling.com/timesift/reference/forest.md),
  [`boosting()`](https://gillescolling.com/timesift/reference/boosting.md)
- **Species distribution classics**:
  [`maxent()`](https://gillescolling.com/timesift/reference/maxent.md),
  [`envelope()`](https://gillescolling.com/timesift/reference/envelope.md),
  [`mars()`](https://gillescolling.com/timesift/reference/mars.md),
  [`discriminant()`](https://gillescolling.com/timesift/reference/discriminant.md),
  [`additive()`](https://gillescolling.com/timesift/reference/additive.md),
  [`perceptron()`](https://gillescolling.com/timesift/reference/perceptron.md)
- **Bayesian**:
  [`hierarchical()`](https://gillescolling.com/timesift/reference/hierarchical.md),
  a logistic model with unit intercepts and a spatial field
- **Neural encoders** (torch):
  [`cnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
  and
  [`rescnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
  reading the sequence of bins,
  [`mlp()`](https://gillescolling.com/timesift/reference/torch_learners.md)
  the bins flattened, each through a joint multi-label head
- **Your own**:
  [`learner()`](https://gillescolling.com/timesift/reference/learner.md)
  takes a fit and a predict pair, and
  [`register_learner()`](https://gillescolling.com/timesift/reference/register_learner.md)
  adds it to the registry; it then goes through the same folds, the same
  cells and the same scoring

The learners run on timesift’s own C++ cores, written from the published
methods, and those cores give the numbers of the references a biomod2
user already fits: glmnet’s elastic net, MASS’s stepwise search, rpart’s
tree, earth’s MARS, mda’s flexible discriminant analysis, mgcv’s GAM,
nnet’s network, maxnet’s Maxent and biomod2’s surface range envelope,
each pinned against the reference’s own output in the fixtures.
Architecture belongs to the constructor and training belongs to
[`train_control()`](https://gillescolling.com/timesift/reference/train_control.md),
so `train_control(epochs = 200, device = "cuda")` reaches every neural
learner of a run at once.

### Comparing grains

- **[`grain_ladder()`](https://gillescolling.com/timesift/reference/grain_ladder.md)**:
  fit at every grain and see where skill saturates
- **[`grain_contrasts()`](https://gillescolling.com/timesift/reference/grain_contrasts.md)**:
  compare every grain against a learner’s best one
- **[`paired_contrast()`](https://gillescolling.com/timesift/reference/paired_contrast.md)**:
  compare two arms cell by cell
- **[`select_grain()`](https://gillescolling.com/timesift/reference/select_grain.md)**:
  choose the grain inside the training data, and score the whole
  procedure
- **[`occlusion()`](https://gillescolling.com/timesift/reference/occlusion.md)**:
  which part of the record a fitted model reads
- **[`simulate_records()`](https://gillescolling.com/timesift/reference/simulate_records.md)**:
  sensor records whose response acts at a known temporal grain, to test
  the procedure against a truth

### Ensemble and prediction

- **[`ensemble()`](https://gillescolling.com/timesift/reference/ensemble.md)**:
  stacking by non-negative weights summing to one, fitted on out-of-fold
  predictions alone and minimising the response head’s own loss;
  `"mean"`, `"median"`, `"weighted"` and `"committee"` combine without
  fitting
- **[`ensemble_weights()`](https://gillescolling.com/timesift/reference/ensemble_weights.md)**:
  the weights fitted on every target, for prediction
- **[`project()`](https://gillescolling.com/timesift/reference/project.md)**
  and
  **[`range_change()`](https://gillescolling.com/timesift/reference/range_change.md)**:
  a fit onto rasters, and how each response’s range changes between two
  projections
- **[`pseudo_absences()`](https://gillescolling.com/timesift/reference/pseudo_absences.md)**,
  **[`block_cv()`](https://gillescolling.com/timesift/reference/cv.md)**,
  **[`env_cv()`](https://gillescolling.com/timesift/reference/cv.md)**:
  background draws and spatial or environmental folds

The combiner is handed the predictions, the response, the fold map and
the mask, and never a model.

## Calendar bins

A month is 28, 30 or 31 days, and a week starts on a Monday. `timesift`
bins on the calendar, so a monthly mean over three years stays in its
months rather than sliding through the seasons as 730-hour blocks would,
and it asserts every unit holds readings in every bin. A record that
fails that assertion is refused rather than padded, and
[`coverage()`](https://gillescolling.com/timesift/reference/coverage.md)
lays the same binning out as a count of readings per unit and bin, so
the logger that stopped early or the month the whole record skipped
shows as a row or a column of zeros.

``` r

attr(grain_matrix(d, plot, t, temp, grain = "month"), "bin_n")[1, 1:3]
#> 2021-09-01T00:00:00Z 2021-10-01T00:00:00Z 2021-11-01T00:00:00Z
#>                  720                  744                  720
```

A calendar of your own is a function: pass one that returns each
reading’s bin start, and seasons cut at the equinoxes bin like any named
grain. A record that begins away from a bin boundary gives a bin the
calendar does not fill; `bin_partial` marks those bins and
`partial = "drop"` removes them.

## Extreme days and extreme readings

`min` and `max` take the coldest and warmest single reading in a bin.
`cold_day` and `warm_day` reduce each day to its own mean first, then
take the extreme over days. `mean_daily_min` and `mean_daily_max` take
the mean of the daily extremes, the exposure a typical day of the bin
brought. One hour at -50 sets `min` to -50 outright and reaches the
day-level statistics only through its twenty-fourth of that day’s mean.

## A maximised TSS is optimistic

TSS is read at the threshold that maximises it, chosen on the same
held-out units the score is read on. That inflates the level where
presences are thin: on the Schrankogel design, by 0.110 on average when
the truth is 0.60.
[`tss_inflation()`](https://gillescolling.com/timesift/reference/tss_inflation.md)
measures it for your design and
[`implied_skill()`](https://gillescolling.com/timesift/reference/implied_skill.md)
says what population skill a level you read is consistent with. How
large the inflation is depends on how a model’s predictions are
distributed, so two models of equal skill can carry different inflations
and a paired TSS difference is not free of it. Runs are therefore scored
by AUC by default, with TSS beside it.

## R and Python agree

`inst/spec/representation.md` is normative, and `inst/spec/fixtures/`
holds a synthetic series with the digest of every grain-by-statistic
combination, alongside the reference coefficients of every shared
learner core. Both test suites assert against the same fixtures, so R
and Python give the same representation and the same baseline fits. [The
contract](https://gillescolling.com/timesift/articles/contract.html)
says what each language carries.

## Reproducing the study

`inst/reproduce/schrankogel.R` runs the published grid from the Zenodo
deposit it was built on, and asserts the plot count, the species count,
the cell count and the bin count of every grain before fitting anything.
[Its
README](https://github.com/gcol33/timesift/blob/master/inst/reproduce/README.md)
says how to run each stage, what it costs, and how every number compares
with the paper.

## Documentation

- [Choosing how a record is
  read](https://gillescolling.com/timesift/articles/timesift.html) -
  Getting started in R
- [Choosing how a record is read, in
  Python](https://gillescolling.com/timesift/articles/python.html) -
  Getting started in Python
- [Representing a
  record](https://gillescolling.com/timesift/articles/representations.html) -
  Grains, statistics, calendars, lookbacks and the arrays they give
- [Learners](https://gillescolling.com/timesift/articles/learners.html) -
  The shipped learners, training, tuning, and a learner of your own
- [Finding the
  grain](https://gillescolling.com/timesift/articles/comparing-grains.html) -
  Ladders, paired contrasts and nested selection on a record with a
  known answer
- [Ensembles, thresholds and
  maps](https://gillescolling.com/timesift/articles/prediction.html) -
  Folds, stacking, prediction, thresholds and projection onto rasters
- [Coming from
  biomod2](https://gillescolling.com/timesift/articles/biomod2.html) -
  The biomod2 models and where they sit in timesift
- [Troubleshooting](https://gillescolling.com/timesift/articles/troubleshooting.html) -
  The errors a run raises, what causes them, and the fix
- [The representation
  contract](https://gillescolling.com/timesift/articles/contract.html) -
  What R and Python share, and what each carries
- [R reference](https://gillescolling.com/timesift/reference/index.html)
  · [Python
  reference](https://gillescolling.com/timesift/articles/python-reference.html)

Bug reports and questions go to the [issue
tracker](https://github.com/gcol33/timesift/issues).

## Support

> “Software is like sex: it’s better when it’s free.” — Linus Torvalds

I’m a PhD student who builds R packages in my free time because I
believe good tools should be free and open. I started these projects for
my own work and figured others might find them useful too.

If this package saved you some time, buying me a coffee is a nice way to
say thanks. It helps with my coffee addiction.

[![Buy Me A
Coffee](https://img.shields.io/badge/-Buy%20me%20a%20coffee-FFDD00?logo=buymeacoffee&logoColor=black)](https://buymeacoffee.com/gcol33)

## Citation

``` bibtex
@software{timesift,
  author = {Colling, Gilles},
  title  = {timesift: Learn Predictive Representations of Time-Varying Data},
  year   = {2026},
  url    = {https://gillescolling.com/timesift/}
}
```

## License

MIT (see the LICENSE file)
