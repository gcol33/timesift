# timesift

[![R-CMD-check](https://github.com/gcol33/timesift/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/gcol33/timesift/actions/workflows/R-CMD-check.yaml)
[![pytest](https://github.com/gcol33/timesift/actions/workflows/pytest.yaml/badge.svg)](https://github.com/gcol33/timesift/actions/workflows/pytest.yaml)
[![contract](https://github.com/gcol33/timesift/actions/workflows/contract.yaml/badge.svg)](https://github.com/gcol33/timesift/actions/workflows/contract.yaml)

`timesift` fits and compares representations of time-varying data
against a prediction target. Give it one row per thing to predict and a
long table of time-stamped readings belonging to those rows. It builds
each candidate representation, fits the learners you name on every one
they can read, scores them all on one set of held-out folds, and stacks
the out-of-fold predictions into an ensemble. What comes back says how
much of the record the prediction actually needed.

The same calls exist in both languages over one shared C++ core, and the
fixtures under `inst/spec/fixtures/` hold the two to the same numbers.

[TABLE]

`fit` prints every candidate it fitted, the score each reached on the
held-out folds, and the weights the stack gave them:

``` r

fit
#> timesift  80 targets, 4 responses, 5-fold random CV, tss
#>
#> candidate                    mean    won  responses
#> elasticnet / month          0.261      0  separate
#> forest / month              0.364      0  separate
#> forest / week               0.391      1  separate
#> elasticnet / day            0.445      2  separate
#> elasticnet / week           0.464      0  separate
#> forest / day                0.504      1  separate
#> ensemble                    0.524      -
#>
#> weights  forest / day 0.59   elasticnet / day 0.28   elasticnet / week 0.13
```

`predict(fit, new_plots, new_logger)` rebuilds every member’s
representation for the new rows and predicts through the ensemble.

## Four things, and one contract

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

## Representations

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
is what a unit carrying several targets through time needs: two targets
a fortnight apart on one sensor read two different stretches of the same
series, anchored by `target_time`.

## Learners

[`elasticnet()`](https://gillescolling.com/timesift/reference/elasticnet.md)
and
[`stepwise()`](https://gillescolling.com/timesift/reference/stepwise.md)
read a block of features,
[`forest()`](https://gillescolling.com/timesift/reference/forest.md)
grows a probability forest over one, and the `torch` encoders
[`mlp()`](https://gillescolling.com/timesift/reference/torch_learners.md),
[`cnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
and
[`rescnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
read a sequence with a joint multi-label head.
[`learner()`](https://gillescolling.com/timesift/reference/learner.md)
takes a fit and a predict pair of your own, which then goes through the
same folds, the same cells and the same scoring.

Architecture belongs to the constructor and training belongs to
[`train_control()`](https://gillescolling.com/timesift/reference/train_control.md),
so `train_control(epochs = 200, device = "cuda")` reaches every neural
learner of a run at once and a learner given its own control overrides
that on the settings it names.

## Calendar bins, not blocks of hours

A month is 28, 30 or 31 days, and a week starts on a Monday. Bins that
count hours instead drift away from both, so a “monthly” mean built from
730-hour blocks slides through the seasons over three years. `timesift`
bins on the calendar and asserts every unit holds readings in every bin.

``` r

attr(grain_matrix(d, plot, t, temp, grain = "month"), "bin_n")[1, 1:3]
#> 2021-09-01T00:00:00Z 2021-10-01T00:00:00Z 2021-11-01T00:00:00Z
#>                  720                  744                  720
```

A calendar of your own is a function: pass one that returns each
reading’s bin start, and seasons cut at the equinoxes bin like any named
grain.

A record that begins away from a bin boundary gives a bin the calendar
does not fill. `bin_partial` marks those bins and `partial = "drop"`
removes them, so the choice between a short bin and a lost end of the
record is one the caller makes.

## An extreme day is not an extreme reading

`min` and `max` take the coldest and warmest single reading in a bin.
`cold_day` and `warm_day` reduce each day to its own mean first, then
take the extreme over days. `mean_daily_min` and `mean_daily_max` take
the mean of the daily extremes, the exposure a typical day of the bin
brought. One hour at -50 sets `min` to -50 outright and reaches the
day-level statistics only through its twenty-fourth of that day’s mean.

## The ensemble

[`ensemble()`](https://gillescolling.com/timesift/reference/ensemble.md)
combines the candidates by stacking: non-negative weights summing to
one, fitted on the out-of-fold predictions alone, minimising the
response head’s own loss over the scorable cells. `"mean"`, `"median"`
and `"weighted"` combine without fitting. The combiner is handed the
predictions, the response, the fold map and the mask, and never a model.

``` r

ensemble_weights(fit)
```

## Ecology, and what the study found

Species distribution modelling from microclimate loggers is the
application the package was built for and the setting it ships defaults
for: presence-absence, a joint multi-label head, and the true skill
statistic. On 894 alpine plots, 101 species and three years of hourly
soil temperature:

- The full hourly series was the best input for none of three
  architectures. Reading every hour cost the convolutional network 0.048
  TSS against its own best grain.
- Skill peaked at the weekly average and fell from monthly on, by 0.080
  at yearly.
- A window’s coldest and warmest **day** carried more than its mean, and
  by more as the window widened: 0.006 weekly to 0.046 yearly.
- A fully connected network on the same series was level with a
  penalised logistic model on 188 hand-built features (-0.002 TSS, p =
  0.63), so the gain came from convolution reading the series at a
  coarse grain rather than from the model being a network.

## The score you report is an upper bound

TSS is read at the threshold that maximises it, chosen on the same
held-out units the score is read on. That inflates the level where
presences are thin: on the Schrankogel design, by 0.110 when the truth
is 0.60.
[`tss_inflation()`](https://gillescolling.com/timesift/reference/tss_inflation.md)
measures it for your design,
[`implied_skill()`](https://gillescolling.com/timesift/reference/implied_skill.md)
says what population skill a level you read is consistent with, and
[`paired_contrast()`](https://gillescolling.com/timesift/reference/paired_contrast.md)
is where it cancels, because both arms carry the same bias on the same
cell.

## The two languages agree

`inst/spec/representation.md` is normative, and `inst/spec/fixtures/`
holds a synthetic series with the digest of every grain-by-statistic
combination. Both test suites assert the same digests, so R and Python
cannot drift apart on the one thing the package is about. [The Python
pages](https://gillescolling.com/timesift/articles/python.html) are that
side, and [the
contract](https://gillescolling.com/timesift/articles/contract.html)
says what each language carries.

## Reproducing the study

`inst/reproduce/schrankogel.R` runs the published grid from the Zenodo
deposit it was built on, and asserts the plot count, the species count,
the cell count and the bin count of every grain before fitting anything.
[`vignette("reproducing-schrankogel")`](https://gillescolling.com/timesift/articles/reproducing-schrankogel.md)
says which setting corresponds to which part of that grid.

## License

MIT.
