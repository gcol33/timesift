# timesift

*choosing the time scale of a sensor record*

![Three years of hourly soil temperature from one alpine plot drawn at
seven time scales, from hourly readings down to yearly means, beside the
true skill statistic a convolutional network reached on each over 101
species: 0.658 hourly, rising to a peak of 0.706 at the weekly mean,
falling to 0.626 yearly](reference/figures/timesift-hero.png)

timesift finds the time scale at which a long sensor record best
predicts something, such as whether a plant species grows at the plot
where a soil temperature logger recorded every hour for three years.

Before a model is fitted, a record like that is usually reduced once, to
monthly means or growing-degree-days, and the reduction is rarely
revisited. timesift makes it an argument. Each time scale is a *grain*:
the record is cut into calendar bins, a real week or a real month, and
each bin is reduced to a statistic such as its mean. The package builds
the record at every grain you name, fits each learner on every grain it
can read, and scores every pairing on the same held-out folds and the
same cells. You can then compare the grains cell by cell, choose one
inside the training data, or combine them into an ensemble.

timesift supports:

- **Calendar grains and lookbacks.** From the readings as recorded
  through half-days, days, weeks, months, seasons and years, with
  day-level statistics such as the coldest and warmest day of each bin,
  and
  [`lookback()`](https://gillescolling.com/timesift/reference/native.html)
  spans that end at each target’s own date.
- **The learners species distribution modelling uses.** Elastic net,
  stepwise GLM, CART, random forest, boosting, Maxent, surface range
  envelope, MARS, flexible discriminant analysis, GAM and a one-layer
  network, plus convolutional and fully connected networks through
  torch, and any learner of your own.
- **Scores you can report.** The grain and the ensemble weights are
  chosen inside each outer training fold, so the held-out score covers
  the choosing too.
- **Thin presences.**
  [`tss_inflation()`](https://gillescolling.com/timesift/reference/tss_inflation.md)
  estimates how far a true skill statistic read at its own best
  threshold overstates skill for your design’s presence counts.
- **R and Python.** Both languages call one C++ core, so they give the
  same arrays and the same baseline fits.

Every learner core is timesift’s own code, written from the published
method and pinned in the test fixtures against the reference package a
species modeller already uses: glmnet, MASS, rpart, earth, mda, mgcv,
nnet, maxnet and biomod2. Both test suites assert the same 257 digests
of the binned record, so a difference between the languages fails a
check. A methods section cites the package version, which fixes every
core and its defaults.

## Installation

The CRAN release installs from a binary on Windows and macOS and needs
no build tools there:

``` r

install.packages("timesift")
```

The development version compiles C++ code, so it needs a compiler:

| Operating system | Build tools |
|----|----|
| Windows | [Rtools](https://cran.r-project.org/bin/windows/Rtools/) matching your R version |
| macOS | `xcode-select --install` |
| Debian / Ubuntu | `r-base-dev` |

Check that R finds them:

``` r

pkgbuild::has_build_tools(debug = TRUE)
```

Then install from GitHub:

``` r

# install.packages("pak")
pak::pak("gcol33/timesift")
```

The C++ code needs no system library: it uses only the C++ standard
library, including its threads, and [cpp11](https://cpp11.r-lib.org/)
headers at build time. The neural learners need
[torch](https://torch.mlverse.org/) (`install.packages("torch")`, then
[`torch::install_torch()`](https://torch.mlverse.org/docs/reference/install_torch.html)).

The Python package has wheels for Linux, Windows and macOS and needs
Python 3.11 or later:

``` bash
pip install timesift

# with the torch encoders, the grain contrasts and the plots
pip install "timesift[torch,contrasts,plot]"
```

## Usage

1.  Bring two tables: `targets`, one row per thing to predict with its
    response, and `series`, the time-stamped readings belonging to those
    rows.
2.  Call
    [`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
    with the response columns, the identifier both tables carry, the
    time column and the grains to compare.
3.  Read the scores, then call
    [`predict()`](https://rdrr.io/r/stats/predict.html) on new rows.

[`simulate_records()`](https://gillescolling.com/timesift/reference/simulate_records.md)
draws a record with a known answer to try this on: 150 sensors with a
reading every three hours for a year, and four responses that depend on
the record’s seasonal level.

``` r

library(timesift)

sim <- simulate_records(n = 150, mechanism = "season", variables = 4,
                        prevalence = 0.3, auc = 0.9)
targets <- data.frame(unit = rownames(sim$y), sim$y)

fit <- timesift(targets, sim$readings, y = -unit, id = unit, time = time,
                sift = grains("day", "week", "month", "season"))
fit
#> timesift  150 targets, 4 responses, 10-fold random CV, roc_auc
#> 
#> candidates, scored on the outer folds
#> candidate                     mean    won  responses
#> elasticnet / day             0.553      0  separate
#> elasticnet / season          0.571      0  separate
#> elasticnet / week            0.612      0  separate
#> elasticnet / month           0.712      4  separate
#> 
#> procedure, chosen and weighted inside each outer training fold
#> selected                     0.712  se 0.077
#> ensemble                     0.712  se 0.077
#> selected elasticnet / month in 10 of 10 folds
#> 
#> choice on every target  elasticnet / month
#> weights on every target  elasticnet / month 1.00
```

`targets` holds one 0/1 column per response and the identifier, here
`unit`; `series` holds that identifier, a date-time column and the
values, one row per reading. `y` takes tidyselect expressions such as
`starts_with("sp_")`. A table with a missing reading in a bin is
refused, and
[`coverage()`](https://gillescolling.com/timesift/reference/coverage.md)
shows which unit and bin lack readings.

A *candidate* is one grain paired with one learner;
[`elasticnet()`](https://gillescolling.com/timesift/reference/elasticnet.md)
is the learner when none is named. The candidate rows score each one on
the outer folds, and the highest of them was picked out on the same
folds it is scored on. The `selected` and `ensemble` rows are the
numbers to report: the candidate and the ensemble weights were chosen
inside each outer training fold and then scored once on the held-out
fold.

`predict(fit, new_targets, new_series)` rebuilds each member’s grain for
the new rows and predicts through the ensemble fitted on every target.
`type = "binary"` cuts each response into presence and absence at a
threshold learned from the out-of-fold predictions.

The same run in Python:

``` python
import timesift as ts

sim = ts.simulate_records(n=150, mechanism="season", variables=4,
                          prevalence=0.3, auc=0.9)
targets = {"unit": sim.y.units}
targets.update(zip(sim.y.variables, sim.y.values.T))

fit = ts.timesift(targets, sim.readings, y=list(sim.y.variables),
                  id="unit", time="time",
                  sift=ts.grains("day", "week", "month", "season"))
print(fit)
```

## The grain in an alpine study

timesift was built for a study of 894 alpine plots on the Schrankogel,
Austria, with 101 plant species and three years of hourly soil
temperature per plot. The figure at the top shows that run:

- The hourly record was the best input for none of three network
  architectures. The convolutional network scored 0.048 TSS lower on it
  than on its best grain.
- Skill peaked at the weekly mean and fell from monthly on, 0.080 lower
  at yearly.
- The coldest and warmest day of a bin carried more than its mean, by
  0.006 weekly up to 0.046 yearly.
- A fully connected network on the same series scored level with a
  penalised logistic model on 188 hand-built features (-0.002 TSS, p =
  0.63).

`inst/reproduce/schrankogel.R` runs that grid from the [Zenodo
deposit](https://doi.org/10.5281/zenodo.17047026) and checks the plot,
species, cell and bin counts before fitting anything. [Its
README](https://github.com/gcol33/timesift/blob/master/inst/reproduce/README.md)
compares every number with the paper.

## Reading a record at several grains

`grains("day", "week", "month")` names a set of grains to compare, and
`grains("auto")` takes every named grain that gives the record at least
two bins. A learner runs across the whole set, or at one grain when it
is given one, as in `cnn(data = grain("week"))`.

A month is 28 to 31 days and a week starts on a Monday, so a monthly
mean over three years stays in its months. `year_start` moves the year
boundary, for example to a hydrological year, and a function of the
reading times can define a calendar of your own, such as seasons cut at
the equinoxes.

`min` and `max` are the coldest and warmest single reading in a bin.
`cold_day` and `warm_day` first reduce each day to its mean, then take
the coldest and warmest day. One hour at -50 sets `min` to -50 and moves
`cold_day` by a twenty-fourth of that.

[`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md)
and
[`lookback_matrix()`](https://gillescolling.com/timesift/reference/lookback_matrix.md)
return the binned arrays, `[unit, bin, channel]`, without fitting
anything. [Representing a
record](https://gillescolling.com/timesift/articles/representations.html)
covers every grain and statistic.

## Choosing a grain and combining candidates

[`grain_ladder()`](https://gillescolling.com/timesift/reference/grain_ladder.md)
fits one learner at every grain and shows where skill levels off.
[`grain_contrasts()`](https://gillescolling.com/timesift/reference/grain_contrasts.md)
compares every grain with a learner’s best one, and
[`paired_contrast()`](https://gillescolling.com/timesift/reference/paired_contrast.md)
compares two candidates on matched cells.
[`select_grain()`](https://gillescolling.com/timesift/reference/select_grain.md)
chooses the grain inside the training data and scores the whole
procedure. [Finding the
grain](https://gillescolling.com/timesift/articles/comparing-grains.html)
runs them on a record with a known answer.

[`ensemble()`](https://gillescolling.com/timesift/reference/ensemble.md)
stacks the candidates with non-negative weights summing to one, fitted
on their out-of-fold predictions alone.
[`project()`](https://gillescolling.com/timesift/reference/project.md)
maps a fit onto rasters and
[`range_change()`](https://gillescolling.com/timesift/reference/range_change.md)
compares two projections. [Ensembles, thresholds and
maps](https://gillescolling.com/timesift/articles/prediction.html) goes
from a scored run to maps.

## TSS read at its own best threshold

The true skill statistic (TSS) is usually read at the threshold that
maximises it, on the same held-out plots it scores. Where a species has
few presences that overstates skill: on the Schrankogel design by 0.110
on average when the true skill is 0.60.
[`tss_inflation()`](https://gillescolling.com/timesift/reference/tss_inflation.md)
estimates the overstatement for your own presence counts and
[`implied_skill()`](https://gillescolling.com/timesift/reference/implied_skill.md)
gives the true skill a score you read is consistent with. Runs are
scored by AUC by default, with TSS beside it.

## Documentation

Start with [Choosing how a record is
read](https://gillescolling.com/timesift/articles/timesift.html): one
path from two tables to a scored comparison and predictions. In Python,
start with [Choosing how a record is read, in
Python](https://gillescolling.com/timesift/articles/python.html).

- [Representing a
  record](https://gillescolling.com/timesift/articles/representations.html):
  grains, statistics, calendars, lookbacks and the arrays they give.
- [Learners](https://gillescolling.com/timesift/articles/learners.html):
  the shipped learners, training settings, tuning and a learner of your
  own.
- [Finding the
  grain](https://gillescolling.com/timesift/articles/comparing-grains.html):
  ladders, paired contrasts and nested selection.
- [Ensembles, thresholds and
  maps](https://gillescolling.com/timesift/articles/prediction.html):
  stacking, prediction, thresholds and projection onto rasters.
- [Coming from
  biomod2](https://gillescolling.com/timesift/articles/biomod2.html):
  each biomod2 model and its timesift learner.
- [Troubleshooting](https://gillescolling.com/timesift/articles/troubleshooting.html):
  the errors a run raises and how to fix them.
- [The representation
  contract](https://gillescolling.com/timesift/articles/contract.html):
  what R and Python share and what each carries.
- [R reference](https://gillescolling.com/timesift/reference/index.html)
  and [Python
  reference](https://gillescolling.com/timesift/articles/python-reference.html):
  every function, grouped by job.

## Getting help

Questions and bug reports go to the [issue
tracker](https://github.com/gcol33/timesift/issues). With a bug report,
include a small reproducible example made with
[reprex](https://reprex.tidyverse.org/).

## Support

> “Software is like sex: it’s better when it’s free.” — Linus Torvalds

I’m a PhD student who builds R packages in my free time because I
believe good tools should be free and open. I started these projects for
my own work and figured others might find them useful too.

If this package saved you some time, buying me a coffee is a nice way to
say thanks. It helps with my coffee addiction.

[![Buy Me A
Coffee](https://img.shields.io/badge/-Buy%20me%20a%20coffee-FFDD00?logo=buymeacoffee&logoColor=black)](https://buymeacoffee.com/gcol33)

## License

MIT (see the LICENSE file)

## Citation

To cite timesift in publications use:

``` text
Colling G (2026). _timesift: Learn Predictive Representations of
Time-Varying Data_. R package version 0.6.0,
<https://gillescolling.com/timesift/>.
```

BibTeX:

``` bibtex
@Manual{colling2026timesift,
  title = {timesift: Learn Predictive Representations of Time-Varying Data},
  author = {Gilles Colling},
  year = {2026},
  note = {R package version 0.6.0},
  url = {https://gillescolling.com/timesift/},
}
```

Download the citation as
[BibTeX](https://gillescolling.com/timesift/timesift.bib) or
[RIS](https://gillescolling.com/timesift/timesift.ris).
