# timesift <img src="https://gillescolling.com/timesift/reference/figures/logo.svg" align="right" height="139" alt="timesift logo" />

*choosing the time scale of a sensor record*

[![PyPI](https://img.shields.io/pypi/v/timesift)](https://pypi.org/project/timesift/)
[![pytest](https://github.com/gcol33/timesift/actions/workflows/pytest.yaml/badge.svg)](https://github.com/gcol33/timesift/actions/workflows/pytest.yaml)
[![contract](https://github.com/gcol33/timesift/actions/workflows/contract.yaml/badge.svg)](https://github.com/gcol33/timesift/actions/workflows/contract.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center"><img src="https://gillescolling.com/timesift/reference/figures/timesift-hero.png" alt="Three years of hourly soil temperature from one alpine plot drawn at seven time scales, from hourly readings down to yearly means, beside the true skill statistic a convolutional network reached on each over 101 species: 0.658 hourly, rising to a peak of 0.706 at the weekly mean, falling to 0.626 yearly" width="100%"></p>

timesift finds the time scale at which a long sensor record best predicts something, such as
whether a plant species grows at the plot where a soil temperature logger recorded every hour for
three years.

Before a model is fitted, a record like that is usually reduced once, to monthly means or
growing-degree-days, and the reduction is rarely revisited. timesift makes it an argument. Each
time scale is a *grain*: the record is cut into calendar bins, a real week or a real month, and
each bin is reduced to a statistic such as its mean. The package builds the record at every grain
you name, fits each learner on every grain it can read, and scores every pairing on the same
held-out folds and the same cells. You can then compare the grains cell by cell, choose one inside
the training data, or combine them into an ensemble.

timesift supports:

- **Calendar grains and lookbacks.** From the readings as recorded through half-days, days, weeks,
  months, seasons and years, with day-level statistics such as the coldest and warmest day of each
  bin, and `lookback()` spans that end at each target's own date.
- **The learners species distribution modelling uses.** Elastic net, stepwise GLM, CART, random
  forest, boosting, Maxent, surface range envelope, MARS, flexible discriminant analysis, GAM and a
  one-layer network, plus convolutional and fully connected networks through torch, and any
  learner of your own.
- **Report evaluation scores.** The grain and the ensemble weights are chosen inside each outer
  training fold, so the held-out score covers the choosing too.
- **Thin presences.** With few presences, the TSS read at its best threshold overstates skill.
  `tss_inflation()` measures the overstatement for your design, and `implied_skill()` turns a
  reported TSS into the skill it is consistent with.
- **Python and R.** This package and the [R package](https://gillescolling.com/timesift/) call one
  C++ core, so they give the same arrays and the same baseline fits.

Every learner core is timesift's own code, written from the published method and pinned in the
test fixtures against the reference package a species modeller already uses: glmnet, MASS, rpart,
earth, mda, mgcv, nnet, maxnet and biomod2. Both test suites assert the same 257 digests of the
binned record, so a difference between the languages fails a check. A methods section cites the
package version, which fixes every core and its defaults.

## Installation

```bash
pip install timesift
```

Wheels are built for CPython 3.11, 3.12 and 3.13 on Linux, Windows and macOS. On another
interpreter pip builds from the source distribution, which needs a C++17 compiler.

At runtime the package needs only NumPy. The optional parts are extras:

| Extra | Adds | For |
|---|---|---|
| `torch` | PyTorch | `cnn()`, `rescnn()` and `mlp()` |
| `contrasts` | SciPy | `grain_contrasts()` |
| `plot` | Matplotlib | `plot()` |
| `grid` | xarray | `project()` and `range_change()` |

```bash
pip install "timesift[torch,contrasts,plot]"
```

A function that needs one of them raises an error naming the package to install when it is
missing.

## Usage

1. Bring two tables: `targets`, one row per thing to predict with its response, and `series`,
   the time-stamped readings belonging to those rows.
2. Call `timesift()` with the response columns, the identifier both tables carry, the time column
   and the grains to compare.
3. Read the scores, then call `fit.predict()` on new rows.

`simulate_records()` draws a record with a known answer to try this on: 150 sensors with a reading
every three hours for a year, and four responses that depend on the record's seasonal level.

```python
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

The printed run lists every candidate with its score on the outer folds, then the `selected` and
`ensemble` rows. A *candidate* is one grain paired with one learner; `elasticnet()` is the learner
when none is named. The `selected` and `ensemble` rows are the numbers to report: the candidate
and the ensemble weights were chosen inside each outer training fold and then scored once on the
held-out fold.

`targets` and `series` are mappings of column name to array, which a pandas data frame satisfies.
`y` selects over its own table by a column name, a list of names, a glob such as `"sp_*"`, or a
function of a name. A table with a missing reading in a bin is refused, and `coverage()` shows
which unit and bin lack readings. A time column that carries a zone bins by that zone's clock.

`fit.predict(new_targets, new_series)` rebuilds each member's grain for the new rows and predicts
through the ensemble fitted on every target. `type="binary"` cuts each response into presence and
absence at a threshold learned from the out-of-fold predictions.

## The grain in an alpine study

timesift was built for a study of 894 alpine plots on the Schrankogel, Austria, with 101 plant
species and three years of hourly soil temperature per plot. The figure at the top shows that run:
the convolutional network scored 0.048 TSS lower on the hourly record than on its best grain, and
skill peaked at the weekly mean. The
[reproduction README](https://github.com/gcol33/timesift/blob/master/inst/reproduce/README.md)
compares every number with the paper.

## Working across R and Python

The binning, the statistics and the elastic net are one implementation, so the same input gives
the same arrays and the same coefficients in both languages. The fold builders draw on each
language's own random stream, so the same seed gives different folds; `write_folds()` saves a fold
map and `read_folds()` reads it in the other language.
[The contract between R and Python](https://gillescolling.com/timesift/articles/contract.html)
lists what each language carries.

## Documentation

Start with [Choosing how a record is read](https://gillescolling.com/timesift/articles/timesift.html?lang=python):
one path from two tables to a scored comparison and predictions.

- [Python reference](https://gillescolling.com/timesift/articles/python-reference.html): every function, grouped by job.
- [Representing a record](https://gillescolling.com/timesift/articles/representations.html), [Learners](https://gillescolling.com/timesift/articles/learners.html), [Finding the grain](https://gillescolling.com/timesift/articles/comparing-grains.html) and [Ensembles, thresholds and maps](https://gillescolling.com/timesift/articles/prediction.html): the concepts, with R code that the Python functions of the same name follow.
- [Troubleshooting](https://gillescolling.com/timesift/articles/troubleshooting.html): the errors a run raises and how to fix them.

## Getting help

Questions and bug reports go to the [issue tracker](https://github.com/gcol33/timesift/issues).
With a bug report, include a small example that reproduces it.

## Contributing

Building from a checkout, the test suites and the rules for the shared fixtures are in
[CONTRIBUTING.md](https://github.com/gcol33/timesift/blob/master/CONTRIBUTING.md).

## Support

> "Software is like sex: it's better when it's free." — Linus Torvalds

I'm a PhD student who builds R packages in my free time because I believe good tools should be free and open. I started these projects for my own work and figured others might find them useful too.

If this package saved you some time, buying me a coffee is a nice way to say thanks. It helps with my coffee addiction.

[![Buy Me A Coffee](https://img.shields.io/badge/-Buy%20me%20a%20coffee-FFDD00?logo=buymeacoffee&logoColor=black)](https://buymeacoffee.com/gcol33)

## License

MIT

## Citation

Download the citation as [BibTeX](https://gillescolling.com/timesift/timesift.bib) or
[RIS](https://gillescolling.com/timesift/timesift.ris), or use GitHub's "Cite this repository" on
the [repository](https://github.com/gcol33/timesift).
