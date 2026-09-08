# timesift

Learn predictive representations of time-varying data. Flagship application: species modelling
from microclimate loggers.

A sensor records every hour for years. Before any model is fitted, that record is reduced: to
monthly means, to growing-degree-days, to whatever the analyst decides. `timesift` makes that
reduction an explicit, testable choice rather than a preprocessing step nobody revisits. It builds
each candidate representation, fits learners on every one they can read, scores them all on one set
of held-out folds, and stacks the out-of-fold predictions.

## Why it exists

From the Schrankogel study (`~/Documents/code/schrankogel-cnn-2026`, paper at
`~/Documents/writing/papers/paper_schrankogel_cnn_2026`), on 894 alpine plots, 101 species,
three years of hourly soil temperature:

- The full hourly series is the best input for none of three architectures. Reading every hour
  cost the convolutional network 0.048 TSS against its own best grain.
- Skill peaks at the weekly average and falls from monthly on (-0.080 at yearly).
- A window's coldest and warmest **day** carry more than its mean, and increasingly so as the
  window widens (+0.006 weekly to +0.046 yearly). They need daily values, so daily storage is the
  floor.
- A fully connected network on the same series is level with a penalised logistic model on 188
  hand-built features (-0.002 TSS, p = 0.63). The gain comes from convolution reading the series at
  a coarse grain, not from the model being a network.

The package is what lets someone else run that test on their own data.

## Positioning

The chain the package sits in is: a record, a representation of it, a temporal grain, a prediction.
Loggers are one input and species models are one application; the question the package hands the
user is at what temporal grain a record should be represented for a given prediction problem.

`timesift` names that: sifting a record for the resolution that carries the signal. It claims
neither a domain nor an input, which is what the earlier `timegrain` and `climgrain` both did
(`timegrain` reads as time-series storage outside ecology and an unrelated Obsidian plugin ranks
above the package under it; `climgrain` narrows to climate, where the code has no specialisation at
all). Ecology and species modelling give the package its audience, and the real specialisation is
presence-absence with rare responses, which lives in the shipped defaults rather than in the name.

## Layout

The R package sits at the repository root so every CRAN and pkgdown convention works unmodified.
`python/` is in `.Rbuildignore`. The contract ships with the package under
`inst/spec/`, so the test that asserts its digests runs on an installed copy rather
than only in the source tree.

`pyproject.toml` and `CMakeLists.txt` sit at the root too, and are in `.Rbuildignore`. That is not
tidiness: a Python source distribution cannot reach above its own project directory, so a
`python/pyproject.toml` would have to carry a vendored copy of `src/` and a CI job asserting the
copy still matched. The project directory is the repository, and there is one copy of the sources.

```
timesift/
  src/                the shared core, compiled into both languages
    ts_core.h ts_calendar.cpp ts_core.cpp
    ts_r.cpp          the cpp11 wrapper; cpp11.cpp is generated
  R/                  R package source
  tests/testthat/     including helper-oracle.R, the pure-R implementation
  man/ vignettes/
  DESCRIPTION NAMESPACE
  CMakeLists.txt pyproject.toml
  inst/
    spec/             the contract both implementations obey
      representation.md
      fixtures/       small input + expected digests, read by both test suites
    reproduce/        the driver that runs the published grid from the Zenodo deposit
    benchmark/        the selection benchmark's design, driver and summary
  python/             the Python twin
    ts_py.cpp         the nanobind wrapper
    timesift/
    tests/            including oracle.py, the pure-NumPy implementation
```

## The contract between the two languages

`grain_matrix()` in R and `grain_matrix()` in Python must return the **same numbers** for the
same input, and so must `lookback_matrix()`. This is not a nicety: the whole claim of the package
is that the grain is what matters, so two implementations that bin differently would make the tool
the confound.

`inst/spec/representation.md` is the normative description. `inst/spec/fixtures/` holds a small
input series and the digests of every grain-by-statistic combination. Both test suites read those
fixtures and assert against the same digests. A change to binning that is not reflected in the
fixtures is a bug in whichever language changed.

Models cannot be byte-identical across torch and libtorch and are not required to be.

## API

```r
fit <- timesift(targets, series, y = starts_with("sp_"), id = plot_id, time = datetime,
                models = c(elasticnet(), forest(), cnn()),
                sift = grains("day", "week", "month"),
                resampling = cv(v = 5))
summary(fit)
predict(fit, new_targets, new_series)
```

A **target** is one row to predict, a **series** is the long time-stamped record belonging to those
rows, a **representation** is how that record becomes an array, and a **learner** is a fit and a
predict pair. A candidate is one (representation, learner) pair, and every candidate emits an
out-of-fold prediction for every scorable cell over the same folds. Everything above that layer
reads only those predictions.

`grain_matrix()` and `lookback_matrix()` are the arrays themselves, reachable without the fitting
layer. Both return a numeric array `[unit, bin, channel]` with dimnames and the binning recorded in
attributes. Neither knows anything about species, or about what the response is.

### Representations

```r
native()                                   # the record unreduced, at grain "native"
grain(grain, stats = "mean")               # one calendar grain
multigrain(grains = NULL, stats = "mean")  # several grains bound into one tabular block
lookback(span, lag = "0 days", bins = 1L, stats = "mean")   # a span anchored on target_time
```

`grains()` and `lookbacks()` are sets of them, and `grains("auto")` expands to the named grains the
record gives at least two bins. A learner left open runs across the whole set; one given
`data = grain("week")` runs at that representation alone.

### Statistic vocabulary

The distinction between an extreme reading and an extreme day is the subtle part, and mixing them
up silently changes the result, so the names keep them apart:

| name | meaning |
|---|---|
| `mean` | mean of the readings in the bin |
| `min`, `max` | coldest and warmest single reading in the bin |
| `cold_day`, `warm_day` | coldest and warmest day, each day first reduced to its own mean |
| `mean_daily_min`, `mean_daily_max` | the bin's average daily minimum and maximum, each day first reduced to its own extreme |

The four day-level statistics are defined only for grains of a day or coarser. The reported input
in the paper is `c("cold_day", "mean", "warm_day")` at the weekly grain.

### Grains

`native`, `halfday`, `day`, `week`, `month`, `season`, `year`. `native` is the record as recorded,
one bin per reading. The four coarse grains follow the calendar rather than a fixed count of hours,
so a bin is a real month or a real week rather than a drifting block of 730 or 168 hours.
`year_start` sets the hydrological-year boundary (default `"09-01"`, the convention in the source
dataset).

A calendar the package does not carry is passed as a function of the reading instants returning
each reading's bin start. That is how the deposit's astronomical seasons, cut at the equinoxes and
solstices rather than on the first of a month, bin like any named grain.

A `lookback()` is the reduction the calendar cannot express: a fixed span ending a fixed lag before
each target's own instant, which is what a unit carrying several targets through time needs. A year
is 365 days and a month is 30 days there, because a lookback of a fixed length is a fixed length.

## Design rules

- **The response head and the metric are registered, not hard-coded.** Presence-absence with a
  joint multi-label head and TSS is the shipped default and the vignette, but adding an abundance
  or phenology response is one registration, never a fork of the fitting code. The head's `loss`
  and `activation` are what every shipped learner fits toward: the encoders train under the loss
  and predict through the activation, the per-response learners take the family the loss names,
  and the combiner minimises the loss. A fit that declares a `head` argument is handed the head,
  as one that declares `control` is handed the control; no learner holds a response of its own.
  Same for learners: `mlp()`, `cnn()`, `rescnn()`, `elasticnet()`, `stepwise()`, `forest()` and
  any user-supplied fit/predict pair go through one interface.
- **A fitted encoder is a plain object.** Its weights are arrays and its device is the setting,
  not the resolved device; the network is rebuilt at prediction. `saveRDS()` and `pickle` round
  trip a fit, and a fit made on one machine predicts on another.
- **Architecture is the constructor's, training is `train_control()`'s.** A training setting is
  defaulted in exactly one place, a run gives one control to every neural learner, and a learner
  given its own control overrides that on the settings it names.
- **The representation layer is one implementation, in C++, compiled into both languages.** The
  calendar binning, the lookback, the statistics, the array assembly and every guard live in
  `src/`, take naive local seconds and know nothing of a zone, a locale or tzdata. Each language
  resolves the columns and the zone above it and wraps the result below it, and that is all either
  holds. A calendar bug fixed in one place is fixed in both, which is the whole reason it is there:
  the four it replaced each existed twice.
- **The pure-R and pure-NumPy implementations are kept as test oracles**, never reachable at
  runtime. The NumPy one was written from the spec rather than from the R source, so it is the
  evidence that the spec is complete; one shared binary would otherwise make the agreement between
  the languages trivially true.
- **No dependency for the representation.** `cpp11` is header-only and `LinkingTo`; nanobind is a
  build dependency of the wheel. Nothing is added at runtime on either side.
- **No primary-plus-fallback paths.** A learner that needs `torch` declares it and errors without
  it. Two code paths diverge.
- **The scorable-cell mask is computed from the response and the fold map alone**, with no model
  involved, so every candidate in a run is scored on the same cells and every paired comparison
  runs on matched cells.
- **The combiner never sees a model.** `ensemble_fit()` is handed the out-of-fold predictions, the
  response, the mask and the fold map, and that is what keeps the stack honest.
- **No name that masks a base or recommended generic.** `native()` rather than `raw()`,
  `lookback()` rather than `window()`, and no `glm()` or `gam()` constructor: a user's own GLM, GAM
  or JSDM reaches the same interface through `register_learner()`.

## What ships, and what it warns about

TSS read at the threshold that maximises it is inflated where presences are thin: the Schrankogel
simulation puts the inflation at +0.110 when the true skill is 0.60, generated in cells holding one
or two presences. Most SDM code carries that silently. `timesift` reports the score and the
expected inflation for the user's own presence counts. That is a reason to switch that has nothing
to do with neural networks.

## Status

Version 0.1.1. Not on CRAN or PyPI yet. The version went down at the rename: 0.1.0 was a first
release under a new name and a general contract, not a fourth release of `climgrain`. `DESCRIPTION`
is where the string is written and `pyproject.toml` reads it from there, so a bump is one edit.

The build order is done on both sides: the representation and the fixtures, the lookback, the fold
map and the scorable-cell mask, the ladder and its plot, the learner registry, the torch learners,
the stack, and above them the paired contrast, the mixed-model grain contrast, the occlusion
profile and the inflation of a self-selected threshold. The Python side carries the same except the
mixed model, and reproduces every one of the 234 representation digests, sixteen of which pin a zone other than
UTC.

Both sides run on one C++ core. The implementations that used to be compared are kept as oracles
the suites check the core against. The last section of `inst/spec/representation.md` says what each
language carries, so a difference between them is a recorded decision: one name per concept, the
three registries and the nested selection on both sides, and the mixed-model contrast, the record
simulator and the plots in R alone.

`inst/reproduce/schrankogel.R` runs the published grid from the deposit and asserts its input at
every step. Verified against the deposit on 2026-09-02, matching the paper exactly:

| quantity | reported | reproduced |
|---|---|---|
| plots, species, rarest species | 894, 101, 26 | same |
| scorable cells | 1003 of 1010 | same |
| bins per grain | 26304, 2192, 1096, 157, 36, 13, 3 | same |
| numbers per plot, weekly three-channel | 471 | same |
| numbers per plot, daily three-channel | 3288 | same |
| inflation at truth 0.60, 0.70, 0.90 | +0.110, +0.095, +0.051 | same |
| elastic net on the 188 aggregates | 0.687 | 0.686 |

The elastic net sits 0.001 low because the two runs seed the inner cross-validation's random fold
draw differently; nothing else in the table carries randomness. The stepwise arm and the network
grid have not been rerun here: forward selection over 188 columns is many hours single-threaded,
and the encoders want the graphics processor they had in the study.

## Related

- Study code: `gcol33/schrankogel-cnn-2026` (private)
- Paper: `~/Documents/writing/papers/paper_schrankogel_cnn_2026`, target Methods in Ecology and
  Evolution as a Methods article with this package as the companion software
- Collaboration notes: `~/Documents/TaskGoblin/colabs/schrankogel-cnn/`
