# timesift 0.1.1

## New

* `coverage()` lays the binning out as a count of readings per `(unit, bin)`, over every bin the
  calendar tiles the record with. `grain_matrix()` refuses a record where a unit misses a bin
  rather than padding it, and this is the table that decision is made on: a logger that started
  late, stopped early or lost a month is a row with zeros in it, and a bin the whole record skips
  is a column of zeros. Nothing here fills a cell. On both sides, and pinned by
  `inst/spec/fixtures/coverage.csv`.

## The response head

* A head carries a `loss` and an `activation`, and they are what every shipped learner fits
  toward. The encoders train under the loss and predict through the activation, the per-response
  learners take the family the loss names, and the combiner minimises the same loss. A registered
  head other than presence-absence is therefore fitted as itself rather than as clamped binary
  cross-entropy under another name.
* A `fit` that declares a `head` argument is handed the head, as one that declares `control` is
  handed the control. Both are read from the fit's own signature, at the one point every fold loop
  fits through, so a learner whose fit takes `(x, y)` is called with `(x, y)`.
* `elasticnet()`, `stepwise()` and `forest()` fit the Gaussian family where the head's loss is
  squared error.
* `ensemble()` left without a response takes the run's head; one naming a different head is
  refused before a candidate is fitted.

## Fitting

* A response is fitted under a seed of its own, taken from its name. A learner that covers the
  responses one at a time is handed one column per call, so a seed spent from one shared stream
  made a response's fit depend on which responses were fitted before it, and a seed read off the
  column's position inside its own call gave every response the same one. Either way the model of
  a response depended on how the columns were batched, and `elasticnet()` and `forest()` both did.
  Fitting a response alone, or beside others, or in a different order now gives the same model,
  which `tests/testthat/test-variable-seeds.R` holds to. This changes the numbers those two
  produce; `stepwise()` spends no randomness and the encoders cover the responses jointly, so
  neither moves.
* `metric` takes a registered name or a function of `(y, p)` everywhere the contract says it does.
  What scores and what a report prints travel with the fit, so a function reads as `<function>` in
  a report rather than as whatever each language calls an anonymous one, and the occlusion profile
  rescores under the metric the fit was scored with. `select_grain()` is the one door that takes a
  name only, because it reports its estimate under every registered metric and selects on a row of
  that table.
* A static predictor enters the design once rather than once per bin. It reaches the array as a
  channel that does not move across the bins, which is what an encoder reads; flattening the bins
  was emitting it once per bin, so a penalised fit saw it as often as the grain had bins and a
  forward search could pick it repeatedly.
* A resampling given as an unnamed vector follows the targets into the order they are fitted in.
  `timesift()` sorts the targets by identifier before it builds the fold map, so such a vector was
  landing on whichever unit had taken that position.
* `elasticnet()` takes `s`, the point of the penalty path to predict at. It also raises what
  `cv.glmnet()` raises rather than turning any failure into the response's own mean, and the
  forward search leaves out a column holding a single value rather than reaching a fitter's error
  on it.
* On the Python side `elasticnet()` honours its mixing -- the scikit-learn floor is 1.8, where
  `l1_ratios` alone selects the elastic net -- and standardises its design before penalising it,
  as glmnet does. The scaler travels with the fit.

## The representation boundary

* A supplied calendar is checked before its bins are read as bins: a bin begins at or before every
  reading it holds, and a bin's readings are a stretch of the record. A calendar shifted by one
  boundary, and one interleaving consecutive readings, used to produce an array that looked like
  any other.
* An identifier is written by one rule in both languages. A whole number is its digits, a
  character id is itself, a factor is its label, and anything else is refused naming the column.
* A time column carrying a zone names the calendar to bin by, in `grain_matrix()`, `coverage()`
  and `lookback_matrix()` alike; a `tz` naming a different zone beside it is refused rather than
  silently preferred.
* The lookback's target table is read by name, `id` and `at`, as every other alignment in the
  package is, and a lookback reaching the fitting layer without an anchor is refused with a
  message that says so.
* A reading that is not a finite number is refused by the shared core, naming the unit and the
  instant of the first one, so the rule is written once rather than once per wrapper.
  `digest_array()` refuses an array that is not finite for the same reason the contract gives.
* A record and any permutation of its rows now reduce to the same bytes. Both reductions walk the
  record by unit and then by instant; the calendar reduction accumulated in the caller's order,
  which moved a mean in its last bits under a shuffle that changed nothing about the record.

## Packaging

* `DESCRIPTION` is where the version is written, and `pyproject.toml` reads it from there.
* `.gitattributes` holds the repository to one line ending.

# timesift 0.1.0

`timesift()` is the whole entry point. A table of targets, a table of time-stamped series belonging
to them, and one call builds every candidate representation, fits the learners that can read each
one, scores them all on one set of held-out folds, and stacks the out-of-fold predictions.

```r
fit <- timesift(plots, logger, y = starts_with("sp_"), id = plot_id, time = datetime,
                models = c(elasticnet(), forest(), cnn()),
                sift = grains("day", "week", "month"))
summary(fit)
```

The version reads 0.1.0 because this is a first release: the package fits any time-varying record
against any prediction target, where its predecessors fitted a climate record at a climate grain.
Species distribution modelling from microclimate loggers is the application it ships defaults for,
and the Schrankogel grid it was built on still reproduces from `inst/reproduce/schrankogel.R`.

## The four concepts

* A **target** is one row to predict, a **series** is the long record belonging to those rows, a
  **representation** is how that record becomes an array, and a **learner** is a fit and a predict
  pair. A candidate is one (representation, learner) pair, and every candidate emits an out-of-fold
  prediction for every scorable cell over the same folds. Comparison, ensembling and importance
  read only those predictions.
* `y`, `x` and `static` take tidyselect expressions over their own table, and `starts_with()`,
  `ends_with()`, `contains()`, `matches()`, `all_of()`, `any_of()`, `everything()` and `where()`
  are re-exported rather than redefined. A column of `targets` that is neither the response nor the
  identifier nor the anchor is a predictor only where `static` names it.
* `predict()` on a fit rebuilds each member's representation for the new targets from the settings
  its own arm was built with, and combines them through the ensemble.

## Representations

* `native()`, `grain()`, `multigrain()` and `lookback()` are what a representation is before any
  record has been read; `grains()` and `lookbacks()` are sets of them, and `grains("auto")` reads
  off the record every named grain it gives at least two bins.
* `c()` combines learners, or representations, into a set: `c(elasticnet(), forest())` and
  `c(grains("day", "week"), lookback("30 days"))`. A set handed back to `c()` splices, so a set can
  be added to rather than rewritten. `models` and `sift` take that, a bare spec, or a list.
* `lookback()` is a fixed span of record ending a fixed lag before each target's own instant, which
  is what a unit carrying several targets through time needs. It is one entry point in `src/`
  beside the calendar reduction, so both languages read it from the same implementation, and
  `lookback_matrix()` exposes it directly.
* `build_representation()` is the one place the fitting layer turns a representation and the two
  tables into an array, so a run and a prediction on new targets reach a record the same way.
* A learner given `data = grain("week")` runs at that representation alone; left open it runs
  across the whole sift. A learner that reads a block of features and one that reads a sequence say
  so, and a pairing neither can carry is reported by name rather than fitted.

## Learners and training

* `elasticnet()`, `stepwise()`, `mlp()`, `cnn()` and `rescnn()` drop the `_learner` suffix and
  gain `data`. `forest()` joins them, a probability forest on `ranger` in R and on scikit-learn in
  Python. `reads` and `multi` are `learner()`'s, where a learner of your own declares what it reads
  and whether one fitted model covers every response.
* `train_control()` is the one place a training setting is defaulted. The architecture constructors
  carry architecture, a run gives one control to every neural learner, and a learner given its own
  control overrides that on the settings it names.
* A learner declares whether one fitted model covers every response or one is fitted per response.
  Either way a candidate emits one `[target, response]` matrix, so nothing above the learner layer
  has to know which it was.

## Resampling and the ensemble

* `cv()` and `grouped_cv()`; `resampling` also takes a fold vector or a `fold_map()` result, which
  is how a split the package has no constructor for reaches the same fitting path.
* `ensemble()` fits non-negative weights summing to one on the out-of-fold predictions alone,
  minimising the response head's own loss over the scorable cells, solved by an exponentiated-
  gradient loop in the package. `"mean"`, `"median"` and `"weighted"` combine without fitting.
  `ensemble_fit()` is handed the predictions, the response, the mask and the fold map, and never a
  model.
* `summary()` reports each candidate's mean, how many responses it scored highest on, whether one
  model covered them, and the level the combination reached under the weights it reached it with.

## Names

* `native()` rather than `raw()` and `lookback()` rather than `window()`, which would have masked
  `base::raw()` and `stats::window()`. `occlusion()` is one generic over a run and a ladder, and
  `bin_occlusion()` is gone. `ensemble_learner()` is gone with it: it fitted its members and
  averaged them, which the stack does over any candidates at all and with the weights fitted rather
  than assumed.
* The bin's name is the grain throughout, in both languages.

# climgrain 0.3.0

The package is renamed from `timegrain` to `timesift`. With it go the C++ prefix (`tg_` to `ts_`
and the four files that carried it), the S3 classes (`timegrain_matrix` and its siblings to
`timesift_matrix`), `timegrain_set()` to `timesift_set()`, and the Python module. No function
signature, default or return type changed, and the fixture digests are untouched.

What the user chooses is the temporal grain of a climate representation, and the name now carries
the thing whose grain it is. The title reads "Temporal Climate Resolution for Ecological
Prediction".

# climgrain 0.2.0

The binning and the reduction are now one implementation, `src/ts_core.cpp` and
`src/ts_calendar.cpp`, compiled into the R package by R itself and into the Python extension by
CMake. The two languages agree by construction rather than by two implementations being checked
against each other after the fact. What each side keeps above it is the boundary: resolving the
columns, resolving the zone, and wrapping the result.

Four bugs that existed twice, once per language, are closed by that (#1, #2, #4, #5), and a fifth
on the Python side with them (#3).

## The calendar

* Bin starts are computed by proleptic Gregorian arithmetic on local time rather than by writing a
  local midnight and parsing it back. A zone that moves its clock at midnight, such as
  `America/Sao_Paulo` before 2019, no longer produces `NA` bin starts and a failure from inside the
  reduction, and a `year_start` landing on such a night is an argument rather than an error (#1).
* A bin start is a local time, so reporting it as an instant now has a stated rule: one the clock
  skipped resolves to the instant the clock jumped to, one the clock repeated to the first of the
  two.
* `grain_matrix()` in Python takes a `tz` argument. The same instants and the same zone now give
  the same answer in both languages, and the fixtures pin it rather than leaving it assumed (#5).
* Instants are read at whole seconds in both languages, so two readings a fraction of a second
  apart are the same reading twice.

## Guards

* Consecutive bin starts must be one bin apart on the grain's own calendar. A bin no unit reaches
  is never built, so a month missing from the whole record used to pass as four adjacent monthly
  bins with one gone, in both languages (#4). Not asserted for `native`, whose bin is the reading
  itself, nor for a supplied calendar, which declares its own bin lengths.
* A day-level statistic requires every calendar day to lie inside one bin, decided from the bins
  rather than from the grain's name. A supplied calendar cutting inside a day used to give a
  mostly-`NA` array in R and a numpy `IndexError` in Python (#2).
* Python no longer rejects a unit called `nan` and no longer misses a genuinely missing id (#3).

## Evidence

* The pure-R and pure-NumPy implementations are kept as test oracles,
  `tests/testthat/helper-oracle.R` and `python/tests/oracle.py`. Neither package reaches them at
  runtime; both suites check them against the core on the fixtures and on random series. The NumPy
  one was written from `inst/spec/representation.md` rather than from the R source, which is what
  makes it evidence that the document is complete.
* A third fixture series and a `tz` column in `digests.csv`: every grain of the aligned series
  read as a `Europe/Vienna` clock, and a short series across the night `America/Sao_Paulo` moved
  its clock at midnight.

## The two languages

The names, the defaults and the extension points had drifted, so a script moved from one side to
the other met them one at a time (#8). One name per concept now, and where the two still differ the
difference is a row in `inst/spec/representation.md` rather than something to be discovered at a
call site.

* `glmnet_learner()` is `elasticnet_learner()`, and its `nfolds` is `n_inner`. The name says which
  model is fitted rather than which package fits it, which is also what the Python side already
  called it. It is registered as `"elasticnet"`.
* Python carries the response and metric registries, so on both sides the response head and the
  metric are registry entries and the fitting path holds no list of names. `learners()`,
  `metrics()` and `responses()` list what a session has, in C collation.
* `stepwise_learner()` and `select_grain()` are on the Python side. The selector's orthogonal
  polynomial basis and its logistic fits agree with R's `poly()` and `glm()` to twelve decimals on
  the same design; the selection procedure is the same nested one, reporting its estimate under
  every registered metric.
* Python's `grain_matrix()` returns one representation when one grain is named, whether as a
  string or as a sequence of one, and a `timesift_set()` for two or more. Its fold map carries the
  units it was drawn for, so aligning one is by name there as it already was in R.
* The torch encoders take `swa` and `swa_start` on both sides, and both refuse a setting the
  learner does not have rather than ignoring it. A misspelled argument used to be dropped in
  silence.
* `select_grain()` searches its candidates in the order they were declared. It read them off a join
  on the names before, which is the collation the rest of the package stopped depending on in this
  version, so an exact tie on the inner score could fall to a different candidate on a different
  machine.

## Build

* `LinkingTo: cpp11`, and flat `.cpp` under `src/`, so R compiles the core with no `Makevars`.
* `pyproject.toml` and `CMakeLists.txt` sit at the repository root rather than under `python/`,
  because a Python source distribution cannot reach above its own project directory and the shared
  sources must not be vendored into a second copy. The Python build is scikit-build-core and
  nanobind; the wheel carries `python/timesift` as `timesift`.
* The wheel depends on `tzdata` on Windows, which ships no IANA database of its own, so a
  zoned record bins there as it does everywhere else. The `sklearn` extra asks for a version
  that the declared floor of Python 3.10 can install.
* `R-CMD-check`, `pytest` and `contract` run on push and on pull requests, the last of them
  running both fixture suites against one `inst/spec/fixtures/` in a single job.

## Documentation

* One site for both languages at <https://gillescolling.com/timesift/>: the R reference from the
  Rd files, the Python reference written from the Python sources by `tools/python_reference.py`,
  and `inst/spec/representation.md` rendered as a page of its own, so the document the two answer
  to is read where the calls are. The `pytest` workflow rewrites the Python pages and fails if what
  is on disk differs, so a docstring cannot change without the page changing with it.
* Every public class, method and property of the Python package carries a docstring.

# climgrain 0.1.0

First release. The package builds the representation, fits at every grain, and reports where
predictive skill saturates.

## Representation

* `grain_matrix()`: reduces a long table of sensor readings to a `[unit, bin, channel]` array at
  one of seven temporal grains, from the unreduced record to a single value per hydrological year.
  Naming several grains returns one representation per grain; passing a function bins by a
  calendar the package does not carry, such as seasons cut at the equinoxes.
* Bins follow the calendar, so a month is 28, 30 or 31 days and a week starts on a Monday, and
  every `(unit, bin)` cell is asserted to hold readings.
* A bin the record does not cover for its whole calendar span is reported on `bin_partial` and
  kept or removed by the `partial` argument, so a record that begins away from a bin boundary
  says so rather than carrying a short bin that looks like any other.
* Seven statistics: `mean`, `min`, `max`, the day-level `cold_day` and `warm_day`, which reduce
  each day to its own mean before taking the extreme over days, and `mean_daily_min` and
  `mean_daily_max`, which take the mean of the daily extremes.
* `calendar_channels()` and `bind_channels()` supply the position of each bin in the year to an
  encoder that would otherwise pool it away.
* `feature_matrix()` brings an already-reduced feature table in as an arm of the same ladder.
* Gaps, duplicated `(unit, time)` pairs and missing values are errors rather than silent padding.

## Fitting and scoring

* `fold_map()` and `scorable_cells()`: one split read by everything that scores, and the mask of
  cells a score is defined on, computed from the response and the fold map with no model involved,
  so every arm shares one denominator and every paired difference runs on matched cells.
* `grain_ladder()` fits every learner at every grain, `plot()` draws the curve, and
  `paired_contrast()` compares two arms inside each cell both scored.
* `bin_occlusion()` holds each bin of the record back and rescores, without refitting, so a fitted
  model says which part of the year its skill rests on.
* `grain_contrasts()` fits `score ~ grain + (1 | variable) + (1 | fold)` and compares every
  grain against a learner's best by Dunnett's procedure. Needs `lme4`, `lmerTest` and `emmeans`.
* `tss()`, `roc_auc()`, `kappa_score()`, `model_agreement()` and `decision_threshold()`.
* `tss_inflation()` measures how much a self-selected threshold inflates a reported level at the
  user's own presence counts, and `implied_skill()` inverts that map to say what population skill
  a level actually read is consistent with.

## Learners

* `elasticnet_learner()` and `stepwise_learner()` on the flattened representation, both redoing their
  selection inside whichever units they are handed.
* `mlp_learner()`, `cnn_learner()` and `rescnn_learner()`, joint multi-label encoders on `torch`,
  sharing one training recipe.
* `ensemble_learner()` averages several members' predicted probabilities before the threshold is
  chosen, and the torch learners take `swa = TRUE` to average the weights of their tail epochs.
* `learner()` takes a fit and a predict pair of your own; `register_learner()`,
  `register_response()` and `register_metric()` extend the three registries the fitting path reads.

## Reproducing the study

* `inst/reproduce/schrankogel.R` runs the published grid from the Zenodo deposit it was built on,
  asserting the plot count, the species count, the cell count and the bin count of every grain
  before fitting anything. See `vignette("reproducing-schrankogel")`.

## The cross-language contract

* `inst/spec/representation.md` is normative for both the R and the Python implementation.
* `inst/spec/fixtures/` carries a synthetic series and the digest of every grain-by-statistic
  combination; both test suites assert against the same digests.
* The Python side implements the representation, the folds, the mask, the metrics, the ladder and
  the same three encoders.
