# Changelog

## timesift 0.1.1

### New

- [`coverage()`](https://gillescolling.com/timesift/reference/coverage.md)
  lays the binning out as a count of readings per `(unit, bin)`, over
  every bin the calendar tiles the record with.
  [`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md)
  refuses a record where a unit misses a bin rather than padding it, and
  this is the table that decision is made on: a logger that started
  late, stopped early or lost a month is a row with zeros in it, and a
  bin the whole record skips is a column of zeros. Nothing here fills a
  cell. On both sides, and pinned by `inst/spec/fixtures/coverage.csv`.

### The response head

- A head carries a `loss` and an `activation`, and they are what every
  shipped learner fits toward. The encoders train under the loss and
  predict through the activation, the per-response learners take the
  family the loss names, and the combiner minimises the same loss. A
  registered head other than presence-absence is therefore fitted as
  itself rather than as clamped binary cross-entropy under another name.
- A `fit` that declares a `head` argument is handed the head, as one
  that declares `control` is handed the control. Both are read from the
  fit’s own signature, at the one point every fold loop fits through, so
  a learner whose fit takes `(x, y)` is called with `(x, y)`.
- [`elasticnet()`](https://gillescolling.com/timesift/reference/elasticnet.md),
  [`stepwise()`](https://gillescolling.com/timesift/reference/stepwise.md)
  and
  [`forest()`](https://gillescolling.com/timesift/reference/forest.md)
  fit the Gaussian family where the head’s loss is squared error.
- [`ensemble()`](https://gillescolling.com/timesift/reference/ensemble.md)
  left without a response takes the run’s head; one naming a different
  head is refused before a candidate is fitted.

### Fitting

- Predicting from a fit whose new targets are missing a static predictor
  names the column that is missing. An absent column read as one that is
  not numeric, and the message sent the reader to encode something their
  table does not hold.
- On the Python side `models` takes one learner as well as a list of
  them, which is the form R has always taken; a single learner was
  reaching the fitting layer as a sequence of nothing.
- A learner is handed the whole response matrix whether it declares
  `joint` or `separate`, and the three that fit one model per response
  already did that inside their own fit. The fitting layer was splitting
  the responses as well, so the flattened block of predictors was
  rebuilt once for every response: on the published grid, 101 copies of
  a 47 MB matrix per fold and per arm. `multi` is now what the learner
  says it does with the matrix and what a report says of the candidate.
  The wrapper that reassembled the columns is gone from both sides, and
  with it the Python `CandidateFit`.
- One fold loop on the Python side, which the run, the ladder and the
  inner search of a selection all call, in place of three copies; one
  `_check_bins` and one `_check_grain`; and one baseline prediction per
  fold in the occlusion, which was rebuilding an encoder from its arrays
  once per response.
- A fit refers to the code that made it rather than carrying a copy of
  it. R writes a closure’s body out beside the closure, so
  [`saveRDS()`](https://rdrr.io/r/base/readRDS.html) on a fit copied
  every function that produced it and a fit read back after an upgrade
  rebuilt the old encoder and loaded the new weights into it. A fitted
  encoder now stores the name of its module builder, and a fit stores
  the name and the settings of its learner wherever the registry can
  rebuild it; a learner defined outside any registry still travels
  whole, because there is nowhere else for it to come from. The Python
  side stores the encoder’s name for the same reason.
- A candidate that holds no number on a scorable cell stops the run and
  names itself. A threshold metric returns `NA` on a cell whose
  predictions are `NaN` or infinite, which is the same `NA` a one-class
  cell returns, so a network whose training diverged was reported as an
  unscorable cell and dropped from the combiner without a word; the
  stack was then fitted on fewer cells than the mask said and nothing
  showed it. The combiner is now fitted on every scorable cell and
  refuses to drop one.
  [`score_predictions()`](https://gillescolling.com/timesift/reference/score_predictions.md)
  refuses the same predictions on both sides.
- [`grain_ladder()`](https://gillescolling.com/timesift/reference/grain_ladder.md)
  and
  [`select_grain()`](https://gillescolling.com/timesift/reference/select_grain.md)
  take a `control`, so the training settings of a ladder are given once
  for the run rather than restated on every neural learner in it. A
  selection hands the same control to the inner search and to the refit,
  and a learner carrying settings of its own still overrides it on the
  ones it names.
- A response is fitted under a seed of its own, taken from its name. A
  learner that covers the responses one at a time is handed one column
  per call, so a seed spent from one shared stream made a response’s fit
  depend on which responses were fitted before it, and a seed read off
  the column’s position inside its own call gave every response the same
  one. Either way the model of a response depended on how the columns
  were batched, and
  [`elasticnet()`](https://gillescolling.com/timesift/reference/elasticnet.md)
  and
  [`forest()`](https://gillescolling.com/timesift/reference/forest.md)
  both did. Fitting a response alone, or beside others, or in a
  different order now gives the same model, which
  `tests/testthat/test-variable-seeds.R` holds to. This changes the
  numbers those two produce;
  [`stepwise()`](https://gillescolling.com/timesift/reference/stepwise.md)
  spends no randomness and the encoders cover the responses jointly, so
  neither moves.
- `metric` takes a registered name or a function of `(y, p)` everywhere
  the contract says it does. What scores and what a report prints travel
  with the fit, so a function reads as `<function>` in a report rather
  than as whatever each language calls an anonymous one, and the
  occlusion profile rescores under the metric the fit was scored with.
  [`select_grain()`](https://gillescolling.com/timesift/reference/select_grain.md)
  is the one door that takes a name only, because it reports its
  estimate under every registered metric and selects on a row of that
  table.
- A static predictor enters the design once rather than once per bin. It
  reaches the array as a channel that does not move across the bins,
  which is what an encoder reads; flattening the bins was emitting it
  once per bin, so a penalised fit saw it as often as the grain had bins
  and a forward search could pick it repeatedly.
- A resampling given as an unnamed vector follows the targets into the
  order they are fitted in.
  [`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
  sorts the targets by identifier before it builds the fold map, so such
  a vector was landing on whichever unit had taken that position.
- [`elasticnet()`](https://gillescolling.com/timesift/reference/elasticnet.md)
  takes `s`, the point of the penalty path to predict at. It also raises
  what `cv.glmnet()` raises rather than turning any failure into the
  response’s own mean, and the forward search leaves out a column
  holding a single value rather than reaching a fitter’s error on it.
- On the Python side
  [`elasticnet()`](https://gillescolling.com/timesift/reference/elasticnet.md)
  honours its mixing – the scikit-learn floor is 1.8, where `l1_ratios`
  alone selects the elastic net – and standardises its design before
  penalising it, as glmnet does. The scaler travels with the fit.

### The representation boundary

- A supplied calendar is checked before its bins are read as bins: a bin
  begins at or before every reading it holds, and a bin’s readings are a
  stretch of the record. A calendar shifted by one boundary, and one
  interleaving consecutive readings, used to produce an array that
  looked like any other.

- An identifier is written by one rule in both languages. A whole number
  is its digits, a character id is itself, a factor is its label, and
  anything else is refused naming the column.

- A time column carrying a zone names the calendar to bin by, in
  [`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md),
  [`coverage()`](https://gillescolling.com/timesift/reference/coverage.md)
  and
  [`lookback_matrix()`](https://gillescolling.com/timesift/reference/lookback_matrix.md)
  alike; a `tz` naming a different zone beside it is refused rather than
  silently preferred.

- The lookback’s target table is read by name, `id` and `at`, as every
  other alignment in the package is, and a lookback reaching the fitting
  layer without an anchor is refused with a message that says so.

- A reading that is not a finite number is refused by the shared core,
  naming the unit and the instant of the first one, so the rule is
  written once rather than once per wrapper.
  [`digest_array()`](https://gillescolling.com/timesift/reference/digest_array.md)
  refuses an array that is not finite for the same reason the contract
  gives.

- A record and any permutation of its rows now reduce to the same bytes.
  Both reductions walk the record by unit and then by instant; the
  calendar reduction accumulated in the caller’s order, which moved a
  mean in its last bits under a shuffle that changed nothing about the
  record.

- The `native` grain is read on the instant rather than on the local
  clock, so the two readings of the hour a zone repeats when it sets its
  clock back are two bins. They used to share a local second and were
  merged into one bin holding their mean, on a grain the contract
  defines as the record unreduced; the `Europe/Vienna` native fixture
  pinned 9599 bins for 9600 readings, and is regenerated. The walk order
  both reductions accumulate in carries the instant as a third key, so
  those two readings are no longer a tie the sort resolved as it liked,
  and a zoned record already in order is no longer sorted again.

- A supplied calendar that returns no bin start for a reading, `NA` in R
  or `NaT` in Python, is refused naming the first such reading. It used
  to reach the core as a bin at the beginning of time, holding the
  reading its real bin then lacked, with nothing raised; the guard
  fixture carries the case as `missing`.

- A time column carried in a zone the database does not know is an error
  in R, as it was in Python. R’s zone resolution used to warn and read
  the clock in UTC.

- The contract now says that a lookback’s length is measured on the
  local clock, as everything below the zone boundary is: a one-day
  lookback ending at a local midnight holds 25 hours of record on the
  night a zone sets its clock back and 23 on the night it sets it
  forward. That is what keeps a calendar day whole inside a bin for the
  day-level statistics; both suites pin it.

- The guards above the core name what they refuse the same way on both
  sides: a missing identifier or instant names its column in Python as
  it did in R, a duplicated reading is named by its instant in UTC in R
  as it was in Python and the noun agrees with the count on both, and
  Python reads `bins = 3.0` as R reads `bins = 3`. A zoned record from
  before 1970 no longer raises `OSError` on Windows in Python. A
  lookback’s rows are named by the row names of `at` whatever kind of
  row names the frame carries, which is what the contract already said.

- The zoned digests have a witness that is not the core. Each oracle
  reads a zoned series as a clock in its zone and bins that clock with
  its own calendar, and both suites assert every `Europe/Vienna` and
  `America/Sao_Paulo` digest against it. The core’s negative-lag guard,
  its out-of-range unit index and its bound on the day table are
  exercised by both suites as well.

- The two orderings among the seven statistics,
  `min <= mean_daily_min <= mean <= mean_daily_max <= max` and
  `min <= cold_day <= mean <= warm_day <= max`, are asserted by both
  test suites on the core’s output and the oracle’s. They used to sit in
  a block of the core that neither build compiled.

- [`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md)
  and
  [`lookback_matrix()`](https://gillescolling.com/timesift/reference/lookback_matrix.md)
  on the Python side refuse a call that names no `value` column before
  reaching the compiled core, and a lookback’s label reads a zero lag as
  zero however it is spelled.

### Names

- [`as_sift()`](https://gillescolling.com/timesift/reference/grains.md)
  is the coercer to a set of representations on both sides, in place of
  R’s `timesift_sift()`; the class keeps its name.
- The last section of the contract records every public name of both
  languages, on the side it is on, and each suite reads that section
  against its own exports, so a name added to one side without a line
  there fails the suite. Python exports `resolve_metric` beside
  `get_learner`, as the contract had said it did.

### What a rare response weighs

- The response head owns it. A head’s `weights(y)` returns one case
  weight per cell of the response, and the encoders, the penalised fit,
  the forest and the forward search all fit under it: the encoders as an
  elementwise weight on the loss, the penalised fit and the forward
  search as case weights, the forest as the probability a unit is drawn
  into a tree’s bootstrap. The shipped presence-absence head carries
  [`positive_weights()`](https://gillescolling.com/timesift/reference/positive_weights.md),
  exported on both sides: each presence weighs the ratio of absences to
  presences among the fitting units, capped at 50, and each absence
  weighs one. A head registered with
  `weights = function(y) positive_weights(y, cap = 20)` weights every
  learner by that cap, and a head without `weights` fits unweighted.
- Three learners used to decide this for themselves, three ways. The
  encoders capped the weight at `pos_weight_cap` from
  [`train_control()`](https://gillescolling.com/timesift/reference/train_control.md),
  the elastic net and the forest weighted uncapped under
  `weight_positives`, and the forward search did not weight at all, so a
  response present in one target of a hundred weighed 99 in two
  learners, 50 in a third and 1 in the fourth, in the rare regime the
  network-against-aggregates comparison is about. `pos_weight_cap` and
  `weight_positives` are gone. On the study’s data, whose rarest species
  has 26 presences in 894 plots, the cap never binds, so the reproduced
  elastic net is the same number.
- Python’s forest draws each tree’s bootstrap by the weights itself, as
  ranger’s `case.weights` does. scikit-learn’s forest takes a sample
  weight into its impurities and leaf values, and a tree grown to pure
  leaves is the same tree under any weight, so the Python forest was
  fitting a rare response unweighted while the R forest was not.
- The forward search fits the quasi-binomial family and reads its
  criterion off the deviance, which is the same number the binomial
  family reports for a 0/1 response and takes a case weight that is not
  a whole number without a warning. The Python forward search runs its
  iteratively reweighted least squares under the same case weights.

### Fitting under a grouping

- A fit is checked once, on the `timesift_fit`, against the bins and the
  channels it was made on, before any learner sees the representation it
  is asked to predict. A calendar grain’s bins are named by their
  instants, so a record from another period with the same number of bins
  is refused by the first bin that differs rather than read by position,
  as
  [`cnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
  fitted on September to November used to read March to May. A
  lookback’s bins are relative to each target and predict any period.
  The check that each learner carried is gone; every learner, including
  one of your own, is held to this one. On both sides.
- The grouping an outer fold map keeps whole reaches every split drawn
  inside a fit. A \[fold_map()\] built with `group` carries it, a `fit`
  that declares a `group` argument is handed the grouping of the units
  it is fitted on, and the encoders’ inner validation set, the elastic
  net’s inner folds and
  [`select_grain()`](https://gillescolling.com/timesift/reference/select_grain.md)’s
  inner map are all dealt by it. Under
  [`grouped_cv()`](https://gillescolling.com/timesift/reference/cv.md)
  with `target_time` a unit kept whole across the outer folds used to be
  split across the inner ones, so the epoch and the penalty were chosen
  on rows the fit had already seen a near-copy of. On both sides. The
  elastic net deals its own inner folds now, through
  [`fold_map()`](https://gillescolling.com/timesift/reference/fold_map.md),
  rather than leaving the draw to `cv.glmnet()` or to scikit-learn.

### Fixes

- [`fold_map()`](https://gillescolling.com/timesift/reference/fold_map.md)
  deals with one round-robin counter that runs on from each stratum into
  the next, on both sides, so the folds are equal in size to within one
  unit whatever `v` is and leave-one-out is every unit in a fold of its
  own. A counter restarted in every stratum used the first labels once
  more than the last in every stratum, and a stratum smaller than `v`
  never reached the last labels at all: ten folds over thirty units came
  back holding two to five units each. The fold-map fixture, and the
  mask and the inflation built on it, are regenerated.
- A fold map given as `resampling` that holds one fold is refused, and
  one built by
  [`fold_map()`](https://gillescolling.com/timesift/reference/fold_map.md)
  keeps its `grouped`, `seed` and `strata` when it reaches the run, so a
  grouped design reports as one. A run under which no `(response, fold)`
  cell is scorable stops before anything is fitted, and
  [`ensemble_fit()`](https://gillescolling.com/timesift/reference/ensemble_fit.md)
  handed no scorable cell says so rather than fitting a combiner on
  nothing.
- The threshold metrics check that the response is 0/1 before coercing
  it, on both sides. They used to coerce first, which read 0.6 as 0 and
  scored a response that was never binary.
- A torch fit puts torch’s generators back where it found them, as it
  does R’s.
- `.simplex_weights()` stops at a relative gain of 1e-14 rather than
  1e-12, on both sides. Near the minimum the loss is flat to second
  order, so the stop settles the gradient to about the square root of
  the tolerance, and the looser one left the carried members’ gradients
  apart by more than a millionth on some boards.
- The within-variable-then-across-variables mean is one function,
  `.cell_means()`, read by the ladder, the report, the stack’s member
  scores and the selection’s estimate; the five copies had four
  spellings.
- Small edges: `models` and `learners` take a character vector of
  registered names, as the docs said; a fit read back through its
  reference keeps a setting held at `NULL`, so a forest refits with its
  default `mtry`; `train_control(early_stopping = Inf)` is a patience
  that never runs out;
  [`predict()`](https://rdrr.io/r/stats/predict.html) on a `timesift`
  fit holds the new targets to one row per identifier, as the fit’s own
  were held; the elastic net refuses a design of one column in its own
  words; and
  [`grain_contrasts()`](https://gillescolling.com/timesift/reference/grain_contrasts.md)
  names the reference it cannot find rather than failing on a
  length-zero test.
- Python: a learner pinned to a representation that shares a sift
  member’s label but not its definition is refused, as R refuses it,
  rather than fitted on the sift’s array.
- Python: `auto_grains()` counts the bins in the zone the time column
  carries, the clock the representation is then binned by. It used to
  probe in UTC, so a zoned record could be offered a grain it does not
  support or denied one it does.
- Python:
  [`grain()`](https://gillescolling.com/timesift/reference/native.md)
  takes a supplied calendar as R does, reported as `custom`;
  [`multigrain()`](https://gillescolling.com/timesift/reference/native.md)
  refuses one on both sides.
- Four differences between the languages are gone rather than recorded.
  [`grouped_cv()`](https://gillescolling.com/timesift/reference/cv.md)
  deals the groups unstratified on both sides. A lookback is labelled
  `<span> x<bins> lag <lag>` on both, and a numeric span as a duration.
  [`lookback()`](https://gillescolling.com/timesift/reference/native.md)
  refuses a span that is not positive when it is built.
  [`forest()`](https://gillescolling.com/timesift/reference/forest.md)
  and
  [`elasticnet()`](https://gillescolling.com/timesift/reference/elasticnet.md)
  derive one seed per response from the response’s name in Python as in
  R, so two species no longer share their bootstrap draws.
- Python:
  [`feature_matrix()`](https://gillescolling.com/timesift/reference/feature_matrix.md)
  names unnamed rows from 1, as every other unit label does, and a fold
  map prints every fold level it holds.
- A response given as a data frame with two or more non-numeric columns
  is refused, naming them, rather than read through the first and
  silently stripped of the rest.
- A presence-absence response holding a missing value says so on the
  Python side, rather than reporting it as a value that is not 0 or 1.
- An arm is matched whole against the ladder’s labels rather than split
  at the `|`, so a learner or a sift whose own name holds one is still
  found, on both sides.
- [`select_grain()`](https://gillescolling.com/timesift/reference/select_grain.md)’s
  `inner` is checked before it is coerced, so a value that is not a
  count is named in the error rather than printed as `NA`.

### Packaging

- `inst/CITATION` cites the package, and `citation("timesift")` prints
  it; the methods article’s entry is added once it has a DOI.
  `codemeta.json` is generated from `DESCRIPTION` with codemetar and
  regenerated at a release. `DESCRIPTION` declares `Language: en-GB`,
  and `inst/WORDLIST` holds the proper nouns and API names the spell
  check would otherwise flag, so
  [`spelling::spell_check_package()`](https://docs.ropensci.org/spelling//reference/spell_check_package.html)
  runs clean.
- The committed site is held to its sources. `build_site.R` writes a
  digest over everything the site is rendered from into
  `docs/site-digest.txt`, and the `site` workflow recomputes it from the
  checkout on every push, so a push that changes the reference, the news
  or an article without rebuilding the site fails there rather than
  serving last week’s pages under this week’s sources. The site stays
  built locally, because the build post-processes its figures in a way a
  plain build in CI would not.
- The Python floor is 3.11. The `sklearn` extra pins the scikit-learn
  release where the mixing parameter alone selects the elastic net, and
  that release has no build for 3.10, so the 3.10 job had failed at
  install on every push. The `test` extra carries pandas, so the suite
  runs
  [`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
  on a data frame with a categorical identifier, a nullable-integer
  response and a zone-aware time column rather than skipping that test
  on every runner.
- `DESCRIPTION` is where the version is written, and `pyproject.toml`
  reads it from there.
- `.gitattributes` holds the repository to one line ending.
- The wheel ships `py.typed`.

## timesift 0.1.0

[`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
is the whole entry point. A table of targets, a table of time-stamped
series belonging to them, and one call builds every candidate
representation, fits the learners that can read each one, scores them
all on one set of held-out folds, and stacks the out-of-fold
predictions.

``` r

fit <- timesift(plots, logger, y = starts_with("sp_"), id = plot_id, time = datetime,
                models = c(elasticnet(), forest(), cnn()),
                sift = grains("day", "week", "month"))
summary(fit)
```

The version reads 0.1.0 because this is a first release: the package
fits any time-varying record against any prediction target, where its
predecessors fitted a climate record at a climate grain. Species
distribution modelling from microclimate loggers is the application it
ships defaults for, and the Schrankogel grid it was built on still
reproduces from `inst/reproduce/schrankogel.R`.

### The four concepts

- A **target** is one row to predict, a **series** is the long record
  belonging to those rows, a **representation** is how that record
  becomes an array, and a **learner** is a fit and a predict pair. A
  candidate is one (representation, learner) pair, and every candidate
  emits an out-of-fold prediction for every scorable cell over the same
  folds. Comparison, ensembling and importance read only those
  predictions.
- `y`, `x` and `static` take tidyselect expressions over their own
  table, and
  [`starts_with()`](https://tidyselect.r-lib.org/reference/starts_with.html),
  [`ends_with()`](https://tidyselect.r-lib.org/reference/starts_with.html),
  [`contains()`](https://tidyselect.r-lib.org/reference/starts_with.html),
  [`matches()`](https://tidyselect.r-lib.org/reference/starts_with.html),
  [`all_of()`](https://tidyselect.r-lib.org/reference/all_of.html),
  [`any_of()`](https://tidyselect.r-lib.org/reference/all_of.html),
  [`everything()`](https://tidyselect.r-lib.org/reference/everything.html)
  and [`where()`](https://tidyselect.r-lib.org/reference/where.html) are
  re-exported rather than redefined. A column of `targets` that is
  neither the response nor the identifier nor the anchor is a predictor
  only where `static` names it.
- [`predict()`](https://rdrr.io/r/stats/predict.html) on a fit rebuilds
  each member’s representation for the new targets from the settings its
  own arm was built with, and combines them through the ensemble.

### Representations

- [`native()`](https://gillescolling.com/timesift/reference/native.md),
  [`grain()`](https://gillescolling.com/timesift/reference/native.md),
  [`multigrain()`](https://gillescolling.com/timesift/reference/native.md)
  and
  [`lookback()`](https://gillescolling.com/timesift/reference/native.md)
  are what a representation is before any record has been read;
  [`grains()`](https://gillescolling.com/timesift/reference/grains.md)
  and
  [`lookbacks()`](https://gillescolling.com/timesift/reference/grains.md)
  are sets of them, and `grains("auto")` reads off the record every
  named grain it gives at least two bins.
- [`c()`](https://rdrr.io/r/base/c.html) combines learners, or
  representations, into a set: `c(elasticnet(), forest())` and
  `c(grains("day", "week"), lookback("30 days"))`. A set handed back to
  [`c()`](https://rdrr.io/r/base/c.html) splices, so a set can be added
  to rather than rewritten. `models` and `sift` take that, a bare spec,
  or a list.
- [`lookback()`](https://gillescolling.com/timesift/reference/native.md)
  is a fixed span of record ending a fixed lag before each target’s own
  instant, which is what a unit carrying several targets through time
  needs. It is one entry point in `src/` beside the calendar reduction,
  so both languages read it from the same implementation, and
  [`lookback_matrix()`](https://gillescolling.com/timesift/reference/lookback_matrix.md)
  exposes it directly.
- [`build_representation()`](https://gillescolling.com/timesift/reference/build_representation.md)
  is the one place the fitting layer turns a representation and the two
  tables into an array, so a run and a prediction on new targets reach a
  record the same way.
- A learner given `data = grain("week")` runs at that representation
  alone; left open it runs across the whole sift. A learner that reads a
  block of features and one that reads a sequence say so, and a pairing
  neither can carry is reported by name rather than fitted.

### Learners and training

- [`elasticnet()`](https://gillescolling.com/timesift/reference/elasticnet.md),
  [`stepwise()`](https://gillescolling.com/timesift/reference/stepwise.md),
  [`mlp()`](https://gillescolling.com/timesift/reference/torch_learners.md),
  [`cnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
  and
  [`rescnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
  drop the `_learner` suffix and gain `data`.
  [`forest()`](https://gillescolling.com/timesift/reference/forest.md)
  joins them, a probability forest on `ranger` in R and on scikit-learn
  in Python. `reads` and `multi` are
  [`learner()`](https://gillescolling.com/timesift/reference/learner.md)’s,
  where a learner of your own declares what it reads and whether one
  fitted model covers every response.
- [`train_control()`](https://gillescolling.com/timesift/reference/train_control.md)
  is the one place a training setting is defaulted. The architecture
  constructors carry architecture, a run gives one control to every
  neural learner, and a learner given its own control overrides that on
  the settings it names.
- A learner declares whether one fitted model covers every response or
  one is fitted per response. Either way a candidate emits one
  `[target, response]` matrix, so nothing above the learner layer has to
  know which it was.

### Resampling and the ensemble

- [`cv()`](https://gillescolling.com/timesift/reference/cv.md) and
  [`grouped_cv()`](https://gillescolling.com/timesift/reference/cv.md);
  `resampling` also takes a fold vector or a
  [`fold_map()`](https://gillescolling.com/timesift/reference/fold_map.md)
  result, which is how a split the package has no constructor for
  reaches the same fitting path.
- [`ensemble()`](https://gillescolling.com/timesift/reference/ensemble.md)
  fits non-negative weights summing to one on the out-of-fold
  predictions alone, minimising the response head’s own loss over the
  scorable cells, solved by an exponentiated- gradient loop in the
  package. `"mean"`, `"median"` and `"weighted"` combine without
  fitting.
  [`ensemble_fit()`](https://gillescolling.com/timesift/reference/ensemble_fit.md)
  is handed the predictions, the response, the mask and the fold map,
  and never a model.
- [`summary()`](https://rdrr.io/r/base/summary.html) reports each
  candidate’s mean, how many responses it scored highest on, whether one
  model covered them, and the level the combination reached under the
  weights it reached it with.

### Names

- [`native()`](https://gillescolling.com/timesift/reference/native.md)
  rather than [`raw()`](https://rdrr.io/r/base/raw.html) and
  [`lookback()`](https://gillescolling.com/timesift/reference/native.md)
  rather than [`window()`](https://rdrr.io/r/stats/window.html), which
  would have masked [`base::raw()`](https://rdrr.io/r/base/raw.html) and
  [`stats::window()`](https://rdrr.io/r/stats/window.html).
  [`occlusion()`](https://gillescolling.com/timesift/reference/occlusion.md)
  is one generic over a run and a ladder, and `bin_occlusion()` is gone.
  `ensemble_learner()` is gone with it: it fitted its members and
  averaged them, which the stack does over any candidates at all and
  with the weights fitted rather than assumed.
- The bin’s name is the grain throughout, in both languages.
