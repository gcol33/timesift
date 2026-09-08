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

### Packaging

- `DESCRIPTION` is where the version is written, and `pyproject.toml`
  reads it from there.
- `.gitattributes` holds the repository to one line ending.

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
