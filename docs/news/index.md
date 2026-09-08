# Changelog

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
  drop the `_learner` suffix and gain `data`, `reads` and `multi`.
  [`forest()`](https://gillescolling.com/timesift/reference/forest.md)
  joins them, a probability forest on `ranger` in R and on scikit-learn
  in Python.
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
