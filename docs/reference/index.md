# Package index

## The one call

Two tables to a scored comparison of representations, and the prediction
that follows from it.

- [`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
  : Fit and compare representations of time-varying data
- [`c(`*`<timesift_learner>`*`)`](https://gillescolling.com/timesift/reference/combine.md)
  [`c(`*`<timesift_models>`*`)`](https://gillescolling.com/timesift/reference/combine.md)
  [`c(`*`<timesift_representation>`*`)`](https://gillescolling.com/timesift/reference/combine.md)
  [`c(`*`<timesift_sift>`*`)`](https://gillescolling.com/timesift/reference/combine.md)
  : Combine learners, or representations, into a set
- [`summary(`*`<timesift>`*`)`](https://gillescolling.com/timesift/reference/timesift_report.md)
  [`print(`*`<timesift>`*`)`](https://gillescolling.com/timesift/reference/timesift_report.md)
  [`print(`*`<timesift_summary>`*`)`](https://gillescolling.com/timesift/reference/timesift_report.md)
  : What a run found
- [`predict(`*`<timesift>`*`)`](https://gillescolling.com/timesift/reference/predict.timesift.md)
  : Predict from a fitted timesift
- [`plot(`*`<timesift>`*`)`](https://gillescolling.com/timesift/reference/plot.timesift.md)
  : Draw a run

## Representations

How a series becomes the array a learner reads, and the set a run is
compared across.

- [`native()`](https://gillescolling.com/timesift/reference/native.md)
  [`grain()`](https://gillescolling.com/timesift/reference/native.md)
  [`multigrain()`](https://gillescolling.com/timesift/reference/native.md)
  [`lookback()`](https://gillescolling.com/timesift/reference/native.md)
  : How a series becomes an array a learner reads
- [`grains()`](https://gillescolling.com/timesift/reference/grains.md)
  [`lookbacks()`](https://gillescolling.com/timesift/reference/grains.md)
  [`timesift_sift()`](https://gillescolling.com/timesift/reference/grains.md)
  : Several representations to run the same learners across
- [`build_representation()`](https://gillescolling.com/timesift/reference/build_representation.md)
  : Build one representation for a set of targets

## Learners

The arms that ship, how they are trained, and the interface a learner of
your own goes through.

- [`elasticnet()`](https://gillescolling.com/timesift/reference/elasticnet.md)
  : Penalised regression on the flattened representation
- [`stepwise()`](https://gillescolling.com/timesift/reference/stepwise.md)
  : Forward selection by AIC on the flattened representation
- [`forest()`](https://gillescolling.com/timesift/reference/forest.md) :
  Random forest on the flattened representation
- [`mlp()`](https://gillescolling.com/timesift/reference/torch_learners.md)
  [`cnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
  [`rescnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
  : Sequence encoders with a joint multi-label head
- [`train_control()`](https://gillescolling.com/timesift/reference/train_control.md)
  : Training settings every neural learner reads
- [`learner()`](https://gillescolling.com/timesift/reference/learner.md)
  : Define a learner
- [`register_learner()`](https://gillescolling.com/timesift/reference/register_learner.md)
  [`learners()`](https://gillescolling.com/timesift/reference/register_learner.md)
  : Register a learner

## The split and the cells

One fold map read by everything that scores, and the cells a score is
defined on, computed with no model involved.

- [`cv()`](https://gillescolling.com/timesift/reference/cv.md)
  [`grouped_cv()`](https://gillescolling.com/timesift/reference/cv.md) :
  How the folds are drawn
- [`fold_map()`](https://gillescolling.com/timesift/reference/fold_map.md)
  : Assign units to cross-validation folds
- [`scorable_cells()`](https://gillescolling.com/timesift/reference/scorable_cells.md)
  : Which cells a score is defined on

## Combining the candidates

Weights fitted on the out-of-fold predictions alone.

- [`ensemble()`](https://gillescolling.com/timesift/reference/ensemble.md)
  : How the candidates are combined
- [`ensemble_fit()`](https://gillescolling.com/timesift/reference/ensemble_fit.md)
  : Fit the combiner on the out-of-fold predictions
- [`ensemble_combine()`](https://gillescolling.com/timesift/reference/ensemble_combine.md)
  : Combine one prediction per member into one prediction
- [`ensemble_weights()`](https://gillescolling.com/timesift/reference/ensemble_weights.md)
  : The weights the combiner fitted

## Scoring and comparison

- [`tss()`](https://gillescolling.com/timesift/reference/tss.md) : The
  true skill statistic
- [`roc_auc()`](https://gillescolling.com/timesift/reference/roc_auc.md)
  : The area under the ROC curve
- [`kappa_score()`](https://gillescolling.com/timesift/reference/kappa_score.md)
  [`decision_threshold()`](https://gillescolling.com/timesift/reference/kappa_score.md)
  [`model_agreement()`](https://gillescolling.com/timesift/reference/kappa_score.md)
  : Cohen's kappa, and where two models disagree
- [`score_predictions()`](https://gillescolling.com/timesift/reference/score_predictions.md)
  : Score held-out predictions on the cells the mask allows
- [`paired_contrast()`](https://gillescolling.com/timesift/reference/paired_contrast.md)
  : Compare two arms cell by cell
- [`grain_contrasts()`](https://gillescolling.com/timesift/reference/grain_contrasts.md)
  : Compare every grain against a learner's best one
- [`tss_inflation()`](https://gillescolling.com/timesift/reference/tss_inflation.md)
  : How much a self-selected threshold inflates the reported level
- [`implied_skill()`](https://gillescolling.com/timesift/reference/implied_skill.md)
  : What population skill a reported level is consistent with
- [`occlusion()`](https://gillescolling.com/timesift/reference/occlusion.md)
  : What part of the record a fitted model reads

## The arrays themselves

Readings in long form to a `[unit, bin, channel]` array, reachable
without the fitting layer.

- [`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md)
  : Reduce sensor series to a temporal grain
- [`lookback_matrix()`](https://gillescolling.com/timesift/reference/lookback_matrix.md)
  : Reduce sensor series to a lookback anchored on each target
- [`coverage()`](https://gillescolling.com/timesift/reference/coverage.md)
  : Which units reach which bins
- [`timesift_set()`](https://gillescolling.com/timesift/reference/timesift_set.md)
  : Several built representations of the same targets
- [`calendar_channels()`](https://gillescolling.com/timesift/reference/calendar_channels.md)
  : Where in the year each bin sits
- [`bind_channels()`](https://gillescolling.com/timesift/reference/bind_channels.md)
  : Put channels side by side
- [`feature_matrix()`](https://gillescolling.com/timesift/reference/feature_matrix.md)
  : Bring an already-reduced feature table into a ladder

## One grain at a time

Fitting across a set of grains on its own split, and reading the grain a
ladder saturates at.

- [`grain_ladder()`](https://gillescolling.com/timesift/reference/grain_ladder.md)
  [`summary(`*`<timesift_ladder>`*`)`](https://gillescolling.com/timesift/reference/grain_ladder.md)
  : Fit at every grain and see where skill saturates
- [`fit_learner()`](https://gillescolling.com/timesift/reference/fit_learner.md)
  [`predict(`*`<timesift_fit>`*`)`](https://gillescolling.com/timesift/reference/fit_learner.md)
  : Fit one learner at one grain
- [`select_grain()`](https://gillescolling.com/timesift/reference/select_grain.md)
  [`summary(`*`<timesift_selection>`*`)`](https://gillescolling.com/timesift/reference/select_grain.md)
  : Choose the grain inside the training data, and score the whole
  procedure
- [`plot(`*`<timesift_ladder>`*`)`](https://gillescolling.com/timesift/reference/plot.timesift_ladder.md)
  : Draw a ladder
- [`plot(`*`<timesift_selection>`*`)`](https://gillescolling.com/timesift/reference/plot.timesift_selection.md)
  : Draw how stable the choice of grain was

## Extending

The response head and the metric are registrations, never a fork of the
fitting code.

- [`register_response()`](https://gillescolling.com/timesift/reference/register_response.md)
  [`responses()`](https://gillescolling.com/timesift/reference/register_response.md)
  : Register a response head
- [`register_metric()`](https://gillescolling.com/timesift/reference/register_metric.md)
  [`metrics()`](https://gillescolling.com/timesift/reference/register_metric.md)
  : Register a metric

## What crosses the boundary

The three artifacts a split is carried in, and the digest that says two
arrays are the same array.

- [`write_folds()`](https://gillescolling.com/timesift/reference/artifacts.md)
  [`read_folds()`](https://gillescolling.com/timesift/reference/artifacts.md)
  [`write_response()`](https://gillescolling.com/timesift/reference/artifacts.md)
  [`read_response()`](https://gillescolling.com/timesift/reference/artifacts.md)
  [`write_cells()`](https://gillescolling.com/timesift/reference/artifacts.md)
  [`read_cells()`](https://gillescolling.com/timesift/reference/artifacts.md)
  : Read and write the artifacts that cross the language boundary
- [`digest_array()`](https://gillescolling.com/timesift/reference/digest_array.md)
  : The cross-language digest of a representation

## A record to test on

- [`simulate_records()`](https://gillescolling.com/timesift/reference/simulate_records.md)
  : Simulate sensor records whose response acts at a known temporal
  grain

## The package

- [`timesift-package`](https://gillescolling.com/timesift/reference/timesift-package.md)
  : timesift: Temporal Climate Resolution for Ecological Prediction
