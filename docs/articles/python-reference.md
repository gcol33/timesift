# Python reference

The Python package’s functions, classes and values, under the sections
of [the R
reference](https://gillescolling.com/timesift/reference/index.md) and in
its order. [Get
started](https://gillescolling.com/timesift/articles/python.md) runs
them on a simulated record.

## [The one call](https://gillescolling.com/timesift/articles/python-one-call.md)

Two tables to a scored comparison of representations, and the prediction
that follows from it.

- [`timesift()`](https://gillescolling.com/timesift/articles/python-one-call.html#timesift):
  Compare every learner across every representation, and estimate
  choosing among them.

- [`Timesift`](https://gillescolling.com/timesift/articles/python-one-call.html#Timesift):
  A fitted sift: every candidate’s out-of-fold predictions, their
  scores, and the combiner.

- [`TimesiftSpec`](https://gillescolling.com/timesift/articles/python-one-call.html#TimesiftSpec):
  How a fit was asked for: the columns each table plays, and the
  calendar they are read in.

- [`summary()`](https://gillescolling.com/timesift/articles/python-one-call.html#summary):
  The fit as text: the candidates on the outer folds, the procedure, and
  the weights.

- [`candidate_table()`](https://gillescolling.com/timesift/articles/python-one-call.html#candidate_table):
  One row per candidate: its level, how many responses it was best on,
  and how it covered them.

- [`procedure_table()`](https://gillescolling.com/timesift/articles/python-one-call.html#procedure_table):
  The selected candidate’s and the stack’s held-out level under the
  run’s own metric, with the standard error across responses: one row
  each, or none where the run made no estimate.

- [`plot()`](https://gillescolling.com/timesift/articles/python-one-call.html#plot):
  Draw a ladder, a run or a selection, and return the table the drawing
  was made from.

- [`project()`](https://gillescolling.com/timesift/articles/python-one-call.html#project):
  Predict one target per cell of a grid, each carrying the record of its
  own cell.

- [`range_change()`](https://gillescolling.com/timesift/articles/python-one-call.html#range_change):
  How the range of each response changes between two maps.

- [`RangeChange`](https://gillescolling.com/timesift/articles/python-one-call.html#RangeChange):

  The change in range of each response: a `table` of counts, and the
  `map` of codes (-2 lost, -1 kept, 0 absent in both, 1 gained) of the
  shape of the inputs.

- [`pseudo_absences()`](https://gillescolling.com/timesift/articles/python-one-call.html#pseudo_absences):
  Draw pseudo-absences from a pool of background units.

- [`select_columns()`](https://gillescolling.com/timesift/articles/python-one-call.html#select_columns):
  The columns a selection names, in the order the selection names them.

- [`column_names()`](https://gillescolling.com/timesift/articles/python-one-call.html#column_names):
  The column names of a table, whether it is a mapping of arrays or a
  data frame.

- [`n_targets()`](https://gillescolling.com/timesift/articles/python-one-call.html#n_targets):
  How many rows of targets there are, read off a column the spec is sure
  of.

- [`target_labels()`](https://gillescolling.com/timesift/articles/python-one-call.html#target_labels):
  What names the rows of every array in one fit.

## [Representations](https://gillescolling.com/timesift/articles/python-representations.md)

How a series becomes the array a learner reads, and the set a run is
compared across.

- [`native()`](https://gillescolling.com/timesift/articles/python-representations.html#native):
  The record unreduced: one bin per reading.

- [`grain()`](https://gillescolling.com/timesift/articles/python-representations.html#grain):

  One calendar grain, named, or supplied as a function of the reading
  instants returning each reading’s bin start, which is reported as
  `custom`.

- [`multigrain()`](https://gillescolling.com/timesift/articles/python-representations.html#multigrain):
  Several grains flattened and bound side by side into one block of
  features.

- [`lookback()`](https://gillescolling.com/timesift/articles/python-representations.html#lookback):
  A stretch of record of fixed length, ending a fixed lag before each
  target’s own instant.

- [`grains()`](https://gillescolling.com/timesift/articles/python-representations.html#grains):

  A sift over calendar grains, named or read off the record with
  `"auto"`.

- [`lookbacks()`](https://gillescolling.com/timesift/articles/python-representations.html#lookbacks):
  A sift over lookbacks of several lengths, all sharing a lag and a
  number of bins.

- [`Representation`](https://gillescolling.com/timesift/articles/python-representations.html#Representation):
  One reduction, named but not yet built.

- [`Sift`](https://gillescolling.com/timesift/articles/python-representations.html#Sift):
  The representations a set of candidates runs across, as a mapping of
  label to spec.

- [`as_sift()`](https://gillescolling.com/timesift/articles/python-representations.html#as_sift):
  A sift, whether it arrived as one, as a representation, as a grain
  name, or as a list.

- [`expand_sift()`](https://gillescolling.com/timesift/articles/python-representations.html#expand_sift):

  The sift with `"auto"` replaced by the grains the record supports.

- [`auto_grains()`](https://gillescolling.com/timesift/articles/python-representations.html#auto_grains):
  The named grains that give the record at least two bins, from the
  finest to the coarsest.

- [`build_representation()`](https://gillescolling.com/timesift/articles/python-representations.html#build_representation):
  The array one representation names, for these targets, in their own
  row order.

## [Learners](https://gillescolling.com/timesift/articles/python-learners.md)

The arms that ship, how they are trained, and the interface a learner of
your own goes through.

- [`elasticnet()`](https://gillescolling.com/timesift/articles/python-learners.html#elasticnet):
  One penalised regression per variable, over every bin-by-channel
  column and, by default, their squares, with the penalty chosen by an
  inner cross-validation on the fitting units.

- [`linear()`](https://gillescolling.com/timesift/articles/python-learners.html#linear):
  One generalised linear model per variable over every bin-by-channel
  column, its terms chosen by Akaike’s criterion. The family is the
  response head’s: logistic under a binary cross-entropy loss, Gaussian
  under a squared-error one and Poisson under a Poisson-deviance one,
  and so are the case weights.

- [`forest()`](https://gillescolling.com/timesift/articles/python-learners.html#forest):
  One random forest per variable, over every bin-by-channel column: a
  probability forest under a presence-absence head and a regression
  forest under a head with a squared-error loss or a count head, a count
  being cut on its variance and a leaf reporting its mean. Trees split
  on one column at a time and pay nothing for columns that carry
  nothing, so a forest reads a wide tabular representation without a
  penalty path and without a selection step.

- [`tree()`](https://gillescolling.com/timesift/articles/python-learners.html#tree):

  One classification or regression tree per variable, over every
  bin-by-channel column, grown under rpart’s rules: the Gini index under
  a presence-absence head, the sum of squares under a head with a
  squared-error loss and the Poisson deviance under a count head, a
  split only between two distinct values of a column, and the
  cost-complexity bookkeeping that keeps a split only where it lowers
  the risk by at least `cp` of the root’s. On the same columns, weights
  and folds the tree is the one rpart grows, split for split, and its
  complexity table the one rpart reports; the tree is grown by the core
  the R package calls, so the two languages grow it identically.

- [`boosting()`](https://gillescolling.com/timesift/articles/python-learners.html#boosting):

  One boosted model per variable, over every bin-by-channel column: a
  logistic model under a presence-absence head, a squared-error one
  under a head with a squared-error loss and a Poisson one with a log
  link under a count head. The score starts at the log-odds of the
  weighted share of presences, or the weighted mean, and each tree is
  fitted to the loss’s gradient at the current score and added to it
  scaled by `shrinkage`. Each tree is grown on a subsample of the units
  drawn without replacement, and reads a subsample of the columns.

- [`maxent()`](https://gillescolling.com/timesift/articles/python-learners.html#maxent):

  One maximum-entropy model per variable, over every bin-by-channel
  column, in the formulation of the maxnet package (Phillips et
  al. 2017) and biomod2’s `MAXNET`: maxnet’s feature classes, its
  regularisation of each feature, and a lasso over them, fitted by the
  penalised core `elasticnet` runs on, which the R package calls too.
  With the maxnet package’s own settings the features and the penalty
  factors are maxnet’s to rounding, and the fit settles at the objective
  glmnet reaches for maxnet.

- [`envelope()`](https://gillescolling.com/timesift/articles/python-learners.html#envelope):

  biomod2’s surface range envelope, one per variable, over every
  bin-by-channel column: for each column the `quantile` and
  `1 - quantile` quantiles of its readings over the units present, and a
  unit predicted present where every column lies between its two, the
  ends included. With the same quantile it draws the envelope `bm_SRE()`
  draws; the quantile is R’s default, type 7.

- [`mars()`](https://gillescolling.com/timesift/articles/python-learners.html#mars):

  One MARS model per variable over every bin-by-channel column, fitted
  as the earth package fits it and as biomod2 fits `MARS`. The forward
  pass starts from the intercept and at each step multiplies a term
  already in the model by a pair of hinges on one column,
  `max(0, x - t)` and `max(0, t - x)`, taking the parent, the column and
  the knot `t` that most reduce the residual sum of squares of a
  least-squares fit to the response; a knot at a column’s least value
  enters the column linearly. `degree` bounds how many hinges a term
  multiplies. The pass stops at `nk` terms, when a step raises the
  R-squared by less than `thresh`, or when no term reduces the
  residuals.

- [`discriminant()`](https://gillescolling.com/timesift/articles/python-learners.html#discriminant):

  One flexible discriminant per variable over every bin-by-channel
  column, fitted as mda’s `fda(method = mars)` fits it and as biomod2
  fits `FDA`. Optimal scoring gives presence and absence one score each
  and regresses the scored response on a MARS basis of the columns; the
  fitted score is the one canonical variate, and the prediction is the
  posterior probability of presence under two normal classes around the
  class centroids on it, the classes’ shares among the fitting units as
  priors.

- [`additive()`](https://gillescolling.com/timesift/articles/python-learners.html#additive):

  One additive model per variable, a smooth function of every
  bin-by-channel column, fitted as mgcv’s
  `gam(y ~ s(x1) + s(x2) + ..., method = "GCV.Cp")` fits it and as
  biomod2 fits `GAM`. Each column enters as a thin plate regression
  spline of `k` basis functions (Wood 2003): a cubic radial function
  centred on each of the column’s distinct values, reduced to its
  `k - 2` directions of greatest eigenvalue, together with the linear
  function, which the penalty on the spline’s squared second derivative
  leaves free. Each smooth sums to zero over the units, beside one
  intercept.

- [`hierarchical()`](https://gillescolling.com/timesift/articles/python-learners.html#hierarchical):
  One Bayesian logistic model per response: a logistic regression on
  every bin-by-channel column of the representation, standardised so the
  prior on a coefficient means the same thing for each, optionally with
  an intercept for each unit and a Gaussian-process field over the
  targets’ coordinates. A unit then carries what its own record says and
  what its neighbours’ presences say, under the folds every other
  learner is scored on.

- [`mlp()`](https://gillescolling.com/timesift/articles/python-learners.html#mlp):
  Flattens the channels and builds in no temporal geometry.

- [`cnn()`](https://gillescolling.com/timesift/articles/python-learners.html#cnn):
  Convolution, batch normalisation, activation and pooling, then global
  average pooling.

- [`rescnn()`](https://gillescolling.com/timesift/articles/python-learners.html#rescnn):
  Dilated residual blocks with channel gates, pooling average and
  maximum together.

- [`train_control()`](https://gillescolling.com/timesift/articles/python-learners.html#train_control):
  The settings every neural learner reads, with anything named here
  replacing its default.

- [`TrainControl`](https://gillescolling.com/timesift/articles/python-learners.html#TrainControl):
  How long to train, on what, and when to stop.

- [`Learner`](https://gillescolling.com/timesift/articles/python-learners.html#Learner):
  A name, a fit and a predict, what has to be installed for them to run,
  and what the learner reads.

- [`flatten()`](https://gillescolling.com/timesift/articles/python-learners.html#flatten):

  `[unit, bin, channel]` to `[unit, bin * channel]` in the array’s own
  order.

- [`register_learner()`](https://gillescolling.com/timesift/articles/python-learners.html#register_learner):
  Make a learner available by name. The learners that ship are
  registered the same way.

- [`get_learner()`](https://gillescolling.com/timesift/articles/python-learners.html#get_learner):

  A `Learner`, whether it arrived as one or as the name of a registered
  one.

- [`learners()`](https://gillescolling.com/timesift/articles/python-learners.html#learners):
  The learners registered under this session.

- [`tune()`](https://gillescolling.com/timesift/articles/python-learners.html#tune):

  A learner that searches `grid` on the units it is fitted on and fits
  the best setting.

- [`Tuned`](https://gillescolling.com/timesift/articles/python-learners.html#Tuned):
  What a tuned learner fitted: the model, the setting chosen, and every
  setting’s score.

- [`register_tuning()`](https://gillescolling.com/timesift/articles/python-learners.html#register_tuning):

  Register the grid a learner is tuned over when `tune` is given none.

- [`tunings()`](https://gillescolling.com/timesift/articles/python-learners.html#tunings):
  The learners a grid is registered for.

## [The split and the cells](https://gillescolling.com/timesift/articles/python-split.md)

One fold map read by everything that scores, and the cells a score is
defined on, computed with no model involved.

- [`cv()`](https://gillescolling.com/timesift/articles/python-split.html#cv):
  Hold out single targets, balanced within equal-count strata of the
  response.

- [`grouped_cv()`](https://gillescolling.com/timesift/articles/python-split.html#grouped_cv):

  Keep every target sharing a value of the `group` column in one fold.

- [`block_cv()`](https://gillescolling.com/timesift/articles/python-split.html#block_cv):

  Hold out a block of targets whole, the blocks cut on the columns `by`
  of `targets`.

- [`env_cv()`](https://gillescolling.com/timesift/articles/python-split.html#env_cv):

  As `block_cv`, on columns centred and scaled first: blocks of
  predictor space.

- [`Resampling`](https://gillescolling.com/timesift/articles/python-split.html#Resampling):
  How the targets are split, named but not yet drawn.

- [`as_resampling()`](https://gillescolling.com/timesift/articles/python-split.html#as_resampling):
  A resampling spec, or a fold map somebody else built, read as a spec
  that returns it.

- [`resolve_folds()`](https://gillescolling.com/timesift/articles/python-split.html#resolve_folds):
  Draw the fold map a resampling spec names, in the row order of the
  response.

- [`fold_map()`](https://gillescolling.com/timesift/articles/python-split.html#fold_map):
  Assign units to folds, balanced within equal-count strata of a
  stratifying value.

- [`scorable_cells()`](https://gillescolling.com/timesift/articles/python-split.html#scorable_cells):
  Which cells admit a score, from the response and the fold map alone.

- [`align_folds()`](https://gillescolling.com/timesift/articles/python-split.html#align_folds):

  A fold map reaches the fitting path as an integer vector in the row
  order of the representation, whether it arrived as a `Folds`, a
  mapping of unit to fold, or a bare vector already in that order.

- [`Folds`](https://gillescolling.com/timesift/articles/python-split.html#Folds):
  Which fold each unit is held out in, named by unit.

- [`Cells`](https://gillescolling.com/timesift/articles/python-split.html#Cells):

  Which `(variable, fold)` cells admit a score, and the counts that
  decided it.

## [Combining the candidates](https://gillescolling.com/timesift/articles/python-combining.md)

Weights fitted on the out-of-fold predictions alone.

- [`ensemble()`](https://gillescolling.com/timesift/articles/python-combining.html#ensemble):
  Ask for an ensemble of the candidates a fit produced.

- [`ensemble_fit()`](https://gillescolling.com/timesift/articles/python-combining.html#ensemble_fit):
  Fit the combiner on the out-of-fold predictions and nothing else.

- [`ensemble_combine()`](https://gillescolling.com/timesift/articles/python-combining.html#ensemble_combine):

  One `[n, response]` matrix from each member’s `[n, response]` matrix.

- [`ensemble_spread()`](https://gillescolling.com/timesift/articles/python-combining.html#ensemble_spread):

  How far the members of an ensemble disagree: biomod2’s `EMcv` and
  `EMci`.

- [`SPREAD_STATISTICS`](https://gillescolling.com/timesift/articles/python-combining.html#SPREAD_STATISTICS):
  A value.

- [`ensemble_weights()`](https://gillescolling.com/timesift/articles/python-combining.html#ensemble_weights):
  The weight the combiner gave each of its members, or nothing where a
  run combined none.

- [`EnsembleSpec`](https://gillescolling.com/timesift/articles/python-combining.html#EnsembleSpec):
  How the candidates are to be combined, which of them are eligible, and
  under which head.

- [`Stack`](https://gillescolling.com/timesift/articles/python-combining.html#Stack):
  A fitted combiner: what it does and what weight it gave each of its
  members.

## [Scoring and comparison](https://gillescolling.com/timesift/articles/python-scoring.md)

The metrics, the paired contrast between two arms on matched cells,
every grain against a learner’s best, the inflation of a score read at
its own best threshold, and what a fitted model read.

- [`tss()`](https://gillescolling.com/timesift/articles/python-scoring.html#tss):
  Sensitivity plus specificity minus one, at the threshold that
  maximises it.

- [`roc_auc()`](https://gillescolling.com/timesift/articles/python-scoring.html#roc_auc):
  The area under the ROC curve, as the rank sum of the presences. Ties
  take the average rank.

- [`average_precision()`](https://gillescolling.com/timesift/articles/python-scoring.html#average_precision):
  The area under the precision-recall curve, as the step sum over the
  distinct predictions.

- [`kappa_score()`](https://gillescolling.com/timesift/articles/python-scoring.html#kappa_score):
  Cohen’s kappa of a model’s decisions against the observed response.

- [`table_metric()`](https://gillescolling.com/timesift/articles/python-scoring.html#table_metric):
  A metric of the two-by-two table of decisions against observations.

- [`boyce_index()`](https://gillescolling.com/timesift/articles/python-scoring.html#boyce_index):
  The continuous Boyce index (Hirzel et al. 2006).

- [`regression_metric()`](https://gillescolling.com/timesift/articles/python-scoring.html#regression_metric):
  A metric of a numeric response and its predictions, as biomod2 reads
  an abundance model.

- [`ordinal_metric()`](https://gillescolling.com/timesift/articles/python-scoring.html#ordinal_metric):
  A metric of ordinal classes, as biomod2 reads a model of an ordinal
  response.

- [`cohen_kappa()`](https://gillescolling.com/timesift/articles/python-scoring.html#cohen_kappa):
  Chance-corrected agreement of two labellings of the same units, in
  either order.

- [`decision_threshold()`](https://gillescolling.com/timesift/articles/python-scoring.html#decision_threshold):

  The probability cut a rule selects. Presence is predicted at
  `p >= threshold`.

- [`model_agreement()`](https://gillescolling.com/timesift/articles/python-scoring.html#model_agreement):
  Agreement between two models’ decisions, with how often each is right
  where they differ.

- [`score_predictions()`](https://gillescolling.com/timesift/articles/python-scoring.html#score_predictions):
  Score held-out predictions on the cells the mask allows.

- [`paired_contrast()`](https://gillescolling.com/timesift/articles/python-scoring.html#paired_contrast):
  The difference between two arms, taken inside each cell both scored.

- [`grain_contrasts()`](https://gillescolling.com/timesift/articles/python-scoring.html#grain_contrasts):
  Compare every grain against the reference grain of one learner.

- [`tss_inflation()`](https://gillescolling.com/timesift/articles/python-scoring.html#tss_inflation):
  How much a threshold chosen on the scored units inflates the level it
  reports.

- [`implied_skill()`](https://gillescolling.com/timesift/articles/python-scoring.html#implied_skill):
  What population skill a level actually read is consistent with.

- [`occlusion()`](https://gillescolling.com/timesift/articles/python-scoring.html#occlusion):
  What one candidate’s score loses when a bin, or a channel, is withheld
  from it.

- [`response_curve()`](https://gillescolling.com/timesift/articles/python-scoring.html#response_curve):
  Vary one predictor of a fitted candidate across the range it takes,
  the rest held fixed.

- [`ResponseCurve`](https://gillescolling.com/timesift/articles/python-scoring.html#ResponseCurve):
  The prediction against the value of a predictor, or of two.

## [The arrays themselves](https://gillescolling.com/timesift/articles/python-arrays.md)

Readings in long form to a `[unit, bin, channel]` array, reachable
without the fitting layer.

- [`grain_matrix()`](https://gillescolling.com/timesift/articles/python-arrays.html#grain_matrix):
  Bin readings by the calendar and summarise every bin.

- [`lookback_matrix()`](https://gillescolling.com/timesift/articles/python-arrays.html#lookback_matrix):
  Read a fixed length of record ending a fixed lag before each target’s
  own instant.

- [`coverage()`](https://gillescolling.com/timesift/articles/python-arrays.html#coverage):
  Which units reach which bins.

- [`Coverage`](https://gillescolling.com/timesift/articles/python-arrays.html#Coverage):

  How many readings each unit has in each bin, over every bin the
  calendar tiles the record with. `count` is `[unit, bin]`; a unit that
  started late, stopped early or lost a month is a row with zeros in it,
  and a bin the whole record skips is a column of zeros.

- [`timesift_set()`](https://gillescolling.com/timesift/articles/python-arrays.html#timesift_set):
  Every entry point that fits across grains takes a representation, a
  set, or a bare mapping, and works on a set. One coercion, so no caller
  repeats the three cases.

- [`calendar_channels()`](https://gillescolling.com/timesift/articles/python-arrays.html#calendar_channels):
  Where in the year, or the day, each bin sits, as the sine and cosine
  of its fractional position in each cycle named, in the order given.

- [`bind_channels()`](https://gillescolling.com/timesift/articles/python-arrays.html#bind_channels):
  Put the channels of several representations of the same units and bins
  side by side.

- [`feature_matrix()`](https://gillescolling.com/timesift/articles/python-arrays.html#feature_matrix):
  Bring an already-reduced feature table into a ladder as a one-channel
  representation.

- [`TimesiftMatrix`](https://gillescolling.com/timesift/articles/python-arrays.html#TimesiftMatrix):

  A `[row, bin, channel]` representation and the reduction that produced
  it.

- [`TimesiftSet`](https://gillescolling.com/timesift/articles/python-arrays.html#TimesiftSet):
  A ladder of representations, one per grain.

- [`GRAINS`](https://gillescolling.com/timesift/articles/python-arrays.html#GRAINS):
  A value.

- [`STATS`](https://gillescolling.com/timesift/articles/python-arrays.html#STATS):
  A value.

- [`DAY_LEVEL_STATS`](https://gillescolling.com/timesift/articles/python-arrays.html#DAY_LEVEL_STATS):
  A value.

## [One grain at a time](https://gillescolling.com/timesift/articles/python-ladder.md)

Fitting across a set of grains on its own split, and reading the grain a
ladder saturates at.

- [`grain_ladder()`](https://gillescolling.com/timesift/articles/python-ladder.html#grain_ladder):
  Cross-validate every learner at every grain, on one fold map and one
  mask of cells.

- [`fit_learner()`](https://gillescolling.com/timesift/articles/python-ladder.html#fit_learner):
  Fit one learner at one grain, under one registered response head.

- [`Fit`](https://gillescolling.com/timesift/articles/python-ladder.html#Fit):
  A fitted learner, the variables it was fitted on, and the bins and
  channels of the representation it was made on.

- [`Ladder`](https://gillescolling.com/timesift/articles/python-ladder.html#Ladder):

  One score per `(grain, learner, variable, fold)` cell, and what
  produced it.

- [`select_grain()`](https://gillescolling.com/timesift/articles/python-ladder.html#select_grain):
  Choose the grain inside each outer fold’s training units, then score
  the whole procedure.

- [`Selection`](https://gillescolling.com/timesift/articles/python-ladder.html#Selection):
  What a nested selection chose, what it scores, and what it was
  searched over.

## [Extending](https://gillescolling.com/timesift/articles/python-extending.md)

The response head and the metric are registrations, never a fork of the
fitting code.

- [`register_response()`](https://gillescolling.com/timesift/articles/python-extending.html#register_response):
  Register a response head: what the values being predicted are and
  where a score is defined.

- [`responses()`](https://gillescolling.com/timesift/articles/python-extending.html#responses):
  The response heads registered under this session.

- [`as_response()`](https://gillescolling.com/timesift/articles/python-extending.html#as_response):

  The response reaches everything downstream as a `Response`, whether it
  arrived as one, as a mapping of variable name to values, or as a
  two-dimensional array with no names at all.

- [`Response`](https://gillescolling.com/timesift/articles/python-extending.html#Response):

  A `[unit, variable]` matrix of observed values, with the units named.

- [`PRESENCE_ABSENCE`](https://gillescolling.com/timesift/articles/python-extending.html#PRESENCE_ABSENCE):
  A value.

- [`CONTINUOUS`](https://gillescolling.com/timesift/articles/python-extending.html#CONTINUOUS):
  A value.

- [`ABUNDANCE`](https://gillescolling.com/timesift/articles/python-extending.html#ABUNDANCE):
  A value.

- [`ORDINAL`](https://gillescolling.com/timesift/articles/python-extending.html#ORDINAL):
  A value.

- [`COUNT`](https://gillescolling.com/timesift/articles/python-extending.html#COUNT):
  A value.

- [`positive_weights()`](https://gillescolling.com/timesift/articles/python-extending.html#positive_weights):
  Case weights that balance a rare response.

- [`register_metric()`](https://gillescolling.com/timesift/articles/python-extending.html#register_metric):

  Register a metric: a function of `(y, p)` on one held-out cell.

- [`metrics()`](https://gillescolling.com/timesift/articles/python-extending.html#metrics):
  The metrics registered under this session.

- [`resolve_metric()`](https://gillescolling.com/timesift/articles/python-extending.html#resolve_metric):
  The function that scores and the name a report prints, from either way
  a metric is given.

## [What crosses the boundary](https://gillescolling.com/timesift/articles/python-artifacts.md)

The three artifacts a split is carried in, and the digest that says two
arrays are the same array.

- [`write_folds()`](https://gillescolling.com/timesift/articles/python-artifacts.html#write_folds):

  Write a fold map as `id,fold`, ordered by unit.

- [`read_folds()`](https://gillescolling.com/timesift/articles/python-artifacts.html#read_folds):
  Read a fold map somebody else built, optionally aligned to a
  representation’s units.

- [`write_response()`](https://gillescolling.com/timesift/articles/python-artifacts.html#write_response):

  Write a response as `id` and one column per variable, ordered by unit.

- [`read_response()`](https://gillescolling.com/timesift/articles/python-artifacts.html#read_response):

  Read a response matrix. The columns after `id` are the variables, in
  the file’s order.

- [`write_cells()`](https://gillescolling.com/timesift/articles/python-artifacts.html#write_cells):
  Write a scorable mask, ordered by variable and then by fold.

- [`read_cells()`](https://gillescolling.com/timesift/articles/python-artifacts.html#read_cells):
  Read a scorable mask the other language computed.

- [`digest_array()`](https://gillescolling.com/timesift/articles/python-artifacts.html#digest_array):
  MD5 of the representation, byte-exactly as the spec defines it.

## [A record to test on](https://gillescolling.com/timesift/articles/python-simulate.md)

A simulated record with a grain planted in it, to test a run against a
known answer.

- [`simulate_records()`](https://gillescolling.com/timesift/articles/python-simulate.html#simulate_records):
  Draw units carrying a record and a presence-absence response acting at
  one known grain.
- [`Simulation`](https://gillescolling.com/timesift/articles/python-simulate.html#Simulation):
  A simulated record, its response, and everything the draw is
  reproducible from.
