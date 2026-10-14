# Story: timesift

## Claim
timesift treats the time scale at which a long sensor record is read as an argument: it builds the
record at each grain or lookback the user names, fits every learner on every representation it can
read, scores them all on the same held-out folds and cells, and chooses or stacks them inside the
training data.

## Audience
Ecologists fitting species distribution models from logger or other time-stamped records, in R or
Python. They know presence-absence modelling and usually a biomod2-style model set (GLM, GAM,
MARS, random forest, Maxent); they have reduced a record to monthly means or bioclimatic
summaries before, once, by hand. They do not need to know torch, cross-validation design or the
inflation of a self-selected threshold; the docs teach those.

## Relation
- to: none as a parent. The learners reproduce the numbers of the reference packages a species
  modeller already uses (glmnet, MASS, rpart, earth, mda, mgcv, nnet, maxnet, and biomod2's own
  envelope) and the model set of biomod2.
- phrase: "pinned in the test fixtures against the reference package a species modeller already
  uses" (learners); "each biomod2 model and its timesift learner" (biomod2)
- same: each learner's numbers, pinned against its reference package in `inst/spec/fixtures/`
- differs: timesift is not a wrapper; every core is its own code, written from the published method
- status: not a successor of anything; `climgrain` and `timegrain` were earlier names of this package
- home: vignettes/articles/biomod2.Rmd (the biomod2 mapping); vignettes/articles/learners.Rmd
  (the reference package of each learner)
- context: sidecars/context.md

## Arcs

### A1 The grain is a choice you can test
- question: "at what time scale should I summarise my logger record?"
- answer: a representation (native, grain, multigrain, lookback) is an argument; calendar bins are
  real weeks and months; extreme days and extreme readings are different statistics
- payoff: the reader names several grains and statistics and lets the run compare them
- evidence: inst/spec/fixtures digests; inst/reproduce/README.md (the Schrankogel grid)
- limits: day-level statistics need a grain of a day or coarser; a bin with a missing reading is
  refused, not imputed
- home: vignettes/articles/representations.Rmd
- also: README, DESCRIPTION, Get started, ?grain_matrix, python.Rmd

### A2 Evaluation scores
- question: "is the best grain's score an honest estimate?"
- answer: every candidate is scored on one fold map and one scorable-cell mask computed without a
  model; the chosen grain and the ensemble weights are chosen inside each outer training fold, so
  the `selected` and `ensemble` rows score the choosing too
- payoff: the reader reports `selected`/`ensemble`, compares grains cell by cell with
  `paired_contrast()` / `grain_contrasts()`
- evidence: tests of the nested selection; comparing-grains article on a planted grain
- limits: the interval is across responses fitted on the same plots and folds; few responses give a
  wide interval
- home: vignettes/articles/comparing-grains.Rmd
- also: README, Get started, python.Rmd

### A3 The learners a species modeller already uses
- question: "can I use the models I know, and will they give the numbers I know?"
- answer: elastic net, stepwise GLM, CART, forest, boosting, Maxent, envelope, MARS, FDA, GAM and a
  one-layer network, each pinned against its reference package, plus torch encoders and any
  user learner through one interface
- payoff: the reader picks learners by name, or registers their own
- evidence: inst/spec/fixtures (penalised, tree, stepwise, mars, fda, additive, perceptron cases)
- limits: neural encoders are not byte-identical across torch and libtorch; tree differs from rpart
  on an exact tie under fractional weights
- home: vignettes/articles/learners.Rmd
- also: README, Get started, biomod2.Rmd, DESCRIPTION

### A4 From a scored run to predictions and maps
- question: "how do I use the fit?"
- answer: `predict()` rebuilds each member's representation for new rows and predicts through the
  ensemble; thresholds come from out-of-fold predictions; `project()` and `range_change()` map it
- payoff: the reader produces a probability or binary map
- evidence: prediction article
- limits: new rows need a series at the grain the fit reads
- home: vignettes/articles/prediction.Rmd
- also: README, Get started

### A5 TSS read at its own best threshold overstates skill
- question: "how much can I trust a TSS on a rare species?"
- answer: TSS read at the threshold maximising it, on the cells it scores, is inflated where
  presences are thin; `tss_inflation()` estimates it for the reader's own presence counts and
  `implied_skill()` inverts it; runs are scored by AUC by default
- payoff: the reader reports the expected inflation beside a TSS, or uses AUC
- evidence: the Schrankogel simulation (inst/reproduce/README.md); ?tss_inflation
- limits: the inflation depends on the distribution of a model's predictions, so a paired TSS
  difference is not free of it
- home: vignettes/articles/comparing-grains.Rmd, "Scores, thresholds and their inflation"
- also: README, Get started, python.Rmd, ?tss_inflation

### A6 R and Python give the same numbers
- question: "if my colleague runs the Python version, do we get the same answer?"
- answer: one C++ core compiled into both; both suites assert the same digests of every
  representation and the same penalised fits; neural models are not required to match
- payoff: the reader can split work across languages; folds cross through `read_folds()`
- evidence: inst/spec/fixtures; the contract job in CI
- limits: fold builders draw on each language's own random stream; tidyselect verbs are R-only
- home: vignettes/articles/contract.Rmd (inst/spec/representation.md)
- also: README, python/README.md, python.Rmd

## Non-goals
- No specialisation to climate: any time-stamped numeric record.
- No imputation of missing readings.
- No data download or species-name handling.
- No byte-identical neural networks across the two languages.

## Documents
- DESCRIPTION: listing text; claim, A1, A2, A3 (penalised baseline), A5 (default metric)
- python/README.md (PyPI): entry point; claim, A6, first Python example, routing
- README / landing: entry point; claim, first example, one line per arc, routing
- ?timesift-package: entry point inside R; claim, routing
- Get started (timesift.Rmd): tutorial; one path from two tables to a scored run and predictions
- python.Rmd: tutorial, the same path in Python
- representations.Rmd: explanation and how-to, A1 home
- learners.Rmd: how-to, A3 home
- comparing-grains.Rmd: how-to, A2 home
- prediction.Rmd: how-to, A4 home
- biomod2.Rmd: comparison, A3 mapping
- troubleshooting.Rmd: how-to, the errors a run raises
- contract.Rmd: explanation, A6 home
- python-*.Rmd reference pages: reference, no journey

## Open
- A5's home sits inside "Finding the grain", where the article menu does not show it: keep it
  there, or give it an article of its own.
- Occlusion, response curves, pseudo-absences and the spatial and environmental folds carry no
  arc: an interpretation arc, or lines under A2 and A4.
