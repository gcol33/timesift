# Context: timesift

The relation to biomod2 and the reference packages. Every Standing figure
was read on 2026-10-14 from the source named in the line; anything not pulled is left blank.

## Standing

### biomod2
- released: first CRAN release in the archive: `biomod2_1.3.1.tar.gz`, 2012-11-09
  (https://cran.r-project.org/src/contrib/Archive/biomod2/, retrieved 2026-10-14). Current CRAN
  version "4.3-4-7", published "2026-10-08" (https://cran.r-project.org/package=biomod2, retrieved
  2026-10-14). The pkgdown site (https://biomodhub.github.io/biomod2/) shows 4.3-5-0, a development
  version ahead of CRAN.
- review: 
- use: reverse imports on CRAN "ecospat, gecko, RRgeo"; reverse suggests "blockCV, caretSDM"
  (CRAN page, retrieved 2026-10-14). Citations of the method papers: 
  Downloads: 
- maintainers: "Maya Guéguen [aut, cre], Hélène Blancheteau [aut], Rémi Lemaire-Patin [aut],
  Wilfried Thuiller [aut]" (first authors listed on the CRAN page, retrieved 2026-10-14).

### reference packages (rpart, randomForest, gbm, xgboost, maxnet, MASS, earth, mda, mgcv, nnet, cito, glmnet)
- released / use / maintainers: not pulled this session. cito: pkgdown site shows 1.1.1
  (https://citoverse.github.io/cito/reference/dnn.html, retrieved 2026-10-14).

### timesift
- released: first commit `ba264dc`, 2026-09-02 (as `timegrain`); version 0.6.0 on master, submitted
  to CRAN 2026-10-06, CRAN carries 0.3.1; PyPI 0.5.0 since 2026-10-03 (CLAUDE.md status, not
  re-checked against CRAN/PyPI this session).
- review: none yet. Paper targeted at Methods in Ecology and Evolution (CLAUDE.md).
- use: the Schrankogel study (`gcol33/schrankogel-cnn-2026`, private).
- validation: fixtures in `inst/spec/fixtures/` pin each core against its reference package's
  output (glmnet, rpart, MASS, earth, mda, mgcv, nnet, maxnet; forest and boosting against
  randomForest, gbm, xgboost per the spec section headings, not read in full); the Schrankogel grid
  reproduced from the deposit (inst/reproduce/README.md).

### burden
- timesift shows that each learner gives its reference package's numbers on the fixture cases, and
  states where it departs from a biomod2 run (loss of `ANN`, cut per statistic, DNN batch and
  burn-in). biomod2 is never asked to justify its choices.

## Debt
- What timesift takes from biomod2: the model set and its names (`CTA` ... `DNN`), the two option
  sets (default and `"bigboss"`) and their values, the ensemble algorithms (`EMmean`, `EMwmean` and
  its decay, `EMca`, `EMcv`/`EMci`) and the selection threshold, the pseudo-absence strategies
  (`random`, `sre`, `disk`), the cross-validation strategies (`kfold`, `strat`, `block`, `env`,
  `user.defined`), the evaluation statistic set, the probit recalibration of FDA, the SRE quantile,
  and the importance definition of `bm_VariablesImportance()`.
- What timesift takes from the reference packages: the numbers each core must give, fixed in the
  fixtures; never their code (CLAUDE.md: written from the published method).
- credit: B:11-21 names biomod2 and the packages but gives no citation of biomod2 (Thuiller et al.)
  and no sentence saying that the model set, presets, ensemble, CV and pseudo-absence vocabulary
  are biomod2's. The debt is visible only row by row in the tables. Proposed: one sentence after
  B:15, e.g. "The model set, its two option sets, the ensemble algorithms and the pseudo-absence and
  cross-validation strategies used below are biomod2's (Thuiller et al. <year>), and each is
  reproduced under biomod2's name." (citation to be pulled; not filled from memory).
  Finding: thin-credit.

## Use case
- A biomod2 user whose predictors come from a sensor record has reduced that record once, by hand.
  timesift fits the same model set at each grain on one fold map and one set of scorable cells, so
  the reduction is compared rather than assumed.

### novelty claims
- claim: "fits the same algorithms to the record summarised at every temporal grain it supports"
  (B:14-15)
  - parent-does: biomod2 takes one table of predictors (`?BIOMOD_FormatingData`, 4.3-5-0); no
    representation layer. Survives.
  - precise-form: as written.
- claim: "biomod2 fits one species per call" / all responses at once (B:226-228)
  - parent-does: `?BIOMOD_FormatingData` 4.3-5-0: "biomod2 models single species at a time (no
    multi-species)". Survives.
- claim: "No weight is therefore fitted on a prediction a member made for a unit it was trained on"
  (stack, B:244-246); `ensemble("stack")` has no biomod2 row (B:242)
  - parent-does: biomod2 `EMwmean` weights by evaluation scores; whether those are calibration or
    validation scores was not checked this session. Survives as a statement about timesift; the
    "none" row is correct for a stack.
- claim: "biomod2, like most species distribution code, reports TSS at a threshold chosen on the
  same predictions it scores" (B:286-287)
  - parent-does: biomod2 master `R/bm_RunModelsLoop.R` (read via WebFetch 2026-10-14, summarised by
    the fetch tool, to be confirmed in the 4.3-4-7 tarball): the `calibration` column is scored at
    the cut `bm_FindOptimStat()` optimises on the calibration lines; the `validation` and
    `evaluation` columns are scored at that calibration cut (`threshold =
    cross.validation$cutoff[...]`). So biomod2 already scores held-out data at a cut learned
    elsewhere. Fails as written: overclaimed-novelty / defect-framing.
  - precise-form: "biomod2 reports TSS on the calibration data at the cut that maximises it there,
    and on held-out data at that calibration cut. `timesift` learns each candidate's cut from its
    out-of-fold predictions, and `tss_inflation()` estimates how far a TSS read at its own best cut
    rises above the true skill for a given set of presence counts." Drop "like most species
    distribution code" (no source).
- claim: "`tune()` ... the settings are never chosen on the units that score the tuned candidate"
  (B:355-357)
  - parent-does: `BIOMOD_Tuning()` not read this session. Keep the sentence about timesift; do not
    imply biomod2 tunes on scoring units.
- claim: "`occlusion()` computes the same quantity ... on the models a run kept for each fold,
  evaluated on that fold's held-out units" (B:367-369)
  - parent-does: `bm_VariablesImportance()` not read this session (which data it permutes on).
    Survives as a description; check before contrasting.

## Discrepancies

Only what the article states.

### ann-loss
- kind: design
- behaviour: biomod2 `ANN` leaves nnet's `entropy = FALSE` (least squares on the logistic output);
  `perceptron()` fits cross-entropy (B:68-71).
- evidence: perceptron fixtures (S: "The network"); biomod2 side not re-read.
- doc: @p:vig.biomod2.ann

### ann-generator
- kind: design
- behaviour: starting weights drawn from timesift's generator rather than R's (B:71-72).
- doc: @p:vig.biomod2.ann

### dnn-batch-standardise-burnin
- kind: design
- behaviour: cito batch default 10% of rows vs `batch_size` in rows; biomod2 standardises each
  column vs `mlp()` per channel across bins; tuned set's burn-in after 30 epochs not done by `mlp()`
  (B:98-101). cito 1.1.1 batch default confirmed on `?dnn`.
- doc: @p:vig.biomod2.dnn-diffs

### cut-per-statistic
- kind: design
- behaviour: biomod2 evaluates each statistic at its own optimised cut out of 100 candidates
  (`nb.thresh = 100`, `?bm_FindOptimStat` 4.3-5-0); `table_metric()` uses one rule's cut for all
  (B:310-313).
- doc: @p:vig.biomod2.one-cut
- note: version not named in the sentence; add "biomod2 4.3-4-7".

### maxent-vs-maxnet
- kind: design (undocumented)
- behaviour: biomod2 lists `MAXENT` and `MAXNET` as separate models (`?BIOMOD_Modeling` 4.3-5-0);
  the mapping table sends both to `maxent()`, maxnet's model (B:57), without a note.
- doc: none yet.

### biomod2-self-selected-tss
- kind: none (the article's claim does not match biomod2's held-out scoring; see Use case).
- upstream: not a defect; do not file.

No upstream-defect entry: the article states no biomod2 defect. Good.

## Evidence scope

### the simulated run (B:112-180)
- shows: on a record with a weekly planted signal, the weekly candidates score highest; the run
  produces candidates under biomod2 names.
- does-not-show: that timesift's learners score as biomod2's do on the same table; that any grain
  is right for real data.
- stated-at: B:114 ("Because the simulation puts the signal at the week, the comparison below can be
  checked against it"). Good.

### the fixtures (B:19-20)
- shows: each core gives its reference package's output on the fixture cases.
- does-not-show: identity with a full biomod2 run (biomod2 adds its own preprocessing: column
  standardisation for `DNN`, `entropy = FALSE` for `ANN`, its own cut search).
- stated-at: B:19-20 and the Differences section; no single sentence saying "the fixtures compare
  against the packages biomod2 calls, not against biomod2". Propose one.

### the side-by-side (B:184-224)
- shows: the call shape.
- does-not-show: matched settings (model sets and selection metric differ). Not run.
- stated-at: B:186 "Neither chunk is evaluated." Add that the model lists differ, or match them.

## Framing
- open-with: biomod2's closest equivalent with matched settings: the side-by-side would serve better
  as the first example if the model lists and metric matched (currently the simulated run comes
  first, with no biomod2 call to compare it to).
- organising question: where the learners agree with the packages biomod2 calls, where a run differs
  from a biomod2 run and why, and when biomod2 is the better tool (B:415-422, a good closing).
- avoid: "Coming from biomod2" (title; migration-assumption per docs-sidecars Checks, even though
  docs-style's own example uses "Coming from CoordinateCleaner" as a title, so the two skills
  disagree; the user decides); "switching" (B:422); "replaces `BIOMOD_RangeSize()`" (B:343);
  "like most species distribution code" (B:286); "Each algorithm is implemented once, in a compiled
  core" for the torch learners.
- title options (for the user): "timesift and biomod2"; "biomod2 models in timesift"; "Mapping
  biomod2 to timesift". Note: the story's Documents line calls the article "comparison, A3 mapping",
  which supports a neutral title.
