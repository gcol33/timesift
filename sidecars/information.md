# Information: timesift

Anchors are journey slugs (`sidecars/journey.md`): `@p:desc.*` DESCRIPTION, `@p:readme.*` README.Rmd,
`@p:py.*` python/README.md, `@p:pkg.*` ?timesift-package, `@p:vig.<article>.*` the vignettes and
articles (`vig.trouble` = troubleshooting.Rmd), `?py.<name> [page]` a Python reference page. Line
numbers appear only inside `warning:` lines and in code `source:` pointers, as read on 2026-10-14.

## Relation

### biomod2
- forms: each biomod2 model and its timesift learner (story.md); Each biomod2 algorithm corresponds to a learner constructor; the same algorithms; its biomod2 name; biomod2 run
- avoid: Coming from biomod2 (as a title); switching; replaces `BIOMOD_RangeSize()`; corresponds to, as a claim of identity where only the mapping is meant
- stated-at: @p:vig.biomod2.position; @p:vig.biomod2.mapping; @p:vig.biomod2.models-intro; @p:vig.biomod2.report; @p:vig.biomod2.side; @p:vig.biomod2.differences; @p:vig.biomod2.better-tool; @p:readme.trust; @p:py.trust
- warning: relation-drift: the story phrase is never used in biomod2.Rmd; the article says "algorithm" where the story says "model" (see Terms biomod2-model).
- warning: unlinked-package: biomod2 is never linked in biomod2.Rmd. Target read from its CRAN page: https://biomodhub.github.io/biomod2/
- reference-for: `envelope()`, pinned against biomod2's own `bm_SRE()` (inst/spec/make_fixtures.R, envelope_*.csv), so biomod2 belongs in the reference-package list at @p:readme.trust and @p:py.trust.

### reference-packages
- forms: pinned in the test fixtures against the reference package a species modeller already uses (story.md, @p:readme.trust, @p:py.trust); Each core reproduces the package biomod2 itself calls ... and the test fixtures check it against that package's output; checked against glmnet, MASS and `glm()`, earth, mgcv, rpart, and gbm or xgboost's `count:poisson`; pinned as a distance
- packages: glmnet, MASS, rpart, earth, mda, mgcv, nnet, maxnet (story.md); also randomForest, gbm, xgboost, cito in the biomod2 mapping
- avoid: wrapper; calls rpart (timesift never calls them at fit time; @p:vig.biomod2.cores says so correctly)
- stated-at: @p:readme.trust; @p:py.trust; @p:vig.biomod2.cores; @p:vig.biomod2.count-head; @p:vig.biomod2.mapping-table (notes column); @p:vig.learners.reference-table; @p:vig.trouble.lang-limits
- warning: relation-drift: "pinned" (story.md, @p:readme.trust) vs "check it against" (biomod2.Rmd:20).
- warning: unlinked-package: none of the reference packages, nor cito, glmnet, terra or torch (troubleshooting.Rmd:395), is linked at first mention in biomod2.Rmd or troubleshooting.Rmd.
- warning: claim-scope: cito (`DNN`) is in the mapping (biomod2.Rmd:64) but not in the list of reproduced packages (biomod2.Rmd:19), and `mlp()` is not a compiled core; biomod2's `MAXENT` (listed apart from `MAXNET` in `?BIOMOD_Modeling`, 4.3-5-0) is mapped onto maxnet's model without saying so.

## Terms

### anchor
- forms: anchor; target's instant; target's own instant
- as: the instant a target is read back from, given by `at` or `target_time`
- defined: @p:vig.representations.lookbacks
- glossed: @p:vig.representations.four-constructors
- other-senses: "anchor" and "anchor week" in comparing-grains.Rmd name the first week of a simulated lag variable's stretch (see driver)
- warning: form-collision: "anchor" means a target's instant in representations.Rmd and the first week of a planted stretch in comparing-grains.Rmd (:65-66, @p:vig.comparing-grains.true-grain, @p:vig.comparing-grains.anchors-result); a reader moving between the articles meets one word for two things.

### arm
- forms: arm; arms; `"grain|learner"`
- as: one representation fitted with one learner, as a ladder or a run's estimate table names it
- defined: @p:vig.comparing-grains.arm-names; none in python.Rmd or timesift.Rmd
- stated-at: @p:vig.python.predict ("its own arm"); @p:vig.python.reps-settings; @p:vig.python.elasticnet; @p:vig.python.paired
- other-senses: "procedure arm" (@p:vig.prediction.report-shape) and the `arm` column of `fit$estimate` (@p:vig.python.estimate) name the `selected` and `ensemble` rows, i.e. the procedures, not candidates; ?py.paired_contrast [scoring] names an arm `grain|learner`
- warning: use-before-define: used at @p:vig.comparing-grains.paired-method (comparing-grains.Rmd:186 "the difference between two arms") 34 lines before @p:vig.comparing-grains.arm-names defines it.
- warning: term-drift: the run's candidates are named `"elasticnet / week"` (comparing-grains.Rmd:351, :401) and the ladder's arms `"week|elasticnet"` (:192); the article never says the two spellings name the same pairing in two objects.
- warning: undefined: python.Rmd uses "arm" four times and never defines it; the reader has "candidate" from @p:vig.python.call-candidates and is not told an arm is the same thing (same in timesift.Rmd:126, 152, 219, 364).
- warning: form-collision: the `arm` column of the estimate means procedure, the prose means candidate (@p:vig.python.estimate vs @p:vig.python.predict).

### bigboss
- forms: `preset = "bigboss"`; tuned set called `"bigboss"`; tuned `DNN`; preset
- as: biomod2's tuned option set, which a learner's `preset` reproduces
- defined: @p:vig.biomod2.presets
- warning: use-before-define: used at biomod2.Rmd:71 and :89 (@p:vig.biomod2.ann, @p:vig.biomod2.dnn-bigboss) before @p:vig.biomod2.presets (:103).

### bin
- forms: bin; bins; sub-bin
- as: the stretch of record one number stands for
- defined: @p:vig.representations.grains (implicitly; "which stretch of time one number stands for" at @p:vig.representations.three-decisions); known in python.Rmd (used from @p:vig.python.call-built, glossed by output at @p:vig.python.candidates-code)
- warning: use-before-define: "bin" appears in `[unit, bin, channel]` and "monthly bins" (@p:vig.representations.three-columns, @p:vig.representations.array-attributes) before any sentence names a bin as the stretch from the opening; one clause at the first use closes it.

### biomod2-model
- forms: model (story.md, learners.Rmd, biomod2.Rmd:226 "biomod2 fits one species per call"); algorithm (biomod2.Rmd throughout)
- as: one of biomod2's named model types (`GLM`, `RF`, ...)
- defined: known
- warning: term-drift: story.md and learners.Rmd use "model"; biomod2.Rmd uses "algorithm". Pick one.

### block
- forms: block; block of features; tabular block; one row of features
- as: a representation flattened to one row of columns per target
- defined: @p:vig.representations.sequence-or-block
- other-senses: "tabular" is the learner declaration (`reads = "tabular"`); "block" is the representation's printout; both kept as forms. "block" also names a spatial fold block (`block_cv()`, @p:vig.prediction.block-settings)

### candidate
- forms: candidate; candidates; candidate rows; pair
- as: one representation and one learner fitted together and scored on the outer folds
- defined: @p:readme.usage.candidate (front door); @p:vig.python.call-candidates (by use: "The candidates are the comparison"); none in representations.Rmd, learners.Rmd, comparing-grains.Rmd, prediction.Rmd, biomod2.Rmd or troubleshooting.Rmd (Get started is the home)
- glossed: @p:vig.comparing-grains.run-setup ("builds each representation ... fits every learner on each")
- warning: undefined: @p:vig.representations.representations-as-settings ("name the candidates it produced") and @p:vig.learners.model-half ("the model half of a candidate") use it as known; link Get started or add a clause.
- warning: undefined: comparing-grains.Rmd:111 "weekly candidate" and prediction.Rmd:150 "candidate rows" rely on Get started.
- warning: use-before-define: biomod2.Rmd first uses it at :123 (@p:vig.biomod2.models-intro, "each candidate is reported") without gloss, and the article is read by someone coming from biomod2 who has not read Get started; troubleshooting.Rmd:598 (@p:vig.trouble.faq) uses it without gloss.
- warning: gloss-order: in python.Rmd the term appears in the printed report (@p:vig.python.call-code) before any sentence says a candidate is a representation-learner pair; that sentence exists in the project CLAUDE.md but not in the article.

### cell
- forms: cell; cells; scorable cell; scorable cells; (response, fold) cell
- as: one response in one outer fold, the unit a score is read on; scorable when both classes are present among its held-out units and among the units fitted on
- defined: @p:vig.prediction.scorable; @p:vig.python.split-cells; @p:vig.trouble.rare; @p:vig.biomod2.differences; none in representations.Rmd or comparing-grains.Rmd
- warning: undefined: representations.Rmd:582, :623 use "scorable cells" with no definition.
- warning: use-before-define: comparing-grains.Rmd:142 "mask of scorable cells" and :150 "All 40 cells" (@p:vig.comparing-grains.fold-map, @p:vig.comparing-grains.all-scorable) before any definition; :169 gives the composition only in passing.
- warning: use-before-define: biomod2.Rmd uses it at :165, :227, :245 (@p:vig.biomod2.count, @p:vig.biomod2.multi, @p:vig.biomod2.stack) before the definition at :403.
- warning: use-before-define: "on the same cells" at @p:vig.python.call-candidates precedes @p:vig.python.split-cells by six sections.
- warning: form-collision: prediction.Rmd uses "cell" for a grid cell (:43, :79, :119), a scoring cell (:122-135) and a raster cell (:399); :132 holds both senses in one sentence ("20 cells is scorable ... 47 and 59 of the 120 cells"). Both comparing-grains.Rmd:397 and prediction.Rmd:467 also use "cell of the representation" for a bin-by-statistic predictor.

### channel
- forms: channel; channels
- as: one statistic of one reading column (or a static column) laid along the bins
- defined: @p:vig.representations.run-columns; @p:vig.python.channels (by example)

### coarsest-adequate
- forms: coarsest adequate; `choose = "coarsest_adequate"`
- as: the coarsest candidate within one standard error of the highest inner score
- defined: @p:vig.comparing-grains.coarsest-rule

### combination
- forms: stack; the stack's weights; ensemble; combiner; combination; members; `ensemble("stack")`
- as: the candidates' out-of-fold predictions combined by fitted non-negative weights summing to one
- defined: @p:vig.prediction.combiner; @p:vig.biomod2.stack; @p:vig.python.call-procedure (stack); @p:vig.python.combiner (combiner); none in representations.Rmd or learners.Rmd
- other-senses: `ensemble` is also the procedure row reported beside `selected`, the `predict()` default `candidate = "ensemble"`, and `ensemble()` the specification of five methods
- warning: term-drift: the summary row and argument are `ensemble`, the prose says "the stack" (learners.Rmd:200, representations.Rmd:608); pick one in prose or say once that the `ensemble` row is the stack.
- warning: use-before-define: comparing-grains.Rmd:119 "the stack's weights were fitted" (@p:vig.comparing-grains.procedure-rows) with no definition in that article.
- warning: synonym-drift: four nouns for one object across @p:vig.python.call-procedure, @p:vig.python.predict, @p:vig.python.combiner, @p:vig.python.weights-read. Proposed forms: "ensemble" for the object, "combiner" only for the fitting step, "stack" only as the method name `"stack"`.

### committee-averaging
- forms: committee averaging; committee
- as: each member's predictions cut to presence or absence and averaged over members
- defined: @p:vig.biomod2.committee-intro

### coverage
- forms: `coverage()`; coverage table
- as: the count of readings per unit and bin
- defined: @p:vig.representations.coverage
- glossed: @p:vig.trouble.coverage

### day-level-statistic
- forms: day-level statistic; day-level statistics; extreme day
- as: a statistic that first reduces each calendar day, then reduces over the days
- defined: @p:vig.representations.statistics
- avoid: daily extreme (ambiguous between `cold_day` and `mean_daily_min`)

### digest
- forms: digest; digests; `digest_array()`
- as: the MD5 of an array written as the specification defines
- defined: @p:vig.representations.digests
- stated-at: @p:readme.trust; @p:py.trust (as "digests of the binned record")

### driver
- forms: driver; mechanism; anchor week
- as: the weighted mean of the anomaly a simulated response depends on; the mechanism is the rule placing the weights, the anchor week the first week of a lag variable's stretch
- defined: @p:vig.comparing-grains.simulator; anchor week at @p:vig.comparing-grains.true-grain
- other-senses: see anchor (form-collision)

### encoder
- forms: encoder; encoders; neural encoders; torch encoders; network
- as: a torch learner that fits all responses jointly through one shared embedding
- defined: @p:vig.learners.who-reads-what
- warning: use-before-define: representations.Rmd:470 "An encoder that ends in global pooling" and :487 "the torch encoders" use the term, and "global pooling", with no gloss; the audience in story.md does not know torch.

### fold-map
- forms: fold map; map; folds; outer folds; split
- as: the assignment of every target to one outer fold, shared by every candidate
- defined: @p:vig.prediction.split-why
- glossed: @p:vig.comparing-grains.fold-map
- avoid: "the split" alone where inner and outer splits are both in play (prediction.Rmd:156-161, @p:vig.prediction.block-settings)
- warning: undefined: representations.Rmd uses it at @p:vig.representations.artifacts-cross with no definition; @p:vig.learners.fit-count uses inner and outer folds with a one-clause gloss only.

### grain
- forms: grain; grains; named grain; temporal grain
- as: how much of the record one bin stands for: a calendar resolution (native, halfday, day, week, month, season, year)
- defined: @p:vig.representations.grains; @p:readme.grain (front door, by mechanism)
- glossed: @p:vig.biomod2.position (by example: monthly means)
- warning: undefined: python.Rmd never defines it (known there; ?py.GRAINS [arrays] lists the values).

### inflation
- forms: inflation; expected inflation; population skill; implied skill
- as: the excess of a TSS read at its own best cut over the skill it was planted at; `implied_skill()` is its inverse
- defined: @p:vig.comparing-grains.tss-why; @p:vig.comparing-grains.metrics-read
- avoid: "true skill" (comparing-grains.Rmd:439) beside "population skill" (:441); pick one.

### inner-folds
- forms: inner folds; inner score
- as: the folds the training units of one outer fold are split into, for choosing and weighting candidates
- defined: @p:vig.python.call-procedure
- glossed: @p:vig.learners.fit-count
- other-senses: `elasticnet()`'s own `n_inner` (its penalty search, printed at @p:vig.python.pin-code) is a second, deeper nesting under the same word
- warning: form-collision: a reader seeing `n_inner = 10` printed after being told the run uses five inner folds has no sentence telling them these are two different searches (@p:vig.python.pin-code, @p:vig.python.call-procedure).

### ladder
- forms: ladder; grain ladder; `grain_ladder()`
- as: every learner at every grain scored on one fold map and mask, without the run around it
- defined: @p:vig.comparing-grains.ladder-intro
- glossed: @p:vig.comparing-grains.roadmap (names it before the definition; roadmap, acceptable)

### learner
- forms: learner; learners; tabular learner; sequence learner
- as: a fit and a predict pair with declarations of what it reads
- defined: @p:vig.learners.model-half; @p:vig.python.own (by the user-learner sentence; known earlier)
- glossed: @p:vig.representations.three-decisions (used without gloss)

### level
- forms: level; a level read
- as: a score as reported (mean over responses), as distinct from the population skill
- defined: none
- other-senses: "Each plot has a level of its own" (@p:vig.python.tables-signal) means the plot's mean temperature; "held-out level" (prediction.Rmd:151) names the procedure estimate
- warning: form-collision: "level" is the plot's mean temperature at @p:vig.python.tables-signal and the reported score at @p:vig.python.honest and @p:vig.python.plot (alt text); same in timesift.Rmd:71, 294-305. Proposed: "a plot's own mean temperature" in the first; keep "level" for the score and gloss it once ("the score a run reports").

### lookback
- forms: lookback; lookbacks; `lookback()`
- as: a fixed span of record ending a fixed lag before a target's instant
- defined: @p:vig.representations.lookbacks; @p:vig.python.reps-sets
- glossed: @p:vig.representations.four-constructors
- warning: python.Rmd says "ending at each target's own instant" (@p:vig.python.reps-sets); the reference says "ending a fixed lag before" (?py.lookback [representations]). The lag defaults to zero, so the tutorial states the default case; one clause would cover it.

### mask
- forms: mask; mask of scorable cells; scorable-cell mask; the mask
- as: which cells hold both presences and absences among held-out and fitting units, computed from the response and the fold map alone, with no model
- defined: @p:vig.prediction.scorable
- glossed: @p:vig.comparing-grains.fold-map (without the fitting-side condition, comparing-grains.Rmd:142-143)
- warning: repeated-definition: defined in both comparing-grains.Rmd and prediction.Rmd; only prediction.Rmd gives both conditions. story.md makes comparing-grains the A2 home, so that article should carry the full definition and prediction.Rmd point to it.

### mixed-model-contrast
- forms: mixed-model contrast; `grain_contrasts()`
- as: a mixed model of per-cell scores comparing every grain with the best by Dunnett's procedure
- defined: @p:vig.comparing-grains.mixed-method

### nested-selection
- forms: nested selection; selection; `select_grain()`; choosing inside the training data
- as: the choice of candidate made on inner folds of each outer training set
- defined: @p:vig.comparing-grains.select-method
- warning: restated-arc: already explained at @p:vig.comparing-grains.procedure-rows (comparing-grains.Rmd:117-120) for the run; :245 explains it again for `select_grain()` without pointing back.

### occlusion-weight
- forms: weight; fall in score; importance
- as: the fall in held-out score when one bin or channel is permuted across held-out units; "importance" is one minus the correlation, as biomod2 computes it
- defined: @p:vig.comparing-grains.occlusion-method; @p:vig.biomod2.importance
- other-senses: "weight" is also the stack weight and the simulation's planted weight in comparing-grains.Rmd (:51, :119, :372)

### paired-contrast
- forms: paired contrast; contrast; step of the ladder
- as: the per-cell difference between two arms averaged within response, summarised across responses
- defined: @p:vig.comparing-grains.paired-method

### partial-bin
- forms: partial bin; partial
- as: a bin whose calendar span the record does not cover in full
- defined: @p:vig.representations.partial-bins; @p:vig.trouble.partial-why
- glossed: @p:vig.representations.array-attributes ("which bins the record does not cover in full")
- warning: use-before-define: @p:vig.representations.custom-partial uses "marked partial" before the definition section; acceptable as glossed.

### procedure-estimate
- forms: procedure rows; procedure; procedure's estimate; selected; ensemble; nested estimate; held-out level; the estimate
- as: the score of choosing or stacking inside each outer training fold, read once on the outer test folds
- defined: @p:vig.comparing-grains.procedure-rows; @p:vig.prediction.procedure-rows; @p:vig.python.call-procedure
- glossed: @p:readme.usage.candidate (front door: why `selected` and `ensemble` are the numbers to report)
- warning: gloss-order: comparing-grains.Rmd:117 opens on "The procedure rows" as a known term.
- warning: term-drift: five wordings across comparing-grains.Rmd and prediction.Rmd; "held-out level" (prediction.Rmd:151) and "nested estimate" (:250) are not introduced as the same thing.

### pseudo-absence
- forms: pseudo-absence; drawn units; background cells
- as: a unit drawn from a background pool to stand as an absence
- defined: @p:vig.prediction.pseudo-method

### python-side
- forms: the Python side; the Python twin; Python
- as: the Python package
- defined: known
- warning: term-drift: representations.Rmd:94 "the Python side", :567 "The Python twin"; mild, both clear.

### range-change-codes
- forms: lost (-2); kept (-1); absent in both (0); gained (1)
- as: biomod2's coding of a cell's change between two binary maps
- defined: @p:vig.prediction.range-read

### reading
- forms: reading; readings; reading-level
- as: one logged value at one instant
- defined: known

### record
- forms: record; series
- as: the time-stamped readings of every unit; `series` names the long table holding it
- defined: known; series at @p:vig.python.tables-def
- other-senses: `series` names the table holding the record

### representation
- forms: representation; representations
- as: a description of how the record becomes an array (grain or span, statistics, year start), made before any record is read
- defined: @p:vig.representations.representations-as-settings; @p:vig.python.reps-settings (settings only)
- warning: use-before-define: used from @p:vig.representations.roadmap and in learners.Rmd:13 as known; in representations.Rmd the definition comes in section 8.
- warning: use-before-define: python.Rmd uses it at @p:vig.python.call-built two sections before @p:vig.python.reps-settings; acceptable as `known` only if the opener glosses it (it does not; it says "reduction").

### response
- forms: response; responses; variable; variables
- as: one column of `y`, in the examples one species
- defined: none in python.Rmd (the code names `sp1..sp6`)
- other-senses: the printed objects say "variables" (`30 cells over 6 variables`, `n_variable`); the Python reference uses "variable" in learner docstrings ("One penalised regression per variable") and "response" in the head docstrings
- warning: term-drift: two words for one concept, in prose vs printed output vs reference. Proposed: "response" in prose everywhere; leave output column names; add one gloss where `n_variable` first prints (@p:vig.python.ladder-code): "one row per response (`variable`)".

### response-head
- forms: response head; head; heads; joint head
- as: the registration saying what response values are, how learners fit them (loss, output) and which metric and cells a score is defined on
- defined: @p:vig.learners.responses; @p:vig.biomod2.heads
- warning: use-before-define: learners.Rmd:105-121 (table column "every head", "joint head", "whatever loss the response head names") and :377 "the response head's case weights" come before the definition at :432.
- warning: use-before-define: biomod2.Rmd uses it at :70 ("the loss of the response head", @p:vig.biomod2.ann) and :245 (@p:vig.biomod2.stack) before @p:vig.biomod2.heads (:318).

### richness
- forms: richness; number of responses present at a unit
- as: the count of responses present at a unit, the stratifying value of `cv()`
- defined: @p:vig.comparing-grains.fold-map
- glossed: @p:vig.prediction.constructors (states it without the word)

### schrankogel-study
- forms: Schrankogel study; Schrankogel deposit; the study; the deposit; the Schrankogel record of the package's origin
- as: the alpine study the package was built from, its hourly soil-temperature record deposited on Zenodo
- defined: @p:vig.learners.encoders-when (the one place it is introduced); @p:readme.study and @p:py.study (front door)
- warning: undefined: representations.Rmd:150 first names "The Schrankogel deposit" with no gloss, and uses it at :196, :255, :302, :665, :674 before saying at :674 what it is; "deposit" (a Zenodo deposit) is never glossed in representations.Rmd or learners.Rmd.
- warning: undefined: troubleshooting.Rmd:489 (@p:vig.trouble.iterate) names the Schrankogel record with no gloss.

### sequence
- forms: sequence; sequence learner
- as: a representation whose bins are ordered in time
- defined: @p:vig.representations.sequence-or-block

### sift
- forms: sift
- as: the set of representations a run compares
- defined: none in python.Rmd (the argument appears at @p:vig.python.call-code; @p:vig.python.call-built says "Every representation named in `sift`")
- other-senses: the Python reference calls a fit "A fitted sift" (?py.Timesift [one-call])

### static
- forms: static predictor; `static`
- as: a column of `targets` carried beside the representation as a predictor, held constant across bins
- defined: @p:vig.python.static; none in representations.Rmd or learners.Rmd
- warning: undefined: representations.Rmd:523 "as a `static` predictor such as elevation does inside a run" relies on `timesift(static = )`, never introduced in representations.Rmd or learners.Rmd.

### statistic
- forms: statistic; statistics; stats
- as: what one number of a bin summarises
- defined: @p:vig.representations.statistics

### supplied-calendar
- forms: supplied calendar; calendar passed as a function; custom
- as: a function of the reading instants returning each reading's bin start
- defined: @p:vig.representations.custom-calendar

### target
- forms: target; targets
- as: one row to predict
- defined: @p:vig.representations.two-tables; @p:vig.python.tables-def
- warning: use-before-define: representations.Rmd:22 "lookbacks anchored on each target" precedes :49.

### threshold
- forms: threshold; cut; decision threshold; rule
- as: the probability at or above which a response is called present, learned from out-of-fold predictions
- defined: @p:vig.prediction.threshold-method

### timesift-set
- forms: `timesift_set`; set
- as: several built arrays over the same targets, keyed by label
- defined: @p:vig.representations.sets-of-arrays
- glossed: @p:vig.representations.grains
- other-senses: "set" also names a `grains()`/`lookbacks()` set of unbuilt representations (@p:vig.representations.sets) and a set of learners from `c()` (@p:vig.learners.fit-predict-contract); three senses of "set" in two articles

### true-grain
- forms: true grain; planted grain; the truth
- as: the coarsest grain at which the driver is an exact linear function of the binned record
- defined: @p:vig.comparing-grains.true-grain
- warning: definition-clash: @p:vig.comparing-grains.other-mechanisms (comparing-grains.Rmd:317) says under `"season"` "every finer grain nests inside it"; ?simulate_records (R/simulate.R:30-31) says month and day nest, and :64-65 of the same article says week bins straddle month boundaries.

### tss
- forms: TSS; true skill statistic
- as: sensitivity plus specificity minus one at the cut maximising it
- defined: @p:vig.comparing-grains.tss-why
- glossed: @p:readme.supports and @p:py.supports ("true skill statistic")
- warning: undefined-expansion: "TSS" is never expanded in comparing-grains.Rmd or prediction.Rmd.

### unit
- forms: unit; units; plot
- as: the thing a record belongs to, named by a column both tables carry, and what the folds deal; plot is the examples' unit
- defined: @p:vig.representations.two-tables
- warning: use-before-define: representations.Rmd:12 "an array of numbers per unit" precedes :50.
- warning: term-drift: python.Rmd uses "units" at @p:vig.python.split-folds ("deals units into folds") and @p:vig.python.split-cells without definition, after "targets" was defined as the thing held out; the fold printout says `60 units`. Where a unit carries one target they coincide, which is the tutorial's case, but nothing says so.

### won
- forms: won
- as: how many responses a candidate scored highest on
- defined: @p:vig.python.call-candidates; @p:vig.prediction.won

## Quantities

Values stay in the source named. Prose numbers in the articles are typed by hand unless an entry says inline `r`; hand-typed values were checked against the site built at dffca49 (representations.Rmd, learners.Rmd) on 2026-10-14.


### digests-257
- metric: count of representation digests both test suites assert
- source: inst/spec/fixtures/digests.csv; lookback_digests.csv; coverage.csv
- stated-at: @p:readme.trust; @p:py.trust
- warning: source-scope: digests.csv holds 188 rows; the stated 257 is reached by row count as digests.csv (188) + lookback_digests.csv (64) + coverage.csv (5), leaving out channels_digests.csv (22). The source line should name the three files, and "digests of the binned record" (README.Rmd:58) covers coverage counts and lookbacks too.

### study-design
- metric: plots, species and years of hourly soil-temperature record in the Schrankogel study; hand-built features of the baseline
- population: the Schrankogel study
- source: inst/reproduce/README.md (bins per grain; the 188 variables)
- stated-at: @p:readme.study; @p:py.study; @p:readme.hero (alt text); @p:py.hero (alt text); @p:vig.representations.memory; @p:vig.learners.encoders-when; @p:vig.trouble.iterate

### hourly-cost
- metric: TSS of the convolutional network on the hourly record below its own best grain
- population: the Schrankogel study
- source: the paper (Schrankogel CNN 2026); inst/reproduce/README.md grid table
- stated-at: @p:readme.study; @p:py.study
- note: consistent by arithmetic with the hero alt text (hourly against weekly score, @p:readme.hero)

### yearly-drop
- metric: TSS at the yearly grain below the weekly peak
- population: the Schrankogel study
- source: the paper; inst/reproduce/README.md grid table
- stated-at: @p:readme.study
- note: consistent by arithmetic with the hero alt text (yearly against weekly score)

### extreme-day-gain
- metric: TSS gain from adding the coldest and warmest day to a bin's mean, at weekly and at yearly grain
- population: the Schrankogel study
- source: the paper only; not in inst/reproduce/README.md
- stated-at: @p:readme.study; @p:vig.representations.schrankogel-extremes
- warning: untraced-number: no package-side source holds the pair (README.Rmd:203, representations.Rmd); held by the paper and CLAUDE.md "Why it exists". The copy in the README and the one in representations.Rmd drift if either is edited.

### mlp-vs-penalised
- metric: TSS of the fully connected network minus the penalised logistic model on the hand-built features, with the paired p-value
- compares: fully connected network; penalised logistic model on the 188 aggregates
- population: the Schrankogel study
- source: the paper
- stated-at: @p:readme.study; @p:vig.learners.encoders-when

### tss-inflation-060
- metric: expected inflation of a self-selected TSS at a true skill of 0.60 on the Schrankogel design
- population: the Schrankogel study's presence counts
- source: inst/reproduce/README.md (inflation row)
- stated-at: @p:readme.tss

### hero-tss
- metric: TSS of the convolutional network at each of the seven grains, over 101 species
- population: the Schrankogel study
- source: study table S12, via data-raw/hero.R
- stated-at: @p:readme.hero (alt text); @p:py.hero (alt text)

### wheel-interpreters
- metric: Python versions the wheels are built for
- source: pyproject.toml `[tool.cibuildwheel]` build
- stated-at: @p:py.install

### python-minimum
- metric: lowest supported Python version
- source: pyproject.toml `requires-python`
- stated-at: @p:readme.install; @p:py.install


### rep-example-design
- metric: example record design (thirty plots, 400 days, hourly, from 1 September 2021)
- source: representations.Rmd `data` chunk (design constant)
- stated-at: @p:vig.representations.two-tables

### rep-ladder-bins
- metric: bins per grain on the example record (native through year), monthly count, partial-drop count
- source: `ladder` and `partial` chunk outputs
- stated-at: @p:vig.representations.array-attributes; @p:vig.representations.ladder-counts; @p:vig.representations.keep-or-drop; @p:vig.representations.auto-set; @p:vig.representations.run-readout
- warning: restated-number: the native bin count is given at @p:vig.representations.ladder-counts and again at @p:vig.representations.auto-set; the second use anchors the "no cap" point, acceptable.

### rep-month-counts
- metric: readings per month, last month's readings
- source: `week-start` chunk output
- stated-at: @p:vig.representations.iso-week-calendar-month
- warning: the prose "the counts run 720, 744 and 672" reads as the first three months; the output's first three are 720, 744, 720 (672 is February). Say "720, 744 or 672 readings for months of 30, 31 and 28 days".

### rep-dst-counts
- metric: readings on the spring-forward day; bin start shift
- source: `zones` chunk output
- stated-at: @p:vig.representations.time-zones; @p:vig.representations.dst-readout

### rep-year-counts
- metric: readings per year on September and January boundaries
- source: `year-start` chunk output
- stated-at: @p:vig.representations.hydro-readout

### rep-stats-p01
- metric: September 2021 statistics of p01, before and after the spike
- source: `stats` and `spike` chunk outputs
- stated-at: @p:vig.representations.stats-order; @p:vig.representations.spike-readout

### rep-custom-counts
- metric: readings in the first and last solstice-season bin
- source: `solstices` chunk output
- stated-at: @p:vig.representations.custom-partial

### rep-gap-counts
- metric: p07 February readings against neighbours
- source: `gap-month` chunk output
- stated-at: @p:vig.representations.three-ways-out

### rep-lookback-means
- metric: weekly means before each visit
- source: `lookback` chunk output
- stated-at: @p:vig.representations.lookback-readout

### rep-lengths
- metric: bins per grain for records of 120 days, one year, three years
- source: `lengths` chunk output
- stated-at: @p:vig.representations.lengths-readout

### rep-memory
- metric: megabytes of native and weekly three-channel arrays at the study size; numbers per plot
- source: `memory` chunk (computed); numbers per plot also in inst/reproduce/README.md
- stated-at: @p:vig.representations.memory-readout

### schrankogel-bins
- metric: hourly readings and day bins of the deposit; months, seasons, years
- population: the Schrankogel study
- source: inst/reproduce/README.md (bins per grain)
- stated-at: @p:vig.representations.fixed-offset-advice; @p:vig.representations.lengths-readout

### digest-precision
- metric: decimal places an array is rounded to before its digest (twelve)
- source: inst/spec/representation.md (design constant)
- stated-at: @p:vig.representations.digests; @p:vig.representations.digest-readout


### lrn-example-design
- metric: units, step, length, prevalence and simulator AUC of the example record (150 units, six-hourly, one year)
- source: learners.Rmd `data` chunk (design constant)
- stated-at: @p:vig.learners.simulated-record

### lrn-native-bins
- metric: native bins of the learners record
- source: design (days x readings per day); `skip-native` output
- stated-at: @p:vig.learners.sequence-refusal-preview

### lrn-run-time
- metric: elapsed seconds for the eight-learner run
- source: inline `r` (computed at knit)
- stated-at: @p:vig.learners.fit-count

### lrn-fit-count
- metric: fits per candidate under 5 outer x 3 inner folds
- source: design arithmetic
- stated-at: @p:vig.learners.fit-count

### lrn-summary-scores
- metric: mean AUC on outer folds; selected and ensemble AUC
- source: `summary` chunk output
- stated-at: @p:vig.learners.summary-readout
- warning: hard-typed (0.750, 0.741, 0.740); the run time beside it is inline `r`, these are not.

### lrn-settings-auc
- metric: one-split held-out AUC of ten settings
- source: `settings` chunk output
- stated-at: @p:vig.learners.settings-readout
- warning: hard-typed.

### lrn-encoder-auc
- metric: encoder AUC after a few epochs
- source: `encoders` chunk output (eval = has_torch)
- stated-at: @p:vig.learners.encoders-readout
- warning: the chunk is conditional on torch; on a build without torch the prose "both encoders score close to 0.5" describes output that is not shown.

### lrn-own-auc
- metric: AUC of `channel_means` and of the weekly elastic net
- source: `register-learner` chunk output
- stated-at: @p:vig.learners.own-readout; @p:vig.learners.wrong-choice (as "below 0.5")

### lrn-weights
- metric: presence weight per response; default cap
- source: `weights` chunk output; `positive_weights()` signature (R/response.R:145)
- stated-at: @p:vig.learners.register-response; @p:vig.learners.cap-two

### lrn-brier
- metric: negative Brier score of elastic net and forest under the capped head
- source: `register-metric` chunk output
- stated-at: @p:vig.learners.capped-readout
- warning: hard-typed.

### lrn-columns
- metric: weekly and monthly column counts of the example record; stepwise term count
- source: design (ISO weeks over 365 days; squares)
- stated-at: @p:vig.learners.choosing; @p:vig.learners.additive-month; @p:vig.learners.start-elasticnet

### lrn-timings
- metric: fit seconds per learner and per `k`
- source: inline `r` (computed at knit)
- stated-at: @p:vig.learners.additive-k-readout; @p:vig.learners.start-elasticnet

### lrn-counts
- metric: shipped learners; tabular per-response learners; learners carrying tuning grids
- source: R/zzz.R:51-65 (registrations); R/tune.R:182-200 (grids: forest, boosting, mars, discriminant, maxent, envelope, perceptron, mlp)
- stated-at: @p:vig.learners.model-half; @p:vig.learners.who-reads-what; @p:vig.learners.registered-grids
- warning: claim-scope: "Eight shipped learners carry the grids that biomod2's `BIOMOD_Tuning()` searches": `mlp` is one of the eight and is not a biomod2 model; check whether BIOMOD_Tuning's ANN grid is what `mlp` carries, or say seven plus mlp.


### cg-design
- metric: n, variables, days, step, prevalence and `auc` of the simulated record
- source: `simulate` chunk call (design constants)
- stated-at: @p:vig.comparing-grains.simulate-call; @p:vig.comparing-grains.ceiling

### cg-weekly-bins
- metric: weekly bins in the record
- source: `ladder` chunk (`dim` output)
- stated-at: @p:vig.comparing-grains.true-grain

### cg-day-columns
- metric: daily bins
- source: `ladder` chunk
- stated-at: @p:vig.comparing-grains.run-result; @p:vig.comparing-grains.variance-bias

### cg-candidate-auc
- metric: mean AUC per grain in the run
- source: `fit` chunk
- stated-at: @p:vig.comparing-grains.run-result

### cg-ceiling-gap
- metric: weekly mean minus the simulator's `auc`
- source: arithmetic on `fit` output and design
- stated-at: @p:vig.comparing-grains.run-result

### cg-cells
- metric: scorable cells in the run
- source: `folds` chunk
- stated-at: @p:vig.comparing-grains.all-scorable

### cg-ladder-cells
- metric: ladder cells
- source: `ladder` chunk
- stated-at: @p:vig.comparing-grains.ladder-matches

### cg-train-units
- metric: training units per outer fold
- source: arithmetic (n, v)
- stated-at: @p:vig.comparing-grains.variance-bias; @p:vig.comparing-grains.select-result; @p:vig.comparing-grains.anchors-result; @p:vig.comparing-grains.guide-size

### cg-contrast-diff
- metric: week minus month, week minus day
- source: `contrast` chunk
- stated-at: @p:vig.comparing-grains.paired-result; @p:vig.comparing-grains.guide-size (restated)

### cg-p-floor
- metric: smallest signed-rank p-value
- source: `p-floor` chunk
- stated-at: @p:vig.comparing-grains.paired-result; @p:vig.comparing-grains.few-responses; @p:vig.comparing-grains.three-result; @p:vig.comparing-grains.guide-paired (restated)

### cg-mixed-width
- metric: width of the mixed-model intervals
- source: `grain-contrasts` chunk (needs lme4, lmerTest, emmeans)
- stated-at: @p:vig.comparing-grains.mixed-read

### cg-select
- metric: procedure score, inner score range
- source: `select` chunk
- stated-at: @p:vig.comparing-grains.select-result

### cg-null-best
- metric: best null-ladder grain mean
- source: `null` chunk
- stated-at: @p:vig.comparing-grains.null-best; @p:vig.comparing-grains.guide-report (restated)

### cg-null-select
- metric: null procedure score, interval, fold choices, inner range, gap
- source: `null-select` chunk
- stated-at: @p:vig.comparing-grains.null-select

### cg-coarsest
- metric: folds whose rule kept the best
- source: `null-coarsest` chunk
- stated-at: @p:vig.comparing-grains.coarsest-read

### cg-mechanism-peaks
- metric: season and event peaks, month score, day-to-week drop
- source: `mechanisms` chunk
- stated-at: @p:vig.comparing-grains.mechanisms-result

### cg-anchor-dates
- metric: dates of the two anchor weeks
- source: `occlusion-check` chunk
- stated-at: @p:vig.comparing-grains.anchors-lead

### cg-occlusion-anchor
- metric: anchor-week fall, off-stretch weight
- source: `occlusion-check` chunk
- stated-at: @p:vig.comparing-grains.anchors-result

### cg-lag-decay
- metric: factor between consecutive weekly weights
- source: ?simulate_records (`lag = exp(-(0:3))`, R/simulate.R:233); design constant
- stated-at: @p:vig.comparing-grains.anchors-result

### cg-channel
- metric: mean channel weight, ratio to cold day
- source: `occlusion-channel` chunk
- stated-at: @p:vig.comparing-grains.channel-result

### cg-curve-span
- metric: span of the response curve per response
- source: `response-curve` chunk
- stated-at: @p:vig.comparing-grains.curve-result

### cg-metric-floors
- metric: average-precision floor at the prevalence, AUC floor
- source: design constants
- stated-at: @p:vig.comparing-grains.metrics-read

### cg-inflation
- metric: inflation at two skills on the example design
- source: `inflation` chunk
- stated-at: @p:vig.comparing-grains.inflation-read

### cg-implied
- metric: observed TSS, implied skill
- source: `score` and `implied` chunks
- stated-at: @p:vig.comparing-grains.implied-read

### cg-rare
- metric: presence range, unscorable cells, inflation on the rare record
- source: `rare` chunk
- stated-at: @p:vig.comparing-grains.rare-result

### cg-held-out
- metric: units held out per fold
- source: arithmetic (n, v)
- stated-at: @p:vig.comparing-grains.guide-size


### pr-grid
- metric: grid size, days, species
- source: `data` chunk code (design constants)
- stated-at: @p:vig.prediction.grid-design; @p:vig.prediction.data-code

### pr-fold-sizes
- metric: cells per fold, transect size
- source: `fold-map` chunk
- stated-at: @p:vig.prediction.fold-sizes

### pr-cells
- metric: scorable cells; presence range
- source: `cells` and `data` chunks
- stated-at: @p:vig.prediction.one-denominator

### pr-random-procedure
- metric: selected and stack AUC under the random split
- source: `estimates` chunk
- stated-at: @p:vig.prediction.procedure-rows; @p:vig.prediction.guide-stack (restated)

### pr-block-procedure
- metric: drop and level under spatial blocks
- source: `estimates` chunk
- stated-at: @p:vig.prediction.block-drop; @p:vig.prediction.guide-stack (restated); @p:vig.prediction.guide-spatial (restated)

### pr-forest-en-drop
- metric: forest and elastic net monthly levels under both splits
- source: `fit` and `fit-block` print
- stated-at: @p:vig.prediction.block-drop

### pr-weights
- metric: refitted weights
- source: `weights` chunk
- stated-at: @p:vig.prediction.weights-read

### pr-fold-weights
- metric: per-fold weekly forest weight
- source: `weights` chunk
- stated-at: @p:vig.prediction.weights-read

### pr-methods-auc
- metric: AUC of each combination; best candidate
- source: `methods-score` chunk; `fit` print
- stated-at: @p:vig.prediction.methods-score

### pr-committee-values
- metric: values a four-vote share takes
- source: arithmetic, design constant
- stated-at: @p:vig.prediction.methods-score

### pr-won
- metric: wins per candidate
- source: `report` chunk
- stated-at: @p:vig.prediction.won

### pr-cuts
- metric: youden and kappa cut for sp4
- source: `thresholds` chunk
- stated-at: @p:vig.prediction.rules

### pr-called
- metric: cells called present per rule
- source: `threshold-counts` chunk
- stated-at: @p:vig.prediction.counts-read; @p:vig.prediction.guide-rule-mpa (restated)

### pr-spread
- metric: cell means and coefficient of variation
- source: `spread-predict` chunk
- stated-at: @p:vig.prediction.spread-read

### pr-pseudo-admitted
- metric: admitted counts; band share
- source: `pseudo` chunk
- stated-at: @p:vig.prediction.pseudo-admits

### pr-po-auc
- metric: presence-background AUC
- source: `pseudo-fit` chunk
- stated-at: @p:vig.prediction.pseudo-auc

### pr-project-time
- metric: elapsed seconds of the projection
- source: inline `r` (`took`)
- stated-at: @p:vig.prediction.project-read; @p:vig.prediction.guide-cost

### pr-warming
- metric: warming applied for the second period
- source: `range` chunk code (design constant)
- stated-at: @p:vig.prediction.range-method

### pr-range
- metric: cells gained, lost, kept per species
- source: `range` chunk
- stated-at: @p:vig.prediction.range-read

### pr-se
- metric: standard errors of selected and stack
- source: `estimates` chunk
- stated-at: @p:vig.prediction.guide-stack


### bm-mapping-rows
- metric: rows of the mapping table
- source: @p:vig.biomod2.mapping-table; matches the models of `?BIOMOD_Modeling` 4.3-5-0 with `MAXENT` and `MAXNET` in one row
- stated-at: @p:vig.biomod2.mapping-table

### bm-nnet-settings
- metric: `size`, `decay`, `rang`, `maxit` of the network; hidden units of `perceptron()`
- source: R/learners_perceptron.R:140, :142; biomod2 side: none found this session (`bm_ModelingOptions` not read)
- stated-at: @p:vig.biomod2.ann

### bm-cito-defaults
- metric: cito's default network (layers, activation, dropout, optimiser, learning rate, epochs) and validation share
- source: cito 1.1.1 `?dnn`, read 2026-10-14; match
- stated-at: @p:vig.biomod2.dnn-default; @p:vig.biomod2.dnn-diffs

### bm-cito-burnin
- metric: epochs before early stopping in biomod2's tuned `DNN`
- source: none found (cito's own default `burnin = Inf`, so the value is biomod2's tuned one; biomod2 bigboss DNN options not read)
- stated-at: @p:vig.biomod2.dnn-diffs

### bm-tree-bigboss
- metric: `min_split`, `min_leaf`, `cp`, `max_depth` and inner folds of the bigboss tree
- source: R/learners_tree.R:107; match
- stated-at: @p:vig.biomod2.presets

### bm-example-design
- metric: units, step, length and responses of the example record
- source: `data` chunk (design constants); match
- stated-at: @p:vig.biomod2.run

### bm-fixed-grain-counts
- metric: weekly columns, coefficients, training units and term count of the two fixed-grain learners
- source: none (hard-coded)
- stated-at: @p:vig.biomod2.fixed-grain
- warning: stale-number: "53 columns", "478 coefficients", "160 training units", "17" (biomod2.Rmd:145-146): 160 = 200 x 4/5 and 17 = 4 x (5 - 1) + 1 match the chunk's `k = 5L`, but 478 = 53 x (10 - 1) + 1 uses the default `k = 10`; under the chunk's `k = 5` it is 213.
- warning: stale-number: "one presence in twenty at each end" (biomod2.Rmd:147): `quantile = 0.025` is one in forty at each end (R/learners_envelope.R:27-28).

### bm-candidates
- metric: candidates in the run (learners x grains plus pinned), outer folds
- source: `models` chunk (twelve entries, two pinned), sift of three grains, `fit` chunk `cv(v = 5)`; match by arithmetic, not inline
- stated-at: @p:vig.biomod2.count

### bm-weekly-top
- metric: how many of the highest mean AUCs belong to weekly candidates
- source: none (prose, not inline)
- stated-at: @p:vig.biomod2.report
- warning: untraced-number: "four highest mean AUCs belong to weekly candidates" (biomod2.Rmd:173) is typed by hand with no inline value.

### bm-top-weights
- metric: the largest stack weights shown; `min_score` cut
- source: `weights` chunk `[1:6]`; `fit` chunk
- stated-at: @p:vig.biomod2.min-score

### bm-threshold-grid
- metric: size of biomod2's threshold grid
- source: `?bm_FindOptimStat` 4.3-5-0 (`nb.thresh` default), read 2026-10-14; match
- stated-at: @p:vig.biomod2.one-cut

### bm-data-types
- metric: response types biomod2 4.3 models
- source: `?BIOMOD_FormatingData` 4.3-5-0 (binary, count, multiclass, ordinal, relative, abundance); the version they arrived in: none found (NEWS not reachable)
- stated-at: @p:vig.biomod2.heads
- warning: claim-scope: "relative" is not mentioned among the types biomod2.Rmd:317 lists.


### tr-example-design
- metric: plots, step, days, start date and species of the example record
- source: `data` chunk (design constants); match
- stated-at: @p:vig.trouble.record

### tr-calendar-dates
- metric: weekdays and dates of the first bins
- source: the calendar (2021-09-01 a Wednesday; 2021-09-06 and 2021-09-13 Mondays); match
- stated-at: @p:vig.trouble.gap; @p:vig.trouble.partial; @p:vig.trouble.partial-run; @p:vig.trouble.predict
- warning: ambiguous: "30 August to 25 October" (troubleshooting.Rmd:410-411) are bin starts of the first and last week; say "weeks beginning ...".

### tr-zone-arithmetic
- metric: UTC offsets and day lengths under Europe/Vienna and Etc/GMT-1
- source: zone arithmetic; correct for CEST / Etc/GMT-1
- stated-at: @p:vig.trouble.zone; @p:vig.trouble.dst; @p:vig.trouble.fixed

### tr-knit-values
- metric: bin counts, timings, fitted targets, megabytes
- source: inline `r` (knit-sourced)
- stated-at: @p:vig.trouble.partial-why; @p:vig.trouble.dst; @p:vig.trouble.ids-count; @p:vig.trouble.cost-read; @p:vig.trouble.iterate

### tr-fold-placement
- metric: which folds hold both presences of `sp3`
- source: none (prose describing a seeded draw, not inline)
- stated-at: @p:vig.trouble.rare-read
- warning: untraced-number: "folds 1, 2 and 4 ... fold 3" (troubleshooting.Rmd:361) breaks if `fold_map()`'s draw changes.

### tr-cost-formula
- metric: fits per run (`v_outer * (n_inner + 1) * candidates`)
- source: R/timesift.R:138 (`n_inner` default); match
- stated-at: @p:vig.trouble.cost


### sp-nested-folds
- metric: least outer folds for `"nested_cv"`, equation (15), Bates, Hastie and Tibshirani (2024)
- source: none checked this session
- stated-at: @p:vig.contract.same-call

### sp-seed-formula
- metric: the seed of each refit (`seed + 10007 r + 101 a + b + 3001 c`)
- source: none checked this session (design constant)
- stated-at: @p:vig.contract.same-call

### sp-df-rule
- metric: degrees-of-freedom rounding, simulation count and level of the contrast intervals
- source: none checked this session (emmeans conventions)
- stated-at: @p:vig.contract.same-call

### sp-agreement-magnitude
- metric: "of the order of a thousandth"
- source: none found
- stated-at: @p:vig.contract.same-call
- warning: unbacked-claim: vague magnitude with no source (representation.md:2240).

### sp-colours
- metric: plot palette `hcl.colors(n, "Dark 3")`
- source: R/plot.R:111; R/select_grain.R:401; match
- stated-at: @p:vig.contract.same-call

### sp-swa-start
- metric: admissible range of `swa_start`
- source: R/control.R:73 (default)
- stated-at: @p:vig.contract.same-call


### py-noise-ratio
- metric: hour-to-hour noise relative to the per-plot signal ("an order of magnitude larger")
- source: @p:vig.python.tables-code (noise sd against the level's sd: a factor of 8)
- stated-at: @p:vig.python.tables-signal
- note: holds as an order of magnitude

### py-outer-folds
- metric: count of outer folds ("the same five folds")
- source: @p:vig.python.call-code (`ts.cv(v=5)`)
- stated-at: @p:vig.python.call-built

### py-responses
- metric: number of responses in the example
- source: @p:vig.python.tables-code
- stated-at: @p:vig.python.interval; @p:vig.python.paired

### py-channels
- metric: channels per candidate
- source: @p:vig.python.candidates-code
- stated-at: @p:vig.python.channels

### py-auto-bins
- metric: the bin count a grain must give to enter `"auto"`
- source: ?py.auto_grains [representations]; python/timesift/specs.py `auto_grains`
- stated-at: @p:vig.python.reps-sets; @p:vig.trouble.faq

### py-month-days
- metric: days in a calendar month
- source: the calendar
- stated-at: @p:vig.python.calendar
- warning: stale-number: "28, 30 or 31 days" omits 29 (February of a leap year). Proposed: "28 to 31 days".

### py-block-hours
- metric: hours in a drifting month and week block
- source: none in the article; arithmetic (8766 / 12 rounded, 7 x 24); same figures in the project CLAUDE.md
- stated-at: @p:vig.python.calendar

### py-day-share
- metric: share of a day's mean one hourly reading carries ("its twenty-fourth")
- source: @p:vig.python.tables-code (`freq="h"`)
- stated-at: @p:vig.python.extremes-def
- note: true for an hourly record only; the sentence does not say so

### py-cold-hour
- metric: the hypothetical cold reading
- source: none (illustrative)
- stated-at: @p:vig.python.extremes-def


### py-maxent-design-size
- metric: features of a weekly three-channel representation under "lqh" and the extra products under "lqph"
- source: arithmetic, verified (columns x knots x hinges; columns choose two); the column count is the weekly three-channel count of inst/reproduce/README.md
- stated-at: ?py.maxent [learners]

### py-maxnet-class-thresholds
- metric: presence counts at which maxnet's default feature classes change
- source: maxnet `maxnet.formula()`; not checked this session
- stated-at: ?py.maxent [learners]

### py-preset-values
- metric: forest, tree, boosting, perceptron `"default"` and `"bigboss"` values
- source: biomod2 option sets / the reference packages; python/timesift/learners.py presets; not traced value by value this session
- stated-at: ?py.forest; ?py.tree; ?py.boosting; ?py.perceptron [learners]

### py-tune-grids
- metric: perceptron hidden and decay grid; mlp width grid
- source: python/timesift/tune.py:101-102 (verified)
- stated-at: ?py.tune [learners]

### py-glm-iterations
- metric: IRLS iterations before a stepwise move is refused
- source: R `glm.control(maxit)`; python/timesift/_stepwise.py not checked
- stated-at: ?py.linear [learners]

### py-earth-radius
- metric: sphere radius for `lonlat` distances
- source: python/timesift/pseudo.py; not checked this session (design constant)
- stated-at: ?py.pseudo_absences [one-call]

### py-signed-rank-normal
- metric: per-variable count at which the signed-rank p-value switches to the normal approximation
- source: R `wilcox.test` exact rule; python/timesift/ladder.py not checked
- stated-at: ?py.paired_contrast [scoring]

### py-nested-cv-benchmark
- metric: "in the package's benchmark that spread exceeded the estimate's where the sample was small or the signal absent"
- source: none found in the page; presumably inst/benchmark/ (not checked)
- stated-at: ?py.select_grain [ladder]
- warning: untraced-number: a claim about a benchmark result with no pointer. Proposed: name the benchmark file or article.

## Defaults

Each entry names the argument, where its value lives (`home`: the signature or `@param`), and every place the docs state it. Values are read in `home`.

### presence-absence-metric
- argument: timesift(metric), the `presence_absence` head's metric (R and Python)
- home: R/response.R:170 (`metric = "roc_auc"`); python/timesift/response.py `PRESENCE_ABSENCE`; ?py.PRESENCE_ABSENCE [extending]
- stated-at: @p:desc.default; @p:pkg.default; @p:readme.tss; @p:vig.timesift.implied; @p:vig.comparing-grains.tss-why; @p:vig.python.inflation-read; @p:vig.contract.same-call
- status: match at every stated-at
- history: DESCRIPTION stated TSS as the default until it was fixed on 2026-10-14.

### timesift-learners
- argument: timesift(learners)
- home: @param R/timesift.R:42 (`elasticnet()`); python/timesift/fit.py:634-643 (`None` -> `[elasticnet()]`)
- stated-at: @p:vig.python.call-built
- warning: unstated: @p:vig.learners.start-elasticnet recommends the elastic net without saying it is the default; one clause there would connect the two.

### timesift-sift
- argument: timesift(sift)
- home: @param R/timesift.R:43-45 (`grains("auto")`)
- stated-at: @p:vig.representations.into-a-run
- warning: wording: "uses `grains("auto")` for the default learner" reads as if the run used the default `learners`, while the code passes an explicit named list. Say "for an elastic net at its defaults".

### timesift-x
- argument: timesift(x)
- home: @param R/timesift.R:31 (every numeric column but id and time)
- stated-at: @p:vig.representations.run-columns
- status: match

### timesift-n-inner
- argument: timesift(n_inner)
- home: R/timesift.R:138; python/timesift/fit.py:231; ?py.timesift [one-call]
- stated-at: @p:vig.prediction.procedure-rows; @p:vig.python.call-procedure; @p:vig.trouble.cost
- status: match; the R @param gives no value, and troubleshooting.Rmd never says the value is the default; link `?timesift`
- warning: unresolved-name: the spec says "`inner` left unset is 5, and `inner = NULL` / `inner=None`" (representation.md:2223-2224, @p:vig.contract.same-call) and "an inner map of `inner` folds" (:2216); the argument is `n_inner` in both languages.

### timesift-ensemble
- argument: timesift(ensemble); ensemble(method)
- home: `ensemble = TRUE` "the default stack"; `ensemble(method = "stack", ...)` first (R/ensemble.R:61); Python `True`
- stated-at: @p:vig.prediction.combiner; @p:vig.biomod2.stack; @p:vig.python.call-code (implied)
- status: match

### timesift-seed
- argument: timesift(seed)
- home: @param "Seed for the inner splits"
- stated-at: @p:vig.trouble.seeds
- status: match

### timesift-keep-fits
- argument: timesift(keep_fits)
- home: Python signature (`False`)
- stated-at: @p:vig.python.occlusion ("With the per-fold fits kept"; the tutorial passes True)
- status: match

### elasticnet-s
- argument: elasticnet(s)
- home: R/learners_linear.R:66, @param :43-46 (`"lambda.1se"`, glmnet's default); Python `'lambda.1se'`
- stated-at: @p:vig.learners.elasticnet-settings; @p:vig.learners.settings-readout; @p:vig.python.elasticnet
- status: match

### elasticnet-path
- argument: elasticnet(n_lambda, tol)
- home: signature; @param :48-50 for `tol` ("as close to the optimum as glmnet's own default does")
- stated-at: @p:vig.learners.elasticnet-settings; @p:vig.comparing-grains.run-setup ("a shorter penalty path and a looser stopping tolerance than its defaults")
- status: match, stated without the value
- warning: unstated: @p:vig.comparing-grains.run-setup also lowers `elasticnet(n_inner)` below its default without saying so.

### elasticnet-n-inner
- argument: elasticnet(n_inner)
- home: signature; @param "ten as in cv.glmnet()"
- stated-at: @p:vig.prediction.block-settings ("Its default of ten"); @p:vig.python.pin-code (printed)
- status: match
- warning: form-collision: printed beside the run's own `n_inner` with no sentence separating the two searches (see Terms inner-folds).

### elasticnet-squares
- argument: elasticnet(squares)
- home: signature (`TRUE`, adds the square of every column)
- stated-at: @p:vig.learners.reference-table ("every column and its square"); @p:vig.learners.elasticnet-settings ("with `squares = TRUE`, its square")
- status: match; the settings sentence does not say TRUE is the default, the table implies it
- warning: stale-count: comparing-grains.Rmd counts "365 columns", "53 columns" (:112, :180, :375; @p:vig.comparing-grains.run-result, @p:vig.comparing-grains.variance-bias, @p:vig.comparing-grains.anchors-result), leaving out the squared columns the default adds.

### linear-select
- argument: linear(select, terms)
- home: signature (`"both"`, `"power"`)
- stated-at: @p:vig.learners.start-elasticnet ("its two-way search", "each column and its square")
- status: match

### forest-trees-mtry
- argument: forest(trees, mtry)
- home: NULL in the signature; values under presence-absence in the help page Details (R/learners_forest.R:23, :92-93), not in @param
- stated-at: @p:vig.learners.forest-boost-maxent-settings
- status: match

### forest-balance
- argument: forest(balance)
- home: signature (`FALSE`)
- stated-at: @p:vig.learners.forest-boost-maxent-settings; @p:vig.biomod2.mapping-table (`RFd` = `balance = TRUE`)
- status: match

### boosting-method
- argument: boosting(method)
- home: signature (`"gbm"` first)
- stated-at: @p:vig.learners.forest-boost-maxent-settings; @p:vig.biomod2.mapping-table
- status: match

### maxent-classes
- argument: maxent(classes)
- home: @param R/learners_maxnet.R:49 (NULL, maxnet's choice by presence count)
- stated-at: @p:vig.learners.forest-boost-maxent-settings
- status: match

### maxent-formulation
- argument: maxent(formulation)
- home: R/learners_maxnet.R:75 (`"background"` first)
- stated-at: @p:vig.biomod2.mapping-table; @p:vig.biomod2.differences
- status: match

### additive-k
- argument: additive(k)
- home: R/learners_additive.R:52; @param gives no default
- stated-at: @p:vig.learners.additive-refuse; @p:vig.biomod2.mapping-table
- status: match; @param should name the default

### envelope-quantile
- argument: envelope(quantile)
- home: signature, "biomod2's default is `0.025`"
- stated-at: @p:vig.biomod2.mapping-table; @p:vig.biomod2.fixed-grain
- status: value match at the mapping table
- warning: default-mismatch: @p:vig.biomod2.fixed-grain wording "one presence in twenty at each end" (see Quantities bm-fixed-grain-counts).

### tree-prune
- argument: tree(prune)
- home: signature (`"se_sum"`, R/learners_tree.R:59); @param lists values, no default named
- stated-at: @p:vig.biomod2.mapping-table ("`"se_sum"` is biomod2's pruning")
- status: match; @param should name the default

### tree-shrink
- argument: tree(shrink)
- home: signature; @param "rpart's default is `1`" (R/learners_tree.R:46-47)
- stated-at: @p:vig.biomod2.count-head (`tree(shrink = 0)` turns it off)
- status: match

### tree-preset
- argument: tree(preset = "bigboss")
- home: R/learners_tree.R:107, :23-24
- stated-at: @p:vig.biomod2.presets
- status: match

### perceptron-hidden
- argument: perceptron(hidden)
- home: settings R/learners_perceptron.R:142; @param "Hidden units." gives no default
- stated-at: @p:vig.biomod2.ann ("two hidden units, which is the default")
- status: match; @param silent

### perceptron-standardise
- argument: perceptron(standardise)
- home: signature (`FALSE`)
- stated-at: @p:vig.learners.limits-list ("unless `standardise = TRUE`"); @p:vig.biomod2.ann ("Neither nnet nor `perceptron()` scales the columns")
- status: match

### mlp-architecture
- argument: mlp(hidden, dropout, activation)
- home: signature (`c(512L, 256L)`, `0.3`, `"relu"`)
- stated-at: @p:vig.biomod2.dnn-default (biomod2's default DNN given with explicit values)
- warning: unstated: the article never says `mlp()`'s own defaults differ from cito's; a reader writing `DNN = mlp()` gets 512/256 relu, not cito's 50/50 selu.

### train-control-batch-size
- argument: train_control(batch_size)
- home: signature
- stated-at: @p:vig.biomod2.dnn-diffs ("`batch_size` counts rows"); @p:vig.python.control-code (printed)
- status: no value stated; fine

### grain-matrix-year-start
- argument: grain_matrix(year_start); grain(); multigrain(); grains()
- home: signature R/grain_matrix.R:140 (`"01-01"`); @param R/grain_matrix.R:19
- stated-at: @p:vig.representations.year-start ("defaults to the calendar year"); @p:vig.trouble.lang-input
- status: match

### grain-matrix-partial
- argument: grain_matrix(partial)
- home: R/grain_matrix.R:141 (`"keep"`); @param :21-23; Python `'keep'`
- stated-at: @p:vig.representations.keep-or-drop; @p:vig.trouble.partial-run; @p:vig.trouble.partial-drop
- status: match; `timesift()` hard-codes "keep" (R/timesift.R:197), as @p:vig.representations.keep-or-drop and @p:vig.trouble.partial-run say
- warning: unstated: python.Rmd shows the partial December bin it keeps at @p:vig.python.calendar-code without explaining it.

### grain-matrix-stats
- argument: grain_matrix(stats); grain(stats); lookback(stats)
- home: signature (`"mean"`; Python `('mean',)`)
- stated-at: @p:vig.representations.choosing-stats ("`mean` is the default"); @p:vig.python.position-code (relied on)
- status: match

### multigrain-grains
- argument: multigrain(grains)
- home: R/representation.R:63 (NULL, every grain the record supports)
- stated-at: @p:vig.representations.four-constructors
- status: match

### lookback-lag
- argument: lookback(lag)
- home: signature (`'0 days'`)
- stated-at: @p:vig.python.reps-sets ("ending at each target's own instant")
- status: match for the default only (see Terms lookback)

### calendar-channels-cycles
- argument: calendar_channels(cycles)
- home: R/channels.R:41 (`"year"`); @param :21 "`"year"`, `"day"` or both, in the order given"; Python `('year',)`
- stated-at: @p:vig.representations.daily-cycle; @p:vig.python.position
- warning: default-mismatch: representations.Rmd:490 "`cycles = "day"` adds the place in the daily cycle as a second pair"; under the signature `cycles = "day"` replaces the year pair; the chunk below it correctly passes `c("year", "day")`. Write "`cycles = c("year", "day")` adds ...".

### positive-weights-cap
- argument: positive_weights(cap)
- home: R/response.R:145; @param :133 does not state it; Python `positive_weights(cap=50.0)` (verified)
- stated-at: @p:vig.learners.register-response; @p:vig.contract.same-call; ?py.positive_weights [extending]
- status: match; the @param line should carry the value since the article states it
- warning: unstated: the floor of one on a presence's weight (`pmax(..., 1)`, R/response.R:153) is stated nowhere in the spec (representation.md:2128).

### cv-strata
- argument: cv(strata, by)
- home: signature (`by = NULL` -> richness)
- stated-at: @p:vig.comparing-grains.fold-map ("balanced on richness"); @p:vig.prediction.constructors ("within strata of responses present")
- status: match

### cv-v
- argument: cv(v)
- home: signature
- stated-at: none (every chunk passes `v`)
- status: no prose claim; `block_cv(v)` and `env_cv(v)` default lower than `cv()`'s, also never stated, fine

### cv-repeats
- argument: cv(repeats); grouped_cv(repeats)
- home: @param repeats (`seed`, `seed + 1`, ...)
- stated-at: @p:vig.prediction.constructors ("draws the split again under the next seed")
- status: match

### simulate-records-year-start
- argument: simulate_records(year_start)
- home: signature (`"09-01"`)
- stated-at: @p:vig.comparing-grains.run-setup (the run's `year_start` "matches the boundary the simulation located its bins with")
- status: match; the article never says the value is the simulator's default

### response-curve-fixed
- argument: response_curve(fixed)
- home: signature (`c("mean", ...)`)
- stated-at: @p:vig.comparing-grains.curve-method ("every other is held at its mean"); @p:vig.prediction.curve-method ("at its `fixed` summary")
- status: match

### occlusion-permutations
- argument: occlusion(permutations)
- home: R ladder method signature; python/timesift/report.py:134, hidden behind `occlusion(x, *args, **kwargs)` in the reference
- stated-at: none (passed explicitly at @p:vig.comparing-grains.occlusion-code, @p:vig.comparing-grains.channel-code, @p:vig.python.occlusion-code)
- warning: default-undocumented: ?py.occlusion does not show the default (see Placement py-occlusion-arguments).

### occlusion-metric
- argument: occlusion(metric)
- home: @param (the metric the fit was scored under)
- stated-at: @p:vig.contract.same-call
- status: match

### decision-threshold-perc
- argument: decision_threshold(perc)
- home: signature; @param "0.9 by default"; R/metrics.R:127
- stated-at: @p:vig.prediction.rules; @p:vig.biomod2.rules ("keeps `perc` of the presences", no value)
- status: match

### threshold-rule
- argument: predict(rule); decision_threshold(rule); table_metric(rule)
- home: R/timesift.R:749 (`rule = c("youden", ...)`); R/metrics_table.R:65; Python `'youden'`
- stated-at: @p:vig.contract.same-call; @p:vig.biomod2.one-cut; @p:vig.python.binary
- status: match in R
- warning: enum-mismatch: @p:vig.python.binary lists youden, kappa and prevalence without naming the default; `THRESHOLD_RULES` also has `"mpa"` (python/timesift/metrics.py:14).

### kappa-score-rule
- argument: kappa_score(rule)
- home: R/metrics.R:143 (`"prevalence"` first)
- stated-at: none in biomod2.Rmd
- warning: claim-scope: @p:vig.biomod2.metric-table maps `KAPPA` to `kappa_score()` without saying its cut is the prevalence rule, while @p:vig.biomod2.one-cut (biomod2.Rmd:310-313) says all statistics share the youden cut.

### ensemble-rule
- argument: ensemble(rule), for committee averaging
- home: NULL; @param "`"youden"` by default"
- stated-at: @p:vig.prediction.methods-read; @p:vig.biomod2.ensemble-table
- status: match

### ensemble-decay
- argument: ensemble(decay)
- home: NULL -> `"proportional"` (@param); a number weights by rank
- stated-at: @p:vig.prediction.methods-list ("`decay` to weight by rank as biomod2's EMwmean.decay"); @p:vig.biomod2.ensemble-table
- status: match; the prose implies `decay` is what turns rank weighting on, and the score-proportional default could be named

### predict-candidate
- argument: predict(candidate); Timesift.predict(candidate)
- home: signature (`"ensemble"`)
- stated-at: @p:vig.prediction.candidate-arg; @p:vig.python.predict
- status: match

### project-chunk
- argument: project(chunk)
- home: R/project.R:50; @param chunk gives no value
- stated-at: @p:vig.prediction.project-read
- status: match; @param could state it

### pseudo-absences-quantile
- argument: pseudo_absences(quantile)
- home: signature
- stated-at: @p:vig.prediction.pseudo-admits (a value passed, "middle 80%")
- status: not stated as default; consistent

### tss-inflation-replicates
- argument: tss_inflation(skill, replicates)
- home: signature
- stated-at: none (passed explicitly at @p:vig.comparing-grains.inflation-code, @p:vig.biomod2.inflation, @p:vig.python.inflation-code)
- status: fine

### select-grain-rule
- argument: select_grain(choose); the selection rule
- home: @param choose; `"argmax"` the default rule (representation.md:2164)
- stated-at: @p:vig.comparing-grains.coarsest-rule (the coarsest-adequate rule, passed); @p:vig.contract.same-call
- status: match

### learner-multi-reads
- argument: Learner(multi); Learner(reads)
- home: python/timesift/learners.py:53-54 (`"separate"`, `"tabular"`)
- stated-at: @p:vig.python.multi (multi); none for reads
- warning: unstated: @p:vig.python.own does not say `reads` defaults to tabular; a reader writing a sequence learner is not told to set `reads="sequence"`.

### register-tuning-overwrite
- argument: register_tuning(overwrite)
- home: signature (`FALSE`)
- stated-at: @p:vig.learners.register-tuning-readout
- status: match

### python-extras
- argument: pip extras of the wheel
- home: pyproject.toml (`torch`, `contrasts = ["scipy>=1.15"]`, `plot`, `grid`)
- stated-at: @p:vig.python.contrasts; @p:vig.python.install-py; every Python reference page
- warning: incomplete: the install line lists `torch,contrasts,plot` and leaves out `grid` (xarray), which `project()` and `range_change()` need.

## Placement

### install-in-articles
- at: @p:vig.representations.install; @p:vig.learners.install; @p:vig.comparing-grains.install; @p:vig.prediction.install; @p:vig.biomod2.install; @p:vig.trouble.install; @p:vig.contract.install
- detail: the full R and Python installation chunks
- needed-for: none
- move-to: Get started only (`docs-articles`: Installation in Get started only); each opening links Get started and names only the extra setup (torch for the encoder section of learners.Rmd)

### r-install-in-python
- at: @p:vig.python.install-r, and the Installation section of all twelve python-*.Rmd pages
- detail: the R `install.packages()` / `pak::pak()` chunk, and twelve copies of the install block on the reference pages
- needed-for: none in a Python article or a reference page
- move-to: python.Rmd alone for Python; R install lives in the R Get started; the reference index may link it

### pandas-not-installed
- at: @p:vig.python.install-py
- detail: the tutorial imports pandas in its first chunk; `pip install timesift` does not install it (pyproject: pandas only in `test`)
- needed-for: @p:vig.python.tables-code and every later chunk
- move-to: add `pip install pandas` (or a note) to the install chunk

### roadmap-paragraphs
- at: @p:vig.representations.roadmap; @p:vig.learners.roadmap
- detail: section inventories of eleven and six items
- needed-for: none (the table of contents carries it)
- move-to: delete; keep the opening's purpose sentence, plus "ends with" where it says where the article lands

### fold-artifacts
- at: @p:vig.representations.artifacts-cross; @p:vig.representations.artifacts-code; @p:vig.representations.read-folds-by-name
- detail: write_folds/read_folds format, CSV, LF, C collation, alignment by name
- needed-for: A6, not A1
- move-to: vignettes/articles/contract.Rmd ("The representation contract"), with a one-sentence link from the digest section

### procedure-rows-explained
- at: @p:vig.learners.summary-readout (sentences 3-5)
- detail: how `selected` and `ensemble` are chosen inside outer training folds and why the candidate mean is optimistic
- needed-for: A2, home comparing-grains.Rmd
- move-to: keep one sentence ("the procedure rows are the scores to report") and link "Finding the grain"

### schrankogel-network-result
- at: @p:vig.learners.encoders-when
- detail: the study's fully connected network level with the penalised model on the hand-built features
- needed-for: the encoder recommendation, so it can stay; the numbers are a copy of the README (see Quantities mlp-vs-penalised)
- move-to: keep, link inst/reproduce or the paper as the source

### cg-build-speed
- at: @p:vig.comparing-grains.run-setup ("to keep this article quick to build")
- detail: why the elastic net settings are loosened
- needed-for: the reader's choice of settings, partly
- move-to: keep, reworded to the reader's terms ("faster settings than the defaults; keep the defaults for a study")

### cg-tss-section
- at: @p:vig.comparing-grains.scores through @p:vig.comparing-grains.rare-result
- detail: the full treatment of TSS inflation and `implied_skill()`
- needed-for: A5
- move-to: either declare comparing-grains the A5 home in story.md (it is, under Open), or move to its own article and leave one paragraph here

### pr-split-section
- at: @p:vig.prediction.split-why through @p:vig.prediction.one-denominator
- detail: resampling constructors, fold maps and the scorable-cell mask
- needed-for: A2 (whose home is comparing-grains), and the block run later in the same article
- move-to: keep the constructors (spatial folds belong with maps); move the mask definition to the A2 home and point to it

### pr-timesift-report
- at: @p:vig.prediction.report-shape ("documented under the topic timesift_report")
- detail: a help-topic name in prose without a link
- needed-for: the reader looking up the report's fields
- move-to: `?timesift_report` as a code span pkgdown autolinks

### torch-message-provenance
- at: @p:vig.trouble.torch
- detail: "quoted from the source here, since the machine this article was built on has torch installed"
- needed-for: none
- move-to: delete (process narration); keep the message

### no-fallback-rationale
- at: @p:vig.trouble.torch ("because two code paths would drift apart")
- detail: the package's internal design rule
- needed-for: none for the reader
- move-to: CLAUDE.md / contributor docs; the reader needs only "there is no fallback"

### penalised-distance
- at: @p:vig.trouble.lang-limits
- detail: why the elastic net is pinned as a distance
- needed-for: A6, home contract.Rmd
- move-to: one sentence and the link to [The representation contract](contract.html)

### python-helpers
- at: @p:vig.contract.one-language
- detail: R's unexported helpers by name (`.as_folds()`, `.as_response()`, ...)
- needed-for: none for a user; contributor-level
- move-to: keep in the spec (it is normative), but the rendered article could stop at the Python names

### fixtures-subsections
- at: @p:vig.contract.sections
- detail: thirteen "The fixtures" and thirteen "How exactly" subsections
- needed-for: implementers
- move-to: name each by its core ("The tree's fixtures") or move the "How exactly" parts to an appendix page

### internal-plumbing-in-reference
- at: ?py.select_columns, ?py.column_names, ?py.n_targets, ?py.target_labels [one-call]; ?py.as_sift, ?py.expand_sift, ?py.build_representation [representations]; ?py.as_resampling, ?py.resolve_folds, ?py.align_folds [split]; ?py.timesift_set [arrays]; ?py.resolve_metric [extending]
- detail: coercion and alignment helpers documented beside user verbs, several in developer terms ("One coercion, so no caller repeats the three cases"; "reaches the fitting path")
- needed-for: none for a user; they are exported (`__all__`), so the generator documents them
- move-to: a trailing "Internals" section of the reference index, or drop from `__all__` if they are not meant as API (a design decision for the user)

### py-occlusion-arguments
- at: ?py.occlusion [scoring]
- detail: signature shown as `occlusion(x, *args, **kwargs)`; `candidate`, `over`, `substitute`, `metric`, `permutations`, `seed` and the returned keys (`weight`, `part`, `variable`) are undocumented, though the tutorial uses three of them
- needed-for: @p:vig.python.occlusion-code; @p:vig.python.channel-hold
- move-to: the docstring of the dispatcher (python/timesift/report.py:121) states the two call forms' arguments
