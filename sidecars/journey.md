# Journey: timesift

## DESCRIPTION
### desc.claim  [A1]
- idea: fits and compares representations
- job: names the package's category in one sentence: representations of time-varying data compared against a prediction target
- leaves: the reader knows what kind of package this is
### desc.path  [A1 A2]
- idea: two tables to a stack
- job: walks the one call from two tables through the candidate representations to scored, stacked predictions
- leaves: the reader knows the shape of a run
### desc.calendar  [A1]
- idea: real weeks and months
- job: states the calendar binning that separates a grain from a block of hours
- leaves: the reader knows bins follow the calendar
### desc.registries
- idea: registered, not hard-coded
- job: says learners, heads and metrics are registrations
- leaves: the reader knows it extends without a fork
### desc.baseline  [A3]
- idea: elastic net baseline, cited
- job: names the penalised baseline and its method paper
- leaves: the reader can cite the baseline
### desc.default  [A5]
- idea: presence-absence default, AUC
- job: states the shipped response and metric and cites TSS
- leaves: the reader knows the defaults fit species distribution modelling

## README
### readme.title  (section)
- idea: name, logo, break line
- job: names the subject in plain words before anything is defined
- leaves: the reader knows the domain: the time scale of a sensor record
### readme.hero  [A1]  (figure)
- idea: seven grains, their scores
- job: shows one plot's record at the study's seven grains beside the skill each gave, so the claim is seen before it is read
- leaves: the reader has seen skill peak at an intermediate grain
### readme.claim  [A1]
- idea: finds the best time scale
- job: says what the package does, with the logger example
- leaves: the reader knows the package's job
### readme.grain  [A1 A2]
- idea: reduction becomes an argument
- job: defines a grain by its mechanism and says what a run does with every grain, ending on the reader's three choices
- leaves: the reader has the core term and knows what they decide
### readme.supports  [A1 A2 A3 A5 A6]
- idea: five things it supports
- job: lists one line per headline arc so a skimmer finds their use case
- leaves: the reader has seen every arc once
### readme.trust  [A3 A6]
- idea: own cores, pinned
- job: says why the numbers can be trusted: each core pinned against its reference package, both languages against the same digests, the version fixing the defaults
- leaves: the reader knows what a methods section cites
### readme.install  (section)
- idea: CRAN, build tools, Python
- job: gets the package installed on each platform, including the compiler and torch setup
- leaves: the reader has a working install
### readme.usage.steps
- idea: three numbered steps
- job: names the two tables and the one call before the code
- leaves: the reader knows what they bring
### readme.usage.sim
- idea: a record with known answer
- job: says what `simulate_records()` draws so the example needs no data
- leaves: the reader can run the example as pasted
### readme.usage.code  [A2]  (code)
- idea: the one call, printed
- job: runs a four-grain comparison and prints the scored run
- leaves: the reader has seen what comes back
### readme.usage.input
- idea: what the tables hold
- job: describes the input columns, selection and the refusal of missing readings
- leaves: the reader can shape their own data
### readme.usage.candidate  [A2]
- idea: which rows to report
- job: defines a candidate and says why `selected` and `ensemble` are the numbers to report
- leaves: the reader reads the printout correctly
### readme.usage.predict  [A4]
- idea: predict and binary cut
- job: says how a fit predicts new rows and where the threshold comes from
- leaves: the reader can use the fit
### readme.usage.python  [A6]  (code)
- idea: same run in Python
- job: shows the Python call for the same run
- leaves: a Python reader has their first call
### readme.study  [A1]  (section)
- idea: the alpine study's findings
- job: states the four findings the package was built from and points at the reproduction
- leaves: the reader has evidence the grain matters on real data
### readme.grains  [A1]  (section)
- idea: grain sets, calendars, statistics
- job: says how grains are named, how the calendar binning works and how an extreme day differs from an extreme reading, linking the home article
- leaves: the reader knows what to pass to `sift`
### readme.choose  [A2 A4]  (section)
- idea: ladder, contrasts, ensemble, maps
- job: names the comparison and combination functions and links their home articles
- leaves: the reader knows where the next step lives
### readme.tss  [A5]  (section)
- idea: TSS inflation at thin presences
- job: states the inflation, its size on the study design and the two functions that handle it
- leaves: the reader knows why the default metric is AUC
### readme.docs  (section)
- idea: where to start
- job: routes to Get started and labels each article by its job
- leaves: the reader knows where to go next
### readme.footer  (section)
- idea: help, contributing, support, citation
- job: gives the channels, the contributor guide and the generated citation
- leaves: the reader can ask, contribute and cite

## ?timesift-package
### pkg.claim  [A1 A2]
- idea: finds the time scale
- job: states the claim and the one call inside R
- leaves: the reader knows the front door
### pkg.arrays  [A1]
- idea: arrays without fitting
- job: names the two array functions reachable alone
- leaves: the reader knows the lower layer exists
### pkg.default  [A5]
- idea: default response and metric
- job: states the shipped default
- leaves: the reader knows what a run assumes
### pkg.route
- idea: where to start
- job: routes to the Get started vignette and the site
- leaves: the reader knows where to read on

## python/README.md
### py.title  (section)
- idea: name, logo, break line
- job: as readme.title, with absolute image links for PyPI
- leaves: the reader knows the domain
### py.hero  [A1]  (figure)
- idea: seven grains, their scores
- job: as readme.hero
- leaves: as readme.hero
### py.claim  [A1]
- idea: finds the best time scale
- job: as readme.claim
- leaves: as readme.claim
### py.grain  [A1 A2]
- idea: reduction becomes an argument
- job: as readme.grain
- leaves: as readme.grain
### py.supports  [A1 A2 A3 A5 A6]
- idea: five things it supports
- job: as readme.supports, with the R package linked as the twin
- leaves: as readme.supports
### py.trust  [A3 A6]
- idea: own cores, pinned
- job: as readme.trust
- leaves: as readme.trust
### py.install  (section)
- idea: pip, wheels, extras
- job: installs the wheel, says when a compiler is needed and what each extra adds
- leaves: the reader has a working install
### py.usage.steps
- idea: three numbered steps
- job: names the two tables and the one call
- leaves: the reader knows what they bring
### py.usage.code  [A2]  (code)
- idea: the one call
- job: runs the four-grain comparison on a simulated record
- leaves: the reader has a scored run
### py.usage.output  [A2]
- idea: which rows to report
- job: says what the printout lists, defines a candidate, says which rows to report
- leaves: the reader reads the printout correctly
### py.usage.input
- idea: mappings and selections
- job: describes the input types, selection forms, refusal of gaps and zoned time columns
- leaves: the reader can shape their own data
### py.usage.predict  [A4]
- idea: predict and binary cut
- job: as readme.usage.predict
- leaves: the reader can use the fit
### py.study  [A1]  (section)
- idea: the alpine study, short
- job: states the headline study finding and links the reproduction
- leaves: the reader has evidence on real data
### py.languages  [A6]  (section)
- idea: same numbers, folds cross
- job: says what the two languages share and how a fold map crosses between them
- leaves: the reader can split work across languages
### py.docs  (section)
- idea: where to start
- job: routes to the Python Get started and reference, and to the R concept articles
- leaves: the reader knows where to go next
### py.footer  (section)
- idea: help, contributing, support, citation
- job: as readme.footer, citation by download link
- leaves: the reader can ask, contribute and cite

## Get started (vignettes/timesift.Rmd)
### vig.timesift.opening  [A1]
- idea: reduction made an argument
- job: states the problem and what the package does about it
- leaves: the reader knows why they are here
### vig.timesift.scope
- idea: the whole path, simulated
- job: says what this vignette covers
- leaves: the reader knows the route
### vig.timesift.install  (section)
- idea: CRAN, GitHub, PyPI
- job: installs both languages
- leaves: a working install
### vig.timesift.tables  (section)
- idea: targets and series
- job: defines the two tables and their link
- leaves: the reader knows the input shape
### vig.timesift.data  (code)
- idea: sixty plots simulated
- job: builds a record whose signal sits under hourly noise
- leaves: data to run on
### vig.timesift.noise
- idea: level buried in noise
- job: says what the simulated record hides
- leaves: the reader knows what a good grain should recover
### vig.timesift.call  [A2]  (code)
- idea: the one call
- job: runs three grains over five folds
- leaves: a scored run
### vig.timesift.report  [A2]
- idea: two parts of report
- job: says what was fitted and that the report has two parts
- leaves: the reader is ready to read the printout
### vig.timesift.candidates  [A2]
- idea: candidates are the comparison
- job: explains the candidate rows and why their best is not reportable
- leaves: the reader does not report the best candidate
### vig.timesift.procedure  [A2]
- idea: procedure rows are reportable
- job: explains the nested choosing and weighting behind `selected` and `ensemble`
- leaves: the reader knows which number to report
### vig.timesift.estimate  [A2]  (code)
- idea: estimate and selections
- job: shows the reported estimate and each fold's choice
- leaves: the reader sees the numbers to report
### vig.timesift.interval  [A2]
- idea: interval across responses
- job: states what the interval does not carry
- leaves: the reader knows its limit
### vig.timesift.static
- idea: static only when named
- job: says a target column is a predictor only through `static`
- leaves: the reader controls the predictors
### vig.timesift.candtable  (code)
- idea: candidate table
- job: shows the representation, bins and channels of each candidate
- leaves: the reader can inspect a candidate
### vig.timesift.channels
- idea: two channels per grain
- job: reads the candidate table
- leaves: the reader knows what static adds
### vig.timesift.plot  [A2]  (figure)
- idea: score against grain
- job: plots the ladder with the ensemble level
- leaves: the reader sees the comparison
### vig.timesift.predict  [A4]
- idea: predict rebuilds representations
- job: says how prediction works and how to predict with the selected candidate
- leaves: the reader can predict
### vig.timesift.predictcode  [A4]  (code)
- idea: predicted probabilities
- job: predicts the targets
- leaves: probabilities in hand
### vig.timesift.binary  [A4]
- idea: thresholds from out-of-fold
- job: explains where the binary cut comes from and the three rules
- leaves: the reader can choose a threshold rule
### vig.timesift.binarycode  [A4]  (code)
- idea: thresholds and binary
- job: shows the cuts and the binary prediction
- leaves: presence-absence predictions
### vig.timesift.map  [A4]
- idea: a map is this
- job: connects a binary prediction to a map
- leaves: the reader sees the step to maps
### vig.timesift.representations  [A1]  (section)
- idea: settings, not data
- job: says a representation is settings and shows the four constructors
- leaves: the reader can name a representation
### vig.timesift.sets  [A1]
- idea: sets, multigrain, lookback
- job: explains the sets, `grains("auto")`, multigrain and lookback
- leaves: the reader knows each kind
### vig.timesift.pinned  [A1]  (code)
- idea: a learner pinned
- job: pins a learner to one grain
- leaves: the reader can restrict a learner
### vig.timesift.compat  [A3]  (section)
- idea: tabular versus sequence learners
- job: explains which representation a learner may be handed and when a pair is refused
- leaves: the reader understands a refusal
### vig.timesift.compatcode  (code)
- idea: a refused pair
- job: shows the error for a tabular learner on native
- leaves: the reader has seen the refusal
### vig.timesift.skipped
- idea: skipped in sets
- job: says a set skips the pair while a named one errors
- leaves: the reader knows the two behaviours
### vig.timesift.split  [A2]  (section)
- idea: one fold map
- job: explains the shared fold map and the fold builders
- leaves: the reader knows the split is shared
### vig.timesift.folds  (code)
- idea: fold map printed
- job: shows the folds
- leaves: the folds in view
### vig.timesift.mask  [A2]
- idea: scorable cells
- job: defines which cells admit a score, from response and folds alone
- leaves: the reader knows the mask
### vig.timesift.cells  (code)
- idea: mask printed
- job: shows the cells
- leaves: the mask in view
### vig.timesift.matched  [A2]
- idea: matched cells, one denominator
- job: says why a model-free mask makes comparisons fair
- leaves: the reader trusts the comparison
### vig.timesift.learners  [A3]  (section)
- idea: shipped learner kinds
- job: describes the tabular learners and the torch encoders and why a joint head helps rare responses
- leaves: the reader knows the learner families
### vig.timesift.baseline  [A3 A6]
- idea: one compiled baseline
- job: says the elastic net is one core for both languages and how to read it at a penalty
- leaves: the reader trusts the baseline
### vig.timesift.control  [A3]
- idea: architecture versus training
- job: splits constructor and `train_control()` and says how an override works
- leaves: the reader can set training
### vig.timesift.controlcode  (code)
- idea: control and override
- job: shows a run control and a learner override
- leaves: the reader has seen both
### vig.timesift.custom  [A3]
- idea: a learner of your own
- job: says a user learner is a fit/predict pair under the same folds, and how weights reach it
- leaves: the reader can write one
### vig.timesift.customcode  [A3]  (code)
- idea: two custom learners
- job: defines a nearest neighbour and a weighted GLM and runs them beside the elastic net
- leaves: a worked custom learner
### vig.timesift.combination  [A2]  (section)
- idea: combiner never sees models
- job: describes the combiners and why the ensemble is not scored with the full-data weights
- leaves: the reader knows how the stack is built
### vig.timesift.weightscode  (code)
- idea: ensemble weights
- job: shows the weights
- leaves: the weights in view
### vig.timesift.weightsread  [A2]
- idea: reading the weights
- job: says how to read a zero weight and the ensemble line
- leaves: the reader can interpret the stack
### vig.timesift.honest  [A5]  (section)
- idea: TSS inflation
- job: explains why TSS at its own threshold is inflated and that `tss_inflation()` measures it
- leaves: the reader knows the problem
### vig.timesift.inflationcode  [A5]  (code)
- idea: inflation for this design
- job: computes the inflation at two skills
- leaves: the inflation in view
### vig.timesift.implied  [A5]
- idea: expectation, inversion, AUC default
- job: explains `implied_skill()`, why paired TSS is not free of it, and the AUC default
- leaves: the reader knows why AUC is the default
### vig.timesift.arrays  [A1]  (section)
- idea: grain_matrix alone
- job: introduces the array without the fitting layer
- leaves: the reader can build an array
### vig.timesift.arraycode  (code)
- idea: weekly three-channel array
- job: builds a weekly array with the study's three statistics
- leaves: an array in view
### vig.timesift.calendar  [A1]
- idea: real months, real weeks
- job: states the calendar binning
- leaves: the reader trusts a bin
### vig.timesift.calendarcode  (code)
- idea: readings per month
- job: shows the varying bin counts
- leaves: the evidence for calendar bins
### vig.timesift.extremes  [A1]  (section)
- idea: extreme day versus reading
- job: defines the extreme statistics and how one cold hour reaches each
- leaves: the reader picks the right statistic
### vig.timesift.statscode  (code)
- idea: all seven statistics
- job: shows every statistic on one bin
- leaves: the difference in numbers
### vig.timesift.calchannels  [A1]
- idea: position in the year
- job: says why calendar channels are given to an encoder
- leaves: the reader knows what calendar channels are for
### vig.timesift.channelcode  (code)
- idea: channels bound
- job: binds the calendar channels
- leaves: their names in view
### vig.timesift.ladder  [A2]  (section)
- idea: ladder on its own
- job: introduces `grain_ladder()` on prebuilt arrays
- leaves: the reader can fit a ladder alone
### vig.timesift.laddercode  (code)
- idea: ladder summary
- job: fits and summarises the ladder
- leaves: a ladder in view
### vig.timesift.paired  [A2]
- idea: paired contrast
- job: explains how the paired contrast is taken and why six responses give a wide interval
- leaves: the reader can test one step
### vig.timesift.pairedcode  (code)
- idea: month versus day
- job: runs one contrast
- leaves: a contrast in view
### vig.timesift.mixed  [A2]
- idea: whole curve contrasts
- job: introduces `grain_contrasts()` for the whole curve
- leaves: the reader can test every grain
### vig.timesift.mixedcode  (code)
- idea: grain contrasts
- job: runs the mixed-model contrast
- leaves: the contrasts in view
### vig.timesift.occlusion  (section)
- idea: what was read
- job: explains occlusion of bins without refitting
- leaves: the reader can ask which bins mattered
### vig.timesift.occlusioncode  (code)
- idea: monthly occlusion
- job: runs occlusion on a monthly fit
- leaves: bin weights in view
### vig.timesift.channelocc  [A1]
- idea: occluding channels
- job: says channel occlusion asks what a statistic carries
- leaves: the reader can test the extremes
### vig.timesift.next  (section)
- idea: where next
- job: routes to the four topic articles, biomod2 and troubleshooting
- leaves: the reader knows where to go

## Representing a record

### vig.representations.three-decisions  [A1]
- idea: three decisions between record and array
- job: Establishes that a learner never reads the logged record, and that the reduction between them is three named arguments the package builds and records.
- leaves: The reader knows the three decisions (span per number, what it summarises, where spans begin) are arguments, and that the array remembers how it was built.

### vig.representations.roadmap  [A1]
- idea: section inventory
- job: Lists the article's topics in reading order.
- leaves: A list of eleven topics the table of contents already gives; nothing the next section needs.

### vig.representations.install  (section)
- idea: install R and Python
- job: Repeats the Get started installation chunks.
- leaves: Nothing new for a reader who reached this article from Get started.

### vig.representations.two-tables  [A1]
- idea: series and targets, joined by unit
- job: Defines the two input tables and the unit column that joins them, and introduces the simulated record used throughout.
- leaves: `series` is one row per reading, `targets` one row per thing to predict, and the example is thirty plots logged hourly.

### vig.representations.sim-record  [A1]  (code)
- idea: simulated hourly record
- job: Builds the example record and the four presence-absence targets.
- leaves: `series` and `targets` in the session, with a known seasonal and daily cycle.

### vig.representations.three-columns  [A1]
- idea: id, time, value
- job: Names the three columns `grain_matrix()` reads and the `[unit, bin, channel]` result.
- leaves: The reader can call `grain_matrix()` and expects an array whose attributes hold the binning.

### vig.representations.first-array  [A1]  (code)
- idea: first monthly array
- job: Shows the first array, its shape and its attribute names.
- leaves: A concrete array to read the next paragraph against.

### vig.representations.array-attributes  [A1]
- idea: what the attributes record
- job: Reads the shape and each binning attribute off the printed array.
- leaves: `bin_start`, `bin_end`, `bin_n` and `bin_partial` are known, and bin names are UTC ISO instants in both languages.

### vig.representations.ids-are-names  [A1 A6]
- idea: identifiers kept as names
- job: Establishes that a unit identifier is a string and rows sort under C collation, so the order is the same everywhere.
- leaves: A numeric id stays its digits; `"100000"` sorts before `"7"`.

### vig.representations.ids-code  [A1]  (code)
- idea: numeric ids demonstrated
- job: Shows the digit and order rule on two plots.
- leaves: Evidence for the previous paragraph.

### vig.representations.refused-ids  [A1]
- idea: what an identifier refuses
- job: States the input refusals (fractional id, missing id or time, duplicated instant) and why a duplicate matters.
- leaves: The reader knows a duplicated reading is refused because it would count twice.

### vig.representations.duplicate-code  [A1]  (code)
- idea: duplicate refused
- job: Shows the refusal message naming the first duplicate.
- leaves: What the error looks like.

### vig.representations.run-columns  [A1]
- idea: timesift() reads the same columns
- job: Connects the array call to the run's `id`, `time` and `x`, and how several reading columns become channels.
- leaves: Several value columns reach a learner as channels named column_statistic.

### vig.representations.time-zones  [A1]  (section)
- idea: bins follow the column's zone
- job: Establishes that binning follows the `tzone` attribute on the local clock, so a day is a wall-clock day.
- leaves: The reader expects a 23-reading day where summer time starts.

### vig.representations.zones-code  [A1]  (code)
- idea: Vienna against fixed offset
- job: Bins the same instants in a summer-time zone and a fixed offset.
- leaves: Two `bin_n` rows to compare and a native identity check.

### vig.representations.dst-readout  [A1]
- idea: the zone changes only the calendar
- job: Reads the two outputs and shows that the zone moves bins, never readings, and that `native` is zone-free.
- leaves: Changing `tzone` changes binning only; `native` is identical in either zone.

### vig.representations.fixed-offset-advice  [A1]
- idea: carry fixed-offset loggers fixed
- job: Recommends carrying a fixed-offset logger in its offset, with the Schrankogel record as the case.
- leaves: A rule for choosing the zone; first mention of the Schrankogel deposit, not introduced.

### vig.representations.grains  [A1]  (section)
- idea: a grain is the span of one bin
- job: Defines a grain and says several grains return a `timesift_set`.
- leaves: The reader can request a ladder of grains in one call.

### vig.representations.ladder-code  [A1]  (code)
- idea: seven-grain ladder
- job: Builds every named grain over the example record.
- leaves: `ladder`, reused through the rest of the article.

### vig.representations.ladder-counts  [A1]
- idea: bin count per grain
- job: Reads the bin count of each grain and states that coarse grains follow the calendar rather than a fixed hour count.
- leaves: The ladder's sizes and the calendar rule.

### vig.representations.week-start-code  [A1]  (code)
- idea: week starts and month counts
- job: Shows the first week starts and the readings per month.
- leaves: Evidence for the calendar rule.

### vig.representations.iso-week-calendar-month  [A1]
- idea: ISO weeks, calendar months
- job: Shows that a week is an ISO week and a month a calendar month, from the outputs.
- leaves: The first week begins before the record; month counts vary with month length.

### vig.representations.year-start  [A1]  (section)
- idea: year_start moves season and year
- job: Defines `year_start` and how seasons are counted from it.
- leaves: The reader can set a hydrological year.

### vig.representations.year-start-code  [A1]  (code)
- idea: September boundary
- job: Bins seasons and years on a September boundary beside the default.
- leaves: Two year splittings to compare.

### vig.representations.hydro-readout  [A1]
- idea: boundary is a system question
- job: Reads both splittings and ties the boundary choice to the system, with the Schrankogel winters as the case.
- leaves: The choice of boundary decides whether a winter falls in one bin or two.

### vig.representations.statistics  [A1]  (section)
- idea: two families of statistics
- job: Defines the seven statistics in their two families, reading-level and day-level.
- leaves: The vocabulary `mean`, `min`, `max`, `cold_day`, `warm_day`, `mean_daily_min`, `mean_daily_max`.

### vig.representations.stats-code  [A1]  (code)
- idea: seven statistics on one plot
- job: Computes all seven monthly statistics for three months of one plot.
- leaves: Numbers to read the ordering against.

### vig.representations.stats-order  [A1]
- idea: families order around the mean
- job: Reads how the families order themselves within a bin and introduces the figure.
- leaves: The reader expects reading-level extremes outside day-level ones.

### vig.representations.weekly-figure  [A1]  (figure)
- idea: weekly extremes over the record
- job: Shows the day-level pair lying inside the reading-level pair across the whole record.
- leaves: A visual of the two families.

### vig.representations.extreme-hour  [A1]  (section)
- idea: one reading can set min
- job: Explains why a single reading can be a fault and sets up the spike experiment.
- leaves: The reader knows the experiment: one hour made 20 degrees colder.

### vig.representations.spike-code  [A1]  (code)
- idea: one hour colder
- job: Reduces the clean and spiked copies with all seven statistics.
- leaves: Two rows to compare.

### vig.representations.spike-readout  [A1]
- idea: an extreme day dilutes a spike
- job: Shows how much each statistic follows the spike and why, ending on the distinction.
- leaves: An extreme day is a state held a whole day; an extreme reading can be one hour.

### vig.representations.schrankogel-extremes  [A1]
- idea: extreme days add skill
- job: Gives the study evidence that extreme days carry more than the mean, more at coarser grains.
- leaves: A reason to add `cold_day` and `warm_day` beyond robustness.

### vig.representations.day-level-floor  [A1]
- idea: day-level needs whole days
- job: States the day grain floor for day-level statistics and their coincidence at `day`.
- leaves: Refused at `native` and `halfday`; equal to the reading-level ones at `day`.

### vig.representations.day-only-code  [A1]  (code)
- idea: refusal and coincidence shown
- job: Shows the halfday refusal and the four statistics agreeing at day.
- leaves: Evidence for the floor rule.

### vig.representations.custom-calendar  [A1]  (section)
- idea: a calendar as a function
- job: Defines a supplied calendar and its usual use, astronomical seasons.
- leaves: The reader can write a function returning each reading's bin start.

### vig.representations.solstice-code  [A1]  (code)
- idea: solstice seasons
- job: Bins the record on a solstice calendar.
- leaves: A custom-grain array with counts and partial flags.

### vig.representations.custom-partial  [A1]
- idea: custom grain's partial rule
- job: Reads the custom array and states how its first and last bins are judged partial, and when day-level statistics are allowed on it.
- leaves: The record's end is a custom calendar's end; a calendar cutting a day refuses day-level statistics.

### vig.representations.schrankogel-seasons  [A1]
- idea: the study's seasons were custom
- job: Connects the supplied calendar to the reproduction driver.
- leaves: The published seasonal rung was built this way.

### vig.representations.gaps  [A1]  (section)
- idea: a gap is refused, not padded
- job: States the one-reading-per-bin rule and why padding is refused, and sets up the lost week.
- leaves: The reader expects a refusal for `p07`.

### vig.representations.gap-code  [A1]  (code)
- idea: lost week refused
- job: Shows the refusal on the gappy record.
- leaves: The message the reader will meet.

### vig.representations.coverage  [A1]
- idea: coverage() finds gaps
- job: Introduces `coverage()` as the table a refused record is read on.
- leaves: A count-only call taking no value column.

### vig.representations.coverage-code  [A1]  (code)
- idea: weekly and monthly coverage
- job: Lays out the counts at two grains.
- leaves: The empty cell visible at week, none at month.

### vig.representations.gap-by-grain  [A1]
- idea: coarser grain absorbs gap
- job: Reads why the gap shows at week and not at month.
- leaves: A gap's visibility depends on the grain.

### vig.representations.gap-month-code  [A1]  (code)
- idea: February counts compared
- job: Shows `p07`'s short February beside a neighbour.
- leaves: Evidence of a mean over fewer readings.

### vig.representations.three-ways-out  [A1]
- idea: three ways out of a gap
- job: Names the analyst's three options and states the package fills nothing.
- leaves: A decision list: coarser grain, drop units, cut the record.

### vig.representations.no-missing-bin  [A1]
- idea: consecutive bins one step apart
- job: States the second rule catching a bin no unit reaches, and its two exemptions.
- leaves: A convolution never reads non-adjacent months as neighbours.

### vig.representations.partial-bins  [A1]  (section)
- idea: partial bins sit at the ends
- job: Defines a partial bin and says where it can occur.
- leaves: Only the record's ends can be partial.

### vig.representations.partial-code  [A1]  (code)
- idea: partial flags per grain
- job: Shows partial flags at three grains and the drop option.
- leaves: Which bins are partial here.

### vig.representations.partial-where  [A1]
- idea: start date decides partial
- job: Reads which bins are partial from where the record starts and stops.
- leaves: `bin_partial` labels kept partial bins whatever `partial` is.

### vig.representations.keep-or-drop  [A1]
- idea: cost of keeping and dropping
- job: States the `partial` default, what each choice costs, and what `timesift()` does.
- leaves: Keeping biases day-level extremes towards the mean; dropping costs the ends; runs keep.

### vig.representations.representations-as-settings  [A1]  (section)
- idea: a representation before any record
- job: Defines a representation object as the description that configures a run and rebuilds at prediction.
- leaves: Four constructors exist.

### vig.representations.constructors-code  [A1]  (code)
- idea: the four constructors printed
- job: Prints one of each constructor.
- leaves: The printed form of each.

### vig.representations.four-constructors  [A1]
- idea: what each constructor takes
- job: Says what each constructor builds and takes.
- leaves: `native()`, `grain()`, `multigrain()`, `lookback()` and their arguments.

### vig.representations.sequence-or-block  [A1 A3]
- idea: sequence or block
- job: Defines the sequence/block distinction and that a learner's declaration decides which pairs run.
- leaves: Which constructors are sequences and which blocks.

### vig.representations.construction-checks  [A1]
- idea: checks before any record
- job: Lists what a constructor refuses without a record.
- leaves: Errors arrive at construction, not mid-run.

### vig.representations.sets  [A1]  (section)
- idea: grains() and lookbacks() build sets
- job: Introduces the set constructors that `sift` takes.
- leaves: Shared settings apply to every member.

### vig.representations.sets-code  [A1]  (code)
- idea: set constructors printed
- job: Prints several sets, including `"auto"` and `c()`.
- leaves: The forms a `sift` can take.

### vig.representations.auto-set  [A1]
- idea: grains("auto") resolved at the record
- job: Defines the `"auto"` set, its resolution time and its lack of a cap.
- leaves: `"auto"` includes `native` on any record with two readings.

### vig.representations.lookbacks  [A1]  (section)
- idea: a span before each target
- job: Defines the lookback for units with several targets through time and the shape of `at`.
- leaves: One row per target, read back from its own instant.

### vig.representations.lookback-code  [A1]  (code)
- idea: three visits, four weeks back
- job: Builds a lookback for three visits on two plots.
- leaves: A `[target, bin, channel]` array.

### vig.representations.lookback-readout  [A1]
- idea: bins named relative to anchor
- job: Reads the target order and relative bin names off the output.
- leaves: Columns align across targets with no shared date.

### vig.representations.fixed-lengths  [A1]
- idea: lookbacks measure fixed lengths
- job: States why a lookback month is 30 days and what day-level statistics need there.
- leaves: Use `grain_matrix()` when the calendar matters.

### vig.representations.lookback-edge-rule  [A1]
- idea: empty lookback bin refused
- job: States the one-reading rule for lookbacks.
- leaves: A lookback reaching before the record is refused.

### vig.representations.lookback-edge-code  [A1]  (code)
- idea: early anchor refused
- job: Shows the refusal naming target and interval.
- leaves: The message.

### vig.representations.lookback-edge-readout  [A1]
- idea: read bin_n near the ends
- job: Explains the refusal and recommends reading `bin_n` near either end.
- leaves: A habit for anchors close to the record's edges.

### vig.representations.calendar-position  [A1]  (section)
- idea: give back position in year
- job: Introduces `calendar_channels()` and `bind_channels()` for encoders that pool away timing.
- leaves: Sine and cosine of the year fraction, at each bin's midpoint.

### vig.representations.channels-code  [A1]  (code)
- idea: weekly position channels
- job: Builds and binds the position channels on a weekly array.
- leaves: A four-channel-plus array with a `position` attribute.

### vig.representations.channels-are-time-index  [A1]
- idea: position channels are a time index
- job: States what the channels are and are not, why the pair, and how encoders treat them.
- leaves: Marked channels reach torch encoders unstandardised.

### vig.representations.daily-cycle  [A1]
- idea: daily cycle on sub-day grains
- job: States when the daily pair is available.
- leaves: The day cycle is refused at a day or coarser.

### vig.representations.channels-day-code  [A1]  (code)
- idea: halfday with both cycles
- job: Computes year and day pairs at halfday.
- leaves: Four channel columns for one plot.

### vig.representations.halfday-sign  [A1]
- idea: halfdays alternate; lookback refused
- job: Reads the alternation and states that a lookback has no calendar position.
- leaves: Why the daily pair flips; lookbacks refuse position channels.

### vig.representations.channels-lookback-code  [A1]  (code)
- idea: lookback refused
- job: Shows the refusal.
- leaves: The message.

### vig.representations.bind-channels  [A1]
- idea: bind_channels() joins arrays
- job: States the binding rules and the second-variable use.
- leaves: A snow product joins temperature through the same call.

### vig.representations.tabular-layout  [A1 A3]  (section)
- idea: as.matrix() for tabular learners
- job: Gives the matrix layout a tabular learner reads.
- leaves: Column order is channel by channel, bin within.

### vig.representations.as-matrix-code  [A1]  (code)
- idea: season min/max matrix
- job: Shows the column count and names.
- leaves: Twelve `channel@bin` columns.

### vig.representations.static-columns  [A1 A3]
- idea: constant channels collapse
- job: States the column naming and that a constant channel becomes one column, and where to call it.
- leaves: A penalised fit sees elevation once; `as.matrix()` is the call for an own block learner.

### vig.representations.feature-matrix  [A1]
- idea: an already-reduced table as array
- job: Introduces `feature_matrix()` for published summaries.
- leaves: Units come from row names or a leading text column.

### vig.representations.feature-matrix-code  [A1]  (code)
- idea: bioclim table wrapped
- job: Wraps a two-feature table.
- leaves: A `[unit, feature, 1]` array.

### vig.representations.feature-table-baseline  [A1]
- idea: feature table is the baseline
- job: Reads the printout and positions a feature table as what a grain is measured against.
- leaves: A feature table competes as a candidate with no time axis.

### vig.representations.sets-of-arrays  [A1]  (section)
- idea: timesift_set holds built arrays
- job: Defines the set of built arrays and its three uses.
- leaves: Grain, custom calendar and feature table can share a set.

### vig.representations.set-code  [A1]  (code)
- idea: mixed set built
- job: Builds and subsets a three-member set.
- leaves: The printed set.

### vig.representations.set-same-targets  [A1]
- idea: members share targets
- job: States the same-targets rule and why the lookback cannot join.
- leaves: Row names are the check.

### vig.representations.set-mismatch-code  [A1]  (code)
- idea: mismatch refused
- job: Shows the refusal.
- leaves: The message.

### vig.representations.digests  [A6]  (section)
- idea: digest pins the specification
- job: Defines `digest_array()` and ties it to the spec and the fixtures in both languages.
- leaves: A binning change shows up as a changed digest.

### vig.representations.digest-code  [A6]  (code)
- idea: two months, one digest
- job: Digests the month built two ways.
- leaves: Two identical strings.

### vig.representations.digest-readout  [A6]
- idea: compare digests across machines
- job: Reads the identity and recommends digests for cross-machine checks.
- leaves: A quick equality check across sessions and languages.

### vig.representations.artifacts-cross  [A6]
- idea: folds cross as files
- job: States which artifacts cross the language boundary as files, and why.
- leaves: Folds, response and cells are CSV in C collation, byte-identical.

### vig.representations.artifacts-code  [A6]  (code)
- idea: fold file round trip
- job: Writes and reads a fold map.
- leaves: `TRUE` for the round trip.

### vig.representations.read-folds-by-name  [A6]
- idea: alignment by name
- job: States `read_folds()` aligns by name and its two edge rules.
- leaves: Missing units error, extras drop.

### vig.representations.into-a-run  [A1 A2]  (section)
- idea: sift and data= reach the run
- job: Shows the two places a run takes representations and introduces the example run.
- leaves: `sift` for open learners, `data =` to pin one.

### vig.representations.run-code  [A1]  (code)
- idea: auto plus pinned run
- job: Runs two elastic nets across `"auto"` and a pinned block.
- leaves: The candidate table.

### vig.representations.run-readout  [A1 A2 A3]
- idea: inapplicable pairs skipped
- job: Reads which grains `"auto"` resolved to, which pair was skipped and the pinned block's size.
- leaves: Mixed candidates are comparable because scored on the same folds and cells.

### vig.representations.target-time  [A1]
- idea: target_time needs lookbacks
- job: States the anchored-run rules: lookbacks only, no default spans, grouped split.
- leaves: How to configure a run with several targets per unit.

### vig.representations.run-lookback-code  [A1]  (code)
- idea: anchored run
- job: Runs two lookbacks on anchored targets.
- leaves: A candidate table no paragraph reads.

### vig.representations.which-grains  [A1]  (section)
- idea: bins a planned record gives
- job: Introduces a helper counting bins per grain for a planned deployment.
- leaves: `coverage()` as a planning tool.

### vig.representations.lengths-code  [A1]  (code)
- idea: three record lengths
- job: Counts bins for 120 days, one year and three years.
- leaves: A table of bins per grain.

### vig.representations.lengths-readout  [A1]
- idea: short records lose coarse grains
- job: Reads the table and ties the three-year row to the Schrankogel bins.
- leaves: Which grains a short record supports.

### vig.representations.two-bin-caution  [A1]
- idea: few bins, partial ends
- job: Warns that a grain in `"auto"` may still be thin, and how to check.
- leaves: Read `bin_n` and `bin_partial` before a long run.

### vig.representations.memory  [A1]  (section)
- idea: array size by grain
- job: Sets up the memory comparison on the study's size.
- leaves: Size is units by bins by channels doubles.

### vig.representations.memory-code  [A1]  (code)
- idea: megabytes compared
- job: Computes the two sizes.
- leaves: Two numbers.

### vig.representations.memory-readout  [A1]
- idea: native is expensive
- job: Reads the sizes and recommends a hand-named set on a long record.
- leaves: `"auto"` builds `native`; a named set is cheaper.

### vig.representations.choosing-stats  [A1]  (section)
- idea: which statistics when
- job: Gives the use of each statistic family and their cost in channels.
- leaves: Start at `mean`, add the extreme days, use `min`/`max` for single-reading events.

## Learners

### vig.learners.model-half  [A3]
- idea: learner is the model half
- job: Defines a learner against a representation and states that every learner goes through one interface.
- leaves: Adding a model never changes the fitting or scoring code; fifteen ship.

### vig.learners.roadmap  [A3]
- idea: section inventory
- job: Lists the article's topics.
- leaves: A list the table of contents already gives.

### vig.learners.simulated-record  [A3]
- idea: lagged weekly simulation
- job: Describes the example record and that its signal lives in each unit's departure from its own mean.
- leaves: The reader can predict that coarse reductions will lose the signal.

### vig.learners.install  (section)
- idea: install R and Python
- job: Repeats the Get started installation chunks.
- leaves: Nothing new.

### vig.learners.data-code  [A3]  (code)
- idea: simulate_records() call
- job: Builds the record and shows the presence counts.
- leaves: `sim` and `targets`.

### vig.learners.one-interface  [A3]  (section)
- idea: fit, predict, three declarations
- job: Defines the learner object and its declarations.
- leaves: What a learner declares; `NULL` means default at fit.

### vig.learners.print-code  [A3]  (code)
- idea: two learners printed
- job: Shows the printed declarations.
- leaves: The print format.

### vig.learners.fit-predict-contract  [A3]
- idea: whole response matrix in
- job: States what `fit` and `predict` exchange and that `c()` builds a set.
- leaves: Per-response learners still get the whole matrix.

### vig.learners.set-code  [A3]  (code)
- idea: a learner set
- job: Prints a set of three.
- leaves: The set form.

### vig.learners.catalogue  [A3]  (section)
- idea: shipped learners registered
- job: Introduces the registry and reads declarations off every constructor.
- leaves: A table of reads, multi, needs.

### vig.learners.catalogue-code  [A3]  (code)
- idea: catalogue built
- job: Builds the declarations table.
- leaves: The table.

### vig.learners.who-reads-what  [A3 A6]
- idea: tabular, joint, torch
- job: Reads the table: which read blocks, which fit jointly, which need torch, and that the rest run in the shared core.
- leaves: Only three need torch; the compiled ones fit the same model in both languages.

### vig.learners.checked-against  [A3]
- idea: written from the method, checked against reference
- job: States the relation to reference packages and links the biomod2 mapping.
- leaves: Each compiled learner's numbers are pinned in the fixtures.

### vig.learners.reference-table  [A3]  (table)
- idea: learner, model, reference, responses
- job: Maps each learner to its model, reference package and supported heads.
- leaves: The reader picks a learner by the model they know.

### vig.learners.every-head  [A3]
- idea: what "every head" means; hierarchical
- job: Glosses the responses column and describes the one learner without a reference.
- leaves: Loss families by head; `hierarchical()` uses INLA-style Laplace and an optional spatial field.

### vig.learners.what-handed  [A3]  (section)
- idea: tabular cannot read native
- job: States why a tabular learner refuses `native()` and that a pinned pair is refused early.
- leaves: Refused before the record is read.

### vig.learners.refuse-native-code  [A3]  (code)
- idea: pinned refusal
- job: Shows the refusal.
- leaves: The message.

### vig.learners.skip-in-sift  [A3]
- idea: skipped inside a sift
- job: States that inside a `sift` the pair is skipped with a note.
- leaves: A run goes on past inapplicable pairs.

### vig.learners.skip-native-code  [A3]  (code)
- idea: mixed sift run
- job: Shows the skipped pair in the candidate table.
- leaves: Status column.

### vig.learners.sequence-refusal-preview  [A3]
- idea: the opposite refusal
- job: Gives the native size and previews the sequence learner's opposite refusal.
- leaves: A forward pointer to the encoder section.

### vig.learners.several  [A2 A3]  (section)
- idea: named list, pinned and loosened
- job: Shows how to run several learners and settings side by side, and why two are adjusted.
- leaves: A named list reports each setting as its own candidate.

### vig.learners.run-code  [A3]  (code)
- idea: eight-learner run, timed
- job: Runs the eight learners at two grains with nested inner folds.
- leaves: `fit` and its run time.

### vig.learners.fit-count  [A2]
- idea: how many fits
- job: Gives the run time and how many times each candidate is fitted.
- leaves: The cost structure of nesting.

### vig.learners.summary-intro  [A2]
- idea: what summary() lists
- job: Says what the summary table holds.
- leaves: Candidates first, then the procedure.

### vig.learners.summary-code  [A2 A3]  (code)
- idea: run summary
- job: Prints the summary.
- leaves: Candidate means, wins, procedure rows.

### vig.learners.summary-readout  [A2 A3]
- idea: weekly wins; report procedure rows
- job: Reads the grain effect, the best candidate and why the procedure rows are the scores to report.
- leaves: The best candidate mean is optimistic; `selected` and `ensemble` score the choosing.

### vig.learners.candidate-table  [A3]
- idea: candidate table records build
- job: States what the candidate table records.
- leaves: Learner, grain, bins, status, chosen settings.

### vig.learners.candidates-code  [A3]  (code)
- idea: candidate table head
- job: Prints four rows.
- leaves: The table form.

### vig.learners.one-array  [A3]  (section)
- idea: fit_learner() without folds
- job: Introduces fitting a learner on one array and sets up the half split.
- leaves: Two weekly arrays over the same calendar weeks.

### vig.learners.one-array-code  [A3]  (code)
- idea: one elastic net fit
- job: Fits and prints one learner.
- leaves: `f`, `week_train`, `week_test`.

### vig.learners.unfitted-stopped  [A3]
- idea: unfitted and stopped responses
- job: Defines the two diagnostics and states bins are checked at predict.
- leaves: A rare response falls back to its prevalence; mismatched bins are refused.

### vig.learners.one-array-refuse-code  [A3]  (code)
- idea: month array refused
- job: Shows the refusal and builds monthly arrays used later.
- leaves: `month_train`, `month_test`.

### vig.learners.settings  [A3]  (section)
- idea: most-changed settings
- job: Scopes the section to the settings most often changed.
- leaves: The rest live in each help page.

### vig.learners.elasticnet-settings  [A3]
- idea: alpha, s, tol
- job: Explains the elastic net's mixing, read point and stopping tolerance.
- leaves: `lambda.1se` is glmnet's default; a looser `tol` is faster.

### vig.learners.forest-boost-maxent-settings  [A3]
- idea: forest · boosting · maxent settings
- job: Explains the settings of three different learners in one paragraph.
- leaves: Forest defaults and balancing, boosting's two methods, maxent's feature classes.

### vig.learners.held-out-helper  [A3]
- idea: one-split AUC helper
- job: Introduces the comparison function.
- leaves: AUC on the other half, averaged over responses.

### vig.learners.settings-code  [A3]  (code)
- idea: ten variants scored
- job: Scores ten settings on one split.
- leaves: A named AUC vector.

### vig.learners.settings-readout  [A3]
- idea: differences, then caution
- job: Reads the variants and warns that one split is noisy.
- leaves: Confirm a setting under the run's folds before acting on it.

### vig.learners.encoders  [A3]  (section)
- idea: architecture vs train_control()
- job: Establishes the split between constructor and `train_control()` and the override rule.
- leaves: One place defaults training; a learner overrides only what it names.

### vig.learners.control-code  [A3]  (code)
- idea: control and override printed
- job: Prints a control and an encoder with one override.
- leaves: The `(default)` marking.

### vig.learners.encoder-run-intro  [A3]
- idea: undertrained example run
- job: Reads the printouts and sets up a short CPU run.
- leaves: The reader expects undertrained networks.

### vig.learners.encoders-code  [A3]  (code)
- idea: mlp and cnn run
- job: Trains two encoders at weekly.
- leaves: A summary with joint responses.

### vig.learners.encoders-readout  [A3]
- idea: joint responses, reported once
- job: Reads the scores and the `joint` column.
- leaves: One network covers all responses.

### vig.learners.refuse-sequence  [A3]
- idea: sequence learner refuses a block
- job: Introduces the refusal promised earlier.
- leaves: Refused before the record is read.

### vig.learners.refuse-sequence-code  [A3]  (code)
- idea: cnn on multigrain refused
- job: Shows the refusal.
- leaves: The message.

### vig.learners.portable-fit  [A3]
- idea: fit travels across machines
- job: States a fitted encoder predicts on a machine without a GPU.
- leaves: `saveRDS()` round trips a fit.

### vig.learners.tuning  [A2 A3]  (section)
- idea: tune() nests a grid search
- job: Defines `tune()` and why its choice is honest inside a run.
- leaves: Each outer fold tunes on its own training units.

### vig.learners.tune-code  [A3]  (code)
- idea: forest grid
- job: Tunes a forest on the training half.
- leaves: The chosen setting and table.

### vig.learners.tune-readout  [A3]
- idea: chosen setting and where it shows
- job: Reads the choice and where a run records it.
- leaves: The `settings` column.

### vig.learners.registered-grids  [A3]
- idea: biomod2's grids registered
- job: States that grids default to registered ones, eight of which follow `BIOMOD_Tuning()`.
- leaves: `tune()` without a grid works for those learners.

### vig.learners.tunings-code  [A3]  (code)
- idea: registered grids listed
- job: Lists them.
- leaves: The registry.

### vig.learners.register-tuning  [A3]
- idea: add a grid
- job: Shows how to register a grid, fixed or data-dependent.
- leaves: The elastic net gets a mixing grid.

### vig.learners.register-tuning-code  [A3]  (code)
- idea: alpha grid tuned
- job: Registers and tunes.
- leaves: The table.

### vig.learners.register-tuning-readout  [A3]
- idea: lasso chosen; session scope
- job: Reads the choice and the registration's scope.
- leaves: Lasts for the session; `overwrite = TRUE` replaces.

### vig.learners.own-learner  [A3]  (section)
- idea: learner() hands what fit declares
- job: Defines the argument-driven hand-over of weights, head, control and group.
- leaves: A fit receives only what it names.

### vig.learners.own-intro  [A3]
- idea: channel-means GLM
- job: Describes the example learner.
- leaves: What the code will do.

### vig.learners.own-code  [A3]  (code)
- idea: channel_means constructor
- job: Defines the learner.
- leaves: A constructor.

### vig.learners.register-learner  [A3]
- idea: register by name; saved by name
- job: States what registration gives and its consequence for `readRDS()`.
- leaves: Re-register before predicting from a saved fit.

### vig.learners.register-learner-code  [A3]  (code)
- idea: own learner in a run
- job: Registers and runs it against the elastic net.
- leaves: A summary.

### vig.learners.own-readout  [A1 A3]
- idea: the reduction discarded the signal
- job: Explains why the channel mean scores below chance, from the simulation's design.
- leaves: A learner cannot recover what the reduction removed.

### vig.learners.responses  [A3]  (section)
- idea: the head defines response
- job: Defines the response head and lists the shipped heads.
- leaves: Four heads beside presence-absence.

### vig.learners.responses-code  [A3]  (code)
- idea: heads listed
- job: Lists them.
- leaves: The registry.

### vig.learners.counts  [A3]
- idea: counts under Poisson
- job: Sets up a count fit with no learner change.
- leaves: The forest fits counts as is.

### vig.learners.counts-code  [A3]  (code)
- idea: count forest
- job: Fits counts.
- leaves: Predictions.

### vig.learners.pa-only  [A3]
- idea: four learners are PA-only
- job: States the presence-absence restriction.
- leaves: Which learners refuse other heads.

### vig.learners.counts-refuse-code  [A3]  (code)
- idea: maxent count refused
- job: Shows the refusal.
- leaves: The message.

### vig.learners.register-response  [A3]
- idea: a head's specification
- job: Lists a head's fields and the allowed activations and losses, and the default presence weighting.
- leaves: Presences weighted by the absence ratio, capped.

### vig.learners.weights-code  [A3]  (code)
- idea: presence weights
- job: Shows the weight per response.
- leaves: Four weights.

### vig.learners.cap-two  [A3]
- idea: a cap of two
- job: Predicts the effect of a lower cap from the weights shown.
- leaves: Which responses change.

### vig.learners.register-response-code  [A3]  (code)
- idea: capped head registered
- job: Registers the head.
- leaves: `presence_absence_cap2`.

### vig.learners.register-metric  [A3]
- idea: a metric of one's own
- job: Defines a metric and the higher-is-better convention.
- leaves: Brier reversed.

### vig.learners.register-metric-code  [A3]  (code)
- idea: capped Brier run
- job: Runs under the new head and metric.
- leaves: A summary.

### vig.learners.capped-readout  [A3]
- idea: run reported by Brier
- job: Reads the result.
- leaves: Elastic net above forest.

### vig.learners.choosing  [A3]  (section)
- idea: cost depends on columns
- job: Sets up the timing of every tabular learner at weekly.
- leaves: Fifty-three columns, all units.

### vig.learners.timings-code  [A3]  (code)
- idea: fit times
- job: Times thirteen learners.
- leaves: Sorted seconds.

### vig.learners.additive-refuse  [A3]
- idea: additive too wide here
- job: Explains why `additive()` is missing.
- leaves: Coefficients exceed units at `k = 10`.

### vig.learners.additive-refuse-code  [A3]  (code)
- idea: refusal
- job: Shows it.
- leaves: The message.

### vig.learners.additive-month  [A3]
- idea: fits at monthly
- job: Moves to monthly.
- leaves: Twelve columns fit.

### vig.learners.additive-month-code  [A3]  (code)
- idea: k timings
- job: Times two `k`.
- leaves: Two seconds.

### vig.learners.additive-k-readout  [A3]
- idea: k drives cost
- job: Reads the two times.
- leaves: Smaller `k` is much cheaper.

### vig.learners.start-elasticnet  [A3]
- idea: start with elastic net; linear slowest
- job: Recommends the elastic net, reads its timing and explains why `linear()` is slowest and how to bound it.
- leaves: A starting learner and a way to make the stepwise search affordable.

### vig.learners.units-vs-columns  [A3]
- idea: units against columns
- job: Introduces the list of fit limits by learner.
- leaves: A frame for the list.

### vig.learners.limits-list  [A3]
- idea: per-learner fit limits
- job: Gives each learner's failure mode when columns grow against units.
- leaves: Which learners suit a fine grain on a small sample.

### vig.learners.encoders-when  [A3]
- idea: encoders need many responses
- job: States when the encoders pay off, with the study's result.
- leaves: Few units and responses favour the elastic net.

### vig.learners.wrong-choice  [A1 A3]
- idea: no learner recovers dropped signal
- job: Gathers the refusals and closes on why learners are compared across grains.
- leaves: The representation, not the learner, bounds what can be found.

Arc tags follow sidecars/story.md. `[none]` marks an entry that carries no story arc (see
findings-B.md, `story-gap`).

## Finding the grain (vignettes/articles/comparing-grains.Rmd)

### vig.comparing-grains.question  [A1 A2]
- idea: a planted grain makes a test
- job: Sets up the article's method: on field data the right grain is unknown, so a simulated record with a fixed answer turns every tool into something that can be checked.
- leaves: The reader knows the article tests the package against a known answer, and that `simulate_records()` provides it.

### vig.comparing-grains.roadmap  [A1 A2]
- idea: weekly plant, four grains, five readings
- job: Lays out the route: one planted record read by five tools, then two records at other grains and one with no signal.
- leaves: The reader can anticipate the sections and knows a null record closes the comparison.

### vig.comparing-grains.install  (section)
- idea: install from CRAN, GitHub, PyPI
- job: Repeats the install commands for both languages.
- leaves: Nothing the reader of an article needs; see findings (`install-outside-get-started`).

### vig.comparing-grains.known-answer  (section)

### vig.comparing-grains.simulator  [A1]
- idea: three parts plus a weighted driver
- job: Explains how the simulated record and its responses are made, so the reader can see why the lag mechanism puts the truth at the week.
- leaves: Readings are cycle plus offset plus anomaly plus noise; the lag driver is an exact linear combination of weekly means.

### vig.comparing-grains.simulate-call  [A1]  (code)
- idea: draw the lag record
- job: Draws the record the rest of the article reads and shows its anchors.
- leaves: A `sim` object with readings, responses, weights and per-variable anchor weeks.

### vig.comparing-grains.true-grain  [A1]
- idea: true grain defined by nesting
- job: Defines the true grain and explains why finer grains keep it and coarser grains lose part of it.
- leaves: Day holds the driver over more columns; month and season average part of it away; anchors index weekly bins.

### vig.comparing-grains.ceiling  [A1 A2]
- idea: driver AUC is a ceiling
- job: Sets the population ceiling the fitted scores are later read against, and says why no model reaches it.
- leaves: `auc` is the driver's own AUC; fitted models fall short because of noise and estimation.

### vig.comparing-grains.weights-lead  [A1]
- idea: first variable's four weeks
- job: Introduces the check of the first variable's planted stretch.
- leaves: The reader expects a date range.

### vig.comparing-grains.weights-code  [A1]  (code)
- idea: date range of the weights
- job: Shows the instants the first variable's driver reads.
- leaves: The planted stretch as dates.

### vig.comparing-grains.tables  [A1]
- idea: targets and readings tables
- job: Names the two tables `timesift()` takes, in the shape the package expects.
- leaves: One row per unit in `targets`; a long readings table.

### vig.comparing-grains.tables-code  [A1]  (code)
- idea: build targets, inspect readings
- job: Builds `targets` and shows the readings' columns.
- leaves: The inputs of the run.

### vig.comparing-grains.one-run  (section)

### vig.comparing-grains.run-setup  [A2]
- idea: run settings explained
- job: Says what the run does and why each non-default setting is there (fast elastic net, `year_start`, `keep_fits`).
- leaves: The reader knows which settings to change for real data and which serve later sections.

### vig.comparing-grains.run-code  [A2]  (code)
- idea: one run over four grains
- job: Runs the comparison the rest of the article reads.
- leaves: A fit with candidate and procedure rows.

### vig.comparing-grains.run-result  [A1 A2]
- idea: week recovered below the ceiling
- job: States that the run recovers the planted grain, ranks the other grains by how much of the driver they lose, and places the level against the ceiling.
- leaves: Week is best on every response; day, month and season fall in that order.

### vig.comparing-grains.procedure-rows  [A2]
- idea: nested procedure scores the choosing
- job: Explains how the procedure rows were computed and why they equal the weekly candidate here but not in general.
- leaves: The selected/ensemble rows score the choice; equality is a property of this run.

### vig.comparing-grains.plot-fit  [A2]  (figure)
- idea: candidate means by grain
- job: Shows the run's grain profile with the ensemble level.
- leaves: The peak at week, visually.

### vig.comparing-grains.estimate-lead  [A2]
- idea: estimate under every metric
- job: Introduces the estimate table and the four metrics a presence-absence study reads.
- leaves: The reader expects one row per metric.

### vig.comparing-grains.estimate-code  [A2]  (code)
- idea: selected arm, four metrics
- job: Reads the selected arm's estimate under four metrics with its interval.
- leaves: Score, interval, number of responses.

### vig.comparing-grains.interval-scope  [A2]
- idea: interval across responses only
- job: States what the interval covers and what it leaves out.
- leaves: The interval is the spread between responses, not their shared error.

### vig.comparing-grains.fold-map  [A2]
- idea: one map, mask before models
- job: Explains how the outer folds were dealt and that the scorable-cell mask is fixed before any fit.
- leaves: Folds balanced on richness; mask from response and fold map alone.

### vig.comparing-grains.folds-code  [A2]  (code)
- idea: folds and cells printed
- job: Shows the fold sizes and the cell table.
- leaves: Five equal folds; the cell table.

### vig.comparing-grains.all-scorable  [A2]
- idea: every cell scorable here
- job: Says why every cell is scorable at this prevalence and points to where rarity changes it.
- leaves: Scorability depends on presences per fold.

### vig.comparing-grains.ladder  (section)

### vig.comparing-grains.ladder-intro  [A1 A2]
- idea: ladder without the run
- job: Introduces `grain_ladder()` on prebuilt arrays and says it reproduces the run when given the run's folds.
- leaves: The ladder is the run's comparison reachable from arrays.

### vig.comparing-grains.ladder-code  [A1 A2]  (code)
- idea: build arrays, run ladder
- job: Builds the four arrays and runs the ladder on the run's folds.
- leaves: Bins per grain; the ladder table.

### vig.comparing-grains.ladder-matches  [A2]
- idea: ladder equals run
- job: Confirms the ladder reproduces the run's candidate scores and says why.
- leaves: Same arrays, learner, folds and mask give the same numbers.

### vig.comparing-grains.plot-ladder-lead  [A2]
- idea: what the ladder plot draws
- job: Tells the reader how to read the ladder plot.
- leaves: Means, intervals across responses, circled best grain.

### vig.comparing-grains.plot-ladder  [A2]  (figure)
- idea: ladder with intervals
- job: Shows the grain profile with its uncertainty.
- leaves: Rise to week, fall after.

### vig.comparing-grains.variance-bias  [A1]
- idea: variance left, bias right
- job: Explains the shape of the curve: fine grains pay in variance, coarse grains in bias.
- leaves: The best grain depends on sample size as well as on the mechanism.

### vig.comparing-grains.paired  (section)

### vig.comparing-grains.paired-method  [A2]
- idea: per-cell differences, responses as replicates
- job: Defines how `paired_contrast()` computes a difference, its interval and its p-value.
- leaves: Responses are the replicates; t interval; Wilcoxon signed-rank p.

### vig.comparing-grains.paired-code  [A2]  (code)
- idea: week against month and day
- job: Runs the two contrasts the ladder suggests.
- leaves: Two contrast rows.

### vig.comparing-grains.paired-result  [A2]
- idea: week wins; p at its floor
- job: States that week beats both neighbours with intervals excluding zero, and that the equal p-values are the test's floor.
- leaves: The p-value floor depends only on the number of responses.

### vig.comparing-grains.p-floor-code  [A2]  (code)
- idea: smallest p by response count
- job: Tabulates the smallest attainable p for three to ten responses.
- leaves: The floor as a function of response count.

### vig.comparing-grains.few-responses  [A2]
- idea: few responses, no test
- job: States the consequence for small studies and sets up the three-response example.
- leaves: Five responses or fewer cannot reach the conventional level.

### vig.comparing-grains.three-code  [A2]  (code)
- idea: contrast on three responses
- job: Repeats the week-month contrast on three responses.
- leaves: A wider interval.

### vig.comparing-grains.three-result  [A2]
- idea: same sign, wider interval
- job: Explains why the interval widens and where the p-value floor sits with three responses.
- leaves: Few responses widen the interval through the t multiplier.

### vig.comparing-grains.arm-names  [A2]
- idea: arms named whole
- job: States that a contrast takes a full arm and why a bare learner is refused.
- leaves: Arms are `"grain|learner"`; choosing a learner's best grain on the scored folds is refused.

### vig.comparing-grains.whole-curve  (section)

### vig.comparing-grains.mixed-method  [A2]
- idea: mixed model against best grain
- job: Defines `grain_contrasts()`: the model, the many-to-one comparison and what the random effects absorb.
- leaves: Each grain compared with the best, within response and fold.

### vig.comparing-grains.mixed-code  [A2]  (code)
- idea: grain contrasts table
- job: Fits the mixed-model contrast when its three packages are installed.
- leaves: Adjusted intervals per grain.

### vig.comparing-grains.mixed-read  [A2]
- idea: pooled variance, chosen reference
- job: Reads the table, explains the singular-fit message and the equal widths, and says the table belongs beside the paired contrasts because its reference was chosen.
- leaves: The mixed-model and paired intervals answer differently; read both.

### vig.comparing-grains.choosing  (section)

### vig.comparing-grains.select-method  [A2]
- idea: choice moved inside training
- job: Explains `select_grain()`'s nested procedure and what its estimate covers.
- leaves: The estimate is of the procedure, choice included.

### vig.comparing-grains.select-code  [A2]  (code)
- idea: nested selection on the arrays
- job: Runs the nested selection and prints each fold's choice.
- leaves: Per-fold choices and inner scores.

### vig.comparing-grains.select-result  [A2]
- idea: every fold chose week
- job: States that every fold chose week and the procedure equals the run's estimate, and why inner scores sit lower.
- leaves: Inner scores rank, they are not the level.

### vig.comparing-grains.plot-select-lead  [A2]
- idea: how to read the selection plot
- job: Tells the reader what the selection plot draws.
- leaves: One line per fold, circle at the choice.

### vig.comparing-grains.plot-select  [A2]  (figure)
- idea: inner scores per fold
- job: Shows the stability of the choice across folds.
- leaves: All circles on week.

### vig.comparing-grains.wandering  [A2]
- idea: wandering circles mean flat grid
- job: Gives the reader the reading rule for an unstable selection.
- leaves: Unstable choice signals a flat grid.

### vig.comparing-grains.null  (section)

### vig.comparing-grains.null-setup  [A2]
- idea: no signal, AUC near half
- job: Sets up the null record and what an honest tool should report on it.
- leaves: Every AUC should sit near 0.5.

### vig.comparing-grains.null-code  [A2]  (code)
- idea: null ladder
- job: Runs the ladder on the null record.
- leaves: A ladder with a nominal best grain.

### vig.comparing-grains.null-best  [A2]
- idea: best of four is chance
- job: Shows that a best grain appears anyway and why reading its score as the level overstates skill.
- leaves: The best mean of several candidates is biased upward.

### vig.comparing-grains.null-select-code  [A2]  (code)
- idea: nested selection on null
- job: Runs the nested selection on the null record.
- leaves: The procedure's estimate and interval.

### vig.comparing-grains.null-select  [A2]
- idea: procedure interval contains half
- job: States that the procedure's interval contains 0.5 and that the gap to the best candidate grows with the search.
- leaves: Report the procedure, not the best candidate.

### vig.comparing-grains.coarsest-rule  [A1 A2]
- idea: coarsest adequate rule
- job: Defines `choose = "coarsest_adequate"` and what it buys on a flat profile.
- leaves: The one-standard-error rule picks the least storage without measured loss.

### vig.comparing-grains.coarsest-code  [A2]  (code)
- idea: coarsest adequate on null
- job: Runs the rule on the null record.
- leaves: Per-fold choices under the rule.

### vig.comparing-grains.coarsest-read  [A2]
- idea: choices scatter on flat profile
- job: Reads the scattered choices as the signature of a record with nothing to find.
- leaves: A fold-to-fold change of choice is what a flat profile looks like.

### vig.comparing-grains.moves  (section)

### vig.comparing-grains.other-mechanisms  [A1]
- idea: season and event mechanisms
- job: Introduces two records whose truth sits at the season and at the day.
- leaves: Which grains nest in each truth (see findings: week does not nest in a season).

### vig.comparing-grains.mechanisms-code  [A1]  (code)
- idea: two more ladders
- job: Builds and runs a ladder for each mechanism.
- leaves: Two ladder summaries.

### vig.comparing-grains.mechanisms-result  [A1]
- idea: peaks follow the truth
- job: States that each ladder peaks at its planted grain and explains the shape of each fall.
- leaves: The ladder tracks the truth, not a preference of the method.

### vig.comparing-grains.plot-event  [A1]  (figure)
- idea: event ladder
- job: Shows the event record's ladder peaking at day.
- leaves: Visual confirmation for the day truth.

### vig.comparing-grains.what-read  (section)

### vig.comparing-grains.occlusion-method  [none]
- idea: hold one bin back
- job: Defines `occlusion()`: a bin's weight is the fall in score when it is permuted in the held-out units, without refitting.
- leaves: Occlusion weight defined.

### vig.comparing-grains.occlusion-code  [none]  (code)
- idea: weekly bin weights
- job: Computes the bin weights of the weekly candidate.
- leaves: A weight per bin and variable.

### vig.comparing-grains.anchors-lead  [none]
- idea: check against planted anchors
- job: Sets up the response-by-response check and names the two anchor weeks.
- leaves: The weeks the profile should single out.

### vig.comparing-grains.anchors-code  [none]  (code)
- idea: heaviest bins per variable
- job: Lists the four heaviest bins of the first two variables.
- leaves: The heaviest bins.

### vig.comparing-grains.anchors-result  [none]
- idea: anchor week heaviest
- job: States that the anchor week is heaviest for both variables, and reads the off-stretch weights as the fit picking up structure the mechanism lacks.
- leaves: Occlusion recovers the planted stretch, with some spill.

### vig.comparing-grains.channel-lead  [A1]
- idea: occlude a whole statistic
- job: Introduces occlusion by channel and the expectation that the mean carries most.
- leaves: The reader expects the mean to dominate.

### vig.comparing-grains.channel-code  [A1]  (code)
- idea: three-channel run, occluded
- job: Fits a three-channel weekly run and occludes by channel.
- leaves: Mean weight per channel.

### vig.comparing-grains.channel-result  [A1]
- idea: mean dominates, extremes share
- job: States that the mean carries most and why the extreme days carry some.
- leaves: Channels are correlated, so occlusion shares weight.

### vig.comparing-grains.response  (section)

### vig.comparing-grains.curve-method  [none]
- idea: move one cell of the representation
- job: Defines `response_curve()` and what a predictor is.
- leaves: A predictor is one bin-by-statistic cell.

### vig.comparing-grains.curve-code  [none]  (code + figure)
- idea: curve at the anchor week
- job: Draws the curve at the first variable's anchor week and tabulates each response's span.
- leaves: A plot and a span per response.

### vig.comparing-grains.curve-result  [none]
- idea: only the planted response moves
- job: States that the first variable rises as planted and the others barely move, with one exception read as a spurious coefficient.
- leaves: The curve confirms the planted sign.

### vig.comparing-grains.scores  (section)

### vig.comparing-grains.tss-why  [A5]
- idea: TSS cut chosen on scored units
- job: States the default metric, defines TSS and explains why choosing its cut on the scored units inflates it.
- leaves: TSS is inflated, more so with few presences per cell.

### vig.comparing-grains.score-code  [A5]  (code)
- idea: one candidate, four metrics
- job: Scores the weekly candidate under four metrics.
- leaves: Four means.

### vig.comparing-grains.metrics-read  [A5]
- idea: each metric's floor and reading
- job: Explains how each metric reads the predictions and introduces `tss_inflation()`.
- leaves: Average precision's floor is the prevalence; Boyce uses no absences; only TSS chooses a cut.

### vig.comparing-grains.inflation-code  [A5]  (code)
- idea: inflation at two skills
- job: Measures the inflation at this design's cell sizes.
- leaves: Reported level per planted skill.

### vig.comparing-grains.inflation-read  [A5]
- idea: inflation shrinks near one
- job: Reads the inflation at the two skills and introduces `implied_skill()` as the inverse.
- leaves: Inflation is larger at moderate skill.

### vig.comparing-grains.implied-code  [A5]  (code)
- idea: invert the observed TSS
- job: Inverts the weekly candidate's TSS.
- leaves: The implied population skill.

### vig.comparing-grains.implied-read  [A5]
- idea: paired TSS not free of inflation
- job: States what the observed TSS implies, the limit of the inversion, and why runs are scored by AUC.
- leaves: TSS beside AUC; a paired TSS difference can mislead.

### vig.comparing-grains.rare  (section)

### vig.comparing-grains.rare-mask  [A2 A5]
- idea: rare responses drop cells
- job: States that a cell needs both classes and that the mask drops cells for every arm alike.
- leaves: The mask shrinks with rarity.

### vig.comparing-grains.rare-code  [A2 A5]  (code)
- idea: rare record, cells, inflation
- job: Draws a rare record and reads its cells and inflation.
- leaves: Presence counts, cell table, inflation.

### vig.comparing-grains.rare-result  [A2 A5]
- idea: fewer cells, larger inflation
- job: States how rarity shrinks the scorable cells and widens the inflation.
- leaves: A four-presence response is scored on at most four folds.

### vig.comparing-grains.guidance  (section)

### vig.comparing-grains.guidance-scope  [A2]
- idea: guidance from these designs
- job: Bounds the guidance to the simulated designs and tells the reader to run the checks on their own.
- leaves: The guidance is design-specific.

### vig.comparing-grains.guide-paired  [A2]
- idea: few species, report intervals
- job: Recommends reporting differences with intervals when responses are few.
- leaves: Test only with enough responses.

### vig.comparing-grains.guide-mixed  [A2]
- idea: mixed model beside paired
- job: Recommends reading the mixed-model contrast beside the paired one.
- leaves: Pooled variance and chosen reference.

### vig.comparing-grains.guide-size  [A2]
- idea: held-out size, inner folds rank
- job: Ties the separation found to the held-out size and restates that inner scores only rank.
- leaves: Sample size sets what the comparison can separate.

### vig.comparing-grains.guide-rare  [A2 A5]
- idea: check cells before fitting
- job: Recommends running `scorable_cells()` and `tss_inflation()` before any fit.
- leaves: Fewer folds keep more presences per held-out cell.

### vig.comparing-grains.guide-report  [A2]
- idea: report the procedure
- job: Recommends the procedure's estimate as the level to report.
- leaves: Where to get it.

### vig.comparing-grains.guide-repeats  [A2]
- idea: repeats show split dependence
- job: Points to `cv(repeats =)` for split dependence, which the article did not use.
- leaves: Single-map results carry split noise.

### vig.comparing-grains.guide-meaning  [A1]
- idea: selected grain depends on n
- job: States what a selected grain does and does not mean.
- leaves: Grain preference depends on sample size as well as mechanism.

## Ensembles, thresholds and maps (vignettes/articles/prediction.Rmd)

### vig.prediction.route  [A4]
- idea: what follows a scored run
- job: Lays out the route from the split through combination and thresholds to maps.
- leaves: The reader knows the order and that one simulated grid carries it.

### vig.prediction.install  (section)
- idea: install from CRAN, GitHub, PyPI
- job: Repeats the install commands for both languages.
- leaves: Nothing new (see findings, `install-outside-get-started`).

### vig.prediction.grid  (section)

### vig.prediction.grid-design  [A4]
- idea: grid, warmth, frost, four species
- job: Describes the simulated grid and how each species is tied to warmth or frost, so later maps can be read against it.
- leaves: sp1 and sp2 follow warmth oppositely, sp3 the snap, sp4 both.

### vig.prediction.data-code  [A4]  (code)
- idea: simulate grid and tables
- job: Builds the series and targets tables.
- leaves: Presence counts per species.

### vig.prediction.raster-order  [A4]
- idea: cell order matches terra
- job: Explains why the temperature matrix doubles as a raster and what `transect` is.
- leaves: The maps later are built from the same matrix.

### vig.prediction.split  (section)

### vig.prediction.split-why  [A2]
- idea: split decides every number
- job: Establishes that the fold map underlies every later score, weight and threshold, and introduces the four constructors.
- leaves: Resampling is a specification turned into a map inside the run.

### vig.prediction.resampling-code  [A2]  (code)
- idea: four constructors printed
- job: Shows the four resampling specifications.
- leaves: Their printed form.

### vig.prediction.constructors  [A2]
- idea: four ways to deal folds
- job: Explains how each constructor deals the units and which take repeats.
- leaves: cv stratified, grouped_cv whole groups, block_cv and env_cv deterministic blocks.

### vig.prediction.fold-map-lead  [A2]
- idea: the map itself
- job: Introduces `fold_map()` as the map drawn from a response.
- leaves: `group` reproduces `grouped_cv()`.

### vig.prediction.fold-map-code  [A2]  (code)
- idea: plain and grouped maps
- job: Draws a plain and a grouped map.
- leaves: Fold sizes and transect-fold table.

### vig.prediction.fold-sizes  [A2]
- idea: equal folds, whole transects
- job: Reads the fold sizes of both maps.
- leaves: Grouping keeps transects whole.

### vig.prediction.scorable  [A2]
- idea: scorable cells defined
- job: Defines a scorable cell by the two conditions on held-out and fitting units.
- leaves: The mask is read from response and fold map alone.

### vig.prediction.cells-code  [A2]  (code)
- idea: cell table
- job: Prints the scorable-cell table.
- leaves: The cell table.

### vig.prediction.one-denominator  [A2]
- idea: no model, one denominator
- job: States why all cells are scorable here and why a model-free mask gives every candidate one denominator.
- leaves: Rare species drop cells for every candidate at once.

### vig.prediction.two-splits  (section)

### vig.prediction.run-design  [A2 A4]
- idea: two learners, two grains, two splits
- job: Sets up the run that the rest of the article reads, made under two splits.
- leaves: What the comparison between splits will show.

### vig.prediction.run-code  [A2 A4]  (code)
- idea: run under random split
- job: Fits the run under a random split.
- leaves: The fit.

### vig.prediction.procedure-rows  [A2]
- idea: nested rows are the level
- job: Explains how the procedure rows were computed and gives their scores under the random split.
- leaves: Selected and stack levels under random folds.

### vig.prediction.block-settings  [A2]
- idea: blocks bound inner folds
- job: Explains why the block run lowers `n_inner` and the elastic net's own `n_inner`.
- leaves: Blocked outer splits keep blocks whole inside too.

### vig.prediction.block-code  [A2]  (code)
- idea: run under spatial blocks
- job: Fits the run under spatial blocks.
- leaves: The blocked fit.

### vig.prediction.fold-maps-lead  [A2]
- idea: maps of the two splits
- job: Introduces the figure of both fold maps.
- leaves: Expectation of scattered versus compact folds.

### vig.prediction.fold-maps  [A2]  (figure)
- idea: scattered versus compact folds
- job: Shows the two splits on the grid.
- leaves: What a block split holds out.

### vig.prediction.estimates-code  [A2]  (code)
- idea: estimates side by side
- job: Prints the AUC estimates of both runs.
- leaves: Two estimate tables.

### vig.prediction.block-drop  [A2]
- idea: blocks lower the level
- job: States that the level falls under blocks, offers the gradient as the likely reason, and says which number describes prediction into unseen regions.
- leaves: Use the block number for maps outside the training region; forests lose more.

### vig.prediction.combination  (section)

### vig.prediction.combiner  [A2 A4]
- idea: combiner sees no model
- job: States what the combiner is handed and defines the default stack.
- leaves: One non-negative weight vector summing to one across responses.

### vig.prediction.weights-code  [A2 A4]  (code)
- idea: weights, fold weights, choices
- job: Prints the refitted weights, the per-fold weights and the per-fold choices.
- leaves: Three tables.

### vig.prediction.weights-read  [A2 A4]
- idea: refit weights versus fold weights
- job: Distinguishes the weights `predict()` uses from those behind the held-out ensemble score, and shows both are unstable across folds.
- leaves: Which weights to quote for what; the stack's composition swings.

### vig.prediction.methods-list  [A4]
- idea: five combination methods
- job: Lists the five methods and the eligibility settings.
- leaves: stack, mean, median, weighted, committee; min_score; scope.

### vig.prediction.ensemble-fit  [A4]
- idea: combiner on its own
- job: Introduces `ensemble_fit()` and `ensemble_combine()`.
- leaves: The combiner can be refitted outside a run.

### vig.prediction.methods-code  [A4]  (code)
- idea: fit five combiners
- job: Fits all five methods on the run's predictions.
- leaves: Stack weights; committee thresholds.

### vig.prediction.methods-read  [A4]
- idea: same weights; committee cuts
- job: Confirms the refitted stack matches and says how committee cuts are learned.
- leaves: Committee cuts by youden per member and response.

### vig.prediction.optimism  [A2 A4]
- idea: fitted combiners score optimistically
- job: Warns which combination scores are held-out and which are optimistic.
- leaves: Quote the nested estimate for stack, weighted and committee.

### vig.prediction.methods-score-code  [A4]  (code)
- idea: score each combination
- job: Scores each combination on the outer folds.
- leaves: Five AUCs.

### vig.prediction.methods-score  [A4]
- idea: mean level, committee coarse
- job: Reads the five scores and explains why the committee ranks coarsely.
- leaves: A few-member committee loses AUC by construction.

### vig.prediction.spread-intro  [A4]
- idea: spread reads members side by side
- job: Defines `ensemble_spread()` and its biomod2 counterparts.
- leaves: Mean, sd, cv and interval per target and response.

### vig.prediction.spread-code  [A4]  (code)
- idea: spread for sp3
- job: Shows the spread for the first targets.
- leaves: A spread table.

### vig.prediction.report  (section)

### vig.prediction.report-shape  [A2]
- idea: summary is a data frame
- job: Says what printing a run shows and that the summary is a filterable table.
- leaves: Weights and choices travel as attributes.

### vig.prediction.report-code  [A2]  (code)
- idea: report columns and choice
- job: Prints the report's main columns and the choice.
- leaves: mean, se, won, scored per row.

### vig.prediction.won  [A2]
- idea: won counts per response
- job: Explains `won` and shows that the stack's heaviest member wins no response, and restates the se scope.
- leaves: Weight in the stack and per-response wins differ.

### vig.prediction.predict  (section)

### vig.prediction.predict-method  [A4]
- idea: predict rebuilds representations
- job: States what `predict()` rebuilds and what new targets must carry.
- leaves: New targets need id, static columns and a series.

### vig.prediction.predict-code  [A4]  (code)
- idea: three candidate choices
- job: Predicts with the ensemble, the selected candidate and one named candidate.
- leaves: Three prediction matrices.

### vig.prediction.candidate-arg  [A4]
- idea: candidate argument values
- job: Explains the three values of `candidate` and why the selected elastic net predicts nearer one half.
- leaves: Default ensemble; selected; named.

### vig.prediction.thresholds  (section)

### vig.prediction.threshold-method  [A4]
- idea: cuts from out-of-fold predictions
- job: States where binary cuts come from and introduces `decision_threshold()` and `rule`.
- leaves: Cuts are learned on units the model was not fitted on.

### vig.prediction.thresholds-code  [A4]  (code)
- idea: four rules' cuts
- job: Computes each rule's cut per species.
- leaves: A cut table.

### vig.prediction.rules  [A4 A5]
- idea: four rules defined
- job: Defines each rule and reads where they agree and differ.
- leaves: youden is TSS's cut; mpa cuts lowest.

### vig.prediction.counts-code  [A4]  (code)
- idea: called versus observed
- job: Counts cells called present under each rule.
- leaves: Called counts per rule and species.

### vig.prediction.counts-read  [A4]
- idea: prevalence matches, mpa overcalls
- job: Reads the counts and states the binary output's type and its `NA` case.
- leaves: mpa trades specificity for presences kept.

### vig.prediction.disagree  (section)

### vig.prediction.spread-type  [A4]
- idea: spread for new targets
- job: States what `type = "spread"` returns.
- leaves: A target-by-response-by-statistic array.

### vig.prediction.spread-predict-code  [A4]  (code)
- idea: spread for sp1
- job: Shows the spread for six targets.
- leaves: Spread table.

### vig.prediction.spread-read  [A4]
- idea: small mean, large cv
- job: Reads the spread and points forward to the uncertainty map.
- leaves: cv is large where the mean is small.

### vig.prediction.presence-only  (section)

### vig.prediction.pseudo-method  [none]
- idea: three pseudo-absence strategies
- job: Defines `pseudo_absences()` and its three strategies.
- leaves: random, sre, disk; presences never drawn.

### vig.prediction.pseudo-code  [none]  (code)
- idea: disk and sre draws
- job: Draws pseudo-absences under two strategies.
- leaves: Two draws with headers.

### vig.prediction.pseudo-admits  [none]
- idea: header counts admitted units
- job: Reads the header's admitted counts and states the error when too many are asked for.
- leaves: `repeats` gives independent draws.

### vig.prediction.pseudo-fit-code  [none]  (code)
- idea: fit on one draw
- job: Fits a presence-background run on one draw.
- leaves: A presence-only fit.

### vig.prediction.pseudo-auc  [none]
- idea: background AUC differs
- job: States that the drawn rows are ordinary rows and that the AUC measures something different from presence-absence AUC.
- leaves: Fit per draw to see draw dependence.

### vig.prediction.maps  (section)

### vig.prediction.project-method  [A4]
- idea: project one target per cell
- job: States what `project()` takes and returns and that the fit needs `id`.
- leaves: Raster record with time layers; static raster named as the fit's columns.

### vig.prediction.rasters-code  [A4]  (code)
- idea: build the rasters
- job: Builds the record and elevation rasters.
- leaves: A 365-layer raster.

### vig.prediction.project-code  [A4]  (code + figure)
- idea: present-day map
- job: Projects the fit and checks it equals `predict()`.
- leaves: Four probability maps.

### vig.prediction.project-read  [A4]
- idea: projection equals predict
- job: States the projection's time, its equality with `predict()`, chunking and the `NA` rule.
- leaves: A missing reading anywhere makes the cell `NA`.

### vig.prediction.range-method  [A4]
- idea: warmer record, range change
- job: Sets up the future map and defines what `range_change()` counts.
- leaves: Same call with another period's record.

### vig.prediction.range-code  [A4]  (code + figure)
- idea: binary maps and change
- job: Projects both periods as binary maps and counts the change.
- leaves: Change table and sp1 map.

### vig.prediction.range-read  [A4]
- idea: warm species gain, cold lose
- job: Reads the change per species against the simulation and defines the change percentage and the codes.
- leaves: biomod2 codes; threshold for probability maps.

### vig.prediction.spread-map-lead  [A4]
- idea: project the disagreement
- job: States what `type = "spread"` projects and how layers are named.
- leaves: `response.statistic` layer names.

### vig.prediction.spread-map  [A4]  (code + figure)
- idea: sd map for sp3
- job: Draws the members' standard deviation for sp3.
- leaves: An uncertainty map.

### vig.prediction.curves  (section)

### vig.prediction.curve-method  [none]
- idea: one predictor moved
- job: Defines `response_curve()` and the two ways to name a predictor.
- leaves: Channel moves every bin; list names one cell.

### vig.prediction.curve-code  [none]  (code + figure)
- idea: ensemble curve over the mean
- job: Draws the ensemble's curve over the mean moved in every week.
- leaves: Curves per species.

### vig.prediction.curve-read  [none]
- idea: year swept, sp3 flat
- job: Reads the curve against the simulation and says how the ensemble curve is formed.
- leaves: sp3's signal is hidden in an all-weeks curve.

### vig.prediction.curve-bin-lead  [none]
- idea: single-week curve for sp3
- job: Motivates a curve over the snap week alone.
- leaves: The coldest week is the snap.

### vig.prediction.curve-bin-code  [none]  (code + figure)
- idea: curve at snap week
- job: Finds the snap week and draws sp3's curve over it.
- leaves: The single-bin curve.

### vig.prediction.curve-bin-read  [none]
- idea: sp3 falls as week warms
- job: Reads the single-bin curve against the simulation.
- leaves: One week barely moves sp1 and sp2.

### vig.prediction.surface-lead  [none]
- idea: two predictors, a surface
- job: Introduces `with` for a two-predictor surface.
- leaves: Expectation of an image.

### vig.prediction.surface  [none]  (code + figure)
- idea: mean by elevation surface
- job: Draws sp1's surface over weekly mean and elevation.
- leaves: The image.

### vig.prediction.surface-read  [none]
- idea: mean matters, elevation not
- job: Reads the surface.
- leaves: Gradient along the mean only.

### vig.prediction.guidance  (section)

### vig.prediction.guide-stack  [A2 A4]
- idea: when a stack pays
- job: Compares stack and selected candidate under both splits and states when a stack pays and how fold weights tell.
- leaves: Unstable weights favour reporting the selected candidate.

### vig.prediction.guide-unfitted  [A4]
- idea: mean when weights unstable
- job: Recommends the plain mean where stack weights are unstable and the committee for binary maps.
- leaves: Committee suits votes, not ranking.

### vig.prediction.guide-rules-lead  [A4]
- idea: threshold rule choice
- job: Introduces the rule list.
- leaves: Expectation of one line per rule.

### vig.prediction.guide-rule-prevalence  [A4]
- idea: prevalence for counts
- job: Recommends prevalence where the observed count or kappa is to be read.
- leaves: When to use prevalence.

### vig.prediction.guide-rule-youden  [A4 A5]
- idea: youden for equal costs
- job: Recommends youden where both errors cost the same.
- leaves: Youden is TSS's cut.

### vig.prediction.guide-rule-kappa  [A4]
- idea: kappa for chance agreement
- job: Recommends kappa where agreement beyond chance is the target.
- leaves: When to use kappa.

### vig.prediction.guide-rule-mpa  [A4]
- idea: mpa when omission costly
- job: Recommends mpa where missing a presence costs more.
- leaves: mpa overcalls.

### vig.prediction.guide-spatial  [A2]
- idea: spatial folds where clustered
- job: Recommends spatial, environmental or grouped folds by data structure, and restates the inner-fold bound.
- leaves: Which split for which data.

### vig.prediction.guide-cost  [A4]
- idea: projection is cheap
- job: States what a projection costs and how to run a large one.
- leaves: Fitting costs more than projecting.

Line numbers refer to the source files as read on 2026-10-14 (master, `dffca49` plus untracked
`sidecars/`). Spine lines are given per section where the section has more than two entries.

## Coming from biomod2

Source: vignettes/articles/biomod2.Rmd. Spine: what timesift does with biomod2's model set ->
how each model maps -> one run read under biomod2 names -> the same run side by side -> ensembles,
thresholds, other responses, maps, tuning, importance, curves -> what differs -> when biomod2 is the
better tool.

### vig.biomod2.setup  (code)
- idea: hidden setup chunk
- job: Loads the package and sets figure defaults; no reader-facing content.
- leaves: nothing.

### vig.biomod2.position  [A1 A3]
- idea: same algorithms, every grain
- job: Places timesift relative to biomod2 by saying what biomod2's predictor table already is (a summary of a record) and that timesift fits the same algorithms at every grain on shared folds before combining.
- leaves: biomod2 fits on one summarised table; timesift varies the summary and keeps the folds fixed.

### vig.biomod2.cores  [A3 A6]
- idea: own cores, pinned to references
- job: Establishes that each learner is timesift's own compiled code, checked in the fixtures against the package biomod2 calls, and that neither biomod2 nor those packages run at fit time.
- leaves: the reference package of each algorithm fixes its numbers, not its code; R and Python share the core.
- warning: covers more than it holds: `mlp()` (biomod2's `DNN`) is a torch encoder, neither compiled nor pinned; `MAXENT` in biomod2 is not maxnet (see findings).

### vig.biomod2.install  (section, code)
- idea: install lines
- job: Gives the R and Python install commands.
- leaves: how to install.
- warning: Installation belongs in Get started only (docs-articles); here it separates the position paragraphs from the mapping.

### vig.biomod2.mapping  [A3]  (section)
- idea: model to constructor
- job: States the rule of the mapping: one constructor per algorithm, variants as arguments.
- leaves: a biomod2 name becomes a constructor call.

### vig.biomod2.mapping-table  [A3]  (table)
- idea: biomod2 model table
- job: Gives each biomod2 model its constructor and the one setting that makes it biomod2's.
- leaves: the thirteen rows the reader looks a model up in.

### vig.biomod2.ann  [A3]
- idea: perceptron vs ANN · presets · column scale
- job: Shows that `perceptron()` is nnet's network and states where it departs from biomod2's `ANN` (loss, generator).
- leaves: same weights from the same start; cross-entropy instead of least squares; columns unscaled.
- warning: two-jobs: the `preset = "bigboss"` settings and the unscaled-columns caution are separate points, and `bigboss` is used here before vig.biomod2.presets defines it.

### vig.biomod2.dnn  [A3]
- idea: DNN as mlp plus control
- job: Maps cito's `dnn()` settings onto `mlp()` and `train_control()`, including the sign convention of `alpha`.
- leaves: which argument holds which cito setting.

### vig.biomod2.dnn-default  [A3]  (code)
- idea: biomod2 default DNN call
- job: Gives the `mlp()` call equal to biomod2's default `DNN`.
- leaves: a call to copy.

### vig.biomod2.dnn-bigboss  [A3]  (code)
- idea: biomod2 tuned DNN call
- job: Gives the `mlp()` call equal to biomod2's tuned `DNN`.
- leaves: a call to copy.

### vig.biomod2.dnn-diffs  [A3]
- idea: three remaining DNN differences
- job: Lists what the two calls still do differently from cito (batch size, standardisation, burn-in).
- leaves: the DNN mapping is close, not exact.

### vig.biomod2.presets  [A3]
- idea: default and bigboss presets
- job: Defines biomod2's two option sets and how `preset` reproduces each, with an explicit setting overriding.
- leaves: `preset = "default"` / `"bigboss"`; which learners have one set only.

### vig.biomod2.run  [A1 A3]  (section)
- idea: simulated record, weekly signal
- job: Introduces the simulated record and says the planted grain is weekly, so the result can be checked.
- leaves: 200 units, a year at six-hourly readings, three responses, signal at the week.

### vig.biomod2.data  (code)
- idea: simulate the record
- job: Builds the targets and readings.
- leaves: `sim`, `targets`.

### vig.biomod2.models-intro  [A3]
- idea: named list, pinned data
- job: Says why the learners are a named list and how `data =` pins a learner to one representation.
- leaves: candidates reported under biomod2 names; pinned vs unpinned learners.

### vig.biomod2.models  [A3]  (code)
- idea: biomod2 model set as list
- job: Writes biomod2's model set as timesift learners.
- leaves: `models`.

### vig.biomod2.fixed-grain  [A3]
- idea: GAM and SRE at season
- job: Explains why the additive model and the envelope are fixed to the seasonal grain.
- leaves: coefficient count per column; envelopes narrow as columns are added.
- warning: numbers mix `k = 10` (478) with the chunk's `k = 5` (17); "one presence in twenty at each end" for `quantile = 0.025` (findings).

### vig.biomod2.fit  [A2]  (code)
- idea: the run
- job: Runs the whole comparison and shows the first rows of the candidate table.
- leaves: `fit`.

### vig.biomod2.count  [A2]
- idea: 32 candidates, shared folds
- job: Accounts for the candidate count and states that all are scored on the same folds and cells.
- leaves: mean scores are comparable across candidates.

### vig.biomod2.scores  [A2]  (code)
- idea: print the run
- job: Shows the report.
- leaves: the printed table.

### vig.biomod2.report  [A1 A2]
- idea: weekly wins · grain per algorithm
- job: Reads the report: the weekly candidates lead, as planted, and the run adds the grain dimension to biomod2's algorithm ranking.
- leaves: what each column of the report means; grain and algorithm ranking can differ.
- warning: "The four highest mean AUCs belong to weekly candidates" is a hard-coded result in prose; inline it or it can go stale.

### vig.biomod2.plot-fit  [A1]  (figure)
- idea: mean AUC by representation
- job: Shows each candidate's score against its grain.
- leaves: the ladder picture.

### vig.biomod2.side  [A3]  (section)
- idea: same run, two packages
- job: Sets up the side-by-side: biomod2's three steps against one timesift call; chunks not run.
- leaves: the two calls are meant to be the same run.

### vig.biomod2.biomod2-side  [A3]  (code)
- idea: biomod2 calls
- job: Gives biomod2's format, model and ensemble calls.
- leaves: the biomod2 side.

### vig.biomod2.timesift-side  [A3]  (code)
- idea: timesift call
- job: Gives the matching timesift call.
- leaves: the timesift side.
- warning: the two model sets differ (biomod2 side has `GAM` and `SRE`, timesift side drops them; selection metric TSS vs AUC), so "the same run" is not matched.

### vig.biomod2.python-intro  [A6]
- idea: Python takes same constructors
- job: Says Python takes the same constructors.
- leaves: one sentence of A6.

### vig.biomod2.python  [A6]  (code)
- idea: Python call
- job: Gives the Python version of the call.
- leaves: the Python side.

### vig.biomod2.multi  [A2 A3]
- idea: all species in one call
- job: States that timesift fits every response at once on one fold map, mask and candidate set, where biomod2 fits one species per call.
- leaves: responses are columns of one result.

### vig.biomod2.ensembling  [A3]  (section)
- idea: ensemble algorithms map
- job: Introduces the mapping of biomod2's ensemble algorithms.
- leaves: they are `ensemble()` options.

### vig.biomod2.ensemble-table  [A3]  (table)
- idea: EM* to ensemble()
- job: Gives each biomod2 ensemble algorithm and selection argument its timesift form, plus the stack.
- leaves: lookup table.

### vig.biomod2.stack  [A2]
- idea: stack default, out-of-fold only
- job: Defines the stack and why it never weights an in-sample prediction.
- leaves: the default combiner.

### vig.biomod2.min-score  [A3]
- idea: min_score as thresh
- job: Ties `min_score` to `metric.select.thresh` and introduces the weights shown.
- leaves: eligibility threshold.

### vig.biomod2.weights  (code)
- idea: six largest weights
- job: Shows the weights.
- leaves: output.

### vig.biomod2.committee-intro  [A3]
- idea: committee averaging defined
- job: Defines committee averaging and sets up a smaller run.
- leaves: votes at learned cuts.

### vig.biomod2.committee  (code)
- idea: committee run
- job: Runs it and shows AUC and TSS estimates.
- leaves: output.

### vig.biomod2.spread-intro  [A4]
- idea: EMcv/EMci as spread
- job: Maps the disagreement summaries to `type = "spread"`.
- leaves: which statistics the spread returns.

### vig.biomod2.spread  (code)
- idea: spread output
- job: Shows the spread array.
- leaves: output.

### vig.biomod2.cv-map  [A4]
- idea: CV gives uncertainty map
- job: Says the coefficient of variation per cell gives an `EMcv`-style map.
- leaves: the link to maps.

### vig.biomod2.thresholds  [A5]  (section)
- idea: self-selected TSS inflates
- job: States that TSS read at its own best cut inflates where presences are few, and that timesift estimates the inflation.
- leaves: why `tss_inflation()` exists.
- warning: "biomod2 ... reports TSS at a threshold chosen on the same predictions it scores" holds for biomod2's `calibration` column only; its `validation` column is scored at the calibration cut (context.md, Discrepancies).

### vig.biomod2.inflation  [A5]  (code)
- idea: inflation for these responses
- job: Shows the estimate for the article's three responses.
- leaves: output.

### vig.biomod2.rules  [A4 A5]
- idea: threshold rules listed
- job: Lists the rules of `decision_threshold()`.
- leaves: youden, kappa, prevalence, mpa.

### vig.biomod2.table-intro  [A3]
- idea: other statistics via table_metric
- job: Introduces `table_metric()` for biomod2's other statistics.
- leaves: confusion-table metrics.

### vig.biomod2.metric-table  [A3]  (table)
- idea: metric names map
- job: Maps biomod2's metric names.
- leaves: lookup table.

### vig.biomod2.one-cut  [A3 A5]
- idea: one cut for all statistics
- job: States the design difference: biomod2 optimises a cut per statistic, `table_metric()` uses one rule's cut for all.
- leaves: why the same statistic can read differently.

### vig.biomod2.heads  [A3]  (section)
- idea: other heads · regression metrics · ordinal · multiclass
- job: Maps biomod2's non-binary data types onto response heads and their metrics.
- leaves: four heads, which learners fit them, metric names.
- warning: two-jobs (heads and learners; metrics; ordinal reading; multiclass) and code-dense.

### vig.biomod2.count-head  [A3]
- idea: count head, Poisson cores
- job: Describes the count head and which cores fit the Poisson family against which reference.
- leaves: Poisson per learner; tree shrinkage.
- warning: chain-not-list: six learners mapped to seven references by "respectively".

### vig.biomod2.maps  [A4]  (section)
- idea: projection and range change
- job: Maps biomod2's projection and range-size steps onto `project()` and `range_change()`.
- leaves: one target per cell; two binary maps give a range change.

### vig.biomod2.project  [A4]  (code)
- idea: project now and later
- job: Shows the calls (not run).
- leaves: a pattern; uses objects the article never builds.

### vig.biomod2.tuning  [A2 A3]  (section)
- idea: tune inside outer folds
- job: Maps `BIOMOD_Tuning()` onto `tune()` and says tuning never sees the scoring units.
- leaves: tuned settings in the candidate table.

### vig.biomod2.tune  (code)
- idea: tune a forest
- job: Shows the call (not run).
- leaves: pattern.

### vig.biomod2.importance  [A3]  (section)
- idea: occlusion as VariablesImportance
- job: Maps biomod2's permutation importance onto `occlusion()` and defines what a predictor is there.
- leaves: same quantity, read on held-out units; bin vs channel.

### vig.biomod2.curves  [A3]  (section)
- idea: response curves mapped
- job: Maps `bm_PlotResponseCurves()` onto `response_curve()`.
- leaves: a predictor is a statistic across bins.

### vig.biomod2.curve  (code)
- idea: warm_day curve
- job: Shows the call (not run).
- leaves: pattern.

### vig.biomod2.differences  [A2 A3]  (section, list)
- idea: absences · folds · cells · weights · PA-only · predictors
- job: Lists the behaviours a biomod2 user must expect to differ.
- leaves: six differences.
- warning: bold bullet lead-ins; the Folds bullet is a second mapping table in prose.

### vig.biomod2.better-tool  [A3]  (section)
- idea: when biomod2 fits better
- job: Names the case where biomod2 is the better choice: a raster stack with no record behind it.
- leaves: the reader can decide whether they need timesift.
- warning: "gains little from switching" assumes migration.

## Troubleshooting

Source: vignettes/articles/troubleshooting.Rmd. Spine: one shared record -> refusals in the order a
run meets them (record, time, units, learner, statistics, responses, torch, prediction) -> cost ->
reproducibility -> cross-language -> short questions -> reporting.

### vig.trouble.setup  (code)
- idea: hidden setup
- job: Seeds and loads the package.
- leaves: nothing.

### vig.trouble.purpose  [A1]
- idea: refusals, reproduced
- job: States the article's job: each common refusal, its cause and the call that works.
- leaves: how each section is built.

### vig.trouble.record  
- idea: shared 24-plot record
- job: Describes the record every example reads.
- leaves: 24 plots, hourly, 60 days from 1 September 2021, two species.

### vig.trouble.install  (section, code)
- idea: install lines
- job: Gives install commands.
- leaves: how to install.
- warning: sits between vig.trouble.record and the chunk it describes.

### vig.trouble.data  (code)
- idea: build the record
- job: Builds `series` and `targets`.
- leaves: the objects.

### vig.trouble.gap  [A1]  (section)
- idea: unit misses a bin
- job: Sets up the gap case.
- leaves: p03 loses one week.

### vig.trouble.gap-code  (code)
- idea: gap error
- job: Shows the refusal.
- leaves: the message.

### vig.trouble.gap-why  [A1]
- idea: no imputation, why
- job: Explains why a missing bin is refused rather than filled, and that the run refuses it the same way.
- leaves: the non-goal "no imputation" with its reason.

### vig.trouble.coverage  [A1]
- idea: coverage finds gaps
- job: Introduces `coverage()` as the way to locate gaps.
- leaves: a diagnostic.

### vig.trouble.coverage-code  (code)
- idea: coverage output
- job: Shows it.
- leaves: output.

### vig.trouble.gap-options  [A1]
- idea: three ways forward
- job: Names the three remedies and leaves the choice to the analysis.
- leaves: drop units, cut the record, coarsen.

### vig.trouble.gap-fix  (code)
- idea: two fixes run
- job: Runs two remedies.
- leaves: output.

### vig.trouble.gap-all  [A1]
- idea: gap in every unit
- job: Explains the second refusal, a bin nobody reaches.
- leaves: why bins are not silently dropped.

### vig.trouble.gap-all-code  (code)
- idea: all-unit gap error
- job: Shows it.
- leaves: message.

### vig.trouble.partial  [A1]  (section)
- idea: partial edge weeks
- job: Explains why the first and last weekly bins are partial.
- leaves: calendar tiling.

### vig.trouble.partial-code  (code)
- idea: inspect partial bins
- job: Shows the bins and their attributes.
- leaves: output.

### vig.trouble.partial-why  [A1]
- idea: partial bins are fine · their extremes shrink
- job: Says partial bins are expected and what `bin_partial` and `bin_n` record, and what fewer readings do to the statistics.
- leaves: partial bins are marked and counted.

### vig.trouble.partial-drop  [A1]  (code + sentence)
- idea: drop partial bins
- job: Shows `partial = "drop"`.
- leaves: the option.

### vig.trouble.partial-none  [A1]
- idea: dropping can empty
- job: Shows that dropping at a coarse grain can leave nothing.
- leaves: an error case.

### vig.trouble.partial-run  [A1]
- idea: run keeps partial · trim yourself
- job: Says the run keeps partial bins and how to fit on whole bins by trimming.
- leaves: the trim recipe.

### vig.trouble.partial-trim  (code)
- idea: seven whole weeks
- job: Shows the trim.
- leaves: output.

### vig.trouble.time  [A1]  (section)
- idea: text or Date times
- job: Says text and Date time columns are refused.
- leaves: POSIXct is required.

### vig.trouble.time-code  (code)
- idea: time class errors
- job: Shows both refusals.
- leaves: messages.

### vig.trouble.time-fix  [A1]
- idea: parse with zone · unknown zone
- job: Gives the fix and the unknown-zone refusal.
- leaves: parse with `tz`.

### vig.trouble.zone-code  (code)
- idea: unknown zone error
- job: Shows it.
- leaves: message.

### vig.trouble.zone  [A1]
- idea: zone sets midnight
- job: Explains the silent case: bins follow the zone's calendar.
- leaves: Vienna days start 22:00 UTC in September.

### vig.trouble.vienna  (code)
- idea: Vienna days
- job: Shows it.
- leaves: output.

### vig.trouble.dst  [A1]
- idea: DST days · fixed offset
- job: Reads the result and gives the fixed-offset alternative.
- leaves: 23/25-hour days; `Etc/GMT-1`.

### vig.trouble.fixed  (code + sentence)
- idea: fixed offset days
- job: Shows days at 23:00 UTC year-round.
- leaves: output.

### vig.trouble.readings  [A1]  (section)
- idea: NA and duplicates refused
- job: States both are refused where the record is read.
- leaves: clean before representing.

### vig.trouble.readings-code  (code)
- idea: NA and duplicate errors
- job: Shows them.
- leaves: messages.

### vig.trouble.readings-why  [A1]
- idea: where duplicates come from
- job: Names the usual cause and the fix.
- leaves: deduplicate by a rule.

### vig.trouble.ids  [A2]  (section)
- idea: target without record
- job: States that a target with no record is refused.
- leaves: id links the tables.

### vig.trouble.ids-code  (code)
- idea: missing unit error
- job: Shows it.
- leaves: message.

### vig.trouble.ids-extra  [A2]
- idea: extra series units ignored
- job: States the converse is not an error.
- leaves: unread units do not enter.

### vig.trouble.ids-extra-code  (code)
- idea: fewer targets run
- job: Shows it.
- leaves: output.

### vig.trouble.ids-count  [A2]
- idea: fitted targets listed
- job: Reads the result and says where fitted targets are listed.
- leaves: `rownames(fit$y)`.

### vig.trouble.ids-dup  [A2]
- idea: duplicate ids need target_time
- job: Explains the duplicate-id refusal.
- leaves: `target_time` disambiguates.

### vig.trouble.reads  [A3]  (section)
- idea: learner declares what it reads
- job: Explains tabular vs sequence learners and the pinned refusal.
- leaves: elastic net cannot read `native()` when pinned.

### vig.trouble.reads-code  (code)
- idea: pinned refusal
- job: Shows it.
- leaves: message.

### vig.trouble.reads-sift  [A3]
- idea: sift skips quietly
- job: Says a sift pairing is skipped and recorded.
- leaves: `fit$candidates` note.

### vig.trouble.reads-none  [A3]
- idea: nothing fittable stops
- job: Says the run stops when no pair fits.
- leaves: message lists reasons.

### vig.trouble.reads-sequence  [A3]
- idea: sequence learners refuse flat
- job: Explains the converse refusals for `cnn()`/`rescnn()`.
- leaves: multigrain and one-bin grains refused.

### vig.trouble.pinned  [A1 A3]  (section)
- idea: pinned representation outside sift
- job: Says a pinned representation is built even outside the sift.
- leaves: output shows it.

### vig.trouble.clash  [A1]
- idea: same name, different settings
- job: Explains the name clash error.
- leaves: the cause.

### vig.trouble.clash-fix  [A1]
- idea: name the sift
- job: Gives the fix.
- leaves: named sifts.

### vig.trouble.daylevel  [A1]  (section)
- idea: day-level stats need days
- job: Explains why day-level statistics refuse sub-daily grains.
- leaves: refused at construction.

### vig.trouble.daylevel-fix  [A1]
- idea: min vs cold_day
- job: Gives the sub-daily alternative.
- leaves: `min` at half-day.

### vig.trouble.rare  [A2]  (section)
- idea: unscorable cells
- job: Defines a scorable cell and sets up two rare responses.
- leaves: cell = response x fold.

### vig.trouble.rare-read  [A2]
- idea: which folds fail, why
- job: Reads the mask and states it is model-free and shared.
- leaves: rare responses contribute nothing.

### vig.trouble.precheck  [A2]
- idea: check before fitting
- job: Shows the mask can be read before a fit.
- leaves: `scorable_cells()`.

### vig.trouble.rare-none  [A2]
- idea: fewer folds · all-absent stops
- job: Advises fewer folds and shows the one stopping case.
- leaves: the remedy.

### vig.trouble.response  [A2]
- idea: response must be 0/1
- job: Shows the response refusals.
- leaves: binary, complete.
- warning: the biomod2 article says four other heads exist; this says the response "must be 0 and 1" without saying "under the default presence-absence head".

### vig.trouble.torch  [A3]  (section)
- idea: torch checked at fit
- job: Says torch is checked when fitted, not constructed, and gives the message.
- leaves: the message text.
- warning: process-narration ("quoted from the source here, since the machine this article was built on has torch installed").

### vig.trouble.predict  [A4]  (section)
- idea: calendar bins must match
- job: Explains why a shorter or later record is refused at prediction.
- leaves: same weeks required.

### vig.trouble.predict-read  [A4]
- idea: three refusals explained
- job: Reads the three errors and shows a valid subset prediction.
- leaves: any subset of units predicts.

### vig.trouble.lookback  [A1 A4]
- idea: lookback predicts other periods
- job: Gives the route to another period: a lookback anchored on `target_time`.
- leaves: the fix for forecasting.

### vig.trouble.cost  [A2]  (section)
- idea: cost is fit count
- job: Gives the fit-count formula for nested and outer-only runs.
- leaves: `n_inner = NULL` trades the estimate for speed.

### vig.trouble.cost-read  
- idea: timings
- job: Reports the two timings.
- leaves: numbers.

### vig.trouble.native-cost  [A1]
- idea: array size follows record
- job: Says each fit costs its array, and native grows with the record.
- leaves: size arithmetic.

### vig.trouble.iterate  [A1]
- idea: iterate coarse first
- job: Advises coarse grains and no inner search while exploring.
- leaves: the workflow advice; `threads`.
- warning: "the Schrankogel record of the package's origin" is undefined in this article.

### vig.trouble.seeds  [A2]  (section)
- idea: seeds the call carries
- job: Lists where each draw is seeded.
- leaves: same seeds, same numbers.

### vig.trouble.stream  [A2]
- idea: fold map restores stream
- job: States the session stream is left as found.
- leaves: no side effect.

### vig.trouble.digest  [A6]
- idea: digest tests input change
- job: Gives the usual cause of drift and a digest test.
- leaves: `digest_array()`.

### vig.trouble.digest-read  [A2]
- idea: row order irrelevant · torch not bit-stable
- job: Explains why the digests agree and states the neural limit.
- leaves: neural reruns may differ.

### vig.trouble.lang  [A6]  (section)
- idea: one core, digest check
- job: States the contract and how to run the digest check, linking the contract article.
- leaves: the check is runnable.

### vig.trouble.lang-limits  [A6]
- idea: penalised tolerance, neural free
- job: States what is pinned as a distance and what is not pinned, and what counts as a bug.
- leaves: when to report.
- warning: restates the contract's penalised-fit rationale; one sentence and the link would do (A6 home is contract.Rmd).

### vig.trouble.lang-input  [A6]
- idea: input causes disagreement
- job: Names the usual cross-language causes: zone and `year_start`.
- leaves: check the input first.

### vig.trouble.faq  (section)
- idea: short questions
- job: Answers eight recurring questions in a line or two each.
- leaves: standardisation, coverage periods, static columns, reported number, missing responses, skipped candidates, saving fits, auto grains.

### vig.trouble.bug  (section)
- idea: what to report
- job: Says which failures are bugs and what a useful report holds.
- leaves: the tracker URL and the report contents.

## The representation contract

Source: vignettes/articles/contract.Rmd, which renders inst/spec/representation.md below its
Installation section. Only the spec's opening (lines 1-4) and its last section (lines 2009-2301)
were read in full; the middle sections are listed from the heading outline and get section-level
entries labelled as such, no job claims about their content.

### vig.contract.install  (section, code)
- idea: install lines
- job: Gives install commands.
- leaves: how to install.
- warning: it is the first thing on the page; the article has no opening of its own saying what the reader will learn (the A6 question).

### vig.contract.normative  [A6]
- idea: spec is normative
- job: States that the document overrides either implementation.
- leaves: a disagreement is a bug in the code, not the spec.
- warning: opens on a verbless fragment ("Normative for both implementations.").

### vig.contract.sections  [A1 A3 A6]  (sections, heading outline only)
- idea: input to combiner
- job: (not read in full) Input, ordering, zone, grains, partial bins, statistics, custom bins, output, lookback, channels, the boundary artifacts, fixtures, and one section per core (penalised fit, maxnet, tree, forest, boosting, envelope, stepwise, MARS, discriminant, additive, network, hierarchical, combiner).
- leaves: unknown until read.
- warning: repeated subsection headings "The fixtures" (13x) and "How exactly" (13x) give duplicate anchors and headings that do not name their subject.

### vig.contract.carries  [A6]  (section)
- idea: differences recorded here
- job: States that every difference between the languages is recorded in this section.
- leaves: where to look.

### vig.contract.names  [A6]  (table)
- idea: one name per concept
- job: Lists each concept and its shared name.
- leaves: the vocabulary on both sides.

### vig.contract.same-call  [A6]  (section, list)
- idea: same call, same behaviour
- job: Lists behaviours that must agree across the languages, one bullet per rule.
- leaves: about thirty rules (selection, intervals, binary cut, weights, encoders, nested run, contrasts, simulator, plot colours).
- warning: bullets of up to twenty lines (`"nested_cv"` interval, encoders standardisation) are paragraphs with several jobs; `inner` is named where both languages call the argument `n_inner`.

### vig.contract.shaped  [A6]  (table)
- idea: same thing, different shape
- job: Lists concepts whose shape differs by language.
- leaves: R vs Python forms.

### vig.contract.shaped-why  [A6]
- idea: shapes, not behaviours
- job: States the shape differences change no behaviour and why R uses methods.
- leaves: `c()` and generics rationale.

### vig.contract.one-language  [A6]  (tables)
- idea: tidyselect R-only · Python helpers
- job: Lists what exists in one language only and why.
- leaves: tidyselect is R-only; Python exposes helpers and types.

### vig.contract.models  [A6]
- idea: models not promised
- job: States the one non-promise: neural fits are not byte-identical.
- leaves: the limit of A6.

Source: vignettes/articles/python.Rmd, generated by tools/python_articles.py from
python/articles/python.md. Edits go to the .md source, not the .Rmd. Line numbers below are the
.Rmd's. Headings mirror the R Get started (vignettes/timesift.Rmd) one for one, except that the R
article ends on `## Where next` and this one has no counterpart.

Spine (one line per section, read in order):
1. The reduction of a record is an argument; this is the R path in Python on one shared core.
2. Install.
3. Two tables: one row per thing to predict, one long record, linked by an id.
4. One call: candidates are the comparison, `selected`/`ensemble` are the numbers to report; then predictions and a binary cut.
5. Representations are settings, built later; sets of them; a learner pinned to one.
6. A learner is refused a representation it cannot read.
7. One fold map and one model-free mask make every comparison matched.
8. Learners, the shared elastic-net core, training control, a learner of your own.
9. The combiner sees predictions only; weights say who carries the signal.
10. TSS at its own best cut is inflated; measure it, or score by AUC.
11. The arrays without the fitting layer: calendar bins, extreme day vs extreme reading, position channels.
12. The ladder on its own, and the two ways to test a step of it.
13. What the fit read: occlusion.

## Choosing how a record is read, in Python

### vig.python.problem  [A1]
- idea: reduction made an argument
- job: Opens on the reader's own habit of reducing a record once by hand and states that timesift makes that reduction an argument it fits at and compares.
- leaves: The package's claim in the reader's terms: the time scale is a setting the run tests, and the output says how much of the record the prediction needed.

### vig.python.scope  [A6]
- idea: R path, one shared core
- job: Sets what the article covers (two tables to a scored comparison and a prediction) and ties it to the R walkthrough and the contract, so a reader in either language knows the numbers are the same.
- leaves: This is a tutorial on simulated data, it mirrors the R article section by section, and binning, statistics and the penalised fit are one compiled core under both languages.

### vig.python.install  (section)
- idea: how to install
- job: Gives the install lines.
- leaves: Section heading only.

### vig.python.install-r  (code)
- idea: R install lines
- job: Installs the R package from CRAN or GitHub.
- leaves: Nothing a Python reader needs; see findings (placement).

### vig.python.install-py  (code)
- idea: pip install, extras
- job: Installs the wheel, with the torch, contrasts and plot extras, or from GitHub.
- leaves: The extras exist and what each brings. Does not say that the article's code needs pandas, which the wheel does not install, nor names the `grid` extra `project()` needs.

### vig.python.tables  (section)
- idea: two tables
- job: Heading for the input shapes.
- leaves: -

### vig.python.tables-def  [A1]
- idea: targets and series defined
- job: Defines the two inputs (a target is one row to predict, a series the long time-stamped record belonging to it) and the identifier that links them.
- leaves: `targets`, `series`, the shared id; either is a mapping of column to array, which a pandas or polars frame is.

### vig.python.tables-code  (code)
- idea: simulated plots and species
- job: Builds 60 plots of 120 days of hourly temperature with a per-plot level, and six presence-absence responses driven by that level with alternating sign.
- leaves: Concrete `targets` (plot, elevation, sp1..sp6) and `series` (plot, t, temp) to run on.

### vig.python.tables-signal
- idea: signal buried in noise
- job: Says what the simulation planted, so the reader knows what a good representation should recover.
- leaves: Each plot's level is the signal; hourly noise is about an order of magnitude larger.

### vig.python.call  (section)
- idea: one call
- job: Heading for the run.
- leaves: -

### vig.python.call-code  [A2]  (code)
- idea: the run and its report
- job: Shows the single call with `y`, `id`, `time`, `static`, `sift` and `resampling`, and the printed report.
- leaves: What a run returns at a glance: candidates with mean, won and responses; the selected and ensemble rows with a standard error; the choice and weights on every target.

### vig.python.call-built  [A3]
- idea: what was fitted
- job: Says that every representation in `sift` was built and the default learner (elastic net) fitted on each over the same folds, and announces a two-part report.
- leaves: `learners` defaults to `[ts.elasticnet()]`; the report has two parts.

### vig.python.call-candidates  [A2]
- idea: candidates are the comparison
- job: Explains how to read the candidate rows and rules out reporting the best of them.
- leaves: `won`, `responses`; the highest candidate mean is selected on the folds it is scored on, so it is not the number to report.

### vig.python.call-procedure  [A2]
- idea: procedure rows to report
- job: Establishes that `selected` and `ensemble` are honest held-out scores because the choosing and weighting happened inside each outer training fold.
- leaves: The nested design in one paragraph; `selected` and `ensemble` are what to report.

### vig.python.estimate  (code)
- idea: estimate table, every metric
- job: Shows `fit.estimate` as a table of every registered metric for both procedure rows.
- leaves: The estimate carries 27 metrics with se and interval. No prose names which one to read; the run's own metric is roc_auc.

### vig.python.selected  (code)
- idea: chosen per fold
- job: Shows which candidate each outer fold chose and its inner score.
- leaves: The choice is per fold; here month in all five.

### vig.python.interval  [A2]
- idea: interval across responses
- job: States the limit of the reported interval: it is across responses on the same plots and folds.
- leaves: The interval does not carry the error the responses share.

### vig.python.static  [A1]
- idea: only named columns predict
- job: Says a column of `targets` reaches a model only when `static` names it.
- leaves: `elevation` is a predictor because it was asked for; stray columns are ignored.

### vig.python.candidates-code  (code)
- idea: bins and channels per grain
- job: Shows each candidate's representation, bin count and channel count.
- leaves: 120, 18 and 4 bins; two channels each.

### vig.python.channels
- idea: second channel is elevation
- job: Explains the two channels.
- leaves: A static column is a channel held constant across bins.

### vig.python.plot  [A2]  (figure)
- idea: score by representation
- job: Draws mean AUC across the representations with the ensemble's held-out level across it.
- leaves: The curve the report tabulates, and where the ensemble sits against it (read later in vig.python.weights-read).

### vig.python.predict  [A4]
- idea: predict rebuilds representations
- job: Says how `predict()` handles new rows: it rebuilds each member's representation from its stored settings and combines through the ensemble refitted on every target; `candidate="selected"` uses the chosen candidate.
- leaves: Default prediction is the ensemble; the alternative is `"selected"`.

### vig.python.predict-code  (code)
- idea: probability matrix
- job: Shows the prediction as a targets-by-responses matrix.
- leaves: Output shape and scale.

### vig.python.binary  [A4]
- idea: binary cut from out-of-fold predictions
- job: Establishes that a binary prediction is cut at a threshold learned from out-of-fold predictions, and names the rules.
- leaves: `type="binary"`, `ts.decision_threshold()`, rules youden, kappa, prevalence (the fourth rule, `"mpa"`, is not named).

### vig.python.threshold-code  (code)
- idea: one cut per response
- job: Shows the prevalence cuts.
- leaves: The cuts as numbers.

### vig.python.binary-code  (code)
- idea: binary prediction
- job: Shows the 0/1 prediction.
- leaves: Output form.

### vig.python.map  [A4]
- idea: map is one target per cell
- job: Tells the reader how a map follows from the same call.
- leaves: A binary map is this prediction on one target per cell with its own series. Does not name `ts.project()` or link the prediction article.

### vig.python.reps  (section)
- idea: representations
- job: Heading.
- leaves: -

### vig.python.reps-settings  [A1]
- idea: settings, not data
- job: Defines a representation as settings alone, which is why one object describes an arm before data and rebuilds for new targets.
- leaves: Why a representation can be named before the record is read. Uses "arm" without defining it.

### vig.python.reps-code  (code)
- idea: four constructors printed
- job: Shows `native`, `grain` with three statistics, `multigrain`, `lookback`.
- leaves: Kind, grain or span, stats of each.

### vig.python.reps-sets  [A1]
- idea: sets, auto, multigrain, lookback
- job: Introduces sets of representations and what multigrain and lookback are for.
- leaves: `ts.grains()`, `ts.lookbacks()`, `"auto"`; multigrain is one tabular block; lookback is a span anchored on `target_time`.

### vig.python.pin  [A1 A3]
- idea: pin a learner
- job: Says a learner runs across the set or at one pinned representation.
- leaves: `data=` pins.

### vig.python.pin-code  (code)
- idea: pinned elastic net printed
- job: Shows a pinned learner and its settings.
- leaves: elasticnet defaults (alpha 0.5, n_inner 10, s lambda.1se ...).

### vig.python.handed  (section)
- idea: what a learner reads
- job: Heading.
- leaves: -

### vig.python.handed-refusal  [A1 A3]
- idea: tabular vs sequence refusal
- job: Explains the tabular/sequence distinction and when each kind of mismatch is refused.
- leaves: Tabular given native is refused before building; sequence given one bin is refused after.

### vig.python.handed-code  (code)
- idea: the error
- job: Shows the refusal message.
- leaves: The message names the fix.

### vig.python.handed-set
- idea: skipped in a set, error by hand
- job: Distinguishes a skipped pair inside a set from an error when pinned.
- leaves: `not applicable` in `ts.summary()`.

### vig.python.split  (section)
- idea: split and cells
- job: Heading.
- leaves: -

### vig.python.split-folds  [A2]
- idea: one fold map
- job: States that one fold map serves every candidate, and names the ways to give one.
- leaves: `ts.cv()`, `ts.grouped_cv()`, a fold vector, `ts.fold_map()`.

### vig.python.folds-code  (code)
- idea: fold sizes
- job: Shows the fit's folds.
- leaves: 60 units in 5 folds of 12.

### vig.python.split-cells  [A2]
- idea: scorable cell defined
- job: Defines a scorable cell from the response and fold map alone.
- leaves: A cell needs both classes in test and in training.

### vig.python.cells-code  (code)
- idea: all 30 cells scorable
- job: Shows the mask summary.
- leaves: 30 of 30.

### vig.python.split-matched  [A2]
- idea: shared denominator
- job: States the consequence: every candidate on the same cells, so means and paired differences are matched.
- leaves: Restates vig.python.call-candidates ("on the same cells, so their means can be read against each other"); see findings.

### vig.python.learners  (section)
- idea: learners and training
- job: Heading.
- leaves: -

### vig.python.learners-shipped  [A3]
- idea: tabular learners and encoders
- job: Sorts the shipped learners by what they read and introduces the joint torch encoders.
- leaves: elasticnet, linear, forest read a block; cnn, rescnn convolve; mlp flattens. Names three of the fourteen; the rest are left to the learners article, which is not linked. Ends on an unsourced claim about pooling.

### vig.python.elasticnet  [A3 A6]
- idea: shared penalised core · s and tol
- job: Says the elastic net is the shared core so it is the same baseline in both languages, then describes `s` and `tol`.
- leaves: Two jobs; the A6 half restates vig.python.scope.

### vig.python.control  [A3]
- idea: architecture vs training
- job: Splits constructor settings from `ts.train_control()` and states the override rule.
- leaves: One control per run; a learner's own control overrides by name.

### vig.python.control-code  (code)
- idea: control and cnn printed
- job: Shows control defaults and a cnn with an epoch override.
- leaves: The defaults, and that `epochs=300` on cnn overrides the run's 200.

### vig.python.own  [A3]
- idea: own learner · flatten · case weights
- job: Says a user learner is a fit/predict pair scored like the shipped ones, then gives `ts.flatten()` and the `weights` argument.
- leaves: Three things; see findings (crowded).

### vig.python.own-code  (code)
- idea: two user learners in a run
- job: Defines a joint 1-nearest-neighbour and a weighted IRLS logistic, and runs them beside elastic net.
- leaves: A report with user learners ranked among shipped ones.

### vig.python.multi  [A3]
- idea: joint vs separate
- job: Explains `multi` and its default and ties it to the report's `responses` column.
- leaves: `multi="joint"` vs `"separate"` (default).

### vig.python.combination  (section)
- idea: combination
- job: Heading.
- leaves: -

### vig.python.combiner  [A2]
- idea: combiner sees predictions · methods · which weights predict
- job: States that the combiner never sees a model, lists the combination methods, and says which weights predict and why they are not the scored ones.
- leaves: Methods stack, mean, median, weighted, committee; `fit.fold_weights`. Does not say where a method is chosen (`ensemble=ts.ensemble(method=...)`).

### vig.python.weights-code  (code)
- idea: weights printed
- job: Shows the stack weights.
- leaves: Month carries 0.96, week 0.04, day effectively 0 (printed as 5.967019e-269).

### vig.python.weights-read  [A2]
- idea: reading weights and the plot
- job: Tells the reader how to read weights and where the ensemble line sits on the plot.
- leaves: Zero weight means passed over; ensemble above every curve means complementary candidates.

### vig.python.honest  (section)
- idea: reading a level honestly
- job: Heading.
- leaves: -

### vig.python.inflation  [A5]
- idea: TSS inflated at own cut
- job: States why TSS at its own best cut is inflated, more so with fewer presences, and names the function that measures it.
- leaves: `ts.tss_inflation()` for the design in hand.

### vig.python.inflation-code  (code)
- idea: inflation at 0.6 and 0.9
- job: Shows the inflation for this design.
- leaves: +0.17 at 0.6, +0.07 at 0.9 here.

### vig.python.inflation-read  [A5]
- idea: expectation, inverse, why AUC
- job: Qualifies the inflation as an expectation, names the inverse, and gives the reason runs default to AUC.
- leaves: `ts.implied_skill()`; paired TSS differences are not free of inflation; AUC is the default metric.

### vig.python.arrays  (section)
- idea: arrays alone
- job: Heading.
- leaves: -

### vig.python.arrays-intro  [A1]
- idea: grain_matrix
- job: Introduces `ts.grain_matrix()` as the representation without the fitting layer.
- leaves: It bins by calendar and summarises.

### vig.python.arrays-code  (code)
- idea: weekly three-channel array
- job: Builds the paper's weekly cold_day/mean/warm_day array.
- leaves: 60 x 18 x 3; the first bin starts 2021-08-30, two days before the record.

### vig.python.calendar  [A1]
- idea: calendar bins
- job: Establishes that bins are real weeks and months, not fixed hour blocks.
- leaves: Weeks start Monday; months are calendar months. Says "28, 30 or 31 days" (omits 29).

### vig.python.calendar-code  (code)
- idea: hours per month bin
- job: Shows readings per monthly bin.
- leaves: 720, 744, 720, 696. The 696 is a partial December (record ends 29 December), which no prose explains.

### vig.python.extremes  (section)
- idea: extreme day vs reading
- job: Subheading, the only one under its parent.
- leaves: -

### vig.python.extremes-def  [A1]
- idea: six extreme statistics
- job: Defines min/max, cold_day/warm_day and mean_daily_min/max, and shows with one cold hour why they differ.
- leaves: The vocabulary the paper's input uses.

### vig.python.extremes-code  (code)
- idea: seven statistics of one week
- job: Shows the ordering on one plot-week.
- leaves: -28.68 to 31.50 for readings, -1.73 to 0.94 for days.

### vig.python.position  [A1]
- idea: calendar position channels
- job: Says why a pooled encoder needs bin position as input.
- leaves: `ts.calendar_channels()` and `ts.bind_channels()`.

### vig.python.position-code  (code)
- idea: sin/cos channels
- job: Shows the bound channels.
- leaves: mean, year_sin, year_cos.

### vig.python.ladder  (section)
- idea: one grain at a time
- job: Heading.
- leaves: -

### vig.python.ladder-intro  [A2]
- idea: ladder on its own
- job: Introduces `ts.grain_ladder()` for prebuilt arrays.
- leaves: Same split, same mask, reachable without `timesift()`.

### vig.python.ladder-code  (code)
- idea: ladder summary
- job: Shows per-grain scores.
- leaves: month best at 0.912.

### vig.python.paired  [A2]
- idea: paired contrast
- job: Says a claim about one step rests on the paired contrast and how it is computed, with the small-sample limit.
- leaves: Difference per cell, averaged per response, summarised across responses; six responses gives a wide interval.

### vig.python.paired-code  (code)
- idea: month vs day
- job: Shows the contrast.
- leaves: +0.042, interval 0.004 to 0.081, 6 of 6 favour, p 0.031.

### vig.python.contrasts  [A2]
- idea: whole-curve contrast
- job: Introduces `ts.grain_contrasts()` and its extra.
- leaves: Mixed model, many-to-one against the best, needs scipy.

### vig.python.contrasts-code  (code)
- idea: Dunnett table
- job: Shows each grain against month.
- leaves: Neither difference is significant after adjustment.

### vig.python.read  (section)
- idea: what was read
- job: Heading.
- leaves: -

### vig.python.occlusion  [A1]
- idea: occlusion profile
- job: Defines the occlusion profile and that nothing is refitted.
- leaves: Weight per bin per response, needing `keep_fits=True`.

### vig.python.occlusion-code  (code)
- idea: monthly weights
- job: Shows the mean weight of each monthly bin.
- leaves: September and October carry most.

### vig.python.channel-hold  [A1]
- idea: hold a channel instead
- job: Points at holding a channel back, without showing it.
- leaves: That the option exists; not how to call it (`over="channel"`) or where to read on. The article ends here, with no routing.
