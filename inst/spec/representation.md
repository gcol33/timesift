# The representation contract

Normative for both implementations. R and Python must produce the same numbers from the same
input; where this document and either implementation disagree, this document is right.

## Input

A long table of readings with three columns of interest:

| column | type | meaning |
|---|---|---|
| id | character or factor | the unit carrying the sensor (a plot, a site, a device) |
| time | POSIXct, UTC | the instant of the reading |
| value | numeric | the reading |

Requirements, each checked and each an error rather than a warning:

- No missing `id`, `time` or `value`.
- No duplicate `(id, time)` pair.
- Every id spans the same set of bins once binned. A record that stops early is not silently
  padded; it is reported with the ids and the bins concerned.

Ordering of the input rows carries no meaning and is not relied on. The output is ordered by
sorted unique id and by bin start.

## Ordering identifiers

Every place an identifier decides a position -- the ids naming the first dimension of the
representation, and the variable names naming the cells of the scorable mask -- they are sorted by
**C collation**: the byte order of the UTF-8 encoding, which for every code point is also the code
point order. It does not depend on the session's locale, on the machine, or on the language.

That has to be stated because the two languages' defaults disagree and so do two R sessions in
different locales. R's `sort()` follows `LC_COLLATE`, which in an English locale orders `_z` before
`a` before `A`, and NumPy's `np.unique()` orders by code point, which puts `A` first. Both
implementations therefore name the rule rather than take a default: R passes
`method = "radix"`, which sorts characters in the C locale whatever `LC_COLLATE` says, and NumPy
already sorts this way.

The failure this prevents is silent rather than loud. Two orderings hold the same numbers in
different rows, so a response matrix built in one language and a representation built in the other
line up row for row while naming different units, and nothing errors. `series_order.csv` carries
ids the two rules order differently, so the fixtures fail rather than the user.

The channel names are never sorted: the channel order is the order the caller named the statistics
in.

## The time zone

Bin membership is decided from the calendar the series is carried in, so the zone has to be named
before anything is binned. It is named once, at the edge of each implementation, and nothing below
that edge knows a zone exists: the binning and the reduction see only *naive local seconds*, the
count of seconds from 1970-01-01T00:00:00 on that calendar. A day there is 86400 of them whatever
the night did, and a month is what the proleptic Gregorian calendar says.

| | how the zone is named |
|---|---|
| R | the `tzone` attribute of the `POSIXct` column; unset means UTC |
| Python | the `tz` argument; `None`, the default, means the instants already read as the calendar to bin by, which is what a zone-free `datetime64` says |

The same instants and the same zone give the same answer in both languages, and the fixtures pin
that rather than leaving it assumed.

Reading an instant as a clock is defined for every instant in every zone. The reverse is not: on
the night a zone moves its clock forward a local time exists on no instant, and on the night it
moves back a local time exists on two. A bin start is a local time, so reporting it as an instant
needs a rule, and the rule is that **a local time the clock skipped resolves to the instant the
clock jumped to, and a local time the clock repeated resolves to the first of the two**. In
`America/Sao_Paulo`, whose clock moved at midnight until 2019, the day beginning 4 November 2018
therefore opens at 01:00 local rather than at a midnight that never happened, and holds 23 readings
rather than 24.

Instants are read at whole seconds. Two readings a fraction of a second apart are the same reading
twice, and are reported as a duplicated `(id, time)` pair.

## Grains

Bin membership is read from the calendar rather than from a running count of hours, so a bin is a
real week or a real month rather than a drifting block of 168 or 730 hours.

| grain | bin |
|---|---|
| `native` | the reading itself, no reduction |
| `halfday` | 00:00-11:59 and 12:00-23:59 of each calendar day |
| `day` | the calendar day |
| `week` | ISO week, Monday to Sunday |
| `month` | the calendar month |
| `season` | three calendar months, counted from `year_start` |
| `year` | the year running from `year_start` |

`year_start` is a `"MM-DD"` string, default `"09-01"`. It sets the boundary of the hydrological
year and therefore also the phase of the seasonal bins. A seasonal bin is three calendar months
counted from that anniversary, so a record of three hydrological years beginning on it holds
twelve of them and no partial one. Cutting seasons anywhere else, at the equinoxes and solstices
for instance, is a different calendar and is passed as a function; see Custom bins.

Bins are contiguous and cover the record with no gap and no overlap, and both halves of that are
asserted:

- every `(id, bin)` cell holds at least one reading, which is what makes a bin the same span for
  every id and a record that stops early an error rather than a padded row;
- consecutive bin starts are one bin apart on the grain's own calendar. A bin no id reaches is
  never built, so a February missing from every logger would otherwise give four "adjacent" monthly
  bins with February simply gone, and a convolution would read January and March as neighbours.

The second is not asserted for `native`, where the bin is the reading itself and the bin sequence is
the record's own sampling grid rather than a calendar, nor for a supplied calendar, which declares
its own bin lengths and is contiguous by construction. It is asserted in local time, so a sequence
stepping across a clock change is contiguous: the civil day a zone shortened is still one bin of
one calendar day.

## Partial bins

A bin is **partial** when the record does not cover its whole calendar span. The record covers
from its first reading to its last plus one sampling interval, taken as the smallest gap between
consecutive distinct reading instants, and a bin is partial when its own span reaches outside
that. Only a bin at an end of the record can, because every id is required to hold readings in
every bin between them.

Which bins those are follows from where the record starts and stops against the calendar, not from
the grain alone. Three years of hourly readings from 1 September on a `"09-01"` boundary carry no
partial month, season or year, and a partial week at each end, because 1 September 2021 is a
Wednesday. A record from an arbitrary deployment date carries one at each end of almost every
grain.

`grain_matrix()` reports the verdict as `bin_partial` and takes a `partial` argument saying what
becomes of such a bin: `"keep"`, the default, returns it alongside the full bins; `"drop"` removes
it, and errors rather than returning an empty representation if that leaves no bin. Dropping is a
choice about the record, not about the implementation: it discards up to three months of readings
at each end of a seasonal grain, while keeping gives a bin whose mean is taken over fewer readings
and whose `cold_day` and `warm_day` are drawn from fewer days, so they sit closer to that bin's
mean than a full bin's would. `bin_n` gives the count each bin was reduced from.

Both implementations obey the same rule and the fixtures pin both settings.

## Statistics

Each named statistic becomes one channel of the output. A grain may carry any subset, and the
channel order in the output is the order given by the caller.

| name | definition | defined for |
|---|---|---|
| `mean` | arithmetic mean of the readings in the bin | every grain |
| `min` | smallest single reading in the bin | every grain |
| `max` | largest single reading in the bin | every grain |
| `cold_day` | smallest daily mean among the days in the bin | `day` and coarser |
| `warm_day` | largest daily mean among the days in the bin | `day` and coarser |
| `mean_daily_min` | mean over the bin's days of each day's smallest reading | `day` and coarser |
| `mean_daily_max` | mean over the bin's days of each day's largest reading | `day` and coarser |

"A day or coarser" is decided from the bins rather than from the grain's name: a day-level
statistic requires every calendar day of the record to lie entirely inside one bin. Naming `native`
or `halfday` is refused before any data is read; a supplied calendar that cuts inside a day is
refused once the bins are in hand, naming the day it splits and the two bins it splits it between.

The four day-level statistics reduce each calendar day first and then reduce again over the days of
the bin. `cold_day` and `warm_day` take the extreme of the daily means; `mean_daily_min` and
`mean_daily_max` take the mean of the daily extremes. They are not `min` and `max`, which act on
single readings, and the difference is the point: an extreme day is a state the site was in, an
extreme reading can be one hour, and an average daily extreme is the exposure a typical day of the
bin brought.

Two orderings follow from the definitions and are asserted:
`min <= mean_daily_min <= mean <= mean_daily_max <= max` and
`min <= cold_day <= mean <= warm_day <= max`. The two day-level pairs are not ordered against each
other, and a bin whose days differ widely in level is where they part: a bin of one day at 0 and
one at 10 has `cold_day` 0 and `mean_daily_min` 5.

At the `day` grain `cold_day`, `warm_day` and `mean` coincide by construction, as do
`mean_daily_min` with `min` and `mean_daily_max` with `max`. Requesting them there is allowed and
returns the identical channels.

## Custom bins

A caller may supply the binning instead of naming a grain, as a function of the reading instants
returning the start of each reading's bin. Everything downstream is unchanged: the bins are still
required to tile the record, and the output still carries the bin starts as its second dimension's
names. This is how a calendar the package does not carry is used, such as seasons cut at the
equinoxes and solstices rather than on the first of a month.

Such a calendar owns its own bin lengths. A record beginning inside one of its seasons gives a
leading bin of a few weeks beside neighbours of three months, and that is the calendar the caller
asked for rather than a bin the record failed to fill: it is what makes three years cut at the
equinoxes thirteen bins where three calendar months from `"09-01"` are twelve. The function
declares where its bins begin, so the package cannot know where the last one was meant to end and
takes the record's end as its end; that final bin is never reported partial, and a leading bin is
judged as any other.

The function must return a bin start for every reading, including any that precede its first
boundary. Deciding that with an interval lookup is the natural way to write one and the two
languages disagree below the first boundary, where R's `findInterval()` gives 0 and NumPy's
`searchsorted() - 1` gives -1: the first silently shortens the result, which is an error, and the
second silently wraps to the last boundary, which is not. Either put the first boundary at or
before the record's first reading, as the fixtures do, or handle the readings below it explicitly.

## Output

A numeric array of shape `[n_id, n_bin, n_channel]`.

- Dimension 1 is named by the sorted unique ids.
- Dimension 2 is named by the ISO-8601 timestamp of each bin's start, in UTC.
- Dimension 3 is named by the statistic.

Attributes carried on the array:

| attribute | content |
|---|---|
| `grain` | the grain name |
| `stats` | the statistic names, in channel order |
| `year_start` | the `"MM-DD"` boundary used |
| `bin_start` | the bin start instants, resolved from local time as **The time zone** describes |
| `bin_end` | the last reading instant assigned to each bin |
| `bin_n` | a `[id, bin]` matrix of how many readings fell in each cell |
| `bin_partial` | a logical vector marking the bins the record does not cover for their whole calendar span |

No standardisation, centring or scaling happens here. Scaling is a property of a fit and belongs
to the fold it is computed on, never to the representation, because computing it over all ids
would leak the held-out units into the training input.

## The lookback

The second reduction, and the one no calendar expresses. Its input is a table of **targets** beside
the readings: one row per thing to predict, carrying the unit whose record it reads and the instant
it is anchored at. Two targets on the same unit a fortnight apart read two different stretches of
that unit's series, which is why the bins are placed relative to the target rather than to a month
or a week.

| column | type | meaning |
|---|---|---|
| id | character or factor | the unit, which the readings must carry |
| at | POSIXct | the instant the target is anchored at |

A target's identity is its position in that table. A unit may carry any number of targets, so the
unit cannot name a row, and the output is in the table's own row order rather than in a sorted one.

### The bin arithmetic

Three numbers describe a lookback: `span`, its length; `lag`, the gap between the anchor and the end
of the lookback; and `bins`, how many sub-bins it is cut into, oldest first. With `a` the target's
anchor and `step = span / bins`, bin `b`, counted from zero, covers naive local

    [a - lag - span + b * step,  a - lag - span + (b + 1) * step)

closed at the left and open at the right, so a reading on a boundary belongs to the later bin.
`span` must divide by `bins` exactly; one that does not is an error rather than a rounded step.
Only the readings whose unit is the target's are read: a lookback is a stretch of one unit's record.

Nothing else changes. The statistics are the seven **Statistics** defines, computed over the
readings of a cell exactly as they are over the readings of a grain's cell, and the day-level pair
reduces the days of a bin oldest first.

### The two guards

- Every `(target, bin)` cell holds at least one reading. A lookback reaching past either end of the
  record is an error naming the target and the interval, never a padded row, for the reason a
  grain's empty cell is one: an invented value in front of a model is worse than a target the
  record cannot answer for.
- The four day-level statistics reduce each calendar day first, so they are defined only where
  every calendar day lies whole inside one bin. A calendar settles that by itself; a lookback has to
  be asked, and the answer is two conditions rather than one. `step` must be a whole number of
  days, which holds for every target or for none, and `a - lag - span` must fall on a day boundary,
  which is a property of the anchor: a table of anchors on the hour rules the four statistics out
  where the same anchors at midnight allow them. Either failing is an error naming the target.

### Durations

`span` and `lag` are a count and a unit, or a bare count of seconds. **A year is 365 days and a
month is 30 days.** A lookback of a fixed length is a fixed length rather than a calendar step:
comparing two targets' representations means each read the same amount of record, and a February
or a leap year takes that away. Where the calendar is what matters, a grain is the reduction that
follows it.

| unit | seconds |
|---|---|
| `second`, `seconds` | 1 |
| `minute`, `minutes` | 60 |
| `hour`, `hours` | 3600 |
| `day`, `days` | 86400 |
| `week`, `weeks` | 604800 |
| `month`, `months` | 2592000 |
| `year`, `years` | 31536000 |

A string is a count, optional space, and one of those names, upper or lower case: `"30 days"`,
`"12 hours"`, `"1 year"`. A string of digits alone is seconds, and so is a number. Nothing else
parses, and a duration that does not parse is an error naming the argument.

### The output of a lookback

A numeric array of shape `[n_target, n_bin, n_channel]`, traversed and digested exactly as a
grain's is, target fastest.

- Dimension 1 is named by the target's own row label where the target table carries one, and by its
  position, from 1, where it does not. R takes the row names of the `at` data frame; Python has no
  row names and always names by position.
- Dimension 2 is named by where the bin opens relative to the anchor, oldest first, written as a
  signed count and the coarsest of `day`, `hour`, `minute` and `second` that divides it exactly,
  singular at one: `-30 days`, `-1 day`, `-12 hours`. In a grain an instant names a bin; here no
  two targets share one.
- Dimension 3 is named by the statistic.

Attributes carried on the array:

| attribute | content |
|---|---|
| `grain` | `"lookback"` |
| `span`, `lag` | the durations, resolved to seconds |
| `bins` | how many bins the lookback was cut into |
| `stats` | the statistic names, in channel order |
| `bin_n` | a `[target, bin]` matrix of how many readings fell in each cell |

There is no `bin_start`, no `bin_end` and no `bin_partial`. A bin is a position relative to an
anchor rather than a span of the calendar, and a cell the record does not cover is an error rather
than a verdict.

## What crosses the language boundary, and what does not

The binning and the reduction are one implementation: `src/ts_core.cpp` and `src/ts_calendar.cpp`,
compiled into the R package by R itself and into the Python extension by CMake. What each language
holds above it is the boundary, which resolves the columns, resolves the zone and wraps the result.
The two agree by construction rather than by two implementations being checked against each other
after the fact.

The digests did not stop meaning anything when that happened. The implementations they used to
compare are kept as test oracles, `tests/testthat/helper-oracle.R` and `python/tests/oracle.py`,
reachable from neither package at runtime and exercised only against the core on the fixtures and
on random series. The NumPy one was written from this document rather than from the R source, which
is what makes it evidence that the document is complete. One implementation in production, two in
evidence.

The representation is normative and is checked byte-exactly. Three things beside it are artifacts
that both sides read rather than each side computing: the response matrix, the fold map, and the
mask of scorable cells that follows from those two. A fold map built from a seed in R and a fold
map built from the same seed in Python are different maps, because the two languages draw on
different random streams; the fix is not to align the streams but to build the map once and read
it in the other language. **The file format below is what makes that possible**, and both sides
carry the reader and the writer for all three.

A model fitted in one language and a model fitted in the other cannot be byte-identical and are
not required to be.

## The file format of the three artifacts

One format for all three: CSV, UTF-8, a header row, `,` as the separator, no quoting, and LF line
endings on every platform. Written the same way in either language, the same artifact gives the
same bytes, so a round trip through a file is checkable and is checked.

A number is written with `%.12g`: twelve significant digits, which renders `0` and `1` as `0` and
`1` and carries any measurement a response holds. A logical is written `TRUE` or `FALSE`, and is
read from either that or `1`/`0`.

### The fold map: `id,fold`

| column | type | |
|---|---|---|
| `id` | character | the unit |
| `fold` | integer | the fold it is held out in, from 1 |

One row per unit, ordered by the id under **Ordering identifiers**. A unit may appear once.

### The response matrix: `id` and one column per variable

| column | type | |
|---|---|---|
| `id` | character | the unit |
| each remaining column | numeric | that variable's value at that unit |

The columns after `id` are the variables, **in the file's own order**, which is the order the
response carries them in; they are not sorted, because a response's column order is the caller's.
Rows are ordered by the id. A presence-absence response holds `0` and `1` and nothing else, and is
checked for that when it is prepared rather than when it is read.

### The scorable mask: `variable,fold,n_occ,pres_train,abs_train,pres_test,abs_test,scorable`

One row per `(variable, fold)` cell, ordered by variable under **Ordering identifiers** and then
by fold ascending. The seven columns after `variable` are integers except `scorable`, which is a
logical. The mask is a pure function of the response and the fold map, so it can be recomputed
rather than carried; it is written because reading it is how a language that did not build it gets
the exact cells the other one scored on.

### A unit the file does not carry

Aligning any of the three to a representation's units is by name, never by position. A unit in the
representation that the file has no row for is an error, reporting how many are missing and naming
the first of them in the representation's own order. A unit the file carries that the
representation does not is dropped without comment: a fold map covering a whole study is a normal
thing to read a subset of.

## Fixtures

Three series, because a record that starts on a bin boundary cannot tell two binning rules apart
and a record in UTC cannot tell two readings of a zone apart.
`spec/fixtures/series.csv` is a synthetic three-unit, 400-day hourly series beginning at midnight
on the default anniversary, so every coarse grain is in phase with it from the first reading.
`series_offset.csv` is a two-unit, 200-day series beginning at 05:00 on 17 October, which is what
a logger deployed when someone could walk to it gives, and puts every grain out of phase.
`series_zoned.csv` is a two-unit, 10-day series across 4 November 2018, the night
`America/Sao_Paulo` moved its clock at midnight, which is the record that tells a calendar read by
arithmetic apart from one read by writing a local midnight and parsing it back. `seasons.csv` holds
the equinox and solstice boundaries that make each series a caller-supplied calendar, which is the
only path the manuscript's seasonal rung ever took.

`digests.csv` holds one row per series, grain, time zone, `year_start`, `partial` setting and
statistic, covering every grain-by-statistic combination, each of the three-channel schemes
(`min+mean+max`, `mean_daily_min+mean+mean_daily_max`, `cold_day+mean+warm_day`), the coarse
grains at anniversaries other than the default, both `partial` settings, the supplied calendar,
and the zone: every grain of the aligned series read as a `Europe/Vienna` clock, which moves twice
inside that record, and the short series read as an `America/Sao_Paulo` clock, which moves at
midnight inside it, including a `year_start` landing on the night it moves. Each row carries `n_unit`, `n_bin`, the first and last bin start, how many bins are
partial, and the digest.

The lookback reads the same three series and three files of its own. `lookback_targets.csv`
holds the anchors, in named sets rather than one set per series, because an anchor that is a local
midnight in one zone is not one in another and an anchor on the hour rules out the day-level
statistics that a midnight allows: `aligned` and `offset` sit on day boundaries in UTC, `hourly`
sits on the hour, and `zoned` sits on local midnights in `America/Sao_Paulo`. `lookback_digests.csv`
holds one row per target set, zone, span, lag, bin count and statistic, covering every statistic
and each of the three-channel schemes, lags of none and of a day and of half a day, one and three
and four and seven bins, a step that is a whole number of days and one that is not, a span written
as a bare count of seconds, and the zoned set read both as UTC and as the clock that moves inside
it. Each row carries `n_target`, `n_bin`, the first and last bin's offset, the unit of the first
and last target, and the digest. `lookback_guards.csv` holds one case per guard -- a lookback reaching
past the record, a day-level statistic over bins shorter than a day, and one over bins that do not
open on a day boundary -- with the substring of the message the two implementations must both
raise.

Both test suites read the series, rebuild every row and assert all of it. The shape is asserted
before the digest, so two implementations that put the record into a different number of bins are
reported as that rather than as an unexplained hash mismatch. A contract that carried one series
starting on the anniversary, at the default anniversary, with no supplied calendar and no bin
count beside the hash would pass while the two sides disagreed on how many seasons three years
hold, which is the whole thing it exists to prevent.

The digest is defined byte-exactly, because a scheme that varies by platform pins nothing:

1. Traverse the array in its own order: unit fastest, then bin, then channel.
2. Format each value with `%.12f`.
3. Join with a line feed, terminate with one, encode UTF-8.
4. MD5 of those bytes.

The line ending is LF on every platform. R's `writeLines()` emits CRLF on Windows, so the bytes
are written explicitly; a digest generated on Windows and checked on Linux must agree.

Twelve places is far below any difference that could change a fitted model and far above the
noise from the two languages accumulating a mean in different orders. Should a combination ever
straddle a rounding boundary at the twelfth place, the fix is to record that combination's
tolerance in this document, not to loosen the scheme for everything.

A digest that moves without a matching change to this document is a bug in whichever
implementation moved. Regenerating the fixtures is a deliberate act with its own commit.

## What else is pinned

The representation is not the only deterministic thing the two languages share, and a contract
that pinned it alone would let everything above it drift. Four more fixtures, generated by the same
script and read by both suites:

`response.csv`, `folds.csv` and `cells.csv` hold one response of 40 units by 6 variables, a fold
map of five folds over it, and the mask that follows. The variable names order differently under
C collation and under an English locale, and the prevalences are chosen so the mask is not all
`TRUE`: one variable is present nowhere and one everywhere, so neither has a scorable cell at all,
and a rare one is scorable in some folds and not others. Both suites read the response and the
fold map, recompute the mask, and assert it cell by cell. Both also write all three back and
assert the bytes, which is what makes the file format a contract rather than a convention.

`metric_cases.csv` and `metrics.csv` hold ten `(y, p)` cases and the value of every threshold
metric on each: `tss`, `roc_auc`, `kappa` under both rules, and `decision_threshold` under all
three. The cases are where the tie rule is the whole answer -- every prediction tied, ties within
a class, ties across the classes, one presence, one absence, all presences, all absences, a
perfect separation and a reversed one. A metric a case defines no value on is written `NA` rather
than left out, so a suite that quietly skipped it fails rather than passes.

`contrast_cells.csv` and `contrast.csv` hold a fixed table of per-cell scores for two arms, with
cells one arm scored and the other did not, and the paired contrast read off it. No model is
involved: the pairing, the per-variable mean and the signed-rank p-value are what the two
languages own, and a fitted model is what they are not required to share.

### How exactly

Everything named above is **byte-exact**, at the twelve significant digits the format writes. It
is arithmetic on the same finite inputs in both languages, so anything less would be a difference
worth finding rather than a tolerance worth allowing. The one exception is the signed-rank
p-value: it is exact where the exact distribution applies, which is fewer than fifty values with
no tie, and the Python side reaches the normal approximation beyond that through a Chebyshev fit
to the complementary error function accurate to about 1.2e-7 relative. The fixture stays on the
exact branch; where a caller lands on the other, the two agree to 1e-6 and no closer.

`tss_inflation()` cannot be pinned as a digest, because it draws replicates from each language's
own random stream, and aligning those streams would be the wrong fix for the same reason it is the
wrong fix for a fold map. Its inflation figure is a headline claim of the package, so what is
required of it is stated rather than left to a hand check: **at 200 replicates the two
implementations agree on the inflation to within 0.02 at each planted skill**, which is well
inside the Monte Carlo error of either one alone and far below the +0.110 the claim rests on. A
disagreement beyond that is a bug in one of them, not sampling.

## What each language carries

The representation and the three artifacts are the contract. Everything built over them is meant to
match too, and where the two sides differ the difference is recorded here rather than found at a
call site.

### One name per concept

| concept | the name, on both sides |
|---|---|
| the whole run | `timesift()`, from a table of targets and a table of series to a scored comparison |
| what a representation is | `native()`, `grain()`, `multigrain()`, `lookback()`, and the sets `grains()` and `lookbacks()` |
| the target-anchored array | `lookback_matrix()` |
| which units reach which bins | `coverage()`, the count of readings per unit and bin over every bin the calendar tiles the record with, which is where a refused record's gaps are read off |
| building one representation | `build_representation()` |
| the penalised learner | `elasticnet()` |
| the forward selector | `stepwise()` |
| the forest | `forest()` |
| the encoders | `mlp()`, `cnn()`, `rescnn()` |
| how an encoder is trained | `train_control()` |
| the resampling | `cv()` and `grouped_cv()` |
| the combiner | `ensemble()`, `ensemble_fit()`, `ensemble_combine()` and `ensemble_weights()` |
| scoring held-out predictions | `score_predictions()`, on the cells the mask allows |
| a set of representations | `timesift_set()`, which reads as a mapping of grain name to representation |
| folds of the inner cross-validation | `n_inner` |
| the digest | `digest_array()`, exported |
| the three registries | `register_learner()` and `learners()`, `register_metric()` and `metrics()`, `register_response()` and `responses()` |

### The same call does the same thing

- Naming one grain returns the representation and naming two or more returns a set, whether the
  one is named as a string or as a sequence of one.
- `grain_ladder()` and `select_grain()` left without a fold map build one with the defaults of
  `fold_map()`. The two languages draw different maps from the same seed, so where both must see
  one split, write it and read it back as the section above describes.
- Held-out predictions are placed by unit and by variable, never by position.
- A setting given at fit time overrides the one the learner carries, and a setting the learner does
  not have is refused rather than ignored.
- The response head and the metric are registry entries. `metric` takes a registered name or a
  function of `(y, p)`, and left unset it is the one the response head carries.
- The encoders take `swa` and `swa_start`: the schedule anneals until the averaging begins and is
  then held flat, the averaged weights get their own pass to rebuild the batch-normalisation
  statistics from a reset, and the default is off, so a default recipe is the same recipe on both
  sides. `swa_start` is at least 0 and under 1, and `pos_weight_cap` is at least 1, on both.
- The encoders standardise every channel by its own centre and sample standard deviation over
  every unit and bin of the fitting units; the inner validation set is one unit from each of as
  many equal-count strata of the response total as it holds; the fitting units are cut into as
  few batches of at most `batch_size` rows as they divide into, of as equal a length as they can
  be; the snapshot early stopping restores, and the running average `swa` keeps, are copies of
  the weights and never the storage the optimiser updates.
- A `static` column enters the array as a channel holding the same number in every bin, which is
  the constant an encoder reads beside the readings. Flattening the bins into a block of features
  reads such a channel once, so a static predictor is one column of the design however many bins
  the grain has.
- A learner is fitted toward the registered response head and holds no response of its own: the
  encoders train under the head's `loss` and predict through its `activation`, and the three
  learners fitting one model per response take the family the loss names, logistic under
  `binary_cross_entropy` and Gaussian under `squared_error`. A fit that declares a `head` argument
  is handed the head, as one that declares `control` is handed the control.
- A fitted encoder holds its weights as arrays and the device *setting* rather than the device it
  resolved to, and rebuilds the network when it predicts, so a fit written with `saveRDS()` or
  `pickle` predicts in a fresh session and on another machine.
- `select_grain()` searches the candidates in the order the grains and the learners were declared
  in, so which candidate an exact tie on the inner score falls to does not depend on how the names
  sort.
- A learner left without a `data =` runs across every representation of the run, and one given a
  representation there runs at that one alone. A pairing the learner cannot read is skipped and
  reported by name inside a set, and is an error where the caller named it.
- A candidate is reported as `learner / representation`, and every candidate emits an out-of-fold
  prediction for every scorable cell over the same folds. The combiner is handed those predictions,
  the response, the mask and the fold map, and never a model.
- The combiner minimises the loss of the head the run was fitted under. `ensemble()` left without
  a `response` takes the run's, and one naming a different head is refused before anything is
  fitted; `ensemble_fit()` called on its own reads an unnamed head as `presence_absence`.

### The same thing, shaped differently

| concept | in R | in Python |
|---|---|---|
| a learner's own training settings | a `control` field holding a partly specified `train_control()` | its `params`, beside the architecture |
| the occlusion profile | `occlusion()`, an S3 generic with methods on a run and on a ladder | `occlusion()`, one function taking either |
| a set of learners, or of representations | `c()`, an S3 method on each spec class | a `list`, and `+` between two of them |

All three are shapes rather than behaviours: a setting given to a learner beats the run's control
on both sides, the profile is one implementation on both sides, and a set is the same set. `c()`
is a method rather than a second constructor because R's own way to combine things of one kind is
`c()`; a Python list already concatenates, so nothing is added there.

### Present in one language only

| in R only | why |
|---|---|
| `grain_contrasts()` | fits a mixed model over the whole ladder and reads Dunnett's comparisons off it, on lme4, lmerTest and emmeans. The Python twin would need a mixed-model fitter of its own or a scientific stack the wheel does not depend on, and nothing in the contract reads it. |
| `simulate_records()` | generates a record with a planted grain, for the vignette and the recovery tests. The Python suite builds its records in its own fixtures. |
| `plot()` on a ladder and on a selection | the wheel depends on numpy alone, and every number a plot draws is on the object it is called on. |

| in Python only | what it is |
|---|---|
| `flatten`, `align_folds`, `as_response`, `get_learner`, `cohen_kappa` | the helpers R keeps unexported, as `.flatten()`, `.as_folds()`, `.as_response()`, `.as_learner()` and `.kappa_table()`. A Python module namespace is flat, and anyone writing a learner or reading an artifact against this side reaches them. |

Models are the one thing neither side promises. A fit in torch and a fit in libtorch cannot be
byte-identical, and the encoders match module for module rather than number for number.
