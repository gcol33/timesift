# The representation contract

Normative for both implementations. R and Python must produce the same
numbers from the same input; where this document and either
implementation disagree, this document is right.

## Input

A long table of readings with three columns of interest:

| column | type | meaning |
|----|----|----|
| id | character, factor, or a whole number | the unit carrying the sensor (a plot, a site, a device) |
| time | POSIXct | the instant of the reading, read on the clock **The time zone** names |
| value | numeric | the reading |

An identifier is a name, and the two languages have to write the same
name for the same value. A character id is itself and a factor is its
label. A whole number is its digits, with no exponent and no decimal
point, so a plot read as `100000` from a file is `100000` in both,
rather than R’s `1e+05` beside Python’s `100000.0`. A number that is not
whole has no such writing and is refused, naming the column, as is a
column of any other type. This holds wherever an identifier names
something: the id column of the readings, of the targets, of the target
table a lookback is anchored by, and of a response.

Requirements, each checked and each an error rather than a warning:

- No missing `id` or `time`.
- Every `value` is a finite number. A missing one propagates through a
  sum and is skipped by a comparison, so a bin’s mean and its minimum
  would disagree about which readings were in it, and an infinity makes
  the mean of a bin holding both signs a not-a-number. The reading
  columns are read by the shared core, so this is one guard rather than
  one per language, and its message names the unit and the instant of
  the first reading it refuses.
- No duplicate `(id, time)` pair.
- Every id spans the same set of bins once binned. A record that stops
  early is not silently padded; it is reported with the ids and the bins
  concerned.

Ordering of the input rows carries no meaning, exactly and not
approximately: both reductions walk the record by unit, then by the
local clock, then by the instant, rather than in the order the caller
wrote it, so a record and any permutation of its rows reduce to the same
bytes and reach the same digest. Addition is not associative, and a
reduction that accumulated in the caller’s order would move in its last
bits under a permutation that changed nothing about the record. The
instant is the third key because two readings can share a local second,
on the night a zone sets its clock back, and an order that left them
tied would leave their sum to whatever the sort did with the tie. A
record already in that order is the common case and pays the scan that
establishes it. The output is ordered by sorted unique id and by bin
start.

## Ordering identifiers

Every place an identifier decides a position – the ids naming the first
dimension of the representation, and the variable names naming the cells
of the scorable mask – they are sorted by **C collation**: the byte
order of the UTF-8 encoding, which for every code point is also the code
point order. It does not depend on the session’s locale, on the machine,
or on the language.

That has to be stated because the two languages’ defaults disagree and
so do two R sessions in different locales. R’s
[`sort()`](https://rdrr.io/r/base/sort.html) follows `LC_COLLATE`, which
in an English locale orders `_z` before `a` before `A`, and NumPy’s
`np.unique()` orders by code point, which puts `A` first. Both
implementations therefore name the rule rather than take a default: R
passes `method = "radix"`, which sorts characters in the C locale
whatever `LC_COLLATE` says, and NumPy already sorts this way.

The failure this prevents is silent rather than loud. Two orderings hold
the same numbers in different rows, so a response matrix built in one
language and a representation built in the other line up row for row
while naming different units, and nothing errors. `series_order.csv`
carries ids the two rules order differently, so the fixtures fail rather
than the user.

The channel names are never sorted: the channel order is the order the
caller named the statistics in.

## The time zone

Bin membership is decided from the calendar the series is carried in, so
the zone has to be named before anything is binned. It is named once, at
the edge of each implementation, and nothing below that edge knows a
zone exists: the binning and the reduction see only *naive local
seconds*, the count of seconds from 1970-01-01T00:00:00 on that
calendar. A day there is 86400 of them whatever the night did, and a
month is what the proleptic Gregorian calendar says.

|  | how the zone is named |
|----|----|
| R | the `tzone` attribute of the `POSIXct` column; unset means UTC |
| Python | the zone the time column carries, where it carries one, else the `tz` argument; `None` and no zone on the column mean the instants already read as the calendar to bin by, which is what a zone-free `datetime64` says |

The same instants and the same zone give the same answer in both
languages, and the fixtures pin that rather than leaving it assumed. On
the Python side reading the column as instants drops the zone it was
written on, so it is taken off the column before that; a `tz` naming a
different zone beside one the column carries is an error, because two
zones are two answers. A zone name the database does not know is an
error on both sides, never a warning and a calendar in UTC.

The `native` grain is the one grain not read on that clock. Its bin is
the reading itself, and a reading is its instant: the two readings of
the hour a zone repeats when it sets its clock back share a local second
and are two bins, not one bin holding two readings, so the record read
at `native` is the same array whichever zone it is carried in, and its
bin starts are the instants themselves.

Reading an instant as a clock is defined for every instant in every
zone. The reverse is not: on the night a zone moves its clock forward a
local time exists on no instant, and on the night it moves back a local
time exists on two. A bin start is a local time, so reporting it as an
instant needs a rule, and the rule is that **a local time the clock
skipped resolves to the instant the clock jumped to, and a local time
the clock repeated resolves to the first of the two**. In
`America/Sao_Paulo`, whose clock moved at midnight until 2019, the day
beginning 4 November 2018 therefore opens at 01:00 local rather than at
a midnight that never happened, and holds 23 readings rather than 24.

Instants are read at whole seconds. Two readings a fraction of a second
apart are the same reading twice, and are reported as a duplicated
`(id, time)` pair.

## Grains

Bin membership is read from the calendar rather than from a running
count of hours, so a bin is a real week or a real month rather than a
drifting block of 168 or 730 hours.

| grain     | bin                                              |
|-----------|--------------------------------------------------|
| `native`  | the reading itself, no reduction                 |
| `halfday` | 00:00-11:59 and 12:00-23:59 of each calendar day |
| `day`     | the calendar day                                 |
| `week`    | ISO week, Monday to Sunday                       |
| `month`   | the calendar month                               |
| `season`  | three calendar months, counted from `year_start` |
| `year`    | the year running from `year_start`               |

`year_start` is a `"MM-DD"` string, default `"01-01"`. It sets the
boundary of the year, a hydrological one where it is not the first of
January, and therefore also the phase of the seasonal bins. A seasonal
bin is three calendar months counted from that anniversary, so a record
of three hydrological years beginning on it holds twelve of them and no
partial one. Cutting seasons anywhere else, at the equinoxes and
solstices for instance, is a different calendar and is passed as a
function; see Custom bins.

Bins are contiguous and cover the record with no gap and no overlap, and
both halves of that are asserted:

- every `(id, bin)` cell holds at least one reading, which is what makes
  a bin the same span for every id and a record that stops early an
  error rather than a padded row;
- consecutive bin starts are one bin apart on the grain’s own calendar.
  A bin no id reaches is never built, so a February missing from every
  logger would otherwise give four “adjacent” monthly bins with February
  simply gone, and a convolution would read January and March as
  neighbours.

The second is not asserted for `native`, where the bin is the reading
itself and the bin sequence is the record’s own sampling grid rather
than a calendar, nor for a supplied calendar, which declares its own bin
lengths and is contiguous by construction. It is asserted in local time,
so a sequence stepping across a clock change is contiguous: the civil
day a zone shortened is still one bin of one calendar day.

## Partial bins

A bin is **partial** when the record does not cover its whole calendar
span. The record covers from its first reading to its last plus one
sampling interval, taken as the smallest gap between consecutive
distinct reading instants, and a bin is partial when its own span
reaches outside that. Only a bin at an end of the record can, because
every id is required to hold readings in every bin between them.

Which bins those are follows from where the record starts and stops
against the calendar, not from the grain alone. Three years of hourly
readings from 1 September on a `"09-01"` boundary carry no partial
month, season or year, and a partial week at each end, because 1
September 2021 is a Wednesday. A record from an arbitrary deployment
date carries one at each end of almost every grain.

[`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md)
reports the verdict as `bin_partial` and takes a `partial` argument
saying what becomes of such a bin: `"keep"`, the default, returns it
alongside the full bins; `"drop"` removes it, and errors rather than
returning an empty representation if that leaves no bin. Dropping is a
choice about the record, not about the implementation: it discards up to
three months of readings at each end of a seasonal grain, while keeping
gives a bin whose mean is taken over fewer readings and whose `cold_day`
and `warm_day` are drawn from fewer days, so they sit closer to that
bin’s mean than a full bin’s would. `bin_n` gives the count each bin was
reduced from.

Both implementations obey the same rule and the fixtures pin both
settings.

## Statistics

Each named statistic becomes one channel of the output. A grain may
carry any subset, and the channel order in the output is the order given
by the caller.

| name | definition | defined for |
|----|----|----|
| `mean` | arithmetic mean of the readings in the bin | every grain |
| `min` | smallest single reading in the bin | every grain |
| `max` | largest single reading in the bin | every grain |
| `cold_day` | smallest daily mean among the days in the bin | `day` and coarser |
| `warm_day` | largest daily mean among the days in the bin | `day` and coarser |
| `mean_daily_min` | mean over the bin’s days of each day’s smallest reading | `day` and coarser |
| `mean_daily_max` | mean over the bin’s days of each day’s largest reading | `day` and coarser |

“A day or coarser” is decided from the bins rather than from the grain’s
name: a day-level statistic requires every calendar day of the record to
lie entirely inside one bin. Naming `native` or `halfday` is refused
before any data is read; a supplied calendar that cuts inside a day is
refused once the bins are in hand, naming the day it splits and the two
bins it splits it between.

The four day-level statistics reduce each calendar day first and then
reduce again over the days of the bin. `cold_day` and `warm_day` take
the extreme of the daily means; `mean_daily_min` and `mean_daily_max`
take the mean of the daily extremes. They are not `min` and `max`, which
act on single readings, and the difference is the point: an extreme day
is a state the site was in, an extreme reading can be one hour, and an
average daily extreme is the exposure a typical day of the bin brought.

Two orderings follow from the definitions, and both test suites assert
them on every bin of every grain, on the core’s output and on the
oracle’s alike: `min <= mean_daily_min <= mean <= mean_daily_max <= max`
and `min <= cold_day <= mean <= warm_day <= max`. The two day-level
pairs are not ordered against each other, and a bin whose days differ
widely in level is where they part: a bin of one day at 0 and one at 10
has `cold_day` 0 and `mean_daily_min` 5.

At the `day` grain `cold_day`, `warm_day` and `mean` coincide by
construction, as do `mean_daily_min` with `min` and `mean_daily_max`
with `max`. Requesting them there is allowed and returns the identical
channels.

## Custom bins

A caller may supply the binning instead of naming a grain, as a function
of the reading instants returning the start of each reading’s bin.
Everything downstream is unchanged: the bins are still required to tile
the record, and the output still carries the bin starts as its second
dimension’s names. This is how a calendar the package does not carry is
used, such as seasons cut at the equinoxes and solstices rather than on
the first of a month.

Such a calendar owns its own bin lengths. A record beginning inside one
of its seasons gives a leading bin of a few weeks beside neighbours of
three months, and that is the calendar the caller asked for rather than
a bin the record failed to fill: it is what makes three years cut at the
equinoxes thirteen bins where three calendar months from `"09-01"` are
twelve. The function declares where its bins begin, so the package
cannot know where the last one was meant to end and takes the record’s
end as its end; that final bin is never reported partial, and a leading
bin is judged as any other.

The function must return a bin start for every reading, including any
that precede its first boundary. Deciding that with an interval lookup
is the natural way to write one and the two languages disagree below the
first boundary, where R’s
[`findInterval()`](https://rdrr.io/r/base/findInterval.html) gives 0 and
NumPy’s `searchsorted() - 1` gives -1: the first silently shortens the
result, and the second silently wraps to the last boundary. Either put
the first boundary at or before the record’s first reading, as the
fixtures do, or handle the readings below it explicitly. A reading the
function returns no bin start for, an `NA` in R or a `NaT` in Python, is
refused before the bins are read, naming how many there are and the unit
and instant of the first: below the boundary a missing time is not
distinguishable from one at the beginning of time, and the bin it would
open there would hold the reading its real bin then lacks.

Two things have to hold of what the function returns, and both are
checked before its bins are read as bins, naming the first reading that
breaks them:

- A bin begins at or before every reading it holds. A calendar shifted
  by one boundary breaks it, and so does the wrap above, whose bin start
  is the last edge of all and therefore later than the reading it was
  asked about.
- A bin’s readings are a stretch of the record. A calendar sending
  alternate readings to two bins breaks it, and the two bins the array
  then carries are not bins the record ever had.

Neither is visible to the empty-cell guard or to the contiguity rule,
which read the bins the calendar declared and can only ask whether every
unit reaches each of them.

## Output

A numeric array of shape `[n_id, n_bin, n_channel]`.

- Dimension 1 is named by the sorted unique ids.
- Dimension 2 is named by the ISO-8601 timestamp of each bin’s start, in
  UTC.
- Dimension 3 is named by the statistic.

Attributes carried on the array:

| attribute | content |
|----|----|
| `grain` | the grain name |
| `stats` | the statistic names, in channel order |
| `year_start` | the `"MM-DD"` boundary used |
| `bin_start` | the bin start instants, resolved from local time as **The time zone** describes |
| `bin_end` | the last reading instant assigned to each bin |
| `bin_n` | a `[id, bin]` matrix of how many readings fell in each cell |
| `bin_partial` | a logical vector marking the bins the record does not cover for their whole calendar span |

No standardisation, centring or scaling happens here. Scaling is a
property of a fit and belongs to the fold it is computed on, never to
the representation, because computing it over all ids would leak the
held-out units into the training input.

## The lookback

The second reduction, and the one no calendar expresses. Its input is a
table of **targets** beside the readings: one row per thing to predict,
carrying the unit whose record it reads and the instant it is anchored
at. Two targets on the same unit a fortnight apart read two different
stretches of that unit’s series, which is why the bins are placed
relative to the target rather than to a month or a week.

| column | type                | meaning                                 |
|--------|---------------------|-----------------------------------------|
| id     | as the readings’ id | the unit, which the readings must carry |
| at     | POSIXct             | the instant the target is anchored at   |

Both columns are read **by name**, as every alignment in the package is,
and a table missing either is refused. A table carrying more columns
than these two is read for these two.

A target’s identity is its position in that table. A unit may carry any
number of targets, so the unit cannot name a row, and the output is in
the table’s own row order rather than in a sorted one.

### The bin arithmetic

Three numbers describe a lookback: `span`, its length; `lag`, the gap
between the anchor and the end of the lookback; and `bins`, how many
sub-bins it is cut into, oldest first. With `a` the target’s anchor and
`step = span / bins`, bin `b`, counted from zero, covers naive local

    [a - lag - span + b * step,  a - lag - span + (b + 1) * step)

closed at the left and open at the right, so a reading on a boundary
belongs to the later bin. `span` must divide by `bins` exactly; one that
does not is an error rather than a rounded step. Only the readings whose
unit is the target’s are read: a lookback is a stretch of one unit’s
record.

Nothing else changes. The statistics are the seven **Statistics**
defines, computed over the readings of a cell exactly as they are over
the readings of a grain’s cell, and the day-level pair reduces the days
of a bin oldest first.

### The two guards

- Every `(target, bin)` cell holds at least one reading. A lookback
  reaching past either end of the record is an error naming the target
  and the interval, never a padded row, for the reason a grain’s empty
  cell is one: an invented value in front of a model is worse than a
  target the record cannot answer for.
- The four day-level statistics reduce each calendar day first, so they
  are defined only where every calendar day lies whole inside one bin. A
  calendar settles that by itself; a lookback has to be asked, and the
  answer is two conditions rather than one. `step` must be a whole
  number of days, which holds for every target or for none, and
  `a - lag - span` must fall on a day boundary, which is a property of
  the anchor: a table of anchors on the hour rules the four statistics
  out where the same anchors at midnight allow them. Either failing is
  an error naming the target.

### Durations

`span` and `lag` are a count and a unit, or a bare count of seconds. **A
year is 365 days and a month is 30 days.** A lookback of a fixed length
is a fixed length rather than a calendar step: comparing two targets’
representations means each read the same amount of record, and a
February or a leap year takes that away. Where the calendar is what
matters, a grain is the reduction that follows it.

The length is measured on the local clock, as everything below the zone
boundary is. In a zone that moves its clock, a lookback of one day
ending at a local midnight holds the whole local day before it, which is
25 hours of record on the night the clock is set back and 23 on the
night it is set forward; a lookback that spans neither night holds 24.
That is what keeps a calendar day whole inside a bin for the four
day-level statistics, which a length fixed in instants could not: a
day-level lookback anchored within `span` after a transition would then
have to be refused. The difference is one hour twice a year, against the
whole day a February differs by.

| unit                | seconds  |
|---------------------|----------|
| `second`, `seconds` | 1        |
| `minute`, `minutes` | 60       |
| `hour`, `hours`     | 3600     |
| `day`, `days`       | 86400    |
| `week`, `weeks`     | 604800   |
| `month`, `months`   | 2592000  |
| `year`, `years`     | 31536000 |

A string is a count, optional space, and one of those names, upper or
lower case: `"30 days"`, `"12 hours"`, `"1 year"`. A string of digits
alone is seconds, and so is a number. Nothing else parses, and a
duration that does not parse is an error naming the argument.

### The output of a lookback

A numeric array of shape `[n_target, n_bin, n_channel]`, traversed and
digested exactly as a grain’s is, target fastest.

- Dimension 1 is named by the target’s own row label where the target
  table carries one, and by its position, from 1, where it does not. R
  takes the row names of the `at` data frame; Python has no row names
  and always names by position.
- Dimension 2 is named by where the bin opens relative to the anchor,
  oldest first, written as a signed count and the coarsest of `day`,
  `hour`, `minute` and `second` that divides it exactly, singular at
  one: `-30 days`, `-1 day`, `-12 hours`. In a grain an instant names a
  bin; here no two targets share one.
- Dimension 3 is named by the statistic.

Attributes carried on the array:

| attribute | content |
|----|----|
| `grain` | `"lookback"` |
| `span`, `lag` | the durations, resolved to seconds |
| `bins` | how many bins the lookback was cut into |
| `stats` | the statistic names, in channel order |
| `bin_n` | a `[target, bin]` matrix of how many readings fell in each cell |

There is no `bin_start`, no `bin_end` and no `bin_partial`. A bin is a
position relative to an anchor rather than a span of the calendar, and a
cell the record does not cover is an error rather than a verdict. R
carries no such attribute and Python carries them as `None`; a column of
not-a-time beside a column of `FALSE` would read as an answer to a
question the reduction does not ask.

## The channels

An array of readings is not the only thing a learner is handed. Two more
reductions put a channel beside those readings, and both return an array
a model reads, so both are normative here.

### Where a bin sits in the year

[`calendar_channels()`](https://gillescolling.com/timesift/reference/calendar_channels.md)
reads a representation and returns an array of the same units and bins
with two channels, `year_sin` and `year_cos`, identical across units. An
encoder that ends in global pooling discards when in the record a
thermal event happened, so the position of a bin in the year is given to
it as an input or it is not used at all; it is the time index of each
bin rather than a summary of the readings, and a sine and a cosine
rather than the fraction itself because the two are continuous across
the turn of the year where the fraction jumps.

For each bin, with `bin_start` and `bin_end` the instants the
representation carries:

    mid  = bin_start + floor((bin_end - bin_start) / 2)
    y    = the calendar year mid falls in, on the proleptic Gregorian calendar in UTC
    frac = (mid - first instant of y) / (length of y in seconds)

    year_sin = sin(2 pi frac)
    year_cos = cos(2 pi frac)

Four things in that are decisions rather than consequences.

The position is read at the **midpoint of the record the bin holds** –
`bin_end` is the last reading assigned to the bin, not the end of its
calendar span – so a bin the record only partly covers sits at the phase
it was actually measured over rather than at the phase of a whole one.

A bin spanning an **odd number of seconds** has its midpoint on a half
second, and the second it began on is the one it is read at. Every named
grain spans a whole number of hours, so nothing reaches that today; a
supplied calendar need not, and the rule is written down rather than
left to whichever language happens to round.

The year is read **in UTC**, on the instants, whatever clock the bins
were placed on. The phase is a place on the orbit rather than a reading
of a clock, and a zone moves it by its offset: under a day on a cycle of
a year, the same shift for every bin of the record.

`frac` is **exact arithmetic on the calendar** and the same bits on
every platform. The sine and the cosine of it are the platform’s
library, accurate to about an ulp and no further, so the contract pins
`frac` and states a tolerance of **1e-12** on the two channels. Both
suites read the fraction back and assert the channels against it.

The result carries every attribute of its input, with `stats` replaced
by `year_sin, year_cos`, no `static` channel, and `position` naming
every channel it holds. A lookback is refused: its bins are placed
relative to a target rather than on the calendar, so they have no
position in the year, and it carries no `bin_start` to read one from.

### Where a bin sits in the day

[`calendar_channels()`](https://gillescolling.com/timesift/reference/calendar_channels.md)
takes `cycles`, the cycles to place each bin in, `year` by default.
Naming `day` adds two channels, `day_sin` and `day_cos`, read at the
same midpoint:

    mid  = bin_start + floor((bin_end - bin_start) / 2)
    frac = (mid mod 86400) / 86400

    day_sin = sin(2 pi frac)
    day_cos = cos(2 pi frac)

with `mod` the floored remainder, so an instant before 1970 sits at its
place in its own day. The channels come two per cycle in the order
`cycles` names them, and `stats` names them the same way.

The day is read **in UTC**, as the year is. A zone moves the phase by
its offset and a site’s longitude moves the solar day against UTC by a
fixed amount; both are the same shift for every bin of a record, which a
model absorbs. A local clock’s summer time would instead move the phase
by an hour twice a year, which is a change in the input rather than a
rotation of it.

The day cycle is **refused on bins that sit a day or more apart**, read
as the smallest gap between consecutive `bin_start`s, and on a
representation of fewer than two bins, where no gap can be read. At a
day or coarser every bin would sit at the same place in the day, and a
constant channel is not a position. The half-daily grain passes and
alternates between the two halves of the day, which is what it records.
An unknown cycle name is refused. The fraction is pinned exactly and the
two channels to the same **1e-12** as the year’s.

### Putting channels side by side

[`bind_channels()`](https://gillescolling.com/timesift/reference/bind_channels.md)
takes two or more representations of the same units and bins and returns
one array carrying every channel, in the order the arguments are given
and, inside each argument, in its own channel order. It is how a
temperature reading, an external product such as snow cover, and the
calendar position of each bin reach a model as one input.

The result carries the **first argument’s** attributes, with `stats` the
joined channel names. Every other argument is read for its channels
alone; nothing of its own binning survives, which is why the units and
the bins have to agree in the first place. The two attributes that name
channels rather than bins, `static` and `position`, name those of every
argument, in the order the channels come.

Four inputs are refused, and the three that concern one argument name
its position from one:

| what | the message both raise |
|----|----|
| fewer than two arguments | `` `bind_channels()` needs at least two representations `` |
| an argument that is not a representation | `argument 2 is a ..., not a representation` |
| an argument covering other units or bins | `argument 2 covers different units or bins from the first` |
| two arguments carrying a channel of one name | `two representations carry a channel of the same name: mean` |

## What crosses the language boundary, and what does not

The binning and the reduction are one implementation: `src/ts_core.cpp`
and `src/ts_calendar.cpp`, compiled into the R package by R itself and
into the Python extension by CMake. So is the penalised fit,
`src/ts_penalised.cpp`, for the same reason: it is the arm the networks
are measured against, and a baseline that moved between the languages
would make the tool the confound. So is the tree, `src/ts_tree.cpp`.
What each language holds above them is the boundary, which resolves the
columns, resolves the zone, deals the folds and wraps the result. The
two agree by construction rather than by two implementations being
checked against each other after the fact.

The digests did not stop meaning anything when that happened. The
implementations they used to compare are kept as test oracles,
`tests/testthat/helper-oracle.R` and `python/tests/oracle.py`, reachable
from neither package at runtime and exercised only against the core on
the fixtures and on random series. The NumPy one was written from this
document rather than from the R source, which is what makes it evidence
that the document is complete. One implementation in production, two in
evidence.

The representation is normative and is checked byte-exactly. Three
things beside it are artifacts that both sides read rather than each
side computing: the response matrix, the fold map, and the mask of
scorable cells that follows from those two. A fold map built from a seed
in R and a fold map built from the same seed in Python are different
maps, because the two languages draw on different random streams; the
fix is not to align the streams but to build the map once and read it in
the other language. **The file format below is what makes that
possible**, and both sides carry the reader and the writer for all
three.

A model fitted in one language and a model fitted in the other cannot be
byte-identical and are not required to be. The penalised fit and the
tree are the models that are: each goes through the shared core, so the
two sides return the same coefficients, and grow the same tree, from the
same design, and what is stated below is how far either sits from glmnet
and from rpart rather than from the other.

## The file format of the three artifacts

One format for all three: CSV, UTF-8, a header row, `,` as the
separator, no quoting, and LF line endings on every platform. Written
the same way in either language, the same artifact gives the same bytes,
so a round trip through a file is checkable and is checked.

A number is written with `%.12g`: twelve significant digits, which
renders `0` and `1` as `0` and `1` and carries any measurement a
response holds. A logical is written `TRUE` or `FALSE`, and is read from
either that or `1`/`0`.

### The fold map: `id,fold`

| column | type      |                                    |
|--------|-----------|------------------------------------|
| `id`   | character | the unit                           |
| `fold` | integer   | the fold it is held out in, from 1 |

One row per unit, ordered by the id under **Ordering identifiers**. A
unit may appear once.

### The response matrix: `id` and one column per variable

| column                | type      |                                    |
|-----------------------|-----------|------------------------------------|
| `id`                  | character | the unit                           |
| each remaining column | numeric   | that variable’s value at that unit |

The columns after `id` are the variables, **in the file’s own order**,
which is the order the response carries them in; they are not sorted,
because a response’s column order is the caller’s. Rows are ordered by
the id. A presence-absence response holds `0` and `1` and nothing else,
and is checked for that when it is prepared rather than when it is read.

### The scorable mask: `variable,fold,n_occ,pres_train,abs_train,pres_test,abs_test,scorable`

One row per `(variable, fold)` cell, ordered by variable under
**Ordering identifiers** and then by fold ascending. The seven columns
after `variable` are integers except `scorable`, which is a logical. The
mask is a pure function of the response and the fold map, so it can be
recomputed rather than carried; it is written because reading it is how
a language that did not build it gets the exact cells the other one
scored on.

### A unit the file does not carry

Aligning any of the three to a representation’s units is by name, never
by position. A unit in the representation that the file has no row for
is an error, reporting how many are missing and naming the first of them
in the representation’s own order. A unit the file carries that the
representation does not is dropped without comment: a fold map covering
a whole study is a normal thing to read a subset of.

## Fixtures

Four series, because a record that starts on a bin boundary cannot tell
two binning rules apart, a record in UTC cannot tell two readings of a
zone apart, and identifiers that agree under every collation rule cannot
tell two row orders apart. `spec/fixtures/series.csv` is a synthetic
three-unit, 400-day hourly series beginning at midnight on the default
anniversary, so every coarse grain is in phase with it from the first
reading. `series_offset.csv` is a two-unit, 200-day series beginning at
05:00 on 17 October, which is what a logger deployed when someone could
walk to it gives, and puts every grain out of phase. `series_zoned.csv`
is a two-unit, 10-day series across 4 November 2018, the night
`America/Sao_Paulo` moved its clock at midnight, which is the record
that tells a calendar read by arithmetic apart from one read by writing
a local midnight and parsing it back. `series_order.csv` is a five-unit,
30-day series whose identifiers C collation and an English locale order
differently, `A1 P10 P9 _x a1` against `_x a1 A1 P10 P9`, and which
arrive in a third order again. `seasons.csv` holds the equinox and
solstice boundaries that make each series a caller-supplied calendar,
which is the only path the manuscript’s seasonal rung ever took.

The digests are the core’s own output: `inst/spec/make_fixtures.R` loads
the R package and calls the same public functions a user calls. They pin
a regression, not an agreement between two implementations. The evidence
that the two agree is the oracles, checked against the core separately,
on these fixtures and on random series.

`digests.csv` holds one row per series, grain, time zone, `year_start`,
`partial` setting and statistic, covering every grain-by-statistic
combination, each of the three-channel schemes (`min+mean+max`,
`mean_daily_min+mean+mean_daily_max`, `cold_day+mean+warm_day`), the
coarse grains at anniversaries other than the default, both `partial`
settings, the supplied calendar over the same grid of statistics a named
grain is read at, and the zone: every grain of the aligned series read
as a `Europe/Vienna` clock, which moves twice inside that record, and
the short series read as an `America/Sao_Paulo` clock, which moves at
midnight inside it, including a `year_start` landing on the night it
moves. Each row carries `n_unit`, `n_bin`, the first and last bin start,
how many bins are partial, and the digest.

`coverage.csv` holds what
[`coverage()`](https://gillescolling.com/timesift/reference/coverage.md)
reports, which is the same binning laid out as a count of readings per
`(unit, bin)` and the one reduction no digest above reaches: every
series here is complete, and a gap is what that table exists to show.
Each case names the readings it takes out, by unit and by span, so both
suites build the same record; a case takes none, a case takes a month
from one unit, a case takes a span from every unit so that the calendar
tiles over a bin no unit reaches at all, and a case reads a supplied
calendar. Each row carries the shape, the first and last bin, how many
cells are empty, how many units have a gap, how many bins no unit
reaches, and the digest of the count matrix.

`grain_guards.csv` holds one case per guard on a supplied calendar, each
naming the series and the calendar that breaks it beside the substring
of the message both implementations must raise. The calendars are named
rather than written out, so both suites build the same function: `late`
gives every reading the midnight after it,
`(floor(t / 86400) + 1) * 86400`; `alternate` sends consecutive readings
to two bins an hour apart, `t0 + 3600 * (((t - t0) / 3600) mod 2)` with
`t0` the record’s first reading; and `missing` bins by the calendar day
but returns no bin start, `NA` or `NaT`, for the reading exactly one day
after the record’s first.

The lookback reads the same series and three files of its own.
`lookback_targets.csv` holds the anchors, in named sets rather than one
set per series, because an anchor that is a local midnight in one zone
is not one in another and an anchor on the hour rules out the day-level
statistics that a midnight allows: `aligned` and `offset` sit on day
boundaries in UTC, `hourly` sits on the hour, and `zoned` sits on local
midnights in `America/Sao_Paulo`. `lookback_digests.csv` holds one row
per target set, zone, span, lag, bin count and statistic, covering every
statistic and each of the three-channel schemes, lags of none and of a
day and of half a day, one and three and four and seven bins, a step
that is a whole number of days and one that is not, a span written as a
bare count of seconds, and the zoned set read both as UTC and as the
clock that moves inside it. Each row carries `n_target`, `n_bin`, the
first and last bin’s offset, the unit of the first and last target, and
the digest. `lookback_guards.csv` holds one case per guard – a lookback
reaching past the record, a day-level statistic over bins shorter than a
day, and one over bins that do not open on a day boundary – with the
substring of the message the two implementations must both raise.

`channels_digests.csv` and `channels_guards.csv` pin the two channel
functions the way the reductions are pinned rather than each side
testing itself. A `calendar` row is a representation’s two channels and
a `bound` row is its readings with those channels beside them, over the
grains whose bins are a day, a week, a month, a season, a year and a
supplied calendar’s, a record out of phase with all of them, both
anniversaries of the two grains that count from one, both `partial`
settings, the two zones, and the two bin widths no other row reaches:
the `native` bin, whose start and end are one instant, and the `halfday`
bin, which is narrower than a day. What each row digests is the year
fraction, not the sine and the cosine of it, for the reason **The
channels** gives; the row carries the tolerance the two channels are
then asserted against. `channels_guards.csv` names each refused input
and the substring of the message both implementations must raise, and
both suites build the input from the case name: `x` at the weekly grain
of the aligned series, `other` at its monthly one, and `back` a 30-day
lookback anchored on its last reading.

Both test suites read the series, rebuild every row and assert all of
it. The shape is asserted before the digest, so two implementations that
put the record into a different number of bins are reported as that
rather than as an unexplained hash mismatch. A contract that carried one
series starting on the anniversary, at the default anniversary, with no
supplied calendar and no bin count beside the hash would pass while the
two sides disagreed on how many seasons three years hold, which is the
whole thing it exists to prevent.

The digest is defined byte-exactly, because a scheme that varies by
platform pins nothing:

1.  Traverse the array in its own order: unit fastest, then bin, then
    channel.
2.  Format each value with `%.12f`.
3.  Join with a line feed, terminate with one, encode UTF-8.
4.  MD5 of those bytes.

The line ending is LF on every platform. R’s
[`writeLines()`](https://rdrr.io/r/base/writeLines.html) emits CRLF on
Windows, so the bytes are written explicitly; a digest generated on
Windows and checked on Linux must agree.

The array has to be finite, and an array holding an infinity or a
missing value is refused rather than digested. `%.12f` writes an
infinity as `Inf` in R and as `inf` in Python, and R tells `NA` apart
from `NaN` where Python has one spelling for both, so any pinned
rendering would either compare the language or conflate two values R
keeps distinct. Nothing built here reaches that case: a reading that is
not a finite number is refused where the record is read, which is the
guard below.

Twelve places is far below any difference that could change a fitted
model and far above the noise from the two languages accumulating a mean
in different orders. Should a combination ever straddle a rounding
boundary at the twelfth place, the fix is to record that combination’s
tolerance in this document, not to loosen the scheme for everything.

A digest that moves without a matching change to this document is a bug
in whichever implementation moved. Regenerating the fixtures is a
deliberate act with its own commit.

## What else is pinned

The representation is not the only deterministic thing the two languages
share, and a contract that pinned it alone would let everything above it
drift. More fixtures, generated by the same script and read by both
suites:

`response.csv`, `folds.csv` and `cells.csv` hold one response of 40
units by 6 variables, a fold map of five folds over it, and the mask
that follows. The variable names order differently under C collation and
under an English locale, and the prevalences are chosen so the mask is
not all `TRUE`: one variable is present nowhere and one everywhere, so
neither has a scorable cell at all, and a rare one is scorable in some
folds and not others. Both suites read the response and the fold map,
recompute the mask, and assert it cell by cell. Both also write all
three back and assert the bytes, which is what makes the file format a
contract rather than a convention.

`metric_cases.csv` and `metrics.csv` hold eleven `(y, p)` cases and the
value of every threshold metric on each: `tss`, `roc_auc`,
`average_precision`, `kappa` under both rules, `decision_threshold`
under all four rules and the ten table metrics at the Youden cut and
`boyce`. The cases are where the tie rule is the whole answer – every
prediction tied, ties within a class, ties across the classes, one
presence, one absence, all presences, all absences, a perfect separation
and a reversed one. A metric a case defines no value on is written `NA`
rather than left out, so a suite that quietly skipped it fails rather
than passes.

[`table_metric()`](https://gillescolling.com/timesift/reference/table_metric.md)
reads a two-by-two table of decisions against observations, with `H` the
hits, `F` the false alarms, `M` the misses, `C` the correct negatives
and `n` their sum: `pod` is `H / (H + M)`, `pofd` `F / (F + C)`, `far`
`F / (H + F)`, `sr` `H / (H + F)`, `accuracy` `(H + C) / n`, `bias`
`(H + F) / (H + M)`, `or` `H C / (M F)`, `orss`
`(H C - M F) / (H C + M F)`, `csi` `H / (H + M + F)` and `ets`
`(H - h) / (H + M + F - h)` with `h = (H + M)(H + F) / n`. The cut is
the one a rule of
[`decision_threshold()`](https://gillescolling.com/timesift/reference/kappa_score.md)
selects, or a `threshold` given, and presence is predicted at
`p >= cut`. A zero denominator is `NA`, so the odds ratio of a table
with no miss or no false alarm is `NA` rather than infinite, and so is a
cell of one class. The cut rule `"mpa"` is the highest cut that predicts
presence at a share `perc` of the presences: the presences’ predictions
sorted in decreasing order, the `ceiling(perc * k - 1e-9)`-th of the
`k`.

[`boyce_index()`](https://gillescolling.com/timesift/reference/boyce_index.md)
is the Spearman correlation, over the windows where it is defined, of
the predicted-to-expected ratio with the window midpoint. With `lo` and
`hi` the least and greatest prediction, the window width is
`w = width (hi - lo)` and window `i = 0, ..., resolution` spans
`[from_i, from_i + w]` with
`from_i = lo + (hi - w - lo) i / resolution`, the last one closed at
`hi`. The ratio is the share of presences inside divided by the share of
all units inside, undefined where no unit is inside, and the result is
`NA` where fewer than three windows define it or it takes one value.
Both languages evaluate `from_i` in that order of operations, which is
what makes their windows agree to the last place. `metric_cases.csv`
carries a case of 25 units for it, the other cases being too small to
hold a unit in most windows.

`blocks_input.csv` and `blocks.csv` hold 23 units on three columns, with
ties in two of them, and the block each unit lands in under five cuts:
[`block_cv()`](https://gillescolling.com/timesift/reference/cv.md) on
two columns into four, three and five blocks, on one column into two,
and [`env_cv()`](https://gillescolling.com/timesift/reference/cv.md) on
three scaled columns into four. A block is cut by halving. With `k`
blocks to make from `n` units, the left part takes `k %/% 2` of the
blocks and `floor((2 n (k %/% 2) + k) / (2 k))` of the units, the units
being ordered by the column of widest range (the first on a tie in
range) and by order of arrival where tied on it. Blocks are numbered in
the order the halving makes them, left before right, and the fold of a
unit is its block.
[`env_cv()`](https://gillescolling.com/timesift/reference/cv.md) centres
each column and divides by its standard deviation with `n - 1` first,
and a column of no spread becomes zero. The resulting map is grouped by
block, so a split drawn inside a fold keeps a block whole.

`numeric_metric_cases.csv` and `numeric_metrics.csv` hold eleven
`(y, p)` cases and the value of every metric of a numeric response and
of ordinal classes on each. With `e = y - p`, `r_squared` is
`1 - sum(e^2) / sum((y - mean(y))^2)`, `pearson` the correlation,
`rmse`, `mse`, `mae` and `max_error` the root mean, the mean, the mean
absolute and the greatest absolute of `e`, and `poisson_deviance` the
mean of `2 (y log(y / p) - (y - p))`, the logarithm taken as zero at
`y = 0`; where `y` or `p` is constant the first two are `NA`, as is any
metric where `p` holds a value that is not a number, and the Poisson
deviance where `p` is negative or is zero beside a count above zero. For
the ordinal metrics each prediction is read as the observed class
nearest to it, the lower class on a tie, `m[i, j]` counts the units of
observed class `j` read as class `i`, and with `k` the classes observed
`accuracy` is `trace(m) / sum(m)`, `recall` is the sum of `m[j, j]` over
the column sums, `precision` the sum of `m[i, i]` over the row sums,
each over classes with a nonzero sum and divided by `k`, and `f1` is
`2 P R / (P + R)` of the last two. The cells of a numeric head hold a
cell scorable where each side of the split holds two distinct values.

`pa_pool.csv`, `pa_presences.csv` and `pa_candidates.csv` hold a pool of
36 units on a six-by-six grid of longitude and latitude, five presences
of which four are in the pool, and for each of four strategies which
units of the pool it admits: `random` every unit whose `id` is not a
presence’s; `sre` those outside the band between the 0.1 and 0.9
quantiles (Hyndman and Fan’s seventh definition) of each of two columns
over the presences, outside where a unit leaves the band in at least one
column and inside where it lies on a bound; `disk_planar` those whose
Euclidean distance to the nearest presence in the coordinates’ units
lies in `[0.5, 1]`; and `disk_lonlat` those whose haversine distance on
a sphere of radius 6371008.8 m lies in `[30000, 70000]`. Which of the
admitted units a draw picks is not pinned: it is the language’s
generator.

`contrast_cells.csv` and `contrast.csv` hold a fixed table of per-cell
scores for two arms, with cells one arm scored and the other did not,
and the paired contrast read off it. No model is involved: the pairing,
the per-variable mean, the interval on Student’s t with one degree of
freedom fewer than there are variables, and the signed-rank p-value with
the method it was read by are what the two languages own, and a fitted
model is what they are not required to share. The p-value is exact below
fifty per-variable differences holding no zero and no tie, and the
normal approximation with continuity and tie corrections otherwise.

`grain_contrast_cells.csv` and `grain_contrast.csv` hold one learner’s
per-cell scores at three grains, over twelve variables and four folds
with five cells missing, and the table R’s
[`grain_contrasts()`](https://gillescolling.com/timesift/reference/grain_contrasts.md)
reads off them against the learner’s best grain. The design is small so
the degrees of freedom are low enough that the way they are read moves
the critical value, and cells are missing so the fit is not the balanced
one a shortcut would get right by accident.

`simulate_design.csv` holds the design of
[`simulate_records()`](https://gillescolling.com/timesift/reference/simulate_records.md)
under each of its three temporal mechanisms, six variables over 400
days: the grain, its bin count, each variable’s anchor, the sum of its
squared weights over the readings, its driver’s population standard
deviation, and the link solved for a prevalence of 0.2 and an area under
the curve of 0.8. None of it is drawn, so all of it is pinned.

### How exactly

Everything named above is **byte-exact**, at the twelve significant
digits the format writes. It is arithmetic on the same finite inputs in
both languages, so anything less would be a difference worth finding
rather than a tolerance worth allowing. The one exception is the
signed-rank p-value: it is exact where the exact distribution applies,
which is fewer than fifty values with no tie, and the Python side
reaches the normal approximation beyond that through a Chebyshev fit to
the complementary error function accurate to about 1.2e-7 relative. The
fixture stays on the exact branch; where a caller lands on the other,
the two agree to 1e-6 and no closer.

The grain contrast is the other exception, and is not byte-exact at all.
Its differences are the fixed effects of a restricted fit each
language’s optimiser finds, and are asserted to 1e-6; its intervals and
its p-values are integrals of a multivariate t each language evaluates
by quasi-Monte Carlo, as emmeans does through mvtnorm, and are asserted
to 2e-4 and 2e-3. The simulator’s design is arithmetic on the same
inputs and is asserted to 1e-10, all but its link, which R’s root finder
settles to 1e-8 and which is asserted to 1e-7.

[`tss_inflation()`](https://gillescolling.com/timesift/reference/tss_inflation.md)
cannot be pinned as a digest, because it draws replicates from each
language’s own random stream, and aligning those streams would be the
wrong fix for the same reason it is the wrong fix for a fold map. Its
inflation figure is a headline claim of the package, so what is required
of it is stated rather than left to a hand check: **at 200 replicates
the two implementations agree on the inflation to within 0.02 at each
planted skill**, which is well inside the Monte Carlo error of either
one alone and far below the +0.110 the claim rests on. A disagreement
beyond that is a bug in one of them, not sampling.

## The penalised fit

[`elasticnet()`](https://gillescolling.com/timesift/reference/elasticnet.md)
is one elastic net per response, over `src/ts_penalised.cpp`, which both
languages compile. Its conventions are glmnet’s, because that is what
the arm has always been and the acceptance criterion for replacing it
was agreement with it rather than an elastic net of our own.

- Case weights are normalised to sum to one, and the objective is the
  mean deviance halved for a Gaussian family and the mean negative log
  likelihood for a binomial or a Poisson one, the Poisson one without
  its `log(y!)`, plus
  `lambda * (alpha * sum |b| + (1 - alpha) / 2 * sum b^2)`.
- Every column is centred on its weighted mean and divided by its
  weighted standard deviation, taken with those weights and no
  correction for degrees of freedom. A Gaussian response is centred and
  scaled the same way, which is what puts the reported penalties on the
  response’s own scale. A column holding one value has no spread to
  divide by, is left out of the descent, and is reported at zero.
- A penalty factor, where one is given, is rescaled to sum to the number
  of columns, so a factor of one everywhere gives the path no factor
  gives.
- The path is `n_lambda` penalties, geometric from the smallest that
  leaves every coefficient at zero down by a ratio of `1e-4` where there
  are more units than columns and `1e-2` otherwise. The largest is
  `max_j |g_j| / (vp_j * max(alpha, 1e-3))` on the standardised scale,
  with the floor under the mixing that gives a ridge a finite start.
- The path ends early where a step explains almost nothing more: a
  Gaussian family reads that share against the deviance explained so far
  and a binomial one reads it outright, which is the difference glmnet’s
  solvers carry. A Poisson family reads the share gained over the last
  four steps, the fifth point’s deviance explained less the first’s,
  relative to the deviance explained now, against ten times that
  threshold. It also ends where a fit explains more than `0.999` of the
  null deviance. Neither rule is read before the fifth point, and
  neither applies to a path of supplied penalties.
- The fit is iteratively reweighted least squares with a cyclic
  coordinate descent inside it, warm-started along the path, restricted
  by Tibshirani’s sequential strong rule and checked against the
  optimality condition on every column the rule discarded. A reweighted
  least squares step is judged on its own move, the coefficients it
  started from against the ones it reached, over the columns that have
  left zero and over the intercept, which is glmnet’s test.
- The cycle over the columns that have left zero is
  Anderson-extrapolated every five passes (Bertrand and Massias, 2021):
  the affine combination of the last six iterates whose successive moves
  come closest to cancelling is taken where the penalised quadratic is
  lower there than at the cycle’s own iterate. The objective only falls,
  and the descent still stops only on a pass that moved nothing.
- A Gaussian fit folds the root of each case weight into the
  standardised columns and into the intercept’s column, so its quadratic
  has unit weight.
- A Poisson response is a count under the log link, none of it negative
  and some above zero, and more than one value. Its null model is the
  weighted mean count, whose logarithm is the starting intercept (a mean
  of one where the fit has no intercept), and the quadratic there, the
  mean as the working weight and `y - mean` times the case weight as the
  residual, is what the largest penalty is read off. A reweighting
  rebuilds it at the current coefficients, the linear predictor held
  within 250 either side. Like a binomial fit it is reweighted least
  squares inside a coordinate descent, and its deviance explained is
  read against the null deviance of the weighted mean. A held-out case’s
  deviance is `2 (y log(y / mu) - (y - mu))` at the predicted mean, the
  logarithm taken as zero at `y = 0`.
- A fit that does not settle at a penalty, inside `max_irls`
  reweightings or inside what is left of the path’s `max_pass` passes,
  ends the path there: the points before it are the path, and `stalled`
  records the 1-based position of the penalty it did not settle at, `0`
  where the path ran to its end. That is glmnet’s `jerr = -m`. A fit
  that does not settle at the first penalty has nothing to return and is
  an error. A cross-validation records each fold’s `stalled` in fold
  order as `fold_stalled`, and reads a fold whose path ended early at
  its last point for every penalty below it, as glmnet’s `predict` reads
  a truncated path.
- A cross-validated penalty fits each fold along a path of its own and
  reads it at the whole-unit path’s penalties, interpolating between the
  two points around each, which is what `cv.glmnet` aligns on. The
  held-out deviance is averaged within a fold and then across the folds,
  weighted by each fold’s weight, and its standard error is the spread
  across the folds over `nfolds - 1`. `lambda.min` is the largest
  penalty of least held-out deviance and `lambda.1se` the largest within
  one standard error of it.
- The folds are dealt by the caller and handed over as one 0-based index
  per unit, so a grouping the outer map keeps whole stays whole where
  the penalty is chosen. Nothing inside the core draws.
- The fit on every unit and the fit of each fold are independent of one
  another, so `threads` runs them at once. What comes back is the same
  numbers either way, to the bit: the fits share the design they read
  and nothing else, and the held-out deviance is summarised after all of
  them have finished, in fold order. Both suites assert it.

### The fixtures

`penalised_input.csv` holds the design both suites fit: a weekly
representation of eighty simulated records, flattened, with the square
of every column beside it, which is what
[`elasticnet()`](https://gillescolling.com/timesift/reference/elasticnet.md)
penalises over. The squares are the scale case the standardisation
exists for, a reading of ten degrees and its square of a hundred, and
the weekly bins of one record are collinear the way adjacent bins are.
Beside it are a binomial response, the continuous driver it was
generated from, a case weight per unit and a five-fold map.
`count_response.csv` holds a count response for the Poisson family, read
by unit: the driver in standard units read as a rate `exp(0.4 + 0.6 z)`,
and that Poisson’s quantiles at a low-discrepancy sequence rather than a
draw, so no random stream is advanced.

`penalised_cases.csv` names eighteen cases, each family at each of three
mixings with and without the weights, and carries the convergence
threshold and the pass budget the reference was read at and the
tolerances a suite is allowed. `penalised_path.csv` holds glmnet’s
coefficients, its intercept and the objective at every tenth point of
each case’s path, and `penalised_cv.csv` holds the path’s length and the
penalty the cross-validation chose.

### How exactly

Not byte-exactly, and not by digest. Two implementations of a coordinate
descent settle at the same point to the tolerance they are run at and no
closer, so what is required is a distance:

- **The path’s length and its penalties are exact**, to `1e-10`
  relative. They follow from the gradient at the null model and a
  geometric ratio, so a difference there is a difference in the
  conventions rather than in the descent.
- **The objective is what is pinned tightly.** A suite’s fit may sit no
  more than `1e-6` above glmnet’s at the same penalty. Sitting below it
  is the fit being closer to the optimum than the reference, which is
  the direction the arm is allowed to move in; sitting above it is the
  arm being weakened, which is the thing this replacement was not
  allowed to do. Measured at a matched threshold of `1e-14`, the core
  sits between `3e-13` and `2e-7` above glmnet across the twelve
  Gaussian and binomial cases, and at most `6.7e-8` above it across the
  six Poisson ones.
- **A coefficient is allowed `1e-4`**, against the largest coefficient
  of the case. It is looser than the objective on purpose: two nearly
  identical columns split one coefficient between them differently in
  any two descents, and that split is not determined to the precision
  the fit is. On a design without collinear columns the two agree far
  closer, to `2e-6` at a matched threshold of `1e-14`.
- **The cross-validated penalty is exact.** `lambda.min` and
  `lambda.1se` are the same point of the same path in all eighteen
  cases, and the held-out deviance agrees to `1e-4` relative, to
  `2.2e-7` on the Poisson ones.

One point of the path is different by construction, and only for a
ridge. glmnet fits its first point at a penalty of `9.9e35` and reports
it under the largest penalty that would have left every coefficient at
zero. Above a mixing of zero those are the same fit, because the
threshold holds every coefficient down at that penalty; at a mixing of
zero there is no threshold, and the core solves the point it reports
while glmnet reports a fit made at an infinite penalty. The fixtures
therefore start at the second point, and the difference at the first is
asserted to be small rather than absent.

A Poisson path that ends on the rule that stops it, rather than at its
last penalty, ends on a comparison with a threshold, and glmnet’s Newton
iteration settles at a tolerance of its own. The generator refuses a
Poisson reference whose comparison at either of its last two points lies
within a thousandth of the threshold, so no length hangs on that
tolerance.

The reference is generated with glmnet’s own pass budget raised well
above its default. At the threshold the reference is read at, a
collinear design runs past that default, and glmnet then warns,
truncates the path and returns the penalties it did reach; the generator
refuses any reference glmnet warned about, so a fixture can never encode
a failure both implementations would have to reproduce to match.

## maxnet

[`maxent()`](https://gillescolling.com/timesift/reference/maxent.md) is
one Maxent model per response in maxnet’s formulation, over
`src/ts_maxnet.cpp`, which both languages compile, and which hands its
design to the penalised fit above. Its features, their regularisation
and its path are the maxnet package’s (0.1.4), because a biomod2 user’s
`MAXNET` is maxnet’s and the model here has to be the one they already
fit.

- The features are built over the rows fitted. A column holding one
  value there takes none. The classes are letters: `l` the column, `q`
  its square, `h` hinges, `t` thresholds, `p` the product of each pair
  of columns. Left open they follow the presence count: `l` under 10,
  `lq` under 15, `lqh` under 80, `lqph` from 80 on.
- The order is maxnet’s model matrix: every linear term, every square,
  each column’s hinges, each column’s thresholds, and the products, pair
  `(a, b)` with `a < b` ordered by `a` and then by `b`.
- Knots are R’s `seq(min, max, length.out = m)`: the ends exactly and
  the interior as `min + i * ((max - min) / (m - 1))`. A column’s
  `knots - 1` forward hinges run from each of the first `knots - 1`
  knots to the maximum and its `knots - 1` reverse hinges from the
  minimum to each of the last `knots - 1`, each
  `min(1, max(0, (x - lo) / (hi - lo)))`. Its thresholds are `x >= k` at
  points `3` to `knots + 1` of `seq(min, max, length.out = knots + 2)`,
  which is maxnet’s `[2:nknots + 1]`: 49 thresholds at the default of
  50.
- A feature’s regularisation is `maxnet.default.regularization()`: with
  `np` presences, the larger of `0.001` times the feature’s range over
  the rows fitted, a floor, and its standard deviation over the
  presences (over `np - 1`) times a class factor over `sqrt(np)`, all
  times `regmult`. The class factor is R’s `approx(rule = 2)` at `np` of
  a table. The linear, quadratic and product features share one table,
  chosen by the richest class present: the product table where a product
  is, else the quadratic one where a square is, else the linear one.
  Hinges read `0.5` and thresholds interpolate from `2` at none to `1`
  at 100. The floor is `0.5 * max(sd, 1 / sqrt(np)) / sqrt(np)` for a
  hinge, `1` for a threshold constant over the presences, and zero
  otherwise.
- The background formulation is maxnet’s model. Every unit is
  background, and each presence is appended to it again, after every
  unit and in the order the presences come, unless an absence carries
  exactly its readings in every column. The rows are weighted `1` for a
  presence and `100` for the background, and the lasso is fitted without
  standardisation, with the regularisation as its penalty factors,
  glmnet’s probability floor at `1e-8` as maxnet sets it, along the 200
  penalties
  `10^seq(4, 0, length.out = 200) * mean(reg) * sum(y) / sum(w)`, and
  read at the last. Its intercept is discarded: the link is
  `sum_k beta_k f_k + alpha`, with `alpha = -log(sum exp(link))` over
  the background rows and the entropy `-sum q log q` of
  `q = exp(link + alpha)`, both taken through the largest link. The
  outputs are maxnet’s: `exp(link)`, the cloglog
  `1 - exp(-exp(entropy + link))` and the logistic
  `1 / (1 + exp(-entropy - link))`.
- The background formulation reads no case weights, as maxnet takes
  none.
- The absence formulation fits the same features and penalty factors,
  without appending anything, as a logistic lasso under the caller’s
  case weights, along the penalised fit’s derived path of `n_lambda`
  points with glmnet’s default probability floor, and reads it at the
  cross-validated `lambda.min` or `lambda.1se` over the folds the caller
  dealt. It predicts the logistic of its own intercept plus the link.
- A fit keeps the features it gave a non-zero coefficient and nothing
  else, with each column’s range over the rows fitted and each kept
  feature’s range. Predicting with `clamp` holds each column inside its
  range, then each feature inside its own, as maxnet’s
  `predict(clamp = TRUE)`.
- A path that does not settle at a penalty is read at the last point it
  settled at, and `stalled` says where it stopped; maxnet stops with an
  error there instead.
- A design of more than `max_design` gigabytes, rows fitted times
  features times eight bytes, is refused before it is allocated, with
  its size in the message.

### The fixtures

`maxnet_cases.csv` names twelve cases on the weekly columns of
`penalised_input.csv`, without the squares: maxnet’s own classes at 27,
12 and 8 presences (`maxnet_response.csv` holds the two thinner
responses, the first 12 and the first 8 presences in unit order), each
richer class set, thresholds alone, hinges at `regmult = 2`, `lq` at
`0.5`, linear features without the presences added to the background, a
presence whose readings an absence repeats, and the absence formulation
with and without the case weights, cross-validated over the fixture’s
five folds. The reference is the maxnet package’s own fit, and
`cv.glmnet` over maxnet’s features for the absence formulation, run at a
threshold of `1e-14`. Each case carries the classes used, the rows and
features fitted, the penalty read, the objective there, and the
background’s entropy and `alpha`. `maxnet_regularization.csv` holds
every feature’s penalty factor and `maxnet_predict.csv` the predictions
on the fixture’s rows and, clamped, on every reading scaled by 1.3.

### How exactly

- **The features and their penalty factors are arithmetic** and are
  asserted to `1e-12` relative; the largest difference from maxnet is
  `7.4e-16`. The classes used and the counts of rows and features are
  asserted exactly.
- **The objective is pinned tightly**, to `1e-10` at the reference’s
  penalty, which is itself asserted to `1e-10` relative. At a matched
  threshold of `1e-14` the core and glmnet agree to `5.5e-14` across the
  twelve cases, the core below glmnet in five of them.
- **The predictions, the entropy and `alpha` are allowed `1e-4`.**
  Hinges at neighbouring knots are nearly collinear, and how a fit
  splits a coefficient between them is not determined to the precision
  the objective is. Across the twelve cases the predictions differ by at
  most `2.3e-5`, the entropy by `7.8e-8` and `alpha`, a number near
  `-10`, by `4.9e-5`.

## The tree

[`tree()`](https://gillescolling.com/timesift/reference/tree.md) is one
classification or regression tree per response, over `src/ts_tree.cpp`,
which both languages compile. Its rules are rpart’s, because a biomod2
user’s classification tree is rpart’s and the tree here has to be the
one they already fit.

- The design is column-major and every value finite; a non-finite value
  is an error rather than a surrogate split. A binomial response is 0 or
  1, and case weights are zero or more and sum to more than zero.
- A binomial family splits on the Gini index of the weighted class
  counts, a Gaussian one on the weighted sum of squares, a Poisson one
  on the Poisson deviance of the unshrunk rates. A node’s risk is the
  weight it misclassifies under its majority class, the first class
  winning a tie, its weighted sum of squares about its weighted mean, or
  its Poisson deviance about its shrunk rate. Its value is the weighted
  share of ones, that mean, or that rate.
- A Poisson response is a count, none of it negative, with some weight
  on a count above zero. A node holding events `S = sum w y` over
  exposure `W = sum w` predicts the rate `(S + a) / (W + b)`, the
  posterior mean of a gamma prior of coefficient of variation `shrink`
  centred on the rate of the units the tree is grown on:
  `a = 1 / shrink^2` and `b = a W_0 / S_0` from those units’ totals, and
  `a = b = 0`, the unshrunk `S / W`, where `shrink` is zero. Its risk is
  `sum 2 w (y log(y / rate) - (y - rate))` about that rate, the
  logarithm taken as zero at `y = 0`. A cut gains the deviance of the
  unshrunk rates it removes,
  `2 (S_l log(S_l / W_l) + S_r log(S_r / W_r) - S log(S / W))` with a
  term of no events counted as zero, as a share of the node’s risk, and
  sends the side of the lower unshrunk rate left. The tree grown on a
  cross-validation fold’s units takes its prior from those units, and a
  held-out observation’s loss is the deviance
  `2 (y log(y / rate) - (y - rate))` of the leaf it reaches.
- A column’s observations are sorted once, by rpart’s own quicksort, and
  every node reads them in that order. The order matters beyond the
  sort: a node’s sums are taken in it, and two sums of the same numbers
  in two orders can differ in the last place and turn a tie between two
  columns the other way.
- A split is tried only between two distinct values of a column, at
  their midpoint, with at least `min_leaf` observations on each side. A
  column’s best split is the first position reaching the largest
  improvement, and a node’s is the first column reaching the largest
  improvement across columns. An improvement at or below `1e-10` of the
  largest the fit has seen is read as none. Observations of zero weight
  are not offered to the split search.
- A node of fewer than `min_split` observations, or `max_depth` levels
  below the root, is not split. The complexity bookkeeping is rpart’s
  `partition()`: a split is kept where the risk its subtree removes, per
  split, is more than `cp` times the root’s risk, and a node whose
  subtree fails that is collapsed after its children were grown. The
  direction of a continuous split sends the side of lower mean, or of
  fewer ones, left.
- The complexity table is rpart’s: one row per distinct complexity,
  largest first, each on the scale of the root’s risk, with the number
  of splits and the relative risk of the tree pruned there. Where folds
  are given, each fold’s complement grows a tree under the complexity
  rescaled by its share of the weight, and every held-out observation is
  run down it at the geometric mean of each pair of adjacent rows;
  `xerror` is the weighted loss summed over the held-out observations
  and `xstd` its spread, both on the root’s scale.
- `prune` reads that table. `"se_sum"` is the row of least
  `xerror + xstd` among the rows that keep a split, the last of them
  where several tie, which is how biomod2 prunes its classification
  tree; `"one_se"` the first row within one `xstd` of the least
  `xerror`; `"min"` the first row of least `xerror`; `"none"` no
  pruning. Pruning at a complexity collapses every split whose own
  complexity is at or below it and keeps the table whole.
- The folds are dealt by the caller and handed over as one 0-based index
  per unit. Nothing inside the core draws.
- Every operation rounds on its own: the core is compiled with
  floating-point contraction off, so no multiply and add fuse into one
  rounding. A fused operation moves a sum by a unit in the last place,
  which is enough to turn a tie between two splits the other way, and
  compilers fuse by default on some machines and not on others.
- rpart’s class priors are the data’s class shares, divided back out by
  shares it computes again in C; the core takes every prior as exactly
  one. The two are the same numbers wherever the weighted class counts
  are exact, which integer weights make them. Under fractional weights
  rpart’s priors sit a unit in the last place from one, computed with
  R’s extended-precision sums, whose width depends on the machine, and a
  split whose complexity ties `cp` exactly can then fall the other way.
  The core falls the same way on every machine.

### The fixtures

`tree_cases.csv` names twenty-two cases: each family, with every weight
one and with the integer counts `tree_weights.csv` holds, under rpart’s
own defaults, biomod2’s tuned option set, and a shallow tree grown with
no complexity threshold at all, and four Poisson cases under the tuned
option set at a `shrink` of zero and of one half. Each is grown on the
design `penalised_input.csv` carries, with its five-fold map as the
cross-validation’s folds, and on the count response of
`count_response.csv` under the Poisson family. `tree_nodes.csv` holds
rpart’s node table, `tree_cptable.csv` its complexity table with the
cross-validated error, and `tree_predict.csv` the predictions of the
tree rpart’s `prune()` leaves at the row biomod2’s rule picks.

### How exactly

The node numbers, the split columns, the split directions, the node
counts and the number of splits in each row of the table are asserted
exactly. Every number of the node table and the complexity table, the
thresholds, weights, risks, complexities, values, cross-validated errors
and their spreads, and every prediction of the pruned tree, is asserted
to `1e-12` relative. On the twelve binomial and Gaussian cases the
largest difference from rpart is `1.1e-16`, and on the ten Poisson ones
`1.8e-14`.

## The forest

[`forest()`](https://gillescolling.com/timesift/reference/forest.md) is
one random forest per response, over the same core. Its split search is
the tree’s, and everything else it does is drawn, from one generator
this section defines, so that a forest is the same forest in either
language and on any number of threads.

### The generator

Every word is an unsigned 32-bit integer and every operation is taken
modulo 2^32.

- SplitMix32 advances a counter by `0x9E3779B9` and returns
  `z ^ (z >> 16)`, where `z = (c ^ (c >> 16)) * 0x85EBCA6B`, then
  `z = (z ^ (z >> 13)) * 0xC2B2AE35`, `c` the advanced counter.
- Tree `t` of a forest seeded `s` starts the counter at
  `s + 4 t * 0x9E3779B9`, which makes its state the outputs `4t + 1` to
  `4t + 4` of one SplitMix32 stream started at `s`. Those four words are
  the state `s0, s1, s2, s3` of a xoshiro128\*\* generator (Blackman and
  Vigna).
- An output is `rotl(s1 * 5, 7) * 9`; the state then moves by
  `t = s1 << 9`, `s2 ^= s0`, `s3 ^= s1`, `s1 ^= s2`, `s0 ^= s3`,
  `s2 ^= t`, `s3 = rotl(s3, 11)`.
- A uniform is an output times 2^-32, on `[0, 1)` and exact in a double.
  An index below `k` is that uniform times `k` rounded down, and `k - 1`
  should the product round up to `k`.
- A seed is taken modulo 2^32. The learners derive one per response as
  every other learner does, from the learner’s seed and the response’s
  name.

### The bootstrap

- The observations are drawn with replacement, each in proportion to its
  case weight: a draw takes a uniform times the total weight and returns
  the first observation, in row order, whose running sum of weights
  exceeds it. The running sum is taken one addition at a time. An
  observation of zero weight is never drawn.
- A forest draws as many observations as there are. A balanced one draws
  from the zeros alone and then from the ones alone, each as many times
  as the smaller class holds observations of positive weight; a class of
  no weight is an error, and so is `balance` on a Gaussian family.
- The draws make one list of rows: every row in ascending order,
  repeated as many times as it was drawn. The order in which the draws
  came does not reach the tree.

### The tree of a forest

- Each drawn observation weighs one. The tree is grown depth first, a
  node before its children and its left child’s subtree before its right
  child’s, and every draw a node makes is made in that order.
- A node holding fewer than `2 * min_leaf` observations, or observations
  whose responses are all equal, is a leaf. Nothing is pruned and there
  is no complexity threshold: a node is split wherever a split improves
  on it.
- A node that may be split first draws its columns. The tree keeps one
  arrangement of the column indices, the identity at its root and never
  reset between nodes; for `c` from 0 to `mtry - 1`, position `c` is
  swapped with position `c + j`, `j` an index below `p - c`. The first
  `mtry` positions are the node’s columns, tried in ascending order.
- For each column the node’s observations are sorted by it, ties kept in
  the order the node holds them, and searched as the tree’s split search
  searches: a split only between two distinct values, at their midpoint,
  at least `min_leaf` observations on each side, the Gini index of the
  counts or the sum of squares, the first position reaching the largest
  improvement. The node takes the first column reaching the largest
  improvement, and an improvement at or below `1e-10` of the largest the
  tree has seen, updated with the improvement read, is none.
- The node’s observations go left or right by its split, each side
  keeping the order the node held them in.
- A leaf’s value is the share of ones among its observations, or their
  mean, the mean’s sum taken one addition at a time in the node’s order.
- The forest’s prediction for a row is the sum of its trees’ leaf
  values, taken in tree order, over the number of trees.

### The settings

`preset = "default"` is randomForest’s own defaults, which is what
biomod2’s default option set fits: 500 trees, `mtry` the square root of
the column count rounded down under a binomial family and a third of it
under a Gaussian or a Poisson one, never below one, and `min_node`, the
`min_leaf` above, one and five under the two. A forest cuts a count as
it cuts a continuous response, on the weighted sum of squares, and a
leaf holds the mean count of its draws. `preset = "bigboss"` is
biomod2’s tuned option set: 500 trees, `mtry = 2`, `min_node = 5`. A
setting given explicitly beats either, and an `mtry` above the column
count is the column count.

### The fixtures

No package grows a forest from this generator, so the reference is the
forest grown from this text in R alone,
`tests/testthat/helper-oracle-forest.R`, which computes the generator’s
arithmetic in doubles where every step is exact. `forest_cases.csv`
names eight cases on the tree’s design: each family, flat weights and
the tree’s counts, a balanced forest, one trying every column, and a
seed of 2^32 - 1. `forest_nodes.csv` holds each tree’s node table and
`forest_predict.csv` the forest’s prediction for every unit, the
thresholds, the values and the predictions as hexadecimal floats: R
reads a seventeen-digit decimal exactly only where it has extended
precision, and on arm64 macOS it reads 5/6 a unit in the last place off,
where a hexadecimal float reads exactly on every platform and in both
languages. `forest_stream.csv` holds the generator’s first eight outputs
for five seed and tree pairs, which the Python suite also checks against
a reimplementation of the generator of its own.

### How exactly

Every field of every node, the thresholds and values included, and every
prediction are asserted exactly: the core and the oracle perform the
same operations in the same order, and no sum in a forest depends on a
machine’s extended precision.

## The boosted trees

[`boosting()`](https://gillescolling.com/timesift/reference/boosting.md)
is one gradient boosted model per response, over `src/ts_boost.cpp`,
which both languages compile. Its first-order trees are gbm’s and its
second-order ones xgboost’s exact greedy trees, because those are what a
biomod2 user fits as `GBM` and `XGBOOST`, and their draws come from the
forest’s generator.

### The fit

- The score of every observation starts at `log(S / (T - S))` under a
  binomial family, `S / T` under a Gaussian one and `log(S / T)` under a
  Poisson one, `S` the sum of weight times response and `T` the sum of
  weights, each taken in row order. A binomial fit with no weight on one
  class is an error, and so is a Poisson one with no weight on a count
  above zero.
- Tree `t` of fit `f` draws from the stream `f * trees + t` of the fit’s
  seed, fit 0 being the fit on every observation and fit `g + 1` the one
  holding fold `g` out. It first draws its bag, gbm’s way: with `k` the
  floor of `subsample` times the fit’s observations, observation `i` of
  `m`, in row order, is kept where a uniform times `m - i` is below `k`
  less those kept so far. It then draws its columns: the floor of
  `colsample` times the column count, at least one, by the forest’s
  partial shuffle of a fresh arrangement of the column indices, tried in
  ascending order.
- The working response is, first order, `y - 1 / (1 + exp(-F))`,
  `y - exp(F)` or `y - F`; second order, the gradient `w (p - y)` and
  hessian `w max(p (1 - p), 1e-16)` under a binomial family,
  `w (mu - y)` and `w mu e^0.7` under a Poisson one, `w (F - y)` and `w`
  under a Gaussian one, `p` the logistic of the score `F` and `mu` its
  exponential.
- Each tree is grown on the bag’s observations, and each column’s
  observations are read in ascending order, ties in row order.
- A leaf’s value is its step times `shrinkage`, and every observation’s
  score moves by the value of the leaf it falls into. A score is the
  starting score plus the leaves’ values in tree order, and a binomial
  prediction the logistic of it.

### The first-order tree

- gbm’s tree is `depth` splits, grown best first. Every terminal node is
  searched once, when it is made, over every column in turn: a candidate
  between two distinct values, at their midpoint, with at least
  `min_leaf` observations on each side, improves by
  `lw rw (ls/lw - rs/rw)^2 / (lw + rw)`, `ls` and `lw` the weighted
  working response and the weight to its left and `rs` and `rw` to its
  right, the right side’s sums the node’s own less the left’s, taken one
  observation at a time. A node keeps the first split reaching its
  largest improvement.
- The terminal node split is the first reaching the largest improvement
  over all of them; none above zero ends the tree. A split leaves the
  left child in the node’s place in that order and appends the right
  child and then an empty node, gbm’s branch for a missing value, which
  holds nothing and is never split but keeps its place in the order.
- A Gaussian leaf’s step is its weighted mean working response, as its
  parent’s split left the sums. A binomial leaf’s is one Newton step,
  the sum of `w z` over the sum of `w (y - z) (1 - y + z)` over the
  bag’s observations in it, in row order, and zero where that is zero. A
  Poisson leaf’s is `log(sum w y / sum w exp(F))` over the bag’s
  observations in it, in row order, `-1` where it holds no count and `0`
  where it holds no exposure, and then held to at most `19` less the
  greatest score in the leaf and at least `-19` less the least.

### The second-order tree

- xgboost’s tree is grown level by level to `depth`. Every node of a
  level is searched over every column in turn: a candidate between two
  distinct values, at their midpoint, with at least `min_leaf` of
  hessian on each side, gains
  `GL^2 / (HL + lambda) + GR^2 / (HR + lambda) - G^2 / (H + lambda)`,
  `GL` and `HL` the gradient and hessian to its left and `GR` and `HR`
  the node’s less those. A node keeps the first split reaching its
  largest gain, and is split where that gain is above `1e-6`.
- The grown tree is pruned from the leaves up: a split whose children
  are both leaves and whose gain is below `gamma` is collapsed, which
  can make its parent such a split in turn.
- A leaf’s step is `-G / (H + lambda)` over the bag’s observations in
  it, in row order, and zero where the denominator is zero. Under a
  Poisson family the step is held within `0.7` either side and a split
  gains what the held steps remove, `-(2 G w + (H + lambda) w^2)` at the
  step `w` of each side and of the node, which is `G^2 / (H + lambda)`
  wherever the step is not held.

### The number of trees

Where folds are given, each fold’s fit scores its held-out observations
after every tree by the weighted deviance,
`-2 sum w (y F - log(1 + exp(F))) / sum w`,
`-2 sum w (y F - exp(F)) / sum w` (the Poisson deviance without its
saturated term, which no tree changes) or `sum w (y - F)^2 / sum w`. The
error after tree `t` is each fold’s deviance times the observations the
fold holds, summed over the folds in order and divided by all the
observations, gbm’s `gbmCrossValErr`, and the fit on every observation
keeps its trees up to the first of least error.

### The settings

Under `method = "gbm"`, `preset = "default"` is gbm’s defaults as
biomod2 passes them: 100 trees of one split, `shrinkage = 0.1`,
`min_leaf = 10`, `subsample = 0.5`, no inner folds. `"bigboss"` is 2500
trees of seven splits, `shrinkage = 0.001`, `min_leaf = 5`,
`subsample = 0.5` and three inner folds. Under `method = "xgboost"`,
`"default"` is xgboost’s: 100 trees of depth 6, `shrinkage = 0.3`,
`min_leaf = 1`, `lambda = 1`, `gamma = 0`, every observation and column;
`"bigboss"` four trees of depth 2 at `shrinkage = 1`. `lambda` and
`gamma` are zero under gbm’s trees and an error to set otherwise.

### The fixtures

`boost_cases.csv` names seventeen cases on the tree’s design. Nine are
gbm’s, each family with flat weights and the tree’s counts, and a
cross-validated fit of each over the design’s five folds; gbm grows them
with `bag.fraction = 1`, each fold’s fit behind `nTrain` as `gbmDoFold`
arranges it, the Poisson ones under its `poisson` distribution on the
count response of `count_response.csv`. Six are xgboost’s
`tree_method = "exact"`, two of them its `count:poisson`, with `lambda`
and `gamma` set and, for a binomial or a Poisson response, the weights
`boost_weights.csv` holds: under equal weights the first round’s
gradients take two values, many splits then tie exactly, and the two
libraries’ sums break such a tie differently. Two draw a subsample and a
column sample and are grown from this text in R alone,
`tests/testthat/helper-oracle-boost.R`. `boost_predict.csv` holds every
case’s prediction for every unit and `boost_cv.csv` the cross-validated
error after every tree.

### How exactly

The first-order predictions and the cross-validated errors are asserted
to `1e-12` relative, the largest difference from gbm being `6.7e-16`;
gbm rescales the weights before it fits, which moves a sum in the last
place. The second-order predictions are asserted to `1e-5`, since
xgboost stores the design and the gradients in single precision; its
Poisson cases differ by at most `1.9e-7`.

## The envelope

[`envelope()`](https://gillescolling.com/timesift/reference/envelope.md)
is one surface range envelope per response, biomod2’s `SRE`, over
`src/ts_envelope.cpp`, which both languages compile.

- Over the rows whose response is one, each column’s `q` and `1 - q`
  quantiles, `q` the `quantile` setting in `[0, 0.5]`. The upper
  probability is formed as `1 - q` before it is used, as R forms it.
- The quantile is R’s type 7, the default of
  [`quantile()`](https://rdrr.io/r/stats/quantile.html) and the one
  `bm_SRE()` reads: over the `m` sorted values, the position
  `h = 1 + (m - 1) * prob`, the value at `floor(h)`, and where `h` lies
  above it and the value at `ceiling(h)` differs,
  `(1 - f) * lo + f * hi` with `f = h - floor(h)`.
- A row is predicted one where every column lies inside its band, both
  ends included, and zero elsewhere.
- The absences and the case weights are not read. A response with no
  presence, or with nothing else, is predicted its mean above the core,
  as every per-response learner predicts one.

### The fixtures

`envelope_cases.csv` names six cases on the weekly columns maxnet’s
fixtures read, over the three responses those carry: quantiles of `0`,
`0.025` at each presence count, `0.1` and `0.5`. `envelope_bounds.csv`
holds each column’s two bounds from `bm_SRE(do.extrem = TRUE)`, and
`envelope_predict.csv` its projection onto the fixture’s rows and onto
every reading scaled by `1.02`. At a quantile of zero a presence sits on
each bound, which is where an exclusive comparison would differ.

### How exactly

- **The bounds are arithmetic on the readings** and are asserted to
  `1e-12` relative; the core and biomod2 agree to the last bit.
- **The projections are asserted exactly.**

## The stepwise model

[`linear()`](https://gillescolling.com/timesift/reference/linear.md) is
one generalised linear model per response, its terms chosen by Akaike’s
criterion, over `src/ts_stepwise.cpp`, which both languages compile.
Each fit is R’s `glm.fit` and the search is MASS’s `stepAIC()`, because
the forward search over column terms is the arm the published comparison
ran and the rest is what a biomod2 user’s `GLM` is.

- **The terms.** A column holding one value over the rows fitted is not
  a term. Under `"column"` a term is the column’s orthogonal polynomial
  of degree `min(degree, distinct values - 1)`; under `"power"` each
  power `1` to `degree` of the column is a term of its own, the square
  taken as a product and any higher power through `pow`, as R’s `^`
  takes them. Terms are catalogued column by column, each column’s
  powers in order.
- **The orthogonal polynomial** is the three-term recurrence of R’s
  [`poly()`](https://rdrr.io/r/stats/poly.html): `P_0 = 1`,
  `norm2_0 = n`, and for `k = 1..d`,
  `alpha_k = sum(v P_{k-1}^2) / norm2_{k-1}`,
  `P_k = (v - alpha_k) P_{k-1} - (norm2_{k-1} / norm2_{k-2}) P_{k-2}`
  (the last term absent at `k = 1`) and `norm2_k = sum(P_k^2)`; the
  columns are `P_k / sqrt(norm2_k)`. The alphas and norms are kept with
  the fit and new rows are mapped through them, never through a basis
  re-derived from themselves.
- **The design** is the intercept followed by the columns of each term
  the model holds, in the order it holds them.
- **Each fit** is iteratively reweighted least squares as `glm.fit`:
  starting means `(w y + 0.5) / (w + 1)` under the binomial family,
  `y + 0.1` under the Poisson one and `y` under the Gaussian one; at
  each iteration the working response `eta + (y - mu) / mu_eta` and
  working weights `sqrt(w mu_eta^2 / V(mu))`, solved by LINPACK’s
  `dqrdc2` Householder decomposition with its limited pivoting at a
  tolerance of `min(1e-7, epsilon / 1000)`, a column falling below that
  share of its original norm moved to the end and given a coefficient of
  zero; and the stopping rule
  `|dev - dev_old| / (|dev| + 0.1) < epsilon`, `epsilon = 1e-8`, within
  25 iterations. The logit link holds the linear predictor at 30 either
  side as R’s does: below `-30` the mean is `eps / (1 + eps)` and above
  `30` it is `1 / (1 + eps)` with `eps` the machine epsilon, and the
  derivative is `eps` outside that range. The log link of the Poisson
  family takes the mean and its derivative as `max(exp(eta), eps)`, and
  its variance function is the mean. The rank is the decomposition’s at
  the last iteration.
- **The criterion** is `deviance + 2 rank` under the binomial family,
  `deviance - 2 sum w (y log y - y - log y!) + 2 rank` under the Poisson
  one, which is `-2 log L + 2 rank` with the saturated log likelihood
  put back, and, under the Gaussian one, \`n (log(2 pi deviance
  / n) + 1) + 2 - sum(log
  23. - 2 rank\`, the deviance weighted by the case weights in all
        three.
- **The search.** `"forward"` and `"both"` start from the intercept,
  `"backward"` and `"none"` from every term. `"none"` stops there.
  Otherwise each step fits the model with each held term dropped, where
  the search drops (`"both"`, `"backward"`), and with each term it lacks
  added, where it adds (`"forward"`, `"both"`) and holds fewer than
  `max_terms`. A move whose fit did not settle is not a candidate. If
  any drop leaves the rank where it was, the last such term is dropped
  and the step ends. Otherwise the moves that change the rank are
  compared in order, drops in the model’s order and then additions in
  catalogue order, and the first to reach the lowest criterion is taken
  if that is strictly below the model’s own; if none is, the search
  stops. A taken addition goes to the end of the model.
- **What a fit keeps**: each term’s column, power and degree, the
  recurrences, the coefficients, the rank, the deviance, the criterion,
  whether the last fit settled, and the number of steps. A model holding
  no term predicts the mean of the response over the rows fitted,
  unweighted. Otherwise the prediction is the linear predictor through
  the link.
- The candidate fits of one step are independent, and `threads` runs
  them at once without changing what comes back.

### The fixtures

`stepwise_cases.csv` names eighteen cases on the weekly columns maxnet’s
fixtures read. Ten are MASS’s `stepAIC()` over biomod2’s formula
`x + I(x^2)` (`I(x^3)` beside them in one) from the intercept or from
every term, in every direction, three of them under the Poisson family
on the count response of `count_response.csv`: two of them carry the
first column twice, so a term is aliased and the backward search drops
it before anything else. MASS’s binomial family rounds a fractional
weight into a count, so its binomial cases are unweighted; its Gaussian
criterion differs from the core’s by a constant, and one Gaussian case
is weighted, as is one of the Poisson ones. Three are
[`glm()`](https://rdrr.io/r/stats/glm.html) on every term, and five are
the forward search over column terms written in R alone,
`tests/testthat/helper-oracle-stepwise.R`, under fractional case
weights. Each case carries the terms chosen as `column:power` in the
order the model holds them, `0` for a polynomial, with the rank, the
deviance, whether the fit settled and the number of steps.
`stepwise_predict.csv` holds the fitted means on the fixture’s rows and
on every reading scaled by `1.01`.

### How exactly

- **The terms, the rank, the steps and whether the fit settled are
  asserted exactly.**
- **The deviance is asserted to `1e-9` relative and the predictions to
  `1e-8`.** The iteration is `glm.fit`’s own and the two settle on the
  same iterate; across the eighteen cases they agree to about `1e-14`.

## MARS

[`mars()`](https://gillescolling.com/timesift/reference/mars.md) is one
multivariate adaptive regression spline per response, over
`src/ts_mars.cpp`, which both languages compile. The forward pass is
earth’s (5.3.6) `ForwardPass`, the pruning pass leaps’ backward
elimination as earth calls it, and the refit earth’s, because a biomod2
user’s `MARS` is `earth(y ~ ., glm = list(family = binomial))`. The
least squares and the logistic refit are R’s `dqrdc2` and `glm.fit`,
from `src/ts_glm.cpp`, which the stepwise model shares.

- **The weights.** Weights that differ from the first by no more than
  `1e-8` are no weights: every weight is one below. Otherwise a weight
  below the mean over `1e8` is raised to it. The basis is weighted
  throughout the two passes: the intercept is `sqrt(w)` and every term
  is its parent times a hinge, so every term carries the root weight
  once.
- **The scaled response.** The forward pass reads
  `ys = (y - mean(y)) / sd(y)`, `sd` over `n - 1`, and a weighted fit
  also `yw = sqrt(w) y / sd(y)`. The null residual sum of squares is
  `sum(w (ys - wm)^2)`, `wm` the weighted mean, held at `1e-8 n` from
  below.
- **The settings.** `nk`, the most terms, defaults to
  `min(200, max(20, 2 p)) + 1`; `penalty` to 2 at degree one and 3
  above. The end span of a term of degree `d` is `endspan`, or where
  that is 0 `floor(7.32193 + log(p) / 0.69315)`, widened at `d >= 2` by
  `floor(adjust * span + 0.5)`, `adjust = 2`, and held to
  `[1, floor(n / 2) - 1]`. The min span is `minspan`, or where that is 0
  `floor((2.9702 + log(p m)) / 1.7329)` with `m` the units where the
  parent is positive, at least
  1.  The first knot counted from the top sits `max(1, endspan + s)`
      units in, `s` the remainder `a - floor(a / minspan) minspan`
      halved over the `a = max(0, n - 2 endspan)` units available (half
      the min span where it divides exactly, half of `a` where `a` is at
      most the min span). A negative `minspan` asks for `-minspan`
      knots: the min span is `ceiling(n / (1 - minspan))` and the first
      knot the smallest multiple of it at or past the end span, less
      one.
- **The order of a column** is its units sorted by value, ties in unit
  order.
- **The basis the search reads** is earth’s: an orthonormal basis of the
  terms held, built by modified Gram-Schmidt in the order they entered,
  the intercept `1 / sqrt(n)` and the first term centred on its mean, a
  term whose residual has a sum of squares of zero held as zero. A
  weighted fit keeps beside it the orthonormal basis of every column of
  the weighted basis so far, the unusable upper hinges included, each
  one whose residual norm is below `1e-8` of its own dropped, which is
  the basis earth’s QR regresses on.
- **A step.** For each of the first `fast_k` parents of the queue whose
  degree is below `degree`, and each column the parent does not already
  use, the step searches the column linearly and then its knots, and
  takes the candidate of greatest reduction in the residual sum of
  squares, the first in parent order and column order where two tie. A
  candidate is *of a new form* unless some term held uses the column and
  uses the same other columns as the parent; only then is the linear
  term a candidate and the pair’s upper hinge usable.
- **The linear candidate** is the parent times the column. Unweighted,
  it is orthogonalised against the basis, and where its residual’s sum
  of squares is at most `0.01` it is zero and the form is no longer new;
  its reduction is `(sum(ys q))^2`. Weighted, it is orthogonalised
  against the weighted basis, dropped where its residual norm is below
  `1e-8` of its own, and its reduction is the one the exact residuals
  give.
- **The knot search** scans the column’s order from its second largest
  value down to the end span. With `b` the parent and
  `h(t) = b max(0, x - t)`, it carries, as each unit enters above the
  knot, the running sums of `b`, `b^2`, `b x` and `b^2 x`, each centred
  column’s covariance with `b`, and the centred response’s covariance
  with `b`, and from them the centred covariance of `h(t)` with each
  basis column (`c_k`), with itself (`c`), and with the centred response
  (`g`): `c += dx (2 S(b^2 x) - S(b^2) (x0 + x1)) + (u^2 - v^2) / n`,
  `v = S(b x) - S(b) x0` and `u` its last value. A weighted fit centres
  on `sqrt(w) / |sqrt(w)|` instead of the constant, with the sums of `b`
  and `b x` weighted by it and no division by `n`. At a unit whose
  parent is positive, and unweighted whose `c` is positive, a counter
  from the first knot’s position runs down, and where it reaches zero it
  is reset to the min span and the knot evaluated:
  `r = g - sum(g_k c_k)` over the basis and the linear candidate, and
  `q = c - sum(c_k^2)`.
  - Unweighted, where `q / c` exceeds `0.01` (below the 15th column) or
    `1e-5`, the knot reduces the residuals by the linear candidate’s
    reduction plus `r^2 / q`, and it is kept where that beats the
    column’s best so far and is below `min(1.01 R, 10 D)`, `R` the
    residual sum of squares and `D` the last step’s reduction.
  - Weighted, where `q` exceeds `1e-10 c` the knot’s residual sum of
    squares is the base’s less `r^2 / q`, and it is kept where that is
    below the best so far by more than `1e-10`; the linear candidate is
    best where no knot is below it. earth fits the whole weighted basis
    by QR at every knot; this is the same residual sum of squares, and a
    hinge within `1e-10` of the basis is the column earth’s QR would
    drop.
- **A step’s term.** The parent’s factors and a new one on the column:
  at the knot’s value, the lower term `max(0, x - t)` over the units
  above the knot in the order and the upper term `max(0, t - x)` over
  the rest, or, for the linear candidate, the column itself at the
  column’s least value, with no upper term. The upper term is used only
  where the candidate was of a new form and, weighted, its residual on
  earth’s basis has a sum of squares above `0.01`; an unusable one keeps
  its place in the basis’s numbering but is never a parent.
- **Fast MARS.** A queue entry per term position in the order they
  entered, starting from the intercept at the null residual sum of
  squares. After each step a parent tried is given its best reduction
  and the count of term positions then, and the new terms enter at
  infinity. The queue is sorted by reduction, larger first and then the
  lower position, and ranked; then, where `fast_beta` is above zero, by
  `rank + fast_beta (positions now - positions at its reduction)`, the
  lower first, then the reduction and then the position. A parent whose
  degree is too high keeps its entry unchanged.
- **Stopping.** After each step the residual sum of squares is the last
  less the step’s reduction, within `1e-10` of zero taken as zero, and
  `R^2 = 1 - R / R_null`. The pass stops without the step where no
  candidate was found and, at a `thresh` above zero, where `R^2` rose by
  less than it (a rise within `1e-10` of zero taken as zero) or the
  generalised R-squared fell below `-10`; with it, where `R^2` reached
  `1 - thresh` or the term positions reached `nk - 1`. The termination
  code is earth’s: 1 at `nk` below 3, 2 or 3 at a generalised R-squared
  below `-10`, 4 at a small rise, 5 at the largest `R^2`, 6 where
  nothing was found, and 7 at `nk`.
- **After the pass**, the usable terms are decomposed by `dqrdc2` at
  `1e-8` and a term it moves past the rank is dropped.
- **The pruning pass** is leaps over the weighted basis and the
  weighted, unscaled response (`sqrt(w) y`): Miller’s orthogonal
  reduction (AS 274) row by row, the tolerances at `5e-10` of each
  column’s scaled row sums, the dependent columns dropped and the
  reduction rebuilt without them. Each subset size first records the
  prefix of the terms in their order. Backward elimination then runs
  from the last position down to the third, the intercept held in: the
  position whose drop raises the residuals least (the lowest where two
  tie, one below its tolerance at once) is moved to the end, and every
  prefix it moved past is recorded where its residual sum of squares
  beats the one held for its size and it is not the same subset. The
  generalised cross-validation of a size `k` is
  `RSS_k / (n (1 - C / n)^2)`, \`C = k + penalty (k
  - 1.  / 2`(0 at`penalty = -1`) and infinite from`C \>=
        n`; the first size of least value up to`nprune`is kept. Unpruned, the first`nprune\`
        terms are.
- **The refit** is over the kept terms unweighted, the weighted basis
  divided by `sqrt(w)`: under the binomial or the Poisson family
  `glm.fit` from `src/ts_glm.cpp` with the head’s weights, under the
  Gaussian one `dqrls` at `1e-7` over the terms and the response each
  times `sqrt(w)`. A dependent kept term is an error.
- **What a fit keeps**: every term of the forward pass as its factors
  (column, direction and cut), the kept terms in their order, their
  coefficients, the termination code, the kept size’s generalised
  cross-validation and whether the refit settled. The prediction is each
  kept term as the product of its factors, `max(0, x - t)`,
  `max(0, t - x)` or `x`, then the linear predictor, through the logit
  link under the binomial family and the exponential under the Poisson
  one.
- The columns of one parent are independent searches, and `threads` runs
  them at once; their results are taken in column order afterwards, so
  what comes back does not depend on it.

### The fixtures

`mars_cases.csv` names thirteen cases on the weekly columns maxnet’s
fixtures read: earth’s own fits over the binomial, the Gaussian and the
Poisson response, at degrees one and two, unweighted and under the
fractional weights, unpruned, with `nprune = 5`, and with `minspan = 3`,
`endspan = 5` and every term a parent. Each case carries every forward
term as `column:direction:cut` factors, the kept terms, the termination
code and the kept size’s generalised cross-validation. `mars_coef.csv`
holds the kept terms’ coefficients, the logistic refit’s under the
binomial response and the Poisson one’s under the Poisson, and
`mars_predict.csv` the predictions on every reading scaled by `1.01`.
The weighted cases are earth’s QR at every knot.

### How exactly

- **The forward terms, the kept terms and the termination code are
  asserted exactly**, the cuts to `1e-12` relative: a cut is a reading.
- **The coefficients and the generalised cross-validation are asserted
  to `1e-8` relative and the predictions to `1e-9`.** Across the
  thirteen cases the core and earth agree to `8e-14` or better, weighted
  cases included.

## The discriminant

[`discriminant()`](https://gillescolling.com/timesift/reference/discriminant.md)
is one flexible discriminant analysis per response, over
`src/ts_fda.cpp`, which both languages compile. A biomod2 user’s `FDA`
is `mda::fda(y ~ ., weights = w, method = mars)` under mda’s (0.5-5)
defaults, so the basis is mda’s own MARS and not earth’s that
[`mars()`](https://gillescolling.com/timesift/reference/mars.md)
reproduces; the scoring, the variate and the posterior are those of
`fda()` and `predict.fda()`. The least squares are the Householder QR
with limited pivoting of `src/ts_glm.cpp`, under R’s rank rule, and the
recalibration its IRLS under a probit link, with the normal distribution
computed as R computes it (Cody’s `pnorm`, Wichura’s AS 241 `qnorm`,
`dnorm` with the argument split above five) in `src/ts_normal.cpp`.

- **The classes.** The response is zero or one with both present; the
  first class is the zeros. The priors are the classes’ unweighted
  shares, `n_j / n`.
- **The scores.** The weights are rescaled to `w n / sum(w)` and the
  classes’ weighted shares are `d_j = sum_{i in j} w_i / n`. With
  `s_j = sqrt(d_j / (d_1 + d_2))`, the 2 by 2 matrix of rows
  `(s_j, c_j s_j)`, `c = (-1, 1)` the Helmert contrast, is decomposed by
  `dqrdc2` at `1e-7`; the scores are its Q’s second column over `s_j`,
  `theta_j = (Q e_2)_j / s_j`, and each unit’s scored response is its
  class’s score.
- **The weights do not reach the basis.** `marss` sets them to one
  before its forward pass, so the basis is the unweighted fit to the
  scored response. Two scores are an affine map of the response, which
  leaves every criterion of the pass unchanged, so the terms are the
  same under any weights; the weights move the scores, the variate and
  the recalibration.
- **The settings.** `nk`, the most terms, defaults to
  `max(21, 2 p + 1)`, an even one taken one lower; `penalty` to 2 at
  degree one and 3 above; `thresh` to `0.001`. `marss` writes two
  tolerances as single-precision literals, and they are compared at the
  single-precision values: `0.01f` (`tolbx`) and `1.01f`.
- **The order of a column** is its units sorted by value, ties in unit
  order.
- **The basis the search reads.** Each term’s column orthogonalised
  against the terms before it by one pass of projections (`orthreg`),
  then scaled to unit norm; the intercept is `1 / sqrt(n)` and the first
  hinge is centred on its mean. A column’s mean is taken before it is
  scaled.
- **A step.** For each term of degree below `degree` as the parent, in
  term order: its min span is `int(-log2(-log(0.95) / (p m)) / 2.5)`,
  `m` the units where the parent is positive, and its end span
  `int(3 - log2(0.05 / p))`. For each column the parent does not use, in
  column order, the candidate is *of a new form* unless some term held
  uses the column with the parent’s other columns. A new form offers the
  column itself, the parent times the column orthogonalised against the
  basis, as a candidate: where its sum of squares is at most `0.01f` it
  is zero and the form is no longer new, and otherwise its reduction is
  `sum_k (sum_i y_ik q_i)^2`. The knot search then runs the column’s
  order from `n - 1` down to 1, carrying with each unit that enters
  above the knot the running sums and the centred covariances of
  `h(t) = b max(0, x - t)` with each basis column and the candidate
  (`c_k`), with itself (`c`), and with the centred response (`g`), as
  [`mars()`](https://gillescolling.com/timesift/reference/mars.md)’s
  search carries them. At position `k` the hinge adds `r^2 / q` to the
  new form’s reduction, `r = g - sum(g_k c_k)` and `q = c - sum(c_k^2)`,
  where `c` is positive and `q / c` exceeds `0.01f`; a sum above
  `1.01f R`, `R` the residual sum of squares, or above twice the last
  step’s reduction is taken as zero. The knot is a candidate where `k`
  is a multiple of the min span, within `[end span, n - end span]`, the
  parent is positive at the unit above, and the unit at the knot does
  not share its value with the one below it. The step takes the greatest
  reduction, the first in parent, column and knot order where two tie.
- **A step’s terms.** The pair `b max(0, x - t)` and `b max(0, t - x)`
  at the knot’s value, the second held only where the step was of a new
  form at a knot; the column itself is the first at the column’s least
  value, with no second.
- **Stopping.** With `D` the terms’ count less one and
  `C = 1 + D + penalty D / 2`, the generalised cross-validation is
  `(R / n) / (1 - C / n)^2`. The pass takes the step where its reduction
  is above `thresh` of `R` and the cross-validation after it below ten
  times the null model’s, and stops without it otherwise, when `R` falls
  to `thresh` of the null’s, or at `nk` terms.
- **After the pass** the terms held are decomposed by `dqrdc2` at
  `0.01`, and the positions past the rank are dropped. The position is
  the one among the terms held, which `marss` reads as a term index: it
  is kept as `marss` keeps it.
- **The pruning pass.** Over the terms of the last forward step, the
  least squares at `0.01` and the diagonal of `(R'R)^-1` from R’s
  inverse; then, while terms remain beside the intercept, the term whose
  `beta_j^2 / v_j` is least (the first where two tie) is dropped, `R`
  raised by that, and the subset kept where its cross-validation is
  below the best so far, the least squares redone on what remains. The
  coefficients and `v_j` are read by the terms’ positions in the pivoted
  decomposition, as `marss` reads them.
- **The fit.** The kept terms’ least squares at `0.01`: coefficients in
  the decomposition’s order, and the fitted scores as the scored
  response less the residuals.
- **The variate.** `m = sum(t_i w_i f_i) / n` over the scored response
  `t`, the rescaled weights and the fitted scores `f`. `lambda = |m|`,
  held at `1 - eps`, and the variate’s sign that of `m`. At
  `lambda <= eps` nothing discriminates and every unit is predicted the
  second class’s share. Otherwise `alpha = sqrt(lambda)`, and the class
  centroids are `theta_j sign(m) / (sqrt(1 - lambda) / alpha)`.
- **The posterior.** A unit’s basis is each kept term as the product of
  its factors over the columns in order, `max(0, dir (x - cut))`, and
  its fitted score `f` the sum of the terms times the coefficients in
  order. Its variate is `z = f sign(m) / (sqrt(1 - lambda) alpha)`, and
  the second class’s posterior `pi_2 e_2 / (pi_1 e_1 + pi_2 e_2)`,
  `e_j = exp(-(d_j - min(d)) / 2)` and `d_j = (z - centroid_j)^2`; mda
  exponentiates without the shift, which is the same number where either
  class’s term is representable.
- **The recalibration.** biomod2’s `FDA` always predicts through a
  probit [`glm()`](https://rdrr.io/r/stats/glm.html) of the response on
  the in-sample posterior, under the case weights. That is `glm.fit`
  over the intercept and the posterior, the start
  `mu = (w y + 0.5) / (w + 1)`, the linear predictor held at
  `-qnorm(eps)` either side, the derivative of the mean held at `eps`
  from below, and the deviance’s relative change below `1e-8` in at most
  25 steps. biomod2 rounds the posterior to three decimals first, which
  is not reproduced. `calibrate = FALSE` predicts the posterior.
- **What a fit keeps**: the kept terms as their factors, their
  coefficients, the count of forward terms, the kept subset’s
  generalised cross-validation, the variate’s sign and scale, the
  centroids and priors, and the recalibration’s two coefficients and
  whether it settled.
- The columns of one parent are independent searches, and `threads` runs
  them at once; their results are taken in column order afterwards, so
  what comes back does not depend on it.

### The fixtures

`fda_cases.csv` names five cases on the weekly columns maxnet’s fixtures
read: mda’s own fits under the fractional weights and without them, at
degree two, unpruned, and on the design with the square of every column
beside it, whose near-collinear hinges are what the decomposition at
`0.01` drops. Each case carries the count of forward terms, the kept
terms as `column:direction:cut` factors and the kept subset’s
generalised cross-validation. `fda_coef.csv` holds the coefficients, and
`fda_predict.csv`, on every reading scaled by `1.01`, the posterior and
biomod2’s probit recalibration of it.

### How exactly

- **The count of forward terms and the kept terms are asserted
  exactly**, the cuts to `1e-12` relative: a cut is a reading.
- **The coefficients and the generalised cross-validation are asserted
  to `1e-9` relative and the posteriors and recalibrated probabilities
  to `1e-10`.** Across the five cases the core and mda agree to
  `1.4e-14` or better in the coefficients and `4.1e-15` in the
  predictions.

## The additive model

[`additive()`](https://gillescolling.com/timesift/reference/additive.md)
is one generalised additive model per response, over
`src/ts_additive.cpp`, which both languages compile. A biomod2 user’s
`GAM` is
`mgcv::gam(y ~ 1 + s(x1) + s(x2) + ..., family, weights = w, method = "GCV.Cp")`
under mgcv’s (1.9-4) defaults: a thin plate regression spline of basis
dimension 10 per column (Wood 2003), the smoothing parameters chosen by
the unbiased risk estimator under the binomial and the Poisson family,
whose scale is known, and by generalised cross-validation under the
Gaussian one (Wood 2008). The least squares are the Householder QR of
`src/ts_glm.cpp`.

- **A column’s term.** Its distinct values, sorted, are `u`. A column of
  one value has no term. A column of two enters linearly: the one column
  `(x - mean(x)) / rms(x - mean(x))`, unpenalised. Otherwise the basis
  dimension is `k' = min(k, length(u))`; mgcv refuses `k` above the
  distinct values, and taking the dimension down to them is the only
  place the two differ.
- **The knots** are `u`, or, where `u` holds more than `max_knots`
  values, the `max_knots` of them R’s
  [`sample()`](https://rdrr.io/r/base/sample.html) draws after
  `set.seed(1)`, sorted. That generator is the Mersenne Twister seeded
  as R seeds it: the seed is taken fifty steps along `s <- 69069 s + 1`
  (modulo 2^32), the next step is a position word that is discarded, and
  the 624 after it are the state. A uniform is the tempered output times
  `2^-32`, moved to `2^-33` or `1 - 2^-33` where it would be zero or
  one. An index below `m` is drawn by rejection: with
  `b = ceiling(log2(m))`, `floor(65536 u)` of successive uniforms are
  packed into an integer sixteen bits at a time, as many times as
  `b + 1` bits need, masked to its low `b` bits, and drawn again until
  it is below `m`. The draws without replacement start from the indices
  `0..n-1`; each takes an index `j` below the count left, keeps the
  index at `j`, and moves the last of those left into its place.
- **The radial basis.** `eta(r) = |r|^3 / 12` among the knots makes the
  symmetric matrix `E`. Its `k'` eigenvectors of greatest eigenvalue in
  magnitude are `U` and their eigenvalues `D`, each eigenvector turned
  so that its entry of greatest magnitude, the first where two tie, is
  positive. With `T` the knots’ rows `(1, t - shift)`, `shift = mean(x)`
  over the units, `Z` is the last `k' - 2` columns of the Q of a
  Householder QR of `U'T`, so that `T'UZ = 0`. The raw basis at a value
  `x` is `eta(|x - t_i|)` over the knots times `UZ`, then `1` and
  `x - shift`: `k'` functions, and the penalty on them `Z'DZ` on the
  first `k' - 2` and nothing on the last two.
- **Scaling.** Every raw column is divided by its root mean square over
  the units, and the penalty by the matching products. The penalty is
  then multiplied by the square of the design’s largest absolute row sum
  over its largest absolute column sum.
- **The constraint.** With `c` the scaled columns’ sums over the units,
  `Z_c` is the last `k' - 1` columns of the Q of a Householder QR of
  `c`, and the term’s columns are the scaled basis times `Z_c`, its
  penalty `S_c = Z_c' S Z_c`.
- **Turning.** `S_c`’s eigenvectors, its eigenvalues descending and each
  eigenvector turned as the radial ones are, make the term’s columns the
  constrained ones times them: `k' - 2` penalised columns, the penalty
  on each its eigenvalue, and last the one column the penalty leaves
  free, whose eigenvalue is zero.
- **The model** is the intercept and every term’s columns, in column
  order. It is refused where it holds more coefficients than units, as
  mgcv refuses it. A free column spanned by the free columns before it,
  the intercept first, by the limited pivoting of `src/ts_glm.cpp` at
  `1e-7`, is dropped and its coefficient held at zero.
- **The fit at smoothing parameters `lambda`.** Penalised iteratively
  reweighted least squares: the working weights `W = w mu (1 - mu)` and
  response `z = eta + (y - mu) / (mu (1 - mu))` under the binomial
  family, `W = w mu` and `z = eta + (y - mu) / mu` under the Poisson
  one, `W = w` and `z = y` under the Gaussian one, and each step the
  least squares of `sqrt(W) z` on `sqrt(W) X` stacked on the diagonal
  `sqrt(lambda_j d_c)` of the penalty. The first step starts from the
  means `(w y + 0.5) / (w + 1)` under the binomial family and `y + 0.1`
  under the Poisson one, later fits from the coefficients the search
  last reached, and a step that raises the penalised deviance is halved
  back towards the coefficients it left, at most thirty times. The fit
  stops once the penalised deviance moves by at most
  `1e-13 (|pdev| + 0.1)`; the Gaussian one is one step. The mean is the
  inverse logit with the linear predictor held at 30 either side, or the
  exponential held off zero by the machine epsilon.
- **The criterion.** With `D` the deviance, `n` the units and
  `tau = tr((X'WX + S)^-1 X'WX)`, the unbiased risk estimator
  `D / n + 2 gamma tau / n - 1` under the binomial and the Poisson
  family and the generalised cross-validation `n D / (n - gamma tau)^2`
  under the Gaussian one.
- **The search.** Newton’s method on `rho = log(lambda)`, each held to
  `[-25, 25]`, with the exact gradient and Hessian, from the start that
  makes each penalty’s mean diagonal that of `X'W_0X` over its columns,
  `W_0` the working weights at the starting means. The derivatives come
  from differentiating the score equations `X'(w (y - mu)) = S beta`
  implicitly: `d beta / d rho_j = -lambda_j H^-1 S_j beta` with
  `H = X'WX + S`, the second derivatives through the derivative of `W`
  in the linear predictor, and the trace’s through `H^-1`. At each step
  a parameter whose gradient is below `tol (|V| + D / n)` stays where it
  is, and so does one at a bound with its gradient pointing out; the
  search has settled when no other is left. The Hessian over the others
  is scaled by the root of its diagonal, its eigenvalues replaced by
  their magnitudes held above `1e-7` of the largest, and the step is at
  most 5 in any coordinate and halved until the criterion falls, at most
  thirty times; where no halving of it does, the steepest descent step
  of at most 2 is tried the same way, and where neither falls the search
  stops unsettled. `tol` is `1e-10` and the steps at most 200.
- **A criterion of several local minima.** Which one a search settles in
  depends on where it starts and how it steps. mgcv starts from a rule
  that reads its own parametrisation, which differs from this one in the
  eigenvectors’ signs and the columns’ rotation, so where the criterion
  has more than one minimum the two searches can settle in different
  ones. Each is a minimum of the same criterion over the same model.
- **What a fit keeps**: every term’s column, basis dimension, columns,
  penalised columns and shift, its knots, `UZ`, the map from its raw
  basis to its columns and the diagonal of its penalty; the coefficients
  held at zero; and per response the coefficients, the smoothing
  parameters, each term’s effective degrees of freedom, the criterion,
  the Newton steps taken and whether the search settled. The prediction
  is the terms’ columns at each value times the coefficients, through
  the inverse logit under the binomial family and the exponential under
  the Poisson one.
- **A fit at given smoothing parameters.** `sp`, one value per smooth in
  column order and shared by the responses, replaces the search: the fit
  is the penalised likelihood’s maximum at those parameters, held as
  given with no bound, and the criterion is its value there. It is
  positive and finite, and as many as the model has smooths. The inner
  fit is the one the search runs at each of its steps, so what a search
  settles on is a fit at its `sp`.
- The terms are built one column at a time and the responses fitted one
  at a time, and `threads` runs either at once; each writes its own
  slot, so what comes back does not depend on it. The responses share
  the terms, so fitting them together is fitting each alone.

### The fixtures

`additive_cases.csv` names ten cases on the weekly columns maxnet’s
fixtures read and on inputs of their own: mgcv’s own fits under the
fractional weights and without them, under the Gaussian response, under
the Poisson response of `count_response.csv` on the late columns at
`k = 5` and unweighted at `k = 10` and on the first three at `k = 5`,
with every one of the eleven columns at `k = 5` under the Gaussian
response, at `gamma = 1.4`, on `additive_input.csv`, whose second column
holds five distinct values and whose third holds two, against mgcv’s
`s(v02, k = 5)` and the linear `v03`, and on `additive_knots_input.csv`,
2100 units of two columns, past the subsample. Each is run to a tight
tolerance (`epsilon = 1e-13`, `mgcv.tol = 1e-15`, Newton’s
`conv.tol = 1e-13`), and refused unless mgcv, restarted under its
default tolerances from smoothing parameters of `1e-2`, `1`, `1e2` and
`1e4`, settles at the same criterion to `1e-6` from each, and, under the
Gaussian response, unless a direct penalised least-squares solve at the
smoothing parameters mgcv reports gives its coefficients: `magic` takes
the square root of the summed penalty by a pivoted Cholesky, which drops
the penalty directions of the moderately smoothed columns as numerically
null where another column’s smoothing parameter is many orders of
magnitude larger. Each case carries the criterion and every column’s
effective degrees of freedom, and `additive_predict.csv` the fitted mean
on the design scaled by `1.01`, at every tenth unit of the subsample’s
input.

A fit at given smoothing parameters is asserted against the penalised
problem solved outright. Both suites read the design and the penalty off
the fit, solve the weighted design stacked on the root of the penalty by
a singular value decomposition, which takes no rank decision, and under
the binomial and the Poisson family repeat that at the working weights
with step halving until the coefficients stop moving. The fitted means,
each term’s effective degrees of freedom and the criterion are asserted
to `1e-7`, `1e-6` and `1e-7` at seven patterns of smoothing parameters
spanning `1e-4` to `1e12`, among them one column at `1e12` beside the
rest at `1e-4`, on the binomial weekly columns, the Gaussian eleven
columns at `k = 5` and `gamma = 1.4`, the binomial `few` input and the
Poisson late columns at `k = 5`. The same reference at smoothing
parameters off by a relative `1e-6` differs by `4e-8` in the fitted
means, so the tolerance holds a parameter to about `1e-5`.

### How exactly

- **The criterion is asserted to `1e-9` relative, the effective degrees
  of freedom to `1e-6` and the fitted means to `1e-7`.** Two searches of
  the same criterion settle at the same point to the tolerance they are
  run at and no closer, and the basis is an eigenproblem each solves to
  its own tolerance. Across the seven cases the core and mgcv agree to
  `5e-10` in the criterion, `3.5e-7` in the degrees of freedom and
  `1e-8` in the fitted means, and across the three Poisson ones to
  `6.2e-10`, `4.8e-7` and `1.2e-7`.

## The network

[`perceptron()`](https://gillescolling.com/timesift/reference/perceptron.md)
is one feed-forward network of one hidden layer per response, nnet’s
model and biomod2’s `ANN`, over `src/ts_perceptron.cpp`, which both
languages compile, fitted by the minimiser of `src/ts_quasi_newton.cpp`.

- `hidden` logistic units each take a bias and every column; the output
  takes a bias, every hidden unit and, with `skip`, every column. The
  logistic function is exactly 0 below `-15` and exactly 1 above `15`,
  for the hidden units and for a logistic output alike.
- The weights are a vector ordered unit by unit: each hidden unit’s bias
  and then its columns, then the output’s bias, its hidden units and its
  skipped columns. Drawn, they are `(2u - 1) * range` for `u` the
  uniforms of stream 0 of the seed, the generator **The forest**
  defines, in that order.
- The objective is the sum over rows of the case weight times the row’s
  error, plus `decay` times the sum of the squared weights, biases
  included. Under the binomial family the output is logistic and the
  error `t log(t / y) + (1 - t) log((1 - t) / (1 - y))`, zero terms
  dropped and a probability below `1e-80` read as `1e-80`; under the
  Gaussian family the output is its sum and the error `(y - t)^2`; under
  the Poisson family the output is the exponential of its sum and the
  error `t log(t / y) - (t - y)`.
- The gradient starts every slope at `2 decay w` and adds the rows in
  order. A row’s error is carried back unweighted, `y - t` under the
  binomial and Poisson families and `2 (y - t)` under the Gaussian,
  through each hidden unit as `e v h (1 - h)`, and the row’s case weight
  multiplies the error where it enters a slope.

The minimiser is Nash’s variable metric method (Compact Numerical
Methods for Computers, 1990, Algorithm 21). From the identity, each
iteration takes the direction `t = -B g` and a step of one, cut by `0.2`
until the objective falls by at least `1e-4` times the step times `g't`,
or until no coordinate of the trial point differs from the current one
when both are added to `10`. An accepted step `s`, with `c` the change
in gradient, updates `B` by
`B += ((1 + c'Bc / s'c) s s' - (Bc) s' - s (Bc)') / s'c` where
`s'c > 0`, and resets it to the identity where not. A direction that is
not downhill, or a search that cannot move, resets `B`; two in a row
from the identity end the search. It also ends at an objective below
`abs_tol`, at a fall of at most `rel_tol` times the objective plus
`rel_tol`, or after `max_iter` gradients counting the first, and `B` is
reset after `2 n` gradients without a reset. `B` is held as its lower
triangle.

### The fixtures

`perceptron_cases.csv` names seven nnet fits on the weekly columns
maxnet’s fixtures read, under the binomial and Gaussian responses, with
and without the fractional weights, at two, three and five hidden units,
with and without skip-layer connections, and one stopped at five
iterations. `perceptron_weights.csv` holds each case’s starting weights,
drawn in the generator and rounded to twelve digits, and the weights
nnet ends at; `perceptron_predict.csv` nnet’s predictions on the design
scaled by `1.01`. Both suites start the core from the written weights.

### How exactly

- **The core ends at nnet’s weights to the bit** on the fixture design
  and on drawn ones, at the ends of paths of up to 200 iterations.
- **The tolerances are the paths’, not the core’s.** A last-bit
  difference in a library’s [`exp()`](https://rdrr.io/r/base/Log.html)
  is carried forward by the minimiser, so each case is asserted at a
  hundred times what its own end point moved when its starting weights
  were jittered in their last bit, from `1e-10` to `1e-4`. A case is
  chosen where that is small: on this design a network with little
  penalty on a path of a hundred iterations moved by up to percent.

## The hierarchical model

[`hierarchical()`](https://gillescolling.com/timesift/reference/hierarchical.md)
is one Bayesian logistic model per response, over
`src/ts_hierarchical.cpp`, which both languages compile, with
`src/ts_sparse.cpp` under its nearest-neighbour field. The linear
predictor of target `i` is `x_i' beta + u_g(i) + f(s_i)`: the fixed
effects, an intercept for the unit the target belongs to and a
Gaussian-process field at the target’s place, each target’s log
likelihood scaled by its case weight.

- **The design.** The intercept’s column, then every flattened column
  that is not constant over the fitting targets, centred on its mean and
  divided by its sample standard deviation. A prediction centres and
  scales by the fit’s own.
- **The priors.** `beta_j ~ N(0, 2.5^2)` for every fixed effect, the
  intercept included. A unit’s intercept is `N(0, sd_u^2)`, and the
  field has a marginal standard deviation `sd_f` and a range `r`. Each
  standard deviation has the penalised-complexity prior with
  `P(sd > 3) = 0.01`, density `lambda exp(-lambda sd)` with
  `lambda = -log(0.01) / 3`, and the range the penalised-complexity
  prior of a two-dimensional field, `P(r < r0) = 0.5`, an exponential
  prior on `1 / r` with rate `-log(0.5) r0`, where `r0` is a fifth of
  the extent. Both are densities in the logarithm of the parameter, so
  each carries its Jacobian.
- **The coordinates** are centred on the column means and divided by one
  factor, the root of the mean of the two columns’ sample variances, so
  that distances keep their proportions. The extent is the diagonal of
  their bounding box.
- **The Hilbert-space field** (`hsgp`) has `m` Laplacian eigenfunctions
  per axis, `sin(pi j (x + L) / (2 L)) / sqrt(L)` for `j = 1..m` on a
  box whose half-width `L` is `boundary` times half the coordinates’
  range, no less than 0.1, about the middle of their range, with the
  eigenvalue `(pi j1 / 2 L1)^2 + (pi j2 / 2 L2)^2`, `j2` fastest. A
  coefficient has the prior `N(0, 1)` and multiplies the square root of
  the squared-exponential spectral density
  `sd_f^2 2 pi r^2 exp(-r^2 w^2 / 2)` at its eigenvalue `w^2`.
- **The nearest-neighbour field** (`nngp`) lives on the distinct
  locations, sorted by the first coordinate then the second. Location
  `i` is conditioned on its `neighbours` nearest among the locations
  before it, nearest first. With `C` their covariance, `c` its
  covariance with `i`, and a nugget of `1e-8` on `C`’s diagonal, the
  regression is `a = C^-1 c` and the conditional variance
  `max(sd_f^2 - c'a, 1e-10)`; the first location has the marginal
  variance. The precision is `(I - A)' D^-1 (I - A)` and its log
  determinant `-sum log D`. The covariance is `sd_f^2` times
  `exp(-d / r)`, `(1 + x) exp(-x)` with `x = sqrt(3) d / r`,
  `(1 + x + x^2 / 3) exp(-x)` with `x = sqrt(5) d / r`, or
  `exp(-(d / r)^2)`. The precision is sparse, its pattern the cliques of
  a location and its neighbours and the pairs of a target’s location and
  unit, and is factored by a Cholesky decomposition under a
  minimum-degree ordering analysed once.
- **The conditional fit** at fixed hyperparameters is Laplace’s method
  over the coefficients, the field and the unit intercepts together:
  Newton’s method with step halving to the mode of the penalised log
  likelihood, and the log marginal likelihood
  `J + (1/2) log |Q| - (1/2) log |H|`, `J` the penalised log likelihood
  at the mode, `Q` the prior’s precision and `H` the information there.
  The unit intercepts are eliminated by their diagonal information under
  a dense field block, and the coefficients by a Schur complement under
  a sparse one.
- **The hyperparameters** are the logarithms of `sd_u` where there are
  unit intercepts, then of `sd_f` and `r` where there is a field, and
  the log posterior is the conditional log marginal plus the log
  hyperprior. Without a field the mode of `log sd_u` is the fit, the
  empirical-Bayes estimate, and without either the fit is the posterior
  mode of the coefficients. With a field the mode is found by BFGS on
  central differences, the curvature there by finite differences, and
  each axis of its eigendecomposition is spread `nodes` points `1.25`
  standard deviations apart, the deviation held within `[0.05, 2]`. Each
  node’s conditional fit is weighted by `exp` of its log posterior, and
  the fit is their weighted mean.
- **A prediction** is the weighted sum over the nodes of `x' beta`, the
  unit’s intercept where the unit was fitted and zero where it was not,
  and the field at the new place: the eigenfunctions times the
  coefficients under `hsgp`, and under `nngp` the conditional mean
  `c'C^-1 f` over the `neighbours` nearest fitted locations of any
  order, a nugget of `1e-6` on `C`’s diagonal.

The coordinates and the unit of each target ride on the array as a
placement, not a channel: no learner reads them as a predictor, and a
split of the targets splits them with it. `timesift(coords = )` names
the two columns, and a learner with a field refuses an array that
carries none.

### The fixtures

`hierarchical_input.csv` holds 60 targets in 20 units of three, and
tulpa’s own fits of `y ~ x1 + x2` to it, at the levels where its answer
is deterministic. `hierarchical_cases.csv` carries the posterior mode of
the coefficients, with the case weights and without, and the
empirical-Bayes fit with an intercept for each unit, each with its
coefficients and log marginal likelihood, and the last with the standard
deviation of the intercepts, whose estimates are in
`hierarchical_ranef.csv`. `hierarchical_hsgp_nodes.csv` and
`hierarchical_nngp_nodes.csv` hold the nodes of tulpa’s nested Laplace
grid that carry weight: the hyperparameters, the log weight and the
latent mode, the coefficients then the field’s values, in target order
for the nearest-neighbour field. The coefficients there have a prior
standard deviation of 100, which is what the nested route fits under,
and the core is asked for it. A node is the latent mode at fixed
hyperparameters, so what is pinned is that mode and the constancy across
the nodes of the difference between the core’s log posterior and tulpa’s
log weight.

A model with unit intercepts and a field is asserted against the same
Laplace marginal written out densely, the Vecchia precision built
location by location, the basis from its formula and Newton’s method on
full matrices, at fixed hyperparameters, under every covariance, in both
suites.

### How exactly

- **The mode and the log marginal likelihood without a field** are
  tulpa’s to `1e-11` and `1e-8` relative to the tolerance the case
  carries, `1e-7`; the empirical-Bayes standard deviation, the
  coefficients, the log marginal likelihood and the intercepts to
  `2.3e-8`, `4.5e-9`, `1.7e-8` and `3.3e-8`.
- **A node’s mode** is tulpa’s to `1e-13` under `hsgp` and `2e-14` under
  `nngp`, and the difference between the core’s log posterior and
  tulpa’s log weight is constant across the nodes to `3e-10` and `4e-9`.
  The grid itself, the mode of the hyperparameters and so the nodes a
  fit places, is not pinned: two searches settle at the same point to
  the tolerance they are run at and no closer, and tulpa’s own search
  differs from this one.
- **The dense marginal** is the core’s to `1e-8` relative and the latent
  mode to `1e-7`.

## The combiner

[`ensemble_fit()`](https://gillescolling.com/timesift/reference/ensemble_fit.md)
is handed each candidate’s out-of-fold predictions, the response, the
mask and the fold map, and the candidates’ scores where the
specification reads them; it never sees a model. Its methods are the
ones biomod2’s `BIOMOD_EnsembleModeling()` offers on the same
predictions, under the names this contract gives them.

### Which candidates are members

A candidate’s score is its mean over the variables of its mean over the
scorable folds of each variable. It is read off the run’s scores, or
recomputed from the out-of-fold predictions on the mask where the
specification names a metric of its own. `min_score`, biomod2’s
`metric.select.thresh`, then keeps the candidates scoring at least it;
biomod2 keeps those scoring above it, which differs only on a score
equal to the threshold. `scope` picks among what is left: the
best-scoring candidate, the first offered among any on its score, fixes
the representation (`"learners"`) or the learner (`"representations"`).
An ensemble of fewer than two members is refused, naming the filter or
the scope that left it.

biomod2’s `em.by` needs no argument of its own here. It groups the
models of its pseudo-absence sets and runs before combining them; a
candidate’s out-of-fold prediction already covers every fold of the one
map, so `em.by = "algo"` is each candidate itself, and `"all"` is
`scope = "all"`.

### The weights

`"stack"` minimises the head’s loss over the scorable cells on the
simplex. The loss is the cross entropy under `binary_cross_entropy`, the
mean squared error under `squared_error`, and under `poisson_deviance`
the mean of `2 (y log(y / p) - (y - p))`, the logarithm taken as zero at
`y = 0`, with a combined mean read no closer to zero than the machine
epsilon. `"mean"`, `"median"` and `"committee"` weigh every member the
same. `"weighted"` with `decay = "proportional"` takes each member’s
score, zero where it is at or below zero, over their sum. With a number
`d` the `K` members scoring above zero are ordered from the highest, the
one in place `r` takes `d^(K - r + 1)`, members on exactly the same
score take the mean of their places’ weights, a member at or below zero
takes none, and the weights are divided by their sum. This is biomod2’s
`EMwmean.decay`, less its rounding of the scores and the weights to
three decimals. Where no member scores above zero, every member weighs
the same under either rule.

### The committee

biomod2’s `EMca`. Each member’s cut on each variable is
[`decision_threshold()`](https://gillescolling.com/timesift/reference/kappa_score.md)
under `rule` (`"youden"` unless named) on that member’s out-of-fold
predictions of every target, the cut a fit’s
[`decision_threshold()`](https://gillescolling.com/timesift/reference/kappa_score.md)
learns for that candidate. The combination on a variable is the weighted
share of the members holding a finite cut on it whose prediction is at
least that cut; a variable on which no member holds a cut is `NA` in R
and NaN in Python. Inside
[`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
the cuts of each outer fold are learned on the inner out-of-fold
predictions of its training targets, as a stack’s weights are.

### The spread

[`ensemble_spread()`](https://gillescolling.com/timesift/reference/ensemble_spread.md),
and `predict(type = "spread")` on a run’s ensemble, read biomod2’s
`EMcv` and `EMci` under the stack’s weights `w`, taken as equal for a
median and a committee. For each unit and variable, with `p` the
members’ predictions:

- `mean` is `m = sum(w p)`;
- `sd` is `s = sqrt(sum(w (p - m)^2) / (1 - sum(w^2)))`, the sample
  standard deviation under equal weights;
- `cv` is `s / m`, a ratio where biomod2 reports a percentage;
- `lower` and `upper` are
  `m -+ t(1 - alpha / 2, n - 1) s sqrt(sum(w^2))`, `n` the members
  carrying a weight above zero, held inside the range of the head’s
  predictions, `[0, 1]` under `binary_cross_entropy` and `[0, Inf)`
  under `poisson_deviance`. Under equal weights this is the t interval
  of a mean of `n` members. biomod2 reads its quantile on `n + 1`
  degrees of freedom; `n - 1` is the interval’s own.

`sd`, `cv` and the interval are missing where fewer than two members
carry weight. The array is `[unit, variable, statistic]` in that order
of statistics on both sides.

### The fixtures

`ensemble_oof.csv` holds five candidates’ out-of-fold predictions of the
response fixture, rounded to six decimals, two of them identical so
their scores tie exactly. `ensemble_cases.csv` names twelve
specifications, one per method, the three committee rules, a decay
alone, a decay after a `min_score`, a `min_score` before a scope, a
stack under a scope and weights under `roc_auc`; each suite scores the
candidates under its own `tss` and hands those scores over as a run
does. `ensemble_weights.csv` holds each case’s members and weights,
`ensemble_thresholds.csv` each committee member’s cut on each variable,
and `ensemble_predict.csv` each case’s combination and spread at
`alpha = 0.1` for every unit and variable. `ensemble_count_response.csv`
holds sixty counts with a five-fold map, `ensemble_count_oof.csv` four
candidates’ out-of-fold means of them rounded to six decimals, and
`ensemble_count_cases.csv` the stack, the mean and the median under the
Poisson deviance, with `ensemble_count_weights.csv` and
`ensemble_count_predict.csv` holding their weights and their combination
and spread. The counts and the means are deterministic. The generator
refuses the stack’s weights unless their loss is within `1e-7` of the
least a derivative-free search over the simplex reaches from five
starts.

### How exactly

Everything is asserted to `1e-10` relative, the count cases to `1e-9`.
The file holds twelve significant digits, and the largest difference
either side reads is `4.8e-12`, the stack’s weights included: the two
solvers take the same steps.

## What each language carries

The representation and the three artifacts are the contract. Everything
built over them is meant to match too, and where the two sides differ
the difference is recorded here rather than found at a call site.

### One name per concept

| concept | the name, on both sides |
|----|----|
| the whole run | [`timesift()`](https://gillescolling.com/timesift/reference/timesift.md), from a table of targets and a table of series to a scored comparison and a nested estimate of choosing among it |
| what a representation is | [`native()`](https://gillescolling.com/timesift/reference/native.md), [`grain()`](https://gillescolling.com/timesift/reference/native.md), [`multigrain()`](https://gillescolling.com/timesift/reference/native.md), [`lookback()`](https://gillescolling.com/timesift/reference/native.md), and the sets [`grains()`](https://gillescolling.com/timesift/reference/grains.md) and [`lookbacks()`](https://gillescolling.com/timesift/reference/grains.md) |
| coercing to a set of representations | [`as_sift()`](https://gillescolling.com/timesift/reference/grains.md), from a representation, a list of them or a vector of grain names |
| the calendar-binned array | [`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md) |
| the target-anchored array | [`lookback_matrix()`](https://gillescolling.com/timesift/reference/lookback_matrix.md) |
| an already-reduced feature table | [`feature_matrix()`](https://gillescolling.com/timesift/reference/feature_matrix.md), a one-channel array with no time axis, so a published set of aggregates can be an arm beside a grain |
| which units reach which bins | [`coverage()`](https://gillescolling.com/timesift/reference/coverage.md), the count of readings per unit and bin over every bin the calendar tiles the record with, which is where a refused record’s gaps are read off |
| building one representation | [`build_representation()`](https://gillescolling.com/timesift/reference/build_representation.md) |
| a channel added to an array | [`bind_channels()`](https://gillescolling.com/timesift/reference/bind_channels.md), and [`calendar_channels()`](https://gillescolling.com/timesift/reference/calendar_channels.md) for the sine and cosine of each bin’s position in the year and, finer than a day, in the day, both as **The channels** defines them |
| the penalised learner | [`elasticnet()`](https://gillescolling.com/timesift/reference/elasticnet.md) |
| the generalised linear model | [`linear()`](https://gillescolling.com/timesift/reference/linear.md), its terms searched two-way, forward, backward or not at all |
| the forest | [`forest()`](https://gillescolling.com/timesift/reference/forest.md) |
| the classification and regression tree | [`tree()`](https://gillescolling.com/timesift/reference/tree.md) |
| gradient boosted trees | [`boosting()`](https://gillescolling.com/timesift/reference/boosting.md) |
| maxnet’s MaxEnt | [`maxent()`](https://gillescolling.com/timesift/reference/maxent.md) |
| the surface range envelope | [`envelope()`](https://gillescolling.com/timesift/reference/envelope.md) |
| multivariate adaptive regression splines | [`mars()`](https://gillescolling.com/timesift/reference/mars.md) |
| flexible discriminant analysis | [`discriminant()`](https://gillescolling.com/timesift/reference/discriminant.md) |
| generalised additive models | [`additive()`](https://gillescolling.com/timesift/reference/additive.md) |
| the network of one hidden layer | [`perceptron()`](https://gillescolling.com/timesift/reference/perceptron.md) |
| the Bayesian logistic model with unit intercepts and a spatial field | [`hierarchical()`](https://gillescolling.com/timesift/reference/hierarchical.md) |
| the encoders | [`mlp()`](https://gillescolling.com/timesift/reference/torch_learners.md), [`cnn()`](https://gillescolling.com/timesift/reference/torch_learners.md), [`rescnn()`](https://gillescolling.com/timesift/reference/torch_learners.md) |
| how an encoder is trained | [`train_control()`](https://gillescolling.com/timesift/reference/train_control.md) |
| fitting one learner on one representation | [`fit_learner()`](https://gillescolling.com/timesift/reference/fit_learner.md) |
| the resampling | [`cv()`](https://gillescolling.com/timesift/reference/cv.md) and [`grouped_cv()`](https://gillescolling.com/timesift/reference/cv.md), and [`block_cv()`](https://gillescolling.com/timesift/reference/cv.md) and [`env_cv()`](https://gillescolling.com/timesift/reference/cv.md) for the spatial and environmental splits |
| the fold map and the mask | [`fold_map()`](https://gillescolling.com/timesift/reference/fold_map.md) and [`scorable_cells()`](https://gillescolling.com/timesift/reference/scorable_cells.md) |
| fitting across a set of grains | [`grain_ladder()`](https://gillescolling.com/timesift/reference/grain_ladder.md), and [`select_grain()`](https://gillescolling.com/timesift/reference/select_grain.md) for the nested selection |
| the combiner | [`ensemble()`](https://gillescolling.com/timesift/reference/ensemble.md), [`ensemble_fit()`](https://gillescolling.com/timesift/reference/ensemble_fit.md), [`ensemble_combine()`](https://gillescolling.com/timesift/reference/ensemble_combine.md) and [`ensemble_weights()`](https://gillescolling.com/timesift/reference/ensemble_weights.md) |
| how far an ensemble’s members disagree | [`ensemble_spread()`](https://gillescolling.com/timesift/reference/ensemble_spread.md), and `predict(type = "spread")` on a run |
| scoring held-out predictions | [`score_predictions()`](https://gillescolling.com/timesift/reference/score_predictions.md), on the cells the mask allows |
| the metrics | [`tss()`](https://gillescolling.com/timesift/reference/tss.md), [`roc_auc()`](https://gillescolling.com/timesift/reference/roc_auc.md), [`average_precision()`](https://gillescolling.com/timesift/reference/average_precision.md), [`kappa_score()`](https://gillescolling.com/timesift/reference/kappa_score.md), [`table_metric()`](https://gillescolling.com/timesift/reference/table_metric.md), [`boyce_index()`](https://gillescolling.com/timesift/reference/boyce_index.md), [`regression_metric()`](https://gillescolling.com/timesift/reference/regression_metric.md) and [`ordinal_metric()`](https://gillescolling.com/timesift/reference/ordinal_metric.md), with [`decision_threshold()`](https://gillescolling.com/timesift/reference/kappa_score.md) and [`model_agreement()`](https://gillescolling.com/timesift/reference/kappa_score.md) beside them |
| the cut a fit applies | [`decision_threshold()`](https://gillescolling.com/timesift/reference/kappa_score.md) given a fit in place of the response, one cut per response learned from a candidate’s out-of-fold predictions |
| two arms on matched cells | [`paired_contrast()`](https://gillescolling.com/timesift/reference/paired_contrast.md) |
| every grain against a learner’s best | [`grain_contrasts()`](https://gillescolling.com/timesift/reference/grain_contrasts.md), the mixed model of the per-cell scores and Dunnett’s many-to-one comparisons off it |
| a record with a planted grain | [`simulate_records()`](https://gillescolling.com/timesift/reference/simulate_records.md), the vignette’s and the recovery tests’ generator |
| the inflation of a self-selected threshold | [`tss_inflation()`](https://gillescolling.com/timesift/reference/tss_inflation.md), and [`implied_skill()`](https://gillescolling.com/timesift/reference/implied_skill.md) for the level it implies |
| a set of representations | [`timesift_set()`](https://gillescolling.com/timesift/reference/timesift_set.md), which reads as a mapping of grain name to representation |
| folds of the inner cross-validation | `n_inner` |
| the three artifacts | [`write_folds()`](https://gillescolling.com/timesift/reference/artifacts.md) and [`read_folds()`](https://gillescolling.com/timesift/reference/artifacts.md), [`write_response()`](https://gillescolling.com/timesift/reference/artifacts.md) and [`read_response()`](https://gillescolling.com/timesift/reference/artifacts.md), [`write_cells()`](https://gillescolling.com/timesift/reference/artifacts.md) and [`read_cells()`](https://gillescolling.com/timesift/reference/artifacts.md) |
| the digest | [`digest_array()`](https://gillescolling.com/timesift/reference/digest_array.md), exported |
| the registries | [`register_learner()`](https://gillescolling.com/timesift/reference/register_learner.md) and [`learners()`](https://gillescolling.com/timesift/reference/register_learner.md), [`register_metric()`](https://gillescolling.com/timesift/reference/register_metric.md) and [`metrics()`](https://gillescolling.com/timesift/reference/register_metric.md), [`register_response()`](https://gillescolling.com/timesift/reference/register_response.md) and [`responses()`](https://gillescolling.com/timesift/reference/register_response.md), [`register_tuning()`](https://gillescolling.com/timesift/reference/register_tuning.md) and [`tunings()`](https://gillescolling.com/timesift/reference/register_tuning.md) |
| pseudo-absences | [`pseudo_absences()`](https://gillescolling.com/timesift/reference/pseudo_absences.md), the random, sre and disk strategies |
| what a rare response weighs | [`positive_weights()`](https://gillescolling.com/timesift/reference/positive_weights.md), the case weights the shipped presence-absence head carries as its weights function and every learner that ships reads through the head |

### The same call does the same thing

- Naming one grain returns the representation and naming two or more
  returns a set, whether the one is named as a string or as a sequence
  of one.
- A guard above the core names what it refuses the same way: a missing
  identifier or instant names its column, a duplicated reading and a
  supplied calendar returning no bin start name the unit and the instant
  in UTC to the second, and the noun agrees with the count. `bins` given
  as a whole number in floating point is that whole number, as R reads
  `3` and Python reads `3.0`.
- A representation refuses a statistic its grain has no definition for,
  and a `year_start` that is not a month and a day, when it is
  constructed rather than when it is built. `"auto"` is the whole set
  the record supports and is refused beside a named grain, and the
  `stats` and `year_start` given to
  [`grains()`](https://gillescolling.com/timesift/reference/grains.md)
  reach every member of the set it returns.
- A representation anchored on the target and a run without
  `target_time` are refused against each other in both directions, over
  the members of the sift and the representations learners pinned
  themselves to alike. A learner’s `data` is a representation or
  nothing; the name of a grain is not one.
- [`grain_ladder()`](https://gillescolling.com/timesift/reference/grain_ladder.md)
  and
  [`select_grain()`](https://gillescolling.com/timesift/reference/select_grain.md)
  left without a fold map build one with the defaults of
  [`fold_map()`](https://gillescolling.com/timesift/reference/fold_map.md).
  The two languages draw different maps from the same seed, so where
  both must see one split, write it and read it back as the section
  above describes.
- [`grain_ladder()`](https://gillescolling.com/timesift/reference/grain_ladder.md)
  and
  [`select_grain()`](https://gillescolling.com/timesift/reference/select_grain.md)
  take a `control` as a run does, and hand it to every learner that
  declares one. A selection hands the same one to the inner search and
  to the refit.
- Held-out predictions are placed by unit and by variable, never by
  position.
- A learner is handed the whole response matrix and returns one column
  per response, whether it declares `joint` or `separate`. `multi` is
  what the learner says it does with that matrix and what a report says
  of the candidate; a learner fitting one model per response does that
  inside its own fit, so the block of predictors is built once for the
  fit rather than once for every response of it.
- A prediction that is not a number on a scorable cell is refused where
  it is scored, naming the arm, the cell and how many there are, rather
  than scored as the `NA` a one-class cell gives. The combiner is fitted
  on every scorable cell and refuses to drop one for the same reason.
- A setting given at fit time overrides the one the learner carries, and
  a setting the learner does not have is refused rather than ignored.
- The response head and the metric are registry entries. `metric` takes
  a registered name or a function of `(y, p)`, and left unset it is the
  one the response head carries, which for the shipped presence-absence
  head is `roc_auc`. Both travel with the fit: the function is what
  scores, and the name is what the report prints. A function has no name
  to print and reads as `<function>` on both sides rather than as
  whatever each language calls an anonymous one.
  [`select_grain()`](https://gillescolling.com/timesift/reference/select_grain.md)
  is the one door that takes a name only, because it reports the
  estimate under every registered metric and the one it selects on has
  to be a row of that table.
- An occlusion profile left without a `metric` is read by the one the
  fit was scored under, so a weight is a fall in the number the summary
  reports rather than in a second one. It reaches the response through
  the head the fit was made under, as everything else does, so a head
  that is not presence-absence is occluded like any other.
- An occlusion profile is read on the scorable cells alone, the mask of
  the response and the fold map, as every score is. A bin is held back
  whole: one permutation of the held-out units per draw moves every
  channel of the bin together, so no unit is shown a coldest day from
  one unit beside a warmest day from another. A channel identical across
  units, as the calendar channels are, is left in place when a bin is
  held back, since it says where the bin sits rather than what a unit
  read there; holding such a channel back across the record with
  `over = "channel"` is still asked for by name. The `fold_mean` and
  `unit_mean` substitutes draw nothing and are defined the same way on
  both sides; the `permute` substitute draws on each language’s own
  random stream, so its weights are the same in distribution and not
  draw for draw.
- The encoders take `swa` and `swa_start`: the schedule anneals until
  the averaging begins and is then held flat, the averaged weights get
  their own pass to rebuild the batch-normalisation statistics from a
  reset, and the default is off, so a default recipe is the same recipe
  on both sides. `swa_start` is at least 0 and under 1 on both.
- What a rare response weighs is the response head’s and not a training
  setting. The head’s weights function returns one case weight per cell
  of the response, and the encoders, the penalised fit, the forest and
  the stepwise search all fit under it: the encoders as an elementwise
  weight on the loss, the penalised fit and the stepwise search as case
  weights, and the forest as the probability a unit is drawn into a
  tree’s bootstrap, since a tree grown to pure leaves is the same tree
  under any weight on its observations. The weights function takes the
  response and a mask of the rows the model is fitted on, reads whatever
  it reads off those rows alone, and weights every row. The shipped
  presence-absence head weights each presence by the ratio of absences
  to presences among the fitting units, capped at 50, and each absence
  by one; a head without a weights function fits unweighted. On both
  sides.
- The encoders standardise every channel by its own centre and sample
  standard deviation over every unit and bin of the fitting units,
  except a channel `position` names, which they read at its own
  amplitude (centre 0, scale 1); the inner validation set is a plain
  random draw of the fitting units, and the loss read on it, which the
  early stopping watches, is weighted by the head as the fitting loss
  is, with the fitting units alone in the count the weights are made
  from; the fitting units are cut into as few batches of at most
  `batch_size` rows as they divide into, of as equal a length as they
  can be; the snapshot early stopping restores, and the running average
  `swa` keeps, are copies of the weights and never the storage the
  optimiser updates.
- The default training control holds no inner validation set back, so an
  encoder trains every fitting unit for the whole budget and keeps the
  last epoch; the patience is read only where `val_frac` holds a set
  back, and a patience that never runs out still restores the epoch of
  lowest validation loss. On both sides.
- A `static` column enters the array as a channel holding the same
  number in every bin, which is the constant an encoder reads beside the
  readings. Flattening the bins into a block of features reads such a
  channel once, so a static predictor is one column of the design
  however many bins the grain has.
- A learner is fitted toward the registered response head and holds no
  response of its own: the encoders train under the head’s `loss` and
  predict through its `activation`, and the three learners fitting one
  model per response take the family the loss names, logistic under
  `binary_cross_entropy`, Gaussian under `squared_error` and Poisson
  under `poisson_deviance`. A fit that declares a `head` argument is
  handed the head, as one that declares `control` is handed the control,
  and one that declares `weights` is handed the head’s case weights, the
  matrix of the response’s shape the learners that ship fit under; a fit
  declaring none fits unweighted.
- A fitted encoder holds its weights as arrays and the device *setting*
  rather than the device it resolved to, and rebuilds the network when
  it predicts, so a fit written with
  [`saveRDS()`](https://rdrr.io/r/base/readRDS.html) or `pickle`
  predicts in a fresh session and on another machine.
- A fit refers to the code that made it rather than carrying a copy of
  it: an encoder stores the name of its module builder, and a fit stores
  the name and the settings of its learner wherever the registry can
  rebuild it. A fit read back therefore predicts through the code the
  package holds now, and one naming a builder or a learner the session
  does not carry says so by name. A learner defined outside any registry
  has no name to be rebuilt from and travels whole.
- [`select_grain()`](https://gillescolling.com/timesift/reference/select_grain.md)
  searches the candidates in the order the grains and the learners were
  declared in, so which candidate an exact tie on the inner score falls
  to does not depend on how the names sort. Its `rule` is `"argmax"` by
  default. `"coarsest_adequate"` takes, among the candidates whose inner
  score is at least the highest minus that candidate’s standard error
  (the standard deviation over the inner folds of each fold’s mean over
  its scored variables, over the square root of the fold count), the one
  with the fewest bins, then the fewest channels, then the higher score,
  then the one declared first; a standard error that cannot be computed
  is zero. Each outer fold reports the chosen score, the highest score
  and that standard error. With a `threshold` rule, each outer fold
  learns one cut per variable by
  [`decision_threshold()`](https://gillescolling.com/timesift/reference/kappa_score.md)
  on the selected candidate’s inner out-of-fold predictions of the outer
  training units, and the test fold is read at it by `tss(threshold =)`,
  presence at `p >= threshold`; the estimate row is `tss_inner_cut`.
- Every estimate and every contrast names the interval it carries.
  `"variables"` is the spread across the response variables of the
  dataset, on Student’s t with one degree of freedom fewer than there
  are variables. `"nested_cv"` is the nested cross-validation interval
  of Bates, Hastie and Tibshirani (2024):
  [`select_grain()`](https://gillescolling.com/timesift/reference/select_grain.md)
  and
  [`grain_ladder()`](https://gillescolling.com/timesift/reference/grain_ladder.md)
  take it as `interval` with `repeats` fold maps, the first being the
  map already cross-validated on, and each repetition fits the procedure
  or the arm once per unordered pair and once per unordered triple of
  outer folds, which needs at least four;
  [`paired_contrast()`](https://gillescolling.com/timesift/reference/paired_contrast.md)
  reads it on the difference of two arms of a ladder fitted with it. A
  fold’s score is the mean over the variables scorable in it, the inner
  estimate is averaged as the reported estimate is, the variance of a
  fold’s score is its delete-one jackknife variance over that fold’s
  units, the root mean squared error is rescaled by `(K - 1) / K` and
  held between the jackknife standard error of the estimate and
  `sqrt(K)` times it, and the centre carries the paper’s bias
  correction, equation (15) at `K` folds. The mean squared error the
  width is read off is that of the bias-corrected estimate: inside every
  outer training set the same nested cross-validation runs once more,
  from the triple fits, at `K - 1` folds, and term (a) is the squared
  gap between the bias-corrected estimate that training set reports and
  the held-out fold’s score, and its root is held between the corrected
  centre’s own jackknife standard error, every prediction held fixed,
  and `sqrt(K)` times it; the paper’s width, read off the plain inner
  estimate and held by the plain estimate’s jackknife standard error, is
  reported beside it as `se_bates`. A fit of the nested cross-validation
  draws its seed as `seed + 10007 r + 101 a + b + 3001 c` from its tag
  `(r, a, b, c)`, the repetition and the folds it leaves out, zero where
  it leaves out fewer than three. Two tables whose contrast is read must
  carry the same response, maps, `repeats` and `seed`.
- A binary prediction, `type = "binary"`, cuts each response at the
  threshold
  [`decision_threshold()`](https://gillescolling.com/timesift/reference/kappa_score.md)
  learns under `rule` from the same candidate’s out-of-fold predictions
  of the fit’s own targets: the member’s own for one candidate, and the
  members’ combined under the refitted stack’s weights for the ensemble.
  Presence is `p >= threshold`. A response whose held-out predictions
  give no cut predicts `NA` in R and NaN in Python, and a fit whose
  response is not 0/1 refuses the cut before anything is built. `rule`
  is `"youden"` by default, as it is for
  [`decision_threshold()`](https://gillescolling.com/timesift/reference/kappa_score.md)
  itself.
- `models` takes one learner, a set or list of them, or the name of a
  registered one, and `learners` on a ladder takes the same three forms.
- Predicting rebuilds each member’s representation for the new targets
  from the settings its own arm was built with, so a new target frame
  has to carry the identifier, the anchor and the static columns the fit
  was made with, and is refused by name where it does not. It carries no
  response column, because a target being predicted has none.
- A learner left without a `data =` runs across every representation of
  the run, and one given a representation there runs at that one alone.
  A pairing the learner cannot read is skipped and reported by name
  inside a set, and is an error where the caller named it.
- A candidate is reported as `learner / representation`, and every
  candidate emits an out-of-fold prediction for every scorable cell over
  the same folds. The combiner is handed those predictions, the
  response, the mask and the fold map, and never a model.
- [`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
  evaluates the procedure nested. Within each outer fold it draws an
  inner map of `inner` folds on the training targets, as
  [`select_grain()`](https://gillescolling.com/timesift/reference/select_grain.md)
  draws one, cross-validates every candidate on it, chooses one by
  `rule` on the inner scores, and fits the stack’s weights on the inner
  out-of-fold predictions over the inner mask; every candidate is then
  refitted on the outer training targets and predicts the test fold,
  which gives each candidate’s outer out-of-fold prediction, the
  selected candidate’s and the stack’s under that fold’s weights. The
  estimate is both of those held-out predictions scored under every
  registered metric, and the run’s own where it is a function, with the
  interval across variables; one outer fold’s choice, weights and
  held-out predictions do not move when that fold’s responses change.
  `inner` left unset is 5, and `inner = NULL` / `inner=None` runs no
  search and makes no estimate. With one candidate there is no search
  and the candidate is its own choice. `choice` is the rule applied to
  the outer scores, with the outer folds as its split, and the stack a
  prediction goes through is fitted on the outer out-of-fold
  predictions, so neither is the one the estimate was read under.
- [`grain_contrasts()`](https://gillescolling.com/timesift/reference/grain_contrasts.md)
  fits `score ~ grain + (1 | variable) + (1 | fold)` by restricted
  maximum likelihood on one learner’s scored cells, with each random
  intercept’s standard deviation relative to the residual one as the
  parameter the fit is found over, as lme4 parametrises it. Each grain
  is compared with the reference by its treatment-coded coefficient, the
  reference being the learner’s best grain unless one is named. A
  contrast’s degrees of freedom are Satterthwaite’s as lmerTest reads
  them: `2 v^2 / (g' A g)`, with `A` twice the inverse Hessian of the
  restricted deviance in the random effects’ relative standard
  deviations and the residual one, and `g` the gradient of the
  contrast’s variance in the same. The `"mvt"` adjustment is emmeans’: a
  degree of freedom is read as `floor(df + 0.25)`, at least one, and as
  the normal above 9999; the p-value is one minus the probability that
  every contrast’s statistic lies within the observed one’s absolute
  value, and the interval is the estimate plus and minus the 0.95 point
  of the largest absolute statistic. Both are integrals of a
  multivariate t the two sides evaluate by quasi-Monte Carlo, so they
  agree to the integrator’s error, of the order of a thousandth; the
  differences agree to the optimiser’s. R reads it with lme4, lmerTest
  and emmeans, and Python with scipy, each the dependency its function
  names and errors without.
- [`simulate_records()`](https://gillescolling.com/timesift/reference/simulate_records.md)
  builds its design from the calendar
  [`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md)
  bins by: the anchors of each variable’s stretch of bins, its weights
  over the readings, the population standard deviation of its driver and
  the link coefficients solved for `prevalence` and `auc` are the same
  numbers on both sides, since none of them is drawn. The units are
  drawn from each language’s own stream, as a fold map is, so two draws
  from one `seed` and `draw` are the same draw in distribution and not
  number for number.
- [`plot()`](https://rdrr.io/r/graphics/plot.default.html) returns the
  table it drew from, and the colours it draws in by default are R’s
  `hcl.colors(n, "Dark 3")` on both sides.
- The combiner minimises the loss of the head the run was fitted under.
  [`ensemble()`](https://gillescolling.com/timesift/reference/ensemble.md)
  left without a `response` takes the run’s, and one naming a different
  head is refused before anything is fitted;
  [`ensemble_fit()`](https://gillescolling.com/timesift/reference/ensemble_fit.md)
  called on its own reads an unnamed head as `presence_absence`.

### The same thing, shaped differently

| concept | in R | in Python |
|----|----|----|
| a learner of your own | [`learner()`](https://gillescolling.com/timesift/reference/learner.md), a constructor taking the fit and the predict | `Learner`, the dataclass, built directly with the same fields |
| a learner’s own training settings | a `control` field holding a partly specified [`train_control()`](https://gillescolling.com/timesift/reference/train_control.md) | its `params`, beside the architecture |
| the occlusion profile | [`occlusion()`](https://gillescolling.com/timesift/reference/occlusion.md), an S3 generic with methods on a run and on a ladder | [`occlusion()`](https://gillescolling.com/timesift/reference/occlusion.md), one function taking either |
| the report on a run | [`summary()`](https://rdrr.io/r/base/summary.html), a method on the base generic, printing the candidates and the procedure | [`summary()`](https://rdrr.io/r/base/summary.html), one function returning the text, with `candidate_table()` and `procedure_table()` for the two tables it prints |
| predicting new targets | [`predict()`](https://rdrr.io/r/stats/predict.html), a method on the base generic | `.predict()`, a method on the fit |
| a binary prediction | an integer matrix of 0 and 1, `NA` where no cut is learned | a float array of 0.0 and 1.0, NaN where no cut is learned |
| a spread | an array whose third dimension is named by the statistic | an array whose last axis is ordered as `SPREAD_STATISTICS` |
| the cuts of a fit | [`decision_threshold()`](https://gillescolling.com/timesift/reference/kappa_score.md), an S3 generic with a method on a fit, returning a named vector | [`decision_threshold()`](https://gillescolling.com/timesift/reference/kappa_score.md), one function taking a fit or a response, returning a dict |
| a representation as a block of predictors | [`as.matrix()`](https://rdrr.io/r/base/matrix.html), a method on the base generic | `flatten()` |
| a set of learners, or of representations | [`c()`](https://rdrr.io/r/base/c.html), an S3 method on each spec class | a `list`, and `+` between two of them |
| drawing a ladder, a run or a selection | [`plot()`](https://rdrr.io/r/graphics/plot.default.html), a method on the base generic for each, on base graphics | [`plot()`](https://rdrr.io/r/graphics/plot.default.html), one function taking any of the three, on matplotlib |
| a simulated record | a `timesift_simulation` list whose `readings` is a data frame | the `Simulation` dataclass, whose `readings` is a mapping of column to array; the first reading instant is `from_`, since `from` is a keyword |
| boosting’s L2 penalty on a leaf | `boosting(lambda =)` | `boosting(lambda_=)`, since `lambda` is a keyword |
| a tuned learner | [`tune()`](https://gillescolling.com/timesift/reference/tune.md), a learner whose fitted model is a `timesift_tuned` holding `model`, `chosen` and `table` | [`tune()`](https://gillescolling.com/timesift/reference/tune.md), a `Learner` whose fitted model is a `Tuned` holding the same three, `table` a list of rows |
| the settings a candidate chose | the `settings` column of `candidates`, `NA` where the learner was not tuned | the `settings` column of `candidates`, an empty string there |
| a map | [`project()`](https://gillescolling.com/timesift/reference/project.md), on `terra` rasters, returning a `SpatRaster` with a layer per response (`response.statistic` under a spread) | [`project()`](https://gillescolling.com/timesift/reference/project.md), on `xarray` grids, returning a `DataArray` over `response`, and `statistic` under a spread, and the two spatial dimensions |
| the change in range | [`range_change()`](https://gillescolling.com/timesift/reference/range_change.md), a list of a `table` data frame and a `map` raster of the codes | [`range_change()`](https://gillescolling.com/timesift/reference/range_change.md), a `RangeChange` whose `table` is a list of rows and whose `map` is a `DataArray` |
| the scores of a repeated resampling | `scores` and `cells` data frames with a `repeat` column | `scores` a mapping of arrays with a `repeat` array; `cells` without one |
| a response curve | [`response_curve()`](https://gillescolling.com/timesift/reference/response_curve.md), an S3 generic with a method on a run, a data frame in long form with a [`plot()`](https://rdrr.io/r/graphics/plot.default.html) method; the second predictor is `with =` | [`response_curve()`](https://gillescolling.com/timesift/reference/response_curve.md), one function returning a `ResponseCurve` whose `prediction` is `[value, response]`; the second predictor is `with_=`, since `with` is a keyword |
| the occlusion of an ensemble | `occlusion(fit, "ensemble", over = "channel")` | the same call, `occlusion(fit, "ensemble", over="channel")` |

All of these are shapes rather than behaviours: a setting given to a
learner beats the run’s control on both sides, the profile is one
implementation on both sides, and a set is the same set.
[`c()`](https://rdrr.io/r/base/c.html) is a method rather than a second
constructor because R’s own way to combine things of one kind is
[`c()`](https://rdrr.io/r/base/c.html); a Python list already
concatenates, so nothing is added there.
[`summary()`](https://rdrr.io/r/base/summary.html) and
[`predict()`](https://rdrr.io/r/stats/predict.html) are methods in R
because R has the generics to add them to, and Python’s
[`summary()`](https://rdrr.io/r/base/summary.html) is a function because
it has none.

### Present in one language only

| in R only | why |
|----|----|
| [`starts_with()`](https://tidyselect.r-lib.org/reference/starts_with.html), [`ends_with()`](https://tidyselect.r-lib.org/reference/starts_with.html), [`contains()`](https://tidyselect.r-lib.org/reference/starts_with.html), [`matches()`](https://tidyselect.r-lib.org/reference/starts_with.html), [`all_of()`](https://tidyselect.r-lib.org/reference/all_of.html), [`any_of()`](https://tidyselect.r-lib.org/reference/all_of.html), [`everything()`](https://tidyselect.r-lib.org/reference/everything.html) and [`where()`](https://tidyselect.r-lib.org/reference/where.html) | tidyselect’s verbs, re-exported so that `y = starts_with("sp_")` is written the way R writes a selection. Python has no non-standard evaluation, so a selection there is a name, a list of names, a glob such as `"sp_*"` or a predicate on the name, resolved by `select_columns()`. |

| in Python only | what it is |
|----|----|
| `align_folds`, `as_response`, `as_resampling`, `get_learner`, `resolve_metric`, `cohen_kappa`, `auto_grains`, `expand_sift`, `resolve_folds`, `n_targets`, `target_labels`, `select_columns`, `column_names` | the helpers R keeps unexported: `.as_folds()`, `.as_response()`, `.as_learner()`, `.as_metric()`, `.kappa_table()`, `.auto_grains()` and `.select_columns()` do the same work by the same name, and `.as_fold_map()`, `.sift_specs()` and `.target_frame()` do what the last five do. A Python module namespace is flat, and anyone writing a learner or reading an artifact against this side reaches them. |
| `Representation`, `Sift`, `Resampling`, `TimesiftSpec`, `Learner`, `TrainControl`, `TimesiftMatrix`, `TimesiftSet`, `Coverage`, `Response`, `Folds`, `Cells`, `Fit`, `Ladder`, `Selection`, `Timesift`, `Stack`, `EnsembleSpec`, `Simulation` | the types. R attaches a class attribute to a list or an array and the constructor is the only door to it; a Python dataclass is the type itself, and a user annotating a function or building one by hand reaches it by name. |
| `GRAINS`, `STATS`, `DAY_LEVEL_STATS`, `SPREAD_STATISTICS`, `PRESENCE_ABSENCE`, `CONTINUOUS`, `ABUNDANCE`, `ORDINAL`, `COUNT` | the grain, statistic and spread vocabularies as tuples, and the shipped heads as the mappings [`register_response()`](https://gillescolling.com/timesift/reference/register_response.md) takes. R holds the vocabularies unexported and prints them in the error that refuses a name; the head is reached through [`responses()`](https://gillescolling.com/timesift/reference/register_response.md) on both sides. |

Models are the one thing neither side promises. A fit in torch and a fit
in libtorch cannot be byte-identical, and the encoders match module for
module rather than number for number.
