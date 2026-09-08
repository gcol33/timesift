# Python: the representation

What a representation is before any record has been read, and the array
it becomes.

## `native()`

``` python
native(stats='mean', year_start='09-01')
```

The record unreduced: one bin per reading.

## `grain()`

``` python
grain(g: str, stats='mean', year_start='09-01')
```

One calendar grain.

## `multigrain()`

``` python
multigrain(grains=None, stats='mean', year_start='09-01')
```

Several grains flattened and bound side by side into one block of
features.

Left at `None` the grains are the ones the record supports, the set
`auto_grains` names. A caller who does not want the record unreduced
among them names the grains instead.

## `lookback()`

``` python
lookback(span, lag='0 days', bins=1, stats='mean')
```

A stretch of record of fixed length, ending a fixed lag before each
target’s own instant.

## `grains()`

``` python
grains(*g, stats='mean', year_start='09-01')
```

A sift over calendar grains, named or read off the record with `"auto"`.

## `lookbacks()`

``` python
lookbacks(*spans, lag='0 days', bins=1, stats='mean')
```

A sift over lookbacks of several lengths, all sharing a lag and a number
of bins.

## `Representation`

``` python
Representation(label, kind, stats, grain, grains, span, lag, bins, sequence, year_start)
```

One reduction, named but not yet built.

`sequence` says whether the bins are ordered in time and mean something
to a convolution: the record unreduced, a calendar grain and a lookback
cut into several bins are sequences; a block of features bound side by
side is not.

Attributes:

- `label` - str
- `kind` - str
- `stats` - tuple\[str, …\]
- `grain` - str \| None
- `grains` - tuple\[str, …\] \| None
- `span` - object
- `lag` - object
- `bins` - int
- `sequence` - bool
- `year_start` - str

## `Sift`

The representations a set of candidates runs across, as a mapping of
label to spec.

## `as_sift()`

``` python
as_sift(x)
```

A sift, whether it arrived as one, as a representation, as a grain name,
or as a list.

## `expand_sift()`

``` python
expand_sift(sift, series, spec: TimesiftSpec)
```

The sift with `"auto"` replaced by the grains the record supports.

## `auto_grains()`

``` python
auto_grains(series, spec: TimesiftSpec, stats=('mean',), year_start='09-01')
```

The named grains that give the record at least two bins, from the finest
to the coarsest.

The count comes from the calendar in the core rather than from
arithmetic here, so a grain is admitted on the same rule that will bin
it. It is read off one reading per distinct instant, which carries the
record’s whole span and its gaps at the cost of a single unit’s memory.

## `build_representation()`

``` python
build_representation(rep: Representation, series, targets, spec: TimesiftSpec)
```

The array one representation names, for these targets, in their own row
order.

## `grain_matrix()`

``` python
grain_matrix(
    data=None,
    id=None,
    time=None,
    value=None,
    *,
    grain='day',
    stats=('mean',),
    year_start='09-01',
    partial='keep',
    tz=None,
)
```

Bin readings by the calendar and summarise every bin.

`data` is a mapping of column name to sequence, or any object with
`__getitem__` over the three column names given by `id`, `time` and
`value`. Naming two or more grains returns a `TimesiftSet`; naming one,
whether as a string or as a sequence of one, returns the representation
itself. `grain` may also be a callable, which is handed the reading
instants and must return the start of each reading’s bin.

`tz` names the calendar to bin by. Left at `None` the instants are taken
as already expressed in that calendar, which is what a zone-free
`datetime64` says and what the R side does for a series carried in UTC.
Given a zone name, the instants are read as UTC and binned by that
zone’s clock, which is what the R side does for a series carrying a
`tzone`: the same instants and the same zone give the same answer in
both languages.

`partial` says what becomes of a bin the record does not cover for its
whole calendar span, which is what a record beginning or ending away
from a bin boundary produces. `"keep"`, the default, returns it
alongside the full bins; `"drop"` removes it. Either way the verdict is
carried on `bin_partial`, so a kept partial bin is labelled rather than
silent. A caller-supplied binning declares its own bins, so the package
cannot know where the last one was meant to end and takes the record’s
end as its end.

## `lookback_matrix()`

``` python
lookback_matrix(
    data=None,
    id=None,
    time=None,
    value=None,
    at=None,
    span=None,
    *,
    lag='0 days',
    bins=1,
    stats='mean',
    tz=None,
)
```

Read a fixed length of record ending a fixed lag before each target’s
own instant.

It is the reduction a calendar cannot express: two targets on the same
unit a fortnight apart read two different stretches of the same series,
so the bins are relative to the target rather than to a month or a week.

`at` is a mapping with an `"id"` array of units and a `"time"` array of
anchor instants, one row per target; a unit may carry any number of
them. Bin `b` of a target anchored at `a` covers
`[a - lag - span + b * step, a - lag - span + (b + 1) * step)`, with
`step` the span divided by `bins` and `b` counted from zero. Only the
readings of the target’s own unit are read, and every `(target, bin)`
cell must hold at least one: a lookback reaching past the record is an
error naming the target, never a padded row.

`span` and `lag` are read from a count and a unit – `"30 days"`,
`"12 hours"`, `"1 year"` – or from a bare number of seconds. A year is
365 days and a month is 30 days here, because a lookback of a fixed
length is a fixed length rather than a calendar step.

`tz` names the calendar, as it does for `grain_matrix`. The anchors are
instants and are read as a clock in that same calendar, so one record is
binned by one calendar.

## `timesift_set()`

``` python
timesift_set(x)
```

Every entry point that fits across grains takes a representation, a set,
or a bare mapping, and works on a set. One coercion, so no caller
repeats the three cases.

## `calendar_channels()`

``` python
calendar_channels(x: TimesiftMatrix)
```

Where in the year each bin sits, as the sine and cosine of its
fractional position.

## `bind_channels()`

``` python
bind_channels(*parts: TimesiftMatrix)
```

Put the channels of several representations of the same units and bins
side by side.

## `feature_matrix()`

``` python
feature_matrix(m, units=None, features=None, label: str = 'features')
```

Bring an already-reduced feature table into a ladder as a one-channel
representation.

It carries no time axis, because it has none: the reduction already
happened, elsewhere, and what reaches the model is a list of numbers per
unit. That is the whole point of comparing against it.

## `TimesiftMatrix`

``` python
TimesiftMatrix(
    values,
    units,
    bins,
    stats,
    grain,
    year_start,
    bin_start,
    bin_end,
    bin_n,
    bin_partial,
    span,
    lag,
)
```

A `[row, bin, channel]` representation and the reduction that produced
it.

A row is a unit where the calendar did the binning and a target where a
lookback did, since a unit carrying several targets cannot name a row on
its own. `span` and `lag` are set by a lookback alone, and are what
rebuilding one for new targets reads.

Attributes:

- `values` - np.ndarray
- `units` - tuple\[str, …\]
- `bins` - tuple\[str, …\]
- `stats` - tuple\[str, …\]
- `grain` - str
- `year_start` - str \| None
- `bin_start` - np.ndarray
- `bin_end` - np.ndarray
- `bin_n` - np.ndarray
- `bin_partial` - np.ndarray
- `span` - int \| None
- `lag` - int \| None

### `shape`

Rows, bins and channels.

### `channel()`

``` python
channel(self, name: str)
```

One statistic as a `[row, bin]` matrix.

### `take_units()`

``` python
take_units(self, index)
```

The representation restricted to a subset of its rows, in the order
given.

## `TimesiftSet`

A ladder of representations, one per grain.

Naming several grains in `grain_matrix` returns one of these:
representations of the same units, differing only in how coarsely the
record was read. It is what `grain_ladder` fits across, and it reads as
a mapping of grain name to representation.

### `units`

The units the set covers, which every grain in it shares.

## `GRAINS`

``` python
GRAINS = ('native', 'halfday', 'day', 'week', 'month', 'season', 'year')
```

## `STATS`

``` python
STATS = ('mean', 'min', 'max') + DAY_LEVEL_STATS
```

## `DAY_LEVEL_STATS`

``` python
DAY_LEVEL_STATS = ('cold_day', 'warm_day', 'mean_daily_min', 'mean_daily_max')
```
