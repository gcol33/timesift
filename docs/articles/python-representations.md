# Python: representations

How a series becomes the array a learner reads, and the set a run is
compared across.

[All of the Python
reference](https://gillescolling.com/timesift/articles/python-reference.md)

## `native()`

``` python
native(stats='mean', year_start='09-01')
```

The record unreduced: one bin per reading.

## `grain()`

``` python
grain(g, stats='mean', year_start='09-01')
```

One calendar grain, named, or supplied as a function of the reading
instants returning each reading’s bin start, which is reported as
`custom`.

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
- `grain` - object
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
record’s whole span and its gaps at the cost of a single unit’s memory,
in the zone the time column carries, so a grain is counted on the clock
it will be binned by.

## `build_representation()`

``` python
build_representation(rep: Representation, series, targets, spec: TimesiftSpec)
```

The array one representation names, for these targets, in their own row
order.
