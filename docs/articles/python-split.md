# Python: the split and the cells

One fold map read by everything that scores, and the cells a score is
defined on, computed with no model involved.

[All of the Python
reference](https://gillescolling.com/timesift/articles/python-reference.md)

## Installation

``` r

# Install from CRAN
install.packages("timesift")

# Or install the development version from GitHub
# install.packages("pak")
pak::pak("gcol33/timesift")
```

``` bash
# Install from PyPI
pip install timesift

# with the torch encoders, the contrasts and the plots
pip install "timesift[torch,contrasts,plot]"

# Or install the development version from GitHub
pip install git+https://github.com/gcol33/timesift
```

## `cv()`

``` python
cv(v: int = 10, seed: int = 1, strata: int = 5, by=None, repeats: int = 1)
```

Hold out single targets, balanced within equal-count strata of the
response.

`by` stratifies on a numeric column of `targets` instead of the richness
of the response. `repeats` draws the split that many times, each with
its own seed (`seed`, `seed + 1`, and so on). In `timesift` every repeat
is a full run on its own fold map: a score stays one fit’s held-out
score, the report averages a response’s scores over folds and repeats,
and the combiner is fitted on the out-of-fold predictions of all the
repeats together.

## `grouped_cv()`

``` python
grouped_cv(group, v: int = 10, seed: int = 1, repeats: int = 1)
```

Keep every target sharing a value of the `group` column in one fold.

It is what repeated targets on the same unit need: two visits to a plot
a fortnight apart are not two independent held-out units, and splitting
them across folds scores a model on a unit it has already read.

## `block_cv()`

``` python
block_cv(by, v: int = 4)
```

Hold out a block of targets whole, the blocks cut on the columns `by` of
`targets`.

The units are split at the median of the column with the widest range,
the lower share going to the left, and each part is cut again until
there are `v` of them, so a block holds as many units as another to
within one. Nothing is drawn. Under the nested selection the inner folds
keep blocks whole too, so `inner` is at most `v - 1`. `by` is column
names, or a numeric array with one row per target.

## `env_cv()`

``` python
env_cv(by, v: int = 4)
```

As `block_cv`, on columns centred and scaled first: blocks of predictor
space.

## `Resampling`

``` python
Resampling(kind, v, seed, strata, group, folds, by, scale, repeats)
```

How the targets are split, named but not yet drawn.

Attributes:

- `kind` - str
- `v` - int
- `seed` - int
- `strata` - int
- `group` - str \| None
- `folds` - object
- `by` - object
- `scale` - bool
- `repeats` - int

## `as_resampling()`

``` python
as_resampling(x)
```

A resampling spec, or a fold map somebody else built, read as a spec
that returns it.

## `resolve_folds()`

``` python
resolve_folds(resampling, y, targets, spec)
```

Draw the fold map a resampling spec names, in the row order of the
response.

## `fold_map()`

``` python
fold_map(y: Response, v: int = 10, seed: int = 1, strata: int = 5, by=None, group=None)
```

Assign units to folds, balanced within equal-count strata of a
stratifying value.

`group` keeps every unit sharing a value in one fold, which is what
repeated targets on the same physical unit need: two visits to a plot
are not two independent held-out units. The deal is then made over the
groups rather than over the units, and a group carries the mean of the
stratifying value of the units in it.

The stream is numpy’s, so a map built here is not the map the R side
builds from the same seed. Where both languages must see identical
splits, build the map once, write it with `write_folds` and read it in
the other with `read_folds`.

## `scorable_cells()`

``` python
scorable_cells(y: Response, folds)
```

Which cells admit a score, from the response and the fold map alone.

A cell needs both classes among the held-out units and both classes
among the units a model is fitted on. Computing the mask without a model
is what lets every arm be restricted to the same cells, so their means
share a denominator and every paired difference runs on matched cells.

## `align_folds()`

``` python
align_folds(folds, units)
```

A fold map reaches the fitting path as an integer vector in the row
order of the representation, whether it arrived as a `Folds`, a mapping
of unit to fold, or a bare vector already in that order.

## `Folds`

``` python
Folds(fold, units, grouped, group)
```

Which fold each unit is held out in, named by unit.

Named rather than positional, because a fold map is aligned to a
representation by unit and never by row: two tables of the same height
are not two tables in the same order.

Attributes:

- `fold` - np.ndarray
- `units` - tuple\[str, …\]
- `grouped` - bool
- `group` - tuple\[str, …\] \| None

### `v`

How many folds the map holds.

### `coerce()`

``` python
coerce(cls, x, units=None)
```

A `Folds`, a mapping of unit to fold, or a bare vector read in `units`
order.

### `align()`

``` python
align(self, units)
```

Put the map into the row order of a representation, by unit and never by
position.

### `as_dict()`

``` python
as_dict(self)
```

The map as unit to fold.

## `Cells`

``` python
Cells(variable, fold, n_occ, pres_train, abs_train, pres_test, abs_test, scorable)
```

Which `(variable, fold)` cells admit a score, and the counts that
decided it.

Attributes:

- `variable` - np.ndarray
- `fold` - np.ndarray
- `n_occ` - np.ndarray
- `pres_train` - np.ndarray
- `abs_train` - np.ndarray
- `pres_test` - np.ndarray
- `abs_test` - np.ndarray
- `scorable` - np.ndarray

### `is_scorable()`

``` python
is_scorable(self, variable: str, fold: int)
```

Whether one `(variable, fold)` cell admits a score.
