# timesift (Python)

The Python side of `timesift`. It fits and compares representations of time-varying data against a
prediction target, from the same one call the R package offers, and it answers to
`../inst/spec/representation.md`, the document both implementations answer to.

```python
import timesift as ts

fit = ts.timesift(plots, logger, y="sp_*", id="plot_id", time="datetime",
                  models=[ts.elasticnet(), ts.forest()],
                  sift=ts.grains("day", "week", "month"),
                  resampling=ts.cv(v=5))
print(ts.summary(fit))
```

```
timesift  80 targets, 4 responses, 5-fold random CV, tss

candidate                    mean    won  responses
forest / week               0.503      0  separate
elasticnet / month          0.504      0  separate
forest / month              0.507      1  separate
elasticnet / day            0.515      1  separate
elasticnet / week           0.528      1  separate
forest / day                0.535      1  separate
ensemble                    0.571      -

weights  elasticnet / week 0.37   elasticnet / day 0.22   forest / month 0.17   elasticnet / month 0.12   forest / day 0.09   forest / week 0.04
```

`targets` and `series` are mappings of column name to array, which a data frame satisfies. `y`, `x`
and `static` are selections over their own table: a column name, a list of names, a glob such as
`"sp_*"`, or a function of a name. `fit.predict(targets, series)` rebuilds each member's
representation for the new rows and combines them.

## What is here

- `timesift`, `Timesift`: the entry point and the fitted run, carrying its candidates, its scores,
  its out-of-fold predictions and its stack.
- `native`, `grain`, `multigrain`, `lookback` and the sets `grains` and `lookbacks`: what a
  representation is, before any record has been read. `build_representation` turns one into the
  array a learner is handed.
- `grain_matrix`, `lookback_matrix`, `calendar_channels`, `bind_channels`, `feature_matrix`,
  `timesift_set`: the arrays themselves, reachable without the fitting layer. `coverage` is the
  count of readings per unit and bin, which is where a refused record's gaps are read off.
- `cv`, `grouped_cv`, `fold_map`, `read_folds`, `scorable_cells`: the split, and which cells admit
  a score.
- `elasticnet`, `stepwise`, `forest` (scikit-learn), `mlp`, `cnn`, `rescnn` (torch), and `Learner`
  for one of your own. `train_control` carries how any of the neural ones is trained.
- `ensemble`, `ensemble_fit`, `ensemble_combine`, `ensemble_weights`: the stack over the
  candidates' out-of-fold predictions.
- `tss`, `roc_auc`, `kappa_score`, `model_agreement`, `decision_threshold`: the metrics.
- `grain_ladder`, `select_grain`, `paired_contrast`, `tss_inflation`, `implied_skill`: fitting
  across a set of grains on its own, comparing two arms cell by cell, and reading a level that was
  taken at its own best threshold.
- `occlusion`: hold each bin or each channel back and rescore, without refitting. It takes a run or
  a ladder.
- `register_learner`, `register_metric`, `register_response`: the three registries the fitting path
  reads.

The mixed-model grain contrast the R side offers as `grain_contrasts()` has no counterpart here.
The contract's last section carries the rest of what each language holds.

The binning and the reduction are not written here. They are `../src/ts_core.cpp`, the same
implementation the R package compiles, reached through the `_core` extension that `../CMakeLists.txt`
builds with nanobind. What is written here is the boundary: resolving the columns, resolving the
zone, and putting the result into a `TimesiftMatrix`.

At runtime the package needs numpy alone. A learner that needs a package declares it and stops
without it.

## The time zone

`grain_matrix` takes a `tz` argument. Left at `None` the instants are taken as already expressed
in the calendar to bin by, which is what a zone-free `datetime64` says. Given a zone name they are
read as UTC and binned by that zone's clock, which is what the R side does for a series carrying a
`tzone`. The same instants and the same zone give the same answer in both languages, and
`digests.csv` carries zone rows that pin it.

A time column that carries a zone of its own names the calendar the same way, so a `pandas` column
in `Europe/Vienna` bins by Vienna days without being told to. Naming a different zone in `tz`
beside one the column carries is an error rather than a silent choice between the two.

## The fold map crosses the language boundary; the fold builder does not

`fold_map` draws on numpy's random stream and the R side draws on R's, so the same seed gives
different maps. Where both languages must see identical splits, build the map once and read it in
the other with `read_folds`. The map is an artifact, like the response and the representation.

## Two rules govern this directory

- **`tests/oracle.py` is implemented from the spec, not transcribed from the R source.** It is the
  NumPy representation as it was written before the two languages shared a core, kept because
  reading the R code and copying it would reproduce its bugs and hide its assumptions. Nothing
  imports it outside the suite; it exists so the shared core is checked against an implementation
  that shares none of its code.
- **A digest mismatch is a bug, never a fixture to regenerate.** Regenerating fixtures happens on
  the R side, deliberately, in its own commit, and only when the spec changed with it.

The project directory is the repository root, because a source distribution cannot reach above
itself and the shared sources are not vendored into a second copy. Build and test from there:

```
pip install -e ".[test,torch,sklearn]"
pytest
```
