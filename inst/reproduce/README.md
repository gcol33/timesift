# Reproducing the Schrankogel grid

`schrankogel.R` runs the published grid from the deposit it was built on.

```
Rscript schrankogel.R <deposit_dir> <out_dir> [--name=value ...]
```

`<deposit_dir>` is the unpacked `data` directory of the Chytrý et al. deposit
(doi:10.5281/zenodo.17047026, CC BY 4.0). Four of its files are read: `spe_wide.csv`,
`logger_data.csv`, `seasons.csv` and `output_temperature_variables_scaled.csv`. Each is checked
against the size and the MD5 sum of the deposit's copy before it is read, so a file that is not
the deposit's stops the run rather than producing a number of the right shape. `run.meta` in
`<out_dir>` records the deposit, the package version, the fold map and the sum of every file
written.

`folds.csv` beside the script is the study's own fold map, 894 loggers in ten folds. It is the
map every number below was checked against, and the default.

## Stages

| stage | what it does | writes |
|---|---|---|
| `contract` | the species filter, the fold map, the mask of scorable cells | `cells.csv` |
| `representation` | the record at all seven grains, with every bin count asserted | `representation.csv` |
| `baseline` | the aggregated-feature arms on the deposit's 188 variables, and the penalised fit on the weekly three-channel series | `baseline.csv`, `baseline_series.csv` |
| `selection` | the study's own procedure through `select_grain()`: 33 candidates, ten outer folds, the study's five inner folds, the convolutional encoder | `selection.csv`, `selection_inner.csv`, `selection_cells_auc.csv`, `selection_cells_tss.csv`, `selection_contrast.csv` |
| `networks` | the encoders, and the eleven-member set, across the ladder | `networks_mean.csv`, `networks_extremeday.csv` |
| `contrasts` | every pair of arms, paired inside each cell both scored | `contrasts.csv` |
| `grains` | each grain against its architecture's best, from the mean-reading network grid | `grain_contrasts.csv` |
| `inflation` | what the reported levels are upper bounds on, and the level each reported score implies | `inflation.csv`, `implied_skill.csv` |

The contract stage always runs, since everything downstream reads its response and its folds. The
default is every stage but `networks` and `selection`, the two that fit encoders.

## The selection stage

The demonstration in the paper chooses one of 33 candidates inside each outer training set, on five
inner folds, by the area under the curve, and refits the winner on the whole training set. That is
what `select_grain()` does, so this stage is the demonstration rather than a description of it:

```
Rscript schrankogel.R <deposit_dir> <out_dir> --stages=contract,baseline,selection \
  --baseline=series --epochs=60
```

A candidate is a window and a summary. The mean is defined at all seven windows, the reading-level
minimum and maximum and their three-channel pair from the half-daily window up, and the two
day-level pairs from the weekly window up, since they reduce to whole days first: 7 + 6 + 6 + 6 + 4
+ 4 = 33. Each is given the two calendar channels the encoders of the study read.

The inner partition is the study's own, `inner_folds.csv` beside the script, which holds one inner
fold per (outer fold, training plot). `--inner=build` deals five inner folds per outer training set
with `fold_map()` instead, which is a different partition of the same design.

It is 1,660 encoder fits at the full candidate set: ten outer folds by five inner folds by 33
candidates, plus one refit per outer fold. The two finest windows are 26,304 and 2,192 steps per
plot and want a graphics processor. `--grains` narrows the windows, and a narrowed run says how
many candidates it searched rather than asserting 33.

## Smoke runs

`--smoke=<outer folds>,<species>` runs the stages on a few folds and the most frequent species, to
see them run before an overnight one. It names every file it writes `smoke_`, records itself as a
smoke run in `run.meta`, and compares nothing with the study. Its output is not the reproduction.

## What each comparison is made at

Every comparison with a published number states its tolerance in the script before anything is
fitted, and `checks.csv` holds the table: the quantity, what this run got, what the study reported,
the difference, the tolerance and whether it is inside. Nothing stops the run, so a number outside
its tolerance is a finding rather than a crash.

The selected window is compared exactly: the study's five-inner-fold selection chose a weekly
candidate in all ten outer folds. Which weekly summary won is reported and not compared: in five of
the ten folds the best and second-best inner scores differ by less than 0.0013, and in one by
0.00001, which is inside the seed noise of a single encoder fit.

## The environment on the output

`run.meta` records the R version and platform, the package version, and the torch and libtorch
versions and the device, since a network fitted under another libtorch or on another device does
not return the same weights.

## Options

- `--stages` comma-separated stage names.
- `--grains` which grains the network grid covers. Default `day,week,month,season,year`, the
  five coarse grains; `native` and `halfday` are 26,304 and 2,192 steps per plot and want a
  graphics processor, as they had in the study.
- `--learners` which encoders, plus `ensemble` for the eleven-member set. Default `cnn`.
- `--baseline` which arms: `elastic_net` and `stepwise` on the deposit's 188 aggregated variables,
  `series` for the penalised fit on the weekly coldest-day, mean and warmest-day reading of the
  record itself, or several. Default `elastic_net`. Forward selection over 188 columns is one fit
  per candidate per step per species per fold and takes many hours single-threaded.
- `--inner` a CSV of `outer_fold`, `logger_ID` and `inner_fold`. Default `inner_folds.csv` beside
  the script, the study's own inner partition; `--inner=build` deals its own.
- `--smoke` `<outer folds>,<species>`, see Smoke runs.
- `--folds` a CSV of `logger_ID` and `fold`. Default `folds.csv` beside the script, the study's
  own map. `--folds=build` draws a map with `fold_map()` instead, which is a different partition
  of the same design: `fold_map()` draws on R's random stream and the study's map came from
  `rsample`.
- `--epochs` epoch budget per network fit. Default 60, the budget the study used.

## The ensemble arm

Asked for with `--learners=...,ensemble`, the eleven members run as arms of their own on the same
folds as every other arm, and one further arm per grain is the mean of their held-out predictions,
scored on the same cells by the same metric. A member's out-of-fold prediction on a fold is its
held-out prediction there, so averaging the eleven and choosing a threshold afterwards is the set
scored as one model rather than as a vote between eleven decisions. All twelve arms are written, so
a member's own level is readable beside the level the set reached.

## What it asserts before fitting anything

The size and the MD5 sum of every deposit file it reads, then the plot count, the species count
after the contract filter, the rarest retained species, the cell count, the reading count, the
readings per plot, and the bin count of every grain. A mismatch stops the run rather than
producing a number nobody can trace to an input.

Every stage reports the fold it is on as it goes, so a run measured in hours is distinguishable
from one that has hung.

## What it costs

Measured on one Windows processor, R 4.6.1:

| stage | time |
|---|---|
| contract | seconds |
| representation, all seven grains | about 2 minutes, 75 s of it reading the 1.2 GB CSV |
| baseline, elastic net, 101 species by 10 folds | tens of minutes |
| inflation, 2000 replicates at three planted levels | minutes |
| networks, coarse grains, one encoder | hours |
| networks, the hourly rung | wants a graphics processor, as it had in the study |

Nothing here caps the data. A stage runs over all 894 plots and all 101 species or it does not run.

## What has been checked against the paper

Run on 2026-09-02 with the study's own fold map, the `folds.csv` shipped beside the script.

| quantity | paper | this run |
|---|---|---|
| plots, species, rarest species | 894, 101, 26 | 894, 101, 26 |
| scorable cells | 1003 of 1010 (99.3%) | 1003 of 1010 (99.3%) |
| species with a scorable fold | 101 of 101 | 101 of 101 |
| bins per grain | 26304, 2192, 1096, 157, 36, 13, 3 | same |
| numbers per plot, weekly three-channel | 471 | 471 |
| numbers per plot, daily three-channel | 3288 | 3288 |
| inflation at truth 0.60, 0.70, 0.90 | +0.110, +0.095, +0.051 | +0.110, +0.095, +0.051 |
| elastic net on the 188 aggregates | 0.687 | 0.686 |

The elastic net sits 0.001 below the published figure. Its penalty is chosen by an inner
cross-validation whose folds are drawn at random, and the two runs seed that stream differently:
the study seeded one stream across its parallel workers, this one seeds once per fitted fold. The
counts and the representation carry no such randomness and reproduce exactly.

The stepwise arm and the network grid have not been rerun here. Forward selection over 188 columns
is many hours single-threaded, and the encoders want the graphics processor they had in the
study.
