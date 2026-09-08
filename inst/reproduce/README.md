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
| `baseline` | the aggregated-feature arms on the deposit's 188 variables | `baseline.csv` |
| `networks` | the encoders, and the eleven-member set, across the ladder | `networks_mean.csv`, `networks_extremeday.csv` |
| `contrasts` | every pair of arms, paired inside each cell both scored | `contrasts.csv` |
| `grains` | each grain against its architecture's best, from the mean-reading network grid | `grain_contrasts.csv` |
| `inflation` | what the reported levels are upper bounds on, and the level each reported score implies | `inflation.csv`, `implied_skill.csv` |

The contract stage always runs, since everything downstream reads its response and its folds. The
default is every stage but `networks`.

## Options

- `--stages` comma-separated stage names.
- `--grains` which grains the network grid covers. Default `day,week,month,season,year`, the
  five coarse grains; `native` and `halfday` are 26,304 and 2,192 steps per plot and want a
  graphics processor, as they had in the study.
- `--learners` which encoders, plus `ensemble` for the eleven-member set. Default `cnn`.
- `--baseline` which aggregated-feature arms: `elastic_net`, `stepwise`, or both. Default
  `elastic_net`. Forward selection over 188 columns is one fit per candidate per step per species
  per fold and takes many hours single-threaded.
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
