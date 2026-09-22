# cran-comments

## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new release.

The note also lists seven possibly misspelled words in the Description. Five are the surnames of
the authors of the two papers cited there: Allouche, Hastie, Kadmon, Tibshirani and Tsoar. The
other two are spelled as intended: `lookback()` is one of the package's exported functions and the
name of the representation it builds, and a record the package leaves unreduced is described as
unreduced.

## Test environments

* local: Windows 11, R 4.6.1, `--as-cran`
* win-builder: R-devel (2026-09-21 r90579) and R-release (4.6.1), both `Status: 1 NOTE`
* GitHub Actions: ubuntu-latest (R-release, R-devel), windows-latest, macos-latest

## Notes on two things a search of the sources will find

**`globalenv()` in `R/folds.R`.** `.seed_state()` and `.restore_seed()` read and write
`.Random.seed` in the global environment, and they do so to leave the session as they found it:
every entry point that seeds a fold draw saves the stream first and restores it through
`on.exit()`, so a call to the package does not move the user's random state. No other object is
assigned there, and nothing else in the package writes to the global environment.

**`write.csv()` in `inst/reproduce/` and `inst/benchmark/`.** Those are command-line drivers, run
by the user with `Rscript` and not reachable from the package, from a test or from a vignette.
Neither carries a default output path: the reproduction driver takes its output directory as its
second argument, and the benchmark summariser writes a file only when `--csv=` names one. Nothing
in `R/` writes to disk except the three exported `write_*()` functions, whose `file` argument is
required and whose examples write to `tempfile()` and `unlink()` it.

**`install.packages()` in three error messages.** `R/contrasts.R`, `R/learner.R` and
`R/learners_torch.R` name the command in the text of a `stop()` when an optional package from
`Suggests` is missing. Nothing is installed by the package.

## The torch learners

`torch` is in `Suggests`. Its runtime library is a separate download rather than part of the CRAN
installation, so the tests that fit a neural learner call
`skip_if_not(torch::torch_is_installed())` and skip themselves where it is absent. The examples on
those learners construct a specification and do not fit, so they run anywhere. No vignette uses
them.

## Downstream dependencies

None; this is a new package.
