# cran-comments

## Reason for this update

This release fixes the installation error the CRAN team reported on 2026-10-06 under the clang23
additional checks (LLVM 23.1, libc++): `src/ts_penalised.cpp` used `std::exception_ptr`,
`std::current_exception()` and `std::rethrow_exception()` without including `<exception>`, which
libc++ 23 no longer reaches transitively. Every source file now includes the header of each
standard-library name it uses; a scan of `src/` against the declaring headers finds nothing else
missing.

It comes four days after 0.3.1 because of that request. It also carries the development since
0.3.1, listed in NEWS.md.

The native learners take a `threads` argument (default 1). When `_R_CHECK_LIMIT_CORES_` is set,
the package starts at most two threads, as the parallel package does.

## R CMD check results

0 errors | 0 warnings | 1 note

The note is the incoming-feasibility "Days since last update: 4", explained above.

## Test environments

* win-builder: R-release (4.6.1), `Status: 1 NOTE` (the one above)
* GitHub Actions: ubuntu-latest (R-release, R-devel), windows-latest, macos-latest

## Notes on things a search of the sources will find

**`globalenv()` in `R/folds.R`.** `.seed_state()` and `.restore_seed()` read and write
`.Random.seed` in the global environment, and they do so to leave the session as they found it:
every entry point that seeds a fold draw saves the stream first and restores it through
`on.exit()`, so a call to the package does not move the user's random state. No other object is
assigned there, and nothing else in the package writes to the global environment.

**`write.csv()` in `inst/reproduce/` and `inst/benchmark/`.** Those are command-line drivers, run
by the user with `Rscript` and not reachable from the package, from a test or from a vignette.
Neither carries a default output path: the reproduction driver takes its output directory as its
second argument, and the benchmark summariser writes a file only when `--csv=` names one. Nothing
in `R/` writes to disk except the exported `write_*()` functions, whose `file` argument is
required and whose examples write to `tempfile()` and `unlink()` it.

**`install.packages()` in error messages.** Several files name the command in the text of a
`stop()` when an optional package from `Suggests` is missing. Nothing is installed by the package.

## The torch learners

`torch` is in `Suggests`. Its runtime library is a separate download rather than part of the CRAN
installation, so the tests that fit a neural learner call
`skip_if_not(torch::torch_is_installed())` and skip themselves where it is absent. The examples on
those learners construct a specification and do not fit, so they run anywhere. No vignette uses
them.

## Downstream dependencies

None.
