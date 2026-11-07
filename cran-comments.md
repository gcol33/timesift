# cran-comments

## Reason for this submission

timesift was archived on 2026-10-26 because the clang23 additional check was not fixed in time
(Prof Ripley's mail of 2026-10-06, deadline 2026-10-27). This version fixes it.

0.3.1 on CRAN failed to install under clang23: `src/ts_penalised.cpp` used `std::exception_ptr`
without including `<exception>`, which libc++ 23 no longer reaches transitively. The 0.6.0
resubmission of 2026-10-15 installed, but CRAN's clang23 rerun failed 11 tests, each expecting the
message of an error raised in the C++ code and receiving "C++ error (unknown cause)". Every one of
the 11 went through a fit that collects failures from a pool of threads, which carried the
exception to the calling thread through `std::exception_ptr` and `std::rethrow_exception()`; the
errors thrown directly kept their messages in the same run. A failure is now handed back as its
kind and message and thrown afresh on the calling thread, so no exception object crosses threads.
The package's CI now runs the full `R CMD check` in r-hub's clang23 container, the toolchain of
CRAN's check, and it passes there.

The version is 0.6.0, the version of the two earlier submissions that did not reach CRAN. It
carries the development since 0.3.1, listed in NEWS.md. The tests that take 0.1 s or more call
`skip_on_cran()`, as they did in the 2026-10-15 submission, after 0.6.0 was first returned on
2026-10-07 for an overall check time of 13 minutes.

The native learners take a `threads` argument (default 1). When `_R_CHECK_LIMIT_CORES_` is set,
the package starts at most two threads, as the parallel package does.

## R CMD check results

0 errors | 0 warnings | 0 notes

## Test environments

* win-builder: R-release (4.6.1), `Status: OK`, check time 276 s (tests 36 s)
* win-builder: R-devel (2026-10-14 r90673), `Status: OK`, check time 372 s (tests 43 s)
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
