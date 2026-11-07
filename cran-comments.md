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

The version is 0.6.1. 0.6.0, the version of the two earlier submissions, did not reach CRAN, but
it was published on PyPI as the package's Python twin, which compiles the same C++ sources; 0.6.1
gives the fixed sources a version of their own on both. This upload replaces the 0.6.0 upload of
earlier today, which differs from it only in the version number. It carries the development since
0.3.1, listed in NEWS.md. The tests that take 0.1 s or more call
`skip_on_cran()`, as they did in the 2026-10-15 submission, after 0.6.0 was first returned on
2026-10-07 for an overall check time of 13 minutes.

The native learners take a `threads` argument (default 1). When `_R_CHECK_LIMIT_CORES_` is set,
the package starts at most two threads, as the parallel package does.

## R CMD check results

0 errors | 0 warnings | 1 note

The note is the incoming feasibility check: a new submission of a package archived on CRAN, and
these possibly misspelled words in DESCRIPTION. Allouche, Tsoar, Kadmon, Hastie and Tibshirani are
authors of the cited methods. "lookback" is the package's term for a span of a record ending at
each target's own time, and the name of the function that builds one; "unreduced" describes the
record as recorded, before any reduction to a coarser grain.

## Test environments

* r-hub clang23 container (LLVM 23, libc++), R-devel, `R CMD check --as-cran`: `Status: OK`,
  `[ FAIL 0 | WARN 0 | SKIP 249 | PASS 1388 ]`. The same container on the previous commit
  reproduced the 11 failures of CRAN's clang23 rerun.
* win-builder: R-devel (2026-11-06 r90845), `Status: 1 NOTE`, check time 289 s (tests 25 s)
* win-builder: R-release (4.6.2), `Status: 1 NOTE`, check time 220 s
* local Windows 11, R 4.6.1, `--as-cran`: 1 NOTE (new submission, archived)
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
