# Contributing to timesift

Bug reports and pull requests are welcome on the
[issue tracker](https://github.com/gcol33/timesift/issues). With a bug report, include a small
example that reproduces it.

## Building and testing

The R package sits at the repository root, and so does the Python project: `pyproject.toml` and
`CMakeLists.txt` build the Python extension from the same `src/` the R package compiles. A Python
source distribution cannot reach above its own project directory, so the project directory is the
repository and there is one copy of the sources.

R, from the repository root:

```r
devtools::test()
```

Python, from the repository root:

```bash
pip install -e ".[test,torch]"
pytest
```

## The shared fixtures

`inst/spec/representation.md` describes what both languages compute, and `inst/spec/fixtures/`
holds a small input and the expected digests of every grain and statistic, plus the reference fits
of the learners. Both test suites read the same fixtures.

- **A digest mismatch is a bug, never a fixture to regenerate.** Fixtures are regenerated on the R
  side with `inst/spec/make_fixtures.R`, in a commit of their own, and only when the spec changes
  with them.
- **The test oracles are written from the spec.** `tests/testthat/helper-oracle.R` and
  `python/tests/oracle.py` are pure-R and pure-NumPy implementations of the representation, used
  only by the suites. The NumPy one was written from the spec rather than from the R source, so the
  shared C++ core is checked against code that shares none of its own.
