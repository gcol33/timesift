# Python

[`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
here and
[`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
in R take the same two tables and do the same thing: build every
candidate representation, fit the learners that can read each one, score
them all on one set of held-out folds, and stack the out-of-fold
predictions. The binning, the statistics and the array assembly are
`src/`, compiled into both languages and answering to [the
representation
contract](https://gillescolling.com/timesift/articles/contract.md), and
the fixtures under `inst/spec/fixtures/` hold the two to the same
numbers.

``` bash
pip install git+https://github.com/gcol33/timesift
```

``` python
import timesift as ts

fit = ts.timesift(plots, logger, y="sp_*", id="plot_id", time="datetime",
                  models=[ts.elasticnet(), ts.forest()],
                  sift=ts.grains("day", "week", "month"))
print(ts.summary(fit))
```

`y`, `x` and `static` are selections over their own table: a column
name, a list of names, a glob such as `"sp_*"`, or a function of a name.
The contract’s last section says what each language carries, so a
difference between the two is a recorded decision.

## The pages

- [The
  representation](https://gillescolling.com/timesift/articles/python-representation.md):

  What a representation is before any record has been read, and the
  array it becomes.

- [The split and the
  cells](https://gillescolling.com/timesift/articles/python-split.md):

  One fold map read by everything that scores, and the cells a score is
  defined on, computed with no model involved.

- [Fitting](https://gillescolling.com/timesift/articles/python-fitting.md):

  The run from targets and series, the combiner over its candidates, and
  fitting across a set of grains on its own.

- [Learners](https://gillescolling.com/timesift/articles/python-learners.md):

  The arms that ship, how they are trained, and the interface a learner
  of your own goes through.

- [Scoring and
  comparison](https://gillescolling.com/timesift/articles/python-scoring.md):

  The metrics, the paired contrast between two arms on matched cells,
  the inflation of a score read at its own best threshold, and what a
  fitted model read.

- [Extending](https://gillescolling.com/timesift/articles/python-extending.md):

  The response head and the metric are registrations, never a fork of
  the fitting code.

- [What crosses the
  boundary](https://gillescolling.com/timesift/articles/python-artifacts.md):

  The three artifacts a split is carried in, and the digest that says
  two arrays are the same array.
