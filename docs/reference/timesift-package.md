# timesift: Learn Predictive Representations of Time-Varying Data

Finds the time scale at which a long time-stamped record best predicts a
response.
[`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
builds the record at each grain or lookback named in `sift`, fits every
learner on every representation it can read, scores them all on the same
held-out folds and cells, and chooses or stacks them inside each
training fold. The arrays themselves come from
[`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md)
and
[`lookback_matrix()`](https://gillescolling.com/timesift/reference/lookback_matrix.md)
without the fitting layer. The shipped default is presence-absence with
a joint multi-label head, scored by AUC with the true skill statistic
beside it.

## Details

Start with
[`vignette("timesift")`](https://gillescolling.com/timesift/articles/timesift.md),
or the articles at <https://gillescolling.com/timesift/>.

## See also

Useful links:

- <https://gillescolling.com/timesift/>

- <https://github.com/gcol33/timesift>

- Report bugs at <https://github.com/gcol33/timesift/issues>

## Author

**Maintainer**: Gilles Colling <gilles.colling051@gmail.com>
([ORCID](https://orcid.org/0000-0003-3070-6066)) \[copyright holder\]

Authors:

- Gilles Colling <gilles.colling051@gmail.com>
  ([ORCID](https://orcid.org/0000-0003-3070-6066)) \[copyright holder\]
