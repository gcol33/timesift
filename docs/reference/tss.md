# The true skill statistic

Sensitivity plus specificity minus one, at the threshold that maximises
it. This is the metric species distribution modelling reports, and the
one the shipped presence-absence response is scored by.

## Usage

``` r
tss(y, p, threshold = NULL)
```

## Arguments

- y:

  Observed presence-absence, `0`/`1` or logical.

- p:

  Predicted scores for the same units, in the same order. Higher means
  presence.

- threshold:

  `NULL` for the maximum over every cut, or one cut, presence being
  predicted at `p >= threshold`.

## Value

One number, or `NA` where the cell defines none.

## Details

A cut may only fall between distinct predictions: units sharing a
prediction are decided together, so the same score comes back whatever
order they arrived in. A cell holding only presences or only absences
has no skill to measure and returns `NA` rather than a number.

The threshold is chosen on the same units the score is then read on,
which is how the metric is defined in the literature and how it is
defined here, and it inflates the level where presences are thin.
[`tss_inflation()`](https://gillescolling.com/timesift/reference/tss_inflation.md)
measures that inflation for a given design, and it cancels in the paired
differences
[`paired_contrast()`](https://gillescolling.com/timesift/reference/paired_contrast.md)
takes. Given a `threshold` learned elsewhere, the score is read at that
cut instead, which is what
[`select_grain()`](https://gillescolling.com/timesift/reference/select_grain.md)
reports under `threshold =` with a cut learned on the inner folds.

## Examples

``` r
tss(c(0, 0, 1, 1), c(0.1, 0.2, 0.8, 0.9))
tss(c(0, 0, 1, 1), c(0.9, 0.8, 0.2, 0.1))
tss(c(0, 0, 0, 0), c(0.1, 0.2, 0.8, 0.9))
tss(c(0, 0, 1, 1), c(0.1, 0.6, 0.8, 0.9), threshold = 0.5)
```
