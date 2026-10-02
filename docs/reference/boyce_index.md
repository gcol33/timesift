# The continuous Boyce index

How consistently the predictions rise with the density of presences,
without a cut and without absences (Hirzel et al. 2006). A window of
`width` times the range of the predictions slides over that range in
`resolution + 1` equal steps. In each, the share of presences falling
inside divided by the share of all units falling inside is the
predicted-to-expected ratio, undefined where no unit falls inside, and
the index is the Spearman correlation of that ratio with the window's
midpoint, between -1 and 1. Predictions that rank no better than chance
read near 0.

## Usage

``` r
boyce_index(y, p, resolution = 100L, width = 0.1)
```

## Arguments

- y:

  Observed presence-absence, `0`/`1` or logical.

- p:

  Predicted scores for the same units, in the same order. Higher means
  presence.

- resolution:

  Number of window steps, `100` by default.

- width:

  Window width as a share of the range of `p`, `0.1` by default.

## Value

One number, or `NA` where the cell defines none, where the predictions
are all equal, or where the ratio is defined in fewer than three windows
or takes one value.

## Details

The units of `p` are the background, which is how biomod2 reads `BOYCE`
on a fit. The windows are closed at both ends and the ratio is not
thinned of repeated values.

## Examples

``` r
boyce_index(c(0, 0, 0, 1, 1, 1, 0, 1), c(0.10, 0.20, 0.35, 0.40, 0.60, 0.90, 0.55, 0.70))
```
