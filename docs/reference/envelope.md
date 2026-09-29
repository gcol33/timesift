# biomod2's surface range envelope on the flattened representation

One envelope per response, over every bin-by-channel column of the
representation: for each column, the `quantile` and `1 - quantile`
quantiles of its readings over the units present, and a unit predicted
present where every column lies between its two, the ends included. It
is biomod2's `SRE`, and with the same quantile it draws the envelope
`bm_SRE()` draws; the quantile is R's default, type 7.

## Usage

``` r
envelope(data = NULL, quantile = 0.025)
```

## Arguments

- data:

  A representation the learner is pinned to, or `NULL` to run across
  every representation of the run.

- quantile:

  The share of the presences left outside at each end of every column,
  in `[0, 0.5]`. biomod2's default is `0.025`.

## Value

A
[`learner()`](https://gillescolling.com/timesift/reference/learner.md).

## Details

The prediction is zero or one. It enters an ensemble as that, and the
combiner weighs it by its held-out loss like any other candidate; a
threshold read off it has two values to choose from, so its TSS is the
one its own zeros and ones give.

An envelope reads the presences and nothing else. The absences do not
move it, and neither do the response head's case weights, which it takes
no notice of.

Every column has to agree for a unit to be inside, so the more columns a
representation has the fewer units any envelope holds: at the default
quantile each column shuts out about one presence in twenty, and a
weekly representation of three years has 157 columns per channel. A
coarse grain is what an envelope is meant for, and pinning it there with
`data = grain("season")` keeps it there while the rest of the run reads
finer ones.

A response with no presence, or with nothing else, is predicted its
share among the fitting units, and the fit names it in `unfitted`. The
learner needs a presence-absence response, and a head whose loss is not
the binary cross-entropy is refused.

## Examples

``` r
envelope()
envelope(data = grain("season"), quantile = 0.05)
```
