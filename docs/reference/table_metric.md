# A metric of the two-by-two table of decisions

The predictions are cut into presence and absence, and the table of
decisions against observations is summarised. With `H` the hits, `F` the
false alarms, `M` the misses and `C` the correct negatives:

## Usage

``` r
table_metric(
  y,
  p,
  metric = names(.table_metrics),
  rule = c("youden", "kappa", "prevalence", "mpa"),
  threshold = NULL,
  perc = 0.9
)
```

## Arguments

- y:

  Observed presence-absence, `0`/`1` or logical.

- p:

  Predicted scores for the same units, in the same order. Higher means
  presence.

- metric:

  One of the names in the table.

- rule:

  Threshold rule: `"youden"`, `"kappa"`, `"prevalence"` or `"mpa"`.

- threshold:

  `NULL` for the maximum over every cut, or one cut, presence being
  predicted at `p >= threshold`.

- perc:

  For `rule = "mpa"`, the share of presences the cut must keep, `0.9` by
  default. Where a rule is named without a call to
  [`decision_threshold()`](https://gillescolling.com/timesift/reference/kappa_score.md)
  it takes that default.

## Value

One number, or `NA` where the cell defines none.

## Details

|  |  |
|----|----|
| `metric` | reads |
| `"pod"` | probability of detection, `H / (H + M)` |
| `"pofd"` | probability of false detection, `F / (F + C)` |
| `"far"` | false alarm ratio, `F / (H + F)` |
| `"sr"` | success ratio, `H / (H + F)` |
| `"accuracy"` | `(H + C) / n` |
| `"bias"` | `(H + F) / (H + M)`, the presences called against the presences there are |
| `"or"` | odds ratio, `H C / (M F)` |
| `"orss"` | odds ratio skill score, `(H C - M F) / (H C + M F)` |
| `"csi"` | critical success index, `H / (H + M + F)` |
| `"ets"` | equitable threat score, `(H - h) / (H + M + F - h)` with `h = (H + M)(H + F) / n` |

These are biomod2's evaluation statistics. biomod2 reads each at the cut
that brings that statistic closest to its own optimum on a grid of 100
cuts; here the cut is the one `rule` of
[`decision_threshold()`](https://gillescolling.com/timesift/reference/kappa_score.md)
selects, `"youden"` by default, or a `threshold` learned elsewhere. Read
at the cut that optimises it, some of these are trivial, so the cut is a
separate choice and stays fixed while the statistic changes. Each is
registered under its name at the default rule, so
`grain_ladder(metric = "csi")` reads it. A registration at another rule
is one line:
`register_metric("csi_kappa", function(y, p) table_metric(y, p, "csi", "kappa"))`.

A value the table does not define, a zero denominator, is `NA`, as is a
cell of one class.

## Examples

``` r
y <- c(0, 0, 0, 1, 1, 1, 0, 1)
p <- c(0.10, 0.20, 0.35, 0.40, 0.60, 0.90, 0.55, 0.70)
table_metric(y, p, "pod")
table_metric(y, p, "csi", threshold = 0.5)
```
