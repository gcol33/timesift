# Cohen's kappa, and where two models disagree

A chance-corrected agreement rate on a two-by-two table. Read against
the observed response it is a skill score beside
[`tss()`](https://gillescolling.com/timesift/reference/tss.md); read
between two models' decisions on the same units it says where the two
part company.

## Usage

``` r
kappa_score(y, p, rule = c("youden", "kappa", "prevalence"))

decision_threshold(y, p, rule = c("youden", "kappa", "prevalence"))

model_agreement(y, p_a, p_b, rule = c("youden", "kappa", "prevalence"))
```

## Arguments

- y:

  Observed presence-absence, `0`/`1` or logical.

- p:

  Predicted scores for the same units, in the same order. Higher means
  presence.

- rule:

  Threshold rule: `"youden"`, `"kappa"` or `"prevalence"`.

- p_a, p_b:

  Two models' predictions for the same units.

## Value

For `kappa_score()`, one number. For `decision_threshold()`, the cut
itself, applied as `p >= threshold`. For `model_agreement()`, a one-row
data frame carrying the agreement kappa between two models cut by the
same rule, the share of units they decide differently, and how often
each is the one that is right there.

## Details

Kappa is read at a threshold rather than maximised over one, so the rule
that picks the threshold is part of the statistic. `"youden"` is the
operating point
[`tss()`](https://gillescolling.com/timesift/reference/tss.md) is
defined at and inherits its selection bias; `"kappa"` maximises kappa
itself and inherits the analogous bias; `"prevalence"` cuts at the
observed presence rate, which selects nothing from the labels and is the
rule to read an absolute level at.

## Examples

``` r
y <- c(0, 0, 0, 1, 1, 1)
kappa_score(y, c(0.1, 0.2, 0.6, 0.4, 0.8, 0.9))
model_agreement(y, c(0.1, 0.2, 0.6, 0.4, 0.8, 0.9), c(0.2, 0.1, 0.3, 0.7, 0.9, 0.8))
```
