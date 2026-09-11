# Python: scoring and comparison

The metrics, the paired contrast between two arms on matched cells, the
inflation of a score read at its own best threshold, and what a fitted
model read.

## `tss()`

``` python
tss(y, p, threshold=None)
```

Sensitivity plus specificity minus one, at the threshold that maximises
it.

Given a `threshold` learned elsewhere, the score is read at that cut
instead, presence being predicted at `p >= threshold`.

## `roc_auc()`

``` python
roc_auc(y, p)
```

The area under the ROC curve, as the rank sum of the presences. Ties
take the average rank.

## `kappa_score()`

``` python
kappa_score(y, p, rule: str = 'youden')
```

Cohen’s kappa of a model’s decisions against the observed response.

## `cohen_kappa()`

``` python
cohen_kappa(a, b)
```

Chance-corrected agreement of two labellings of the same units, in
either order.

## `decision_threshold()`

``` python
decision_threshold(y, p, rule: str = 'youden')
```

The probability cut a rule selects. Presence is predicted at
`p >= threshold`.

## `model_agreement()`

``` python
model_agreement(y, p_a, p_b, rule: str = 'youden')
```

Agreement between two models’ decisions, with how often each is right
where they differ.

## `score_predictions()`

``` python
score_predictions(y, p, folds, cells=None, metric: str = 'tss')
```

Score held-out predictions on the cells the mask allows.

The scoring every arm of a ladder and every candidate of a run goes
through, reachable on its own for a prediction matrix that came from
somewhere else: a combination of arms, a model fitted outside the
package, predictions read back from a file. A cell is one response in
one fold, and the mask is computed from the response and the fold map
alone, never from a model, which is what keeps two arms comparable.

## `paired_contrast()`

``` python
paired_contrast(ladder: Ladder, a: str, b: str, interval: str = 'variables')
```

The difference between two arms, taken inside each cell both scored.

Two arms scored on the same held-out units do not necessarily have the
same set of defined cells, so a difference of two marginal means is not
a difference between the arms. Pairing also cancels what a
threshold-selected metric carries in its level, since both arms carry
the same bias on the same cell.

Each arm is named whole, `grain|learner`. A learner named alone would
take its best grain, chosen on the scores the contrast is then read off,
and pairing does not cancel that choice; `select_grain` chooses a grain
on inner folds instead and contrasts the selection through `compare`.
The interval is Student’s t on one degree of freedom fewer than there
are variables, and `p_method` says whether the signed-rank p-value is
`"exact"` or the `"normal"` approximation, which it is when the
per-variable differences hold a zero or a tie or number fifty or more.

`interval="nested_cv"` replaces that interval with one for the
difference in the two arms’ risk on a new sample, by the nested
cross-validation of Bates, Hastie and Tibshirani (2024) read on the
difference of the two arms’ cell scores, which needs a ladder fitted
with `grain_ladder(interval="nested_cv")`.

## `tss_inflation()`

``` python
tss_inflation(
    y: Response,
    folds,
    skill=(0.6, 0.7, 0.9),
    replicates: int = 200,
    seed: int = 1,
)
```

How much a threshold chosen on the scored units inflates the level it
reports.

Predictions are simulated under a normal model whose population skill is
exactly the value planted, at the cell sizes and presence counts of this
design, and read back the way a ladder reports a level. The gap is the
inflation. It cancels in a paired difference and does not cancel in a
level, so a level is an upper bound on the skill a population has.

## `implied_skill()`

``` python
implied_skill(
    y: Response,
    folds,
    observed,
    grid=None,
    replicates: int = 200,
    seed: int = 1,
)
```

What population skill a level actually read is consistent with.

`tss_inflation` maps a population skill to the level a design reports
for it; this inverts that map. It is the only honest way to read a level
as a statement about a population rather than about a scoring rule, and
it says nothing about a difference between two arms, where the inflation
cancels and the reported number stands as it is.

## `occlusion()`

``` python
occlusion(x, *args, **kwargs)
```

What one candidate’s score loses when a bin, or a channel, is withheld
from it.

Takes a `timesift.fit.timesift` run or a `timesift.ladder.grain_ladder`
result, and the profile itself is one implementation either way. The
models kept per fold are the ones read, so the profile is measured where
the score was: on the units each model held out.
