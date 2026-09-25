# Python: scoring and comparison

The metrics, the paired contrast between two arms on matched cells,
every grain against a learner’s best, the inflation of a score read at
its own best threshold, what a fitted model read, and a record with a
planted grain to test all of it on.

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

## `average_precision()`

``` python
average_precision(y, p)
```

The area under the precision-recall curve, as the step sum over the
distinct predictions.

At each distinct prediction, the precision of calling every unit at or
above it a presence, weighted by the share of presences that cut adds.
Units sharing a prediction enter together. Its floor is the prevalence
rather than one half, which makes it the reading of how well presences
are ranked above absences where presences are rare.

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
score_predictions(y, p, folds, cells=None, metric: str = 'roc_auc')
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
a difference between the arms. Pairing removes the variation between
variables and the part of a threshold-selected metric’s bias that the
design sets, but not the part that belongs to each arm: how far TSS read
at its best cut is inflated depends on how an arm’s predictions are
distributed as well as on the presence counts, so two arms of equal
skill on the same cell can carry different biases. A TSS contrast is
best read beside the same contrast under `roc_auc`, which has no cut to
choose.

Each arm is named whole, `grain|learner`. A learner named alone would
take its best grain, chosen on the scores the contrast is then read off,
and pairing does not remove that choice; `select_grain` chooses a grain
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

## `grain_contrasts()`

``` python
grain_contrasts(
    ladder: Ladder,
    learner: str | None = None,
    reference: str | None = None,
    adjust: str = 'mvt',
)
```

Compare every grain against the reference grain of one learner.

A mixed model on the learner’s per-cell scores,
`score ~ grain + (1 | variable) + (1 | fold)`, fitted by restricted
maximum likelihood, and every grain compared against `reference` by
Dunnett’s many-to-one procedure. The design is balanced across
variables, folds and grains, so each grain is compared within a variable
and within a fold, and the variation between variables cancels from the
comparison.

`learner` is the only one in the ladder by default, and `reference` the
learner’s best grain. Taking the best-observed grain as the reference
favours it, so read this beside `paired_contrast`, which singles out no
grain. `adjust` is `"mvt"`, the adjustment R’s default reads, or
`"none"`.

Returns one row per grain other than the reference: `learner`, `grain`,
`reference`, the difference from the reference `diff`, its interval
`lower` and `upper`, and the adjusted `p_value`. Needs scipy.

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
inflation. It is an expectation: a level is optimistic on average, not a
bound every reading sits above. The model planted is one distribution of
predictions, and another at the same skill inflates by a different
amount, which is also why a paired difference is not free of it.

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
for it; this inverts that map, under the distribution of predictions
`tss_inflation` plants; a model whose predictions are distributed
otherwise is inflated by a different amount. It does not correct a
difference between two arms, whose inflations need not be equal.

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

## `simulate_records()`

``` python
simulate_records(
    n: int = 300,
    mechanism: str = 'none',
    variables: int = 10,
    prevalence: float = 0.1,
    auc: float = 0.75,
    from_: str = '2021-09-01',
    days: int = 365,
    step_hours: float = 3,
    seasonal: float = 8,
    offset_sd: float = 1,
    anomaly_sd: float = 1,
    anomaly_days: float = 2,
    offset_effect: float = 0,
    sensor_sd: float = 0.3,
    year_start: str = '09-01',
    seed: int = 1,
    draw: int = 1,
)
```

Draw units carrying a record and a presence-absence response acting at
one known grain.

The response is driven by `g_ij = sum_t w_j(t) a_i(t)`, a weighted mean
of unit `i`’s latent anomaly: the record with the shared seasonal cycle
and the unit’s own offset taken out. The weights are constant within the
bins of one grain and zero outside a short stretch of them, so the true
grain is the coarsest grain at which `g` is still an exact linear
functional of the representation. `"none"` draws the driver
independently of the record; `"event"` reads three consecutive days,
`"season"` one whole season, and `"lag"` four consecutive weeks under a
geometric decay.

The driver is standardised by its population mean and standard
deviation, computed in closed form from the settings, and the response
is `Bernoulli(expit(b0 + b1 z))` with `b0` and `b1` solved so the
marginal prevalence is `prevalence` and the population area under the
ROC curve of `z` is `auc`. `auc` is a ceiling no fitted model reaches.

`seed` fixes the design and `draw` the units, so two calls with one
`seed` and two `draw` values are two samples of one population. `from_`
is R’s `from`, renamed because `from` is a Python keyword.

## `Simulation`

``` python
Simulation(readings, y, driver, grain, weights, link, design, grain_stat, anchor)
```

A simulated record, its response, and everything the draw is
reproducible from.

`readings` is the long table `grain_matrix` takes, as a mapping of
`unit`, `time` and `reading`. `y` is the `[unit, variable]` 0/1 response
and `driver` the standardised driver `z` behind it. `grain` is the true
grain, or `None` where the response does not read the record. `weights`
is the `[reading, variable]` matrix defining the driver, `link` the
solved `b0` and `b1`, and `design` the settings of the draw.

Attributes:

- `readings` - dict
- `y` - Response
- `driver` - np.ndarray
- `grain` - str \| None
- `weights` - np.ndarray
- `link` - dict
- `design` - dict
- `grain_stat` - str
- `anchor` - np.ndarray
