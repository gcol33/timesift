# Maxent on the flattened representation

One maximum-entropy model per response, over every bin-by-channel column
of the representation, in the formulation of the maxnet package
(Phillips et al. 2017) and biomod2's `MAXNET`: maxnet's feature classes,
its regularisation of each feature, and a lasso over them, fitted by the
penalised core
[`elasticnet()`](https://gillescolling.com/timesift/reference/elasticnet.md)
runs on, which the Python package calls too. With the maxnet package's
own settings the features and the penalty factors are maxnet's to
rounding, and the fit settles at the objective glmnet reaches for
maxnet.

## Usage

``` r
maxent(
  data = NULL,
  classes = NULL,
  regmult = 1,
  formulation = c("background", "absence"),
  type = NULL,
  knots = 50L,
  add_samples = TRUE,
  clamp = TRUE,
  n_inner = 5L,
  s = c("lambda.min", "lambda.1se"),
  tol = 1e-08,
  max_design = 2,
  threads = 1L,
  seed = 1L
)
```

## Arguments

- data:

  A representation the learner is pinned to, or `NULL` to run across
  every representation of the run.

- classes:

  Feature classes, letters of `"lqpht"`, or `NULL` for maxnet's choice
  at each response's presence count.

- regmult:

  Multiplier on every feature's regularisation.

- formulation:

  `"background"` for maxnet's presence-background model, `"absence"` for
  a logistic lasso reading the absences as absences.

- type:

  The background formulation's output: `"cloglog"` or `"logistic"`. The
  absence formulation predicts a probability, which is its `"logistic"`.

- knots:

  Points over each column's range the hinges and thresholds are placed
  at.

- add_samples:

  Add each presence to the background, as maxnet's
  `addsamplestobackground`.

- clamp:

  Hold each column and each feature inside the range it was fitted on.

- n_inner:

  Folds of the absence formulation's inner cross-validation.

- s:

  Where the absence formulation reads its path: `"lambda.min"` or
  `"lambda.1se"`.

- tol:

  Where the coordinate descent stops, as the elastic net's `tol`.

- max_design:

  Gigabytes the expanded design may take.

- threads:

  How many fits of the absence formulation's inner cross-validation run
  at once. What comes back does not depend on it.

- seed:

  Seed for the inner cross-validation's fold draw, which is random and
  would otherwise make the fit irreproducible.

## Value

A
[`learner()`](https://gillescolling.com/timesift/reference/learner.md).

## Details

The feature classes are the letters of `classes`: `l` the column itself,
`q` its square, `p` the product of each pair of columns, `h` forward and
reverse hinges at the interior of `knots` equally spaced points of each
column's range, and `t` thresholds at 49 interior points of it. Left
`NULL`, they follow the response's presence count as `maxnet.formula()`
has them: `"l"` under 10 presences, `"lq"` under 15, `"lqh"` under 80,
and `"lqph"` from 80 on. A column holding one value over the units
fitted takes no feature.

`formulation` says what the absences are. `"background"` is maxnet's own
and what biomod2 fits as `MAXNET`: every unit is background, each
presence joins the background again unless an absence carries the same
readings (`add_samples`), the background is weighted 100 against a
presence's 1, and the model is read at the last of maxnet's 200
penalties, which scale with `regmult`. Its output is maxnet's `type`,
`"cloglog"` by default as biomod2 predicts it. `"absence"` reads the
absences as absences: a logistic lasso over the same features and
penalty factors under the response head's case weights,
[`positive_weights()`](https://gillescolling.com/timesift/reference/positive_weights.md)
under presence-absence, with the penalty chosen by an inner
cross-validation dealt as the elastic net's is, and a probability as
output.

The background formulation takes no case weights, as maxnet takes none
and biomod2 passes none: the background weight is what sets a presence's
weight there. Either formulation holds each column inside the range it
was fitted on, and each feature inside its own, before predicting, as
maxnet's `predict(clamp = TRUE)` does; `clamp = FALSE` reads them as
they are.

A hinge per column per knot makes the design large: a weekly
three-channel representation, 471 columns, is 47,100 features under
`"lqh"`, and its products under `"lqph"` 110,685 more. The design is
held in memory with a centred copy beside it, and a fit whose design
would take more than `max_design` gigabytes is refused with the size it
would have taken. A coarser representation (`data = grain("month")`) or
fewer classes is then the way to fit it.

A response with fewer than two presences, or one whose inner training
sets cannot each hold two of each outcome under the absence formulation,
is predicted its share among the fitting units, and the fit names it in
`unfitted`. A path that does not settle at a penalty ends there, as the
elastic net's does, and is read at its last settled point; the fit names
every such response in `stopped`.

The learner needs a presence-absence response: maxnet has no model for a
continuous one, and a head whose loss is not the binary cross-entropy is
refused.

## Examples

``` r
maxent()
maxent(classes = "lqh", regmult = 2)
maxent(formulation = "absence")
```
