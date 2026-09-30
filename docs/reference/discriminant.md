# Flexible discriminant analysis on the flattened representation

One discriminant per response, over every bin-by-channel column of the
representation, fitted as mda's `fda(method = mars)` fits it and as
biomod2 fits `FDA`. Optimal scoring gives presence and absence one score
each and regresses the scored response on a MARS basis of the columns;
the fitted score is the one canonical variate, and a unit's prediction
is its posterior probability of presence under two normal classes around
the class centroids on that variate, with the classes' shares among the
fitting units as priors.

## Usage

``` r
discriminant(
  data = NULL,
  degree = 1L,
  penalty = NULL,
  nk = NULL,
  thresh = 0.001,
  prune = TRUE,
  calibrate = TRUE,
  threads = 1L
)
```

## Arguments

- data:

  A representation the learner is pinned to, or `NULL` to run across
  every representation of the run.

- degree:

  The most hinges a term of the basis multiplies.

- penalty:

  The generalised cross-validation's charge per term beyond its own
  degree of freedom, taken at half. `NULL` is mda's, 2 at degree one and
  3 above.

- nk:

  The most terms the forward pass reaches, the intercept included.
  `NULL` is `max(21, 2 p + 1)` for `p` columns; an even number is taken
  one lower.

- thresh:

  The least share of the residuals a forward step is kept for.

- prune:

  Whether the pruning pass runs. Without it every term of the forward
  pass is kept.

- calibrate:

  Whether the posterior is recalibrated by a probit regression, as
  biomod2 does.

- threads:

  Columns searched at once. The model is the same on any number.

## Value

A
[`learner()`](https://gillescolling.com/timesift/reference/learner.md).

## Details

The basis is mda's own MARS, which biomod2 reaches through `fda()` and
which differs from earth's that
[`mars()`](https://gillescolling.com/timesift/reference/mars.md)
reproduces. Each forward step adds a column linearly or a pair of hinges
on it, `max(0, x - t)` and `max(0, t - x)`, choosing by Friedman's
running updates, and the pass stops when a step lowers the residuals by
less than `thresh` of them, when they fall to `thresh` of the null
model's, when the generalised cross-validation passes ten times the null
model's, or at `nk` terms. The pruning pass drops the term of least t
statistic, one at a time, and keeps the subset of least generalised
cross-validation, which counts each term beyond the intercept as
`1 + penalty / 2` degrees of freedom. The defaults are mda's, which
biomod2 passes unchanged under its default option set and under
`"bigboss"`: degree one, `penalty = 2`, `thresh = 0.001` and
`nk = max(21, 2 p + 1)` for `p` columns.

The case weights are the response head's,
[`positive_weights()`](https://gillescolling.com/timesift/reference/positive_weights.md)
under presence-absence. They set the classes' scores and the variate, as
`fda()` reads them, and not the basis: mda's forward pass sets them to
one.

biomod2 always recalibrates an `FDA` posterior by a probit regression of
the response on it, under the case weights, and predicts through that
regression. `calibrate = TRUE` does the same, on the units the
discriminant was fitted on; `FALSE` predicts the posterior itself.
biomod2 rounds the posterior to three decimals before recalibrating it,
which is not reproduced.

The forward pass, the pruning, the scoring and the recalibration live in
the core the Python package calls, pinned against mda and R's
[`glm()`](https://rdrr.io/r/stats/glm.html) in the fixtures, so the two
languages keep the same terms and predict the same probabilities. A
response holding one value is predicted its mean, and so is one whose
scored response the basis does not reach; both are named in `unfitted`.
The learner needs a presence-absence response, and a head whose loss is
not the binary cross-entropy is refused.

## Examples

``` r
discriminant()
discriminant(data = grain("season"), calibrate = FALSE)
```
