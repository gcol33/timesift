# Stepwise selection by AIC on the flattened representation

One generalised linear model per variable, its terms chosen by Akaike's
criterion over every bin-by-channel column. The family is the response
head's: logistic under a binary cross-entropy loss, Gaussian under a
squared-error one. So are the case weights, so a rare response weighs
here what it weighs in every other learner.

## Usage

``` r
stepwise(
  data = NULL,
  max_terms = 3L,
  degree = 2L,
  direction = c("forward", "both", "backward", "none"),
  terms = c("column", "power"),
  threads = 1L
)
```

## Arguments

- data:

  A representation the learner is pinned to, or `NULL` to run across
  every representation of the run.

- max_terms:

  Terms a forward or two-way search holds at most; `Inf` for no bound.

- degree:

  Polynomial degree each column enters at.

- direction:

  `"forward"`, `"both"`, `"backward"` or `"none"`.

- terms:

  `"column"` for a column's polynomial as one term, `"power"` for each
  power its own.

- threads:

  How many of one step's candidate fits run at once. What comes back
  does not depend on it.

## Value

A
[`learner()`](https://gillescolling.com/timesift/reference/learner.md).

## Details

`terms` says what one term is. Under `"column"` it is a column's
orthogonal polynomial of degree `degree`, so a column enters with its
curvature at once and can be non-monotone in the reading the way a niche
optimum is. Under `"power"` each power of a column, `x`, `x^2` and so
on, is a term of its own, which is how biomod2 writes a quadratic
formula and how
[`MASS::stepAIC()`](https://rdrr.io/pkg/MASS/man/stepAIC.html) walks it.
A column holding one value over the fitting units is not a term.

`direction` is the search. `"forward"` starts from the intercept and
admits the term that lowers the criterion most, while one does and the
model holds fewer than `max_terms`. `"both"` does the same but also
weighs dropping each term it holds at every step, and `"backward"`
starts from every term and drops alone. `"none"` fits every term and
selects nothing: with `terms = "power"` and `degree = 2` that is the
model biomod2's GLM fits, `y ~ x + I(x^2)` over every column. The
two-way and backward searches are MASS's `stepAIC()`, step for step: the
model as it stands wins a tie, a term whose removal leaves the rank
unchanged is dropped first, and an addition that does not raise the rank
is not offered. `max_terms` bounds what a forward or two-way search
adds, and the backward and unselected fits start from every term
whatever it is.

Each fit is R's `glm.fit`: iteratively reweighted least squares, the
rank read off the same pivoted decomposition, and the same stopping
rule. A move whose fit does not settle within its 25 iterations is
refused rather than taken, and the fit names every response whose final
model did not settle in `stopped`.

A model with nothing but the intercept predicts the response's share
among the fitting units. Selection happens inside whichever units the
learner is handed, so under
[`grain_ladder()`](https://gillescolling.com/timesift/reference/grain_ladder.md)
it is redone in every fold. Reported beside a penalised fit it also
prices discrete selection: choosing a handful of columns out of hundreds
is high variance, and that variance is a cost of the selector rather
than of the features. An unselected fit over hundreds of columns
separates any response it is given and is meant for a coarse grain,
`data = grain("season")`.

The search and the fits run on the core the Python package calls, so the
two select the same terms and return the same coefficients.

## Examples

``` r
stepwise(max_terms = 3)
stepwise(direction = "both", terms = "power", max_terms = Inf)
stepwise(data = grain("season"), direction = "none", terms = "power")
```
