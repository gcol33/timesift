# Choose the grain inside the training data, and score the whole procedure

[`grain_ladder()`](https://gillescolling.com/timesift/reference/grain_ladder.md)
fits every candidate against one fold map and reports the grid, so
reading the best grain off it and quoting that grain's score quotes a
number the held-out units helped choose. This does the choosing inside
the training data instead. Within each outer fold the training units are
split again, every candidate is fitted on part of them and scored on the
rest, the best is refitted on the whole outer training set, and the
outer test fold is predicted once. The estimate that comes back is
therefore of the procedure including its choice of grain, which is what
an ecologist applying it to a new site would run.

## Usage

``` r
select_grain(
  x,
  y,
  learners,
  folds = NULL,
  inner = 5L,
  rule = c("argmax", "coarsest_adequate"),
  threshold = NULL,
  interval = c("variables", "nested_cv"),
  repeats = 1L,
  response = "presence_absence",
  metric = NULL,
  compare = NULL,
  control = train_control(),
  seed = 1L,
  verbose = TRUE
)

# S3 method for class 'timesift_selection'
summary(object, ...)
```

## Arguments

- x:

  A
  [`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md)
  result, a
  [`timesift_set()`](https://gillescolling.com/timesift/reference/timesift_set.md),
  or a named list of representations. Its names are the grains being
  chosen between.

- y:

  The response for the same units.

- learners:

  A learner, a list of them, or names of registered ones, as
  [`grain_ladder()`](https://gillescolling.com/timesift/reference/grain_ladder.md)
  takes. Named alongside the grains they form the candidate set.

- folds:

  The outer fold map, from
  [`fold_map()`](https://gillescolling.com/timesift/reference/fold_map.md)
  or any named integer vector. Built with the defaults of
  [`fold_map()`](https://gillescolling.com/timesift/reference/fold_map.md)
  when not given.

- inner:

  Number of inner folds the selection is made on, or a function of the
  outer training response returning a fold map for those units. A count
  deals the inner folds by the grouping the outer fold map carries, so
  what
  [`grouped_cv()`](https://gillescolling.com/timesift/reference/cv.md)
  kept whole outside stays whole inside.

- rule:

  How a candidate is chosen from its inner scores. `"argmax"` takes the
  highest. `"coarsest_adequate"` takes the coarsest candidate whose
  inner score lies within one standard error of the highest, the
  one-standard-error rule of Breiman, Friedman, Olshen and Stone (1984)
  and of Hastie, Tibshirani and Friedman (2009, section 7.10) with
  coarseness in place of model complexity. See Choosing a candidate.

- threshold:

  `NULL`, or the rule of
  [`decision_threshold()`](https://gillescolling.com/timesift/reference/kappa_score.md)
  a presence-absence cut is learned by: `"youden"`, the cut that
  maximises TSS, `"kappa"` or `"prevalence"`. See A cut learned inside
  the training data.

- interval:

  Which interval to report beside the across-variable one, which is
  always reported: `"variables"` for that one alone, or `"nested_cv"`
  for an interval for the procedure's risk. See What the interval is
  for.

- repeats:

  Repetitions of the nested cross-validation, each on its own fold map.
  The first is the map the estimate was computed on.

- response:

  Name of the registered response head.

- metric:

  Name of a registered metric the selection is made on, or `NULL` for
  the response's own. The estimate is reported under every registered
  metric whichever this is.

- compare:

  A
  [`grain_ladder()`](https://gillescolling.com/timesift/reference/grain_ladder.md)
  result on the same units, response and outer fold map, whose arms the
  selected procedure is contrasted against cell by cell. `NULL` for no
  contrast.

- control:

  [`train_control()`](https://gillescolling.com/timesift/reference/train_control.md),
  the training settings every neural learner reads, in the inner search
  and in the refit alike. A learner carrying a control of its own
  overrides it on the settings that control names.

- seed:

  Seed for the inner splits. Each outer fold splits under `seed` plus
  its own number, so no two outer folds inherit the same inner
  partition.

- verbose:

  Report each outer fold and what it selected as it runs.

- object:

  A selection.

- ...:

  Ignored.

## Value

A `timesift_selection`: a list carrying `selected`, one row per outer
fold with the candidate it chose, the inner score it chose on, the
highest inner score in that fold (`inner_best`) and that score's
standard error (`inner_se`); `estimate`, the nested score under every
registered metric with its standard error across variables; `contrast`,
one
[`paired_contrast()`](https://gillescolling.com/timesift/reference/paired_contrast.md)
row against each arm of `compare`, or `NULL`; `candidates`, the set that
was searched; and `scores`, the per-cell rows of the selected procedure
under the selection metric, in the layout
[`grain_ladder()`](https://gillescolling.com/timesift/reference/grain_ladder.md)
returns. The held-out prediction of every unit is in the `predictions`
attribute and the scorable-cell mask in `cells`. `inner` holds every
candidate's inner score and standard error in every outer fold. With
`threshold` set, the estimate carries the score, its interval and the
interval's name in `interval`, one row per metric and interval. With
`interval = "nested_cv"` it also carries `nested_cv`, the same rows with
the estimator's own quantities beside them, and `final`, the procedure
fitted on every unit, whose risk that interval is for. With `threshold`
set, the estimate carries one further row, `tss_inner_cut`, the
procedure's TSS at the learned cuts; `thresholds` holds the cut of every
outer fold and variable; and `cut_scores` the per-cell rows it is
averaged from, in the layout of `scores`. Both are `NULL` otherwise.

## Details

A candidate is a `(grain, learner)` pair: the grains are the elements of
the representation set, which is where a grain and the statistic its
grains are summarised by are both named, and the learners are the ones
passed. Both are registry entries or objects built by
[`learner()`](https://gillescolling.com/timesift/reference/learner.md),
so a new grain, a new grain summary or a new candidate model widens the
search with no change here.

What the estimate is of: the expected held-out score of the whole
pipeline, selection included, on units drawn as these were. What it is
not: the score of the winning grain. That is higher, by the amount
selection buys itself, and the difference between the two is the
quantity this function exists to keep out of a reported number. It also
does not say the selected grain is the one a mechanism acts at; it says
that grain predicted best on the units the selector saw.

The cost is the ladder's, multiplied by the number of inner folds:
`v_outer * (v_inner * candidates + 1)` fits. With a neural learner that
is where an overnight run goes.

## Choosing a candidate

Inside each outer fold every candidate carries an inner score, the mean
over variables of its per-variable mean over the inner folds, and a
standard error, the standard deviation over the inner folds of the
fold's own score (the mean over the variables scored in that fold)
divided by the square root of the number of inner folds.
`rule = "argmax"` takes the highest inner score, and on an exact tie the
candidate declared first.

`rule = "coarsest_adequate"` first finds that highest score and its
standard error, calls every candidate scoring at least the highest minus
one standard error adequate, and takes the coarsest adequate one.
Coarseness is read off the representation as the package holds it: fewer
bins is coarser, and between two candidates with the same number of
bins, fewer channels is coarser. A tie on both goes to the higher inner
score, then to the candidate declared first. Where one candidate scores
more than a standard error above every other, the two rules agree; where
the inner profile is flat, this one returns the least storage the record
can be kept at without a measured loss inside the training data. A
standard error that cannot be computed, because a candidate was scored
in fewer than two inner folds, is taken as zero, so the rule falls back
to the candidates tied with the highest score.

## What the interval is for

The across-variable interval, the one every level of the package
reports, is the estimate plus or minus a Student's t quantile times the
standard error across the response variables. Its spread is the spread
of true skill between variables, and it cannot see the error every
variable shares, since all of them are fitted and scored on the same
units and the same folds. It is an interval over the variables of this
dataset, and not an interval for what the procedure would score on a new
sample.

`interval = "nested_cv"` adds one that is, by the nested
cross-validation of Bates, Hastie and Tibshirani (2024). Inside every
repetition, each outer training set is cross-validated again over the
remaining folds of the same map, which gives the mean squared error of a
cross-validation estimate as the difference of two terms it can
estimate: the squared gap between the inner estimate and the held-out
fold's score, less the variance of that fold's score. The paper's error
is a mean of per-unit losses; here a fold's score is the mean over the
variables scorable in it, the inner estimate is averaged as the reported
estimate is, and the variance of a fold's score is its delete-one
jackknife variance over the units of the fold, which for a mean of
per-unit losses is exactly the paper's `var(e) / |I_k|`. The square root
of the estimated mean squared error is held between the jackknife
standard error of the estimate and the square root of the fold count
times it, as the paper's section 4.3.2 has it, and the centre carries
its bias correction, so the interval is for the risk of the procedure
fitted on a sample of this size, which is the fit `final` holds.

The cost is the selection's, multiplied: one repetition fits the
procedure once for every unordered pair of outer folds,
`v_outer * (v_outer - 1) / 2` fits, and every repetition after the first
refits the outer folds as well. More repetitions steady the estimate of
the mean squared error; the paper uses two hundred random splits, which
is affordable where a fit is cheap and is not where a fit is a neural
network.

## A cut learned inside the training data

TSS read at the cut that maximises it on the scored units is biased
upward, most where presences are few
([`tss_inflation()`](https://gillescolling.com/timesift/reference/tss_inflation.md)).
With `threshold` set, each outer fold learns one cut per variable on the
inner out-of-fold predictions of the candidate it selected, which the
inner search has already made for every outer training unit, by
[`decision_threshold()`](https://gillescolling.com/timesift/reference/kappa_score.md)
under the rule named. The cut is then frozen and the outer test fold's
predictions are read at it by
[`tss()`](https://gillescolling.com/timesift/reference/tss.md). No unit
of an outer test fold enters the cut its own fold is read at. The inner
out-of-fold predictions come from models fitted on part of the outer
training set and the held-out predictions from the refit on all of it,
so the cut is learned on predictions of the same candidate from slightly
smaller training sets.

## References

Bates, S., Hastie, T. and Tibshirani, R. (2024). Cross-validation: what
does it estimate and how well does it do it? *Journal of the American
Statistical Association* **119**(546), 1434-1445.
[doi:10.1080/01621459.2023.2197686](https://doi.org/10.1080/01621459.2023.2197686)

## See also

[`grain_ladder()`](https://gillescolling.com/timesift/reference/grain_ladder.md)
for the grid this selects from, and
[`paired_contrast()`](https://gillescolling.com/timesift/reference/paired_contrast.md)
for the comparison the `contrast` element holds.

## Examples

``` r
set.seed(1)
t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24 * 200)
units <- sprintf("p%02d", 1:60)
warmth <- rnorm(60)
d <- data.frame(
  plot = rep(units, each = length(t)), t = rep(t, length(units)),
  temp = as.numeric(vapply(warmth, function(w) w + sin(seq_along(t) / 300) + rnorm(length(t)),
                           numeric(length(t)))))
y <- matrix(rbinom(120, 1, plogis(c(warmth, -warmth))), nrow = 60,
            dimnames = list(units, c("sp1", "sp2")))
x <- grain_matrix(d, plot, t, temp, grain = c("week", "month"))
sel <- select_grain(x, y, elasticnet(), folds = fold_map(y, v = 3), inner = 3,
                    verbose = FALSE)
sel
sel$estimate
```
