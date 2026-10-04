# Python: learners

The arms that ship, how they are trained, and the interface a learner of
your own goes through.

[All of the Python
reference](https://gillescolling.com/timesift/articles/python-reference.md)

## `elasticnet()`

``` python
elasticnet(
    data=None,
    alpha=0.5,
    n_inner=10,
    squares=True,
    s='lambda.1se',
    n_lambda=100,
    tol=1e-08,
    threads=1,
    seed=1,
)
```

One penalised regression per variable, over every bin-by-channel column
and, by default, their squares, with the penalty chosen by an inner
cross-validation on the fitting units.

There is no discrete selection step: the penalty path uses every column
and shrinks, and nothing about the model is decided outside the fold it
is fitted in. The family is the response head’s: logistic under a binary
cross-entropy loss, log-linear under a Poisson-deviance one, linear
under a squared-error one, and so are the case weights,
`timesift.response.positive_weights` under presence-absence, which every
learner that ships fits under.

The path is fitted by the same core the R package calls, so the two
return the same coefficients for the same input. Its conventions are
glmnet’s, which is what the arm is measured against: weights normalised
to sum to one, columns centred and scaled by their weighted mean and
weighted standard deviation, a hundred penalties down from the smallest
that leaves every coefficient at zero, and the held-out deviance read
fold by fold. `s` is where the fit is read: `"lambda.min"`,
`"lambda.1se"`, or a penalty of its own, which is interpolated between
the two points of the path around it. The defaults are cv.glmnet’s: ten
inner folds, read at `"lambda.1se"`, the largest penalty within one
standard error of the least held-out deviance. `tol` is where the
coordinate descent stops, read off the largest coefficient move of a
pass.

The inner folds are dealt for each response and stratified on it, so a
rare outcome is spread over them as evenly as its count allows. A
presence-absence response whose inner training sets cannot each hold two
of each outcome, the fewest a logistic path is fitted to, has too few of
one outcome to choose a penalty on. It is predicted its share among the
fitting units, as a response holding one outcome is, and the fit names
every such response in `unfitted`.

A reweighted fit that does not settle at a penalty ends its path there
and keeps the points before it, which is what glmnet does with the same
event, and the penalty is chosen over the points fitted. It happens
where a rare outcome is nearly separable at the small end of the path.
The fit names every response whose path on the fitting units, or on any
inner fold, ended that way in `stopped`.

`threads` is how many fits of one response’s inner cross-validation run
at once. The path on every fitting unit and the path of each inner fold
are one independent fit each, so they parallelise without sharing
anything, and `n_inner + 1` threads is as many as a response can use.
The default is serial, because a package does not take a machine’s cores
without being asked. What comes back does not depend on it.

## `linear()`

``` python
linear(data=None, select='both', terms='power', max_terms=math.inf, degree=2, threads=1)
```

One generalised linear model per variable over every bin-by-channel
column, its terms chosen by Akaike’s criterion. The family is the
response head’s: logistic under a binary cross-entropy loss, Gaussian
under a squared-error one and Poisson under a Poisson-deviance one, and
so are the case weights.

The defaults are biomod2’s `GLM`: every column enters as `x + I(x^2)`,
and the terms are searched in both directions by AIC as
[`MASS::stepAIC()`](https://rdrr.io/pkg/MASS/man/stepAIC.html) searches
them, with no bound on how many are kept. Under the shipped
presence-absence head those weights are on, so a default
[`linear()`](https://gillescolling.com/timesift/reference/linear.md) is
biomod2’s `GLM` specification fitted under them; a head registered
without `weights` fits it unweighted.

`terms` says what one term is. Under `"power"` each power of a column is
a term of its own, which is how biomod2 writes a quadratic formula and
how `stepAIC()` walks it. Under `"column"` a term is a column’s
orthogonal polynomial of degree `degree`, so a column enters with its
curvature at once and can be non-monotone in the reading the way a niche
optimum is. A column holding one value over the fitting units is not a
term.

`select` is the search. `"both"` starts from the intercept and at every
step takes the move that lowers the criterion most, adding a term or
dropping one it holds, while one does and the model holds fewer than
`max_terms`. `"forward"` only adds, and `"backward"` starts from every
term and only drops. `"none"` fits every term and selects nothing. The
two-way and backward searches are `stepAIC()`, step for step.
`max_terms` bounds what a forward or two-way search adds; `math.inf` for
no bound.

Each fit is R’s `glm.fit`: iteratively reweighted least squares, the
rank read off the same pivoted decomposition, and the same stopping
rule. A move whose fit does not settle within its 25 iterations is
refused rather than taken, and the fit names every response whose final
model did not settle in `stopped`. A model with nothing but the
intercept predicts the response’s share among the fitting units. The
search runs on the core the R package calls, so the two select the same
terms and return the same coefficients; `threads` runs one step’s
candidate fits at once and does not change what comes back.

## `forest()`

``` python
forest(
    data=None,
    trees=None,
    mtry=None,
    min_node=None,
    balance=False,
    preset='default',
    seed=1,
    threads=1,
)
```

One random forest per variable, over every bin-by-channel column: a
probability forest under a presence-absence head and a regression forest
under a head with a squared-error loss or a count head, a count being
cut on its variance and a leaf reporting its mean. Trees split on one
column at a time and pay nothing for columns that carry nothing, so a
forest reads a wide tabular representation without a penalty path and
without a selection step.

Each tree is grown on a bootstrap draw of the units, as many draws as
there are units, and each node is split on the best of `mtry` columns
drawn for it, by the Gini index or the sum of squares `tree` splits by.
A tree is grown out: a node is split while it holds at least twice
`min_node` units and its responses differ. A leaf reports the share of
presences among the draws it holds, or their mean, and the forest the
mean over its trees. The forest is grown by the core the R package
calls, and every draw comes from one generator seeded per tree, so the
two languages grow the same forest on any number of threads.

`balance=True` is the down-sampled forest biomod2 fits as `RFd`: each
tree draws as many units from each class as the smaller class holds.

`preset` says whose defaults the settings left `None` take. `"default"`
is randomForest’s own, which is what biomod2’s default option set fits:
500 trees, `mtry` the square root of the column count under
presence-absence and a third of it under a squared-error loss, and
`min_node` 1 and 5 under the two. `"bigboss"` is biomod2’s tuned option
set: 500 trees, `mtry=2` and `min_node=5`. A setting given explicitly
beats either, and an `mtry` above the column count is the column count.

The case weights are the response head’s,
`timesift.response.positive_weights` under presence-absence, and weight
the bootstrap draw: a unit is drawn in proportion to its weight, and
within its class under `balance`. Under the shipped presence-absence
head those weights are on, so a default
[`forest()`](https://gillescolling.com/timesift/reference/forest.md) is
randomForest’s specification fitted under them; a head registered
without `weights` fits it unweighted.

## `tree()`

``` python
tree(
    data=None,
    min_split=None,
    min_leaf=None,
    cp=None,
    max_depth=None,
    prune='se_sum',
    n_inner=None,
    preset='default',
    shrink=1.0,
    seed=1,
)
```

One classification or regression tree per variable, over every
bin-by-channel column, grown under rpart’s rules: the Gini index under a
presence-absence head, the sum of squares under a head with a
squared-error loss and the Poisson deviance under a count head, a split
only between two distinct values of a column, and the cost-complexity
bookkeeping that keeps a split only where it lowers the risk by at least
`cp` of the root’s. On the same columns, weights and folds the tree is
the one rpart grows, split for split, and its complexity table the one
rpart reports; the tree is grown by the core the R package calls, so the
two languages grow it identically.

The grown tree is pruned back by an inner cross-validation. Its folds
are dealt for each response and stratified on it, as the elastic net’s
are, and `prune` names the rule that reads the complexity table:
`"se_sum"` takes the row of least cross-validated error plus its
standard error among the rows that keep a split, the last of them where
several tie, which is how biomod2 prunes its classification tree;
`"one_se"` takes the smallest tree within one standard error of the
least cross-validated error; `"min"` the first row reaching the least
error; and `"none"` keeps the tree as grown.

`preset` says whose defaults the settings left `None` take. `"default"`
is rpart’s own, which is what biomod2’s default option set fits:
`min_split=20`, `min_leaf=round(min_split / 3)` (or
`min_split=3 * min_leaf` where only `min_leaf` is given), `cp=0.01`,
`max_depth=30` and ten inner folds. `"bigboss"` is biomod2’s tuned
option set: `min_split=5`, `min_leaf=5`, `cp=0.001`, `max_depth=10` and
five inner folds. A setting given explicitly beats either.

The case weights are the response head’s,
`timesift.response.positive_weights` under presence-absence. They weigh
every class count, sum of squares and event count the tree is grown on;
`min_split` and `min_leaf` count observations, as rpart’s do. Under the
shipped presence-absence head those weights are on, so a default
[`tree()`](https://gillescolling.com/timesift/reference/tree.md) is
rpart’s specification fitted under them; a head registered without
`weights` fits it unweighted.

Under a count head a leaf predicts a rate, and the rate is shrunk
towards the rate of the units the tree is grown on, as rpart’s
`method = "poisson"` shrinks it: the posterior mean of a gamma prior
whose coefficient of variation is `shrink`, `0` for none and rpart’s
default `1`. A split is chosen on the deviance of the unshrunk rates,
and a subtree’s risk, its complexity and the pruning’s cross-validated
error are read on the shrunk ones.

## `boosting()`

``` python
boosting(
    data=None,
    method='gbm',
    trees=None,
    depth=None,
    shrinkage=None,
    min_leaf=None,
    subsample=None,
    colsample=None,
    lambda_=None,
    gamma=None,
    n_inner=None,
    preset='default',
    seed=1,
    threads=1,
)
```

One boosted model per variable, over every bin-by-channel column: a
logistic model under a presence-absence head, a squared-error one under
a head with a squared-error loss and a Poisson one with a log link under
a count head. The score starts at the log-odds of the weighted share of
presences, or the weighted mean, and each tree is fitted to the loss’s
gradient at the current score and added to it scaled by `shrinkage`.
Each tree is grown on a subsample of the units drawn without
replacement, and reads a subsample of the columns.

`method` picks the trees. Under `"gbm"` they are Friedman’s gradient
boosting machine as the gbm package grows it, which is what biomod2 fits
as `GBM`: `depth` splits grown best first, each the one that most
reduces the weighted squared error of the working response with at least
`min_leaf` units on each side, and a leaf that takes one Newton step on
the loss. Under `"xgboost"` they are XGBoost’s exact greedy trees (Chen
and Guestrin 2016), which biomod2 fits as `XGBOOST`: grown level by
level to `depth`, each split chosen by the second-order gain under the
L2 penalty `lambda_` with at least `min_leaf` of hessian on each side,
pruned where a split gains less than `gamma`, and a leaf the step
`-G / (H + lambda)`. Either way `depth` is the order of interaction a
tree can hold. With `subsample=1` the first-order fit is gbm’s own to
rounding, and the second-order one xgboost’s to its single-precision
storage; the model is grown by the core the R package calls, so the two
languages fit the same model.

`n_inner` folds, when above zero, choose how many trees are kept: the
fit is repeated on each fold’s complement, and the number of trees of
least held-out deviance, summed over the folds and weighted by how many
units each holds, is kept, as gbm’s `cv.folds` chooses it. The folds are
dealt for each response and stratified on it, as the elastic net’s are.

`preset` says whose defaults the settings left `None` take. `"default"`
is the fitting package’s own, which is what biomod2’s default option set
fits: under gbm 100 trees of one split, `shrinkage=0.1`, `min_leaf=10`
and `subsample=0.5`; under xgboost 100 trees of depth 6,
`shrinkage=0.3`, `min_leaf=1`, `lambda_=1` and every unit and column.
`"bigboss"` is biomod2’s tuned option set: under gbm 2500 trees of seven
splits, `shrinkage=0.001`, `min_leaf=5`, `subsample=0.5` and three inner
folds; under xgboost four trees of depth 2 at `shrinkage=1`. A setting
given explicitly beats either.

`lambda_` is R’s `lambda`, spelled apart from Python’s keyword. The case
weights are the response head’s, `timesift.response.positive_weights`
under presence-absence, and weigh the gradient and every sum a tree is
grown on; `min_leaf` counts units under gbm, as `n.minobsinnode` does.
Under the shipped presence-absence head those weights are on, so a
default
[`boosting()`](https://gillescolling.com/timesift/reference/boosting.md)
is gbm’s specification fitted under them; a head registered without
`weights` fits it unweighted.

## `maxent()`

``` python
maxent(
    data=None,
    classes=None,
    regmult=1.0,
    formulation='background',
    type=None,
    knots=50,
    add_samples=True,
    clamp=True,
    n_inner=5,
    s='lambda.min',
    tol=1e-08,
    max_design=2.0,
    threads=1,
    seed=1,
)
```

One maximum-entropy model per variable, over every bin-by-channel
column, in the formulation of the maxnet package (Phillips et al. 2017)
and biomod2’s `MAXNET`: maxnet’s feature classes, its regularisation of
each feature, and a lasso over them, fitted by the penalised core
`elasticnet` runs on, which the R package calls too. With the maxnet
package’s own settings the features and the penalty factors are maxnet’s
to rounding, and the fit settles at the objective glmnet reaches for
maxnet.

The feature classes are the letters of `classes`: `l` the column itself,
`q` its square, `p` the product of each pair of columns, `h` forward and
reverse hinges at the interior of `knots` equally spaced points of each
column’s range, and `t` thresholds at 49 interior points of it. Left
`None`, they follow the response’s presence count as `maxnet.formula()`
has them: `"l"` under 10 presences, `"lq"` under 15, `"lqh"` under 80,
and `"lqph"` from 80 on. A column holding one value over the units
fitted takes no feature.

`formulation` says what the absences are. `"background"` is maxnet’s own
and what biomod2 fits as `MAXNET`: every unit is background, each
presence joins the background again unless an absence carries the same
readings (`add_samples`), the background is weighted 100 against a
presence’s 1, and the model is read at the last of maxnet’s 200
penalties, which scale with `regmult`. Its output is maxnet’s `type`,
`"cloglog"` by default as biomod2 predicts it. `"absence"` reads the
absences as absences: a logistic lasso over the same features and
penalty factors under the response head’s case weights,
`timesift.response.positive_weights` under presence-absence, with the
penalty chosen by an inner cross-validation dealt as the elastic net’s
is (`n_inner` folds, read at `s`), and a probability as output.

The background formulation takes no case weights, as maxnet takes none
and biomod2 passes none: the background weight is what sets a presence’s
weight there. Either formulation holds each column inside the range it
was fitted on, and each feature inside its own, before predicting, as
maxnet’s `predict(clamp=TRUE)` does; `clamp=False` reads them as they
are.

A hinge per column per knot makes the design large: a weekly
three-channel representation, 471 columns, is 47,100 features under
`"lqh"`, and its products under `"lqph"` 110,685 more. The design is
held in memory with a centred copy beside it, and a fit whose design
would take more than `max_design` gigabytes is refused with the size it
would have taken.

A response with fewer than two presences, or one whose inner training
sets cannot each hold two of each outcome under the absence formulation,
is predicted its share among the fitting units, and the fit names it in
`unfitted`. A path that does not settle at a penalty ends there and is
read at its last settled point; the fit names every such response in
`stopped`. The learner needs a presence-absence response, under a head
whose loss is the binary cross-entropy.

## `envelope()`

``` python
envelope(data=None, quantile=0.025)
```

biomod2’s surface range envelope, one per variable, over every
bin-by-channel column: for each column the `quantile` and `1 - quantile`
quantiles of its readings over the units present, and a unit predicted
present where every column lies between its two, the ends included. With
the same quantile it draws the envelope `bm_SRE()` draws; the quantile
is R’s default, type 7.

The prediction is zero or one, and enters an ensemble as that. The
envelope reads the presences and nothing else: neither the absences nor
the head’s case weights move it. Every column has to agree for a unit to
be inside, so the more columns a representation has the fewer units any
envelope holds, and a coarse grain, `data=grain("season")`, is what it
is meant for.

A response with no presence, or with nothing else, is predicted its
share among the fitting units and named in `unfitted`. The learner needs
a presence-absence response, under a head whose loss is the binary
cross-entropy.

## `mars()`

``` python
mars(
    data=None,
    degree=1,
    penalty=None,
    max_terms=None,
    min_gain=0.001,
    minspan=0,
    endspan=0,
    fast_k=20,
    fast_beta=1.0,
    prune=True,
    nprune=None,
    threads=1,
)
```

One MARS model per variable over every bin-by-channel column, fitted as
the earth package fits it and as biomod2 fits `MARS`. The forward pass
starts from the intercept and at each step multiplies a term already in
the model by a pair of hinges on one column, `max(0, x - t)` and
`max(0, t - x)`, taking the parent, the column and the knot `t` that
most reduce the residual sum of squares of a least-squares fit to the
response; a knot at a column’s least value enters the column linearly.
`degree` bounds how many hinges a term multiplies. The pass stops at
`max_terms` terms, when a step raises the R-squared by less than
`min_gain`, or when no term reduces the residuals.

The pruning pass removes terms one at a time, each time the one whose
loss raises the residuals least, and keeps the subset of least
generalised cross-validation, which charges `penalty` per knot. Under a
binary cross-entropy head the kept terms are refitted as a logistic
model, as earth’s `glm = list(family = binomial)` refits them, and the
prediction is its probability; under a count head they are refitted as a
Poisson model with a log link, as earth’s `glm = list(family = poisson)`
refits them, and the prediction is its mean count; under a squared-error
head they are refitted by least squares.

The defaults are earth’s, which biomod2 uses under its default and
`"bigboss"` option sets alike: degree one, `penalty` 2 (3 above degree
one), `min_gain=0.001`, `max_terms = min(200, max(20, 2 p)) + 1` for `p`
columns, Friedman’s spans between knots (`minspan=0`, `endspan=0`) and
Fast MARS over the `fast_k=20` best parents. `prune=False` keeps every
forward term; `nprune` caps the terms kept. The case weights are the
head’s and weigh both passes and the refit; under the shipped
presence-absence head they are on, so a default
[`mars()`](https://gillescolling.com/timesift/reference/mars.md) is
earth’s specification fitted under them, and a head registered without
`weights` fits it unweighted. The passes run on the core the R package
calls, so the two keep the same terms and return the same coefficients;
`threads` searches that many columns at once and does not change what
comes back. A variable holding one value is predicted its mean.

## `discriminant()`

``` python
discriminant(
    data=None,
    degree=1,
    penalty=None,
    max_terms=None,
    min_gain=0.001,
    prune=True,
    calibrate=True,
    threads=1,
)
```

One flexible discriminant per variable over every bin-by-channel column,
fitted as mda’s `fda(method = mars)` fits it and as biomod2 fits `FDA`.
Optimal scoring gives presence and absence one score each and regresses
the scored response on a MARS basis of the columns; the fitted score is
the one canonical variate, and the prediction is the posterior
probability of presence under two normal classes around the class
centroids on it, the classes’ shares among the fitting units as priors.

The basis is mda’s own MARS, not earth’s that `mars` reproduces: each
forward step adds a column linearly or a pair of hinges on it, and the
pass stops when a step lowers the residuals by less than `min_gain` of
them, when they fall to `min_gain` of the null model’s, when the
generalised cross-validation passes ten times the null model’s, or at
`max_terms` terms. The pruning drops the term of least t statistic, one
at a time, and keeps the subset of least generalised cross-validation,
which counts each term beyond the intercept as `1 + penalty / 2` degrees
of freedom. The defaults are mda’s, which biomod2 passes unchanged:
degree one, `penalty` 2 (3 above degree one), `min_gain=0.001` and
`max_terms = max(21, 2 p + 1)` for `p` columns.

The head’s case weights set the classes’ scores and the variate, and not
the basis, whose forward pass mda runs unweighted; under the shipped
presence-absence head they are on, so a default
[`discriminant()`](https://gillescolling.com/timesift/reference/discriminant.md)
is mda’s specification fitted under them, and a head registered without
`weights` fits it unweighted. `calibrate=True` recalibrates the
posterior by a probit regression of the response on it under the case
weights, on the fitting units, as biomod2 always does for `FDA`; `False`
predicts the posterior. The passes run on the core the R package calls,
so the two keep the same terms and predict the same probabilities;
`threads` searches that many columns at once and does not change what
comes back. A variable holding one value, or one the basis does not
reach, is predicted its mean and named in `unfitted`. The learner needs
a presence-absence response, under a head whose loss is the binary
cross-entropy.

## `additive()`

``` python
additive(data=None, k=10, gamma=1.0, max_knots=2000, threads=1)
```

One additive model per variable, a smooth function of every
bin-by-channel column, fitted as mgcv’s
`gam(y ~ s(x1) + s(x2) + ..., method = "GCV.Cp")` fits it and as biomod2
fits `GAM`. Each column enters as a thin plate regression spline of `k`
basis functions (Wood 2003): a cubic radial function centred on each of
the column’s distinct values, reduced to its `k - 2` directions of
greatest eigenvalue, together with the linear function, which the
penalty on the spline’s squared second derivative leaves free. Each
smooth sums to zero over the units, beside one intercept.

The coefficients maximise the penalised likelihood and the smoothing
parameters, one per column, minimise the unbiased risk estimator under a
binary cross-entropy head and the generalised cross-validation score
under a squared-error one, by Newton’s method with the exact derivatives
(Wood 2008). `gamma` multiplies the charge each effective degree of
freedom adds to that criterion. Over columns as alike as neighbouring
weeks the criterion can have more than one local minimum; the one the
search settles in then depends on where it starts, and mgcv, starting
from a rule of its own parametrisation, can settle in another. The
defaults are mgcv’s, which biomod2 passes unchanged. Above `max_knots`
distinct values the radial functions are centred on that many of them,
drawn as mgcv draws them. A column of fewer than `k` distinct values
takes as many basis functions as it holds values, a column of two enters
linearly and a column of one is left out; a column whose linear part the
columns before it span keeps only its penalised part.

The head’s case weights enter the likelihood as mgcv’s prior weights;
under the shipped presence-absence head they are on, so a default
[`additive()`](https://gillescolling.com/timesift/reference/additive.md)
is mgcv’s specification fitted under them, and a head registered without
`weights` fits it unweighted. The model holds at most as many
coefficients as there are units, one for the intercept and `k - 1` per
column. The fit runs on the core the R package calls, so the two
languages fit the same model; `threads` works that many columns and
variables at once and does not change what comes back. A variable
holding one value is predicted its mean and named in `unfitted`, and one
whose smoothing parameter search stopped short of its tolerance is named
in `stopped`.

## `perceptron()`

``` python
perceptron(
    data=None,
    hidden=None,
    decay=None,
    range=None,
    max_iter=None,
    skip=False,
    abs_tol=0.0001,
    rel_tol=1e-08,
    preset='default',
    max_hessian=2.0,
    seed=1,
)
```

One network per variable over every bin-by-channel column, fitted as the
nnet package fits it and as biomod2 fits `ANN`. Each of `hidden`
logistic units takes a bias and every column, and the output takes a
bias, every hidden unit and, with `skip`, every column again. The
weights start uniform on `[-range, range]` and are fitted by the
variable metric (BFGS) method of Nash (1990), the minimiser nnet uses,
on the response head’s loss plus `decay` times the sum of the squared
weights, biases included. The fit stops after `max_iter` iterations,
when the objective falls below `abs_tol`, or when an iteration lowers it
by no more than `rel_tol` of itself.

Under a presence-absence head the output is the logistic function of its
sum and the loss the cross-entropy, nnet’s `entropy = TRUE`; under a
head with a squared-error loss the output is the sum itself and the loss
the sum of squares, nnet’s `linout = TRUE`; under a count head the
output is the exponential of the sum and the loss the Poisson deviance,
which nnet does not offer. biomod2 leaves nnet’s own `entropy = FALSE`,
fitting a presence-absence response by least squares on the logistic
output; the learner fits the head’s loss, as every learner does.

`preset` says whose defaults the settings left `None` take. `"default"`
is what biomod2’s default option set fits: two hidden units, as biomod2
sets them, and nnet’s own `decay=0`, `range=0.7` and `max_iter=100`.
`"bigboss"` is biomod2’s tuned option set: five hidden units,
`decay=0.1`, `range=0.1` and `max_iter=200`. A setting given explicitly
beats either.

The network, its objective and the minimiser live in the core the R
package calls, pinned against nnet in the fixtures from the same
starting weights, so the two languages fit the same network. Neither
scales the columns, as nnet does not, so a record read in its own units
saturates the hidden units sooner the wider its range. The minimiser
holds an approximate inverse Hessian of one number per pair of weights;
a network that would need more than `max_hessian` gigabytes for it is
refused with the size. The case weights are the head’s and weigh each
unit’s term of the loss; under the shipped presence-absence head they
are on, so a default
[`perceptron()`](https://gillescolling.com/timesift/reference/perceptron.md)
is biomod2’s `ANN` specification fitted under them, and a head
registered without `weights` fits it unweighted. A variable holding one
value is predicted its mean.

## `hierarchical()`

``` python
hierarchical(
    data=None,
    spatial='none',
    random=False,
    cov='exponential',
    neighbours=15,
    m=6,
    boundary=1.5,
    nodes=5,
    threads=1,
)
```

One Bayesian logistic model per response: a logistic regression on every
bin-by-channel column of the representation, standardised so the prior
on a coefficient means the same thing for each, optionally with an
intercept for each unit and a Gaussian-process field over the targets’
coordinates. A unit then carries what its own record says and what its
neighbours’ presences say, under the folds every other learner is scored
on.

The coefficients, the intercept included, have a `N(0, 2.5^2)` prior. A
unit’s intercept is `N(0, sd^2)` and the field has a marginal standard
deviation and a range, each standard deviation under a
penalised-complexity prior (Simpson et al. 2017) with
`P(sd > 3) = 0.01`, and the range under one anchored at a fifth of the
coordinates’ extent with `P(range < anchor) = 0.5` (Fuglstad et
al. 2019). Coordinates are centred and divided by one factor, so
distances keep their proportions.

The field is a Gaussian process on the two columns named by `coords` in
`timesift.timesift`. `"hsgp"` is the Hilbert-space approximation (Solin
and Sarkka 2020) with `m` Laplacian eigenfunctions per axis, and
`"nngp"` the nearest-neighbour process (Datta et al. 2016) over the
distinct locations, each conditioned on its `neighbours` nearest among
those before it in lexicographic order of the coordinates, with the
covariance `cov`, one of `"exponential"`, `"matern32"`, `"matern52"` and
`"gaussian"`, whose range is the field’s. Its sparse precision is
factored once per set of hyperparameters, so the fit scales with the
number of locations rather than their square.

Inference is Laplace’s method over the coefficients, the intercepts and
the field together, and the hyperparameters are integrated over on a
grid of `nodes` points per hyperparameter, centred on the mode of their
posterior and weighted by it (Rue, Martino and Chopin 2009). With an
intercept alone its standard deviation is set at the mode of its
posterior; with neither the fit is the posterior mode of the
coefficients, so at `spatial="none"` and `random=False` the learner is a
penalised logistic model, every coefficient shrunk towards zero by its
prior. A prediction at new units interpolates the field to their
coordinates, so the targets it is given carry the same coordinate
columns. The head’s case weights enter the likelihood in every
configuration. A response holding one value is predicted its mean. The
learner fits a presence-absence head.

With `random=True` each unit, as named by `id` in `timesift.timesift`,
gets an intercept, which absorbs what its several targets share beyond
the record: it is identified where a unit carries more than one target.
A prediction adds a unit’s intercept where the unit was in the fit and
leaves it at zero, the population level, where it was not, so a unit
held out whole is predicted from its record and its place alone.

Both languages call one C++ core, so the same input gives the same fit
in either.

## `mlp()`

``` python
mlp(data=None, hidden=(512, 256), dropout=0.3, activation='relu', **settings)
```

Flattens the channels and builds in no temporal geometry.

`hidden`, `dropout` and `activation` (`"relu"`, `"gelu"` or `"selu"`)
are the architecture; anything else named is a training setting applied
on top of the
[`train_control()`](https://gillescolling.com/timesift/reference/train_control.md)
the learner is fitted under.

## `cnn()`

``` python
cnn(data=None, channels=(16, 32, 64, 128), kernel=7, dropout=0.3, **settings)
```

Convolution, batch normalisation, activation and pooling, then global
average pooling.

## `rescnn()`

``` python
rescnn(
    data=None,
    channels=(32, 64, 128, 256),
    blocks_per_stage=2,
    kernel=7,
    dilations=(1, 2, 4, 8),
    dropout=0.3,
    **settings,
)
```

Dilated residual blocks with channel gates, pooling average and maximum
together.

## `train_control()`

``` python
train_control(**settings)
```

The settings every neural learner reads, with anything named here
replacing its default.

`epochs`, `batch_size`, `learning_rate`, `weight_decay`,
`early_stopping`, `val_frac`, `device` and `seed` are the settings a run
is described by, and `swa` with `swa_start` average the weights over the
tail of the schedule rather than keeping one epoch out of it. What a
rare response weighs is the response head’s, not a training setting.

`optimizer` is `"adamw"`, `"adam"` or `"sgd"`, torch’s optimisers at
their own defaults besides the learning rate and the weight decay;
`"sgd"` takes no momentum. The weight decay is decoupled from the
gradient under `"adamw"` and added to it as `weight_decay` times each
parameter under the other two, as torch’s optimisers take it.

`penalty` weighs a penalty added to the loss of every batch,
`penalty * (alpha * sum(abs(W)) + (1 - alpha) * sqrt(sum(W ** 2)))`
summed over every weight matrix and kernel of the network, its biases
and normalisation scales left out. This is cito’s `lambda`, the penalty
biomod2’s `DNN` fits under, and 0 adds nothing. `alpha` is read as
[`elasticnet()`](https://gillescolling.com/timesift/reference/elasticnet.md)
reads its `alpha`: 1 penalises the absolute weights alone and 0 the norm
alone, so cito’s `alpha` is `1 - alpha`.

`schedule` moves the learning rate over the epochs: `"cosine"` anneals
it to zero over the budget, `"constant"` holds it, and `"plateau"`
multiplies it by `plateau_factor` once the loss has not improved for
`plateau_patience` epochs, reading the validation loss where `val_frac`
holds a set back and the epoch’s mean training loss where it does not,
as cito’s `reduce_on_plateau` reads them. torch’s own relative threshold
of `1e-4` decides an improvement.

`batch_size` is the most targets an optimiser step reads: the fitting
targets are cut into as few batches of at most that many as they divide
into, of as equal a length as they can be. `val_frac` is held back from
every fit alike by a plain random permutation, the fit on all targets a
run ends with included, and `early_stopping` is read only on that set:
an epoch count of patience, or `math.inf`, the default, to train the
whole budget and restore the epoch with the lowest validation loss. At
the `val_frac` default of 0 nothing is held back: every fitting target
is trained on, the whole budget runs, and the fit keeps the last epoch,
where the cosine schedule has annealed the learning rate to zero.

`device` is `"auto"` for the graphics processor where there is one,
NVIDIA’s or Apple’s, or the name of a device to train on. A fitted
encoder carries the setting rather than the device it resolved to, so a
fit made on one machine predicts on another.

## `TrainControl`

``` python
TrainControl(
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    optimizer,
    penalty,
    alpha,
    schedule,
    plateau_factor,
    plateau_patience,
    early_stopping,
    val_frac,
    device,
    seed,
    swa,
    swa_start,
)
```

How long to train, on what, and when to stop.

Attributes:

- `epochs` - int
- `batch_size` - int
- `learning_rate` - float
- `weight_decay` - float
- `optimizer` - str
- `penalty` - float
- `alpha` - float
- `schedule` - str
- `plateau_factor` - float
- `plateau_patience` - int
- `early_stopping` - float
- `val_frac` - float
- `device` - str
- `seed` - int
- `swa` - bool
- `swa_start` - float

### `override()`

``` python
override(self, settings: dict)
```

The control with the settings a learner or a call gave applied on top of
it.

## `Learner`

``` python
Learner(name, fit, predict, needs, params, data, reads, multi)
```

A name, a fit and a predict, what has to be installed for them to run,
and what the learner reads.

The one interface every arm goes through, the ones that ship and a pair
of your own alike. `data` pins the learner to one representation, or is
`None` to run it across every representation offered. `reads` is whether
it takes a tabular block or an ordered sequence of bins, and `multi` is
whether one fitted model covers every response or one is fitted per
response and the matrix assembled from them.

Attributes:

- `name` - str
- `fit` - Callable
- `predict` - Callable
- `needs` - tuple\[str, …\]
- `params` - dict
- `data` - object
- `reads` - str
- `multi` - str

### `require()`

``` python
require(self)
```

Error, naming the install, unless what the learner needs is importable.

## `flatten()`

``` python
flatten(x: TimesiftMatrix)
```

`[unit, bin, channel]` to `[unit, bin * channel]` in the array’s own
order.

A channel that holds the same number in every bin carries no bin of its
own and is one predictor, read once: repeating it would put the same
column in front of a penalised fit as often as the grain has bins, and
give it that many chances of being drawn by a forest or picked by a
forward search.

## `register_learner()`

``` python
register_learner(name: str, constructor: Callable, overwrite: bool = False)
```

Make a learner available by name. The learners that ship are registered
the same way.

`constructor` is called with no arguments and returns a `Learner`, so a
learner asked for by name is built with its own defaults.

## `get_learner()`

``` python
get_learner(learner)
```

A `Learner`, whether it arrived as one or as the name of a registered
one.

## `learners()`

``` python
learners()
```

The learners registered under this session.

## `tune()`

``` python
tune(learner, grid: dict | None = None, metric=None, n_inner: int = 5, seed: int = 1)
```

A learner that searches `grid` on the units it is fitted on and fits the
best setting.

The search is a cross-validation inside those units, so in a run the
outer folds never see it: each fold chooses from its own training units,
and the score it is then read at is not selected on. biomod2’s
`BIOMOD_Tuning()` searches a grid per algorithm by the same device.

With `grid` left unset the learner is searched over the grid registered
under its name by `register_tuning`, which for the learners that ship is
the one `BIOMOD_Tuning()` searches: `mtry` of a forest from 1 to the
smaller of 10 and the number of columns; `trees`, `depth` and
`shrinkage` of a gbm-style boosting, `shrinkage` and `colsample` of the
second-order one; `degree` and `nprune` of `mars`; `degree` of
`discriminant`; `regmult` of `maxent`; `quantile` of `envelope`;
`hidden` of a `perceptron` at 2, 4, 6 and 8 with `decay` at 0.01, 0.05
and 0.1; and the layer width of `mlp` at 2, 4, 6 and 8.

A learner’s settings are the ones it carries as `params`. `grid` names
some of them and gives the values to try, and the grid is every
combination, the first setting varying fastest as in R, so a tie between
two combinations falls to the same one on both sides. A setting is
scored by the mean over the responses of the mean over inner folds of
`metric` on the cells a score is defined on. The inner folds keep the
grouping the outer fold map keeps whole. What was chosen is on the
fitted model as `chosen` and `table` and in the `settings` column of a
run’s candidate table.

## `Tuned`

``` python
Tuned(model, chosen, table)
```

What a tuned learner fitted: the model, the setting chosen, and every
setting’s score.

Attributes:

- `model` - object
- `chosen` - dict
- `table` - list

## `register_tuning()`

``` python
register_tuning(name: str, grid, overwrite: bool = False)
```

Register the grid a learner is tuned over when `tune` is given none.

`name` is the learner’s, as it reports under. `grid` is a dict of the
values to try, or a function of `(learner, x)` returning one, where `x`
is the representation the learner is fitted on: the second form is for a
grid that depends on the data, as the number of columns does, or on a
setting the learner carries. The grids of the learners that ship are
registered the same way.

## `tunings()`

``` python
tunings()
```

The learners a grid is registered for.
