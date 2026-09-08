# Case weights that balance a rare response

The weight every learner that ships fits a presence-absence response
under: each presence of a response weighs the ratio of absences to
presences among the units handed in, capped, and each absence weighs
one. A response with a presence in one target of a hundred is otherwise
fitted away by any learner that minimises a mean loss, and the encoders,
the penalised fit, the forest and the forward search would each have to
decide that for themselves.

## Usage

``` r
positive_weights(y, cap = 50)
```

## Arguments

- y:

  The response matrix, `[unit, variable]`, 0/1.

- cap:

  Ceiling on the weight a presence is given, at least one.

## Value

A numeric `[unit, variable]` matrix of case weights, one per cell of
`y`.

## Details

The weights are the response head's: the shipped presence-absence head
carries this function as its `weights`, and a head registered with
`weights = function(y) positive_weights(y, cap = 20)` weights every
learner by that cap instead. A head without `weights` is fitted
unweighted.

## Examples

``` r
y <- cbind(rare = c(1, 0, 0, 0, 0, 0), common = c(1, 1, 1, 0, 0, 0))
positive_weights(y)
```
