# Case weights that balance a rare response

The weight every learner that ships fits a presence-absence response
under: each presence of a response weighs the ratio of absences to
presences among the fitting units, capped, and each absence weighs one.
A response with a presence in one target of a hundred is otherwise
fitted away by any learner that minimises a mean loss, and the encoders,
the penalised fit, the forest and the forward search would each have to
decide that for themselves.

## Usage

``` r
positive_weights(y, cap = 50, fitting = NULL)
```

## Arguments

- y:

  The response matrix, `[unit, variable]`, 0/1.

- cap:

  Ceiling on the weight a presence is given, at least one.

- fitting:

  A logical vector over the rows of `y`, `TRUE` for the units the model
  is fitted on, or `NULL` for all of them. The ratio is read off those
  rows alone.

## Value

A numeric `[unit, variable]` matrix of case weights, one per cell of
`y`.

## Details

The ratio is read off the units the model is fitted on and the weight
applies to every unit handed in. An encoder holds part of its units back
as an inner validation set and reads the loss it stops on under the same
weights, so the loss that stops the fit is the loss the fit minimises; a
unit held back that way is not in the count the ratio is made from.

The weights are the response head's: the shipped presence-absence head
carries this function as its `weights`, and a head registered with
`weights = function(y, fitting) positive_weights(y, cap = 20, fitting = fitting)`
weights every learner by that cap instead. A head without `weights` is
fitted unweighted.

## Examples

``` r
y <- cbind(rare = c(1, 0, 0, 0, 0, 0), common = c(1, 1, 1, 0, 0, 0))
positive_weights(y)
positive_weights(y, fitting = c(TRUE, TRUE, TRUE, TRUE, FALSE, FALSE))
```
