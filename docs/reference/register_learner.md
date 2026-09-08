# Register a learner

Makes a learner available by name to
[`grain_ladder()`](https://gillescolling.com/timesift/reference/grain_ladder.md)
and to `learners()`. The learners that ship are registered the same way,
so there is no list of names inside the fitting code.

## Usage

``` r
register_learner(name, constructor, overwrite = FALSE)

learners()
```

## Arguments

- name:

  Name the learner is asked for by.

- constructor:

  A function returning a
  [`learner()`](https://gillescolling.com/timesift/reference/learner.md).

- overwrite:

  Replace an existing registration.

## Value

The constructor, invisibly.

## Examples

``` r
learners()
```
