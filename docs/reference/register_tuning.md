# Register the grid a learner is tuned over

Makes `tune(learner)` search `grid` when it is given no grid of its own.
The grids of the learners that ship are registered the same way.

## Usage

``` r
register_tuning(name, grid, overwrite = FALSE)

tunings()
```

## Arguments

- name:

  Name of the learner the grid belongs to, as it reports under.

- grid:

  A named list of values to try, or a function of `(learner, x)`
  returning one, where `x` is the representation the learner is fitted
  on. The second form is for a grid that depends on the data, as the
  number of columns does, or on a setting the learner carries.

- overwrite:

  Replace an existing registration.

## Value

The grid, invisibly.

## Examples

``` r
register_tuning("flat_glm", list(thresh = c(1e-4, 1e-6)), overwrite = TRUE)
tunings()
```
