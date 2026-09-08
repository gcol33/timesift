# Combine learners, or representations, into a set

[`c()`](https://rdrr.io/r/base/c.html) on learners is the set of them,
and on representations the set of those. A set handed to
[`c()`](https://rdrr.io/r/base/c.html) again splices, so a set can be
added to rather than rewritten, which is what
[`list()`](https://rdrr.io/r/base/list.html) cannot do: `c(base, cnn())`
where `base` is already a set.

## Usage

``` r
# S3 method for class 'timesift_learner'
c(...)

# S3 method for class 'timesift_models'
c(...)

# S3 method for class 'timesift_representation'
c(...)

# S3 method for class 'timesift_sift'
c(...)
```

## Arguments

- ...:

  Learners, representations, or sets of either.

## Value

A `timesift_models` for learners and a `timesift_sift` for
representations.

## Details

`models` and `sift` take either form. A length-one string is the name of
a registered learner.

## Examples

``` r
base <- c(elasticnet(), forest())
base
c(base, stepwise())

c(grains("day", "week"), lookback("30 days"))
```
