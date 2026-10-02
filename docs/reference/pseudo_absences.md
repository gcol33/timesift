# Draw pseudo-absences from a pool of background units

Presence-only records give a response of ones. A model needs units where
the response is zero, and these are drawn from `pool`, a table of
background units that carry the same columns as the targets (an
identifier, the `static` predictors, the `coords`, and a record in
`series` for each). The strategies are `bm_PseudoAbsences()`'s:

## Usage

``` r
pseudo_absences(
  pool,
  presences,
  n,
  strategy = c("random", "sre", "disk"),
  id = NULL,
  env = NULL,
  quantile = 0.025,
  coords = NULL,
  dist_min = 0,
  dist_max = Inf,
  lonlat = FALSE,
  repeats = 1L,
  seed = 1L
)
```

## Arguments

- pool:

  A data frame of background units.

- presences:

  A data frame of the presences, with the columns the strategy reads.

- n:

  Number of pseudo-absences in each draw.

- strategy:

  `"random"`, `"sre"` or `"disk"`.

- id:

  The column naming a unit in both tables, or `NULL` where no unit of
  the pool is a presence.

- env:

  For `"sre"`, the numeric columns the envelope is drawn on.

- quantile:

  For `"sre"`, the share of the presences left outside at each end of a
  column, in `[0, 0.5]`.

- coords:

  For `"disk"`, the two columns holding the coordinates.

- dist_min, dist_max:

  For `"disk"`, the least and greatest distance from the nearest
  presence.

- lonlat:

  For `"disk"`, whether `coords` are longitude and latitude in degrees.

- repeats:

  Number of independent draws, each with its own seed.

- seed:

  Random seed of the first draw; draw `r` uses `seed + r - 1`.

## Value

The drawn rows of `pool`, with the columns `set` (the draw) and `pseudo`
(`TRUE`), of class `timesift_pseudo_absences`.

## Details

- `"random"`: any unit of the pool that is not a presence.

- `"sre"`: a unit outside the envelope of the presences, the band
  between the `quantile` and `1 - quantile` quantiles of each of the
  `env` columns over the presences, as
  [`envelope()`](https://gillescolling.com/timesift/reference/envelope.md)
  draws it. A unit is outside where it leaves the band in at least one
  column.

- `"disk"`: a unit whose distance to the nearest presence lies between
  `dist_min` and `dist_max`, both included, on `coords`. The distance is
  planar in the units of the coordinates, or in metres from longitude
  and latitude in degrees where `lonlat` is `TRUE`, on a sphere of
  radius 6371008.8 m.

A presence is never its own absence: a unit of the pool whose `id` is
one of the presences' is not a candidate. The units a strategy admits
are the same in both languages; which `n` of them are drawn depends on
the language's generator, as the folds of
[`fold_map()`](https://gillescolling.com/timesift/reference/fold_map.md)
do. To fix the units, draw once and keep the table. A set of one's own
is the rows of `pool` it names, and needs no strategy.

The drawn units are not flagged in a fit: bind them to the presences
with a response of zero and fit as one table. Their `pseudo` column says
which they are, and `set` which draw.

## Examples

``` r
pool <- data.frame(cell = 1:200, x = rep(1:20, 10), y = rep(1:10, each = 20),
                   temp = rnorm(200))
presences <- pool[c(5, 25, 45), ]
pseudo_absences(pool, presences, n = 10, strategy = "disk", id = "cell",
                coords = c("x", "y"), dist_min = 3, dist_max = 8)
```
