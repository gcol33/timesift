# Project a fit onto rasters

Predicts one target per cell of a raster, each carrying the record of
its own cell, and returns the predictions as a raster with one layer per
response. This is `BIOMOD_Projection()` and
`BIOMOD_EnsembleForecasting()`: a map is the fit applied to one target
per cell, at the grain the fit reads.

## Usage

``` r
project(
  fit,
  series = NULL,
  static = NULL,
  candidate = "ensemble",
  type = c("response", "binary", "spread"),
  chunk = 5000L,
  ...
)
```

## Arguments

- fit:

  A
  [`timesift()`](https://gillescolling.com/timesift/reference/timesift.md)
  result.

- series:

  A
  [terra::SpatRaster](https://rspatial.github.io/terra/reference/SpatRaster-class.html)
  whose layers are instants, for a fit made on a record; for a record of
  several variables, a list of them named as the fit's `x` columns.
  Every raster shares one geometry and one set of instants.

- static:

  A `SpatRaster` whose layers are named as the fit's `static` columns.

- candidate, type, ...:

  As for
  [`predict.timesift()`](https://gillescolling.com/timesift/reference/predict.timesift.md):
  the candidate (the ensemble by default), `"response"`, `"binary"` or
  `"spread"`, and the arguments they take.

- chunk:

  Number of cells predicted at once.

## Value

A `SpatRaster` of the geometry of the inputs, one layer per response.
Under `type = "spread"` one layer per response and statistic, named
`response.statistic`.

## Details

The record of a cell is its values through the layers of `series`, which
are placed in time by
[`terra::time()`](https://rspatial.github.io/terra/reference/time.html),
and the static predictors of a cell are its values in the layers of
`static`, named as the columns of `targets` were when the fit was made.
A fit made with `coords` reads the centre of each cell. Cells are
predicted in chunks, so a raster larger than memory is read a chunk of
cells at a time. A cell is predicted where every input holds a value at
that cell; a cell with a missing reading anywhere is `NA` in every
layer.

A map for a later period is the same call with the later record, and the
binary maps of the two are what
[`range_change()`](https://gillescolling.com/timesift/reference/range_change.md)
compares.

## Examples

``` r
if (requireNamespace("terra", quietly = TRUE)) {
  set.seed(1)
  r <- terra::rast(nrows = 6, ncols = 5, xmin = 0, xmax = 5, ymin = 0, ymax = 6)
  elev <- terra::setValues(r, rnorm(30))
  slope <- terra::setValues(r, rnorm(30))
  names(elev) <- "elev"
  names(slope) <- "slope"
  static <- c(elev, slope)
  cells <- data.frame(cell = seq_len(30), elev = elev[][, 1], slope = slope[][, 1])
  cells$sp1 <- rbinom(30, 1, plogis(cells$elev))
  cells$sp2 <- rbinom(30, 1, plogis(-cells$slope))
  fit <- timesift(cells, y = c(sp1, sp2), id = cell, static = c(elev, slope),
                  learners = elasticnet(), ensemble = FALSE, n_inner = NULL,
                  resampling = cv(v = 3), verbose = FALSE)
  project(fit, static = static, candidate = "elasticnet / static")
}
```
