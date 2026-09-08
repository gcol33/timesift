# Read and write the artifacts that cross the language boundary

The representation is built twice, once per language, from one shared
core. The response matrix, the fold map and the mask of scorable cells
are not: they are built once and read wherever they are needed, because
a fold map drawn from a seed in R and one drawn from the same seed in
Python are different maps, and aligning the two random streams would be
the wrong fix.

## Usage

``` r
write_folds(x, file)

read_folds(file, units = NULL)

write_response(y, file)

read_response(file, units = NULL)

write_cells(cells, file)

read_cells(file)
```

## Arguments

- x:

  A fold map from
  [`fold_map()`](https://gillescolling.com/timesift/reference/fold_map.md),
  or any named integer vector.

- file:

  Path to write to or read from.

- units:

  The units to align to, in the order they are wanted. A unit the file
  has no row for is an error; a unit the file carries beyond these is
  dropped. `NULL`, the default, returns the file's own rows in the
  file's own order.

- y:

  A response matrix, as
  [`scorable_cells()`](https://gillescolling.com/timesift/reference/scorable_cells.md)
  takes one.

- cells:

  A mask from
  [`scorable_cells()`](https://gillescolling.com/timesift/reference/scorable_cells.md).

## Value

The writers return `file`, invisibly. `read_folds()` returns a named
integer vector of class `timesift_folds`, `read_response()` a numeric
matrix with the units in its row names, and `read_cells()` a
`timesift_cells` data frame.

## Details

These six functions are the handover. The Python side carries
`write_folds()` and its siblings under the same names; they write the
same bytes from the same artifact and read the same files, so a fold map
built in either language is usable in the other without the caller
knowing the format.

The format is normative and is given in `inst/spec/representation.md`:
CSV, UTF-8, a header row, no quoting, LF line endings on every platform,
numbers at twelve significant digits, and rows ordered by the identifier
under C collation.

## Examples

``` r
set.seed(1)
y <- matrix(rbinom(120, 1, 0.3), nrow = 20,
            dimnames = list(sprintf("p%02d", 1:20), paste0("sp", 1:6)))
f <- fold_map(y, v = 4)

path <- tempfile(fileext = ".csv")
write_folds(f, path)
identical(read_folds(path, names(f)), f[names(f)])
unlink(path)
```
