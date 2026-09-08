# Python: what crosses the boundary

The three artifacts a split is carried in, and the digest that says two
arrays are the same array.

## `write_folds()`

``` python
write_folds(x, file)
```

Write a fold map as `id,fold`, ordered by unit.

## `read_folds()`

``` python
read_folds(file, units=None)
```

Read a fold map somebody else built, optionally aligned to a
representation’s units.

## `write_response()`

``` python
write_response(y: Response, file)
```

Write a response as `id` and one column per variable, ordered by unit.

## `read_response()`

``` python
read_response(file, units=None)
```

Read a response matrix. The columns after `id` are the variables, in the
file’s order.

## `write_cells()`

``` python
write_cells(cells: Cells, file)
```

Write a scorable mask, ordered by variable and then by fold.

## `read_cells()`

``` python
read_cells(file)
```

Read a scorable mask the other language computed.

## `digest_array()`

``` python
digest_array(values)
```

MD5 of the representation, byte-exactly as the spec defines it.
