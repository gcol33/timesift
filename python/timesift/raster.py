"""Maps: a fit applied to one target per cell of a grid, and how the range of a response changes
between two of them. Grids are xarray's, which is what this module needs beyond numpy."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np

__all__ = ["RangeChange", "project", "range_change"]

CODES = {"lost": -2, "kept": -1, "absent": 0, "gained": 1}


def _xarray():
    if importlib.util.find_spec("xarray") is None:
        raise ImportError("`project()` and `range_change()` read grids through xarray. Install it "
                          "with pip install xarray")
    import xarray
    return xarray


def project(fit, series=None, static=None, candidate: str = "ensemble", type: str = "response",
            chunk: int = 5000, **kwargs):
    """Predict one target per cell of a grid, each carrying the record of its own cell.

    This is ``BIOMOD_Projection()`` and ``BIOMOD_EnsembleForecasting()``: a map is the fit applied
    to one target per cell, at the grain the fit reads. ``series`` is an ``xarray.DataArray`` with
    a ``time`` dimension and two spatial ones, or a mapping of the fit's ``x`` column names to
    such; ``static`` is an ``xarray.Dataset``, or a mapping of the fit's ``static`` column names to
    ``DataArray`` over the two spatial dimensions. Every input shares one grid, and one set of
    instants. Cells are predicted in chunks of ``chunk``, and a cell is predicted where every
    input holds a value at that cell: a cell with a missing reading anywhere is NaN in every
    layer. ``candidate``, ``type`` and the other keywords are those of ``Timesift.predict``.

    Returns an ``xarray.DataArray`` over ``response`` and the two spatial dimensions of the
    inputs; for ``type="spread"`` a ``statistic`` dimension as well.
    """
    xr = _xarray()
    if type not in ("response", "binary", "spread"):
        raise ValueError(f"`type` is one of response, binary, spread, got {type!r}")
    if not isinstance(chunk, (int, np.integer)) or isinstance(chunk, bool) or chunk < 1:
        raise ValueError("`chunk` is a number of cells of 1 or more")
    spec = fit.spec
    if spec.target_time is not None:
        raise ValueError("this fit anchors each target in time with `target_time`, and a grid "
                         "cell has one record: project a fit made on calendar grains")
    if spec.id is None:
        raise ValueError("a projection names each cell by its number, so the fit has to have "
                         "been made with `id`")
    records, statics, template = _inputs(spec, series, static)
    dims = tuple(d for d in template.dims if d != "time")
    if len(dims) != 2:
        raise ValueError(f"a grid has two spatial dimensions besides `time`, got {len(dims)}")
    shape = tuple(template.sizes[d] for d in dims)
    total = int(np.prod(shape))

    flat_static = {k: np.asarray(v.transpose(*dims).values, dtype=np.float64).reshape(total)
                   for k, v in statics.items()}
    flat_static.update(_cell_coordinates(spec, template, dims, shape))
    flat_series = {k: np.asarray(v.transpose("time", *dims).values,
                                 dtype=np.float64).reshape(len(v["time"]), total)
                   for k, v in records.items()}
    mask = np.ones(total, dtype=bool)
    for v in flat_static.values():
        mask &= np.isfinite(v)
    for v in flat_series.values():
        mask &= np.isfinite(v[0])
    cells = np.flatnonzero(mask)
    if not len(cells):
        raise ValueError("no cell holds a value in every input, so there is nothing to predict")

    out = None
    for start in range(0, len(cells), int(chunk)):
        piece = cells[start:start + int(chunk)]
        kept, p = _project_chunk(fit, spec, flat_static, flat_series, records, piece, candidate,
                                 type, kwargs)
        if out is None:
            out = np.full((total,) + p.shape[1:], np.nan)
        out[kept] = p
    variables = list(fit.y.variables)
    coords = {d: template[d] for d in dims if d in template.coords}
    if type == "spread":
        from .stack import SPREAD_STATISTICS
        grid = out.reshape(shape + out.shape[1:]).transpose(2, 3, 0, 1)
        return xr.DataArray(grid, dims=("response", "statistic") + dims,
                            coords={"response": variables, "statistic": list(SPREAD_STATISTICS),
                                    **coords})
    grid = out.reshape(shape + (len(variables),)).transpose(2, 0, 1)
    return xr.DataArray(grid, dims=("response",) + dims, coords={"response": variables, **coords})


def _inputs(spec, series, static):
    xr = _xarray()
    wanted = tuple(spec.x or ())
    if wanted:
        if series is None:
            raise ValueError(f"this fit reads a record of {', '.join(wanted)}, so `series` is "
                             "needed: an xarray.DataArray with a time dimension, or a mapping of "
                             "them named for the variables")
        if isinstance(series, xr.DataArray):
            series = {wanted[0]: series}
        missing = [w for w in wanted if w not in series]
        if missing:
            raise ValueError(f"`series` has to be a mapping of grids named {', '.join(wanted)}")
        records = {w: series[w] for w in wanted}
        for name, grid in records.items():
            if "time" not in grid.dims:
                raise ValueError(f"`{name}` of `series` needs a `time` dimension")
        first = next(iter(records.values()))
        if any(not np.array_equal(g["time"].values, first["time"].values)
               for g in records.values()):
            raise ValueError("the variables of `series` have to share their instants")
    else:
        records = {}
    if spec.static:
        if static is None or any(s not in static for s in spec.static):
            raise ValueError(f"`static` has to hold grids named {', '.join(spec.static)}")
        statics = {s: static[s] for s in spec.static}
    else:
        statics = {}
    grids = list(records.values()) + list(statics.values())
    template = grids[0]
    spatial = tuple(d for d in template.dims if d != "time")
    for grid in grids:
        if tuple(d for d in grid.dims if d != "time") != spatial or any(
                grid.sizes[d] != template.sizes[d] for d in spatial):
            raise ValueError("the inputs have to share one grid: the same spatial dimensions and "
                             "sizes")
    return records, statics, template


def _cell_coordinates(spec, template, dims, shape) -> dict:
    """The columns a fit made with ``coords`` reads each target's place from, one value per cell.

    A spatial dimension named as a coordinate column supplies that column, and a column no
    dimension is named for takes the dimensions left, in their order, so a grid with dimensions
    ``("y", "x")`` and a fit with ``coords=("x", "y")`` places each cell by its own ``x`` and
    ``y`` whichever way the grid is laid out.
    """
    if not spec.coords:
        return {}
    free = [d for d in dims if d not in spec.coords]
    index = np.indices(shape)
    out = {}
    for c in spec.coords:
        d = c if c in dims else free.pop(0)
        along = np.asarray(template[d].values if d in template.coords else np.arange(len(
            template[d])), dtype=np.float64)
        out[c] = along[index[dims.index(d)].reshape(-1)]
    return out


def _project_chunk(fit, spec, flat_static, flat_series, records, cells, candidate, type, kwargs):
    ids = [f"c{int(c) + 1:09d}" for c in cells]
    targets = {spec.id: ids}
    for name, values in flat_static.items():
        targets[name] = values[cells]
    series = None
    if flat_series:
        keep = np.ones(len(cells), dtype=bool)
        for v in flat_series.values():
            keep &= np.isfinite(v[:, cells]).all(axis=0)
        if not keep.all():
            targets = {k: np.asarray(v)[keep] for k, v in targets.items()}
            ids = [i for i, k in zip(ids, keep) if k]
            cells = cells[keep]
        first = next(iter(records.values()))
        times = np.asarray(first["time"].values).astype("datetime64[s]")
        series = {spec.id: np.repeat(np.asarray(ids), len(times)),
                  spec.time: np.tile(times, len(ids))}
        for name, v in flat_series.items():
            series[name] = v[:, cells].T.reshape(-1)
    p = np.asarray(fit.predict(targets, series, candidate=candidate, type=type, **kwargs))
    return cells, p


@dataclass(frozen=True)
class RangeChange:
    """The change in range of each response: a ``table`` of counts, and the ``map`` of codes
    (-2 lost, -1 kept, 0 absent in both, 1 gained) of the shape of the inputs."""

    table: list
    map: np.ndarray
    variables: tuple


def range_change(now, later, threshold=None) -> RangeChange:
    """How the range of each response changes between two maps.

    Counts the cells a response is lost from, kept in and gained, as ``BIOMOD_RangeSize()`` does,
    a cell being in the range where the response is predicted present. With ``L`` the cells lost,
    ``K`` kept, ``G`` gained and ``A`` absent in both: the current range is ``L + K``, the later
    one ``K + G``, ``percent_loss`` is ``100 L / (L + K)``, ``percent_gain`` is ``100 G / (L + K)``
    and ``change`` is ``percent_gain - percent_loss``. A cell that is NaN in either map is left out
    of every count. ``now`` and ``later`` are ``xarray.DataArray`` with a ``response`` dimension,
    such as ``project()`` returns, or arrays of cells by responses. ``threshold`` is None where the
    maps are already 0 and 1, and otherwise one cut per response, or one for all.
    """
    a, names = _maps(now)
    b, _ = _maps(later)
    if a.shape != b.shape:
        raise ValueError(f"`now` and `later` have to hold the same cells and the same responses, "
                         f"got {a.shape} and {b.shape}")
    a, b = _present(a, threshold, "now"), _present(b, threshold, "later")
    codes = np.where((a == 1) & (b == 0), -2.0,
                     np.where((a == 1) & (b == 1), -1.0, np.where((a == 0) & (b == 1), 1.0, 0.0)))
    codes[np.isnan(a) | np.isnan(b)] = np.nan
    table = []
    for j, name in enumerate(names):
        col = codes[:, j]
        lost, kept = int((col == -2).sum()), int((col == -1).sum())
        gained, absent = int((col == 1).sum()), int((col == 0).sum())
        current = lost + kept
        loss = 100 * lost / current if current else float("nan")
        gain = 100 * gained / current if current else float("nan")
        table.append(dict(variable=name, lost=lost, kept=kept, gained=gained, absent=absent,
                          current=current, later=kept + gained, percent_loss=loss,
                          percent_gain=gain, change=gain - loss))
    return RangeChange(table=table, map=_unflatten(codes, now), variables=tuple(names))


def _maps(x):
    """A map as cells by responses, and the names of the responses."""
    if hasattr(x, "dims") and "response" in x.dims:
        dims = ("response",) + tuple(d for d in x.dims if d != "response")
        values = np.asarray(x.transpose(*dims).values, dtype=np.float64)
        return values.reshape(values.shape[0], -1).T, [str(r) for r in x["response"].values]
    m = np.asarray(x, dtype=np.float64)
    if m.ndim == 1:
        m = m[:, None]
    return m, [f"V{j + 1}" for j in range(m.shape[1])]


def _present(m, threshold, what):
    if threshold is None:
        finite = m[np.isfinite(m)]
        if not np.isin(finite, (0.0, 1.0)).all():
            raise ValueError(f"`{what}` holds values other than 0 and 1: give `threshold`, the "
                             "cut at or above which a cell is present")
        return m
    cut = np.atleast_1d(np.asarray(threshold, dtype=np.float64))
    if cut.size not in (1, m.shape[1]) or np.isnan(cut).any():
        raise ValueError("`threshold` is one cut, or one per response")
    out = (m >= cut).astype(np.float64)
    out[np.isnan(m)] = np.nan
    return out


def _unflatten(codes, like):
    if hasattr(like, "dims") and "response" in like.dims:
        xr = _xarray()
        dims = ("response",) + tuple(d for d in like.dims if d != "response")
        shape = tuple(like.sizes[d] for d in dims[1:])
        grid = codes.T.reshape((codes.shape[1],) + shape)
        return xr.DataArray(grid, dims=dims, coords={d: like[d] for d in dims if d in like.coords})
    return codes
