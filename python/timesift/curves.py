"""How a prediction responds to one predictor, the others held at a reference."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .representation import TimesiftMatrix
from .stack import ensemble_combine, ensemble_spread

__all__ = ["ResponseCurve", "response_curve"]

FIXED = {"mean": np.mean, "median": np.median, "min": np.min, "max": np.max}


@dataclass(frozen=True)
class ResponseCurve:
    """The prediction against the value of a predictor, or of two.

    ``prediction`` is ``[value, response]``; with a second predictor ``value`` and ``value_with``
    run over the grid of the two, the first fastest, as R's table does. ``sd``, ``lower`` and
    ``upper`` are the ensemble members' spread at every value, and ``None`` where it was not asked.
    """

    value: np.ndarray
    variable: tuple[str, ...]
    prediction: np.ndarray
    candidate: str
    predictor: str
    fixed: str
    value_with: np.ndarray | None = None
    with_label: str | None = None
    sd: np.ndarray | None = None
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None


def response_curve(fit, candidate: str = "ensemble", predictor=None, with_=None,
                   fixed: str = "mean", n: int = 50, spread: bool = False) -> ResponseCurve:
    """Vary one predictor of a fitted candidate across the range it takes, the rest held fixed.

    A predictor is a cell of the representation the candidate reads: one statistic in one bin.
    Given as the name of a channel it is that statistic moved together in every bin, which is the
    way a column named in ``static`` is moved; given as ``{"bin": ..., "channel": ...}`` it is one
    cell, and a bare bin name is enough where the representation has a single channel. The
    reference is one made-up unit whose every cell holds the ``fixed`` summary (``"mean"``,
    ``"median"``, ``"min"`` or ``"max"``) of that cell over the targets; a cell that is the same
    for every target keeps its value. The curve is read off the model fitted on all targets.

    For the ensemble every member is moved the same way, a member that does not carry the predictor
    is held at the reference, and the members' predictions are combined by the stack; ``spread``
    adds the members' standard deviation and interval. ``with_`` is a second predictor, and the
    prediction is then read over the grid of the two; it is ``with`` in R, which Python reserves.
    """
    if fixed not in FIXED:
        raise ValueError(f"fixed must be one of {tuple(FIXED)}, got {fixed!r}")
    if predictor is None:
        raise ValueError("`predictor` names what to vary: a channel, a bin, or "
                         "{'bin': ..., 'channel': ...}")
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 2:
        raise ValueError("`n` is a number of values of 2 or more")
    stack = getattr(fit, "stack", None)
    if candidate == "ensemble":
        if stack is None:
            raise ValueError("this fit made no ensemble to read")
        members = list(stack.members)
    else:
        if spread:
            raise ValueError("`spread` is the disagreement of an ensemble's members")
        members = [candidate]
    arrays = [fit.representations[fit.representation_of(c)] for c in members]
    models = [fit.models[c] for c in members]

    first = _cell(arrays, predictor, "predictor")
    second = _cell(arrays, with_, "with_") if with_ is not None else None
    value, other = _grid(first["range"], second["range"] if second else None, int(n))

    preds = {name: model.predict(_curve_array(m, first["spec"], second["spec"] if second else None,
                                              value, other, fixed))
             for name, m, model in zip(members, arrays, models)}
    p = ensemble_combine(stack, preds) if candidate == "ensemble" else preds[members[0]]
    extra = {}
    if spread:
        s = ensemble_spread(stack, preds)
        extra = {"sd": s[:, :, 1], "lower": s[:, :, 3], "upper": s[:, :, 4]}
    return ResponseCurve(value=value, variable=tuple(fit.y.variables),
                         prediction=np.asarray(p, dtype=np.float64), candidate=candidate,
                         predictor=first["label"], fixed=fixed, value_with=other,
                         with_label=second["label"] if second else None, **extra)


def _cell(arrays, predictor, what: str) -> dict:
    """A predictor resolved against every array of a candidate: the cell it names in each, the
    range it takes over the targets, and a label. An array that does not carry it has no cell and
    is held at the reference."""
    spec = None
    for m in arrays:
        try:
            spec = _spec(m, predictor, what)
            break
        except ValueError:
            continue
    if spec is None:
        _spec(arrays[0], predictor, what)
    cells = [_find(m, spec) for m in arrays]
    if all(c is None for c in cells):
        raise ValueError(f"no representation of this candidate carries `{_label(spec)}`")
    values = np.concatenate([m.values[:, c[0], c[1]].ravel()
                             for m, c in zip(arrays, cells) if c is not None])
    lo, hi = float(values.min()), float(values.max())
    if not np.isfinite(hi - lo) or hi == lo:
        raise ValueError(f"`{_label(spec)}` takes one value over the targets, so there is no curve")
    return {"range": (lo, hi), "spec": spec, "label": _label(spec)}


def _spec(m: TimesiftMatrix, predictor, what: str) -> dict:
    """A predictor as ``{bin, channel}`` with ``None`` for every bin. A string names a channel
    where the array has one called that, and a bin where it has one channel."""
    if isinstance(predictor, dict):
        return {"bin": predictor.get("bin"), "channel": predictor.get("channel")}
    if not isinstance(predictor, str):
        raise ValueError(f"`{what}` is a channel, a bin, or a dict with `bin` and `channel`")
    if predictor in m.stats:
        return {"bin": None, "channel": predictor}
    if len(m.stats) == 1 and predictor in m.bins:
        return {"bin": predictor, "channel": m.stats[0]}
    raise ValueError(f'`{what}` is "{predictor}", which is neither a channel nor a bin of a '
                     f'one-channel representation. The channels are {", ".join(m.stats)}')


def _label(spec: dict) -> str:
    return spec["channel"] if spec["bin"] is None else f'{spec["channel"]} @ {spec["bin"]}'


def _find(m: TimesiftMatrix, spec: dict):
    """The indices of the cell a spec names in one array: the bins and the channel, or None."""
    if spec["channel"] not in m.stats:
        return None
    if spec["bin"] is None:
        bins = np.arange(len(m.bins))
    elif spec["bin"] in m.bins:
        bins = np.asarray([m.bins.index(spec["bin"])])
    else:
        return None
    return bins, m.stats.index(spec["channel"])


def _grid(range_, range_with, n: int):
    def step(r):
        return r[0] + (r[1] - r[0]) * np.arange(n) / (n - 1)

    value = step(range_)
    if range_with is None:
        return value, None
    return np.tile(value, n), np.repeat(step(range_with), n)


def _curve_array(m: TimesiftMatrix, spec: dict, spec_with, value: np.ndarray, other,
                 fixed: str) -> TimesiftMatrix:
    """The made-up units: one copy of the reference per value, the predictor cells set."""
    n = len(value)
    reference = FIXED[fixed](m.values, axis=0)
    values = np.broadcast_to(reference, (n,) + reference.shape).copy()
    for which, column in ((spec, value), (spec_with, other)):
        if which is None:
            continue
        cell = _find(m, which)
        if cell is None:
            continue
        bins, channel = cell
        for b in bins:
            values[:, b, channel] = column
    return replace(m, values=values, bin_n=np.repeat(m.bin_n[:1], n, axis=0),
                   units=tuple(f"curve{i:04d}" for i in range(n)))
