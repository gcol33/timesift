"""Pseudo-absences: units drawn from a pool of background units where a response is zero."""

from __future__ import annotations

import numpy as np

from ._envelope import envelope_fit, envelope_predict

__all__ = ["pseudo_absences"]

STRATEGIES = ("random", "sre", "disk")
EARTH_RADIUS = 6371008.8


def pseudo_absences(pool, presences, n: int, strategy: str = "random", id=None, env=None,
                    quantile: float = 0.025, coords=None, dist_min: float = 0.0,
                    dist_max: float = np.inf, lonlat: bool = False, repeats: int = 1,
                    seed: int = 1) -> dict:
    """Draw pseudo-absences from a pool of background units.

    Presence-only records give a response of ones. A model needs units where the response is
    zero, and these are drawn from ``pool``, a mapping of column to array of background units that
    carry the same columns as the targets. The strategies are ``bm_PseudoAbsences()``'s:
    ``"random"`` admits any unit of the pool that is not a presence; ``"sre"`` a unit outside the
    envelope of the presences, the band between the ``quantile`` and ``1 - quantile`` quantiles of
    each of the ``env`` columns over the presences, a unit being outside where it leaves the band
    in at least one column; ``"disk"`` a unit whose distance to the nearest presence lies between
    ``dist_min`` and ``dist_max``, both included, on the two ``coords`` columns, planar in the
    units of the coordinates or in metres from longitude and latitude in degrees where ``lonlat``
    is true, on a sphere of radius 6371008.8 m.

    A presence is never its own absence: a unit of the pool whose ``id`` is one of the presences'
    is not a candidate. The units a strategy admits are the same in both languages; which ``n`` of
    them are drawn depends on the language's generator, as the folds of ``fold_map`` do. The result
    is a mapping of column to array holding the drawn rows of the pool, with ``set`` (the draw,
    from 1) and ``pseudo`` (true) added; draw ``r`` uses ``seed + r - 1``.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy is one of {', '.join(STRATEGIES)}, got {strategy!r}")
    for name, value in (("n", n), ("repeats", repeats)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
            raise ValueError(f"`{name}` is one whole number of 1 or more, got {value!r}")
    open_ = pa_candidates(pool, presences, strategy, id, env, quantile, coords, dist_min,
                          dist_max, lonlat)
    candidates = np.flatnonzero(open_)
    if len(candidates) < n:
        raise ValueError(f"the {strategy} strategy admits {len(candidates)} unit"
                         f"{'s' if len(candidates) != 1 else ''} of the pool, and {n} were "
                         "asked for")
    picked, sets = [], []
    for r in range(1, int(repeats) + 1):
        rng = np.random.default_rng(seed + r - 1)
        picked.append(candidates[np.sort(rng.choice(len(candidates), size=n, replace=False))])
        sets.append(np.full(n, r))
    rows = np.concatenate(picked)
    out = {k: np.asarray(v)[rows] for k, v in pool.items()}
    out["set"] = np.concatenate(sets)
    out["pseudo"] = np.ones(len(rows), dtype=bool)
    return out


def pa_candidates(pool, presences, strategy, id, env, quantile, coords, dist_min, dist_max,
                  lonlat) -> np.ndarray:
    """Which units of the pool a strategy admits, as a boolean array over its rows."""
    n_pool = len(next(iter(pool.values())))
    open_ = np.ones(n_pool, dtype=bool)
    if id is not None:
        if id not in pool or id not in presences:
            raise ValueError("`id` has to name a column of both tables")
        open_ = ~np.isin(np.asarray(pool[id]).astype(str), np.asarray(presences[id]).astype(str))
    if not len(next(iter(presences.values()))):
        raise ValueError("a strategy that reads the presences needs at least one")
    if strategy == "random":
        return open_
    if strategy == "sre":
        if not env or any(c not in pool or c not in presences for c in env):
            raise ValueError("`env` names the numeric columns of both tables the envelope is "
                             "drawn on")
        if not (isinstance(quantile, (int, float)) and 0 <= quantile <= 0.5):
            raise ValueError(f"`quantile` is one number in [0, 0.5], got {quantile!r}")
        x = np.column_stack([np.asarray(presences[c], dtype=np.float64) for c in env])
        band = envelope_fit(x, np.ones(len(x)), quantile)
        inside = envelope_predict(band, np.column_stack(
            [np.asarray(pool[c], dtype=np.float64) for c in env]))
        return open_ & (np.asarray(inside) == 0)
    if not coords or len(coords) != 2 or any(c not in pool or c not in presences for c in coords):
        raise ValueError("`coords` names the two coordinate columns of both tables")
    if dist_min < 0 or dist_max < dist_min:
        raise ValueError("`dist_min` and `dist_max` are distances with 0 <= dist_min <= dist_max")
    near = _nearest_distance(
        np.column_stack([np.asarray(pool[c], dtype=np.float64) for c in coords]),
        np.column_stack([np.asarray(presences[c], dtype=np.float64) for c in coords]), lonlat)
    return open_ & (near >= dist_min) & (near <= dist_max)


def _nearest_distance(src: np.ndarray, to: np.ndarray, lonlat: bool) -> np.ndarray:
    out = np.empty(len(src))
    for i, (a, b) in enumerate(src):
        if lonlat:
            out[i] = _haversine(a, b, to[:, 0], to[:, 1]).min()
        else:
            out[i] = np.sqrt((to[:, 0] - a) ** 2 + (to[:, 1] - b) ** 2).min()
    return out


def _haversine(lon1, lat1, lon2, lat2):
    rad = np.pi / 180
    a = (np.sin((lat2 - lat1) * rad / 2) ** 2
         + np.cos(lat1 * rad) * np.cos(lat2 * rad) * np.sin((lon2 - lon1) * rad / 2) ** 2)
    return 2 * EARTH_RADIUS * np.arcsin(np.sqrt(a))
