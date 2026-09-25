"""Sensor records whose response acts at a known temporal grain.

The twin of R's ``simulate_records()``. The design -- where each variable reads the calendar, how
its driver is standardised, and the link that turns the driver into a response -- draws nothing
at random and is the same numbers on both sides. The units are drawn from numpy's stream, so a
draw here is the same draw in distribution as R's and not number for number, as a fold map is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .representation import grain_matrix
from .response import Response

__all__ = ["MECHANISMS", "Simulation", "simulate_records"]

MECHANISMS = ("none", "event", "season", "lag")

# The true grain of each mechanism, and the shape of the weights inside it. Written as a table so a
# fifth mechanism is a row rather than a branch in the generator.
_GRAIN = {"event": "day", "season": "season", "lag": "week"}
_SHAPE = {"event": np.ones(3), "season": np.ones(1), "lag": np.exp(-np.arange(4.0))}


@dataclass
class Simulation:
    """A simulated record, its response, and everything the draw is reproducible from.

    ``readings`` is the long table :func:`grain_matrix` takes, as a mapping of ``unit``, ``time``
    and ``reading``. ``y`` is the ``[unit, variable]`` 0/1 response and ``driver`` the
    standardised driver ``z`` behind it. ``grain`` is the true grain, or ``None`` where the
    response does not read the record. ``weights`` is the ``[reading, variable]`` matrix defining
    the driver, ``link`` the solved ``b0`` and ``b1``, and ``design`` the settings of the draw.
    """

    readings: dict
    y: Response
    driver: np.ndarray
    grain: str | None
    weights: np.ndarray
    link: dict
    design: dict
    grain_stat: str = "mean"
    anchor: np.ndarray = field(default=None, repr=False)

    def __repr__(self) -> str:  # pragma: no cover - display only
        d = self.design
        truth = ("none, the response does not read the record" if self.grain is None
                 else f"{self.grain} ({d['bins']} bins), stat mean")
        return (f"<timesift simulation> {d['mechanism']} over {d['n']} units x "
                f"{self.weights.shape[0]} readings\n"
                f"true grain: {truth}\n"
                f"{d['variables']} variables at prevalence {d['prevalence']:.3f} "
                f"(drawn {self.y.values.mean():.3f}), population AUC {d['auc']:.2f}")


def simulate_records(n: int = 300, mechanism: str = "none", variables: int = 10,
                     prevalence: float = 0.1, auc: float = 0.75, from_: str = "2021-09-01",
                     days: int = 365, step_hours: float = 3, seasonal: float = 8,
                     offset_sd: float = 1, anomaly_sd: float = 1, anomaly_days: float = 2,
                     offset_effect: float = 0, sensor_sd: float = 0.3,
                     year_start: str = "09-01", seed: int = 1, draw: int = 1) -> Simulation:
    """Draw units carrying a record and a presence-absence response acting at one known grain.

    The response is driven by ``g_ij = sum_t w_j(t) a_i(t)``, a weighted mean of unit ``i``'s
    latent anomaly: the record with the shared seasonal cycle and the unit's own offset taken out.
    The weights are constant within the bins of one grain and zero outside a short stretch of them,
    so the true grain is the coarsest grain at which ``g`` is still an exact linear functional of
    the representation. ``"none"`` draws the driver independently of the record; ``"event"`` reads
    three consecutive days, ``"season"`` one whole season, and ``"lag"`` four consecutive weeks
    under a geometric decay.

    The driver is standardised by its population mean and standard deviation, computed in closed
    form from the settings, and the response is ``Bernoulli(expit(b0 + b1 z))`` with ``b0`` and
    ``b1`` solved so the marginal prevalence is ``prevalence`` and the population area under the
    ROC curve of ``z`` is ``auc``. ``auc`` is a ceiling no fitted model reaches.

    ``seed`` fixes the design and ``draw`` the units, so two calls with one ``seed`` and two
    ``draw`` values are two samples of one population. ``from_`` is R's ``from``, renamed because
    ``from`` is a Python keyword.
    """
    if mechanism not in MECHANISMS:
        raise ValueError(f'`mechanism` is one of {", ".join(MECHANISMS)}, got "{mechanism}"')
    n = _whole(n, "n", 2)
    variables = _whole(variables, "variables", 1)
    days = _whole(days, "days", 1)
    draw = _whole(draw, "draw", 1)
    if 24 % step_hours != 0:
        raise ValueError(f"`step_hours` must divide 24, got {_num(step_hours)}")
    if not 0 < prevalence < 1:
        raise ValueError(f"`prevalence` must lie strictly between 0 and 1, got {_num(prevalence)}")
    if not 0.5 < auc < 1:
        raise ValueError(f"`auc` must lie strictly between 0.5 and 1, got {_num(auc)}")

    per_day = int(24 // step_hours)
    step = int(round(step_hours * 3600))
    start = np.datetime64(f"{from_}T00:00:00", "s")
    when = start + np.arange(days * per_day, dtype=np.int64) * np.timedelta64(step, "s")
    seconds = (when - when[0]).astype(np.int64).astype(float)
    phi = float(np.exp(-step_hours / (24 * anomaly_days)))
    season_wave = seasonal * np.cos(2 * np.pi * seconds / (365.25 * 86400))

    design = _design(mechanism, variables, when, year_start, phi, offset_effect * offset_sd,
                     anomaly_sd, prevalence, auc)

    units = tuple(f"d{draw:03d}u{i:05d}" for i in range(1, n + 1))
    rng = np.random.default_rng((seed + 100003 * draw) % 2147483647)
    offset = rng.normal(0.0, offset_sd, n) if offset_sd > 0 else np.zeros(n)
    anomaly = _ar1_field(rng, n, len(when), phi, anomaly_sd)
    reading = anomaly + season_wave[None, :] + offset[:, None] + \
        rng.normal(0.0, 1.0, anomaly.shape) * sensor_sd

    if mechanism == "none":
        z = rng.normal(0.0, 1.0, (n, variables))
    else:
        z = (anomaly + offset_effect * offset[:, None]) @ design["weights"] / design["sigma"]
    z = z * design["sign"]
    link = design["link"]
    p = 1.0 / (1.0 + np.exp(-(link["b0"] + link["b1"] * z)))
    y = (rng.random(z.shape) < p).astype(np.float64)
    names = tuple(f"v{j:02d}" for j in range(1, variables + 1))

    # Time-major, as R lays the table out: every unit's first reading, then every unit's second.
    readings = {"unit": np.tile(np.asarray(units, dtype=object), len(when)),
                "time": np.repeat(when, n),
                "reading": reading.T.reshape(-1)}
    return Simulation(
        readings=readings, y=Response(values=y, units=units, variables=names), driver=z,
        grain=design["grain"], weights=design["weights"], link=link, anchor=design["anchor"],
        design=dict(n=n, mechanism=mechanism, variables=variables, prevalence=prevalence,
                    auc=auc, from_=from_, days=days, step_hours=step_hours, seasonal=seasonal,
                    offset_sd=offset_sd, anomaly_sd=anomaly_sd, anomaly_days=anomaly_days,
                    offset_effect=offset_effect, sensor_sd=sensor_sd, year_start=year_start,
                    seed=seed, draw=draw, bins=design["bins"], anchor=design["anchor"]))


def _design(mechanism, variables, when, year_start, phi, offset_sd, anomaly_sd, prevalence, auc):
    """Everything two draws share. It depends on the settings and the reading grid, never on how
    many units are drawn, so samples of different sizes are drawn from one population."""
    link = link_coefficients(prevalence, auc)
    direction = np.resize(np.array([1.0, -1.0]), variables)
    if mechanism == "none":
        return dict(grain=None, bins=None, anchor=np.full(variables, -1),
                    weights=np.zeros((len(when), variables)), sigma=np.ones(variables),
                    sign=direction, link=link)
    grain = _GRAIN[mechanism]
    start, partial = _bin_edges(when, grain, year_start)
    bin_of = np.searchsorted(start, when, side="right") - 1
    count = np.bincount(bin_of, minlength=len(start))
    shape = _SHAPE[mechanism]
    anchor = _anchors(mechanism, variables, partial, len(shape))
    weights = np.empty((len(when), variables))
    for j in range(variables):
        k = np.zeros(len(start))
        k[anchor[j]:anchor[j] + len(shape)] = shape
        k = k / k.sum()
        weights[:, j] = k[bin_of] / count[bin_of]
    sigma = np.sqrt(offset_sd ** 2 + anomaly_sd ** 2 *
                    np.array([_ar1_quadform(weights[:, j], phi) for j in range(variables)]))
    return dict(grain=grain, bins=len(start), anchor=anchor, weights=weights, sigma=sigma,
                sign=direction, link=link)


def _anchors(mechanism, variables, partial, width):
    """The first bin of each variable's stretch, zero-based, spread over the positions whose whole
    stretch falls on bins the record covers for their full span."""
    starts = np.arange(len(partial) - width + 1)
    ok = np.array([i for i in starts if not partial[i:i + width].any()], dtype=int)
    if not len(ok):
        raise ValueError(f"the record holds no run of {width} whole {_GRAIN[mechanism]} bins, "
                         f"which the {mechanism} mechanism needs. Lengthen `days`.")
    if variables >= len(ok):
        return np.resize(ok, variables)
    # R rounds half away from zero where numpy rounds half to even; the positions are 1-based there.
    at = np.floor(np.linspace(1, len(ok), variables) + 0.5).astype(int) - 1
    return ok[at]


def _bin_edges(when, grain, year_start):
    """The bins the weights are defined on, from grain_matrix() itself on a two-unit record over
    the same instants, so a mechanism is anchored to the calendar the representation is built on."""
    probe = {"unit": np.repeat(np.array(["a", "b"], dtype=object), len(when)),
             "time": np.tile(when, 2), "reading": np.zeros(2 * len(when))}
    m = grain_matrix(probe, "unit", "time", "reading", grain=grain, stats=("mean",),
                     year_start=year_start)
    return np.asarray(m.bin_start).astype("datetime64[s]"), np.asarray(m.bin_partial, dtype=bool)


def _ar1_quadform(w, phi):
    """Var(sum_t w_t e_t) for an AR(1) of unit marginal variance, in one pass."""
    acc = 0.0
    s = np.empty(len(w))
    for i, wi in enumerate(w):
        acc = phi * acc + wi
        s[i] = acc
    return float(2 * np.sum(w * s) - np.sum(w ** 2))


def _ar1_field(rng, n, steps, phi, sd):
    e = rng.normal(0.0, 1.0, (n, steps)) * (sd * np.sqrt(1 - phi ** 2))
    e[:, 0] = rng.normal(0.0, 1.0, n) * sd
    for k in range(1, steps):
        e[:, k] += phi * e[:, k - 1]
    return e


_LINK_CACHE: dict = {}


def link_coefficients(prevalence: float, auc: float) -> dict:
    """``b0`` and ``b1`` such that a standard normal driver gives the asked-for prevalence and area
    under the ROC curve, both integrated on the fixed grid R integrates on."""
    key = (float(prevalence), float(auc))
    if key in _LINK_CACHE:
        return dict(_LINK_CACHE[key])
    z = np.linspace(-8, 8, 8001)
    dz = z[1] - z[0]
    f = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi) * dz

    def expit(x):
        return 1.0 / (1.0 + np.exp(-x))

    def intercept(b1):
        span = 8 * b1 + 40
        return _root(lambda b0: float(np.sum(f * expit(b0 + b1 * z))) - prevalence,
                     -span, span, 1e-12)

    def area(b1):
        p = expit(intercept(b1) + b1 * z)
        pos = f * p
        neg = f * (1 - p)
        below = np.cumsum(neg) - 0.5 * neg
        return float(np.sum(pos * below) / (np.sum(pos) * np.sum(neg)))

    b1 = _root(lambda b: area(b) - auc, 1e-4, 20, 1e-10)
    out = {"b0": intercept(b1), "b1": b1}
    _LINK_CACHE[key] = out
    return dict(out)


def _root(fn, lo, hi, tol):
    """A bracketed root by bisection, which is all the monotone link equations need."""
    flo = fn(lo)
    if flo * fn(hi) > 0:
        raise ValueError("the link equations have no root in their bracket")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        fm = fn(mid)
        if fm == 0:
            return mid
        if (fm < 0) == (flo < 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _whole(x, name, least):
    # R's as.integer(): a number is truncated to its whole part, as the R side reads it.
    try:
        v = int(float(x))
    except (TypeError, ValueError, OverflowError):
        v = None
    if v is None or v < least:
        raise ValueError(f"`{name}` must be a single whole number of at least {least}")
    return v


def _num(x):
    return f"{x:g}"
