"""The distributions the package reads, written out rather than imported.

Each fits in a page and each is needed in one place, so carrying a scientific computing stack for
them would cost every install more than they are worth. The normal quantile is Wichura's AS241,
which is the algorithm R's ``qnorm`` uses. Student's t quantile is solved on the closed-form
distribution function a whole number of degrees of freedom has, to the precision R's ``qt``
reports. The signed-rank test follows ``wilcox.test`` in taking the exact distribution below fifty
values with no tie and no zero and the normal approximation with continuity and tie corrections
otherwise, and returns which of the two it took, so a p-value read here and a p-value read on the R
side agree and say the same thing about how they were read.
"""

from __future__ import annotations

import math

import numpy as np

_A = (3.3871328727963666080e0, 1.3314166789178437745e2, 1.9715909503065514427e3,
      1.3731693765509461125e4, 4.5921953931549871457e4, 6.7265770927008700853e4,
      3.3430575583588128105e4, 2.5090809287301226727e3)
_B = (1.0, 4.2313330701600911252e1, 6.8718700749205790830e2, 5.3941960214247511077e3,
      2.1213794301586595867e4, 3.9307895800092710610e4, 2.8729085735721942674e4,
      5.2264952788528545610e3)
_C = (1.42343711074968357734e0, 4.63033784615654529590e0, 5.76949722146069140550e0,
      3.64784832476320460504e0, 1.27045825245236838258e0, 2.41780725177450611770e-1,
      2.27238449892691845833e-2, 7.74545014278341407640e-4)
_D = (1.0, 2.05319162663775882187e0, 1.67638483018380384940e0, 6.89767334985100004550e-1,
      1.48103976427480074590e-1, 1.51986665636164571966e-2, 5.47593808499534494600e-4,
      1.05075007164441684324e-9)
_E = (6.65790464350110377720e0, 5.46378491116411436990e0, 1.78482653991729133580e0,
      2.96560571828504891230e-1, 2.65321895265761230930e-2, 1.24266094738807843860e-3,
      2.71155556874348757815e-5, 2.01033439929228813265e-7)
_F = (1.0, 5.99832206555887937690e-1, 1.36929880922735805310e-1, 1.48753612908506148525e-2,
      7.86869131145613259100e-4, 1.84631831751005468180e-5, 1.42151175831644588870e-7,
      2.04426310338993978564e-15)


def norm_ppf(p: float) -> float:
    """The standard normal quantile, to full double precision (Wichura, AS241)."""
    if not 0 < p < 1:
        raise ValueError(f"a probability must lie strictly between 0 and 1, got {p}")
    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        return q * _poly(_A, r) / _poly(_B, r)
    r = p if q < 0 else 1 - p
    r = np.sqrt(-np.log(r))
    if r <= 5:
        r -= 1.6
        value = _poly(_C, r) / _poly(_D, r)
    else:
        r -= 5
        value = _poly(_E, r) / _poly(_F, r)
    return -value if q < 0 else value


def _poly(coefs, r):
    out = 0.0
    for c in reversed(coefs):
        out = out * r + c
    return out


def t_ppf(p: float, df: int) -> float:
    """Student's t quantile on a whole number of degrees of freedom.

    For a whole number of degrees of freedom the distribution function is a finite sum
    (Abramowitz and Stegun 26.7.3 and 26.7.4), and the quantile is solved on it by Newton's
    method from the normal quantile, which lies below it. The function is concave beyond zero, so
    every step lands short of the root and the iteration climbs to it without overshooting.
    """
    if not 0 < p < 1:
        raise ValueError(f"a probability must lie strictly between 0 and 1, got {p}")
    if df < 1 or int(df) != df:
        raise ValueError(f"the degrees of freedom are a whole number of at least one, got {df}")
    df = int(df)
    if p < 0.5:
        return -t_ppf(1 - p, df)
    if p == 0.5:
        return 0.0
    target = 2 * p - 1
    log_scale = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
    t = norm_ppf(p)
    for _ in range(200):
        density = math.exp(log_scale - (df + 1) / 2 * math.log1p(t * t / df))
        step = (_t_central(t, df) - target) / (2 * density)
        t -= step
        if abs(step) <= 1e-15 * t:
            break
    return t


def _t_central(t: float, df: int) -> float:
    """P(|T| < t) for t >= 0, as the finite sum a whole number of degrees of freedom gives."""
    theta = math.atan(t / math.sqrt(df))
    c2 = math.cos(theta) ** 2
    if df % 2:
        if df == 1:
            return 2 * theta / math.pi
        term = total = math.cos(theta)
        for k in range(1, (df - 1) // 2):
            term *= c2 * (2 * k) / (2 * k + 1)
            total += term
        return 2 / math.pi * (theta + math.sin(theta) * total)
    term = total = 1.0
    for k in range(1, df // 2):
        term *= c2 * (2 * k - 1) / (2 * k)
        total += term
    return math.sin(theta) * total


def wilcoxon_p(values) -> tuple[float, str | None]:
    """Two-sided p-value of the signed-rank test on one sample against a centre of zero, and the
    method it was read by: ``"exact"`` or ``"normal"``."""
    values = np.asarray(values, dtype=float)
    x = values[values != 0]
    n = len(x)
    if n < 1:
        return float("nan"), None
    ranks = _average_ranks(np.abs(x))
    statistic = float(ranks[x > 0].sum())
    ties = len(np.unique(np.abs(x))) != n

    if n < 50 and not ties and n == len(values):
        counts = _signed_rank_counts(n)
        total = 2.0 ** n
        below = counts[:int(statistic) + 1].sum() / total
        above = counts[int(statistic):].sum() / total
        return float(min(1.0, 2 * min(below, above))), "exact"

    expected = n * (n + 1) / 4
    _, tie_counts = np.unique(np.abs(x), return_counts=True)
    correction = (tie_counts ** 3 - tie_counts).sum() / 48
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24 - correction)
    if sigma == 0:
        return float("nan"), None
    z = (statistic - expected - np.sign(statistic - expected) * 0.5) / sigma
    return float(min(1.0, 2 * _norm_sf(abs(z)))), "normal"


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or sorted_values[i] != sorted_values[start]:
            ranks[order[start:i]] = (start + i + 1) / 2
            start = i
    return ranks


def _signed_rank_counts(n: int) -> np.ndarray:
    """How many of the 2**n sign assignments give each possible rank sum."""
    counts = np.zeros(n * (n + 1) // 2 + 1)
    counts[0] = 1
    for r in range(1, n + 1):
        counts[r:] = counts[r:] + counts[:-r]
    return counts


def _norm_sf(z: float) -> float:
    """The upper tail of the standard normal, through the complementary error function."""
    return 0.5 * _erfc(z / np.sqrt(2))


def _erfc(x: float) -> float:
    # Numerical Recipes' Chebyshev fit, accurate to about 1.2e-7 relative, which is far below the
    # precision a p-value is read at.
    z = abs(x)
    t = 2.0 / (2.0 + z)
    ty = 4.0 * t - 2.0
    coefs = (-1.3026537197817094, 6.4196979235649026e-1, 1.9476473204185836e-2,
             -9.561514786808631e-3, -9.46595344482036e-4, 3.66839497852761e-4,
             4.2523324806907e-5, -2.0278578112534e-5, -1.624290004647e-6,
             1.303655835580e-6, 1.5626441722e-8, -8.5238095915e-8,
             6.529054439e-9, 5.059343495e-9, -9.91364156e-10,
             -2.27365122e-10, 9.6467911e-11, 2.394038e-12,
             -6.886027e-12, 8.94487e-13, 3.13092e-13,
             -1.12708e-13, 3.81e-16, 7.106e-15)
    d, dd = 0.0, 0.0
    for c in reversed(coefs[1:]):
        d, dd = ty * d - dd + c, d
    value = t * np.exp(-z * z + 0.5 * (coefs[0] + ty * d) - dd)
    return value if x >= 0 else 2.0 - value
