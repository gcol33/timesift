"""Every grain against a learner's best one, on a mixed model of the per-cell scores.

The twin of R's ``grain_contrasts()``, which fits ``score ~ grain + (1 | variable) + (1 | fold)``
by restricted maximum likelihood in lme4, reads Satterthwaite's degrees of freedom off lmerTest
and Dunnett's many-to-one comparisons off emmeans. The model is written out here rather than
borrowed, since the Python scientific stack has no fitter of crossed random effects the wheel could
lean on; the multivariate t it reads the comparisons against is scipy's, which is the one thing
this function needs beyond numpy, as R's needs its three packages.

The fit follows lme4: the random effects are parametrised by ``theta``, each one's standard
deviation relative to the residual one, and the restricted deviance is minimised over ``theta``
with the residual variance profiled out. The degrees of freedom follow lmerTest: the covariance of
``(theta, sigma)`` is twice the inverse Hessian of the restricted deviance there, and a contrast's
degrees of freedom are ``2 v^2 / (g' A g)``, with ``v`` its variance and ``g`` the gradient of that
variance in the same parameters. The adjustment follows emmeans' ``"mvt"``: each contrast's
degrees of freedom are read as ``floor(df + 0.25)``, the p-value is one minus the probability that
every contrast's statistic lies inside the observed one's, and the interval's critical value is the
0.95 point of the largest absolute statistic. Both are integrals of a multivariate t, so both carry
the Monte Carlo error of the integrator, in R as here, of the order of a thousandth.
"""

from __future__ import annotations

import importlib.util

import numpy as np

from .ladder import Ladder

__all__ = ["grain_contrasts"]

# The integrator's own randomisation is seeded, so one ladder returns one table.
_SEED = 20260925
_MAXPTS = 200_000


def grain_contrasts(ladder: Ladder, learner: str | None = None, reference: str | None = None,
                    adjust: str = "mvt") -> list[dict]:
    """Compare every grain against the reference grain of one learner.

    A mixed model on the learner's per-cell scores, ``score ~ grain + (1 | variable) +
    (1 | fold)``, fitted by restricted maximum likelihood, and every grain compared against
    ``reference`` by Dunnett's many-to-one procedure. The design is balanced across variables,
    folds and grains, so each grain is compared within a variable and within a fold, and the
    variation between variables cancels from the comparison.

    ``learner`` is the only one in the ladder by default, and ``reference`` the learner's best
    grain. Taking the best-observed grain as the reference favours it, so read this beside
    :func:`paired_contrast`, which singles out no grain. ``adjust`` is ``"mvt"``, the adjustment
    R's default reads, or ``"none"``.

    Returns one row per grain other than the reference: ``learner``, ``grain``, ``reference``,
    the difference from the reference ``diff``, its interval ``lower`` and ``upper``, and the
    adjusted ``p_value``. Needs scipy.
    """
    if importlib.util.find_spec("scipy") is None:
        raise ImportError("`grain_contrasts()` needs scipy. Install it with pip install scipy")
    if not isinstance(ladder, Ladder):
        raise TypeError(f"expected a grain_ladder() result, got {type(ladder).__name__}")
    if adjust not in ("mvt", "none"):
        raise ValueError(f'`adjust` is "mvt" or "none", got "{adjust}"')
    arms = list(dict.fromkeys(str(v) for v in ladder.learner))
    if learner is None:
        if len(arms) != 1:
            raise ValueError(f"this ladder holds {len(arms)} learners; name the one to fit: "
                             f"{', '.join(arms)}.")
        learner = arms[0]
    mine = np.asarray([str(v) == learner for v in ladder.learner])
    score = np.asarray(ladder.score, dtype=float)
    keep = mine & ~np.isnan(score)
    if not keep.any():
        raise ValueError(f'no scored cell for learner "{learner}".')
    grains = list(dict.fromkeys(str(g) for g in np.asarray(ladder.grain)[mine]))
    if len(grains) < 2:
        raise ValueError(f"a grain contrast needs at least two grains, this ladder has "
                         f"{len(grains)}.")
    if reference is None:
        best = [r["grain"] for r in ladder.summary() if r["learner"] == learner and r["best"]]
        if len(best) != 1:
            raise ValueError(f'no grain of learner "{learner}" has a score to take as the '
                             f"reference. Name `reference`.")
        reference = best[0]
    if reference not in grains:
        raise ValueError(f'"{reference}" is not a grain of this ladder.')

    levels = [reference] + [g for g in grains if g != reference]
    grain = np.asarray([str(g) for g in np.asarray(ladder.grain)[keep]])
    variable = np.asarray([str(v) for v in np.asarray(ladder.variable)[keep]])
    fold = np.asarray([str(f) for f in np.asarray(ladder.fold)[keep]])
    fit = _reml(score[keep], _treatment(grain, levels), [variable, fold])

    # Treatment coding makes every grain's coefficient its difference from the reference, which is
    # the trt.vs.ctrl contrast of the marginal means: a grain below the reference reads negative.
    k = len(levels) - 1
    est = fit["beta"][1:]
    cov = fit["vcov"][1:, 1:]
    se = np.sqrt(np.diag(cov))
    df = np.array([_satterthwaite(fit, np.eye(len(levels))[j + 1]) for j in range(k)])
    t = est / se
    corr = cov / np.outer(se, se)
    p, crit = _adjusted(t, df, corr, adjust)
    return [dict(learner=learner, grain=levels[j + 1], reference=reference, diff=float(est[j]),
                 lower=float(est[j] - crit[j] * se[j]), upper=float(est[j] + crit[j] * se[j]),
                 p_value=float(p[j])) for j in range(k)]


def _treatment(grain, levels):
    x = np.zeros((len(grain), len(levels)))
    x[:, 0] = 1.0
    for j, g in enumerate(levels[1:], start=1):
        x[:, j] = grain == g
    return x


def _reml(y, x, factors):
    """lme4's restricted fit of independent random intercepts, one per grouping factor.

    With ``V = sigma^2 (I + Z L L' Z')`` and ``L`` the diagonal of relative standard deviations,
    every quantity the deviance needs is a product of ``Z'Z``, ``Z'X`` and ``Z'y`` with matrices
    the size of the random effects, so no matrix the size of the data is formed.
    """
    from scipy.optimize import minimize

    blocks = []
    for f in factors:
        names, index = np.unique(f, return_inverse=True)
        z = np.zeros((len(y), len(names)))
        z[np.arange(len(y)), index] = 1.0
        blocks.append(z)
    z = np.hstack(blocks)
    sizes = [b.shape[1] for b in blocks]
    parts = dict(ztz=z.T @ z, ztx=z.T @ x, zty=z.T @ y, xtx=x.T @ x, xty=x.T @ y, yty=y @ y,
                 n=len(y), p=x.shape[1], sizes=sizes)

    def profiled(theta):
        return _deviance(parts, theta, None)[0]

    # The deviance reads each theta through its square, so zero is a stationary point of every one
    # of them whether or not it is the optimum, and a search bounded below by zero stops there. The
    # search is therefore unbounded, from lme4's start, and the sign it lands on is dropped: a
    # derivative-free simplex, as lme4's bobyqa is, polished by a gradient search.
    simplex = minimize(profiled, np.ones(len(factors)), method="Nelder-Mead",
                       options=dict(xatol=1e-10, fatol=1e-13, maxiter=20000, maxfev=20000))
    polish = minimize(profiled, simplex.x, method="BFGS", options=dict(gtol=1e-10))
    best = polish if polish.fun <= simplex.fun else simplex
    theta = np.abs(np.asarray(best.x, dtype=float))
    _, beta, sigma2, xtwx = _deviance(parts, theta, None)
    sigma = float(np.sqrt(sigma2))
    varpar = np.concatenate([theta, [sigma]])
    return dict(parts=parts, varpar=varpar, beta=beta,
                vcov=sigma2 * np.linalg.inv(xtwx),
                acov=2.0 * np.linalg.inv(_hessian(lambda v: _deviance(parts, v[:-1], v[-1])[0],
                                                  varpar)))


def _deviance(parts, theta, sigma):
    """The restricted deviance at ``theta``, at ``sigma`` or with it profiled out, and the fixed
    effects and ``X' V^-1 X`` (on the scale of ``sigma^2``) it was read at."""
    lam = np.repeat(np.asarray(theta, dtype=float), parts["sizes"])
    m = np.eye(len(lam)) + lam[:, None] * parts["ztz"] * lam[None, :]
    chol = np.linalg.cholesky(m)
    lzx = np.linalg.solve(chol, lam[:, None] * parts["ztx"])
    lzy = np.linalg.solve(chol, lam * parts["zty"])
    xtwx = parts["xtx"] - lzx.T @ lzx
    xtwy = parts["xty"] - lzx.T @ lzy
    ytwy = parts["yty"] - lzy @ lzy
    beta = np.linalg.solve(xtwx, xtwy)
    rwr = float(ytwy - beta @ xtwy)
    n, p = parts["n"], parts["p"]
    sigma2 = rwr / (n - p) if sigma is None else float(sigma) ** 2
    logdet_m = 2.0 * np.sum(np.log(np.diag(chol)))
    logdet_x = np.linalg.slogdet(xtwx)[1]
    dev = (n - p) * np.log(2 * np.pi * sigma2) + logdet_m + logdet_x + rwr / sigma2
    return float(dev), beta, sigma2, xtwx


def _satterthwaite(fit, contrast):
    """lmerTest's degrees of freedom of one contrast of the fixed effects."""
    parts = fit["parts"]

    def variance(v):
        _, _, _, xtwx = _deviance(parts, v[:-1], v[-1])
        return float(v[-1] ** 2 * contrast @ np.linalg.solve(xtwx, contrast))

    v = float(contrast @ fit["vcov"] @ contrast)
    g = _gradient(variance, fit["varpar"])
    return 2.0 * v * v / float(g @ fit["acov"] @ g)


def _step(x):
    return 1e-4 * np.maximum(np.abs(x), 1e-2)


def _gradient(fn, x):
    """Central differences with one Richardson step, which is what numDeriv reads a Jacobian by."""
    x = np.asarray(x, dtype=float)
    h = _step(x)
    out = np.empty(len(x))
    for i in range(len(x)):
        e = np.zeros(len(x))
        e[i] = h[i]
        d1 = (fn(x + e) - fn(x - e)) / (2 * h[i])
        d2 = (fn(x + e / 2) - fn(x - e / 2)) / h[i]
        out[i] = (4 * d2 - d1) / 3
    return out


def _hessian(fn, x):
    """Central second differences with one Richardson step."""
    x = np.asarray(x, dtype=float)
    h = _step(x)
    k = len(x)
    out = np.empty((k, k))

    def second(i, j, s):
        ei = np.zeros(k)
        ej = np.zeros(k)
        ei[i] = h[i] * s
        ej[j] = h[j] * s
        return (fn(x + ei + ej) - fn(x + ei - ej) - fn(x - ei + ej) + fn(x - ei - ej)) / \
            (4 * h[i] * h[j] * s * s)

    for i in range(k):
        for j in range(i, k):
            out[i, j] = out[j, i] = (4 * second(i, j, 0.5) - second(i, j, 1.0)) / 3
    return out


def _fix_df(d):
    """emmeans' reading of a degree of freedom before it reaches the integrator: at least one, and
    zero, which the integrator reads as the normal, above 9999 or where it is infinite."""
    if d > 0:
        d = max(1.0, d)
    if not np.isfinite(d) or d > 9999:
        return 0
    return int(np.floor(d + 0.25))


def _adjusted(t, df, corr, adjust):
    from scipy import stats
    from scipy.optimize import brentq

    k = len(t)
    fixed = [_fix_df(d) for d in df]
    if adjust == "none" or k == 1:
        p = np.empty(k)
        crit = np.empty(k)
        for j in range(k):
            dist = stats.norm if fixed[j] == 0 else stats.t(fixed[j])
            p[j] = 2 * dist.sf(abs(t[j]))
            crit[j] = dist.ppf(0.975)
        return p, crit

    def inside(c, d):
        upper = np.full(k, c)
        if d == 0:
            return float(stats.multivariate_normal.cdf(upper, mean=np.zeros(k), cov=corr,
                                                       lower_limit=-upper, maxpts=_MAXPTS,
                                                       abseps=1e-5, rng=_SEED))
        return float(stats.multivariate_t.cdf(upper, loc=np.zeros(k), shape=corr, df=d,
                                              lower_limit=-upper, maxpts=_MAXPTS,
                                              random_state=_SEED))

    p = np.array([min(1.0, max(0.0, 1.0 - inside(abs(t[j]), fixed[j]))) for j in range(k)])
    crit = np.empty(k)
    for d in set(fixed):
        dist = stats.norm if d == 0 else stats.t(d)
        lo = dist.ppf(0.975)
        hi = dist.ppf(1 - 0.025 / k) + 0.5
        c = brentq(lambda c: inside(c, d) - 0.95, lo, hi, xtol=1e-6)
        crit[[j for j in range(k) if fixed[j] == d]] = c
    return p, crit
