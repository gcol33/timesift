"""The ladder, the run and the selection, drawn.

The twin of R's ``plot()`` methods on a ladder, a run and a selection, drawn by matplotlib, which
is what this module needs beyond numpy, as the encoders need torch. Every number a plot draws is on
the object it is called on, and each returns the table it drew from, as R's return theirs, so a
figure is never the only place a number can be read.

The default colours are R's ``hcl.colors(n, "Dark 3")``, computed from the same polar LUV
coordinates, so a curve is the same colour on both sides.
"""

from __future__ import annotations

import importlib.util

import numpy as np

from ._stats import t_ppf
from .fit import Timesift
from .ladder import Ladder, mean_se, per_variable, scored_cells, variable_means
from .selection import Selection

__all__ = ["plot"]


def plot(x, col=None, interval: bool = True, ax=None, **kwargs) -> list[dict]:
    """Draw a ladder, a run or a selection, and return the table the drawing was made from.

    A ladder is drawn as one line per learner across the grains, at the across-variable mean of
    the per-variable score, with a 95 percent interval from its standard error across variables on
    Student's t with one degree of freedom fewer than there are variables; an open circle marks
    each learner's best grain. A run is drawn the same way across the representations, with the
    stack's held-out score across them as a dashed line: the curves are scored on the folds a
    choice among them would be judged on, so the best of them sits a little high, and the stack's
    line does not. A selection is drawn as every candidate's inner score in every outer fold, one
    line per fold, an open circle on the candidate the fold chose.

    ``col`` is one colour per line, recycled. ``ax`` is the matplotlib axes to draw on, a new
    figure's where it is left unset, and ``kwargs`` reach ``ax.set()``, so ``title=`` or
    ``ylim=`` are set as they would be there. ``interval`` draws the interval on a ladder or a run
    and is ignored on a selection, which draws none. Needs matplotlib.
    """
    if importlib.util.find_spec("matplotlib") is None:
        raise ImportError("`plot()` needs matplotlib. Install it with pip install matplotlib")
    if isinstance(x, Ladder):
        return _plot_ladder(x, col, interval, ax, **kwargs)
    if isinstance(x, Timesift):
        return _plot_run(x, col, interval, ax, **kwargs)
    if isinstance(x, Selection):
        return _plot_selection(x, col, ax, **kwargs)
    raise TypeError(f"plot() draws a grain_ladder(), a timesift() or a select_grain() result, "
                    f"got {type(x).__name__}")


def _plot_ladder(x: Ladder, col, interval, ax, **kwargs):
    per = per_variable(x)
    arm = [k[1] for k in per]
    level = [k[0] for k in per]
    stat = _curve_stats(arm, level, list(per.values()), arms=_unique(x.learner),
                        levels=_unique(x.grain))
    _draw_curves(stat, col, interval, "grain", x.metric, ax, None, **kwargs)
    return [dict(learner=s["arm"], grain=s["level"], score=s["score"], se=s["se"]) for s in stat]


def _plot_run(x: Timesift, col, interval, ax, **kwargs):
    candidate, variable, score = scored_cells(x.scores)
    keep = np.asarray([str(c) != "ensemble" for c in candidate], dtype=bool)
    level = variable_means(candidate[keep], variable[keep], score[keep])
    if not level:
        raise ValueError("this run scored no cell, so there is nothing to draw.")
    table = x.candidates
    order = [str(c) for c in table["candidate"]]
    representation = dict(zip(order, (str(v) for v in table["representation"])))
    learner = dict(zip(order, (str(v) for v in table["learner"])))
    names = [c for c in order if c in level]
    arm, lev, val = [], [], []
    for c in names:
        for _, m in sorted(level[c].items()):
            arm.append(learner[c])
            lev.append(representation[c])
            val.append(m)
    stat = _curve_stats(arm, lev, val, arms=_unique(learner[c] for c in names),
                        levels=_unique(representation[c] for c in names))
    rule = None
    for row in x.estimate or []:
        if row.get("arm") == "ensemble" and row["metric"] == x.metric:
            rule = row["score"]
            break
    _draw_curves(stat, col, interval, "representation", x.metric, ax, rule, **kwargs)
    return [dict(learner=s["arm"], representation=s["level"], score=s["score"], se=s["se"])
            for s in stat]


def _plot_selection(x: Selection, col, ax, **kwargs):
    import matplotlib.pyplot as plt

    label = [_label(c["grain"], c["learner"]) for c in x.candidates]
    position = {name: i + 1 for i, name in enumerate(label)}
    chosen = {(r["fold"], _label(r["grain"], r["learner"])) for r in x.selected}
    inner = [dict(row, at=position[_label(row["grain"], row["learner"])],
                  selected=(row["fold"], _label(row["grain"], row["learner"])) in chosen)
             for row in x.inner]
    folds = sorted({r["fold"] for r in inner})
    colours = _colours(col, len(folds))
    if ax is None:
        _, ax = plt.subplots()
    finite = [r["score"] for r in inner if np.isfinite(r["score"])]
    if finite:
        ax.set_ylim(min(finite), max(finite))
    ax.set_xticks(range(1, len(label) + 1))
    ax.set_xticklabels(label, rotation=90, fontsize="small")
    ax.set_xlim(0.5, len(label) + 0.5)
    ax.set_ylabel(f"{x.metric} inside the training data")
    ax.yaxis.grid(True, color="#E5E5E5")
    ax.set_axisbelow(True)
    for k, fold in enumerate(folds):
        rows = sorted((r for r in inner if r["fold"] == fold), key=lambda r: r["at"])
        at = [r["at"] for r in rows]
        sc = [r["score"] for r in rows]
        ax.plot(at, sc, color=colours[k], lw=1.5)
        ax.plot(at, sc, "o", color=colours[k], ms=4)
        won = [r for r in rows if r["selected"]]
        ax.plot([r["at"] for r in won], [r["score"] for r in won], "o", mfc="none",
                mec=colours[k], ms=13, mew=2)
    if kwargs:
        ax.set(**kwargs)
    return inner


def _curve_stats(arm, level, score, arms, levels):
    """A level and the spread around it for every arm at every level, in the layout both curves
    draw from, so a ladder and a run report the same quantity computed the same way."""
    arm = np.asarray(arm, dtype=object)
    level = np.asarray(level, dtype=object)
    score = np.asarray(score, dtype=float)
    out = []
    for a in arms:
        for w in levels:
            v = score[(arm == a) & (level == w)]
            m, se = mean_se(v)
            n = int(np.sum(~np.isnan(v)))
            half = t_ppf(0.975, max(n - 1, 1)) * se if n > 1 else float("nan")
            out.append(dict(arm=a, level=w, score=m, se=se, half=half))
    return out


def _draw_curves(stat, col, interval, xlabel, ylabel, ax, rule, **kwargs):
    import matplotlib.pyplot as plt

    levels = _unique(s["level"] for s in stat)
    arms = _unique(s["arm"] for s in stat)
    colours = _colours(col, len(arms))
    ruled = rule is not None and np.isfinite(rule)
    if ax is None:
        _, ax = plt.subplots()

    score = np.array([s["score"] for s in stat], dtype=float)
    half = np.array([s["half"] for s in stat], dtype=float)
    if interval and np.isfinite(half).any():
        span = np.concatenate([score - half, score + half])
    else:
        span = score
    span = span[np.isfinite(span)]
    if ruled:
        span = np.append(span, rule)
    if len(span):
        low, high = float(span.min()), float(span.max())
        pad = 0.04 * (high - low) if high > low else 0.01
        ax.set_ylim(low - pad, high + pad)
    at_of = {w: i + 1 for i, w in enumerate(levels)}
    ax.set_xticks(list(at_of.values()))
    ax.set_xticklabels(levels)
    ax.set_xlim(0.5, len(levels) + 0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color="#E5E5E5")
    ax.set_axisbelow(True)
    if ruled:
        ax.axhline(rule, ls="--", color="#666666", lw=2, label="ensemble")

    for k, a in enumerate(arms):
        rows = {s["level"]: s for s in stat if s["arm"] == a}
        at = np.array([at_of[w] for w in levels], dtype=float)
        sc = np.array([rows[w]["score"] if w in rows else np.nan for w in levels], dtype=float)
        hf = np.array([rows[w]["half"] if w in rows else np.nan for w in levels], dtype=float)
        ok = np.isfinite(hf) & (hf > 0)
        if interval and ok.any():
            ax.errorbar(at[ok], sc[ok], yerr=hf[ok], fmt="none", ecolor=colours[k], capsize=3,
                        lw=1)
        ax.plot(at, sc, color=colours[k], lw=2, label=a)
        ax.plot(at, sc, "o", color=colours[k], ms=5)
        if np.isfinite(sc).any():
            best = int(np.nanargmax(sc))
            ax.plot(at[best], sc[best], "o", mfc="none", mec=colours[k], ms=15, mew=2)
    if len(arms) + int(ruled) > 1:
        ax.legend(loc="lower left", frameon=False)
    if kwargs:
        ax.set(**kwargs)
    return ax


def _colours(col, n):
    if col is None:
        return hcl_palette(max(n, 2))[:n]
    col = [col] if isinstance(col, str) else list(col)
    return [col[i % len(col)] for i in range(n)]


def hcl_palette(n: int) -> list[str]:
    """R's ``hcl.colors(n, "Dark 3")``: ``n`` hues evenly round the circle from 0 degrees at
    chroma 80 and luminance 60, converted from polar LUV to sRGB as R's ``hcl()`` converts them."""
    return [_hcl_hex(h, 80.0, 60.0) for h in np.linspace(0.0, 360.0 * (n - 1) / n, n)]


def _hcl_hex(h, c, lum):
    white_y, white_u, white_v = 100.0, 0.1978398, 0.4683363
    hr = np.deg2rad(h)
    u_ = c * np.cos(hr)
    v_ = c * np.sin(hr)
    y = white_y * (((lum + 16) / 116) ** 3 if lum > 7.999592 else lum / 903.3)
    u = u_ / (13 * lum) + white_u
    v = v_ / (13 * lum) + white_v
    x = 9.0 * y * u / (4 * v)
    z = -x / 3 - 5 * y + 3 * y / v
    rgb = ((3.240479 * x - 1.537150 * y - 0.498535 * z) / white_y,
           (-0.969256 * x + 1.875992 * y + 0.041556 * z) / white_y,
           (0.055648 * x - 0.204043 * y + 1.057311 * z) / white_y)

    def gamma(t):
        return 12.92 * t if t <= 0.0031308 else 1.055 * t ** (1 / 2.4) - 0.055

    return "#" + "".join(f"{int(round(255 * min(1.0, max(0.0, gamma(t))))):02X}" for t in rgb)


def _label(grain, learner) -> str:
    return f"{grain}|{learner}"


def _unique(values) -> list:
    return list(dict.fromkeys(str(v) for v in values))
