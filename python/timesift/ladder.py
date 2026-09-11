"""Fit at every grain, see where skill saturates, and compare two arms cell by cell."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._stats import norm_ppf, t_ppf, wilcoxon_p
from .learners import fit_learner
from .metrics import tss
from .registry import RESPONSES, get_learner, resolve_metric
from .representation import timesift_set
from .response import (Folds, Response, align_folds, as_response, fold_map,
                       scorable_cells)

__all__ = ["Ladder", "concat_ladders", "grain_ladder", "implied_skill", "ladder_from_rows",
           "learner_dict", "mean_se", "out_of_fold", "paired_contrast", "per_variable", "place",
           "score_arm", "score_predictions",
           "scored_cells", "table_columns", "tss_inflation", "variable_means"]


@dataclass
class Ladder:
    """One score per ``(grain, learner, variable, fold)`` cell, and what produced it."""

    grain: np.ndarray
    learner: np.ndarray
    variable: np.ndarray
    fold: np.ndarray
    score: np.ndarray
    scorable: np.ndarray
    predictions: dict
    cells: object
    folds: Folds
    metric: str
    scorer: object
    response: str
    fits: dict

    def arm(self, name: str):
        """A mask over the rows of one grain-and-learner arm, named ``grain|learner``."""
        grain, learner = _split_arm(self, name)
        return (self.grain == grain) & (self.learner == learner)

    def summary(self):
        """The across-variable mean of the per-variable score, one row per arm."""
        level = variable_means(*_defined(self))
        rows = []
        for w in _ordered(self.grain):
            for ln in _ordered(self.learner):
                per = level.get((w, ln))
                if per is None:
                    continue
                rows.append(dict(learner=ln, grain=w,
                                 score=float(np.mean(list(per.values()))), n_variable=len(per)))
        for ln in {r["learner"] for r in rows}:
            same = [r for r in rows if r["learner"] == ln]
            best = max(same, key=lambda r: r["score"])
            for r in same:
                r["best"] = r is best
        return rows

    def __repr__(self) -> str:  # pragma: no cover - display only
        lines = [f"<timesift ladder> {len(set(self.grain))} grains x "
                 f"{len(set(self.learner))} learners",
                 f"metric: {self.metric} on {int(self.scorable.sum())} scorable cells"]
        for r in self.summary():
            lines.append(f"  {r['learner']:<10} {r['grain']:<10} {r['score']:.4f}"
                         f"{'  <- best' if r['best'] else ''}")
        return "\n".join(lines)


def grain_ladder(x, y, learners, folds=None, response: str = "presence_absence", metric=None,
                  control=None, keep_fits: bool = False, verbose: bool = True) -> Ladder:
    """Cross-validate every learner at every grain, on one fold map and one mask of cells.

    Every arm sees identical splits and is restricted to identical cells, so the arms' means share
    a denominator and any two of them can be compared with ``paired_contrast``.

    ``folds`` left at ``None`` builds one with the defaults of ``fold_map``. Where both languages
    must see the same splits, build it once and read it in the other with ``read_folds``.

    ``control`` is the ``train_control`` every neural learner of the ladder trains under; a learner
    carrying settings of its own overrides it on the ones it names.
    """
    grains = timesift_set(x)
    units = grains.units
    spec = RESPONSES.get(response)
    y = spec["prepare"](as_response(y)).align(units)
    if folds is None:
        folds = fold_map(y)
    folds = Folds.coerce(folds, units).align(units)
    f = folds.fold
    cells = spec["cells"](y, folds)
    score, metric_name = resolve_metric(metric, spec["metric"])
    learners = learner_dict(learners)
    levels = np.unique(f)

    grain, learner, variable, fold, value, ok = [], [], [], [], [], []
    predictions, fits = {}, {}
    for w, m in grains.items():
        for name, ln in learners.items():
            arm = f"{w}|{name}"
            if verbose:
                print(f"fitting {name} at the {w} grain")
            p, per_fold = out_of_fold(m, y, f, levels, ln, response, control, keep_fits,
                                      group=folds.group)
            for k, fit in per_fold.items():
                fits[f"{arm}|{k}"] = fit
            predictions[arm] = p
            scored = score_arm(w, name, y, p, f, levels, cells, score)
            for key, into in (("grain", grain), ("learner", learner), ("variable", variable),
                              ("fold", fold), ("score", value), ("scorable", ok)):
                into.extend(scored[key])

    return ladder_from_rows(dict(grain=grain, learner=learner, variable=variable, fold=fold,
                                 score=value, scorable=ok),
                            predictions=predictions, cells=cells,
                            folds=Folds(fold=f, units=units), metric=metric_name, scorer=score,
                            response=response, fits=fits)


def out_of_fold(m, y: Response, f: np.ndarray, levels, learner, response: str, control=None,
                keep_fits: bool = False, group=None):
    """One candidate over every fold, into the out-of-fold matrix the layers above read.

    The one fold loop of the package: a run, a ladder and the inner search of a selection all fit
    on the training units of a fold and predict its held-out ones, so they do it here rather than
    each carrying a copy. Nothing here knows whether the learner covers the responses jointly or
    one at a time; it is handed the whole response matrix either way.
    """
    p = np.full(y.values.shape, np.nan)
    fits = {}
    for k in levels:
        train = np.flatnonzero(f != k)
        held = m.take_units(np.flatnonzero(f == k))
        fit = fit_learner(learner, m.take_units(train), y.take_units(train), response=response,
                          control=control, group=_group_of(group, train))
        place(p, y, held.units, fit.variables, fit.predict(held))
        if keep_fits:
            fits[int(k)] = fit
    return p, fits


def _group_of(group, rows):
    """The grouping of the rows a fit is handed, or ``None`` where the map carries none."""
    return None if group is None else tuple(group[i] for i in rows)


def place(p: np.ndarray, y: Response, units, variables, predicted: np.ndarray) -> np.ndarray:
    """Write a block of predictions into the out-of-fold matrix, keyed on both axes.

    Keyed rather than positional: a learner returning its responses in another order would
    otherwise scramble which prediction belongs to which one, silently.
    """
    row = {u: i for i, u in enumerate(y.units)}
    column = {v: j for j, v in enumerate(y.variables)}
    for a, u in enumerate(units):
        for b, v in enumerate(variables):
            p[row[u], column[v]] = predicted[a, b]
    return p


def score_arm(grain, learner, y: Response, p: np.ndarray, f: np.ndarray, levels,
              cells, score) -> dict:
    """One arm's cells: which of them a score is defined on, and the score of each.

    A ladder and a selection both read a level off this, so the two report the same quantity
    computed the same way rather than each computing it.
    """
    rows = dict(grain=[], learner=[], variable=[], fold=[], score=[], scorable=[])
    unsettled = []
    for k in levels:
        take = f == k
        for j, v in enumerate(y.variables):
            ok = cells.is_scorable(v, int(k))
            if ok and not np.isfinite(p[take, j]).all():
                unsettled.append((v, int(k)))
            rows["grain"].append(grain)
            rows["learner"].append(learner)
            rows["variable"].append(v)
            rows["fold"].append(int(k))
            rows["scorable"].append(ok)
            rows["score"].append(score(y.values[take, j], p[take, j]) if ok else np.nan)
    _check_finite(unsettled, grain, learner)
    return rows


def _check_finite(unsettled, grain, learner) -> None:
    """A scorable cell is one every candidate has to carry a prediction for.

    A threshold metric returns NaN on a cell holding a prediction that is not a number, which is
    the same NaN a cell with no score in it returns, so a network whose training diverged would be
    reported as an unscorable cell and dropped from the combiner without a word. It is a fit that
    did not settle, and it stops the run where it happened.
    """
    if not unsettled:
        return
    arm = "the predictions hold" if grain is None and learner is None         else f"the {grain}|{learner} arm holds"
    plural = "s" if len(unsettled) > 1 else ""
    raise ValueError(f"{arm} no number on {len(unsettled)} scorable cell{plural}, first "
                     f"{unsettled[0][0]} in fold {unsettled[0][1]}. A scorable cell is one every "
                     "candidate carries, so this is a fit that did not settle rather than a cell "
                     "with no score in it.")


def score_predictions(y, p, folds, cells=None, metric: str = "tss") -> dict:
    """Score held-out predictions on the cells the mask allows.

    The scoring every arm of a ladder and every candidate of a run goes through, reachable on its
    own for a prediction matrix that came from somewhere else: a combination of arms, a model
    fitted outside the package, predictions read back from a file. A cell is one response in one
    fold, and the mask is computed from the response and the fold map alone, never from a model,
    which is what keeps two arms comparable.
    """
    y = as_response(y)
    f = align_folds(folds, y.units)
    if cells is None:
        cells = scorable_cells(y, f)
    rows = score_arm(None, None, y, np.asarray(p), f, np.unique(f), cells,
                     resolve_metric(metric)[0])
    return {k: v for k, v in rows.items() if k not in ("grain", "learner")}


def ladder_from_rows(rows: dict, predictions: dict, cells, folds: Folds, metric: str,
                     fits: dict, scorer=None, response: str = "presence_absence") -> Ladder:
    scorer = resolve_metric(metric)[0] if scorer is None else scorer
    return Ladder(grain=np.asarray(rows["grain"]), learner=np.asarray(rows["learner"]),
                  variable=np.asarray(rows["variable"]), fold=np.asarray(rows["fold"]),
                  score=np.asarray(rows["score"], dtype=float),
                  scorable=np.asarray(rows["scorable"]), predictions=predictions, cells=cells,
                  folds=folds, metric=metric, scorer=scorer, response=response, fits=fits)


def concat_ladders(a: Ladder, b: Ladder) -> Ladder:
    """Two arms' cells in one table, which is what a contrast between them reads."""
    if a.metric != b.metric:
        raise ValueError(f"one table is scored by {a.metric} and the other by {b.metric}")
    return Ladder(
        grain=np.concatenate([a.grain, b.grain]),
        learner=np.concatenate([a.learner, b.learner]),
        variable=np.concatenate([a.variable, b.variable]),
        fold=np.concatenate([a.fold, b.fold]),
        score=np.concatenate([a.score, b.score]),
        scorable=np.concatenate([a.scorable, b.scorable]),
        predictions={**a.predictions, **b.predictions}, cells=a.cells, folds=a.folds,
        metric=a.metric, scorer=a.scorer, response=a.response, fits={})


def variable_means(group, variable, value) -> dict:
    """Each group's mean within each variable, as ``{group: {variable: mean}}``.

    A variable is the independent replicate, so a cell mean is taken within a variable before
    anything is averaged across variables. Averaging cells directly would weight a variable by how
    many folds it happened to be scorable in. A ladder's arms, a selection's rows and a stack's
    candidates are all levelled off this, so the three report the same quantity computed the same
    way rather than each computing it.
    """
    acc: dict = {}
    for g, v, x in zip(group, variable, value):
        acc.setdefault(g, {}).setdefault(str(v), []).append(float(x))
    return {g: {v: float(np.mean(xs)) for v, xs in per.items()} for g, per in acc.items()}


def per_variable(ladder: Ladder) -> dict:
    """The mean score of each arm's variables over the folds it was scorable in."""
    level = variable_means(*_defined(ladder))
    return {(str(w), str(ln), str(v)): m
            for (w, ln), per in level.items() for v, m in per.items()}


def _defined(ladder: Ladder):
    """An arm-keyed view of the cells a score was defined on."""
    keep = ~np.isnan(ladder.score)
    return (list(zip(ladder.grain[keep], ladder.learner[keep])), ladder.variable[keep],
            ladder.score[keep])


def table_columns(table, names) -> dict:
    """A score table as arrays keyed by column, whether it arrived as a mapping or an object."""
    out = {}
    for name in names:
        holds = name in table if isinstance(table, dict) else hasattr(table, name)
        if not holds:
            raise KeyError(f'the table has no column called "{name}"')
        out[name] = np.asarray(table[name] if isinstance(table, dict) else getattr(table, name))
    return out


def scored_cells(scores):
    """The cells of a score table a level may be read from: scorable, and carrying a number.

    A fitted object's scores and a ladder's rows are the same table under two names, so the level
    of a candidate and the level of an arm are read off one rule.
    """
    table = table_columns(scores, ("candidate", "variable", "score", "scorable"))
    keep = table["scorable"].astype(bool) & np.isfinite(table["score"].astype(float))
    return table["candidate"][keep], table["variable"][keep], table["score"][keep]


def mean_se(values) -> tuple[float, float]:
    """A level and the spread of the variables it was averaged over, in one place, so a ladder and
    a selection report the same quantity computed the same way."""
    v = np.asarray([x for x in np.asarray(values, dtype=float) if not np.isnan(x)])
    if not len(v):
        return float("nan"), float("nan")
    se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
    return float(v.mean()), se


def paired_contrast(ladder: Ladder, a: str, b: str) -> dict:
    """The difference between two arms, taken inside each cell both scored.

    Two arms scored on the same held-out units do not necessarily have the same set of defined
    cells, so a difference of two marginal means is not a difference between the arms. Pairing also
    cancels what a threshold-selected metric carries in its level, since both arms carry the same
    bias on the same cell.

    Each arm is named whole, ``grain|learner``. A learner named alone would take its best grain,
    chosen on the scores the contrast is then read off, and pairing does not cancel that choice;
    ``select_grain`` chooses a grain on inner folds instead and contrasts the selection through
    ``compare``. The interval is Student's t on one degree of freedom fewer than there are
    variables, and ``p_method`` says whether the signed-rank p-value is ``"exact"`` or the
    ``"normal"`` approximation, which it is when the per-variable differences hold a zero or a tie
    or number fifty or more.
    """
    hit_a, hit_b = ladder.arm(a), ladder.arm(b)
    key = np.char.add(np.char.add(ladder.variable.astype(str), "|"), ladder.fold.astype(str))
    defined_a = hit_a & ~np.isnan(ladder.score)
    defined_b = hit_b & ~np.isnan(ladder.score)
    shared = np.intersect1d(key[defined_a], key[defined_b])
    if not len(shared):
        raise ValueError("the two arms share no cell both scored")

    ix_a = {k: i for i, k in zip(np.flatnonzero(defined_a), key[defined_a])}
    ix_b = {k: i for i, k in zip(np.flatnonzero(defined_b), key[defined_b])}
    diff = np.asarray([ladder.score[ix_a[k]] - ladder.score[ix_b[k]] for k in shared])
    variable = np.asarray([k.split("|")[0] for k in shared])

    by_variable = np.asarray([diff[variable == v].mean() for v in np.unique(variable)])
    n = len(by_variable)
    d, se = mean_se(by_variable)
    half = t_ppf(0.975, n - 1) * se if n > 1 else float("nan")
    p, method = _wilcoxon(by_variable)
    return dict(a=_label(ladder, a), b=_label(ladder, b), diff=d, lower=d - half,
                upper=d + half, n_variable=n, n_cell=len(shared),
                n_favour=int((by_variable > 0).sum()), p_value=p, p_method=method)


def tss_inflation(y: Response, folds, skill=(0.6, 0.7, 0.9), replicates: int = 200,
                  seed: int = 1) -> list[dict]:
    """How much a threshold chosen on the scored units inflates the level it reports.

    Predictions are simulated under a normal model whose population skill is exactly the value
    planted, at the cell sizes and presence counts of this design, and read back the way a ladder
    reports a level. The gap is the inflation. It cancels in a paired difference and does not
    cancel in a level, so a level is an upper bound on the skill a population has.
    """
    cells = scorable_cells(y, folds)
    keep = cells.scorable
    if not keep.any():
        raise ValueError("no cell of this design is scorable, so there is no level to measure")

    rng = np.random.default_rng(seed)
    out = []
    for target in skill:
        delta = 2 * norm_ppf((target + 1) / 2)
        per_replicate = np.empty(replicates)
        for r in range(replicates):
            by_variable = []
            for v in np.unique(cells.variable[keep]):
                rows = keep & (cells.variable == v)
                by_variable.append(np.mean([
                    tss(np.r_[np.ones(n_pos), np.zeros(n_neg)],
                        np.r_[rng.normal(delta, 1, n_pos), rng.normal(0, 1, n_neg)])
                    for n_pos, n_neg in zip(cells.pres_test[rows], cells.abs_test[rows])]))
            per_replicate[r] = float(np.mean(by_variable))
        out.append(dict(skill=target, reported=float(per_replicate.mean()),
                        inflation=float(per_replicate.mean()) - target,
                        lower=float(np.quantile(per_replicate, 0.025)),
                        upper=float(np.quantile(per_replicate, 0.975)),
                        replicates=replicates))
    return out


def implied_skill(y: Response, folds, observed, grid=None, replicates: int = 200,
                  seed: int = 1) -> list[dict]:
    """What population skill a level actually read is consistent with.

    ``tss_inflation`` maps a population skill to the level a design reports for it; this inverts
    that map. It is the only honest way to read a level as a statement about a population rather
    than about a scoring rule, and it says nothing about a difference between two arms, where the
    inflation cancels and the reported number stands as it is.
    """
    grid = np.arange(0, 0.96, 0.05) if grid is None else np.asarray(grid, dtype=float)
    forward = tss_inflation(y, folds, skill=grid, replicates=replicates, seed=seed)
    reported = np.asarray([r["reported"] for r in forward])
    skill = np.asarray([r["skill"] for r in forward])
    order = np.argsort(reported)
    reported, skill = reported[order], skill[order]
    observed = np.atleast_1d(np.asarray(observed, dtype=float))
    return [dict(observed=float(o), skill=float(np.interp(o, reported, skill)),
                 within_grid=bool(reported[0] <= o <= reported[-1])) for o in observed]


def learner_dict(learners):
    if isinstance(learners, dict):
        return {k: get_learner(v) for k, v in learners.items()}
    if not isinstance(learners, (list, tuple)):
        learners = [learners]
    out = {}
    for l in learners:
        l = get_learner(l)
        if l.name in out:
            raise ValueError(f'two learners are reported under the name "{l.name}". '
                             "Pass a dict to tell them apart.")
        out[l.name] = l
    return out


def _split_arm(ladder: Ladder, name: str):
    """The grain and the learner an arm names, matched whole against the labels the ladder
    carries rather than split at a ``|``, so a grain or a learner whose own name holds one is
    still found. A learner named alone is refused: its best grain would be chosen on the scores a
    contrast is then read off."""
    for grain, learner in zip(ladder.grain, ladder.learner):
        if f"{grain}|{learner}" == name:
            return str(grain), str(learner)
    whole = [f"{g}|{l}" for g, l in zip(ladder.grain, ladder.learner) if str(l) == name]
    if whole:
        raise KeyError(f'"{name}" names a learner and no grain. Its best grain would be chosen on '
                       f'the scores the contrast is read off, so name the arm whole, as '
                       f'"{whole[0]}", or let select_grain choose the grain on inner folds and '
                       f'contrast the selection through its `compare`')
    raise KeyError(f'no arm called "{name}" in this ladder')


def _best_arm(ladder: Ladder, name: str) -> str:
    """The arm a learner named alone is read at: its best grain on the ladder's own scores. That is
    the natural arm for describing a fitted model, and never the arm a contrast is computed on."""
    if any(f"{g}|{l}" == name for g, l in zip(ladder.grain, ladder.learner)):
        return name
    rows = [r for r in ladder.summary() if r["learner"] == name and r["score"] == r["score"]]
    return f'{max(rows, key=lambda r: r["score"])["grain"]}|{name}' if rows else name


def _label(ladder: Ladder, name: str) -> str:
    grain, learner = _split_arm(ladder, name)
    return f"{grain}|{learner}"


def _ordered(values):
    seen = []
    for v in values:
        if v not in seen:
            seen.append(v)
    return seen


def _wilcoxon(values) -> tuple[float, str | None]:
    if len(values) < 2 or not np.any(values != 0):
        return float("nan"), None
    return wilcoxon_p(values)
