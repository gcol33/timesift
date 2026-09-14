"""What a fitted ``timesift`` says about itself.

Two tables. The candidates, what each of them was best at and how it covered the responses, read
off the scores on the outer folds: the comparison. The procedure, the selected candidate and the
stack, read off the held-out predictions whose choice and weights were made inside each outer
training fold: the level to report. And the weights fitted on every target, for prediction.
"""

from __future__ import annotations

import numpy as np

from .ladder import Ladder, scored_cells, table_columns, variable_means
from .occlusion import ladder_occlusion, occlusion_profile
from .response import align_folds, as_response

__all__ = ["candidate_table", "ensemble_weights", "occlusion", "procedure_table", "summary"]

_MISSING = object()


def summary(fit) -> str:
    """The fit as text: the candidates on the outer folds, the procedure, and the weights."""
    rows = list(candidate_table(fit))
    procedure = procedure_table(fit)
    width = max([len("candidate")] + [len(r["candidate"]) for r in rows + procedure])

    def line(candidate, mean, won, responses):
        return f"{candidate:<{width}} {mean:>14} {won:>6}  {responses}".rstrip()

    lines = [_header(fit), "", "candidates, scored on the outer folds",
             line("candidate", "mean", "won", "responses")]
    for r in rows:
        lines.append(line(r["candidate"],
                          "not applicable" if r["mean"] is None else f"{r['mean']:.3f}",
                          "-" if r["won"] is None else str(r["won"]), r["responses"]))
    if procedure:
        lines += ["", "procedure, chosen and weighted inside each outer training fold"]
        for r in procedure:
            lines.append(f"{r['candidate']:<{width}} {r['mean']:>14.3f}  se {r['se']:.3f}")
        counts = _selection_counts(_field(fit, "selected", None))
        if counts:
            lines.append("selected " + ", ".join(f"{n} in {c}" for n, c in counts)
                         + f" of {sum(c for _, c in counts)} folds")

    choice = _field(fit, "choice", None)
    if choice is not None:
        lines += ["", f"choice on every target  {choice}"]
    weights = ensemble_weights(fit)
    if weights:
        # A member whose weight rounds to nothing is not a member of the combination in any way a
        # reader can act on; ensemble_weights() still carries every one of them.
        shown = sorted(((n, w) for n, w in weights.items() if w >= 0.005),
                       key=lambda kv: (-kv[1], kv[0]))
        lines.append("weights on every target  "
                     + "   ".join(f"{n} {w:.2f}" for n, w in shown))
    return "\n".join(lines)


def _selection_counts(selected) -> list[tuple[str, int]]:
    """How often each candidate was chosen across the outer folds, most often first."""
    if not selected:
        return []
    counts: dict = {}
    for row in selected:
        counts[row["candidate"]] = counts.get(row["candidate"], 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])


def _header(fit) -> str:
    y = as_response(_field(fit, "y"))
    v = len(np.unique(align_folds(_field(fit, "folds"), y.units)))
    kind = "grouped" if getattr(_field(fit, "folds"), "grouped", False) else "random"
    return (f"timesift  {_plural(y.values.shape[0], 'target')}, "
            f"{_plural(y.values.shape[1], 'response')}, {v}-fold {kind} CV, "
            f"{_field(fit, 'metric')}")


def _plural(n: int, what: str) -> str:
    return f"{n} {what}" if n == 1 else f"{n} {what}s"


def candidate_table(fit) -> list[dict]:
    """One row per candidate: its level, how many responses it was best on, and how it covered
    them.

    A candidate whose learner and representation could not be paired carries no level, and is
    listed under the ones that do.
    """
    names, multi = _candidates(fit)
    level = variable_means(*scored_cells(_field(fit, "scores")))
    won = _wins(level)
    rows = [dict(candidate=n, responses=multi[n],
                 won=won.get(n, 0) if n in level else None,
                 mean=float(np.mean(list(level[n].values()))) if n in level else None)
            for n in names]
    scored = [r for r in rows if r["mean"] is not None]
    scored.sort(key=lambda r: (r["mean"], r["candidate"]))
    return scored + [r for r in rows if r["mean"] is None]


def procedure_table(fit) -> list[dict]:
    """The selected candidate's and the stack's held-out level under the run's own metric, with
    the standard error across responses: one row each, or none where the run made no estimate."""
    estimate = _field(fit, "estimate", None)
    if not estimate:
        return []
    metric = _field(fit, "metric")
    return [dict(candidate=row["arm"], mean=row["score"], se=row["se"])
            for row in estimate if row["metric"] == metric]


def ensemble_weights(fit) -> dict | None:
    """The weight the combiner gave each of its members, or nothing where a run combined none."""
    stack = _field(fit, "stack", None)
    return None if stack is None else stack.weights


def occlusion(x, *args, **kwargs):
    """What one candidate's score loses when a bin, or a channel, is withheld from it.

    Takes a :func:`~timesift.fit.timesift` run or a :func:`~timesift.ladder.grain_ladder` result,
    and the profile itself is one implementation either way. The models kept per fold are the ones
    read, so the profile is measured where the score was: on the units each model held out.
    """
    if isinstance(x, Ladder):
        return ladder_occlusion(x, *args, **kwargs)
    return _run_occlusion(x, *args, **kwargs)


def _run_occlusion(fit, candidate: str, over: str = "bin", substitute: str = "permute",
                   metric=None, permutations: int = 20, seed: int = 1):
    m = _representation(fit, candidate)
    return occlusion_profile(_kept_fits(fit, candidate), m, _field(fit, "y"),
                             _field(fit, "folds"), over=over, substitute=substitute,
                             metric=_field(fit, "scorer") if metric is None else metric,
                             response=_field(fit, "response"),
                             permutations=permutations, seed=seed)


def _kept_fits(fit, candidate: str) -> dict:
    fits = _field(fit, "fits", None)
    if not fits:
        raise ValueError("this fit kept no model per fold, and an occlusion profile reads the "
                         "models on the units they held out; refit with "
                         "timesift(..., keep_fits=True)")
    prefix = f"{candidate}|"
    kept = {int(str(key)[len(prefix):]): model for key, model in fits.items()
            if str(key).startswith(prefix)}
    if not kept:
        raise KeyError(f'this fit kept no model for the candidate "{candidate}"')
    return kept


def _representation(fit, candidate: str):
    column = _candidate_column(fit, "representation")
    if str(candidate) not in column:
        raise KeyError(f'no candidate called "{candidate}" in this fit. The candidates are '
                       f'{", ".join(column)}.')
    built = _field(fit, "representations")
    name = column[str(candidate)]
    if name not in built:
        raise KeyError(f'this fit carries no representation called "{name}"')
    return built[name]


def _candidates(fit):
    multi = _candidate_column(fit, "multi")
    return list(multi), multi


def _candidate_column(fit, name: str) -> dict:
    """One column of the candidates table, keyed by candidate and in the table's own row order."""
    columns = table_columns(_field(fit, "candidates"), ("candidate", name))
    return {str(c): str(v) for c, v in zip(columns["candidate"], columns[name])}


def _wins(level: dict) -> dict:
    """How many responses each candidate was highest on, ties going to the first name in C
    order."""
    by_response: dict = {}
    for name, per in level.items():
        for variable, mean in per.items():
            by_response.setdefault(variable, {})[str(name)] = mean
    won: dict = {}
    for scored in by_response.values():
        best, top = None, -np.inf
        for name in sorted(scored):
            if scored[name] > top:
                best, top = name, scored[name]
        won[best] = won.get(best, 0) + 1
    return won


def _field(fit, name: str, default=_MISSING):
    value = fit.get(name, default) if isinstance(fit, dict) else getattr(fit, name, default)
    if value is _MISSING:
        raise AttributeError(f'this fitted object carries no "{name}"')
    return value
