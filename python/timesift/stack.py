"""The ensemble: how several candidates' out-of-fold predictions become one prediction.

The combiner never sees a model. It is handed the out-of-fold predictions, the response, the mask
of scorable cells and the fold map, and that is what keeps it honest: a weight cannot be earned on
a cell the candidate that carries it was fitted on.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .ladder import scored_cells, variable_means
from .registry import METRICS, RESPONSES
from .response import align_folds, as_response

__all__ = ["CLAMP", "EnsembleSpec", "METHODS", "SCOPES", "STACK_LOSSES", "Stack", "as_ensemble",
           "candidate_means", "ensemble", "ensemble_combine", "ensemble_fit", "in_scope",
           "run_ensemble", "simplex_weights", "stack_loss"]

METHODS = ("stack", "mean", "median", "weighted")
SCOPES = ("all", "learners", "representations")

# A candidate is named for the learner and the representation it pairs, and the ensemble reads the
# pair back out of the name to keep a scope on one of the two axes.
SEPARATOR = " / "

# The loss is unbounded where a combined prediction reaches zero or one, so it is read at most this
# far into either corner. Both languages read it at the same distance, or their weights would be
# minimising two different functions.
CLAMP = 1e-7


def _clamp(p: np.ndarray) -> np.ndarray:
    return np.clip(p, CLAMP, 1.0 - CLAMP)


# A loss reaches the solver as its value and its derivative in the combined prediction, both
# averaged over the cells, so the solver is the same forty lines whatever the response head is.
STACK_LOSSES = {
    "binary_cross_entropy": dict(
        value=lambda p, y: -float(np.mean(y * np.log(_clamp(p)) + (1 - y)
                                          * np.log1p(-_clamp(p)))),
        gradient=lambda p, y: (_clamp(p) - y) / (_clamp(p) * (1 - _clamp(p))) / len(y)),
    "squared_error": dict(
        value=lambda p, y: float(np.mean((p - y) ** 2)),
        gradient=lambda p, y: 2 * (p - y) / len(y)),
}


@dataclass(frozen=True)
class EnsembleSpec:
    """How the candidates are to be combined, which of them are eligible, and under which head."""

    method: str = "stack"
    scope: str = "all"
    metric: object = None
    response: str | None = None


@dataclass(frozen=True)
class Stack:
    """A fitted combiner: what it does and what weight it gave each of its members."""

    method: str
    weights: dict
    members: tuple[str, ...]

    def __repr__(self) -> str:  # pragma: no cover - display only
        shown = sorted(self.weights.items(), key=lambda kv: (-kv[1], kv[0]))
        return (f"<timesift stack> {self.method} over {len(self.members)} candidates\n"
                + "  ".join(f"{k} {v:.3f}" for k, v in shown))


def ensemble(method: str = "stack", scope: str = "all", metric=None,
             response: str | None = None) -> EnsembleSpec:
    """Ask for an ensemble of the candidates a fit produced.

    ``stack`` fits non-negative weights summing to one on the out-of-fold predictions, ``mean`` and
    ``median`` combine without fitting, and ``weighted`` uses each candidate's own mean score
    rescaled to sum to one. ``scope`` is which candidates are eligible: every one of them, only the
    several learners sharing the best candidate's representation, or only its learner across the
    representations. ``metric`` names the metric the ensemble is reported in, or ``None`` for the
    fit's own, and ``response`` is the registered head whose loss the weights minimise, or ``None``
    for the head the run was fitted under. Naming a head the run does not fit toward is an error
    rather than an override, and :func:`ensemble_fit` called on its own reads ``None`` as
    ``"presence_absence"``.
    """
    if method not in METHODS:
        raise ValueError(f"`method` is one of {', '.join(METHODS)}, got {method!r}")
    if scope not in SCOPES:
        raise ValueError(f"`scope` is one of {', '.join(SCOPES)}, got {scope!r}")
    if metric is not None:
        METRICS.get(metric)
    if response is not None:
        RESPONSES.get(response)
    return EnsembleSpec(method=method, scope=scope, metric=metric, response=response)


def run_ensemble(spec, response: str) -> EnsembleSpec | None:
    """The spec a run combines under.

    The run was fitted toward one response head and the combiner minimises that head's loss, so
    the run's response is what reaches the spec. A spec naming another head is a contradiction
    rather than an override: it would stack an abundance run under a presence-absence loss.
    """
    spec = as_ensemble(spec)
    if spec is None or spec.response == response:
        return spec
    if spec.response is None:
        return replace(spec, response=response)
    raise ValueError(f"the run fits the {response} response and ensemble(response="
                     f"{spec.response!r}) names another. The combiner minimises the loss of the "
                     f"head the run was fitted under.")


def as_ensemble(x) -> EnsembleSpec | None:
    """An ``EnsembleSpec``, whether it was asked for as one, as a method name, or as a yes or no."""
    if x is None or x is False:
        return None
    if x is True:
        return ensemble()
    if isinstance(x, EnsembleSpec):
        return x
    if isinstance(x, str):
        return ensemble(method=x)
    raise TypeError("`ensemble` is an ensemble(), a method name, True or False")


def stack_loss(response: str) -> dict:
    """The loss the combiner minimises under one registered response head.

    The head says what it is trained under and the combiner minimises that same thing, so adding a
    response is a registration here too rather than a second branch in the solver.
    """
    name = RESPONSES.get(response)["loss"]
    if name not in STACK_LOSSES:
        raise ValueError(f'the {response} response is trained under "{name}", which the combiner '
                         f"cannot minimise. It knows {', '.join(sorted(STACK_LOSSES))}.")
    return STACK_LOSSES[name]


def ensemble_fit(oof: dict, y, cells, folds, spec=None, scores=None) -> Stack:
    """Fit the combiner on the out-of-fold predictions and nothing else.

    ``oof`` is one ``[target, response]`` matrix per candidate, in the response's own row order.
    Only the cells the mask admits are read, so every candidate is weighted on the same cells its
    score was read on.
    """
    spec = as_ensemble(ensemble() if spec is None else spec)
    if spec is None:
        raise ValueError("`spec` asks for no ensemble, so there is nothing to fit")
    # Fitted from a run, the spec arrives carrying the run's head; fitted on its own, there is no
    # run to read one off and the shipped default is the one the package defaults to everywhere.
    if spec.response is None:
        spec = replace(spec, response="presence_absence")
    y = as_response(y)
    members = in_scope(tuple(oof), spec.scope, scores)
    if len(members) < 2:
        raise ValueError(f"an ensemble needs at least two candidates, and the {spec.scope} scope "
                         f"leaves {len(members)}")
    if spec.method == "stack":
        p, observed = _stacking_block({name: oof[name] for name in members}, y, cells, folds)
        w = simplex_weights(p, observed, stack_loss(spec.response))
    elif spec.method == "weighted":
        w = _score_weights(members, scores)
    else:
        w = np.full(len(members), 1.0 / len(members))
    return Stack(method=spec.method, members=members,
                 weights={name: float(value) for name, value in zip(members, w)})


def ensemble_combine(stack: Stack, preds: dict) -> np.ndarray:
    """One ``[n, response]`` matrix from each member's ``[n, response]`` matrix."""
    missing = [m for m in stack.members if m not in preds]
    if missing:
        raise KeyError(f"{len(missing)} member{'s have' if len(missing) > 1 else ' has'} no "
                       f"prediction to combine, first: {missing[0]}")
    p = np.stack([np.asarray(preds[m], dtype=np.float64) for m in stack.members])
    if stack.method == "median":
        return np.median(p, axis=0)
    w = np.asarray([stack.weights[m] for m in stack.members], dtype=np.float64)
    return np.tensordot(w, p, axes=(0, 0))


def candidate_means(scores) -> dict:
    """Each candidate's level: the mean within each response first, then across responses."""
    return {name: float(np.mean(list(per.values())))
            for name, per in variable_means(*scored_cells(scores)).items()}


def in_scope(names, scope: str, scores) -> tuple[str, ...]:
    """The candidates a scope leaves eligible, in the order they were offered.

    ``learners`` keeps the several learners sharing the representation of the best candidate and
    ``representations`` keeps the one learner across its representations, so an ensemble under
    either scope varies one axis and holds the other.
    """
    if scope == "all":
        return tuple(names)
    if scope not in SCOPES:
        raise ValueError(f"`scope` is one of {', '.join(SCOPES)}, got {scope!r}")
    axis = 1 if scope == "learners" else 0
    held = _split_candidate(_best(names, scores))[axis]
    return tuple(n for n in names if _split_candidate(n)[axis] == held)


def simplex_weights(p: np.ndarray, y: np.ndarray, loss: dict, iterations: int = 500,
                    tol: float = 1e-12) -> np.ndarray:
    """Non-negative weights summing to one that minimise a loss of the mixture ``p w``.

    ``p`` is one column of predictions per member and ``y`` the observed values they are read
    against. Every loss the combiner knows is convex in the mixture, so the minimum over the
    simplex is one point and an exponentiated-gradient loop walks to it: the multiplicative update
    keeps every weight positive and the renormalisation keeps the sum at one, so the iterate never
    leaves the simplex and no projection step is needed.

    The step is halved until the loss falls, which makes the sequence of losses monotone and the
    stopping point the same on every machine, and the gradient is divided by its largest entry, so
    a step means the same thing whatever scale the loss is on. The loop stops when a step buys less
    than ``tol`` of the loss it is on.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    k = p.shape[1]
    if k < 1:
        raise ValueError("there is nothing to weight")
    w = np.full(k, 1.0 / k)
    value = loss["value"](p @ w, y)
    step = 1.0
    for _ in range(iterations):
        g = p.T @ loss["gradient"](p @ w, y)
        largest = float(np.max(np.abs(g)))
        if not np.isfinite(largest) or largest <= 0:
            break
        g = g / largest
        while True:
            moved_to = w * np.exp(-step * g)
            moved_to = moved_to / moved_to.sum()
            moved = loss["value"](p @ moved_to, y)
            if np.isfinite(moved) and moved <= value:
                break
            step *= 0.5
            if step < 1e-12:
                break
        if step < 1e-12:
            break
        gain = value - moved
        w, value = moved_to, moved
        if gain <= tol * max(1.0, abs(value)):
            break
        step *= 1.5
    return w


def _stacking_block(oof: dict, y, cells, folds):
    """The cells every candidate is weighted on: one column per member, one row per scorable cell
    that carries a prediction from all of them."""
    read = np.stack([_matrix(p, y, name) for name, p in oof.items()])
    keep = _scorable_mask(cells, y, align_folds(folds, y.units)) & np.isfinite(read).all(axis=0)
    if not keep.any():
        raise ValueError("no scorable cell carries a prediction from every candidate")
    return read[:, keep].T, y.values[keep]


def _matrix(p, y, name: str) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if p.shape != y.values.shape:
        raise ValueError(f'"{name}" predicts a {p.shape[0]} by {p.shape[1]} block for a '
                         f"{y.values.shape[0]} by {y.values.shape[1]} response")
    return p


def _scorable_mask(cells, y, f: np.ndarray) -> np.ndarray:
    """The ``[target, response]`` cells a score is defined on, read off the same mask every
    candidate was scored against."""
    admits = {(str(v), int(k)): bool(ok)
              for v, k, ok in zip(cells.variable, cells.fold, cells.scorable)}
    return np.array([[admits.get((str(v), int(k)), False) for v in y.variables] for k in f])


def _levels(scores, why: str) -> dict:
    if scores is None:
        raise ValueError(f"{why} reads the candidates' own scores, and none were given")
    return candidate_means(scores)


def _score_weights(members, scores) -> np.ndarray:
    level = _levels(scores, "the weighted method")
    w = np.asarray([max(level.get(name, 0.0), 0.0) for name in members], dtype=np.float64)
    if w.sum() <= 0:
        raise ValueError("no candidate scores above zero, so there is nothing to weight them by")
    return w / w.sum()


def _best(names, scores) -> str:
    level = _levels(scores, "a scope on one axis")
    best, top = None, -np.inf
    for name in sorted(names):
        value = level.get(name, -np.inf)
        if value > top:
            best, top = name, value
    if best is None:
        raise ValueError("no candidate carries a score to read a scope off")
    return best


def _split_candidate(name: str) -> tuple[str, str]:
    learner, _, representation = str(name).partition(SEPARATOR)
    return learner, representation
