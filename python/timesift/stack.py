"""The ensemble: how several candidates' out-of-fold predictions become one prediction.

The combiner never sees a model. It is handed the out-of-fold predictions, the response, the mask
of scorable cells and the fold map, and that is what keeps it honest: a weight cannot be earned on
a cell the candidate that carries it was fitted on.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ._stats import t_ppf
from .ladder import score_arm, scored_cells, variable_means
from .registry import METRICS, RESPONSES
from .response import align_folds, as_response

__all__ = ["CLAMP", "EnsembleSpec", "METHODS", "SCOPES", "SPREAD_STATISTICS", "STACK_LOSSES",
           "Stack", "as_ensemble", "candidate_means", "ensemble", "ensemble_combine",
           "ensemble_fit", "ensemble_spread", "in_scope", "run_ensemble", "score_weights",
           "simplex_weights", "stack_loss"]

METHODS = ("stack", "mean", "median", "weighted", "committee")
SCOPES = ("all", "learners", "representations")
RULES = ("youden", "kappa", "prevalence")
SPREAD_STATISTICS = ("mean", "sd", "cv", "lower", "upper")

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
        range=(0.0, 1.0),
        value=lambda p, y: -float(np.mean(y * np.log(_clamp(p)) + (1 - y)
                                          * np.log1p(-_clamp(p)))),
        gradient=lambda p, y: (_clamp(p) - y) / (_clamp(p) * (1 - _clamp(p))) / len(y)),
    "squared_error": dict(
        range=(-np.inf, np.inf),
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
    min_score: float | None = None
    decay: object = None
    rule: str | None = None


@dataclass(frozen=True)
class Stack:
    """A fitted combiner: what it does and what weight it gave each of its members.

    A committee also carries each member's cut on each response, ``thresholds[member, variable]``
    in the response's own variable order, NaN where a member holds no cut.
    """

    method: str
    weights: dict
    members: tuple[str, ...]
    loss: str = "binary_cross_entropy"
    thresholds: np.ndarray | None = None
    variables: tuple[str, ...] | None = None

    def __repr__(self) -> str:  # pragma: no cover - display only
        shown = sorted(self.weights.items(), key=lambda kv: (-kv[1], kv[0]))
        return (f"<timesift stack> {self.method} over {len(self.members)} candidates\n"
                + "  ".join(f"{k} {v:.3f}" for k, v in shown))


def ensemble(method: str = "stack", scope: str = "all", metric=None,
             response: str | None = None, min_score: float | None = None, decay=None,
             rule: str | None = None) -> EnsembleSpec:
    """Ask for an ensemble of the candidates a fit produced.

    ``stack`` fits non-negative weights summing to one on the out-of-fold predictions, ``mean`` and
    ``median`` combine without fitting, and ``weighted`` uses each candidate's own mean score,
    its positive part rescaled to sum to one; where no candidate scores above zero every candidate
    weighs the same. ``decay``, biomod2's ``EMwmean.decay``, changes that: given a number ``d``,
    the ``K`` candidates scoring above zero take ``d**K`` for the best down to ``d**1`` for the
    ``K``-th, candidates on the same score share the mean of their ranks' weights, and a candidate
    at or below zero takes none.

    ``committee`` is biomod2's committee averaging: each member cuts each response at the
    threshold ``decision_threshold()`` learns under ``rule`` from that member's out-of-fold
    predictions of every target, and the combination is the share of members voting presence. A
    member holding no cut on a response does not vote on it.

    ``min_score``, biomod2's ``metric.select.thresh``, leaves a candidate whose mean score is below
    it ineligible, before ``scope`` picks among what is left. ``scope`` is which candidates are
    eligible: every one of them, only the several learners sharing the best candidate's
    representation, or only its learner across the representations. ``metric`` names the metric
    the eligibility, ``min_score`` and the weighted weights are read by, or ``None`` for the scores
    the run already carries, and ``response`` is the registered head whose loss the weights
    minimise, or ``None`` for the head the run was fitted under. Naming a head the run does not fit
    toward is an error rather than an override, and :func:`ensemble_fit` called on its own reads
    ``None`` as ``"presence_absence"``.
    """
    if method not in METHODS:
        raise ValueError(f"`method` is one of {', '.join(METHODS)}, got {method!r}")
    if scope not in SCOPES:
        raise ValueError(f"`scope` is one of {', '.join(SCOPES)}, got {scope!r}")
    if metric is not None:
        METRICS.get(metric)
    if response is not None:
        RESPONSES.get(response)
    if min_score is not None and (isinstance(min_score, bool)
                                  or not isinstance(min_score, (int, float))
                                  or not np.isfinite(min_score)):
        raise ValueError("`min_score` is one finite number, or None for no filter")
    if decay is not None:
        if method != "weighted":
            raise ValueError(f'`decay` sets how the "weighted" method turns scores into weights, '
                             f'and the method here is "{method}"')
        if decay != "proportional" and (isinstance(decay, bool)
                                        or not isinstance(decay, (int, float))
                                        or not np.isfinite(decay) or decay < 1):
            raise ValueError('`decay` is "proportional" or one number of at least one')
    if rule is not None:
        if method != "committee":
            raise ValueError(f"`rule` sets where each member of a committee cuts its prediction, "
                             f'and the method here is "{method}"')
        if rule not in RULES:
            raise ValueError(f"`rule` is one of {', '.join(RULES)}, got {rule!r}")
    return EnsembleSpec(method=method, scope=scope, metric=metric, response=response,
                        min_score=None if min_score is None else float(min_score),
                        decay=(decay or "proportional") if method == "weighted" else None,
                        rule=(rule or "youden") if method == "committee" else None)


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

    The weights are fitted to the response on those predictions, so the combination scored against
    the same response is scored on the data its weights were fitted to, and that score is
    optimistic. ``timesift`` evaluates the stack the other way: each outer fold's weights are
    fitted on inner out-of-fold predictions of its training targets and applied to the outer test
    fold.
    """
    spec = as_ensemble(ensemble() if spec is None else spec)
    if spec is None:
        raise ValueError("`spec` asks for no ensemble, so there is nothing to fit")
    # Fitted from a run, the spec arrives carrying the run's head; fitted on its own, there is no
    # run to read one off and the shipped default is the one the package defaults to everywhere.
    if spec.response is None:
        spec = replace(spec, response="presence_absence")
    y = as_response(y)
    level = _member_levels(oof, y, cells, folds, spec, scores)
    members = in_scope(_passing(tuple(oof), level, spec.min_score), spec.scope, level)
    if len(members) < 2:
        raise ValueError(f"an ensemble needs at least two candidates, and the {spec.scope} scope "
                         f"leaves {len(members)}")
    p, observed = _stacking_block({name: oof[name] for name in members}, y, cells, folds)
    loss_name = RESPONSES.get(spec.response)["loss"]
    thresholds = None
    if spec.method == "stack":
        w = simplex_weights(p, observed, stack_loss(spec.response))
    elif spec.method == "weighted":
        w = score_weights([level[name] for name in members], spec.decay or "proportional")
    else:
        w = np.full(len(members), 1.0 / len(members))
    if spec.method == "committee":
        thresholds = _member_thresholds({name: oof[name] for name in members}, y,
                                        spec.rule or "youden")
    return Stack(method=spec.method, members=members,
                 weights={name: float(value) for name, value in zip(members, w)},
                 loss=loss_name, thresholds=thresholds, variables=tuple(y.variables))


def ensemble_combine(stack: Stack, preds: dict) -> np.ndarray:
    """One ``[n, response]`` matrix from each member's ``[n, response]`` matrix."""
    p = _member_block(stack, preds)
    w = np.asarray([stack.weights[m] for m in stack.members], dtype=np.float64)
    if stack.method == "median":
        return np.median(p, axis=0)
    if stack.method == "committee":
        return _committee_share(p, w, stack.thresholds)
    return np.tensordot(w, p, axes=(0, 0))


def ensemble_spread(stack: Stack, preds: dict, alpha: float = 0.05) -> np.ndarray:
    """How far the members of an ensemble disagree: biomod2's ``EMcv`` and ``EMci``.

    Returns an ``[n, response, statistic]`` array, the statistics being ``SPREAD_STATISTICS``: the
    weighted mean ``m`` of the members' predictions under the stack's weights; their weighted
    standard deviation ``s``, the square root of ``sum(w (p - m)**2) / (1 - sum(w**2))``, which is
    the sample standard deviation when the weights are equal; the coefficient of variation
    ``s / m``; and the interval ``m -+ t(1 - alpha / 2, n - 1) s sqrt(sum(w**2))``, ``n`` the
    number of members carrying weight, the t interval of a mean of ``n`` members when the weights
    are equal. Under a head whose predictions are probabilities the interval is held inside zero
    and one. A committee's and a median's members are read at equal weight, and a committee's
    spread is that of the members' predictions rather than of their votes. ``sd``, ``cv`` and the
    interval are NaN where fewer than two members carry weight.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ValueError("`alpha` is one number strictly between 0 and 1")
    p = _member_block(stack, preds)
    k = p.shape[0]
    if stack.method in ("median", "committee"):
        w = np.full(k, 1.0 / k)
    else:
        w = np.asarray([stack.weights[m] for m in stack.members], dtype=np.float64)
        w = w / w.sum()
    m = np.tensordot(w, p, axes=(0, 0))
    sq = float(np.sum(w ** 2))
    n = int(np.sum(w > 0))
    if n >= 2:
        s = np.sqrt(np.tensordot(w, (p - m) ** 2, axes=(0, 0)) / (1 - sq))
        half = t_ppf(1 - alpha / 2, n - 1) * s * np.sqrt(sq)
    else:
        s = np.full(m.shape, np.nan)
        half = s
    low, high = STACK_LOSSES[stack.loss]["range"]
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = s / m
    return np.stack([m, s, cv, np.maximum(m - half, low), np.minimum(m + half, high)], axis=-1)


def _member_block(stack: Stack, preds: dict) -> np.ndarray:
    """The members' predictions stacked in the stack's order, ``[member, n, response]``."""
    missing = [m for m in stack.members if m not in preds]
    if missing:
        raise KeyError(f"{len(missing)} member{'s have' if len(missing) > 1 else ' has'} no "
                       f"prediction to combine, first: {missing[0]}")
    p = [np.asarray(preds[m], dtype=np.float64) for m in stack.members]
    wrong = [m for m, one in zip(stack.members, p) if one.shape != p[0].shape]
    if wrong:
        raise ValueError(f"every member's prediction must have the same shape; "
                         f"{', '.join(wrong)} does not")
    return np.stack(p)


def _committee_share(p: np.ndarray, w: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Each response's vote: the weighted share of the members holding a cut on it that read
    presence, ``p >= cut``. A response no member holds a cut on predicts NaN."""
    out = np.full(p.shape[1:], np.nan)
    for j in range(p.shape[2]):
        voting = np.isfinite(thresholds[:, j])
        if not voting.any():
            continue
        share = w[voting] / w[voting].sum()
        votes = (p[voting, :, j] >= thresholds[voting, j][:, None]).astype(np.float64)
        out[:, j] = share @ votes
    return out


def _member_thresholds(oof: dict, y, rule: str) -> np.ndarray:
    """A committee member's cut on each response, learned from its own out-of-fold predictions of
    every target as ``decision_threshold()`` learns a fit's: ``[member, variable]``."""
    from .metrics import decision_threshold
    return np.array([[decision_threshold(y.values[:, j], np.asarray(p, dtype=np.float64)[:, j],
                                         rule)
                      for j in range(len(y.variables))] for p in oof.values()],
                    dtype=np.float64)


def candidate_means(scores) -> dict:
    """Each candidate's level: the mean within each response first, then across responses."""
    return {name: float(np.mean(list(per.values())))
            for name, per in variable_means(*scored_cells(scores)).items()}


def in_scope(names, scope: str, level: dict) -> tuple[str, ...]:
    """The candidates a scope leaves eligible, in the order they were offered.

    ``level`` is each candidate's mean score, as :func:`candidate_means` reads it. ``learners``
    keeps the several learners sharing the representation of the best candidate and
    ``representations`` keeps the one learner across its representations, so an ensemble under
    either scope varies one axis and holds the other.
    """
    if scope == "all":
        return tuple(names)
    if scope not in SCOPES:
        raise ValueError(f"`scope` is one of {', '.join(SCOPES)}, got {scope!r}")
    axis = 1 if scope == "learners" else 0
    held = _split_candidate(_best(names, level))[axis]
    return tuple(n for n in names if _split_candidate(n)[axis] == held)


def score_weights(score, decay="proportional") -> np.ndarray:
    """biomod2's EMwmean: weights from the members' mean scores, summing to one.

    Proportional weights are the positive part of each score. With a number for ``decay``, the
    ``K`` members scoring above zero take ``decay**K`` for the best down to ``decay**1`` for the
    ``K``-th, members on exactly the same score share the mean of their ranks' weights, and a
    member at or below zero takes none. Where no member scores above zero every member weighs the
    same.
    """
    score = np.asarray([s if np.isfinite(s) else 0.0 for s in score], dtype=np.float64)
    positive = score > 0
    if not positive.any():
        return np.full(len(score), 1.0 / len(score))
    if decay == "proportional":
        w = np.maximum(score, 0.0)
    else:
        k = int(positive.sum())
        rank_weight = np.zeros(len(score))
        top = np.argsort(-score, kind="stable")[:k]
        rank_weight[top] = float(decay) ** (k - np.arange(k))
        w = np.zeros(len(score))
        for value in np.unique(score[positive]):
            tied = score == value
            w[tied] = rank_weight[tied].mean()
    return w / w.sum()


def simplex_weights(p: np.ndarray, y: np.ndarray, loss: dict, iterations: int = 500,
                    tol: float = 1e-14) -> np.ndarray:
    """Non-negative weights summing to one that minimise a loss of the mixture ``p w``.

    ``p`` is one column of predictions per member and ``y`` the observed values they are read
    against. Every loss the combiner knows is convex in the mixture, so the minimum over the
    simplex is one point and an exponentiated-gradient loop walks to it: the multiplicative update
    keeps every weight positive and the renormalisation keeps the sum at one, so the iterate never
    leaves the simplex and no projection step is needed.

    The step is halved until the loss falls, which makes the sequence of losses monotone and the
    stopping point the same on every machine, and the gradient is divided by its largest entry, so
    a step means the same thing whatever scale the loss is on. The loop stops when a step buys less
    than ``tol`` of the loss it is on. Near the minimum the loss is flat to second order in the
    weights, so the stop settles the gradient to a precision of about the square root of ``tol``.
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
            # A step grown past what exp() can hold gives inf / inf, which the finiteness check
            # below refuses and halves, so the overflow is the step being too long and not an error.
            with np.errstate(over="ignore", invalid="ignore"):
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
    """The cells every candidate is weighted on: one column per member, one row per scorable cell.

    Every scorable cell, and no other. The mask says which cells every candidate was scored on and
    the combiner is fitted on exactly those; dropping the ones a candidate left without a number
    would fit the stack on fewer cells than the report says it was, and the report would not show
    it.
    """
    read = np.stack([_matrix(p, y, name) for name, p in oof.items()])
    keep = _scorable_mask(cells, y, align_folds(folds, y.units))
    if not keep.any():
        raise ValueError("no cell of this design is scorable, so there is nothing to fit the "
                         "combiner on")
    missing = [(name, int((~np.isfinite(read[j][keep])).sum()))
               for j, name in enumerate(oof)]
    missing = [one for one in missing if one[1]]
    if missing:
        raise ValueError("the combiner is fitted on every scorable cell, and "
                         + ", ".join(f"{name} ({n})" for name, n in missing)
                         + " hold no number on some of them. A candidate that cannot predict a "
                         "scorable cell is a fit that did not settle; drop it from the run rather "
                         "than from the cells.")
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


def _member_levels(oof: dict, y, cells, folds, spec: EnsembleSpec, scores) -> dict:
    """Each candidate's mean score: recomputed from the out-of-fold predictions where the spec
    names its own metric, and read off the run's own scores where it does not. A combination that
    weighs no candidate against another needs neither, and gets NaN for every one."""
    if spec.metric is not None:
        score = METRICS.get(spec.metric)
        f = align_folds(folds, y.units)
        table = dict(candidate=[], variable=[], score=[], scorable=[])
        for name, p in oof.items():
            got = score_arm("", name, y, _matrix(p, y, name), f, np.unique(f), cells, score)
            table["candidate"] += [name] * len(got["variable"])
            for key in ("variable", "score", "scorable"):
                table[key] += list(got[key])
        level = candidate_means(table)
    elif scores is not None:
        level = candidate_means(scores)
    elif spec.scope != "all" or spec.method == "weighted" or spec.min_score is not None:
        raise ValueError(f"a {spec.method} combination over the {spec.scope} candidates weighs "
                         f"them by their score, which is read off `scores` or recomputed from a "
                         f"metric named in ensemble(metric=). Neither was given.")
    else:
        level = {}
    return {name: level.get(name, float("nan")) for name in oof}


def _passing(names, level: dict, min_score) -> tuple[str, ...]:
    """biomod2's metric.select.thresh, read as the name says: a candidate scoring at least the
    minimum is kept, and one carrying no score is not."""
    if min_score is None:
        return tuple(names)
    keep = tuple(n for n in names if np.isfinite(level[n]) and level[n] >= min_score)
    if len(keep) < 2:
        best = max((v for v in level.values() if np.isfinite(v)), default=float("nan"))
        raise ValueError(f"`min_score = {min_score:g}` leaves {len(keep)} candidate"
                         f"{'' if len(keep) == 1 else 's'} to combine; the best scores {best:.3g}")
    return keep


def _best(names, level: dict) -> str:
    """The best-scoring candidate, the first offered among any on the same score."""
    best, top = None, -np.inf
    for name in names:
        value = level.get(name, -np.inf)
        if np.isfinite(value) and value > top:
            best, top = name, value
    if best is None:
        raise ValueError("no candidate carries a score to read a scope off")
    return best


def _split_candidate(name: str) -> tuple[str, str]:
    learner, _, representation = str(name).partition(SEPARATOR)
    return learner, representation
