"""One call: build the representations, compare every candidate, and estimate choosing among them.

A candidate is one (representation, learner) pair, named for the two, and the contract every
candidate obeys is that it emits an out-of-fold prediction for every scorable cell over the same
folds. Everything above it -- the comparison, the ensemble, the importance -- reads only those
predictions, so a candidate whose learner fits one model covering every response and one whose
learner fits a model per response are the same thing by the time they are compared.

The fold map and the mask of scorable cells are drawn once, before anything is fitted, so every
candidate is scored on identical cells and any two of them can be contrasted cell by cell. Inside
each outer training fold the candidates are searched again on an inner split, one is chosen and
the stack's weights are fitted there, so the estimate of the selected candidate and of the stack
is read on folds neither the choice nor the weights saw.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ladder import ladder_from_rows, score_arm
from .learners import fit_learner
from .selection import (RULES, _choose_candidate, _estimate_row, _inner_se, _inner_splitter,
                        _join_candidates, _nested_estimate, inner_search, refit_candidates,
                        selection_context)
from .registry import RESPONSES, get_learner, resolve_metric
from .representation import TimesiftMatrix, timesift_set
from .response import Folds, Response
from .select import column_names, select_columns
# A candidate is named for the learner and the representation it pairs, and the ensemble reads the
# pair back out of that name, so the two share one separator rather than agreeing on one.
from .stack import SEPARATOR, run_ensemble
from .specs import (Representation, Sift, TimesiftSpec, _needs_target_time, as_sift,
                    build_representation, expand_sift, grains, resolve_folds, target_labels)

__all__ = ["Timesift", "timesift"]

# The one grain a tabular learner cannot be handed: it gives a column per reading rather than a
# reduction, which is the whole reason a sequence learner is the one that reads it.
UNREDUCED = "native"

SCORE_COLUMNS = ("candidate", "variable", "fold", "score", "scorable")


@dataclass
class Timesift:
    """A fitted sift: every candidate's out-of-fold predictions, their scores, and the combiner.

    ``candidates`` and ``scores`` are tables held as columns; ``oof``, ``models`` and ``fits`` are
    keyed by candidate name. What reads them -- the summary, the weights, the occlusion profile --
    reads them and never a model, so a number reported here was read where the score was.
    """

    candidates: dict
    scores: dict
    oof: dict
    representations: dict
    sift: Sift
    stack: object
    weights: object
    models: dict
    folds: Folds
    cells: object
    y: Response
    metric: str
    scorer: object
    response: str
    spec: TimesiftSpec
    fits: dict = field(default_factory=dict)
    control: object = None
    estimate: list | None = None
    selected: list | None = None
    inner: list | None = None
    fold_weights: list | None = None
    predictions: dict | None = None
    choice: str | None = None

    def representation_of(self, candidate: str) -> str:
        """Which representation a candidate reads."""
        for name, label in zip(self.candidates["candidate"], self.candidates["representation"]):
            if name == candidate:
                return label
        raise KeyError(f'no candidate called "{candidate}" in this fit. The candidates are '
                       f'{", ".join(self.candidates["candidate"])}.')

    def predict(self, targets, series=None, candidate: str = "ensemble") -> np.ndarray:
        """Predict new targets, rebuilding each member's representation from the stored settings.

        Every candidate is refitted on all the targets at the end of a fit, so what predicts here
        is one model per candidate rather than a fold's worth of them. ``"ensemble"`` combines them
        under the weights fitted on every target, ``"selected"`` predicts with the candidate the
        rule chose on every target (``choice``), and any other value names one candidate.
        """
        if candidate == "selected":
            candidate = self.choice
        if candidate == "ensemble":
            if self.stack is None:
                raise ValueError("this fit has no ensemble. Name a candidate: "
                                 f"{', '.join(self.models)}.")
            from .stack import ensemble_combine
            built = self._rebuild(self.stack.members, targets, series)
            return ensemble_combine(self.stack, {name: self.models[name].predict(built[name])
                                                 for name in self.stack.members})
        if candidate not in self.models:
            raise KeyError(f'no candidate called "{candidate}" was fitted. '
                           f"Fitted: {', '.join(self.models)}.")
        built = self._rebuild([candidate], targets, series)
        return self.models[candidate].predict(built[candidate])

    def _rebuild(self, names, targets, series) -> dict:
        """One build per representation, however many candidates read it."""
        labels = {name: self.representation_of(name) for name in names}
        built = {label: build_representation(self.sift[label], series, targets, self.spec)
                 for label in set(labels.values())}
        return {name: built[label] for name, label in labels.items()}

    def __repr__(self) -> str:  # pragma: no cover - display only
        from .report import summary
        return summary(self)


def timesift(targets, series=None, *, y, x=None, id=None, time=None, target_time=None,
             static=None, models=None, sift=None, ensemble=True, resampling=None, inner=5,
             rule: str = "argmax", response: str = "presence_absence", metric=None,
             control=None, keep_fits: bool = False, seed: int = 1,
             verbose: bool = True) -> Timesift:
    """Compare every learner across every representation, and estimate choosing among them.

    ``targets`` is one row per thing to predict and ``series`` is the long, time-stamped record
    belonging to it; both are mappings of column name to array, which a data frame satisfies.
    ``y``, ``x`` and ``static`` are selections over their own table: a name, a list of names, a
    glob such as ``"sp_*"``, or a function of a name.

    Within each outer fold of ``resampling`` the training targets are split again into ``inner``
    folds. Every candidate is cross-validated on that inner split, ``rule`` picks one on its inner
    score (``"argmax"`` or ``"coarsest_adequate"``, as in ``select_grain``), and the stack's
    weights are fitted on the inner out-of-fold predictions. Every candidate is then refitted on
    the whole outer training set and predicts the outer test fold, and the selected candidate's
    prediction and the prediction combined under that fold's weights are kept. Nothing the outer
    test fold holds enters the choice or the weights it is scored under, so ``estimate`` is of the
    procedure, selection and stacking included. Its interval is across the response variables of
    this dataset, all fitted and scored on the same targets and folds, and not one for a new
    sample.

    The same refits give every candidate an out-of-fold prediction on the outer folds, which
    ``scores`` holds: the comparison, whose highest level was picked out on the folds it is scored
    on. ``inner=None`` runs no inner search and makes no estimate. ``choice``, ``models`` and
    ``stack`` are the procedure applied to every target, for prediction: the rule read on the
    outer scores and weights fitted on the outer out-of-fold predictions.

    Columns of ``targets`` that are neither the response nor the identifier nor the anchor are
    ignored unless ``static`` names them: a predictor is never picked up because it happened to be
    in the table.
    """
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    spec = _resolve_spec(targets, series, y, x, id, time, target_time, static, response, metric)
    labels = target_labels(targets, spec)
    _check_rows(labels, spec)

    head = RESPONSES.get(response)
    y_mat = head["prepare"](_response(targets, spec, labels))
    folds = resolve_folds(resampling, y_mat, targets, spec).align(labels)
    cells = head["cells"](y_mat, folds)
    score, metric_name = resolve_metric(metric, head["metric"])
    # Refused here rather than after the fitting, because a contradiction between the run's head
    # and the combiner's is not worth a grid of fits to find out about.
    ensemble = run_ensemble(ensemble, response)

    members = _members(sift, series, spec)
    learners = _as_learners(models)
    _check_anchored(members, learners, spec)
    pairs = _pair(members, learners)

    representations, used = {}, {}
    for pair in pairs:
        if pair["reason"]:
            continue
        label = pair["representation"]
        if label in representations:
            # Two representations under one name would be one array fitted by both: the one
            # built first, whichever of the two a pinned learner asked for.
            if used[label]["spec"] != pair["spec"]:
                pinned = pair if pair["learner"].data is not None else used[label]
                raise ValueError(f'two representations are reported under the name "{label}": '
                                 f"the one the {pinned['learner'].name} learner is pinned to and "
                                 f"the one in the sift. Name the sift to tell them apart.")
            continue
        if verbose:
            print(f"building the {pair['representation']} representation")
        representations[pair["representation"]] = build_representation(pair["spec"], series,
                                                                       targets, spec)
        used[pair["representation"]] = pair
    _refuse_one_bin(pairs, representations)
    if all(pair["reason"] for pair in pairs):
        raise ValueError("no learner can read any of the representations:\n  "
                         + "\n  ".join(pair["reason"] for pair in pairs))
    if verbose:
        for pair in pairs:
            if pair["reason"]:
                print(f"not applicable: {pair['reason']}")

    fitted_pairs = [pair for pair in pairs if not pair["reason"]]
    names = [pair["candidate"] for pair in fitted_pairs]
    n_cand = len(fitted_pairs)
    nested = inner is not None
    stacking = ensemble is not None and n_cand >= 2
    # The candidate set as the selection engine reads it: `grain` names the array and `learner`
    # the candidate, and the order the candidates were declared in is both the fitting order and
    # the order a tie falls in.
    candidates = [dict(grain=pair["representation"], learner=pair["candidate"])
                  for pair in fitted_pairs]
    ctx = selection_context(timesift_set(dict(representations)), y_mat,
                            {pair["candidate"]: pair["learner"] for pair in fitted_pairs},
                            candidates, rule,
                            _inner_splitter(inner, folds.group) if nested else None, response,
                            metric if metric is not None else head["metric"], control,
                            folds.group)

    f = folds.fold
    levels = np.unique(f)
    shape = y_mat.values.shape
    oof = {name: np.full(shape, np.nan) for name in names}
    p_selected, p_ensemble = np.full(shape, np.nan), np.full(shape, np.nan)
    fits: dict = {}
    chosen, inner_rows, fold_weights = [], [], []
    for i, k in enumerate(levels, start=1):
        train, test = np.flatnonzero(f != k), np.flatnonzero(f == k)
        search = None
        if nested and n_cand >= 2:
            search = inner_search(ctx, train, seed + i, fold=int(k))
        refit = refit_candidates(ctx, candidates, train, test)
        for name, one in zip(names, refit):
            oof[name][test] = one["pred"]
            if keep_fits:
                fits[f"{name}|{int(k)}"] = one["fit"]
        won = 0 if search is None else names.index(search["won"]["learner"])
        if nested:
            p_selected[test] = refit[won]["pred"]
            chosen.append(_fold_choice(int(k), fitted_pairs[won], search, len(train), len(test)))
            if search is not None:
                inner_rows.extend(dict(g, candidate=g["learner"], representation=g["grain"])
                                  for g in search["grid"])
            if stacking:
                weights, combined = _fold_stack(search["lad"], candidates,
                                                y_mat.take_units(train), ensemble, refit)
                p_ensemble[test] = combined
                fold_weights.append(dict(fold=int(k), **weights))
        if verbose:
            picked = f" selected {names[won]}" if nested else ""
            print(f"fold {int(k)} of {len(levels)}{picked}")

    table = {key: [] for key in SCORE_COLUMNS}
    grain_of = []
    for pair in fitted_pairs:
        name, label = pair["candidate"], pair["representation"]
        rows = score_arm(label, name, y_mat, oof[name], f, levels, cells, score)
        table["candidate"].extend(rows["learner"])
        grain_of.extend(rows["grain"])
        for column in ("variable", "fold", "score", "scorable"):
            table[column].extend(rows[column])
    scores = {"candidate": np.asarray(table["candidate"]),
              "variable": np.asarray(table["variable"]),
              "fold": np.asarray(table["fold"], dtype=np.int64),
              "score": np.asarray(table["score"], dtype=float),
              "scorable": np.asarray(table["scorable"], dtype=bool)}

    if verbose:
        print(f"refitting every candidate on all {shape[0]} targets")
    fitted = {pair["candidate"]: fit_learner(pair["learner"],
                                             representations[pair["representation"]], y_mat,
                                             response=response, control=control,
                                             group=folds.group)
              for pair in fitted_pairs}
    stack, weights = _combine(ensemble, oof, y_mat, cells, folds, scores, verbose)

    estimate = predictions = None
    if nested:
        predictions = {"selected": p_selected}
        estimate = _run_estimate("selected", y_mat, p_selected, f, levels, cells, response,
                                 score, metric_name)
        if stacking:
            predictions["ensemble"] = p_ensemble
            estimate += _run_estimate("ensemble", y_mat, p_ensemble, f, levels, cells, response,
                                      score, metric_name)

    return Timesift(candidates=_candidate_table(pairs, representations), scores=scores, oof=oof,
                    representations=representations,
                    sift=Sift({k: v["spec"] for k, v in used.items()}), stack=stack,
                    weights=weights,
                    models=fitted, folds=folds, cells=cells, y=y_mat, metric=metric_name,
                    scorer=score, response=response, spec=spec, fits=fits, control=control,
                    estimate=estimate, selected=chosen if nested else None,
                    inner=inner_rows if nested and n_cand >= 2 else None,
                    fold_weights=fold_weights if nested and stacking else None,
                    predictions=predictions, choice=_run_choice(scores, grain_of, ctx))


# ---- the nested evaluation -----------------------------------------------------------------------

def _fold_choice(k: int, pair: dict, search, n_train: int, n_test: int) -> dict:
    won = None if search is None else search["won"]
    return dict(fold=k, candidate=pair["candidate"], representation=pair["representation"],
                learner=pair["learner"].name,
                inner_score=float("nan") if won is None else won["score"],
                inner_best=float("nan") if won is None else search["best"]["score"],
                inner_se=float("nan") if won is None else search["best"]["se"],
                n_train=n_train, n_test=n_test)


def _fold_stack(lad, candidates, y_train, spec, refit):
    """The weights one outer fold is combined under, fitted on the inner out-of-fold predictions of
    its own training targets over the inner split's scorable cells, and the outer test fold's
    predictions combined under them. The combiner never sees a prediction for a target of the
    outer test fold, nor that target's response."""
    from .stack import ensemble_combine, ensemble_fit
    inner_oof = {c["learner"]: lad.predictions[f"{c['grain']}|{c['learner']}"]
                 for c in candidates}
    inner_scores = dict(candidate=lad.learner, variable=lad.variable, fold=lad.fold,
                        score=lad.score, scorable=lad.scorable)
    stack = ensemble_fit(inner_oof, y_train, lad.cells, lad.folds, spec, inner_scores)
    combined = ensemble_combine(stack, {c["learner"]: one["pred"]
                                        for c, one in zip(candidates, refit)})
    return stack.weights, combined


def _run_estimate(arm, y, p, f, levels, cells, response, score, metric_name) -> list[dict]:
    """One arm of the estimate under every registered metric, and under the run's own where that
    is a function no registry holds, so the number the choice was made on is always a row."""
    rows = _nested_estimate(y, p, f, levels, cells, response)
    if metric_name not in {r["metric"] for r in rows}:
        scored = ladder_from_rows(score_arm(arm, arm, y, p, f, levels, cells, score),
                                  predictions={}, cells=cells, folds=Folds(fold=f, units=y.units),
                                  metric=metric_name, scorer=score, response=response, fits={})
        rows.append(_estimate_row(metric_name, scored))
    return [dict(row, arm=arm) for row in rows]


def _run_choice(scores: dict, grain_of: list, ctx: dict) -> str:
    """The procedure applied to every target: the rule, read on the outer scores with the outer
    folds as the split it chooses on. One candidate is its own choice."""
    candidates = ctx["candidates"]
    if len(candidates) < 2:
        return candidates[0]["learner"]
    lad = ladder_from_rows(dict(grain=grain_of, learner=scores["candidate"],
                                variable=scores["variable"], fold=scores["fold"],
                                score=scores["score"], scorable=scores["scorable"]),
                           predictions={}, cells=None, folds=None, metric="", fits={},
                           scorer=_unused_scorer)
    grid = _join_candidates(candidates, lad.summary(), _inner_se(lad), -1)
    return _choose_candidate(grid, ctx["size"], ctx["rule"])["learner"]


def _unused_scorer(y, p):
    raise RuntimeError("a table rebuilt from stored scores is read, never rescored")


def _combine(spec, oof, y, cells, folds, scores, verbose):
    """Fit the combiner on the out-of-fold predictions and nothing else."""
    from .stack import ensemble_fit
    if spec is None or len(oof) < 2:
        return None, None
    if verbose:
        print("fitting the ensemble")
    stack = ensemble_fit(oof, y, cells, folds, spec, scores)
    return stack, stack.weights


# ---- what is fitted, and on what -----------------------------------------------------------------

def _members(sift, series, spec) -> Sift:
    if series is None:
        return Sift({"static": Representation(label="static", kind="static", sequence=False)})
    if sift is None and spec.target_time is not None:
        raise ValueError("with `target_time` every representation is anchored on the target, and "
                         "there is no defensible default set of spans. Give `sift` as "
                         "lookbacks(...).")
    return expand_sift(as_sift(sift) if sift is not None else grains("auto"), series, spec)


def _check_anchored(members: Sift, learners, spec) -> None:
    """Whether the targets are anchored in time and whether a representation is has to be one
    answer, over every representation the run will build: the members of the sift and the ones
    learners pinned themselves to through ``data``. A lookback left without ``target_time``
    reaches the builder with no anchor to place its bins against."""
    reps = list(members.values()) + [one.data for one in learners if one.data is not None]
    if spec.target_time is None:
        wrong = _labels(r for r in reps if r.kind == "lookback")
        if wrong:
            _needs_target_time(wrong)
        return
    wrong = _labels(r for r in reps if r.kind != "lookback")
    if wrong:
        raise ValueError(f"with `target_time` every representation must be anchored on the "
                         f"target, so {wrong} cannot be used. Give `sift` as lookbacks(...).")


def _labels(reps) -> str:
    seen = list(dict.fromkeys(r.label for r in reps))
    return ", ".join(seen)


def _pair(members: Sift, learners) -> list:
    """Every (representation, learner) pair, and the reason for any of them that cannot be fitted.

    A pair named through a learner's own ``data =`` is an error where it cannot be read; the same
    pair reached by expanding the sift is carried with its reason instead, because a sift is a set
    to try and not a set of claims.
    """
    pairs = []
    for learner in learners:
        offered = [(learner.data.label, learner.data)] if learner.data is not None \
            else list(members.items())
        for label, rep in offered:
            reason = _refused(rep, learner)
            if reason and learner.data is not None:
                raise ValueError(reason)
            name = f"{learner.name}{SEPARATOR}{label}"
            if any(p["candidate"] == name for p in pairs):
                raise ValueError(f"two candidates are called {name}. Give one of the learners its "
                                 "own name.")
            pairs.append(dict(candidate=name, representation=label, spec=rep, learner=learner,
                              reason=reason))
    return pairs


def _refused(rep: Representation, learner) -> str | None:
    if learner.reads == "tabular" and rep.kind == "grain" and rep.grain == UNREDUCED:
        return (f"{learner.name}() reads a tabular representation; {rep.label} gives it one column "
                "per reading. Use grain(), multigrain() or lookback().")
    return None


def _refuse_one_bin(pairs, representations) -> None:
    """A representation of a single bin is refused once the array exists, so the message can say
    what it actually got rather than what the specification promised."""
    for pair in pairs:
        if pair["reason"]:
            continue
        m = representations[pair["representation"]]
        if pair["learner"].reads == "sequence" and m.values.shape[1] < 2:
            pair["reason"] = (f"{pair['learner'].name}() reads a sequence; "
                              f"{pair['representation']} gives one row of "
                              f"{m.values.shape[2]} features")


def _candidate_table(pairs, representations) -> dict:
    """One row per pair, the ones that could not be fitted among them: a pair carrying no score is
    what the summary reads as not applicable."""
    columns: dict = {k: [] for k in ("candidate", "representation", "learner", "grain", "bins",
                                     "channels", "multi", "reason")}
    for pair in pairs:
        m = representations.get(pair["representation"])
        columns["candidate"].append(pair["candidate"])
        columns["representation"].append(pair["representation"])
        columns["learner"].append(pair["learner"].name)
        columns["grain"].append(pair["spec"].grain or pair["spec"].kind)
        columns["bins"].append(0 if m is None else int(m.values.shape[1]))
        columns["channels"].append(0 if m is None else int(m.values.shape[2]))
        columns["multi"].append(pair["learner"].multi)
        columns["reason"].append(pair["reason"] or "")
    return {k: np.asarray(v) for k, v in columns.items()}


def _as_learners(models) -> list:
    """A learner, the name of a registered one, or a list of either.

    One learner is not a list of learners and reads in Python as a sequence of nothing, so it is
    wrapped here rather than left to iterate into its own fields. ``learner_dict`` takes the same
    three forms, and R's `models` takes a learner, a set of them or a list.
    """
    if models is None:
        models = _default_models()
    if not isinstance(models, (list, tuple)):
        models = [models]
    return [get_learner(m) for m in models]


def _default_models() -> list:
    from .learners import elasticnet
    return [elasticnet()]


# ---- reading the arguments -----------------------------------------------------------------------

def _resolve_spec(targets, series, y, x, id, time, target_time, static, response,
                  metric) -> TimesiftSpec:
    columns = column_names(targets)
    y_names = select_columns(columns, y, "`y`")
    if not y_names:
        raise ValueError("`y` names no column of `targets`")
    for name, arg in ((id, "`id`"), (target_time, "`target_time`")):
        if name is not None and name not in columns:
            raise ValueError(f"{arg} names {name}, which is not a column of `targets`. "
                             f"Available: {', '.join(columns)}")
    taken = tuple(y_names) + tuple(n for n in (id, target_time) if n is not None)
    static_names = () if static is None else tuple(
        select_columns([c for c in columns if c not in taken], static, "`static`", exclude=taken))

    if series is None:
        if target_time is not None:
            raise ValueError("`target_time` anchors a representation in a record, and no `series` "
                             "was given")
        if not static_names:
            raise ValueError("without `series` there is nothing to predict from. Name the "
                             "predictor columns in `static`.")
        return TimesiftSpec(y=tuple(y_names), id=id, static=static_names, response=response,
                            metric=metric)

    offered = column_names(series)
    if id is None:
        raise ValueError("`id` names the column linking `targets` to `series`, and is needed "
                         "wherever a series is given")
    if time is None:
        raise ValueError("`time` names the column of reading instants in `series`")
    for name, arg in ((id, "`id`"), (time, "`time`")):
        if name not in offered:
            raise ValueError(f"{arg} names {name}, which is not a column of `series`. "
                             f"Available: {', '.join(offered)}")
    value_columns = [c for c in offered if c not in (id, time)]
    x_names = tuple(value_columns) if x is None else tuple(
        select_columns(value_columns, x, "`x`", exclude=(id, time)))
    if not x_names:
        raise ValueError("`x` names no value column of `series`")
    return TimesiftSpec(y=tuple(y_names), x=x_names, id=id, time=time, target_time=target_time,
                        static=static_names, response=response, metric=metric)


def _check_rows(labels, spec) -> None:
    if spec.target_time is not None or spec.id is None:
        return
    seen, repeated = set(), []
    for one in labels:
        if one in seen and one not in repeated:
            repeated.append(one)
        seen.add(one)
    if repeated:
        shown = ", ".join(repeated[:5]) + (", ..." if len(repeated) > 5 else "")
        raise ValueError(f"{len(repeated)} id{'s appear' if len(repeated) > 1 else ' appears'} in "
                         f"more than one row of `targets`: {shown}. Give `target_time` to anchor "
                         "repeated targets in time.")


def _response(targets, spec, labels) -> Response:
    columns = []
    for v in spec.y:
        try:
            columns.append(np.asarray(targets[v], dtype=np.float64))
        except (TypeError, ValueError) as e:
            raise ValueError(f"the response column {v} is not numeric") from e
    return Response(values=np.column_stack(columns), units=labels, variables=tuple(spec.y))


