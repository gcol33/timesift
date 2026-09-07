"""One call: build the representations, fit every candidate over one fold map, combine them.

A candidate is one (representation, learner) pair, named for the two, and the contract every
candidate obeys is that it emits an out-of-fold prediction for every scorable cell over the same
folds. Everything above it -- the comparison, the ensemble, the importance -- reads only those
predictions, so a candidate whose learner fits one model covering every response and one whose
learner fits a model per response are the same thing by the time they are compared.

The fold map and the mask of scorable cells are drawn once, before anything is fitted, so every
candidate is scored on identical cells and any two of them can be contrasted cell by cell.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ladder import score_arm
from .learners import Fit, fit_learner
from .registry import METRICS, RESPONSES, get_learner
from .representation import TimesiftMatrix
from .response import Folds, Response
from .select import column_names, select_columns
# A candidate is named for the learner and the representation it pairs, and the ensemble reads the
# pair back out of that name, so the two share one separator rather than agreeing on one.
from .stack import SEPARATOR, run_ensemble
from .specs import (Representation, Sift, TimesiftSpec, as_sift, build_representation, expand_sift,
                    grains, resolve_folds, target_labels)

__all__ = ["CandidateFit", "Timesift", "timesift"]

# The one grain a tabular learner cannot be handed: it gives a column per reading rather than a
# reduction, which is the whole reason a sequence learner is the one that reads it.
UNREDUCED = "native"

SCORE_COLUMNS = ("candidate", "variable", "fold", "score", "scorable")


@dataclass
class CandidateFit:
    """One candidate's fitted models, whichever way its learner covers the responses.

    A joint learner is one model over the whole response matrix and a separate one is a model per
    response; either way this predicts the same ``[target, response]`` matrix, in the response
    order the candidate was fitted on.
    """

    learner: object
    fits: tuple[Fit, ...]
    variables: tuple[str, ...]

    def predict(self, x: TimesiftMatrix) -> np.ndarray:
        """One ``[row, response]`` matrix, whatever the candidate is made of.

        A joint learner contributes one fit covering every response and a separate one contributes
        a fit per response; the columns are assembled by name either way, so nothing above a
        candidate can tell which it was.
        """
        columns = [v for f in self.fits for v in f.variables]
        p = np.concatenate([f.predict(x) for f in self.fits], axis=1)
        position = {v: j for j, v in enumerate(columns)}
        missing = [v for v in self.variables if v not in position]
        if missing:
            raise ValueError(f"the {self.learner.name} learner returned no column for "
                             f"{missing[0]}")
        return p[:, [position[v] for v in self.variables]]


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
    response: str
    spec: TimesiftSpec
    fits: dict = field(default_factory=dict)
    control: object = None

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
        is one model per candidate rather than a fold's worth of them.
        """
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
             static=None, models=None, sift=None, ensemble=True, resampling=None,
             response: str = "presence_absence", metric=None, control=None,
             keep_fits: bool = False, verbose: bool = True) -> Timesift:
    """Fit every learner across every representation, on one fold map, and combine them.

    ``targets`` is one row per thing to predict and ``series`` is the long, time-stamped record
    belonging to it; both are mappings of column name to array, which a data frame satisfies.
    ``y``, ``x`` and ``static`` are selections over their own table: a name, a list of names, a
    glob such as ``"sp_*"``, or a function of a name.

    Columns of ``targets`` that are neither the response nor the identifier nor the anchor are
    ignored unless ``static`` names them: a predictor is never picked up because it happened to be
    in the table.
    """
    spec = _resolve_spec(targets, series, y, x, id, time, target_time, static, response, metric)
    labels = target_labels(targets, spec)
    _check_rows(labels, spec)

    head = RESPONSES.get(response)
    y_mat = head["prepare"](_response(targets, spec, labels))
    folds = resolve_folds(resampling, y_mat, targets, spec).align(labels)
    cells = head["cells"](y_mat, folds)
    score = metric if callable(metric) else METRICS.get(metric or head["metric"])
    metric_name = _metric_name(metric, head)
    # Refused here rather than after the fitting, because a contradiction between the run's head
    # and the combiner's is not worth a grid of fits to find out about.
    ensemble = run_ensemble(ensemble, response)

    members = _members(sift, series, spec)
    learners = [get_learner(m) for m in (models if models is not None else _default_models())]
    pairs = _pair(members, learners)

    representations, used = {}, {}
    for pair in pairs:
        if pair["reason"] or pair["representation"] in representations:
            continue
        if verbose:
            print(f"building the {pair['representation']} representation")
        representations[pair["representation"]] = build_representation(pair["spec"], series,
                                                                       targets, spec)
        used[pair["representation"]] = pair["spec"]
    _refuse_one_bin(pairs, representations)
    if all(pair["reason"] for pair in pairs):
        raise ValueError("no learner can read any of the representations:\n  "
                         + "\n  ".join(pair["reason"] for pair in pairs))
    if verbose:
        for pair in pairs:
            if pair["reason"]:
                print(f"not applicable: {pair['reason']}")

    levels = np.unique(folds.fold)
    table = {k: [] for k in SCORE_COLUMNS}
    oof, fitted, fits = {}, {}, {}
    for pair in pairs:
        if pair["reason"]:
            continue
        name, label = pair["candidate"], pair["representation"]
        learner, m = pair["learner"], representations[label]
        if verbose:
            print(f"fitting {name}")
        p = _out_of_fold(m, y_mat, folds, levels, learner, response, control, name, fits,
                         keep_fits)
        oof[name] = p
        rows = score_arm(label, name, y_mat, p, folds.fold, levels, cells, score)
        table["candidate"].extend(rows["learner"])
        for column in ("variable", "fold", "score", "scorable"):
            table[column].extend(rows[column])
        fitted[name] = _fit_candidate(learner, m, y_mat, response, control)

    scores = {"candidate": np.asarray(table["candidate"]),
              "variable": np.asarray(table["variable"]),
              "fold": np.asarray(table["fold"], dtype=np.int64),
              "score": np.asarray(table["score"], dtype=float),
              "scorable": np.asarray(table["scorable"], dtype=bool)}
    stack, weights = _combine(ensemble, oof, y_mat, cells, folds, scores, verbose)

    return Timesift(candidates=_candidate_table(pairs, representations), scores=scores, oof=oof,
                    representations=representations, sift=Sift(used), stack=stack, weights=weights,
                    models=fitted, folds=folds, cells=cells, y=y_mat, metric=metric_name,
                    response=response, spec=spec, fits=fits, control=control)


# ---- the fitting loop ---------------------------------------------------------------------------

def _out_of_fold(m, y, folds, levels, learner, response, control, name, fits,
                 keep_fits) -> np.ndarray:
    p = np.full(y.values.shape, np.nan)
    row = {u: i for i, u in enumerate(y.units)}
    column = {v: j for j, v in enumerate(y.variables)}
    for k in levels:
        train = np.flatnonzero(folds.fold != k)
        held = m.take_units(np.flatnonzero(folds.fold == k))
        fit = _fit_candidate(learner, m.take_units(train), y.take_units(train), response, control)
        predicted = fit.predict(held)
        # Keyed on both axes rather than positional: a learner returning its responses in another
        # order would otherwise scramble which prediction belongs to which one, silently.
        for a, u in enumerate(held.units):
            for b, v in enumerate(fit.variables):
                p[row[u], column[v]] = predicted[a, b]
        if keep_fits:
            fits[f"{name}|{int(k)}"] = fit
    return p


def _fit_candidate(learner, x, y, response, control) -> CandidateFit:
    """The learner declares whether one fitted model covers every response; where it does not, this
    is where the responses are taken one at a time, so the candidate emits one matrix either way.
    """
    given = {} if control is None else {"control": control}
    if learner.multi == "joint":
        parts = (fit_learner(learner, x, y, response=response, **given),)
    else:
        parts = tuple(fit_learner(learner, x, y.take_variables([j]), response=response, **given)
                      for j in range(len(y.variables)))
    return CandidateFit(learner=learner, fits=parts, variables=y.variables)


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
    members = expand_sift(as_sift(sift) if sift is not None else grains("auto"), series, spec)
    anchored = [k for k, v in members.items() if v.kind == "lookback"]
    if spec.target_time is None:
        if anchored:
            raise ValueError(f"{', '.join(anchored)} reads a lookback ending at each target's own "
                             "instant, so `target_time` has to name the column holding it.")
        return members
    calendar = [k for k in members if k not in anchored]
    if calendar:
        raise ValueError(f"with `target_time` every representation must be anchored on the "
                         f"target, so {', '.join(calendar)} cannot be used. Give `sift` as "
                         "lookbacks(...).")
    return members


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


def _metric_name(metric, head) -> str:
    if metric is None:
        return head["metric"]
    return metric if isinstance(metric, str) else getattr(metric, "__name__", "metric")
