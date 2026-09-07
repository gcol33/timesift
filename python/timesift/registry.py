"""One registry mechanism, used by the three things the package is meant to be extended with.

Learners, response heads and metrics all register the same way, so adding any of them is a
registration and never an edit to the code that fits or scores: the fitting path holds no list of
names. The R package carries the same three registries under the same names.
"""

from __future__ import annotations

from typing import Callable


class Registry:
    """A named collection of one kind of extension."""

    def __init__(self, what: str):
        self.what = what
        self._entries: dict[str, object] = {}

    def set(self, name: str, value, overwrite: bool = False):
        if not isinstance(name, str) or not name:
            raise ValueError(f"a {self.what} needs a single non-empty name")
        if name in self._entries and not overwrite:
            raise ValueError(f'{self.what} "{name}" is already registered. '
                             "Pass overwrite=True to replace it.")
        self._entries[name] = value
        return value

    def get(self, name: str):
        if name not in self._entries:
            raise KeyError(f'unknown {self.what} "{name}". Registered: '
                           f'{", ".join(self.names())}.')
        return self._entries[name]

    def has(self, name: str) -> bool:
        return name in self._entries

    def remove(self, name: str) -> None:
        self._entries.pop(name, None)

    def names(self) -> list[str]:
        # C collation, so a registry lists the same order on every machine and beside R's.
        return sorted(self._entries)


LEARNERS = Registry("learner")
RESPONSES = Registry("response")
METRICS = Registry("metric")


def resolve_metric(metric, default: str | None = None):
    """The function that scores and the name a report prints, from either way a metric is given.

    A metric reaches a run as a registered name or as a function of ``(y, p)``, and everything
    that rescores afterwards -- the ensemble row of a report, an occlusion profile -- needs the
    function rather than a name to look up again. Both travel with the fit. A function has no name
    to print, and reads as ``<function>`` on both sides rather than as whatever the language calls
    an anonymous one.
    """
    if metric is None:
        metric = default
    if callable(metric):
        return metric, "<function>"
    if not isinstance(metric, str):
        raise TypeError("`metric` is the name of a registered metric or a function of (y, p), "
                        f"got {type(metric).__name__}")
    return METRICS.get(metric), metric


def register_learner(name: str, constructor: Callable, overwrite: bool = False):
    """Make a learner available by name. The learners that ship are registered the same way.

    ``constructor`` is called with no arguments and returns a ``Learner``, so a learner asked for
    by name is built with its own defaults.
    """
    if not callable(constructor):
        raise ValueError("a learner is registered as a constructor taking no arguments")
    return LEARNERS.set(name, constructor, overwrite)


def learners() -> list[str]:
    """The learners registered under this session."""
    return LEARNERS.names()


def get_learner(learner):
    """A ``Learner``, whether it arrived as one or as the name of a registered one."""
    from .learners import Learner
    if isinstance(learner, Learner):
        return learner
    return LEARNERS.get(learner)()


def register_metric(name: str, fn: Callable, overwrite: bool = False):
    """Register a metric: a function of ``(y, p)`` on one held-out cell.

    ``y`` and ``p`` are the observed values of the units in one fold and a model's predictions for
    them, both the same length. It returns one number, or a value that is not finite where the cell
    defines none. Registering one makes it available to ``grain_ladder`` by name, with no change
    to the fitting code.
    """
    if not callable(fn):
        raise ValueError("a metric is a function of (y, p)")
    return METRICS.set(name, fn, overwrite)


def metrics() -> list[str]:
    """The metrics registered under this session."""
    return METRICS.names()


RESPONSE_FIELDS = ("prepare", "activation", "loss", "metric", "cells")


def register_response(name: str, spec: dict, overwrite: bool = False):
    """Register a response head: what the values being predicted are and where a score is defined.

    ``spec`` is a mapping with ``prepare(y)``, returning the response a learner is fitted on;
    ``activation``, the name of the output transform (``"sigmoid"`` or ``"identity"``); ``loss``,
    the name of the training objective (``"binary_cross_entropy"`` or ``"squared_error"``);
    ``metric``, the default metric name; and ``cells(y, folds)``, returning the mask of scorable
    cells. Presence-absence with a joint multi-label head is what ships; an abundance or
    phenology response is a registration rather than a second fitting path. Every learner that
    ships reads ``loss`` and ``activation`` from here: the encoders train under the loss and
    predict through the activation, and the learners fitting one model per response take the
    family the loss names, logistic or Gaussian. The combiner minimises the same loss.
    """
    missing = [f for f in RESPONSE_FIELDS if f not in spec]
    if missing:
        raise ValueError(f"a response specification needs {', '.join(missing)}")
    return RESPONSES.set(name, spec, overwrite)


def responses() -> list[str]:
    """The response heads registered under this session."""
    return RESPONSES.names()
