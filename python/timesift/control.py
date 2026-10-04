"""The training settings, defaulted once.

Every learner that trains by gradient descent reads the same settings, so they are declared here
and nowhere else: an architecture constructor carries architecture and whatever it was told to
override, never a second copy of the defaults. A setting given to a learner is applied on top of
the control it is fitted under, so overriding one is the same call in either language.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace

__all__ = ["CONTROL_SETTINGS", "TrainControl", "as_control", "train_control"]

OPTIMIZERS = ("adamw", "adam", "sgd")
SCHEDULES = ("cosine", "constant", "plateau")


@dataclass(frozen=True)
class TrainControl:
    """How long to train, on what, and when to stop."""

    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    penalty: float = 0.0
    alpha: float = 0.5
    schedule: str = "cosine"
    plateau_factor: float = 0.1
    plateau_patience: int = 10
    early_stopping: float = math.inf
    val_frac: float = 0.0
    device: str = "auto"
    seed: int = 1
    swa: bool = False
    swa_start: float = 0.7

    def __post_init__(self):
        if self.epochs < 1:
            raise ValueError(f"`epochs` must be at least 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"`batch_size` must be at least 1, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"`learning_rate` must be positive, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ValueError(f"`weight_decay` cannot be negative, got {self.weight_decay}")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"`optimizer` is one of {', '.join(OPTIMIZERS)}, got "
                             f"{self.optimizer!r}")
        if self.penalty < 0:
            raise ValueError(f"`penalty` cannot be negative, got {self.penalty}")
        if not 0 <= self.alpha <= 1:
            raise ValueError(f"`alpha` must be in [0, 1], got {self.alpha}")
        if self.schedule not in SCHEDULES:
            raise ValueError(f"`schedule` is one of {', '.join(SCHEDULES)}, got "
                             f"{self.schedule!r}")
        if not 0 < self.plateau_factor < 1:
            raise ValueError(f"`plateau_factor` must be in (0, 1), got {self.plateau_factor}")
        if self.plateau_patience < 0:
            raise ValueError(f"`plateau_patience` cannot be negative, got "
                             f"{self.plateau_patience}")
        if self.early_stopping < 1:
            raise ValueError(f"`early_stopping` must be at least 1, or math.inf for never, "
                             f"got {self.early_stopping}")
        if not 0 <= self.val_frac < 1:
            raise ValueError(f"`val_frac` must be in [0, 1), got {self.val_frac}")
        if not 0 <= self.swa_start < 1:
            raise ValueError(f"`swa_start` must be in [0, 1), got {self.swa_start}")

    def __repr__(self) -> str:  # pragma: no cover - display only
        lines = ["<timesift control>"]
        for f in fields(self):
            value = getattr(self, f.name)
            lines.append(f"  {f.name:<16} {describe(value)}"
                         + ("   (default)" if value == f.default else ""))
        return "\n".join(lines)

    def override(self, settings: dict) -> "TrainControl":
        """The control with the settings a learner or a call gave applied on top of it."""
        return replace(self, **check_settings(settings))


CONTROL_SETTINGS = tuple(f.name for f in fields(TrainControl))


def describe(value) -> str:
    """A setting as the printed forms of a control and a learner show it: a sequence joined by
    slashes, as R's ``format()`` of a vector is."""
    if isinstance(value, (tuple, list)):
        return "/".join(str(v) for v in value)
    return str(value)


def check_settings(settings: dict) -> dict:
    """The settings given, once every name in them is one the control carries."""
    unknown = [k for k in settings if k not in CONTROL_SETTINGS]
    if unknown:
        raise TypeError(f"there is no training setting called {', '.join(sorted(unknown))}. "
                        f"The settings are {', '.join(CONTROL_SETTINGS)}")
    return dict(settings)


def train_control(**settings) -> TrainControl:
    """The settings every neural learner reads, with anything named here replacing its default.

    ``epochs``, ``batch_size``, ``learning_rate``, ``weight_decay``, ``early_stopping``,
    ``val_frac``, ``device`` and ``seed`` are the settings a run is described by, and ``swa`` with
    ``swa_start`` average the weights over the tail of the schedule rather than keeping one epoch
    out of it. What a rare response weighs is the response head's, not a training setting.

    ``optimizer`` is ``"adamw"``, ``"adam"`` or ``"sgd"``, torch's optimisers at their own defaults
    besides the learning rate and the weight decay; ``"sgd"`` takes no momentum. The weight decay is
    decoupled from the gradient under ``"adamw"`` and added to it as ``weight_decay`` times each
    parameter under the other two, as torch's optimisers take it.

    ``penalty`` weighs a penalty added to the loss of every batch,
    ``penalty * (alpha * sum(abs(W)) + (1 - alpha) * sqrt(sum(W ** 2)))`` summed over every weight
    matrix and kernel of the network, its biases and normalisation scales left out. This is cito's
    ``lambda``, the penalty biomod2's ``DNN`` fits under, and 0 adds nothing. ``alpha`` is read as
    ``elasticnet()`` reads its ``alpha``: 1 penalises the absolute weights alone and 0 the norm
    alone, so cito's ``alpha`` is ``1 - alpha``.

    ``schedule`` moves the learning rate over the epochs: ``"cosine"`` anneals it to zero over the
    budget, ``"constant"`` holds it, and ``"plateau"`` multiplies it by ``plateau_factor`` once the
    loss has not improved for ``plateau_patience`` epochs, reading the validation loss where
    ``val_frac`` holds a set back and the epoch's mean training loss where it does not, as cito's
    ``reduce_on_plateau`` reads them. torch's own relative threshold of ``1e-4`` decides an
    improvement.

    ``batch_size`` is the most targets an optimiser step reads: the fitting targets are cut into as
    few batches of at most that many as they divide into, of as equal a length as they can be.
    ``val_frac`` is held back from every fit alike by a plain random permutation, the fit on all
    targets a run ends with included, and ``early_stopping`` is read only on that set: an epoch
    count of patience, or ``math.inf``, the default, to train the whole budget and restore the
    epoch with the lowest validation loss. At the ``val_frac`` default of 0 nothing is held back:
    every fitting target is trained on, the whole budget runs, and the fit keeps the last epoch,
    where the cosine schedule has annealed the learning rate to zero.

    ``device`` is ``"auto"`` for the graphics processor where there is one, NVIDIA's or Apple's,
    or the name of a device to train on. A fitted encoder carries the setting rather than the
    device it resolved to, so a fit made on one machine predicts on another.
    """
    return TrainControl(**check_settings(settings))


def as_control(control) -> TrainControl:
    """A ``TrainControl``, whether it arrived as one, as a mapping of settings, or not at all."""
    if control is None:
        return TrainControl()
    if isinstance(control, TrainControl):
        return control
    if isinstance(control, dict):
        return train_control(**control)
    raise TypeError("`control` is a train_control(), a mapping of settings, or None")
