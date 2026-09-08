"""The training settings, defaulted once.

Every learner that trains by gradient descent reads the same settings, so they are declared here
and nowhere else: an architecture constructor carries architecture and whatever it was told to
override, never a second copy of the defaults. A setting given to a learner is applied on top of
the control it is fitted under, so overriding one is the same call in either language.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

__all__ = ["CONTROL_SETTINGS", "TrainControl", "as_control", "train_control"]


@dataclass(frozen=True)
class TrainControl:
    """How long to train, on what, and when to stop."""

    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping: int = 10
    val_frac: float = 0.15
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
        if self.early_stopping < 1:
            raise ValueError(f"`early_stopping` must be at least 1, got {self.early_stopping}")
        if not 0 <= self.val_frac < 1:
            raise ValueError(f"`val_frac` must be in [0, 1), got {self.val_frac}")
        if not 0 <= self.swa_start < 1:
            raise ValueError(f"`swa_start` must be in [0, 1), got {self.swa_start}")

    def override(self, settings: dict) -> "TrainControl":
        """The control with the settings a learner or a call gave applied on top of it."""
        return replace(self, **check_settings(settings))


CONTROL_SETTINGS = tuple(f.name for f in fields(TrainControl))


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

    ``batch_size`` is the most targets an optimiser step reads: the fitting targets are cut into as
    few batches of at most that many as they divide into, of as equal a length as they can be.
    ``val_frac`` is held back from every fit alike, the fit on all targets a run ends with
    included, one target from each of as many equal-count strata of the response total as the
    set holds.

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
