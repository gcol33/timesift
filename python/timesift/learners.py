"""Learners: a fit and a predict pair, and the ones that ship.

Everything the package fits goes through the same pair, so a learner of your own sits beside the
ones here and needs no change to the ladder, the folds or the scoring. A learner that needs a
package says so and stops; there is no second path that runs without it.

A learner also declares what it can be handed and how it covers several responses, so the layer
above can pair it with a representation and assemble its predictions without asking what kind of
model it is.
"""

from __future__ import annotations

import importlib.util
import inspect
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .control import CONTROL_SETTINGS, as_control, check_settings
from .registry import get_learner
from .representation import TimesiftMatrix
from .response import fitting_rows

__all__ = ["Fit", "Learner", "READS", "MULTI", "boosting", "cnn", "elasticnet", "fit_learner",
           "envelope", "flatten", "mlp", "rescnn", "forest", "stepwise", "tree"]

READS = ("tabular", "sequence")
MULTI = ("joint", "separate")


@dataclass
class Learner:
    """A name, a fit and a predict, what has to be installed for them to run, and what the learner
    reads.

    The one interface every arm goes through, the ones that ship and a pair of your own alike.
    ``data`` pins the learner to one representation, or is ``None`` to run it across every
    representation offered. ``reads`` is whether it takes a tabular block or an ordered sequence of
    bins, and ``multi`` is whether one fitted model covers every response or one is fitted per
    response and the matrix assembled from them.
    """

    name: str
    fit: Callable
    predict: Callable
    needs: tuple[str, ...] = ()
    params: dict = field(default_factory=dict)
    data: object = None
    reads: str = "tabular"
    multi: str = "separate"

    def __post_init__(self):
        if self.reads not in READS:
            raise ValueError(f"`reads` is one of {', '.join(READS)}, got {self.reads!r}")
        if self.multi not in MULTI:
            raise ValueError(f"`multi` is one of {', '.join(MULTI)}, got {self.multi!r}")
        # A representation, never the name of one: what a learner is pinned to is read for its
        # kind and its label long before anything is built, and a string carries neither.
        from .specs import Representation
        if self.data is not None and not isinstance(self.data, Representation):
            raise ValueError(f"the {self.name} learner's `data` must be a representation, such as "
                             f"grain(\"week\"), or None, got {type(self.data).__name__}.")

    def require(self) -> None:
        """Error, naming the install, unless what the learner needs is importable."""
        missing = [p for p in self.needs if importlib.util.find_spec(p) is None]
        if missing:
            raise ImportError(f"the {self.name} learner needs {' and '.join(missing)}. "
                              f"Install it with pip install {missing[0]}")


@dataclass
class Fit:
    """A fitted learner, the variables it was fitted on, and the bins and channels of the
    representation it was made on.

    A representation asked to predict is checked against those once, before any learner sees it:
    a calendar grain's bins are named by their starts, so a record from another period is refused
    by the first bin that differs rather than read by position.
    """

    learner: Learner
    model: object
    variables: tuple[str, ...]
    response: str = "presence_absence"
    bins: tuple[str, ...] = ()
    channels: tuple[str, ...] = ()

    def predict(self, x: TimesiftMatrix) -> np.ndarray:
        """Predictions for a representation, as a `[unit, variable]` matrix."""
        self._check_same_representation(x)
        p = np.asarray(self.learner.predict(self.model, x), dtype=np.float64)
        if p.shape[0] != x.values.shape[0]:
            raise ValueError(f"the learner returned {p.shape[0]} rows for "
                             f"{x.values.shape[0]} units")
        return p

    def _check_same_representation(self, x: TimesiftMatrix) -> None:
        head = "the representation predicted on has different channels or bins from the fitted one: "
        channels, bins = tuple(x.stats), tuple(x.bins)
        if channels != tuple(self.channels):
            raise ValueError(head + f"channels {', '.join(channels)} here and "
                             f"{', '.join(self.channels)} in the fit.")
        if len(bins) != len(self.bins):
            raise ValueError(head + f"{len(bins)} bin{'s' if len(bins) != 1 else ''} here and "
                             f"{len(self.bins)} bin{'s' if len(self.bins) != 1 else ''} in the "
                             "fit.")
        for i, (here, fitted) in enumerate(zip(bins, self.bins)):
            if here != fitted:
                raise ValueError(head + f"bin {i + 1} is {here} here and {fitted} in the fit. A "
                                 "calendar grain is read by its bins' instants, so a fit "
                                 "predicts a record over the same period; a lookback reads a "
                                 "span relative to each target and predicts any period.")


def fit_learner(learner, x: TimesiftMatrix, y, response: str = "presence_absence", control=None,
                group=None, **kwargs) -> Fit:
    """Fit one learner at one grain, under one registered response head.

    A ``fit`` that declares a ``head`` argument is handed the registered head, whose ``loss`` and
    ``activation`` say what it is fitting toward; the learners that ship read both from there and
    hold no response of their own. A ``fit`` that declares a ``control`` is handed the run's
    training settings the same way, and one that declares ``weights`` the head's case weights, an
    array of the response's shape, which is what a rare response weighs in every learner that
    ships; a fit that declares none fits unweighted.
    """
    from .registry import RESPONSES
    from .response import as_response
    learner = get_learner(learner)
    learner.require()
    head = RESPONSES.get(response)
    y = head["prepare"](as_response(y)).align(x.units)
    if group is not None and len(group) != x.values.shape[0]:
        raise ValueError(f"`group` must have one value per unit, got {len(group)} for "
                         f"{x.values.shape[0]}")
    given = _declared(learner.fit, head=head, control=control, variables=y.variables,
                      group=None if group is None else tuple(str(g) for g in group))
    if "weights" in inspect.signature(learner.fit).parameters:
        given["weights"] = _head_weights(head, y.values)
    model =learner.fit(x, y.values, **{**learner.params, **kwargs, **given})
    return Fit(learner=learner, model=model, variables=y.variables, response=response,
               bins=tuple(x.bins), channels=tuple(x.stats))


def _declared(fit, **given) -> dict:
    """The arguments a fit is handed because it declares them by name.

    A learner that trains under a control declares one, and the resolved control reaches it through
    that argument and through nothing else; the response head reaches a fit the same way, and so
    do the names of the response's variables and the grouping the outer fold map keeps whole, one
    value per unit or ``None``, by which a fit draws any split of its own. A fit that declares
    none is called with none,
    whatever the run carries, so a two-argument ``fit(x, y)`` is a learner like any other rather
    than one that has to absorb keywords it never asked for.
    """
    parameters = inspect.signature(fit).parameters
    return {name: value for name, value in given.items() if name in parameters}


def _head_weights(head, y: np.ndarray, fitting=None) -> np.ndarray:
    """The case weights a fit is made under, one per cell of the response it is handed: the head's
    where it carries ``weights``, and one everywhere where it does not. Every learner that ships
    reads them here, so what a rare response weighs is decided once, by the head. ``fitting``
    marks the rows the model is fitted on, for a learner that holds some back: the head reads
    what it reads off those rows and weights every row."""
    if head.get("weights") is None:
        return np.ones(y.shape, dtype=np.float64)
    w = np.asarray(head["weights"](y, fitting_rows(fitting, y.shape[0])), dtype=np.float64)
    if w.shape != y.shape or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("a response head's `weights(y, fitting)` returns a numeric array of "
                         "the response's shape, with no missing or negative entry.")
    return w


# The family a learner fitting one model per response fits under is read off the response head's
# loss, so a head registered with a squared-error loss reaches the same learners as a
# presence-absence one and each fits the model that loss names.
FAMILIES = {"binary_cross_entropy": "binomial", "squared_error": "gaussian"}


def _family(head) -> str:
    loss = head["loss"]
    if loss not in FAMILIES:
        raise ValueError(f"a learner fitting one model per response has no family for the "
                         f"{loss!r} loss. It knows {' and '.join(FAMILIES)}.")
    return FAMILIES[loss]


def flatten(x: TimesiftMatrix) -> np.ndarray:
    """``[unit, bin, channel]`` to ``[unit, bin * channel]`` in the array's own order.

    A channel that holds the same number in every bin carries no bin of its own and is one
    predictor, read once: repeating it would put the same column in front of a penalised fit as
    often as the grain has bins, and give it that many chances of being drawn by a forest or
    picked by a forward search.
    """
    n_u = x.values.shape[0]
    if not x.static:
        return x.values.reshape(n_u, -1, order="F")
    constant = np.array([s in set(x.static) for s in x.stats])
    moving = x.values[:, :, ~constant].reshape(n_u, -1, order="F")
    return np.ascontiguousarray(np.concatenate([moving, x.values[:, 0, constant]], axis=1))


# ---- the encoders ----------------------------------------------------------------------------

def _torch():
    try:
        import torch  # noqa: F401
    except ImportError as e:  # pragma: no cover - the message is the point
        raise ImportError("this learner needs torch. Install it with pip install torch") from e
    return importlib.import_module("torch")


class _LenSafePool:
    """Pools while there is something to halve, and is the identity once there is not, so the same
    stack runs at every grain of a ladder including one bin per year."""

    def __new__(cls, kernel: int = 2):
        torch = _torch()
        nn = torch.nn

        class Pool(nn.Module):
            def __init__(self):
                super().__init__()
                self.kernel = kernel
                self.pool = nn.MaxPool1d(kernel)

            def forward(self, z):
                return z if z.shape[-1] < self.kernel else self.pool(z)

        return Pool()


def _mlp_module(in_ch, in_len, n_out, hidden, dropout):
    nn = _torch().nn
    layers = [nn.Flatten()]
    prev = in_ch * in_len
    for k, h in enumerate(hidden):
        layers += [nn.Linear(prev, h), nn.ReLU()]
        if k < len(hidden) - 1:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers += [nn.Dropout(dropout), nn.Linear(prev, n_out)]
    return nn.Sequential(*layers)


def _cnn_module(in_ch, in_len, n_out, channels, kernel, dropout):
    nn = _torch().nn
    layers, prev = [], in_ch
    for c in channels:
        layers += [nn.Conv1d(prev, c, kernel, padding=kernel // 2), nn.BatchNorm1d(c),
                   nn.ReLU(), _LenSafePool()]
        prev = c
    layers += [nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout),
               nn.Linear(prev, n_out)]
    return nn.Sequential(*layers)


def _rescnn_module(in_ch, in_len, n_out, channels, blocks_per_stage, kernel, dilations, dropout):
    torch = _torch()
    nn = torch.nn

    class SE(nn.Module):
        def __init__(self, c, r=8):
            super().__init__()
            h = max(c // r, 4)
            self.fc = nn.Sequential(nn.Linear(c, h), nn.GELU(), nn.Linear(h, c), nn.Sigmoid())

        def forward(self, z):
            return z * self.fc(z.mean(dim=-1)).unsqueeze(-1)

    class ResBlock(nn.Module):
        def __init__(self, c, kernel, dilation, dropout):
            super().__init__()
            pad = (kernel // 2) * dilation
            self.conv = nn.Sequential(
                nn.Conv1d(c, c, kernel, padding=pad, dilation=dilation), nn.BatchNorm1d(c),
                nn.GELU(), nn.Dropout(dropout),
                nn.Conv1d(c, c, kernel, padding=pad, dilation=dilation), nn.BatchNorm1d(c))
            self.se = SE(c)

        def forward(self, z):
            return nn.functional.gelu(z + self.se(self.conv(z)))

    class ResCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(in_ch, channels[0], kernel, padding=kernel // 2),
                nn.BatchNorm1d(channels[0]), nn.GELU())
            layers, prev = [], channels[0]
            for si, c in enumerate(channels):
                if c != prev:
                    layers += [nn.Conv1d(prev, c, 1), nn.BatchNorm1d(c), nn.GELU()]
                d = dilations[si % len(dilations)]
                layers += [ResBlock(c, kernel, d, dropout) for _ in range(blocks_per_stage)]
                layers.append(_LenSafePool())
                prev = c
            self.features = nn.Sequential(*layers)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(prev * 2, n_out))

        def forward(self, z):
            h = self.features(self.stem(z))
            return self.head(torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1))

    return ResCNN()


# ---- the training recipe ---------------------------------------------------------------------

# One learner factory for every encoder: the architecture is a module constructor and nothing
# else, so the training recipe, the standardiser, the class weighting and the early stopping have
# one definition and cannot drift between architectures. Architecture reaches the module builder
# and everything else reaches the control, so a setting given at fit time is applied to whichever
# of the two it belongs to.
def _torch_learner(name, arch, given, data, reads) -> Learner:
    settings = check_settings(given)
    return Learner(name=name, needs=("torch",), params={**arch, **settings},
                   fit=_TorchFitter(name, arch), predict=_torch_predict, data=data,
                   reads=reads, multi="joint")


# The fit of an encoder is an object rather than a closure, and the fitted encoder holds its
# weights as arrays rather than as a live network, so a learner and a fit both pickle: what is
# written out is the encoder's name, an architecture, and numbers. The module builder is looked up
# by that name when the network is rebuilt, so a fit read back after an upgrade is this version's
# architecture loaded with the weights, never the last version's.
@dataclass(frozen=True)
class _TorchFitter:
    name: str
    arch: dict

    def __call__(self, x, y, *, head, control=None, group=None, **passed):
        unknown = set(passed) - set(self.arch) - set(CONTROL_SETTINGS)
        if unknown:
            raise TypeError(f"the {self.name} learner has no setting called "
                            f"{', '.join(sorted(unknown))}")
        return _torch_fit(x, y, self.name,
                          {**self.arch, **{k: v for k, v in passed.items() if k in self.arch}},
                          as_control(control).override(
                              {k: v for k, v in passed.items() if k in CONTROL_SETTINGS}),
                          head, group)


def _resolve_device(name):
    torch = _torch()
    if name in (None, "auto"):
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return name


# ---- the objective ---------------------------------------------------------------------------

# The response head names its loss and its activation, and the trainer looks both up here. Each
# loss takes the head's case weights for the cells of the batch, so a rare response weighs what
# the head says it weighs; an activation is applied at prediction and nowhere else.
def _binary_cross_entropy(torch):
    def loss(out, target, weight):
        return torch.nn.functional.binary_cross_entropy_with_logits(out, target, weight=weight)
    return loss


def _squared_error(torch):
    def loss(out, target, weight):
        return torch.mean(weight * (out - target) ** 2)
    return loss


TORCH_LOSSES = {"binary_cross_entropy": _binary_cross_entropy,
                "squared_error": _squared_error}
TORCH_ACTIVATIONS = {"sigmoid": lambda torch, out: torch.sigmoid(out),
                     "identity": lambda torch, out: out}


def _objective(head):
    if head["loss"] not in TORCH_LOSSES:
        raise ValueError(f"the encoders do not train under the {head['loss']!r} loss. "
                         f"They know {' and '.join(TORCH_LOSSES)}.")
    if head["activation"] not in TORCH_ACTIVATIONS:
        raise ValueError(f"the encoders have no {head['activation']!r} activation. "
                         f"They know {' and '.join(TORCH_ACTIVATIONS)}.")
    return TORCH_LOSSES[head["loss"]], head["activation"]


# ---- the training recipe ---------------------------------------------------------------------

def _torch_fit(x: TimesiftMatrix, y: np.ndarray, module, arch, cfg, head, group=None):
    torch = _torch()
    device = _resolve_device(cfg.device)
    make_loss, activation = _objective(head)

    m = np.transpose(x.values, (0, 2, 1))
    centre, scale = _channel_scaler(m, [s in x.position for s in x.stats])
    m = (m - centre) / scale

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    n = m.shape[0]
    val = _validation_split(y, cfg.val_frac, rng, group)
    fit_idx = np.setdiff1d(np.arange(n), val)

    xt = torch.tensor(m, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    # The weights are the head's, read off the fitting units alone and applied to every unit: the
    # validation units are held out of the count a rare response's weight is made from, as they
    # are held out of the fit, and the loss the early stopping reads on them is weighted as the
    # loss the fit minimises, so the epoch it keeps is the one the fit's own objective prefers.
    weights = _head_weights(head, y, fitting=np.isin(np.arange(n), fit_idx))
    wt = torch.tensor(weights, dtype=torch.float32, device=device)
    loss_fn = make_loss(torch)

    net = _torch_module(module)(in_ch=m.shape[1], in_len=m.shape[2],
                                n_out=y.shape[1], **arch).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.learning_rate,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    best_loss, best_state, bad = float("inf"), None, 0
    # The schedule anneals until the averaging begins and is then held flat, and the epochs it
    # averages over neither validate nor stop early: averaging the tail is a way of walking the
    # flat basin rather than of picking a single epoch out of it.
    swa_from = max(1, int(cfg.swa_start * cfg.epochs)) if cfg.swa else cfg.epochs + 1
    average, n_average = None, 0
    for epoch in range(1, cfg.epochs + 1):
        net.train()
        for b in _batches(rng.permutation(fit_idx), cfg.batch_size):
            idx = torch.tensor(b, dtype=torch.long, device=device)
            opt.zero_grad()
            loss_fn(net(xt[idx]), yt[idx], wt[idx]).backward()
            opt.step()
        if epoch < swa_from:
            sched.step()
        else:
            average, n_average = _accumulate(net, average, n_average)
            continue
        if not len(val):
            continue
        vloss = _torch_loss(net, loss_fn, xt, yt, wt, val, cfg.batch_size, device)
        if vloss < best_loss - 1e-4:
            best_loss, bad = vloss, 0
            best_state = _snapshot(net)
        else:
            bad += 1
            if bad >= cfg.early_stopping:
                break
    if n_average:
        net.load_state_dict(average)
        net.to(device)
        _refresh_batchnorm(net, xt, fit_idx, cfg.batch_size, device)
    elif best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    # The weights leave here as arrays rather than as a live network, and the fit carries the
    # device setting rather than the device it resolved to, so a fit made on one machine is a
    # plain object that predicts on another.
    return dict(module=module, arch=arch, shape=(m.shape[1], m.shape[2], y.shape[1]),
                state={k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()},
                centre=centre, scale=scale, device=cfg.device, activation=activation,
                channels=x.stats, bins=x.values.shape[1], batch_size=cfg.batch_size)


def _torch_restore(torch, model, device):
    """The network a fitted encoder is, built from its architecture and loaded with its weights."""
    in_ch, in_len, n_out = model["shape"]
    net = _torch_module(model["module"])(in_ch=in_ch, in_len=in_len, n_out=n_out, **model["arch"])
    net.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in model["state"].items()})
    return net.to(device).eval()


def _torch_loss(net, loss_fn, xt, yt, wt, idx_all, batch_size, device) -> float:
    torch = _torch()
    net.eval()
    total = 0.0
    with torch.no_grad():
        for b in _batches(idx_all, batch_size):
            idx = torch.tensor(b, dtype=torch.long, device=device)
            total += float(loss_fn(net(xt[idx]), yt[idx], wt[idx])) * len(b)
    return total / len(idx_all)


def _validation_split(y: np.ndarray, val_frac: float, rng, group=None) -> np.ndarray:
    """The inner validation set, ``val_frac`` of the units drawn by a plain random permutation.

    Under a grouping the draw is over the groups, and the rows of a drawn group are held out
    together: a unit the outer folds kept whole is not split across the fit and the loss it stops
    on. A draw stratified on the response total was tried and dropped for this: on the Schrankogel
    weekly arm the plain permutation reads 0.0047 AUC above it, five seeds each, p = 0.019
    (``inst/reproduce/README.md``). A single rare response can still draw no presence into the
    validation set here, where the stratified draw guaranteed one.
    """
    if group is None:
        key = np.arange(y.shape[0])
    else:
        _, key = np.unique(np.asarray([str(g) for g in group]), return_inverse=True)
    n = int(key.max()) + 1
    n_val = max(1, int(round(val_frac * n)))
    if val_frac <= 0 or n - n_val < 2:
        return np.empty(0, dtype=int)
    drawn = rng.permutation(n)[:n_val]
    return np.flatnonzero(np.isin(key, drawn))


def _snapshot(net) -> dict:
    """A copy of the weights as they stand, off the device and off the optimiser's storage."""
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def _accumulate(net, average, n):
    """A running mean of the weights. Only the floating-point entries are weights: a
    batch-normalisation module also carries an integer count of the batches it has seen, and a
    running mean of that is not a number."""
    state = _snapshot(net)
    if not n:
        return state, 1
    n += 1
    for k, v in state.items():
        if v.dtype.is_floating_point:
            average[k] = average[k] + (v - average[k]) / n
        else:
            average[k] = v
    return average, n


def _refresh_batchnorm(net, xt, fit_idx, batch_size, device):
    """Batch normalisation carries running statistics that belong to the weights that produced
    them, so an average of weights needs its own pass over the fitting units before it predicts
    anything. The statistics are reset and recomputed as the plain mean over the batches of that
    pass, weighting the k-th batch by 1/k, rather than folded into whatever the last epoch left
    behind."""
    torch = _torch()
    norms = [mod for mod in net.modules()
             if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)]
    if not norms:
        return net
    momentum = [mod.momentum for mod in norms]
    for mod in norms:
        mod.reset_running_stats()
    net.train()
    with torch.no_grad():
        for k, b in enumerate(_batches(fit_idx, batch_size), start=1):
            for mod in norms:
                mod.momentum = 1.0 / k
            net(xt[torch.tensor(b, dtype=torch.long, device=device)])
    for mod, was in zip(norms, momentum):
        mod.momentum = was
    return net


def _batches(idx: np.ndarray, size: int) -> list:
    """As few batches of at most ``size`` rows as the rows divide into, of as equal a length as
    they can be. Cutting fixed-length batches instead leaves a remainder, and a remainder of one
    row has no variance for batch normalisation to standardise by: the layer's running statistics
    take a NaN and every prediction after it is NaN. Fewer than two rows per batch is refused for
    the same reason, so a handful of rows is one batch."""
    n = len(idx)
    return np.array_split(idx, max(1, min(-(-n // size), n // 2)))


def _channel_scaler(m: np.ndarray, position):
    """One centre and one scale per channel of a ``[unit, channel, bin]`` array, over every unit
    and bin, the sample standard deviation as the R side reads it, and the identity on a channel
    ``position`` marks."""
    centre = m.mean(axis=(0, 2), keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = m.std(axis=(0, 2), ddof=1, keepdims=True)
    scale = np.where(np.isfinite(scale) & (scale >= 1e-8), scale, 1.0)
    position = np.asarray(position, dtype=bool).reshape(1, -1, 1)
    return np.where(position, 0.0, centre), np.where(position, 1.0, scale)


def _torch_predict(model, x: TimesiftMatrix) -> np.ndarray:
    torch = _torch()
    device = _resolve_device(model["device"])
    net = _torch_restore(torch, model, device)
    activation = TORCH_ACTIVATIONS[model["activation"]]
    m = (np.transpose(x.values, (0, 2, 1)) - model["centre"]) / model["scale"]
    xt = torch.tensor(m, dtype=torch.float32, device=device)
    out = []
    with torch.no_grad():
        for b in _batches(np.arange(m.shape[0]), model["batch_size"]):
            idx = torch.tensor(b, dtype=torch.long, device=device)
            out.append(activation(torch, net(xt[idx])).cpu().numpy())
    return np.concatenate(out, axis=0)


# The encoders' module builders, by name. A fitted encoder stores the name and the builder is
# looked up when the network is rebuilt, so a fit carries no reference to a function at all.
TORCH_MODULES = {"mlp": _mlp_module, "cnn": _cnn_module, "rescnn": _rescnn_module}


def _torch_module(name):
    if name not in TORCH_MODULES:
        raise ValueError(f"this fit was made by the {name!r} encoder, which this version of the "
                         f"package does not carry. It has {', '.join(TORCH_MODULES)}.")
    return TORCH_MODULES[name]


def mlp(data=None, hidden=(512, 256), dropout=0.3, **settings) -> Learner:
    """Flattens the channels and builds in no temporal geometry.

    ``hidden`` and ``dropout`` are the architecture; anything else named is a training setting
    applied on top of the ``train_control()`` the learner is fitted under.
    """
    return _torch_learner("mlp", dict(hidden=tuple(hidden), dropout=dropout),
                          settings, data=data, reads="tabular")


def cnn(data=None, channels=(16, 32, 64, 128), kernel=7, dropout=0.3, **settings) -> Learner:
    """Convolution, batch normalisation, activation and pooling, then global average pooling."""
    return _torch_learner("cnn",
                          dict(channels=tuple(channels), kernel=kernel, dropout=dropout),
                          settings, data=data, reads="sequence")


def rescnn(data=None, channels=(32, 64, 128, 256), blocks_per_stage=2, kernel=7,
           dilations=(1, 2, 4, 8), dropout=0.3, **settings) -> Learner:
    """Dilated residual blocks with channel gates, pooling average and maximum together."""
    return _torch_learner("rescnn",
                          dict(channels=tuple(channels), blocks_per_stage=blocks_per_stage,
                               kernel=kernel, dilations=tuple(dilations), dropout=dropout),
                          settings, data=data, reads="sequence")


# ---- one model per response ------------------------------------------------------------------

# The learners that cover the responses one at a time share how a response is fitted and how the
# matrix is put back together, so the difference between them is the model and nothing else. A
# response with one outcome among the fitting units has no model to fit and is predicted its own
# share, which is the level a fitted model would collapse to.
def _inner_folds(yj: np.ndarray, v: int, seed: int, group=None) -> np.ndarray:
    """Inner folds for a fit that chooses a setting by cross-validation inside itself: one 0-based
    fold index per unit, dealt for one response under the fit's own seed, over the groups where
    the outer map carries a grouping, and stratified on the response, so a rare outcome is spread
    over the inner folds as evenly as its count allows."""
    from .response import _deal_response
    fold = _deal_response(yj, v, seed, None if group is None else list(group))
    labels = np.unique(fold)
    return np.searchsorted(labels, fold).astype(np.int32)


def _inner_fittable(yj: np.ndarray, fold: np.ndarray) -> bool:
    """Whether every inner training set of a presence-absence response holds at least two of each
    outcome, the fewest a logistic path is fitted to. A response short of that has too few of one
    outcome to choose a penalty on."""
    return all((yj[fold != k] == 1).sum() >= 2 and (yj[fold != k] == 0).sum() >= 2
               for k in np.unique(fold))


def _fit_columns(m: np.ndarray, y: np.ndarray, make, seeds, weights) -> list:
    out = []
    for j in range(y.shape[1]):
        yj = y[:, j]
        out.append(float(yj.mean()) if len(np.unique(yj)) < 2
                   else make(m, yj, seeds[j], weights[:, j]))
    return out


def _variable_seeds(seed: int, variables) -> list:
    """One seed per response, offset from the learner's by a hash of the response's name, as the R
    side derives them: the model of one response is then the same model whether it was fitted on
    its own or beside others, and in whatever order, and two responses never share a draw."""
    return [int(seed) + _name_offset(name) for name in variables]


def _name_offset(name: str) -> int:
    code = 0
    for b in name.encode("utf-8"):
        code = (code * 31 + b) % 104729
    return code


def _predict_columns(models: list, m: np.ndarray, predict) -> np.ndarray:
    """One column per response: the mean where the response held one value, else ``predict``
    of that response's fit on ``m``."""
    return np.column_stack([np.full(m.shape[0], f) if isinstance(f, float) else predict(f, m)
                            for f in models])


def _design(x: TimesiftMatrix, squares: bool) -> np.ndarray:
    m = flatten(x)
    return np.hstack([m, m ** 2]) if squares else m


def elasticnet(data=None, alpha=0.5, n_inner=5, squares=True, s="lambda.min", n_lambda=100,
               thresh=1e-8, threads=1, seed=1) -> Learner:
    """One penalised regression per variable, over every bin-by-channel column and, by default,
    their squares, with the penalty chosen by an inner cross-validation on the fitting units.

    There is no discrete selection step: the penalty path uses every column and shrinks, and
    nothing about the model is decided outside the fold it is fitted in. The family is the
    response head's: logistic under a binary cross-entropy loss, linear under a squared-error
    one, and so are the case weights, :func:`~timesift.response.positive_weights` under
    presence-absence, which every learner that ships fits under.

    The path is fitted by the same core the R package calls, so the two return the same
    coefficients for the same input. Its conventions are glmnet's, which is what the arm is
    measured against: weights normalised to sum to one, columns centred and scaled by their
    weighted mean and weighted standard deviation, a hundred penalties down from the smallest
    that leaves every coefficient at zero, and the held-out deviance read fold by fold. ``s`` is
    where the fit is read: ``"lambda.min"``, ``"lambda.1se"``, or a penalty of its own, which is
    interpolated between the two points of the path around it.

    The inner folds are dealt for each response and stratified on it, so a rare outcome is spread
    over them as evenly as its count allows. A presence-absence response whose inner training
    sets cannot each hold two of each outcome, the fewest a logistic path is fitted to, has too
    few of one outcome to choose a penalty on. It is predicted its share among the fitting units,
    as a response holding one outcome is, and the fit names every such response in ``unfitted``.

    A reweighted fit that does not settle at a penalty ends its path there and keeps the points
    before it, which is what glmnet does with the same event, and the penalty is chosen over the
    points fitted. It happens where a rare outcome is nearly separable at the small end of the
    path. The fit names every response whose path on the fitting units, or on any inner fold,
    ended that way in ``stopped``.

    ``threads`` is how many fits of one response's inner cross-validation run at once. The path on
    every fitting unit and the path of each inner fold are one independent fit each, so they
    parallelise without sharing anything, and ``n_inner + 1`` threads is as many as a response can
    use. The default is serial, because a package does not take a machine's cores without being
    asked. What comes back does not depend on it.
    """
    return Learner(name="elasticnet", fit=_elasticnet_fit, predict=_elasticnet_predict,
                   data=data, reads="tabular", multi="separate",
                   params=dict(alpha=alpha, n_inner=n_inner, squares=squares, s=s,
                               n_lambda=n_lambda, thresh=thresh, threads=threads, seed=seed))


def _elasticnet_fit(x, y, alpha, n_inner, squares, s, n_lambda, thresh, threads, seed, head,
                    variables, group=None, **_):
    from .penalised import penalised_cv
    family = _family(head)
    m = _design(x, squares)

    # The inner folds are dealt here rather than inside the path, so a grouping the outer folds
    # keep whole stays whole where the penalty is chosen, and a rare outcome is spread over them
    # rather than left to a plain deal.
    def make(design, yj, seed_j, w):
        fold = _inner_folds(yj, n_inner, seed_j, group)
        if family == "binomial" and not _inner_fittable(yj, fold):
            return float(yj.mean())
        return penalised_cv(design, yj, w, family, alpha, fold, int(fold.max()) + 1,
                            n_lambda=n_lambda, thresh=thresh, threads=threads)

    models = _fit_columns(m, y, make, _variable_seeds(seed, variables), _head_weights(head, y))
    return dict(models=models, squares=squares, s=s, n_col=m.shape[1], family=family,
                unfitted=[str(v) for v, f in zip(variables, models) if isinstance(f, float)],
                stopped=[str(v) for v, f in zip(variables, models) if _penalised_stopped(f)])


def _penalised_stopped(fit) -> bool:
    """Whether a cross-validated fit's path, on every unit or on any fold, ended at a penalty it
    did not settle at."""
    return isinstance(fit, dict) and (fit["stalled"] > 0 or bool(np.any(fit["fold_stalled"] > 0)))


def _elasticnet_predict(model, x):
    from .penalised import penalised_predict
    return _predict_columns(model["models"], _design(x, model["squares"]),
                            lambda f, m: penalised_predict(f, m, model["s"]))


def forest(data=None, trees=None, mtry=None, min_node=None, balance=False, preset="package",
           seed=1, threads=1) -> Learner:
    """One random forest per variable, over every bin-by-channel column: a probability forest
    under a presence-absence head and a regression forest under a head with a squared-error loss.
    Trees split on one column at a time and pay nothing for columns that carry nothing, so a forest
    reads a wide tabular representation without a penalty path and without a selection step.

    Each tree is grown on a bootstrap draw of the units, as many draws as there are units, and each
    node is split on the best of ``mtry`` columns drawn for it, by the Gini index or the sum of
    squares :func:`tree` splits by. A tree is grown out: a node is split while it holds at least
    twice ``min_node`` units and its responses differ. A leaf reports the share of presences among
    the draws it holds, or their mean, and the forest the mean over its trees. The forest is grown
    by the core the R package calls, and every draw comes from one generator seeded per tree, so
    the two languages grow the same forest on any number of threads.

    ``balance=True`` is the down-sampled forest biomod2 fits as ``RFd``: each tree draws as many
    units from each class as the smaller class holds.

    ``preset`` says whose defaults the settings left ``None`` take. ``"package"`` is
    randomForest's own, which is what biomod2's default option set fits: 500 trees, ``mtry`` the
    square root of the column count under presence-absence and a third of it under a squared-error
    loss, and ``min_node`` 1 and 5 under the two. ``"bigboss"`` is biomod2's tuned option set: 500
    trees, ``mtry=2`` and ``min_node=5``. A setting given explicitly beats either, and an ``mtry``
    above the column count is the column count.

    The case weights are the response head's, :func:`~timesift.response.positive_weights` under
    presence-absence, and weight the bootstrap draw: a unit is drawn in proportion to its weight,
    and within its class under ``balance``.
    """
    if preset not in ("package", "bigboss"):
        raise ValueError(f'`preset` is "package" or "bigboss", got {preset!r}.')
    return Learner(name="forest", fit=_rf_fit, predict=_rf_predict, data=data, reads="tabular",
                   multi="separate",
                   params=dict(trees=trees, mtry=mtry, min_node=min_node, balance=bool(balance),
                               preset=preset, seed=int(seed), threads=int(threads)))


def _forest_settings(preset, family, n_column, trees, mtry, min_node) -> dict:
    """The settings a forest is grown under: those given, and the preset's for the rest.
    randomForest's own defaults depend on the family and on how many columns there are, so they
    are settled at the fit."""
    binomial = family == "binomial"
    if preset == "bigboss":
        base = dict(trees=500, mtry=2, min_node=5)
    else:
        base = dict(trees=500,
                    mtry=max(1, int(np.floor(np.sqrt(n_column))) if binomial else n_column // 3),
                    min_node=1 if binomial else 5)
    given = dict(trees=trees, mtry=mtry, min_node=min_node)
    out = {k: int(base[k] if v is None else v) for k, v in given.items()}
    out["mtry"] = min(out["mtry"], n_column)
    return out


def _rf_fit(x, y, trees, mtry, min_node, balance, preset, seed, threads, head, variables, **_):
    from ._tree import forest_fit
    family = _family(head)
    m = flatten(x)
    settings = _forest_settings(preset, family, m.shape[1], trees, mtry, min_node)

    def make(design, yj, seed_j, w):
        return forest_fit(design, yj, w, family, settings["trees"], settings["mtry"],
                          settings["min_node"], balance, seed_j, threads)

    return dict(models=_fit_columns(m, y, make, _variable_seeds(seed, variables),
                                    _head_weights(head, y)),
                n_col=m.shape[1], family=family)


def _rf_predict(model, x):
    from ._tree import forest_predict
    return _predict_columns(model["models"], flatten(x), forest_predict)


def boosting(data=None, trees=None, depth=None, shrinkage=None, min_leaf=None, subsample=None,
             colsample=None, newton=False, lambda_=None, gamma=None, n_inner=None,
             preset="package", seed=1, threads=1) -> Learner:
    """One boosted model per variable, over every bin-by-channel column: a logistic model under a
    presence-absence head and a squared-error one under a head with a squared-error loss. The score
    starts at the log-odds of the weighted share of presences, or the weighted mean, and each tree
    is fitted to the loss's gradient at the current score and added to it scaled by ``shrinkage``.
    Each tree is grown on a subsample of the units drawn without replacement, and reads a subsample
    of the columns.

    ``newton`` picks the trees. Off, they are gbm's, which is what biomod2 fits as ``GBM``:
    ``depth`` splits grown best first, each the one that most reduces the weighted squared error of
    the working response with at least ``min_leaf`` units on each side, and a leaf that takes one
    Newton step on the loss. On, they are xgboost's exact greedy trees, which biomod2 fits as
    ``XGBOOST``: grown level by level to ``depth``, each split chosen by the second-order gain under
    the L2 penalty ``lambda_`` with at least ``min_leaf`` of hessian on each side, pruned where a
    split gains less than ``gamma``, and a leaf the step ``-G / (H + lambda)``. Either way
    ``depth`` is the order of interaction a tree can hold. With ``subsample=1`` the first-order fit
    is gbm's own to rounding, and the second-order one xgboost's to its single-precision storage;
    the model is grown by the core the R package calls, so the two languages fit the same model.

    ``n_inner`` folds, when above zero, choose how many trees are kept: the fit is repeated on each
    fold's complement, and the number of trees of least held-out deviance, summed over the folds
    and weighted by how many units each holds, is kept, as gbm's ``cv.folds`` chooses it. The folds
    are dealt for each response and stratified on it, as the elastic net's are.

    ``preset`` says whose defaults the settings left ``None`` take. ``"package"`` is the fitting
    package's own, which is what biomod2's default option set fits: under gbm 100 trees of one
    split, ``shrinkage=0.1``, ``min_leaf=10`` and ``subsample=0.5``; under xgboost 100 trees of
    depth 6, ``shrinkage=0.3``, ``min_leaf=1``, ``lambda_=1`` and every unit and column.
    ``"bigboss"`` is biomod2's tuned option set: under gbm 2500 trees of seven splits,
    ``shrinkage=0.001``, ``min_leaf=5``, ``subsample=0.5`` and three inner folds; under xgboost
    four trees of depth 2 at ``shrinkage=1``. A setting given explicitly beats either.

    ``lambda_`` is R's ``lambda``, spelled apart from Python's keyword. The case weights are the
    response head's, :func:`~timesift.response.positive_weights` under presence-absence, and weigh
    the gradient and every sum a tree is grown on; ``min_leaf`` counts units under gbm, as
    ``n.minobsinnode`` does.
    """
    if preset not in ("package", "bigboss"):
        raise ValueError(f'`preset` is "package" or "bigboss", got {preset!r}.')
    settings = _boost_settings(preset, bool(newton), trees, depth, shrinkage, min_leaf, subsample,
                               colsample, lambda_, gamma, n_inner)
    return Learner(name="boosting", fit=_boost_fit, predict=_boost_predict, data=data,
                   reads="tabular", multi="separate",
                   params=dict(settings, newton=bool(newton), seed=int(seed),
                               threads=int(threads)))


_BOOST_PRESETS = {
    ("package", False): dict(trees=100, depth=1, shrinkage=0.1, min_leaf=10, subsample=0.5,
                             colsample=1, lambda_=0, gamma=0, n_inner=0),
    ("bigboss", False): dict(trees=2500, depth=7, shrinkage=0.001, min_leaf=5, subsample=0.5,
                             colsample=1, lambda_=0, gamma=0, n_inner=3),
    ("package", True): dict(trees=100, depth=6, shrinkage=0.3, min_leaf=1, subsample=1,
                            colsample=1, lambda_=1, gamma=0, n_inner=0),
    ("bigboss", True): dict(trees=4, depth=2, shrinkage=1, min_leaf=1, subsample=1, colsample=1,
                            lambda_=1, gamma=0, n_inner=0),
}


def _boost_settings(preset, newton, trees, depth, shrinkage, min_leaf, subsample, colsample,
                    lambda_, gamma, n_inner) -> dict:
    """The settings boosted trees are fitted under: those given, and the preset's for the rest,
    which are gbm's or xgboost's as ``newton`` picks. gbm's trees take no penalty and no least
    gain."""
    penalised = any(v is not None and v != 0 for v in (lambda_, gamma))
    if not newton and penalised:
        raise ValueError("`lambda_` and `gamma` are the second-order trees' settings; "
                         "set `newton=True` to use them.")
    base = _BOOST_PRESETS[(preset, newton)]
    given = dict(trees=trees, depth=depth, shrinkage=shrinkage, min_leaf=min_leaf,
                 subsample=subsample, colsample=colsample, lambda_=lambda_, gamma=gamma,
                 n_inner=n_inner)
    out = {k: base[k] if v is None else v for k, v in given.items()}
    for k in ("trees", "depth", "n_inner"):
        out[k] = int(out[k])
    for k in ("shrinkage", "min_leaf", "subsample", "colsample", "lambda_", "gamma"):
        out[k] = float(out[k])
    return out


def _boost_fit(x, y, trees, depth, shrinkage, min_leaf, subsample, colsample, newton, lambda_,
               gamma, n_inner, seed, threads, head, variables, group=None, **_):
    from ._tree import boost_fit
    family = _family(head)
    m = flatten(x)

    def make(design, yj, seed_j, w):
        fold, n_fold = None, 0
        if n_inner > 0:
            fold = _inner_folds(yj, n_inner, seed_j, group)
            n_fold = int(fold.max()) + 1
        return boost_fit(design, yj, w, family, trees, depth, shrinkage, min_leaf, subsample,
                         colsample, newton, lambda_, gamma, seed_j, fold, n_fold, threads)

    return dict(models=_fit_columns(m, y, make, _variable_seeds(seed, variables),
                                    _head_weights(head, y)),
                n_col=m.shape[1], family=family)


def _boost_predict(model, x):
    from ._tree import boost_predict
    return _predict_columns(model["models"], flatten(x), boost_predict)


def maxnet(data=None, classes=None, regmult=1.0, formulation="background", type=None, knots=50,
           add_samples=True, clamp=True, n_inner=5, s="lambda.min", thresh=1e-8, max_design=2.0,
           threads=1, seed=1) -> Learner:
    """One maxnet model per variable, over every bin-by-channel column: maxnet's feature classes,
    its regularisation of each feature, and a lasso over them, fitted by the penalised core
    :func:`elasticnet` runs on, which the R package calls too. With the maxnet package's own
    settings the features and the penalty factors are maxnet's to rounding, and the fit settles at
    the objective glmnet reaches for maxnet.

    The feature classes are the letters of ``classes``: ``l`` the column itself, ``q`` its square,
    ``p`` the product of each pair of columns, ``h`` forward and reverse hinges at the interior of
    ``knots`` equally spaced points of each column's range, and ``t`` thresholds at 49 interior
    points of it. Left ``None``, they follow the response's presence count as ``maxnet.formula()``
    has them: ``"l"`` under 10 presences, ``"lq"`` under 15, ``"lqh"`` under 80, and ``"lqph"``
    from 80 on. A column holding one value over the units fitted takes no feature.

    ``formulation`` says what the absences are. ``"background"`` is maxnet's own and what biomod2
    fits as ``MAXNET``: every unit is background, each presence joins the background again unless
    an absence carries the same readings (``add_samples``), the background is weighted 100 against
    a presence's 1, and the model is read at the last of maxnet's 200 penalties, which scale with
    ``regmult``. Its output is maxnet's ``type``, ``"cloglog"`` by default as biomod2 predicts it.
    ``"absence"`` reads the absences as absences: a logistic lasso over the same features and
    penalty factors under the response head's case weights,
    :func:`~timesift.response.positive_weights` under presence-absence, with the penalty chosen by
    an inner cross-validation dealt as the elastic net's is (``n_inner`` folds, read at ``s``), and
    a probability as output.

    The background formulation takes no case weights, as maxnet takes none and biomod2 passes none:
    the background weight is what sets a presence's weight there. Either formulation holds each
    column inside the range it was fitted on, and each feature inside its own, before predicting,
    as maxnet's ``predict(clamp=TRUE)`` does; ``clamp=False`` reads them as they are.

    A hinge per column per knot makes the design large: a weekly three-channel representation,
    471 columns, is 47,100 features under ``"lqh"``, and its products under ``"lqph"`` 110,685
    more. The design is held in memory with a centred copy beside it, and a fit whose design would
    take more than ``max_design`` gigabytes is refused with the size it would have taken.

    A response with fewer than two presences, or one whose inner training sets cannot each hold two
    of each outcome under the absence formulation, is predicted its share among the fitting units,
    and the fit names it in ``unfitted``. A path that does not settle at a penalty ends there and is
    read at its last settled point; the fit names every such response in ``stopped``. The learner
    needs a presence-absence response, under a head whose loss is the binary cross-entropy.
    """
    if formulation not in ("background", "absence"):
        raise ValueError(f'`formulation` is "background" or "absence", got {formulation!r}.')
    if s not in ("lambda.min", "lambda.1se"):
        raise ValueError(f'`s` is "lambda.min" or "lambda.1se", got {s!r}.')
    if classes is not None and (not isinstance(classes, str) or not classes
                                or any(c not in "lqpht" for c in classes)):
        raise ValueError("`classes` is a string of the letters l, q, p, h and t, or None, "
                         f"got {classes!r}.")
    return Learner(name="maxnet", fit=_maxnet_fit, predict=_maxnet_predict, data=data,
                   reads="tabular", multi="separate",
                   params=dict(classes=classes, regmult=float(regmult), formulation=formulation,
                               type=_maxnet_type(formulation, type), knots=int(knots),
                               add_samples=bool(add_samples), clamp=bool(clamp),
                               n_inner=int(n_inner), s=s, thresh=float(thresh),
                               max_design=float(max_design), threads=int(threads),
                               seed=int(seed)))


def _maxnet_type(formulation, type) -> str:
    """The output a formulation predicts: maxnet's cloglog by default under the background, and the
    probability, which is the logistic output, under the absences."""
    if type is None:
        return "cloglog" if formulation == "background" else "logistic"
    allowed = ("cloglog", "logistic") if formulation == "background" else ("logistic",)
    if type not in allowed:
        raise ValueError(f"the {formulation} formulation predicts "
                         f"{' or '.join(repr(a) for a in allowed)}, got {type!r}.")
    return type


def _maxnet_fit(x, y, classes, regmult, formulation, type, knots, add_samples, clamp, n_inner, s,
                thresh, max_design, threads, seed, head, variables, group=None, **_):
    from ._maxnet import maxnet_fit
    if _family(head) != "binomial":
        raise ValueError("maxnet fits a presence-absence response, under a head whose loss is the "
                         f"binary cross-entropy; this head's loss is {head['loss']!r}.")
    m = flatten(x)

    def make(design, yj, seed_j, w):
        if (yj == 1).sum() < 2:
            return float(yj.mean())
        fold, n_fold = None, 0
        if formulation == "absence":
            fold = _inner_folds(yj, n_inner, seed_j, group)
            if not _inner_fittable(yj, fold):
                return float(yj.mean())
            n_fold = int(fold.max()) + 1
        return maxnet_fit(design, yj, w, classes=classes, knots=knots, regmult=regmult,
                          formulation=formulation, add_samples=add_samples, thresh=thresh,
                          one_se=s == "lambda.1se", fold=fold, n_fold=n_fold, threads=threads,
                          max_design=max_design)

    models = _fit_columns(m, y, make, _variable_seeds(seed, variables), _head_weights(head, y))
    return dict(models=models, n_col=m.shape[1], type=type, clamp=clamp,
                unfitted=[str(v) for v, f in zip(variables, models) if isinstance(f, float)],
                stopped=[str(v) for v, f in zip(variables, models)
                         if isinstance(f, dict) and (f["stalled"] > 0 or f["fold_stalled"] > 0)])


def _maxnet_predict(model, x):
    from ._maxnet import maxnet_predict
    return _predict_columns(model["models"], flatten(x),
                            lambda f, m: maxnet_predict(f, m, model["clamp"], model["type"]))


def tree(data=None, min_split=None, min_leaf=None, cp=None, max_depth=None, prune="se_sum",
         n_inner=None, preset="package", seed=1) -> Learner:
    """One classification or regression tree per variable, over every bin-by-channel column,
    grown under rpart's rules: the Gini index under a presence-absence head and the sum of squares
    under a head with a squared-error loss, a split only between two distinct values of a column,
    and the cost-complexity bookkeeping that keeps a split only where it lowers the risk by at
    least ``cp`` of the root's. On the same columns, weights and folds the tree is the one rpart
    grows, split for split, and its complexity table the one rpart reports; the tree is grown by
    the core the R package calls, so the two languages grow it identically.

    The grown tree is pruned back by an inner cross-validation. Its folds are dealt for each
    response and stratified on it, as the elastic net's are, and ``prune`` names the rule that
    reads the complexity table: ``"se_sum"`` takes the row of least cross-validated error plus its
    standard error among the rows that keep a split, the last of them where several tie, which is
    how biomod2 prunes its classification tree; ``"one_se"`` takes the smallest tree within one
    standard error of the least cross-validated error; ``"min"`` the first row reaching the least
    error; and ``"none"`` keeps the tree as grown.

    ``preset`` says whose defaults the settings left ``None`` take. ``"package"`` is rpart's own,
    which is what biomod2's default option set fits: ``min_split=20``,
    ``min_leaf=round(min_split / 3)`` (or ``min_split=3 * min_leaf`` where only ``min_leaf`` is
    given), ``cp=0.01``, ``max_depth=30`` and ten inner folds. ``"bigboss"`` is biomod2's tuned
    option set: ``min_split=5``, ``min_leaf=5``, ``cp=0.001``, ``max_depth=10`` and five inner
    folds. A setting given explicitly beats either.

    The case weights are the response head's, :func:`~timesift.response.positive_weights` under
    presence-absence. They weigh every class count and sum of squares the tree is grown on;
    ``min_split`` and ``min_leaf`` count observations, as rpart's do.
    """
    from ._tree import PRUNE_RULES
    if prune not in PRUNE_RULES:
        raise ValueError(f"`prune` is one of {', '.join(PRUNE_RULES)}, got {prune!r}.")
    settings = _tree_settings(preset, min_split, min_leaf, cp, max_depth, n_inner)
    return Learner(name="tree", fit=_tree_fit, predict=_tree_predict, data=data,
                   reads="tabular", multi="separate",
                   params=dict(settings, prune=prune, seed=int(seed)))


def _tree_settings(preset, min_split, min_leaf, cp, max_depth, n_inner) -> dict:
    """The settings a tree is grown under: those given, and the preset's for the rest. Under
    rpart's own defaults a ``min_leaf`` left open follows ``min_split`` and a ``min_split`` left
    open follows a given ``min_leaf``, as ``rpart.control()`` has them."""
    if preset == "bigboss":
        base = dict(min_split=5, min_leaf=5, cp=0.001, max_depth=10, n_inner=5)
    elif preset == "package":
        split = min_split if min_split is not None else (
            20 if min_leaf is None else 3 * int(min_leaf))
        base = dict(min_split=split, min_leaf=round(split / 3), cp=0.01, max_depth=30,
                    n_inner=10)
    else:
        raise ValueError(f'`preset` is "package" or "bigboss", got {preset!r}.')
    given = dict(min_split=min_split, min_leaf=min_leaf, cp=cp, max_depth=max_depth,
                 n_inner=n_inner)
    out = {k: base[k] if v is None else v for k, v in given.items()}
    for k in ("min_split", "min_leaf", "max_depth", "n_inner"):
        out[k] = int(out[k])
    out["cp"] = float(out["cp"])
    return out


def _tree_fit(x, y, min_split, min_leaf, cp, max_depth, n_inner, prune, seed, head, variables,
              group=None, **_):
    from ._tree import tree_fit, tree_prune, tree_prune_cp
    family = _family(head)
    m = flatten(x)

    def make(design, yj, seed_j, w):
        fold, n_fold = None, 0
        if prune != "none":
            fold = _inner_folds(yj, n_inner, seed_j, group)
            n_fold = int(fold.max()) + 1
        grown = tree_fit(design, yj, w, family, min_split, min_leaf, cp, max_depth, fold, n_fold)
        at = tree_prune_cp(grown, prune)
        return grown if at is None else tree_prune(grown, at)

    return dict(models=_fit_columns(m, y, make, _variable_seeds(seed, variables),
                                    _head_weights(head, y)),
                n_col=m.shape[1], family=family)


def _tree_predict(model, x):
    from ._tree import tree_predict
    return _predict_columns(model["models"], flatten(x), tree_predict)


def stepwise(data=None, max_terms=3, degree=2, direction="forward", terms="column",
             threads=1) -> Learner:
    """One generalised linear model per variable, its terms chosen by Akaike's criterion over every
    bin-by-channel column. The family is the response head's: logistic under a binary
    cross-entropy loss, Gaussian under a squared-error one, and so are the case weights.

    ``terms`` says what one term is. Under ``"column"`` it is a column's orthogonal polynomial of
    degree ``degree``, so a column enters with its curvature at once and can be non-monotone in the
    reading the way a niche optimum is. Under ``"power"`` each power of a column is a term of its
    own, which is how biomod2 writes a quadratic formula and how ``MASS::stepAIC()`` walks it. A
    column holding one value over the fitting units is not a term.

    ``direction`` is the search. ``"forward"`` starts from the intercept and admits the term that
    lowers the criterion most, while one does and the model holds fewer than ``max_terms``.
    ``"both"`` also weighs dropping each term it holds at every step, and ``"backward"`` starts
    from every term and drops alone. ``"none"`` fits every term and selects nothing: with
    ``terms="power"`` and ``degree=2`` that is the model biomod2's GLM fits. The two-way and
    backward searches are MASS's ``stepAIC()``, step for step. ``max_terms`` bounds what a forward
    or two-way search adds; ``float("inf")`` for no bound.

    Each fit is R's ``glm.fit``: iteratively reweighted least squares, the rank read off the same
    pivoted decomposition, and the same stopping rule. A move whose fit does not settle within its
    25 iterations is refused rather than taken, and the fit names every response whose final model
    did not settle in ``stopped``. A model with nothing but the intercept predicts the response's
    share among the fitting units. The search runs on the core the R package calls, so the two
    select the same terms and return the same coefficients; ``threads`` runs one step's candidate
    fits at once and does not change what comes back.
    """
    if direction not in ("forward", "both", "backward", "none"):
        raise ValueError('`direction` is "forward", "both", "backward" or "none", '
                         f"got {direction!r}.")
    if terms not in ("column", "power"):
        raise ValueError(f'`terms` is "column" or "power", got {terms!r}.')
    if isinstance(max_terms, bool) or not isinstance(max_terms, (int, float))             or np.isnan(max_terms) or max_terms < 0:
        raise ValueError(f"`max_terms` is one number of zero or more, or inf, got {max_terms!r}.")
    if isinstance(degree, bool) or not float(degree).is_integer() or degree < 1:
        raise ValueError(f"`degree` is one whole number of one or more, got {degree!r}.")
    return Learner(name="stepwise", fit=_stepwise_fit, predict=_stepwise_predict,
                   data=data, reads="tabular", multi="separate",
                   params=dict(max_terms=max_terms, degree=int(degree), direction=direction,
                               terms=terms, threads=int(threads)))


def _stepwise_fit(x, y, max_terms, degree, direction, terms, threads, head, variables, **_):
    from ._stepwise import stepwise_fit
    family = _family(head)
    m = flatten(x)

    def make(design, yj, seed_j, w):
        return stepwise_fit(design, yj, w, family, max_terms=max_terms, degree=degree,
                            direction=direction, terms=terms, threads=threads)

    models = _fit_columns(m, y, make, [0] * y.shape[1], _head_weights(head, y))
    return dict(models=models, n_col=m.shape[1], family=family,
                stopped=[str(v) for v, f in zip(variables, models)
                         if isinstance(f, dict) and not f["converged"]])


def _stepwise_predict(model, x):
    from ._stepwise import stepwise_predict
    return _predict_columns(model["models"], flatten(x), stepwise_predict)


def envelope(data=None, quantile=0.025) -> Learner:
    """biomod2's surface range envelope, one per variable, over every bin-by-channel column: for
    each column the ``quantile`` and ``1 - quantile`` quantiles of its readings over the units
    present, and a unit predicted present where every column lies between its two, the ends
    included. With the same quantile it draws the envelope ``bm_SRE()`` draws; the quantile is R's
    default, type 7.

    The prediction is zero or one, and enters an ensemble as that. The envelope reads the presences
    and nothing else: neither the absences nor the head's case weights move it. Every column has to
    agree for a unit to be inside, so the more columns a representation has the fewer units any
    envelope holds, and a coarse grain, ``data=grain("season")``, is what it is meant for.

    A response with no presence, or with nothing else, is predicted its share among the fitting
    units and named in ``unfitted``. The learner needs a presence-absence response, under a head
    whose loss is the binary cross-entropy.
    """
    if isinstance(quantile, bool) or not isinstance(quantile, (int, float))             or not 0.0 <= quantile <= 0.5:
        raise ValueError(f"`quantile` is one number in [0, 0.5], got {quantile!r}.")
    return Learner(name="envelope", fit=_envelope_fit, predict=_envelope_predict, data=data,
                   reads="tabular", multi="separate", params=dict(quantile=float(quantile)))


def _envelope_fit(x, y, quantile, head, variables, **_):
    from ._envelope import envelope_fit
    if _family(head) != "binomial":
        raise ValueError("the envelope is drawn around presences, under a head whose loss is the "
                         f"binary cross-entropy; this head's loss is {head['loss']!r}.")
    m = flatten(x)
    models = _fit_columns(m, y, lambda design, yj, seed_j, w: envelope_fit(design, yj, quantile),
                          [0] * y.shape[1], np.ones(y.shape))
    return dict(models=models, n_col=m.shape[1],
                unfitted=[str(v) for v, f in zip(variables, models) if isinstance(f, float)])


def _envelope_predict(model, x):
    from ._envelope import envelope_predict
    return _predict_columns(model["models"], flatten(x), envelope_predict)
