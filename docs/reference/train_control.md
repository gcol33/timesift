# Training settings every neural learner reads

The one place a training setting is defaulted. An architecture
constructor carries its architecture and nothing else, the control
carries how that architecture is trained, and both a whole run and a
single learner take one, so there is never a second table of defaults to
keep in step with this one.

## Usage

``` r
train_control(
  epochs = 60L,
  batch_size = 64L,
  learning_rate = 0.001,
  weight_decay = 1e-04,
  optimizer = c("adamw", "adam", "sgd"),
  penalty = 0,
  alpha = 0.5,
  schedule = c("cosine", "constant", "plateau"),
  plateau_factor = 0.1,
  plateau_patience = 10L,
  early_stopping = Inf,
  val_frac = 0,
  device = "auto",
  seed = 1L,
  swa = FALSE,
  swa_start = 0.7
)
```

## Arguments

- epochs:

  Epoch budget the cosine schedule anneals over.

- batch_size:

  Most targets per optimiser step. The fitting targets are cut into as
  few batches of at most this many as they divide into, of as equal a
  length as they can be, so no batch is a remainder of one.

- learning_rate:

  Learning rate, the one the schedule starts from.

- weight_decay:

  The optimiser's weight decay: decoupled from the gradient under
  `"adamw"`, and added to it as `weight_decay` times each parameter
  under `"adam"` and `"sgd"`, as torch's optimisers take it.

- optimizer:

  `"adamw"`, `"adam"` or `"sgd"`, torch's optimisers at their own
  defaults besides the learning rate and the weight decay; `"sgd"` takes
  no momentum.

- penalty:

  The weight of a penalty added to the loss of every batch:
  `penalty * (alpha * sum(abs(W)) + (1 - alpha) * sqrt(sum(W^2)))`
  summed over every weight matrix and kernel of the network, its biases
  and normalisation scales left out. This is cito's `lambda`, the
  penalty biomod2's `DNN` fits under. 0 adds nothing.

- alpha:

  The share of the penalty on the absolute weights, as
  [`elasticnet()`](https://gillescolling.com/timesift/reference/elasticnet.md)
  reads its `alpha`: 1 penalises the absolute weights alone and 0 the
  norm alone. cito's `alpha` is `1 - alpha`.

- schedule:

  How the learning rate moves over the epochs: `"cosine"` anneals it to
  zero over the budget, `"constant"` holds it, and `"plateau"`
  multiplies it by `plateau_factor` once the loss has not improved for
  `plateau_patience` epochs, reading the validation loss where
  `val_frac` holds a set back and the epoch's mean training loss where
  it does not, as cito's `reduce_on_plateau` reads them. torch's own
  relative threshold of `1e-4` decides an improvement.

- plateau_factor, plateau_patience:

  The factor a plateau multiplies the learning rate by, and the epochs
  without improvement that make one.

- early_stopping:

  Epochs without an inner-validation improvement before training stops.
  Read only where `val_frac` holds a validation set back. `Inf`, the
  default, never stops: the whole budget is trained and the epoch with
  the lowest validation loss is restored.

- val_frac:

  Share of the fitting targets held back as an inner validation set,
  used for early stopping and for nothing else. It is never scored as a
  result. The set is drawn from every fit alike by a plain random
  permutation, so the fit on all targets that a run ends with also
  trains on the rest. At the default of 0 nothing is held back: every
  fitting target is trained on, the whole budget runs, and the fit keeps
  the last epoch, where the cosine schedule has annealed the learning
  rate to zero. On the Schrankogel weekly arm that epoch scores 0.007
  AUC above the epoch early stopping keeps on a 15 percent split, and
  within 0.001 of the best epoch read on the test folds.

- device:

  `"auto"` to take a graphics processor where there is one, NVIDIA's or
  Apple's, or a device name such as `"cuda"`, `"mps"` or `"cpu"`. A
  fitted encoder carries the setting rather than the device it resolved
  to, so a fit made on one machine predicts on another.

- seed:

  Seed for initialisation, batching and the inner validation split.

- swa:

  Average the weights of the tail epochs instead of keeping a single
  epoch. The schedule anneals to `swa_start` of the epoch budget and is
  then held flat while the remaining epochs' weights are averaged, and
  the batch-normalisation statistics are recomputed for the average.
  Early stopping is off while an average is being accumulated, so the
  averaging grain always runs.

- swa_start:

  Share of the epoch budget after which averaging begins.

## Value

A `timesift_control`.

## Details

A control records which of its settings were named in the call. Merging
two controls therefore moves only the settings that were asked for: a
learner given `train_control(epochs = 200)` reads 200 epochs and takes
every other setting from the control the run was given.

## Examples

``` r
train_control()
train_control(epochs = 200L, device = "cpu")
# biomod2's tuned DNN, cito's adam with a penalty and a plateau schedule
train_control(optimizer = "adam", learning_rate = 0.05, weight_decay = 0, penalty = 0.001,
              alpha = 0, schedule = "plateau", plateau_patience = 7L, epochs = 150L,
              batch_size = 100L, val_frac = 0.2, early_stopping = 14L)
```
