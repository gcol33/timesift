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
  early_stopping = 10L,
  val_frac = 0.15,
  device = "auto",
  seed = 1L,
  pos_weight_cap = 50,
  swa = FALSE,
  swa_start = 0.7
)
```

## Arguments

- epochs:

  Epoch budget the cosine schedule anneals over.

- batch_size:

  Targets per optimiser step.

- learning_rate:

  Learning rate.

- weight_decay:

  AdamW weight decay.

- early_stopping:

  Epochs without an inner-validation improvement before training stops.

- val_frac:

  Share of the fitting targets held back as an inner validation set,
  used for early stopping and for nothing else. It is never scored as a
  result.

- device:

  `"auto"` to take a graphics processor where there is one, or a device
  name such as `"cuda"` or `"cpu"`.

- seed:

  Seed for initialisation, batching and the inner validation split.

- pos_weight_cap:

  Ceiling on the per-response positive-class weight, which is the ratio
  of absences to presences among the fitting targets.

- swa:

  Average the weights of the tail epochs instead of restoring the best
  single epoch. The schedule anneals to `swa_start` of the epoch budget
  and is then held flat while the remaining epochs' weights are
  averaged, and the batch-normalisation statistics are recomputed for
  the average. Early stopping is off while an average is being
  accumulated, so the averaging grain always runs.

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
```
