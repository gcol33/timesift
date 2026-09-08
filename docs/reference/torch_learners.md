# Sequence encoders with a joint multi-label head

Three encoders, all fitted the same way and all ending the same way: one
encoder maps a unit's representation to an embedding and a single linear
layer maps that embedding to one output per variable, so every variable
is predicted together from a shared representation. Pooling strength
across variables is what makes the rarer ones learnable at all at these
sample sizes.

## Usage

``` r
mlp(data = NULL, hidden = c(512L, 256L), dropout = 0.3, ...)

cnn(
  data = NULL,
  channels = c(16L, 32L, 64L, 128L),
  kernel = 7L,
  dropout = 0.3,
  ...
)

rescnn(
  data = NULL,
  channels = c(32L, 64L, 128L, 256L),
  blocks_per_stage = 2L,
  kernel = 7L,
  dilations = c(1L, 2L, 4L, 8L),
  dropout = 0.3,
  ...
)
```

## Arguments

- data:

  A representation the learner is pinned to, or `NULL` to run across
  every representation of the run.

- hidden:

  Hidden layer widths, for the fully connected encoder.

- dropout:

  Dropout rate.

- ...:

  Training settings for this learner, named as in
  [`train_control()`](https://gillescolling.com/timesift/reference/train_control.md).

- channels:

  Channel width of each stage.

- kernel:

  Convolution kernel width.

- blocks_per_stage:

  Residual blocks in each stage.

- dilations:

  Dilation of each stage, cycled if shorter than `channels`.

## Value

A
[`learner()`](https://gillescolling.com/timesift/reference/learner.md).

## Details

`mlp()` flattens the channels and builds in no temporal geometry. It is
what separates the effect of the model class from the effect of the
representation: where it matches a penalised regression, the difference
a convolutional encoder makes is the convolution rather than the
network.

`cnn()` is blocks of a one-dimensional convolution, batch normalisation,
a rectified linear activation and max pooling, then global average
pooling. Pooling becomes an identity once a sequence is shorter than its
kernel, so the same stack runs at every grain of a ladder, including one
bin per year.

`rescnn()` adds dilated residual blocks with squeeze-excitation channel
gates, whose stage dilations widen the receptive field toward the
seasonal scale without widening the kernel, and concatenates global
average with global maximum pooling so extremes reach the head beside
the level.

The three constructors carry architecture. How that architecture is
trained is
[`train_control()`](https://gillescolling.com/timesift/reference/train_control.md),
which the run supplies; a setting named in `...` here overrides the
run's control for this learner alone. What the head is trained toward is
the response head's: its `loss` is the training objective and its
`activation` the output transform, so a head registered with a
squared-error loss and an identity activation trains the same encoders
on a continuous response.

Every channel is standardised by its own centre and scale, computed over
every unit and bin of the fitting units, so a static predictor appended
as a channel sits on the same footing as a reading whatever its units
are.

A fitted encoder holds its weights as plain arrays and rebuilds the
network when it predicts, so a fit saved with
[`saveRDS()`](https://rdrr.io/r/base/readRDS.html) predicts after
[`readRDS()`](https://rdrr.io/r/base/readRDS.html) in a fresh session.

## Examples

``` r
cnn(epochs = 5)
```
