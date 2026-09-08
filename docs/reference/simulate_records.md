# Simulate sensor records whose response acts at a known temporal grain

Draws units carrying a year of sensor readings and a multi-variable
presence-absence response whose dependence on the record is a fixed
linear functional of the record at one named grain. The grain is
therefore known before anything is fitted, which is what makes the
output usable for asking whether
[`select_grain()`](https://gillescolling.com/timesift/reference/select_grain.md)
finds it.

## Usage

``` r
simulate_records(
  n = 300L,
  mechanism = c("none", "event", "season", "lag"),
  variables = 10L,
  prevalence = 0.1,
  auc = 0.75,
  from = "2021-09-01",
  days = 365L,
  step_hours = 3,
  seasonal = 8,
  offset_sd = 1,
  anomaly_sd = 1,
  anomaly_days = 2,
  offset_effect = 0,
  sensor_sd = 0.3,
  year_start = "09-01",
  seed = 1L,
  draw = 1L
)
```

## Arguments

- n:

  Number of units to draw.

- mechanism:

  Which generating mechanism, see The mechanisms.

- variables:

  Number of response variables. Each gets its own weights within the
  mechanism.

- prevalence:

  Marginal probability of presence, shared by every variable.

- auc:

  Population area under the ROC curve of the driver against the
  response.

- from:

  First reading instant, `"YYYY-MM-DD"`, read as UTC.

- days:

  Length of the record in days.

- step_hours:

  Sampling step in hours. Must divide 24.

- seasonal:

  Amplitude of the seasonal cycle shared by every unit, in reading
  units.

- offset_sd:

  Standard deviation of the unit-level thermal offset.

- anomaly_sd:

  Marginal standard deviation of each unit's AR(1) anomaly.

- anomaly_days:

  Correlation time of that anomaly, in days.

- offset_effect:

  Weight the unit-level offset enters the driver with. At the default
  `0` the driver reads the unit's *anomaly* alone, so a grain coarse
  enough to average the anomaly away loses the signal. At `1` the offset
  carries the response as well, and since every grain however coarse
  reports the offset, every grain is then equally good: that is the
  grain-invariant control, not a temporal mechanism.

- sensor_sd:

  Standard deviation of the measurement noise added to the latent
  record. The response is generated from the latent record; the readings
  returned carry this noise.

- year_start:

  `"MM-DD"` boundary of the hydrological year, passed to
  [`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md)
  when the mechanism's bins are located. Use the same value when
  representing the readings.

- seed:

  Seed of the design: the weights and the link coefficients. Two calls
  with the same `seed` and different `draw` share a design and draw
  independent units, which is what lets a held-out deployment sample be
  drawn from the same population as a training sample.

- draw:

  Seed of the unit draw. Also names the units, so two draws never
  collide.

## Value

A `timesift_simulation`: a list with `readings`, the long table
[`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md)
takes; `y`, the `[unit, variable]` 0/1 response; `driver`, the
standardised driver `z` behind it; `grain`, the true grain or `NA`;
`weights`, the `[reading, variable]` weights defining the driver;
`link`, the solved `b0` and `b1`; and `design`, the settings the draw is
reproducible from.

## What the true grain is

The response is driven by `g_ij = sum_t w_j(t) * a_i(t)`, a weighted
mean of unit `i`'s latent *anomaly*: the record with the seasonal cycle
every unit shares and the unit's own constant offset taken out, since
neither of those is temporally located and a grain of any width reports
both. The weights `w_j` are constant within the bins of one grain and
zero outside a short stretch of them, so `g` is exactly a linear
combination of that grain's bin means. The true grain of a mechanism is
the **coarsest grain of
[`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md)
at which `g` is still an exact linear functional of the
representation**: at that grain and at every grain whose bins nest
inside it, no information about `g` has been averaged away, and at any
coarser grain some has. Finer grains keep the information but spread it
over more coefficients, so they lose to the true grain by variance
rather than by bias, which is the tension the selection has to resolve.

## The mechanisms

- `"none"`:

  No temporal signal. The driver is a standard normal drawn
  independently of the record, so nothing in the readings carries
  information about the response and no grain is correct. `grain` is
  `NA`.

- `"event"`:

  An isolated event. The weights are uniform over three consecutive
  **day** bins at a fixed calendar position, one position per variable.
  True grain `"day"`: a week bin mixes the three days with four others
  and cannot be unmixed.

- `"season"`:

  A smooth seasonal response. The weights are uniform over one whole
  **season** bin, one season per variable, cycled. True grain
  `"season"`: month and day bins nest inside a season so they are exact
  too, a year bin mixes all four seasons.

- `"lag"`:

  A lagged, cumulative response. The weights decay geometrically over
  four consecutive **week** bins from a fixed anchor, one anchor per
  variable. True grain `"week"`: day bins nest inside weeks so they are
  exact too, month bins straddle week boundaries.

## How the skill is set rather than emergent

The driver is standardised to a standard normal by its population mean
and standard deviation, both computed in closed form from the generating
parameters rather than from the drawn units, so every draw and every
chunk of a draw is on the same scale. The response is
`y_ij ~ Bernoulli(plogis(b0 + b1 z_ij))` with `b0` and `b1` solved by
numerical integration so that the marginal prevalence is `prevalence`
and the population area under the ROC curve of `z` is `auc`. `auc` is
therefore a ceiling no fitted model reaches: the response is generated
from the latent record and the readings carry `sensor_sd` of measurement
noise on top of it, and the weights have to be estimated.

## See also

[`select_grain()`](https://gillescolling.com/timesift/reference/select_grain.md),
which this exists to test, and
[`grain_matrix()`](https://gillescolling.com/timesift/reference/grain_matrix.md),
whose calendar the weights are defined on.

## Examples

``` r
sim <- simulate_records(n = 40L, mechanism = "event", variables = 2L, days = 60L)
sim
sim$grain
x <- grain_matrix(sim$readings, unit, time, reading, grain = c("day", "month"))
dim(x$day)
```
