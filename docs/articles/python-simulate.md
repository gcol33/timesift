# Python: a record to test on

A simulated record with a grain planted in it, to test a run against a
known answer.

[All of the Python
reference](https://gillescolling.com/timesift/articles/python-reference.md)

## Installation

``` r

# Install from CRAN
install.packages("timesift")

# Or install the development version from GitHub
# install.packages("pak")
pak::pak("gcol33/timesift")
```

``` bash
# Install from PyPI
pip install timesift

# with the torch encoders, the contrasts and the plots
pip install "timesift[torch,contrasts,plot]"

# Or install the development version from GitHub
pip install git+https://github.com/gcol33/timesift
```

## `simulate_records()`

``` python
simulate_records(
    n: int = 300,
    mechanism: str = 'none',
    variables: int = 10,
    prevalence: float = 0.1,
    auc: float = 0.75,
    from_: str = '2021-09-01',
    days: int = 365,
    step_hours: float = 3,
    seasonal: float = 8,
    offset_sd: float = 1,
    anomaly_sd: float = 1,
    anomaly_days: float = 2,
    offset_effect: float = 0,
    sensor_sd: float = 0.3,
    year_start: str = '09-01',
    seed: int = 1,
    draw: int = 1,
)
```

Draw units carrying a record and a presence-absence response acting at
one known grain.

The response is driven by `g_ij = sum_t w_j(t) a_i(t)`, a weighted mean
of unit `i`’s latent anomaly: the record with the shared seasonal cycle
and the unit’s own offset taken out. The weights are constant within the
bins of one grain and zero outside a short stretch of them, so the true
grain is the coarsest grain at which `g` is still an exact linear
functional of the representation. `"none"` draws the driver
independently of the record; `"event"` reads three consecutive days,
`"season"` one whole season, and `"lag"` four consecutive weeks under a
geometric decay.

The driver is standardised by its population mean and standard
deviation, computed in closed form from the settings, and the response
is `Bernoulli(expit(b0 + b1 z))` with `b0` and `b1` solved so the
marginal prevalence is `prevalence` and the population area under the
ROC curve of `z` is `auc`. `auc` is a ceiling no fitted model reaches.

`seed` fixes the design and `draw` the units, so two calls with one
`seed` and two `draw` values are two samples of one population. `from_`
is R’s `from`, renamed because `from` is a Python keyword.

## `Simulation`

``` python
Simulation(readings, y, driver, grain, weights, link, design, grain_stat, anchor)
```

A simulated record, its response, and everything the draw is
reproducible from.

`readings` is the long table `grain_matrix` takes, as a mapping of
`unit`, `time` and `reading`. `y` is the `[unit, variable]` 0/1 response
and `driver` the standardised driver `z` behind it. `grain` is the true
grain, or `None` where the response does not read the record. `weights`
is the `[reading, variable]` matrix defining the driver, `link` the
solved `b0` and `b1`, and `design` the settings of the draw.

Attributes:

- `readings` - dict
- `y` - Response
- `driver` - np.ndarray
- `grain` - str \| None
- `weights` - np.ndarray
- `link` - dict
- `design` - dict
- `grain_stat` - str
- `anchor` - np.ndarray
