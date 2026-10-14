# Installing torch

The neural learners
[`cnn()`](https://gillescolling.com/timesift/reference/torch_learners.md),
[`rescnn()`](https://gillescolling.com/timesift/reference/torch_learners.md)
and
[`mlp()`](https://gillescolling.com/timesift/reference/torch_learners.md)
require torch. This article covers GPU setup, download problems and
installation checks. The README describes the default installation. The
other learners use timesift’s C++ code.

torch for R is an R package plus two libraries it downloads after
installation, LibTorch and LibLantern. torch for Python is one wheel
from PyPI or from PyTorch’s own index. The steps below follow the [torch
for R installation
guide](https://torch.mlverse.org/docs/articles/installation.html) and
[PyTorch’s install page](https://pytorch.org/get-started/locally/);
those pages are the reference when a version here has moved on.

## R

### Processor and Apple Silicon

``` r

install.packages("torch")
torch::install_torch(timeout = 600)
```

In an interactive session torch installs the two libraries by itself the
first time it is needed; `install_torch()` does it explicitly, and is
what a script or a cluster job runs. A non-interactive session also
downloads on load when the environment variable `TORCH_INSTALL` is `1`.
On Apple Silicon the same build runs on the graphics processor through
MPS.

### NVIDIA graphics processor

The simplest route is torch’s pre-built binary, which carries CUDA and
cuDNN inside it, so neither has to be installed on the system. It is
about 2 GB, hence the longer timeout:

``` r

options(timeout = 600)
kind <- "cu128"
version <- available.packages()["torch", "Version"]
options(repos = c(
  torch = sprintf("https://torch-cdn.mlverse.org/packages/%s/%s/", kind, version),
  CRAN = "https://cloud.r-project.org"
))
install.packages("torch")
```

`kind` is `"cu128"` for CUDA on Windows and Linux, `"cpu"` for the
processor build, and `"cpu-intel"` or `"cpu-m1"` on macOS.

The alternative is a system CUDA toolkit: version 12.8 on Windows, 12.6,
12.8 or 12.9 on Linux, with a matching cuDNN, then
`install.packages("torch")` as above. `Sys.setenv(CUDA = "12.8")` before
installing forces the version torch looks for. The CUDA runtime can also
come as an R package instead of a system toolkit:

``` r

install.packages("cuda12.8",
                 repos = c("https://mlverse.r-universe.dev", "https://cloud.r-project.org"))
install.packages("torch")
```

### Checking what torch found

``` r

torch::torch_is_installed()
torch::cuda_is_available()
torch::backends_mps_is_available()
```

`train_control(device = "auto")`, the default, takes CUDA where
`cuda_is_available()` is `TRUE`, MPS where `backends_mps_is_available()`
is, and the processor otherwise. A fit carries the setting rather than
the device it resolved to, so a model trained on a GPU machine predicts
on a laptop. To pin one device for a whole run:

``` r

fit <- timesift(plots, logger, y = starts_with("sp_"), id = plot, time = datetime,
                learners = cnn(), sift = grains("week"),
                control = train_control(device = "cuda"))
```

### When the download fails

- **“Timeout of 60 seconds was reached”**:
  `torch::install_torch(timeout = 600)`.
- **More detail on what failed**: `Sys.setenv(TORCH_INSTALL_DEBUG = 1)`
  before installing.
- **No internet on the machine, or a blocked download**:
  [`torch::get_install_libs_url()`](https://torch.mlverse.org/docs/reference/install_torch_from_file.html)
  prints the two URLs. Download both files on a machine that can reach
  them, copy them over, and point the install at the local copies:

``` r

Sys.setenv(TORCH_URL = "/path/to/libtorch.zip")
Sys.setenv(LANTERN_URL = "/path/to/lantern.zip")
torch::install_torch()
```

- **Windows, a missing `lantern.dll` module**: the Microsoft Visual C++
  runtime is missing; the [torch installation
  guide](https://torch.mlverse.org/docs/articles/installation.html)
  links the fix.

## Python

### Processor, Apple Silicon and Linux with CUDA

``` bash
pip install "timesift[torch]"
```

The `torch` extra pulls torch from PyPI. On Linux that wheel brings its
CUDA libraries with it, so an NVIDIA graphics processor works without
further setup. On macOS it runs on Apple Silicon’s graphics processor
through MPS. On Windows it runs on the processor only.

### Windows with an NVIDIA graphics processor

Install torch from PyTorch’s own index first, then timesift; pip keeps
the torch already there:

``` bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install "timesift[torch]"
```

`cu130` is CUDA 13.0; `cu126` and `cu132` are also published for the
current torch. The graphics processor’s driver has to support the CUDA
version chosen; `nvidia-smi` prints the highest one it does. No CUDA
toolkit is needed beyond the driver.

### Checking what torch found

``` bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

On Windows a version without a `+cu` suffix (such as `2.14.1` rather
than `2.14.1+cu130`) is the PyPI processor build, and the index step
above was skipped. `ts.train_control(device="auto")` resolves the device
as in R.
