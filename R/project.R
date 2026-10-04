#' Project a fit onto rasters
#'
#' Predicts one target per cell of a raster, each carrying the record of its own cell, and returns
#' the predictions as a raster with one layer per response. This is `BIOMOD_Projection()` and
#' `BIOMOD_EnsembleForecasting()`: a map is the fit applied to one target per cell, at the grain
#' the fit reads.
#'
#' The record of a cell is its values through the layers of `series`, which are placed in time by
#' [terra::time()], and the static predictors of a cell are its values in the layers of `static`,
#' named as the columns of `targets` were when the fit was made. A fit made with `coords` reads the
#' centre of each cell. Cells are predicted in chunks, so a raster larger than memory is read a
#' chunk of cells at a time. A cell is predicted where every input holds a value at that cell; a
#' cell with a missing reading anywhere is `NA` in every layer.
#'
#' A map for a later period is the same call with the later record, and the binary maps of the
#' two are what [range_change()] compares.
#'
#' @param fit A [timesift()] result.
#' @param series A [terra::SpatRaster] whose layers are instants, for a fit made on a record; for a
#'   record of several variables, a list of them named as the fit's `x` columns. Every raster
#'   shares one geometry and one set of instants.
#' @param static A `SpatRaster` whose layers are named as the fit's `static` columns.
#' @param candidate,type,... As for [predict.timesift()]: the candidate (the ensemble by default),
#'   `"response"`, `"binary"` or `"spread"`, and the arguments they take.
#' @param chunk Number of cells predicted at once.
#'
#' @return A `SpatRaster` of the geometry of the inputs, one layer per response. Under
#'   `type = "spread"` one layer per response and statistic, named `response.statistic`.
#'
#' @examples
#' if (requireNamespace("terra", quietly = TRUE)) {
#'   set.seed(1)
#'   r <- terra::rast(nrows = 6, ncols = 5, xmin = 0, xmax = 5, ymin = 0, ymax = 6)
#'   elev <- terra::setValues(r, rnorm(30))
#'   slope <- terra::setValues(r, rnorm(30))
#'   names(elev) <- "elev"
#'   names(slope) <- "slope"
#'   static <- c(elev, slope)
#'   cells <- data.frame(cell = seq_len(30), elev = elev[][, 1], slope = slope[][, 1])
#'   cells$sp1 <- rbinom(30, 1, plogis(cells$elev))
#'   cells$sp2 <- rbinom(30, 1, plogis(-cells$slope))
#'   fit <- timesift(cells, y = c(sp1, sp2), id = cell, static = c(elev, slope),
#'                   learners = elasticnet(), ensemble = FALSE, n_inner = NULL,
#'                   resampling = cv(v = 3), verbose = FALSE)
#'   project(fit, static = static, candidate = "elasticnet / static")
#' }
#'
#' @export
project <- function(fit, series = NULL, static = NULL, candidate = "ensemble",
                    type = c("response", "binary", "spread"), chunk = 5000L, ...) {
  .check_run(fit)
  type <- match.arg(type)
  if (!requireNamespace("terra", quietly = TRUE)) {
    stop("`project()` reads rasters through terra. Install it with install.packages(\"terra\").",
         call. = FALSE)
  }
  if (!is.numeric(chunk) || length(chunk) != 1L || chunk < 1L) {
    stop("`chunk` is a number of cells of 1 or more.", call. = FALSE)
  }
  spec <- fit$spec
  if (!is.null(spec$target_time)) {
    stop("this fit anchors each target in time with `target_time`, and a raster cell has one ",
         "record: project a fit made on calendar grains.", call. = FALSE)
  }
  inputs <- .project_inputs(spec, series, static)
  template <- inputs$template
  total <- terra::ncell(template)

  mask <- rep(TRUE, total)
  for (r in inputs$static) mask <- mask & stats::complete.cases(terra::values(r))
  for (r in inputs$series) mask <- mask & !is.na(terra::values(r[[1L]])[, 1L])
  cells <- which(mask)
  pieces <- split(cells, ceiling(seq_along(cells) / chunk))

  out <- NULL
  for (piece in pieces) {
    p <- .project_chunk(fit, inputs, piece, candidate, type, ...)
    if (is.null(out)) {
      layers <- colnames(p)
      out <- matrix(NA_real_, nrow = total, ncol = length(layers), dimnames = list(NULL, layers))
    }
    out[piece, ] <- p[match(as.character(piece), rownames(p)), , drop = FALSE]
  }
  if (is.null(out)) {
    stop("no cell holds a value in every input, so there is nothing to predict.", call. = FALSE)
  }
  result <- terra::rast(template, nlyrs = ncol(out))
  names(result) <- colnames(out)
  terra::values(result) <- out
  result
}

# The rasters a projection reads, checked against what the fit was made on, and one of them as the
# geometry the result is laid on.
.project_inputs <- function(spec, series, static) {
  if (is.null(spec$id)) {
    stop("a projection names each cell by its number, so the fit has to have been made with ",
         "`id`.", call. = FALSE)
  }
  wanted <- spec$value
  if (length(wanted)) {
    if (is.null(series)) {
      stop("this fit reads a record of ", .listing(wanted), ", so `series` is needed: a SpatRaster ",
           "whose layers are instants, or a list of them named for the variables.", call. = FALSE)
    }
    if (inherits(series, "SpatRaster")) {
      series <- stats::setNames(list(series), wanted[1L])
    }
    absent <- setdiff(wanted, names(series))
    if (is.null(names(series)) || length(absent)) {
      stop("`series` has to be a list of rasters named ", .listing(wanted), ".", call. = FALSE)
    }
    series <- series[wanted]
    times <- lapply(series, terra::time)
    if (any(vapply(times, function(t) anyNA(t) || is.null(t), logical(1L)))) {
      stop("every layer of `series` needs an instant: set it with terra::time().", call. = FALSE)
    }
    if (length(unique(lapply(times, as.numeric))) > 1L) {
      stop("the variables of `series` have to share their instants.", call. = FALSE)
    }
  } else {
    series <- list()
  }
  if (length(spec$static)) {
    if (!inherits(static, "SpatRaster") || !all(spec$static %in% names(static))) {
      stop("`static` has to be a SpatRaster with layers named ", .listing(spec$static), ".",
           call. = FALSE)
    }
    static <- list(static[[spec$static]])
  } else {
    static <- list()
  }
  rasters <- c(series, static)
  template <- rasters[[1L]]
  same <- vapply(rasters, function(r) terra::compareGeom(template, r, stopOnError = FALSE),
                 logical(1L))
  if (!all(same)) {
    stop("the inputs have to share one geometry: the same extent, resolution and projection.",
         call. = FALSE)
  }
  list(series = series, static = static, template = template[[1L]])
}

# One chunk of cells as the targets and the long record `predict()` reads.
.project_chunk <- function(fit, inputs, cells, candidate, type, ...) {
  spec <- fit$spec
  targets <- data.frame(cells, stringsAsFactors = FALSE)
  names(targets) <- spec$id
  for (r in inputs$static) {
    values <- as.matrix(r[cells])
    for (nm in colnames(values)) targets[[nm]] <- as.numeric(values[, nm])
  }
  if (length(spec$coords)) {
    xy <- terra::xyFromCell(inputs$template, cells)
    for (i in seq_along(spec$coords)) targets[[spec$coords[i]]] <- xy[, i]
  }
  series <- NULL
  if (length(inputs$series)) {
    read <- lapply(inputs$series, function(r) as.matrix(r[cells]))
    keep <- Reduce(`&`, lapply(read, stats::complete.cases))
    targets <- targets[keep, , drop = FALSE]
    for (v in names(read)) {
      m <- read[[v]][keep, , drop = FALSE]
      when <- terra::time(inputs$series[[v]])
      if (!inherits(when, "POSIXct")) {
        when <- as.POSIXct(as.character(when), tz = "UTC")
      }
      block <- data.frame(id = rep(targets[[1L]], times = ncol(m)),
                          time = rep(when, each = nrow(m)), value = as.numeric(m))
      names(block) <- c(spec$id, spec$time, v)
      series <- if (is.null(series)) block else cbind(series, block[v])
    }
  }
  p <- stats::predict(fit, targets, series, candidate = candidate, type = type, ...)
  if (length(dim(p)) == 3L) {
    stats <- dimnames(p)[[3L]]
    flat <- do.call(cbind, lapply(seq_along(stats), function(s) p[, , s]))
    colnames(flat) <- unlist(lapply(stats, function(s) paste(colnames(p), s, sep = ".")))
    rownames(flat) <- rownames(p)
    return(flat)
  }
  p
}
