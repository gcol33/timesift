#' How a prediction responds to one predictor
#'
#' Varies one predictor of a fitted candidate across the range it takes while every other predictor
#' is held at a reference value, and records the prediction, as biomod2's response curves do. A
#' predictor is a cell of the representation the candidate reads: one statistic in one bin. Given as
#' the name of a channel, `"warm_day"`, it is that statistic moved together in every bin, which is
#' the way a column named in `static` is moved and the way a statistic of the whole record is
#' asked about. Given as `list(bin = "2021-11", channel = "mean")` it is one cell, and a bare bin
#' name is enough where the representation has a single channel.
#'
#' The reference is one made-up unit whose every cell holds the `fixed` summary of that cell over
#' the targets. A cell that is the same for every target, as a calendar channel is, keeps the value
#' it has. The curve is read off the model fitted on all targets, not off the per-fold models.
#'
#' For the ensemble, every member is moved the same way: a member that does not carry the predictor
#' is held at the reference, and the members' predictions are combined by the stack. A member
#' reading another grain carries a channel by its name, so an ensemble is asked about by channel.
#'
#' @param x A [timesift()] result.
#' @param candidate The candidate to read, as `summary()` names it, or `"ensemble"`.
#' @param predictor The predictor to vary: a channel, a bin of a one-channel representation, or
#'   `list(bin = , channel = )`.
#' @param with A second predictor, given the same way. The prediction is then read over the grid of
#'   the two.
#' @param fixed What the other predictors are held at: `"mean"`, `"median"`, `"min"` or `"max"` of
#'   the cell over the targets.
#' @param n Number of values of a predictor, equally spaced over the range it takes.
#' @param spread For the ensemble, also return the members' standard deviation and interval at
#'   every value, as `predict(type = "spread")` reads them.
#' @param ... Passed to the method.
#'
#' @return A data frame of one row per value and response, of class `timesift_response_curve`:
#'   the value of the predictor (`value`, and `value_with` for a second one), the response
#'   (`variable`) and the `prediction`, with `sd`, `lower` and `upper` under `spread`.
#'
#' @examples
#' set.seed(1)
#' t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24 * 120)
#' units <- sprintf("p%02d", 1:40)
#' warmth <- rnorm(40)
#' d <- data.frame(
#'   plot = rep(units, each = length(t)), t = rep(t, length(units)),
#'   temp = as.numeric(vapply(warmth, function(w) w + sin(seq_along(t) / 300) + rnorm(length(t)),
#'                            numeric(length(t)))))
#' targets <- data.frame(plot = units, sp1 = rbinom(40, 1, plogis(warmth)),
#'                       sp2 = rbinom(40, 1, plogis(-warmth)))
#' fit <- timesift(targets, d, y = starts_with("sp"), id = plot, time = t, x = temp,
#'                 learners = elasticnet(),
#'                 sift = grains("month", stats = c("cold_day", "warm_day")),
#'                 ensemble = FALSE, n_inner = NULL, resampling = cv(v = 3), verbose = FALSE)
#' head(response_curve(fit, "elasticnet / month", "warm_day", n = 5))
#'
#' @export
response_curve <- function(x, ...) UseMethod("response_curve")

#' @rdname response_curve
#' @export
response_curve.default <- function(x, ...) {
  stop("expected a timesift() result, got ", class(x)[1L], ".", call. = FALSE)
}

#' @rdname response_curve
#' @export
response_curve.timesift <- function(x, candidate = "ensemble", predictor, with = NULL,
                                    fixed = c("mean", "median", "min", "max"), n = 50L,
                                    spread = FALSE, ...) {
  fixed <- match.arg(fixed)
  if (missing(predictor)) {
    stop("`predictor` names what to vary: a channel, a bin, or list(bin = , channel = ).",
         call. = FALSE)
  }
  if (!is.numeric(n) || length(n) != 1L || n < 2L) {
    stop("`n` is a number of values of 2 or more.", call. = FALSE)
  }
  n <- as.integer(n)
  ensemble <- identical(candidate, "ensemble")
  if (ensemble) {
    if (is.null(x$stack)) {
      stop("this run fitted no ensemble to read.", call. = FALSE)
    }
    members <- names(x$stack$weights)
  } else {
    if (spread) {
      stop("`spread` is the disagreement of an ensemble's members.", call. = FALSE)
    }
    members <- candidate
  }
  rows <- lapply(members, function(cd) .candidate_row(x, cd))
  arrays <- lapply(rows, function(r) x$representations[[r$representation]])
  models <- lapply(members, function(cd) x$models[[cd]])

  first <- .curve_cell(arrays, predictor, "predictor")
  second <- if (!is.null(with)) .curve_cell(arrays, with, "with")
  grid <- .curve_grid(first$range, if (!is.null(second)) second$range, n)

  preds <- Map(function(m, model) {
    stats::predict(model, .curve_array(m, grid, first$name, second$name, fixed, length(grid$value)))
  }, arrays, models)
  names(preds) <- members
  p <- if (ensemble) ensemble_combine(x$stack, preds) else preds[[1L]]
  variables <- colnames(x$y)
  out <- data.frame(value = rep(grid$value, times = length(variables)),
                    variable = rep(variables, each = length(grid$value)),
                    prediction = as.numeric(p), stringsAsFactors = FALSE)
  if (!is.null(second)) {
    out$value_with <- rep(grid$with, times = length(variables))
    out <- out[c("value", "value_with", "variable", "prediction")]
  }
  if (spread) {
    s <- ensemble_spread(x$stack, preds)
    for (stat in c("sd", "lower", "upper")) {
      out[[stat]] <- as.numeric(s[, , stat])
    }
  }
  structure(out, class = c("timesift_response_curve", "data.frame"), candidate = candidate,
            predictor = first$label, with = second$label, fixed = fixed)
}

# A predictor resolved against every array of a candidate: the cell it names in each, the range it
# takes over the targets, and a label. An array that does not carry the predictor has no cell, and
# is held at the reference.
.curve_cell <- function(arrays, predictor, what) {
  spec <- NULL
  for (m in arrays) {
    spec <- tryCatch(.curve_spec(m, predictor, what), error = function(e) NULL)
    if (!is.null(spec)) break
  }
  if (is.null(spec)) {
    .curve_spec(arrays[[1L]], predictor, what)
  }
  cells <- lapply(arrays, function(m) .curve_find(m, spec))
  carried <- !vapply(cells, is.null, logical(1L))
  if (!any(carried)) {
    stop("no representation of this candidate carries `", .curve_label(spec), "`.", call. = FALSE)
  }
  values <- unlist(Map(function(m, cell) {
    if (is.null(cell)) numeric() else as.numeric(.curve_slice(m, cell))
  }, arrays, cells), use.names = FALSE)
  range <- range(values)
  if (!is.finite(diff(range)) || diff(range) == 0) {
    stop("`", .curve_label(spec), "` takes one value over the targets, so there is no curve.",
         call. = FALSE)
  }
  list(range = range, name = spec, label = .curve_label(spec))
}

# A predictor as a list(bin, channel) with NULL for "every bin". A string names a channel where one
# of the first array's channels is called that, and a bin where the representation has one channel.
.curve_spec <- function(m, predictor, what) {
  if (is.list(predictor)) {
    return(list(bin = predictor$bin, channel = predictor$channel))
  }
  if (!is.character(predictor) || length(predictor) != 1L) {
    stop("`", what, "` is a channel, a bin, or list(bin = , channel = ).", call. = FALSE)
  }
  if (predictor %in% dimnames(m)[[3L]]) {
    return(list(bin = NULL, channel = predictor))
  }
  if (dim(m)[3L] == 1L && predictor %in% dimnames(m)[[2L]]) {
    return(list(bin = predictor, channel = dimnames(m)[[3L]]))
  }
  stop("`", what, "` is \"", predictor, "\", which is neither a channel nor a bin of a ",
       "one-channel representation. The channels are ", .listing(dimnames(m)[[3L]]), ".",
       call. = FALSE)
}

.curve_label <- function(spec) {
  if (is.null(spec$bin)) spec$channel else paste(spec$channel, spec$bin, sep = " @ ")
}

# The indices of the cell a spec names in one array, or NULL where the array lacks it.
.curve_find <- function(m, spec) {
  channel <- match(spec$channel, dimnames(m)[[3L]])
  bin <- if (is.null(spec$bin)) seq_len(dim(m)[2L]) else match(spec$bin, dimnames(m)[[2L]])
  if (is.na(channel) || anyNA(bin)) NULL else list(bin = bin, channel = channel)
}

.curve_slice <- function(m, cell) {
  m[, cell$bin, cell$channel]
}

.curve_grid <- function(range, range_with, n) {
  step <- function(r) r[1L] + (r[2L] - r[1L]) * (seq_len(n) - 1L) / (n - 1L)
  value <- step(range)
  if (is.null(range_with)) {
    return(list(value = value, with = NULL))
  }
  with <- step(range_with)
  list(value = rep(value, times = n), with = rep(with, each = n))
}

# The made-up units: `n` copies of the reference, the predictor cells set to the grid.
.curve_array <- function(m, grid, spec, spec_with, fixed, n) {
  summary <- switch(fixed, mean = mean, median = stats::median, min = min, max = max)
  reference <- apply(m, c(2L, 3L), summary)
  sub <- .subset_units(m, rep(1L, n))
  dimnames(sub)[[1L]] <- sprintf("curve%04d", seq_len(n))
  for (i in seq_len(n)) {
    sub[i, , ] <- reference
  }
  set <- function(sub, spec, values) {
    cell <- .curve_find(m, spec)
    if (is.null(cell)) {
      return(sub)
    }
    for (b in cell$bin) {
      sub[, b, cell$channel] <- values
    }
    sub
  }
  sub <- set(sub, spec, grid$value)
  if (!is.null(spec_with)) {
    sub <- set(sub, spec_with, grid$with)
  }
  sub
}

#' @export
print.timesift_response_curve <- function(x, ...) {
  cat("<timesift response curve>", attr(x, "candidate"), "over", attr(x, "predictor"),
      if (!is.null(attr(x, "with"))) paste("and", attr(x, "with")), "with the rest at their",
      attr(x, "fixed"), "\n")
  print(utils::head(as.data.frame(x), 6L))
  invisible(x)
}

#' Draw a response curve
#'
#' One line per response against the value of the predictor. For two predictors, the prediction of
#' one response as a filled image over the grid of the two.
#'
#' @param x A [response_curve()] result.
#' @param variable For two predictors, the response to draw; the first by default.
#' @param col One colour per response, recycled.
#' @param ... Passed to [graphics::plot()] or [graphics::image()].
#'
#' @return The data frame drawn, invisibly.
#'
#' @export
plot.timesift_response_curve <- function(x, variable = NULL, col = NULL, ...) {
  tbl <- as.data.frame(x)
  if ("value_with" %in% names(tbl)) {
    variable <- variable %||% tbl$variable[1L]
    one <- tbl[tbl$variable == variable, , drop = FALSE]
    values <- sort(unique(one$value))
    withs <- sort(unique(one$value_with))
    z <- matrix(one$prediction[order(one$value_with, one$value)], nrow = length(values))
    graphics::image(values, withs, z, xlab = attr(x, "predictor"), ylab = attr(x, "with"),
                    main = variable, ...)
    return(invisible(tbl))
  }
  variables <- unique(tbl$variable)
  col <- rep_len(col %||% seq_along(variables), length(variables))
  graphics::plot(range(tbl$value), range(tbl$prediction, na.rm = TRUE), type = "n",
                 xlab = attr(x, "predictor"), ylab = "prediction", ...)
  for (i in seq_along(variables)) {
    one <- tbl[tbl$variable == variables[i], , drop = FALSE]
    graphics::lines(one$value, one$prediction, col = col[i], lwd = 2)
  }
  graphics::legend("topright", legend = variables, col = col, lwd = 2, bty = "n")
  invisible(tbl)
}
