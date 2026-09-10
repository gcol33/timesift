#' Where in the year each bin sits
#'
#' An encoder that ends in global pooling discards when a thermal event happened, so the position
#' of a bin in the year has to be given to it as input if it is to be used at all. These two
#' channels carry that position as the sine and cosine of the bin's fractional place in the year,
#' which is continuous across the turn of the year where the fraction itself is not.
#'
#' They are the time index of each bin, not a summary of the readings, so adding them introduces no
#' hand-built thermal feature: whatever a model does with them it could have done with a calendar.
#'
#' The position is read at the midpoint of the record each bin holds, on the Gregorian calendar in
#' UTC, so a bin the record only partly covers sits at the phase it was actually measured over.
#' `inst/spec/representation.md` is the normative description.
#'
#' @param x A [grain_matrix()] result. A [lookback_matrix()] result has no place in the calendar
#'   and is refused.
#'
#' @return An array of the same units and bins with two channels, `year_sin` and `year_cos`,
#'   identical across units. Combine it with the readings using [bind_channels()].
#'
#' @examples
#' t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24 * 400)
#' d <- data.frame(plot = "a", t = t, temp = sin(seq_along(t) / 500))
#' x <- grain_matrix(d, plot, t, temp, grain = "month")
#' round(calendar_channels(x)[1, 1:4, ], 3)
#'
#' @export
calendar_channels <- function(x) {
  .check_matrix(x)
  if (is.null(attr(x, "bin_start"))) {
    stop("a lookback's bins are placed relative to a target rather than on the calendar, so they ",
         "have no position in the year. `calendar_channels()` reads a grain_matrix().",
         call. = FALSE)
  }
  phase <- ts_year_phase_(as.numeric(attr(x, "bin_start")), as.numeric(attr(x, "bin_end")))

  out <- array(rep(phase, each = dim(x)[1L]), dim = c(dim(x)[1L], dim(x)[2L], 2L),
               dimnames = list(dimnames(x)[[1L]], dimnames(x)[[2L]], c("year_sin", "year_cos")))
  .carry_attrs(out, x, stats = c("year_sin", "year_cos"))
}

#' Put channels side by side
#'
#' Joins representations of the same units and bins into one array, in the order given. It is how
#' a temperature reading, an external product such as snow cover, and the calendar position of each
#' bin reach a model as one input.
#'
#' @param ... Two or more representations of shape `[unit, bin, channel]`, agreeing on their units
#'   and bins.
#'
#' @return One array carrying every channel, in the order the arguments are given and, inside each
#'   argument, in its own channel order. The attributes are the first argument's, with `stats` the
#'   joined names.
#'
#' @examples
#' t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24 * 60)
#' d <- data.frame(plot = rep(c("a", "b"), each = length(t)), t = rep(t, 2),
#'                 temp = rnorm(2 * length(t)))
#' x <- grain_matrix(d, plot, t, temp, grain = "week")
#' dimnames(bind_channels(x, calendar_channels(x)))[[3]]
#'
#' @export
bind_channels <- function(...) {
  parts <- list(...)
  if (length(parts) < 2L) {
    stop("`bind_channels()` needs at least two representations.", call. = FALSE)
  }
  first <- parts[[1L]]
  for (k in seq_along(parts)) {
    if (!inherits(parts[[k]], "timesift_matrix")) {
      stop("argument ", k, " is a ", class(parts[[k]])[1L], ", not a representation.",
           call. = FALSE)
    }
    if (!identical(dimnames(parts[[k]])[1:2], dimnames(first)[1:2])) {
      stop("argument ", k, " covers different units or bins from the first.", call. = FALSE)
    }
  }
  names <- unlist(lapply(parts, function(p) dimnames(p)[[3L]]))
  if (anyDuplicated(names)) {
    stop("two representations carry a channel of the same name: ",
         paste(unique(names[duplicated(names)]), collapse = ", "), ".", call. = FALSE)
  }
  out <- array(unlist(lapply(parts, as.numeric), use.names = FALSE),
               dim = c(dim(first)[1:2], length(names)),
               dimnames = c(dimnames(first)[1:2], list(names)))
  .carry_attrs(out, first, stats = names)
}

.check_matrix <- function(x) {
  if (!inherits(x, "timesift_matrix")) {
    stop("expected a grain_matrix() result, got ", class(x)[1L], ".", call. = FALSE)
  }
  invisible(TRUE)
}

.carry_attrs <- function(out, from, stats) {
  attr(out, "grain") <- attr(from, "grain")
  attr(out, "stats") <- stats
  attr(out, "year_start") <- attr(from, "year_start")
  attr(out, "bin_start") <- attr(from, "bin_start")
  attr(out, "bin_end") <- attr(from, "bin_end")
  attr(out, "bin_n") <- attr(from, "bin_n")
  attr(out, "bin_partial") <- attr(from, "bin_partial")
  class(out) <- c("timesift_matrix", "array")
  out
}
