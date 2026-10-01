#' How the range of each response changes between two projections
#'
#' Counts the cells a response is lost from, kept in and gained, between a map for the present and
#' a map for another period or scenario, as `BIOMOD_RangeSize()` does. A cell is in the range
#' where the response is predicted present.
#'
#' With `L` the cells lost, `K` the cells kept, `G` the cells gained and `A` the cells absent in
#' both: the current range is `L + K`, the later one `K + G`, `percent_loss` is `100 L / (L + K)`,
#' `percent_gain` is `100 G / (L + K)` and `change` is `percent_gain - percent_loss`, the change in
#' range size as a share of the current one. A cell that is `NA` in either map is left out of
#' every count.
#'
#' The `map` codes a cell as biomod2 does: `-2` lost, `-1` kept, `0` absent in both and `1`
#' gained.
#'
#' @param now,later The two maps: a [terra::SpatRaster] with a layer per response, such as
#'   [project()] with `type = "binary"` returns, or a matrix of cells by responses. Layers are
#'   matched by position and must be as many.
#' @param threshold `NULL` where the maps are already 0 and 1. Otherwise one cut per response, or
#'   one for all, at or above which a cell is present; the cuts [decision_threshold()] learns
#'   for a fit are what to give.
#'
#' @return A `timesift_range_change`: a list with the `table`, one row per response, and the
#'   `map`, a raster or a matrix of the codes above, of the shape of the inputs.
#'
#' @examples
#' now <- cbind(sp1 = c(1, 1, 1, 0, 0, NA), sp2 = c(0, 0, 1, 1, 1, 1))
#' later <- cbind(sp1 = c(1, 0, 0, 1, 1, 1), sp2 = c(0, 0, 1, 1, 0, 0))
#' range_change(now, later)$table
#'
#' @export
range_change <- function(now, later, threshold = NULL) {
  a <- .range_matrix(now, "now")
  b <- .range_matrix(later, "later")
  if (!identical(dim(a), dim(b))) {
    stop("`now` and `later` have to hold the same cells and the same responses, got ",
         paste(dim(a), collapse = " x "), " and ", paste(dim(b), collapse = " x "), ".",
         call. = FALSE)
  }
  a <- .range_present(a, threshold, "now")
  b <- .range_present(b, threshold, "later")
  codes <- ifelse(a == 1 & b == 0, -2, ifelse(a == 1 & b == 1, -1, ifelse(a == 0 & b == 1, 1, 0)))
  dimnames(codes) <- dimnames(a)
  count <- function(code) colSums(codes == code, na.rm = TRUE)
  lost <- count(-2)
  kept <- count(-1)
  gained <- count(1)
  absent <- count(0)
  current <- lost + kept
  table <- data.frame(
    variable = colnames(a), lost = lost, kept = kept, gained = gained, absent = absent,
    current = current, later = kept + gained,
    percent_loss = ifelse(current > 0, 100 * lost / current, NA_real_),
    percent_gain = ifelse(current > 0, 100 * gained / current, NA_real_),
    stringsAsFactors = FALSE)
  table$change <- table$percent_gain - table$percent_loss
  rownames(table) <- NULL
  map <- if (inherits(now, "SpatRaster")) {
    out <- terra::rast(now[[1L]], nlyrs = ncol(codes))
    names(out) <- colnames(codes)
    terra::values(out) <- codes
    out
  } else {
    codes
  }
  structure(list(table = table, map = map), class = "timesift_range_change")
}

.range_matrix <- function(x, what) {
  if (inherits(x, "SpatRaster")) {
    m <- terra::values(x)
    colnames(m) <- names(x)
    return(m)
  }
  m <- as.matrix(x)
  if (!is.numeric(m)) {
    stop("`", what, "` is a raster or a numeric matrix of cells by responses.", call. = FALSE)
  }
  if (is.null(colnames(m))) {
    colnames(m) <- paste0("V", seq_len(ncol(m)))
  }
  m
}

# A map as 0 and 1, cut at `threshold` where it is not one already.
.range_present <- function(m, threshold, what) {
  if (is.null(threshold)) {
    if (!all(m[!is.na(m)] %in% c(0, 1))) {
      stop("`", what, "` holds values other than 0 and 1: give `threshold`, the cut at or above ",
           "which a cell is present.", call. = FALSE)
    }
    return(m)
  }
  if (!is.numeric(threshold) || !length(threshold) %in% c(1L, ncol(m)) || anyNA(threshold)) {
    stop("`threshold` is one cut, or one per response.", call. = FALSE)
  }
  cut <- matrix(rep_len(threshold, ncol(m)), nrow = nrow(m), ncol = ncol(m), byrow = TRUE)
  out <- (m >= cut) * 1
  dimnames(out) <- dimnames(m)
  out
}

#' @export
print.timesift_range_change <- function(x, ...) {
  cat("<timesift range change>", .plural(nrow(x$table), "response"), "\n")
  print(x$table, digits = 4L, row.names = FALSE)
  invisible(x)
}

#' Draw the change in range
#'
#' The map of cells lost, kept, absent and gained, for one response.
#'
#' @param x A [range_change()] result made from rasters.
#' @param variable The response to draw; the first by default.
#' @param ... Passed to [terra::plot()].
#'
#' @return `x`, invisibly.
#'
#' @export
plot.timesift_range_change <- function(x, variable = NULL, ...) {
  if (!inherits(x$map, "SpatRaster")) {
    stop("only a change in range made from rasters can be drawn.", call. = FALSE)
  }
  variable <- variable %||% x$table$variable[1L]
  layer <- x$map[[variable]]
  terra::plot(layer, type = "interval", breaks = c(-2.5, -1.5, -0.5, 0.5, 1.5),
              col = c("#C0392B", "#7F8C8D", "#ECF0F1", "#27AE60"), main = variable, ...)
  invisible(x)
}
