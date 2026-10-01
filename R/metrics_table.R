# The threshold rules every consumer of a cut accepts, in the order the signatures list them.
.threshold_rules <- c("youden", "kappa", "prevalence", "mpa")

# Each metric of a two-by-two table of decisions against observations, as a function of its four
# cells: hits `tp`, false alarms `fp`, misses `fn` and correct negatives `tn`.
.table_metrics <- list(
  pod = function(tp, fp, fn, tn) tp / (tp + fn),
  pofd = function(tp, fp, fn, tn) fp / (fp + tn),
  far = function(tp, fp, fn, tn) fp / (tp + fp),
  sr = function(tp, fp, fn, tn) tp / (tp + fp),
  accuracy = function(tp, fp, fn, tn) (tp + tn) / (tp + fp + fn + tn),
  bias = function(tp, fp, fn, tn) (tp + fp) / (tp + fn),
  or = function(tp, fp, fn, tn) tp * tn / (fn * fp),
  orss = function(tp, fp, fn, tn) (tp * tn - fn * fp) / (tp * tn + fn * fp),
  csi = function(tp, fp, fn, tn) tp / (tp + fn + fp),
  ets = function(tp, fp, fn, tn) {
    by_chance <- (tp + fn) * (tp + fp) / (tp + fp + fn + tn)
    (tp - by_chance) / (tp + fn + fp - by_chance)
  }
)

#' A metric of the two-by-two table of decisions
#'
#' The predictions are cut into presence and absence, and the table of decisions against
#' observations is summarised. With `H` the hits, `F` the false alarms, `M` the misses and `C` the
#' correct negatives:
#'
#' | `metric` | reads |
#' |---|---|
#' | `"pod"` | probability of detection, `H / (H + M)` |
#' | `"pofd"` | probability of false detection, `F / (F + C)` |
#' | `"far"` | false alarm ratio, `F / (H + F)` |
#' | `"sr"` | success ratio, `H / (H + F)` |
#' | `"accuracy"` | `(H + C) / n` |
#' | `"bias"` | `(H + F) / (H + M)`, the presences called against the presences there are |
#' | `"or"` | odds ratio, `H C / (M F)` |
#' | `"orss"` | odds ratio skill score, `(H C - M F) / (H C + M F)` |
#' | `"csi"` | critical success index, `H / (H + M + F)` |
#' | `"ets"` | equitable threat score, `(H - h) / (H + M + F - h)` with `h = (H + M)(H + F) / n` |
#'
#' These are biomod2's evaluation statistics. biomod2 reads each at the cut that brings that
#' statistic closest to its own optimum on a grid of 100 cuts; here the cut is the one `rule` of
#' [decision_threshold()] selects, `"youden"` by default, or a `threshold` learned elsewhere. Read
#' at the cut that optimises it, some of these are trivial, so the cut is a separate choice and
#' stays fixed while the statistic changes. Each is registered under its name at the default rule,
#' so `grain_ladder(metric = "csi")` reads it. A registration at another rule is one line:
#' `register_metric("csi_kappa", function(y, p) table_metric(y, p, "csi", "kappa"))`.
#'
#' A value the table does not define, a zero denominator, is `NA`, as is a cell of one class.
#'
#' @inheritParams tss
#' @inheritParams kappa_score
#' @param metric One of the names in the table.
#'
#' @return One number, or `NA` where the cell defines none.
#'
#' @examples
#' y <- c(0, 0, 0, 1, 1, 1, 0, 1)
#' p <- c(0.10, 0.20, 0.35, 0.40, 0.60, 0.90, 0.55, 0.70)
#' table_metric(y, p, "pod")
#' table_metric(y, p, "csi", threshold = 0.5)
#'
#' @export
table_metric <- function(y, p, metric = names(.table_metrics),
                         rule = c("youden", "kappa", "prevalence", "mpa"), threshold = NULL,
                         perc = 0.9) {
  metric <- match.arg(metric)
  rule <- match.arg(rule)
  cut <- if (is.null(threshold)) decision_threshold.default(y, p, rule, perc) else threshold
  y <- .check_labels(y, p)
  if (is.null(y) || length(cut) != 1L || !is.finite(cut)) {
    return(NA_real_)
  }
  hit <- as.numeric(p) >= cut
  value <- .table_metrics[[metric]](tp = sum(hit & y == 1L), fp = sum(hit & y == 0L),
                                    fn = sum(!hit & y == 1L), tn = sum(!hit & y == 0L))
  if (is.finite(value)) value else NA_real_
}

#' The continuous Boyce index
#'
#' How consistently the predictions rise with the density of presences, without a cut and without
#' absences (Hirzel et al. 2006). A window of `width` times the range of the predictions slides over
#' that range in `resolution + 1` equal steps. In each, the share of presences falling inside
#' divided by the share of all units falling inside is the predicted-to-expected ratio, undefined
#' where no unit falls inside, and the index is the Spearman correlation of that ratio with the
#' window's midpoint, between -1 and 1. Predictions that rank no better than chance read near 0.
#'
#' The units of `p` are the background, which is how biomod2 reads `BOYCE` on a fit. The windows are
#' closed at both ends and the ratio is not thinned of repeated values.
#'
#' @inheritParams tss
#' @param resolution Number of window steps, `100` by default.
#' @param width Window width as a share of the range of `p`, `0.1` by default.
#'
#' @return One number, or `NA` where the cell defines none, where the predictions are all equal,
#'   or where the ratio is defined in fewer than three windows or takes one value.
#'
#' @examples
#' boyce_index(c(0, 0, 0, 1, 1, 1, 0, 1), c(0.10, 0.20, 0.35, 0.40, 0.60, 0.90, 0.55, 0.70))
#'
#' @export
boyce_index <- function(y, p, resolution = 100L, width = 0.1) {
  y <- .check_labels(y, p)
  if (is.null(y)) {
    return(NA_real_)
  }
  p <- as.numeric(p)
  lo <- min(p)
  hi <- max(p)
  if (hi == lo) {
    return(NA_real_)
  }
  w <- (hi - lo) * width
  steps <- seq.int(0L, as.integer(resolution))
  from <- lo + (hi - w - lo) * steps / resolution
  to <- from + w
  to[length(to)] <- hi
  inside <- function(x) {
    vapply(seq_along(from), function(i) sum(x >= from[i] & x <= to[i]), numeric(1L))
  }
  ratio <- (inside(p[y == 1L]) / sum(y == 1L)) / (inside(p) / length(p))
  keep <- is.finite(ratio)
  if (sum(keep) < 3L || length(unique(ratio[keep])) < 2L) {
    return(NA_real_)
  }
  stats::cor((from + to)[keep] / 2, ratio[keep], method = "spearman")
}
