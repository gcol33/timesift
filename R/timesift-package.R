#' timesift: Learn Predictive Representations of Time-Varying Data
#'
#' Finds the time scale at which a long time-stamped record best predicts a response. [timesift()]
#' builds the record at each grain or lookback named in `sift`, fits every learner on every
#' representation it can read, scores them all on the same held-out folds and cells, and chooses
#' or stacks them inside each training fold. The arrays themselves come from [grain_matrix()] and
#' [lookback_matrix()] without the fitting layer. The shipped default is presence-absence with a
#' joint multi-label head, scored by AUC with the true skill statistic beside it.
#'
#' Start with `vignette("timesift")`, or the articles at <https://gillescolling.com/timesift/>.
#'
#' @keywords internal
#' @useDynLib timesift, .registration = TRUE
"_PACKAGE"
