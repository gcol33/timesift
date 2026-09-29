#' biomod2's surface range envelope on the flattened representation
#'
#' One envelope per response, over every bin-by-channel column of the representation: for each
#' column, the `quantile` and `1 - quantile` quantiles of its readings over the units present, and a
#' unit predicted present where every column lies between its two, the ends included. It is
#' biomod2's `SRE`, and with the same quantile it draws the envelope `bm_SRE()` draws; the quantile
#' is R's default, type 7.
#'
#' The prediction is zero or one. It enters an ensemble as that, and the combiner weighs it by its
#' held-out loss like any other candidate; a threshold read off it has two values to choose from, so
#' its TSS is the one its own zeros and ones give.
#'
#' An envelope reads the presences and nothing else. The absences do not move it, and neither do the
#' response head's case weights, which it takes no notice of.
#'
#' Every column has to agree for a unit to be inside, so the more columns a representation has the
#' fewer units any envelope holds: at the default quantile each column shuts out about one presence
#' in twenty, and a weekly representation of three years has 157 columns per channel. A coarse grain
#' is what an envelope is meant for, and pinning it there with `data = grain("season")` keeps it
#' there while the rest of the run reads finer ones.
#'
#' A response with no presence, or with nothing else, is predicted its share among the fitting units,
#' and the fit names it in `unfitted`. The learner needs a presence-absence response, and a head
#' whose loss is not the binary cross-entropy is refused.
#'
#' @inheritParams elasticnet
#' @param quantile The share of the presences left outside at each end of every column, in
#'   `[0, 0.5]`. biomod2's default is `0.025`.
#'
#' @return A [learner()].
#'
#' @examples
#' envelope()
#' envelope(data = grain("season"), quantile = 0.05)
#'
#' @export
envelope <- function(data = NULL, quantile = 0.025) {
  if (!is.numeric(quantile) || length(quantile) != 1L || !is.finite(quantile) ||
      quantile < 0 || quantile > 0.5) {
    stop("`quantile` is one number in [0, 0.5], got ", .describe(quantile), ".", call. = FALSE)
  }
  learner(
    name = "envelope",
    data = data, reads = "tabular", multi = "separate",
    params = list(quantile = quantile),
    fit = function(x, y, quantile, head, ...) {
      if (!identical(.head_family(head), "binomial")) {
        stop("the envelope is drawn around presences, under a head whose loss is the binary ",
             "cross-entropy; this head's loss is ", .describe(head$loss), ".", call. = FALSE)
      }
      m <- .flatten(x)
      models <- lapply(seq_len(ncol(y)), function(j) {
        yj <- y[, j]
        if (length(unique(yj)) < 2L) {
          return(mean(yj))
        }
        .envelope_fit(m, yj, quantile)
      })
      list(models = models, columns = colnames(m),
           unfitted = colnames(y)[vapply(models, is.numeric, logical(1L))])
    },
    predict = function(model, x) {
      m <- .flatten(x)
      .as_predictions(vapply(model$models, function(f) {
        if (is.numeric(f)) rep(f, nrow(m)) else .envelope_predict(f, m)
      }, numeric(nrow(m))), nrow(m))
    }
  )
}

# The envelope, over the core `src/ts_envelope.cpp` compiles into both languages. A fit is each
# column's two bounds, a plain list of numbers.
.envelope_fit <- function(x, y, quantile = 0.025) {
  ts_envelope_fit_(as.numeric(x), as.numeric(y), nrow(x), ncol(x), as.numeric(quantile))
}

.envelope_predict <- function(fit, newx) {
  ts_envelope_predict_(fit, as.numeric(newx), nrow(newx), ncol(newx))
}
