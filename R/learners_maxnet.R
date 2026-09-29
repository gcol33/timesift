#' maxnet's MaxEnt on the flattened representation
#'
#' One maxnet model per response, over every bin-by-channel column of the representation: maxnet's
#' feature classes, its regularisation of each feature, and a lasso over them, fitted by the
#' penalised core [elasticnet()] runs on, which the Python package calls too. With the maxnet
#' package's own settings the features and the penalty factors are maxnet's to rounding, and the
#' fit settles at the objective glmnet reaches for maxnet.
#'
#' The feature classes are the letters of `classes`: `l` the column itself, `q` its square, `p`
#' the product of each pair of columns, `h` forward and reverse hinges at the interior of `knots`
#' equally spaced points of each column's range, and `t` thresholds at 49 interior points of it.
#' Left `NULL`, they follow the response's presence count as `maxnet.formula()` has them: `"l"`
#' under 10 presences, `"lq"` under 15, `"lqh"` under 80, and `"lqph"` from 80 on. A column holding
#' one value over the units fitted takes no feature.
#'
#' `formulation` says what the absences are. `"background"` is maxnet's own and what biomod2 fits
#' as `MAXNET`: every unit is background, each presence joins the background again unless an absence
#' carries the same readings (`add_samples`), the background is weighted 100 against a presence's
#' 1, and the model is read at the last of maxnet's 200 penalties, which scale with `regmult`. Its
#' output is maxnet's `type`, `"cloglog"` by default as biomod2 predicts it. `"absence"` reads the
#' absences as absences: a logistic lasso over the same features and penalty factors under the
#' response head's case weights, [positive_weights()] under presence-absence, with the penalty
#' chosen by an inner cross-validation dealt as the elastic net's is, and a probability as output.
#'
#' The background formulation takes no case weights, as maxnet takes none and biomod2 passes none:
#' the background weight is what sets a presence's weight there. Either formulation holds each
#' column inside the range it was fitted on, and each feature inside its own, before predicting,
#' as maxnet's `predict(clamp = TRUE)` does; `clamp = FALSE` reads them as they are.
#'
#' A hinge per column per knot makes the design large: a weekly three-channel representation,
#' 471 columns, is 47,100 features under `"lqh"`, and its products under `"lqph"` 110,685 more.
#' The design is held in memory with a centred copy beside it, and a fit whose design would take
#' more than `max_design` gigabytes is refused with the size it would have taken. A coarser
#' representation (`data = grain("month")`) or fewer classes is then the way to fit it.
#'
#' A response with fewer than two presences, or one whose inner training sets cannot each hold two
#' of each outcome under the absence formulation, is predicted its share among the fitting units,
#' and the fit names it in `unfitted`. A path that does not settle at a penalty ends there, as the
#' elastic net's does, and is read at its last settled point; the fit names every such response in
#' `stopped`.
#'
#' The learner needs a presence-absence response: maxnet has no model for a continuous one, and a
#' head whose loss is not the binary cross-entropy is refused.
#'
#' @inheritParams elasticnet
#' @param classes Feature classes, letters of `"lqpht"`, or `NULL` for maxnet's choice at each
#'   response's presence count.
#' @param regmult Multiplier on every feature's regularisation.
#' @param formulation `"background"` for maxnet's presence-background model, `"absence"` for a
#'   logistic lasso reading the absences as absences.
#' @param type The background formulation's output: `"cloglog"` or `"logistic"`. The absence
#'   formulation predicts a probability, which is its `"logistic"`.
#' @param knots Points over each column's range the hinges and thresholds are placed at.
#' @param add_samples Add each presence to the background, as maxnet's `addsamplestobackground`.
#' @param clamp Hold each column and each feature inside the range it was fitted on.
#' @param n_inner Folds of the absence formulation's inner cross-validation.
#' @param s Where the absence formulation reads its path: `"lambda.min"` or `"lambda.1se"`.
#' @param thresh Where the coordinate descent stops, as the elastic net's `thresh`.
#' @param max_design Gigabytes the expanded design may take.
#' @param threads How many fits of the absence formulation's inner cross-validation run at once.
#'   What comes back does not depend on it.
#'
#' @return A [learner()].
#'
#' @examples
#' maxnet()
#' maxnet(classes = "lqh", regmult = 2)
#' maxnet(formulation = "absence")
#'
#' @export
maxnet <- function(data = NULL, classes = NULL, regmult = 1,
                   formulation = c("background", "absence"), type = NULL, knots = 50L,
                   add_samples = TRUE, clamp = TRUE, n_inner = 5L,
                   s = c("lambda.min", "lambda.1se"), thresh = 1e-8, max_design = 2,
                   threads = 1L, seed = 1L) {
  formulation <- match.arg(formulation)
  s <- match.arg(s)
  type <- .maxnet_type(formulation, type)
  if (!is.null(classes) && (!is.character(classes) || length(classes) != 1L ||
                              !grepl("^[lqpht]+$", classes))) {
    stop("`classes` is a string of the letters l, q, p, h and t, or NULL, got ",
         .describe(classes), ".", call. = FALSE)
  }
  learner(
    name = "maxnet",
    data = data, reads = "tabular", multi = "separate",
    params = list(classes = classes, regmult = regmult, formulation = formulation, type = type,
                  knots = as.integer(knots), add_samples = isTRUE(add_samples),
                  clamp = isTRUE(clamp), n_inner = as.integer(n_inner), s = s, thresh = thresh,
                  max_design = max_design, threads = as.integer(threads), seed = as.integer(seed)),
    fit = function(x, y, classes, regmult, formulation, type, knots, add_samples, clamp, n_inner,
                   s, thresh, max_design, threads, seed, head, weights, group = NULL, ...) {
      if (!identical(.head_family(head), "binomial")) {
        stop("maxnet fits a presence-absence response, under a head whose loss is the binary ",
             "cross-entropy; this head's loss is ", .describe(head$loss), ".", call. = FALSE)
      }
      m <- .flatten(x)
      seeds <- .variable_seeds(seed, y)
      models <- lapply(seq_len(ncol(y)), function(j) {
        yj <- y[, j]
        if (length(unique(yj)) < 2L || sum(yj == 1) < 2L) {
          return(mean(yj))
        }
        fold <- NULL
        n_fold <- 0L
        if (identical(formulation, "absence")) {
          inner <- .inner_folds(yj, n_inner, seeds[j], group)
          if (!.inner_fittable(yj, inner)) {
            return(mean(yj))
          }
          labels <- sort(unique(inner))
          fold <- match(inner, labels) - 1L
          n_fold <- length(labels)
        }
        .maxnet_fit(m, yj, weights[, j], classes = classes, knots = knots, regmult = regmult,
                    formulation = formulation, add_samples = add_samples, thresh = thresh,
                    one_se = identical(s, "lambda.1se"), fold = fold, n_fold = n_fold,
                    threads = threads, max_design = max_design)
      })
      unfitted <- colnames(y)[vapply(models, is.numeric, logical(1L))]
      stopped <- colnames(y)[vapply(models, function(f) {
        is.list(f) && (f$stalled > 0L || f$fold_stalled > 0L)
      }, logical(1L))]
      list(models = models, columns = colnames(m), type = type, clamp = clamp,
           unfitted = unfitted, stopped = stopped)
    },
    predict = function(model, x) {
      m <- .flatten(x)
      .as_predictions(vapply(model$models, function(f) {
        if (is.numeric(f)) rep(f, nrow(m)) else .maxnet_predict(f, m, model$clamp, model$type)
      }, numeric(nrow(m))), nrow(m))
    }
  )
}

# The output a formulation predicts: maxnet's cloglog by default under the background, and the
# probability, which is the logistic output, under the absences.
.maxnet_type <- function(formulation, type) {
  if (is.null(type)) {
    return(if (identical(formulation, "background")) "cloglog" else "logistic")
  }
  allowed <- if (identical(formulation, "background")) c("cloglog", "logistic") else "logistic"
  if (!is.character(type) || length(type) != 1L || !type %in% allowed) {
    stop("the ", formulation, " formulation predicts ",
         paste0("\"", allowed, "\"", collapse = " or "), ", got ", .describe(type), ".",
         call. = FALSE)
  }
  type
}

# maxnet, over the core `src/ts_maxnet.cpp` compiles into both languages. Nothing here decides
# anything: the design, the response, the case weights and the inner folds are settled above, and
# what is left is to hand them over column-major. A fit is a plain list of numbers, so it round
# trips through `saveRDS()` and predicts on another machine.
.maxnet_fit <- function(x, y, w, classes = NULL, knots = 50L, regmult = 1,
                        formulation = "background", add_samples = TRUE, thresh = 1e-8,
                        max_pass = 1e8, n_lambda = 100L, one_se = FALSE, fold = NULL,
                        n_fold = 0L, threads = 1L, max_design = 2) {
  ts_maxnet_fit_(as.numeric(x), as.numeric(y), as.numeric(w), nrow(x), ncol(x),
                 classes %||% "", as.integer(knots), as.numeric(regmult), formulation,
                 isTRUE(add_samples), as.numeric(thresh), as.numeric(max_pass),
                 as.integer(n_lambda), isTRUE(one_se),
                 if (is.null(fold)) NULL else as.integer(fold), as.integer(n_fold),
                 as.integer(threads), as.numeric(max_design))
}

.maxnet_design <- function(x, y, classes = NULL, knots = 50L, regmult = 1,
                           formulation = "background", add_samples = TRUE, max_design = 2) {
  ts_maxnet_design_(as.numeric(x), as.numeric(y), nrow(x), ncol(x), classes %||% "",
                    as.integer(knots), as.numeric(regmult), formulation, isTRUE(add_samples),
                    as.numeric(max_design))
}

.maxnet_predict <- function(fit, newx, clamp = TRUE, type = "cloglog") {
  ts_maxnet_predict_(fit, as.numeric(newx), nrow(newx), ncol(newx), isTRUE(clamp), type)
}
