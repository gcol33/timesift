#' Define a learner
#'
#' A learner is a pair of functions: one that fits a model to a representation and a response, and
#' one that predicts from it. Everything the package fits goes through this pair, so a learner of
#' your own sits beside the ones that ship and needs no change to the ladder, the folds or the
#' scoring.
#'
#' A learner declares what it can be handed and how it covers several responses. `reads` is
#' `"tabular"` where the bins reach it as a block of predictors and `"sequence"` where their order
#' in time is what it reads, and `multi` is `"joint"` where one fitted model covers every response
#' and `"separate"` where one is fitted per response. Either way a candidate emits one
#' `[target, response]` matrix, so nothing above the learner layer has to know which it was.
#'
#' `data` pins a learner to one representation. Left `NULL` the learner runs across every
#' representation of the run.
#'
#' @param name Name the learner is reported under.
#' @param fit A function of `(x, y, ...)`, where `x` is a `[unit, bin, channel]` array and `y` the
#'   response matrix for the same units, returning a fitted object. A `fit` that declares a
#'   `control` argument is handed the resolved [train_control()], and one that declares a `head`
#'   argument is handed the registered response head, whose `loss` and `activation` say what it
#'   is fitting toward. The learners that ship read both from there and hold no response of their
#'   own.
#' @param predict A function of `(model, x)` returning a `[unit, variable]` matrix of predictions
#'   for the units of `x`, in that order.
#' @param data A representation the learner is pinned to, or `NULL` to run across every
#'   representation of the run.
#' @param reads `"tabular"` or `"sequence"`.
#' @param multi `"separate"` where one model is fitted per response, `"joint"` where one model
#'   covers them all.
#' @param control A [train_control()] for this learner alone. The settings it names override the
#'   control the run was given; everything else is taken from that one.
#' @param needs Packages the learner requires. A learner that cannot run says so at once rather
#'   than falling back to something else.
#' @param params Settings carried with the learner and passed to `fit`.
#'
#' @return A `timesift_learner`.
#'
#' @examples
#' # The bin means of a unit, fed to one logistic regression per variable.
#' flat_glm <- learner(
#'   "flat_glm",
#'   fit = function(x, y, ...) {
#'     f <- as.data.frame(apply(x, c(1, 3), mean))
#'     lapply(seq_len(ncol(y)), function(j)
#'       stats::glm(y[, j] ~ ., data = f, family = stats::binomial()))
#'   },
#'   predict = function(model, x) {
#'     f <- as.data.frame(apply(x, c(1, 3), mean))
#'     vapply(model, function(m) stats::predict(m, f, type = "response"), numeric(nrow(f)))
#'   }
#' )
#' flat_glm
#'
#' @export
learner <- function(name, fit, predict, data = NULL, reads = c("tabular", "sequence"),
                    multi = c("separate", "joint"), control = NULL, needs = character(),
                    params = list()) {
  if (!is.function(fit) || !is.function(predict)) {
    stop("a learner needs a `fit` and a `predict` function.", call. = FALSE)
  }
  reads <- match.arg(reads)
  multi <- match.arg(multi)
  if (!is.null(data) && !inherits(data, "timesift_representation")) {
    stop("`data` is a representation such as grain(\"week\"), or NULL, got ",
         class(data)[1L], ".", call. = FALSE)
  }
  structure(list(name = name, fit = fit, predict = predict, data = data, reads = reads,
                 multi = multi, control = if (is.null(control)) NULL else .as_control(control),
                 needs = needs, params = params),
            class = "timesift_learner")
}

#' @export
print.timesift_learner <- function(x, ...) {
  cat("<timesift learner>", x$name, "\n")
  cat("reads   :", x$reads, "; one model per response:",
      if (identical(x$multi, "joint")) "no, joint" else "yes, separate", "\n")
  cat("data    :", if (is.null(x$data)) "every representation of the run" else .rep_label(x$data),
      "\n")
  if (length(x$params)) {
    cat("settings:", paste(names(x$params), vapply(x$params, .describe, character(1L)),
                           sep = " = ", collapse = ", "), "\n")
  }
  if (!is.null(x$control)) {
    named <- attr(x$control, "given")
    cat("training:", paste(named, vapply(named, function(nm) .describe(x$control[[nm]]),
                                         character(1L)),
                           sep = " = ", collapse = ", "), "\n")
  }
  if (length(x$needs)) {
    cat("needs   :", paste(x$needs, collapse = ", "), "\n")
  }
  invisible(x)
}

# A representation names itself; the fitting layer builds the objects and this reads the label off
# one without knowing anything else about it.
.rep_label <- function(rep) {
  label <- rep$label
  if (is.null(label) || !nzchar(label)) class(rep)[1L] else label
}

.describe <- function(v) {
  if (is.null(v)) "NULL" else paste(format(v), collapse = "/")
}

#' Register a learner
#'
#' Makes a learner available by name to [grain_ladder()] and to [learners()]. The learners that
#' ship are registered the same way, so there is no list of names inside the fitting code.
#'
#' @param name Name the learner is asked for by.
#' @param constructor A function returning a [learner()].
#' @param overwrite Replace an existing registration.
#'
#' @return The constructor, invisibly.
#'
#' @examples
#' learners()
#'
#' @export
register_learner <- function(name, constructor, overwrite = FALSE) {
  if (!is.function(constructor)) {
    stop("a learner registration takes a function returning a learner().", call. = FALSE)
  }
  .learners_reg$set(name, constructor, overwrite)
}

#' @rdname register_learner
#' @export
learners <- function() .learners_reg$names()

#' Fit one learner at one grain
#'
#' @param learner A [learner()], or the name of a registered one.
#' @param x A [grain_matrix()] result.
#' @param y The response for the same units.
#' @param response Name of the registered response head. `"presence_absence"` ships.
#' @param control The run's [train_control()]. The learner's own control overrides it on the
#'   settings that control names, and a setting given in `...` overrides both.
#' @param ... Passed to the learner's `fit`.
#'
#' @return A `timesift_fit`, which [stats::predict()] takes a new representation.
#'
#' @examples
#' set.seed(1)
#' t <- seq(as.POSIXct("2021-09-01", tz = "UTC"), by = "hour", length.out = 24 * 120)
#' units <- sprintf("p%02d", 1:40)
#' d <- data.frame(plot = rep(units, each = length(t)), t = rep(t, length(units)),
#'                 temp = as.numeric(replicate(length(units), rnorm(length(t)))))
#' x <- grain_matrix(d, plot, t, temp, grain = "month")
#' y <- matrix(rbinom(80, 1, 0.4), nrow = 40, dimnames = list(units, c("sp1", "sp2")))
#' fit <- fit_learner(elasticnet(), x, y)
#' dim(stats::predict(fit, x))
#'
#' @export
fit_learner <- function(learner, x, y, response = "presence_absence", control = NULL, ...) {
  learner <- .as_learner(learner)
  .require_packages(learner)
  .check_matrix(x)
  spec <- .responses_reg$get(response)
  y <- spec$prepare(y)
  y <- .align_response(y, dimnames(x)[[1L]])
  # A setting given here overrides the one the learner carries, rather than reaching `fit` twice.
  given <- list(...)
  carried <- learner$params[setdiff(names(learner$params), names(given))]
  args <- c(list(x = x, y = y), carried, given)
  # A learner that trains under a control declares one; the resolved control reaches it through
  # that argument and through nothing else, so a learner with no training settings never sees one.
  # The response head reaches a fit the same way, so what a learner fits toward is the
  # registration and never a second copy of it inside the learner.
  declared <- names(formals(learner$fit))
  if ("control" %in% declared) {
    args$control <- .resolve_control(control, learner$control)
  }
  if ("head" %in% declared) {
    args$head <- spec
  }
  model <- do.call(learner$fit, args)
  structure(list(learner = learner, model = model, response = response,
                 variables = colnames(y), grain = attr(x, "grain"),
                 stats = attr(x, "stats")),
            class = "timesift_fit")
}

# A response's own randomness starts from a seed of its own, so the model of one response is the
# same model whether it was fitted on its own or beside others, and in whatever order. The offset
# is taken from the response's name rather than from its position, because a learner that covers
# the responses one at a time is handed a single column and cannot see where it sat.
.variable_seeds <- function(seed, y) {
  labels <- colnames(y)
  offsets <- if (is.null(labels)) seq_len(ncol(y)) else vapply(labels, .name_offset, integer(1L))
  as.integer(seed) + as.integer(offsets)
}

.name_offset <- function(name) {
  code <- 0L
  for (b in utf8ToInt(name)) {
    code <- (code * 31L + b) %% 104729L
  }
  code
}

#' @param object A `timesift_fit`.
#' @param newdata A representation of the same channels for the units to predict.
#' @rdname fit_learner
#' @export
predict.timesift_fit <- function(object, newdata, ...) {
  p <- object$learner$predict(object$model, newdata)
  p <- as.matrix(p)
  if (nrow(p) != dim(newdata)[1L]) {
    stop("the learner returned ", nrow(p), " rows for ", dim(newdata)[1L], " units.",
         call. = FALSE)
  }
  dimnames(p) <- list(dimnames(newdata)[[1L]], object$variables)
  p
}

#' @export
print.timesift_fit <- function(x, ...) {
  cat("<timesift fit>", x$learner$name, "at the", x$grain, "grain\n")
  cat("channels:", paste(x$stats, collapse = ", "), "\n")
  cat("response:", x$response, "on", .plural(length(x$variables), "variable"), "\n")
  invisible(x)
}

#' Combine learners, or representations, into a set
#'
#' `c()` on learners is the set of them, and on representations the set of those. A set handed to
#' `c()` again splices, so a set can be added to rather than rewritten, which is what `list()`
#' cannot do: `c(base, cnn())` where `base` is already a set.
#'
#' `models` and `sift` take either form. A length-one string is the name of a registered learner.
#'
#' @param ... Learners, representations, or sets of either.
#' @return A `timesift_models` for learners and a `timesift_sift` for representations.
#'
#' @examples
#' base <- c(elasticnet(), forest())
#' base
#' c(base, stepwise())
#'
#' c(grains("day", "week"), lookback("30 days"))
#'
#' @name combine
NULL

#' @rdname combine
#' @export
c.timesift_learner <- function(...) {
  .models(.splice(list(...), "timesift_learner"))
}

#' @rdname combine
#' @export
c.timesift_models <- function(...) {
  .models(.splice(list(...), "timesift_learner"))
}

#' @export
print.timesift_models <- function(x, ...) {
  cat("<timesift models>", .plural(length(x), "learner"), "\n")
  for (nm in names(x)) {
    cat(sprintf("  %-14s reads %s, %s\n", nm, x[[nm]]$reads, x[[nm]]$multi))
  }
  invisible(x)
}

# The set a `c()` of learners is, validated by the same call every entry point resolves `models`
# with, so a set and a bare list cannot disagree about what a valid one is.
.models <- function(x) {
  structure(.learner_list(x), class = "timesift_models")
}

.as_learner <- function(learner) {
  if (inherits(learner, "timesift_learner")) {
    return(learner)
  }
  if (is.character(learner) && length(learner) == 1L) {
    return(.learners_reg$get(learner)())
  }
  stop("expected a learner() or the name of a registered one, got ", class(learner)[1L], ".",
       call. = FALSE)
}

# A learner that needs a package says so and stops. Two code paths, one with the package and one
# without, drift apart and the difference surfaces as a result nobody can place.
.require_packages <- function(learner) {
  missing <- learner$needs[!vapply(learner$needs, requireNamespace, logical(1L), quietly = TRUE)]
  if (length(missing)) {
    stop("the ", learner$name, " learner needs ", paste(missing, collapse = " and "),
         ". Install it with install.packages(\"", missing[1L], "\").", call. = FALSE)
  }
  invisible(TRUE)
}

.align_response <- function(y, units) {
  if (identical(rownames(y), units)) {
    return(y)
  }
  missing <- setdiff(units, rownames(y))
  if (length(missing)) {
    stop(length(missing), " unit", if (length(missing) > 1L) "s are" else " is",
         " in the representation but not the response, first: ", missing[1L], ".", call. = FALSE)
  }
  y[units, , drop = FALSE]
}

# Flatten [unit, bin, channel] to [unit, bin * channel] in the array's own order, so the column
# names say which bin and which channel every predictor came from. A channel that holds the same
# number in every bin carries no bin of its own and is one predictor, read once: repeating it
# would put the same column in front of a penalised fit as often as the grain has bins, and give
# it that many chances of being drawn by a forest or picked by a forward search.
.flatten <- function(x) {
  d <- dim(x)
  channels <- dimnames(x)[[3L]]
  constant <- channels %in% (attr(x, "static") %||% character(0))
  moving <- unclass(x)[, , !constant, drop = FALSE]
  out <- matrix(as.numeric(moving), nrow = d[1L], ncol = d[2L] * sum(!constant))
  dimnames(out) <- list(dimnames(x)[[1L]],
                        paste(rep(channels[!constant], each = d[2L]),
                              rep(dimnames(x)[[2L]], times = sum(!constant)), sep = "@"))
  if (!any(constant)) {
    return(out)
  }
  once <- matrix(as.numeric(unclass(x)[, 1L, constant, drop = FALSE]), nrow = d[1L],
                 dimnames = list(dimnames(x)[[1L]], channels[constant]))
  cbind(out, once)
}

# The family a learner fitting one model per response fits under is read off the response head's
# loss, so a head registered with a squared-error loss reaches the same learners as a
# presence-absence one and each fits the model that loss names.
.head_family <- function(head) {
  families <- c(binary_cross_entropy = "binomial", squared_error = "gaussian")
  if (!is.character(head$loss) || !head$loss %in% names(families)) {
    stop("a learner fitting one model per response has no family for the ",
         .describe(head$loss), " loss. It knows ", paste(names(families), collapse = " and "),
         ".", call. = FALSE)
  }
  families[[head$loss]]
}

.glm_family <- function(family) {
  switch(family, binomial = stats::binomial(), gaussian = stats::gaussian())
}

# Scaling belongs to the fold it is computed on. A per-column scaler is what a linear model wants
# over the bin-by-channel columns; the same scaler over the channels, each read across every unit
# and bin, is what a sequence encoder wants, because it puts every channel on a workable scale
# while preserving the differences between units and between bins that carry the signal. Both are
# computed on the fitting units alone.
.scaler <- function(m, per_column = TRUE) {
  if (per_column) {
    centre <- colMeans(m)
    scale <- apply(m, 2L, stats::sd)
  } else {
    centre <- rep(mean(m), ncol(m))
    scale <- rep(stats::sd(as.numeric(m)), ncol(m))
  }
  scale[!is.finite(scale) | scale < 1e-8] <- 1
  list(centre = centre, scale = scale)
}

.apply_scaler <- function(s, m) {
  sweep(sweep(m, 2L, s$centre, "-"), 2L, s$scale, "/")
}
