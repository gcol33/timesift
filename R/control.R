#' Training settings every neural learner reads
#'
#' The one place a training setting is defaulted. An architecture constructor carries its
#' architecture and nothing else, the control carries how that architecture is trained, and both a
#' whole run and a single learner take one, so there is never a second table of defaults to keep in
#' step with this one.
#'
#' A control records which of its settings were named in the call. Merging two controls therefore
#' moves only the settings that were asked for: a learner given `train_control(epochs = 200)` reads
#' 200 epochs and takes every other setting from the control the run was given.
#'
#' @param epochs Epoch budget the cosine schedule anneals over.
#' @param batch_size Most targets per optimiser step. The fitting targets are cut into as few
#'   batches of at most this many as they divide into, of as equal a length as they can be, so no
#'   batch is a remainder of one.
#' @param learning_rate Learning rate.
#' @param weight_decay AdamW weight decay.
#' @param early_stopping Epochs without an inner-validation improvement before training stops.
#' @param val_frac Share of the fitting targets held back as an inner validation set, used for
#'   early stopping and for nothing else. It is never scored as a result. The set is drawn from
#'   every fit alike, one target from each of as many equal-count strata of the response total as
#'   the set holds, so the fit on all targets that a run ends with also trains on the rest.
#' @param device `"auto"` to take a graphics processor where there is one, NVIDIA's or Apple's, or
#'   a device name such as `"cuda"`, `"mps"` or `"cpu"`. A fitted encoder carries the setting
#'   rather than the device it resolved to, so a fit made on one machine predicts on another.
#' @param seed Seed for initialisation, batching and the inner validation split.
#' @param pos_weight_cap Ceiling on the per-response positive-class weight, which is the ratio of
#'   absences to presences among the fitting targets. At least one: a weight under one would
#'   weight presences down.
#' @param swa Average the weights of the tail epochs instead of restoring the best single epoch.
#'   The schedule anneals to `swa_start` of the epoch budget and is then held flat while the
#'   remaining epochs' weights are averaged, and the batch-normalisation statistics are recomputed
#'   for the average. Early stopping is off while an average is being accumulated, so the averaging
#'   grain always runs.
#' @param swa_start Share of the epoch budget after which averaging begins.
#'
#' @return A `timesift_control`.
#'
#' @examples
#' train_control()
#' train_control(epochs = 200L, device = "cpu")
#'
#' @export
train_control <- function(epochs = 60L, batch_size = 64L, learning_rate = 1e-3,
                          weight_decay = 1e-4, early_stopping = 10L, val_frac = 0.15,
                          device = "auto", seed = 1L,
                          pos_weight_cap = 50, swa = FALSE, swa_start = 0.7) {
  given <- names(as.list(match.call()))[-1L]
  # `early_stopping = Inf` is a patience that never runs out, so the whole budget is trained; an
  # integer cannot hold it, and the largest one is the same thing.
  if (is.numeric(early_stopping) && length(early_stopping) == 1L && is.infinite(early_stopping)) {
    early_stopping <- if (early_stopping > 0) .Machine$integer.max else -1L
  }
  settings <- list(
    epochs = as.integer(epochs), batch_size = as.integer(batch_size),
    learning_rate = learning_rate, weight_decay = weight_decay,
    early_stopping = as.integer(early_stopping), val_frac = val_frac,
    device = device, seed = as.integer(seed), pos_weight_cap = pos_weight_cap,
    swa = isTRUE(swa), swa_start = swa_start)
  .check_control(settings)
  structure(settings, given = given, class = "timesift_control")
}

.check_control <- function(settings) {
  positive <- c("epochs", "batch_size", "learning_rate")
  for (nm in positive) {
    if (length(settings[[nm]]) != 1L || is.na(settings[[nm]]) || settings[[nm]] <= 0) {
      stop("`", nm, "` is a single positive number, got ",
           paste(format(settings[[nm]]), collapse = ", "), ".", call. = FALSE)
    }
  }
  if (length(settings$pos_weight_cap) != 1L || is.na(settings$pos_weight_cap) ||
      settings$pos_weight_cap < 1) {
    stop("`pos_weight_cap` is a single number of at least 1, got ",
         paste(format(settings$pos_weight_cap), collapse = ", "), ".", call. = FALSE)
  }
  for (nm in c("val_frac", "swa_start")) {
    if (length(settings[[nm]]) != 1L || is.na(settings[[nm]]) ||
        settings[[nm]] < 0 || settings[[nm]] >= 1) {
      stop("`", nm, "` is a share of the run, at least 0 and under 1, got ",
           paste(format(settings[[nm]]), collapse = ", "), ".", call. = FALSE)
    }
  }
  if (length(settings$early_stopping) != 1L || is.na(settings$early_stopping) ||
        settings$early_stopping < 1L) {
    stop("`early_stopping` is a count of epochs of at least one, or Inf for never, got ",
         paste(format(settings$early_stopping), collapse = ", "), ".", call. = FALSE)
  }
  if (length(settings$weight_decay) != 1L || is.na(settings$weight_decay) ||
        settings$weight_decay < 0) {
    stop("`weight_decay` is a single number that is not negative, got ",
         paste(format(settings$weight_decay), collapse = ", "), ".", call. = FALSE)
  }
  if (!is.character(settings$device) || length(settings$device) != 1L) {
    stop("`device` is \"auto\" or the name of a device, got ", class(settings$device)[1L], ".",
         call. = FALSE)
  }
  invisible(TRUE)
}

#' @export
print.timesift_control <- function(x, ...) {
  cat("<timesift control>\n")
  named <- attr(x, "given")
  for (nm in names(x)) {
    cat(sprintf("  %-15s %s%s\n", nm, .describe(x[[nm]]),
                if (nm %in% named) "" else "   (default)"))
  }
  invisible(x)
}

#' @export
`$.timesift_control` <- function(x, name) {
  if (!name %in% names(unclass(x))) {
    stop("a training control has no setting called ", name, ". It carries ",
         paste(names(unclass(x)), collapse = ", "), ".", call. = FALSE)
  }
  unclass(x)[[name]]
}

.control_names <- function() names(formals(train_control))

# Controls stack: the shipped defaults first, then each control given, each moving only the
# settings its own call named. That is what lets a learner override one setting of a run's control
# without restating the rest of it.
.resolve_control <- function(...) {
  out <- unclass(train_control())
  for (given in Filter(Negate(is.null), list(...))) {
    given <- .as_control(given)
    for (nm in attr(given, "given")) {
      out[[nm]] <- unclass(given)[[nm]]
    }
  }
  .check_control(out)
  structure(out, given = names(out), class = "timesift_control")
}

.as_control <- function(x) {
  if (inherits(x, "timesift_control")) {
    return(x)
  }
  if (is.list(x) && length(x) && !is.null(names(x))) {
    return(do.call(train_control, x))
  }
  stop("expected a train_control(), got ", class(x)[1L], ".", call. = FALSE)
}

# The training settings a learner constructor was handed, as a partial control, or NULL where it
# was handed none. An unknown name is refused here rather than reaching the trainer as a setting
# nothing reads.
.given_control <- function(given, what) {
  if (!length(given)) {
    return(NULL)
  }
  labels <- names(given) %||% rep("", length(given))
  labels[!nzchar(labels)] <- "<unnamed>"
  unknown <- unique(setdiff(labels, .control_names()))
  if (length(unknown)) {
    stop(what, " has no setting called ",
         paste(sort(unknown, method = "radix"), collapse = ", "),
         ". Its architecture is set by its own arguments and its training by ",
         paste(.control_names(), collapse = ", "), ".", call. = FALSE)
  }
  do.call(train_control, given)
}

.torch_device <- function(device) {
  if (!identical(device, "auto")) {
    return(device)
  }
  torch <- .torch()
  if (torch$cuda_is_available()) {
    "cuda"
  } else if (torch$backends_mps_is_available()) {
    "mps"
  } else {
    "cpu"
  }
}
