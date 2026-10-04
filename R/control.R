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
#' @param learning_rate Learning rate, the one the schedule starts from.
#' @param weight_decay The optimiser's weight decay: decoupled from the gradient under `"adamw"`,
#'   and added to it as `weight_decay` times each parameter under `"adam"` and `"sgd"`, as torch's
#'   optimisers take it.
#' @param optimizer `"adamw"`, `"adam"` or `"sgd"`, torch's optimisers at their own defaults besides
#'   the learning rate and the weight decay; `"sgd"` takes no momentum.
#' @param penalty The weight of a penalty added to the loss of every batch:
#'   `penalty * (alpha * sum(abs(W)) + (1 - alpha) * sqrt(sum(W^2)))` summed over every weight matrix
#'   and kernel of the network, its biases and normalisation scales left out. This is cito's
#'   `lambda`, the penalty biomod2's `DNN` fits under. 0 adds nothing.
#' @param alpha The share of the penalty on the absolute weights, as `elasticnet()` reads its
#'   `alpha`: 1 penalises the absolute weights alone and 0 the norm alone. cito's `alpha` is
#'   `1 - alpha`.
#' @param schedule How the learning rate moves over the epochs: `"cosine"` anneals it to zero over
#'   the budget, `"constant"` holds it, and `"plateau"` multiplies it by `plateau_factor` once the
#'   loss has not improved for `plateau_patience` epochs, reading the validation loss where
#'   `val_frac` holds a set back and the epoch's mean training loss where it does not, as cito's
#'   `reduce_on_plateau` reads them. torch's own relative threshold of `1e-4` decides an improvement.
#' @param plateau_factor,plateau_patience The factor a plateau multiplies the learning rate by, and
#'   the epochs without improvement that make one.
#' @param early_stopping Epochs without an inner-validation improvement before training stops.
#'   Read only where `val_frac` holds a validation set back. `Inf`, the default, never stops: the
#'   whole budget is trained and the epoch with the lowest validation loss is restored.
#' @param val_frac Share of the fitting targets held back as an inner validation set, used for
#'   early stopping and for nothing else. It is never scored as a result. The set is drawn from
#'   every fit alike by a plain random permutation, so the fit on all targets that a run ends with
#'   also trains on the rest. At the default of 0 nothing is held back: every fitting target is trained on, the whole budget runs,
#'   and the fit keeps the last epoch, where the cosine schedule has annealed the learning rate to
#'   zero. On the Schrankogel weekly arm that epoch scores 0.007 AUC above the epoch early stopping
#'   keeps on a 15 percent split, and within 0.001 of the best epoch read on the test folds.
#' @param device `"auto"` to take a graphics processor where there is one, NVIDIA's or Apple's, or
#'   a device name such as `"cuda"`, `"mps"` or `"cpu"`. A fitted encoder carries the setting
#'   rather than the device it resolved to, so a fit made on one machine predicts on another.
#' @param seed Seed for initialisation, batching and the inner validation split.
#' @param swa Average the weights of the tail epochs instead of keeping a single epoch.
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
#' # biomod2's tuned DNN, cito's adam with a penalty and a plateau schedule
#' train_control(optimizer = "adam", learning_rate = 0.05, weight_decay = 0, penalty = 0.001,
#'               alpha = 0, schedule = "plateau", plateau_patience = 7L, epochs = 150L,
#'               batch_size = 100L, val_frac = 0.2, early_stopping = 14L)
#'
#' @export
train_control <- function(epochs = 60L, batch_size = 64L, learning_rate = 1e-3,
                          weight_decay = 1e-4, optimizer = c("adamw", "adam", "sgd"),
                          penalty = 0, alpha = 0.5,
                          schedule = c("cosine", "constant", "plateau"), plateau_factor = 0.1,
                          plateau_patience = 10L, early_stopping = Inf, val_frac = 0,
                          device = "auto", seed = 1L, swa = FALSE, swa_start = 0.7) {
  given <- names(as.list(match.call()))[-1L]
  # `Inf` is a patience that never runs out, which an integer cannot hold, so it is kept as it is.
  if (!(is.numeric(early_stopping) && length(early_stopping) == 1L &&
          is.infinite(early_stopping))) {
    early_stopping <- as.integer(early_stopping)
  }
  optimizer <- match.arg(optimizer)
  schedule <- match.arg(schedule)
  settings <- list(
    epochs = as.integer(epochs), batch_size = as.integer(batch_size),
    learning_rate = learning_rate, weight_decay = weight_decay, optimizer = optimizer,
    penalty = penalty, alpha = alpha, schedule = schedule, plateau_factor = plateau_factor,
    plateau_patience = as.integer(plateau_patience), early_stopping = early_stopping, val_frac = val_frac,
    device = device, seed = as.integer(seed), swa = isTRUE(swa), swa_start = swa_start)
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
  for (nm in c("weight_decay", "penalty")) {
    if (length(settings[[nm]]) != 1L || is.na(settings[[nm]]) || settings[[nm]] < 0) {
      stop("`", nm, "` is a single number that is not negative, got ",
           paste(format(settings[[nm]]), collapse = ", "), ".", call. = FALSE)
    }
  }
  if (length(settings$alpha) != 1L || is.na(settings$alpha) || settings$alpha < 0 ||
        settings$alpha > 1) {
    stop("`alpha` is a single number in [0, 1], got ",
         paste(format(settings$alpha), collapse = ", "), ".", call. = FALSE)
  }
  if (length(settings$plateau_factor) != 1L || is.na(settings$plateau_factor) ||
        settings$plateau_factor <= 0 || settings$plateau_factor >= 1) {
    stop("`plateau_factor` is a single number in (0, 1), got ",
         paste(format(settings$plateau_factor), collapse = ", "), ".", call. = FALSE)
  }
  if (length(settings$plateau_patience) != 1L || is.na(settings$plateau_patience) ||
        settings$plateau_patience < 0L) {
    stop("`plateau_patience` is a count of epochs of zero or more, got ",
         paste(format(settings$plateau_patience), collapse = ", "), ".", call. = FALSE)
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
    cat(sprintf("  %-16s %s%s\n", nm, .describe(x[[nm]]),
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
