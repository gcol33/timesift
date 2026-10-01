#' Draw pseudo-absences from a pool of background units
#'
#' Presence-only records give a response of ones. A model needs units where the response is zero,
#' and these are drawn from `pool`, a table of background units that carry the same columns as the
#' targets (an identifier, the `static` predictors, the `coords`, and a record in `series` for
#' each). The strategies are `bm_PseudoAbsences()`'s:
#'
#' * `"random"`: any unit of the pool that is not a presence.
#' * `"sre"`: a unit outside the envelope of the presences, the band between the `quantile` and
#'   `1 - quantile` quantiles of each of the `env` columns over the presences, as [envelope()]
#'   draws it. A unit is outside where it leaves the band in at least one column.
#' * `"disk"`: a unit whose distance to the nearest presence lies between `dist_min` and
#'   `dist_max`, both included, on `coords`. The distance is planar in the units of the
#'   coordinates, or in metres from longitude and latitude in degrees where `lonlat` is `TRUE`,
#'   on a sphere of radius 6371008.8 m.
#'
#' A presence is never its own absence: a unit of the pool whose `id` is one of the presences' is
#' not a candidate. The units a strategy admits are the same in both languages; which `n` of them
#' are drawn depends on the language's generator, as the folds of [fold_map()] do. To fix the
#' units, draw once and keep the table. A set of one's own is the rows of `pool` it names, and
#' needs no strategy.
#'
#' The drawn units are not flagged in a fit: bind them to the presences with a response of zero
#' and fit as one table. Their `pseudo` column says which they are, and `set` which draw.
#'
#' @param pool A data frame of background units.
#' @param presences A data frame of the presences, with the columns the strategy reads.
#' @param n Number of pseudo-absences in each draw.
#' @param strategy `"random"`, `"sre"` or `"disk"`.
#' @param id The column naming a unit in both tables, or `NULL` where no unit of the pool is a
#'   presence.
#' @param env For `"sre"`, the numeric columns the envelope is drawn on.
#' @param quantile For `"sre"`, the share of the presences left outside at each end of a column,
#'   in `[0, 0.5]`.
#' @param coords For `"disk"`, the two columns holding the coordinates.
#' @param dist_min,dist_max For `"disk"`, the least and greatest distance from the nearest
#'   presence.
#' @param lonlat For `"disk"`, whether `coords` are longitude and latitude in degrees.
#' @param repeats Number of independent draws, each with its own seed.
#' @param seed Random seed of the first draw; draw `r` uses `seed + r - 1`.
#'
#' @return The drawn rows of `pool`, with the columns `set` (the draw) and `pseudo` (`TRUE`),
#'   of class `timesift_pseudo_absences`.
#'
#' @examples
#' pool <- data.frame(cell = 1:200, x = rep(1:20, 10), y = rep(1:10, each = 20),
#'                    temp = rnorm(200))
#' presences <- pool[c(5, 25, 45), ]
#' pseudo_absences(pool, presences, n = 10, strategy = "disk", id = "cell",
#'                 coords = c("x", "y"), dist_min = 3, dist_max = 8)
#'
#' @export
pseudo_absences <- function(pool, presences, n, strategy = c("random", "sre", "disk"), id = NULL,
                            env = NULL, quantile = 0.025, coords = NULL, dist_min = 0,
                            dist_max = Inf, lonlat = FALSE, repeats = 1L, seed = 1L) {
  strategy <- match.arg(strategy)
  .check_count(n, "n", 1L)
  .check_count(repeats, "repeats", 1L)
  open <- .pa_candidates(pool, presences, strategy, id, env, quantile, coords, dist_min, dist_max,
                         lonlat)
  candidates <- which(open)
  if (length(candidates) < n) {
    stop("the ", strategy, " strategy admits ", length(candidates), " unit",
         if (length(candidates) != 1L) "s", " of the pool, and ", n, " were asked for.",
         call. = FALSE)
  }
  old <- .seed_state()
  on.exit(.restore_seed(old), add = TRUE)
  draws <- lapply(seq_len(repeats), function(r) {
    set.seed(seed + r - 1L)
    picked <- candidates[sample.int(length(candidates), n)]
    out <- pool[picked, , drop = FALSE]
    out$set <- r
    out$pseudo <- TRUE
    out
  })
  out <- do.call(rbind, draws)
  rownames(out) <- NULL
  structure(out, class = c("timesift_pseudo_absences", "data.frame"), strategy = strategy,
            candidates = length(candidates))
}

# Which units of the pool a strategy admits, as a logical vector over its rows.
.pa_candidates <- function(pool, presences, strategy, id, env, quantile, coords, dist_min,
                           dist_max, lonlat) {
  pool <- as.data.frame(pool)
  presences <- as.data.frame(presences)
  open <- rep(TRUE, nrow(pool))
  if (!is.null(id)) {
    if (!id %in% names(pool) || !id %in% names(presences)) {
      stop("`id` has to name a column of both tables.", call. = FALSE)
    }
    open <- !(pool[[id]] %in% presences[[id]])
  }
  if (!nrow(presences)) {
    stop("a strategy that reads the presences needs at least one.", call. = FALSE)
  }
  switch(strategy,
    random = open,
    sre = {
      if (!length(env) || !all(env %in% names(pool)) || !all(env %in% names(presences))) {
        stop("`env` names the numeric columns of both tables the envelope is drawn on.",
             call. = FALSE)
      }
      if (!is.numeric(quantile) || length(quantile) != 1L || !is.finite(quantile) ||
          quantile < 0 || quantile > 0.5) {
        stop("`quantile` is one number in [0, 0.5], got ", .describe(quantile), ".",
             call. = FALSE)
      }
      x <- as.matrix(presences[, env, drop = FALSE])
      band <- .envelope_fit(x, rep(1, nrow(x)), quantile)
      inside <- .envelope_predict(band, as.matrix(pool[, env, drop = FALSE]))
      open & inside == 0
    },
    disk = {
      if (length(coords) != 2L || !all(coords %in% names(pool)) ||
          !all(coords %in% names(presences))) {
        stop("`coords` names the two coordinate columns of both tables.", call. = FALSE)
      }
      if (!is.numeric(dist_min) || !is.numeric(dist_max) || dist_min < 0 || dist_max < dist_min) {
        stop("`dist_min` and `dist_max` are distances with 0 <= dist_min <= dist_max.",
             call. = FALSE)
      }
      near <- .nearest_distance(as.matrix(pool[, coords, drop = FALSE]),
                                as.matrix(presences[, coords, drop = FALSE]), lonlat)
      open & near >= dist_min & near <= dist_max
    })
}

# The distance from each row of `from` to the nearest row of `to`, planar or on the sphere.
.nearest_distance <- function(from, to, lonlat) {
  vapply(seq_len(nrow(from)), function(i) {
    min(if (lonlat) {
      .haversine(from[i, 1L], from[i, 2L], to[, 1L], to[, 2L])
    } else {
      sqrt((to[, 1L] - from[i, 1L])^2 + (to[, 2L] - from[i, 2L])^2)
    })
  }, numeric(1L))
}

.haversine <- function(lon1, lat1, lon2, lat2) {
  rad <- pi / 180
  a <- sin((lat2 - lat1) * rad / 2)^2 +
    cos(lat1 * rad) * cos(lat2 * rad) * sin((lon2 - lon1) * rad / 2)^2
  2 * 6371008.8 * asin(sqrt(a))
}

#' @export
print.timesift_pseudo_absences <- function(x, ...) {
  cat("<timesift pseudo-absences>", attr(x, "strategy"), "strategy,",
      .plural(attr(x, "candidates"), "candidate"), "in the pool,",
      .plural(length(unique(x$set)), "draw"), "of", sum(x$set == x$set[1L]), "\n")
  print(utils::head(as.data.frame(x), 6L))
  invisible(x)
}
