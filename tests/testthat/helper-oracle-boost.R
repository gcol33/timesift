# gbm's boosting as `inst/spec/representation.md` describes it, in R alone, with the subsample and
# the column draw the spec defines. gbm itself pins the fit where nothing is drawn; this pins what
# the draws do. It uses the generator `helper-oracle-forest.R` defines.

oracle_boost <- function(x, y, w, family, trees, depth, shrinkage, min_leaf, subsample,
                         colsample, seed, model = 0L) {
  m <- nrow(x)
  p <- ncol(x)
  binomial <- family == "binomial"
  s <- oracle_serial_sum(w * y)
  t <- oracle_serial_sum(w)
  init <- if (binomial) log(s / (t - s)) else s / t
  ord <- lapply(seq_len(p), function(v) order(x[, v], method = "radix"))
  f <- rep(init, m)
  bag <- floor(subsample * m)
  n_col <- max(1, floor(colsample * p))
  out <- vector("list", trees)
  for (tree in seq_len(trees) - 1L) {
    stream <- oracle_stream(seed, model * trees + tree)
    inbag <- logical(m)
    bagged <- 0
    for (k in seq_len(m)) {
      inbag[k] <- stream$uniform() * (m - k + 1) < bag - bagged
      if (inbag[k]) bagged <- bagged + 1
    }
    perm <- seq_len(p) - 1L
    for (c in seq_len(n_col) - 1L) {
      j <- c + stream$below(p - c)
      held <- perm[c + 1L]
      perm[c + 1L] <- perm[j + 1L]
      perm[j + 1L] <- held
    }
    cols <- sort(perm[seq_len(n_col)])
    z <- if (binomial) y - 1 / (1 + exp(-f)) else y - f

    # The tree, best first: every terminal node a search; each split adds the left child in the
    # node's place, then the right child and gbm's empty missing-value node at the end.
    nodes <- list(list(col = -1L, thr = 0, left = 0L, right = 0L,
                       mean = oracle_serial_sum(w[inbag] * z[inbag]) /
                         oracle_serial_sum(w[inbag])))
    slots <- list(list(node = 1L, open = TRUE, sum = oracle_serial_sum(w[inbag] * z[inbag]),
                       weight = oracle_serial_sum(w[inbag]), count = sum(inbag), best = 0))
    assign <- rep(1L, m)
    for (step in seq_len(depth)) {
      for (v in cols) {
        for (i in seq_along(slots)) {
          if (!slots[[i]]$open) next
          slots[[i]]$ls <- 0; slots[[i]]$lw <- 0; slots[[i]]$ln <- 0
          slots[[i]]$rs <- slots[[i]]$sum; slots[[i]]$rw <- slots[[i]]$weight
          slots[[i]]$rn <- slots[[i]]$count; slots[[i]]$last <- -Inf
        }
        for (k in ord[[v + 1L]]) {
          if (!inbag[k]) next
          i <- assign[k]
          sl <- slots[[i]]
          if (!sl$open) next
          xk <- x[k, v + 1L]
          if (sl$last != xk && sl$ln >= min_leaf && sl$rn >= min_leaf) {
            d <- sl$ls / sl$lw - sl$rs / sl$rw
            imp <- sl$lw * sl$rw * d * d / (sl$lw + sl$rw)
            if (isTRUE(imp > sl$best)) {
              sl$best <- imp
              sl$var <- v
              sl$split <- 0.5 * (sl$last + xk)
              sl$bls <- sl$ls; sl$blw <- sl$lw; sl$bln <- sl$ln
              sl$brs <- sl$rs; sl$brw <- sl$rw; sl$brn <- sl$rn
            }
          }
          wz <- w[k] * z[k]
          sl$ls <- sl$ls + wz; sl$lw <- sl$lw + w[k]; sl$ln <- sl$ln + 1
          sl$rs <- sl$rs - wz; sl$rw <- sl$rw - w[k]; sl$rn <- sl$rn - 1
          sl$last <- xk
          slots[[i]] <- sl
        }
      }
      for (i in seq_along(slots)) slots[[i]]$open <- FALSE
      bests <- vapply(slots, function(s) s$best, 0)
      if (!any(bests > 0)) break
      at <- which(bests == max(bests))[1L]
      ch <- slots[[at]]
      left <- length(nodes) + 1L
      right <- left + 1L
      nodes[[left]] <- list(col = -1L, thr = 0, mean = ch$bls / ch$blw)
      nodes[[right]] <- list(col = -1L, thr = 0, mean = ch$brs / ch$brw)
      nodes[[ch$node]]$col <- ch$var
      nodes[[ch$node]]$thr <- ch$split
      nodes[[ch$node]]$left <- left
      nodes[[ch$node]]$right <- right
      r <- length(slots) + 1L
      slots[[r]] <- list(node = right, open = TRUE, sum = ch$brs, weight = ch$brw,
                         count = ch$brn, best = 0)
      slots[[r + 1L]] <- list(node = -1L, open = FALSE, best = 0)
      slots[[at]] <- list(node = left, open = TRUE, sum = ch$bls, weight = ch$blw,
                          count = ch$bln, best = 0)
      go <- assign == at & !(x[, ch$var + 1L] < ch$split)
      assign[go] <- r
    }
    leaf_of <- function(k) {
      at <- 1L
      while (nodes[[at]]$col >= 0L) {
        at <- if (x[k, nodes[[at]]$col + 1L] < nodes[[at]]$thr) nodes[[at]]$left else
          nodes[[at]]$right
      }
      at
    }
    leaf <- vapply(seq_len(m), leaf_of, 1L)
    value <- numeric(length(nodes))
    for (a in unique(leaf)) {
      rows <- which(leaf == a & inbag)
      step_value <- if (binomial) {
        num <- oracle_serial_sum(w[rows] * z[rows])
        den <- oracle_serial_sum(w[rows] * (y[rows] - z[rows]) * (1 - y[rows] + z[rows]))
        if (den == 0) 0 else num / den
      } else {
        nodes[[a]]$mean
      }
      value[a] <- shrinkage * step_value
    }
    f <- f + value[leaf]
    out[[tree + 1L]] <- list(nodes = nodes, value = value)
  }
  list(init = init, trees = out, f = f)
}

oracle_boost_predict <- function(fit, x, binomial) {
  f <- rep(fit$init, nrow(x))
  for (tr in fit$trees) {
    for (k in seq_len(nrow(x))) {
      at <- 1L
      while (tr$nodes[[at]]$col >= 0L) {
        at <- if (x[k, tr$nodes[[at]]$col + 1L] < tr$nodes[[at]]$thr) tr$nodes[[at]]$left else
          tr$nodes[[at]]$right
      }
      f[k] <- f[k] + tr$value[at]
    }
  }
  if (binomial) 1 / (1 + exp(-f)) else f
}
