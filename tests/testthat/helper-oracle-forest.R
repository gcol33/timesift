# The forest as `inst/spec/representation.md` describes it, in R alone.
#
# Nothing dispatches to this and the package never reaches it at runtime. The core grows the
# forest both languages call, so the two agreeing says nothing about whether the core grows what
# the spec says; this is the forest grown from the spec's text, and `make_fixtures.R` writes the
# forest fixtures from it for the Python suite to read.
#
# R has no unsigned 32-bit integer, so the generator's words are doubles holding 0 to 2^32 - 1,
# and every operation is done where a double is exact: a product of two words in 16-bit halves, an
# exclusive or on the halves, a shift as a product or a quotient by a power of two.

oracle_u32 <- 2^32

oracle_u32_mul <- function(a, b) {
  ah <- a %/% 65536
  al <- a %% 65536
  bh <- b %/% 65536
  bl <- b %% 65536
  (al * bl + ((ah * bl + al * bh) %% 65536) * 65536) %% oracle_u32
}

oracle_u32_xor <- function(a, b) {
  bitwXor(as.integer(a %/% 65536), as.integer(b %/% 65536)) * 65536 +
    bitwXor(as.integer(a %% 65536), as.integer(b %% 65536))
}

oracle_u32_shl <- function(a, k) (a * 2^k) %% oracle_u32

oracle_u32_shr <- function(a, k) a %/% 2^k

oracle_u32_rotl <- function(a, k) oracle_u32_shl(a, k) + oracle_u32_shr(a, 32 - k)

oracle_golden <- 2654435769 # 0x9E3779B9

# The generator of tree `tree` of a forest seeded `seed`: xoshiro128**, its state the outputs
# 4 * tree + 1 to 4 * tree + 4 of a SplitMix32 stream started at the seed.
oracle_stream <- function(seed, tree) {
  counter <- (seed %% oracle_u32 + oracle_u32_mul((4 * tree) %% oracle_u32, oracle_golden)) %%
    oracle_u32
  s <- numeric(4L)
  for (k in 1:4) {
    counter <- (counter + oracle_golden) %% oracle_u32
    z <- counter
    z <- oracle_u32_mul(oracle_u32_xor(z, oracle_u32_shr(z, 16)), 2246822507) # 0x85EBCA6B
    z <- oracle_u32_mul(oracle_u32_xor(z, oracle_u32_shr(z, 13)), 3266489909) # 0xC2B2AE35
    s[k] <- oracle_u32_xor(z, oracle_u32_shr(z, 16))
  }
  env <- new.env()
  env$s <- s
  env$next_word <- function() {
    s <- env$s
    result <- oracle_u32_mul(oracle_u32_rotl(oracle_u32_mul(s[2], 5), 7), 9)
    t <- oracle_u32_shl(s[2], 9)
    s[3] <- oracle_u32_xor(s[3], s[1])
    s[4] <- oracle_u32_xor(s[4], s[2])
    s[2] <- oracle_u32_xor(s[2], s[3])
    s[1] <- oracle_u32_xor(s[1], s[4])
    s[3] <- oracle_u32_xor(s[3], t)
    s[4] <- oracle_u32_rotl(s[4], 11)
    env$s <- s
    result
  }
  env$uniform <- function() env$next_word() * 2^-32
  env$below <- function(k) min(floor(env$uniform() * k), k - 1)
  env
}

# A running sum taken one addition at a time, as the core takes it; `cumsum()` and `sum()`
# accumulate in extended precision.
oracle_running <- function(v) Reduce(`+`, v, accumulate = TRUE)

oracle_serial_sum <- function(v) Reduce(`+`, v, 0)

oracle_urn_draw <- function(stream, rows, cum) {
  target <- stream$uniform() * cum[length(cum)]
  rows[min(sum(cum <= target) + 1L, length(rows))]
}

# The best cut of one column's observations, sorted by it, every observation weighing one.
oracle_best_cut <- function(xs, ys, edge, gini, my_risk) {
  k <- length(xs)
  none <- list(improve = 0, split = NA_real_, direction = NA_integer_)
  if (gini) {
    right <- c(sum(ys == 0), sum(ys == 1))
    left <- c(0, 0)
    rwt <- k
    lwt <- 0
    total <- 0
    for (c in 1:2) {
      t <- right[c] / rwt
      total <- total + rwt * (t * (1 - t))
    }
    best <- total
    where <- NA
    dir <- -1L
    rtot <- k
    ltot <- 0
    i <- 0L
    while (rtot > edge) {
      i <- i + 1L
      j <- if (ys[i] == 0) 1L else 2L
      rwt <- rwt - 1
      lwt <- lwt + 1
      rtot <- rtot - 1
      ltot <- ltot + 1
      right[j] <- right[j] - 1
      left[j] <- left[j] + 1
      if (ltot >= edge && xs[i + 1L] != xs[i]) {
        temp <- 0
        lmean <- 0
        rmean <- 0
        for (c in 1:2) {
          pr <- left[c] / lwt
          temp <- temp + lwt * (pr * (1 - pr))
          lmean <- lmean + pr * (c - 1)
          pr <- right[c] / rwt
          temp <- temp + rwt * (pr * (1 - pr))
          rmean <- rmean + pr * (c - 1)
        }
        if (temp < best) {
          best <- temp
          where <- i
          dir <- if (lmean < rmean) -1L else 1L
        }
      }
    }
    improve <- total - best
    if (!(improve > 0)) return(list(improve = improve, split = NA_real_, direction = NA_integer_))
    return(list(improve = improve, split = (xs[where] + xs[where + 1L]) / 2, direction = dir))
  }
  right_sum <- oracle_serial_sum(ys)
  right_wt <- k
  grandmean <- right_sum / right_wt
  left_sum <- 0
  left_wt <- 0
  right_sum <- 0
  best <- 0
  where <- NA
  dir <- -1L
  right_n <- k
  left_n <- 0
  i <- 0L
  while (right_n > edge) {
    i <- i + 1L
    left_wt <- left_wt + 1
    right_wt <- right_wt - 1
    left_n <- left_n + 1
    right_n <- right_n - 1
    temp <- ys[i] - grandmean
    left_sum <- left_sum + temp
    right_sum <- right_sum - temp
    if (xs[i + 1L] != xs[i] && left_n >= edge) {
      t <- left_sum * left_sum / left_wt + right_sum * right_sum / right_wt
      if (t > best) {
        best <- t
        where <- i
        dir <- if (left_sum < right_sum) -1L else 1L
      }
    }
  }
  improve <- best / my_risk
  if (!(best > 0)) return(list(improve = improve, split = NA_real_, direction = NA_integer_))
  list(improve = improve, split = (xs[where] + xs[where + 1L]) / 2, direction = dir)
}

# One tree, its nodes depth first, left before right, children counted from 0 at its first node.
oracle_grow_tree <- function(x, y, w, gini, mtry, min_leaf, balance, seed, tree) {
  n <- nrow(x)
  p <- ncol(x)
  stream <- oracle_stream(seed, tree)
  count <- integer(n)
  if (balance) {
    urns <- lapply(0:1, function(cls) which(y == cls))
    each <- min(vapply(urns, function(r) sum(w[r] > 0), integer(1L)))
    draws <- c(each, each)
  } else {
    urns <- list(seq_len(n))
    draws <- n
  }
  for (u in seq_along(urns)) {
    rows <- urns[[u]]
    cum <- oracle_running(w[rows])
    for (k in seq_len(draws[u])) {
      r <- oracle_urn_draw(stream, rows, cum)
      count[r] <- count[r] + 1L
    }
  }
  obs <- rep(seq_len(n), count)

  perm <- seq_len(p) - 1L
  out <- new.env()
  out$column <- integer(0)
  out$threshold <- numeric(0)
  out$less_left <- integer(0)
  out$left <- integer(0)
  out$right <- integer(0)
  out$value <- numeric(0)
  iscale <- 0

  node <- function(obs) {
    at <- length(out$column)
    k <- length(obs)
    ys <- y[obs]
    if (gini) {
      value <- sum(ys == 1) / k
      risk <- 0
    } else {
      mean <- oracle_serial_sum(ys) / k
      risk <- oracle_serial_sum((ys - mean) * (ys - mean))
      value <- mean
    }
    out$column <- c(out$column, -1L)
    out$threshold <- c(out$threshold, 0)
    out$less_left <- c(out$less_left, 0L)
    out$left <- c(out$left, -1L)
    out$right <- c(out$right, -1L)
    out$value <- c(out$value, value)
    if (k < 2 * min_leaf || all(ys == ys[1L])) return(invisible(at))

    for (c in seq_len(mtry) - 1L) {
      j <- c + stream$below(p - c)
      held <- perm[c + 1L]
      perm[c + 1L] <<- perm[j + 1L]
      perm[j + 1L] <<- held
    }
    chosen <- sort(perm[seq_len(mtry)])
    found <- FALSE
    best <- 0
    for (v in chosen) {
      o <- obs[order(x[obs, v + 1L], method = "radix")]
      xs <- x[o, v + 1L]
      if (xs[1L] == xs[k]) next
      cut <- oracle_best_cut(xs, y[o], min_leaf, gini, risk)
      if (cut$improve > iscale) iscale <<- cut$improve
      if (cut$improve > iscale * 1e-10 && (!found || cut$improve > best)) {
        found <- TRUE
        best <- cut$improve
        var <- v
        spoint <- cut$split
        dir <- cut$direction
      }
    }
    if (!found) return(invisible(at))
    side <- ifelse(x[obs, var + 1L] < spoint, dir, -dir)
    out$column[at + 1L] <- var
    out$threshold[at + 1L] <- spoint
    out$less_left[at + 1L] <- as.integer(dir == -1L)
    out$left[at + 1L] <- length(out$column)
    node(obs[side == -1L])
    out$right[at + 1L] <- length(out$column)
    node(obs[side == 1L])
    invisible(at)
  }
  node(obs)
  list(column = out$column, threshold = out$threshold, less_left = out$less_left,
       left = out$left, right = out$right, value = out$value)
}

oracle_forest <- function(x, y, w, family, trees, mtry, min_leaf, balance, seed) {
  lapply(seq_len(trees) - 1L, function(t) {
    oracle_grow_tree(x, y, w, family == "binomial", mtry, min_leaf, balance, seed, t)
  })
}

# The core's forest cut back into one list per tree, children counted from the tree's first node.
oracle_split_forest <- function(forest) {
  lapply(seq_len(length(forest$offset) - 1L), function(t) {
    at <- (forest$offset[t] + 1L):forest$offset[t + 1L]
    list(column = forest$column[at], threshold = forest$threshold[at],
         less_left = forest$less_left[at], left = forest$left[at], right = forest$right[at],
         value = forest$value[at])
  })
}
