pa_tables <- function() {
  pool <- data.frame(cell = 1:100, x = rep(1:10, each = 10), y = rep(1:10, times = 10),
                     e1 = (7 * (1:100)) %% 17, e2 = (3 * (1:100)) %% 13)
  list(pool = pool, presences = pool[c(12, 45, 46, 78), ])
}

test_that("the random strategy admits every unit that is not a presence", {
  d <- pa_tables()
  out <- pseudo_absences(d$pool, d$presences, n = 20L, id = "cell")
  expect_s3_class(out, "timesift_pseudo_absences")
  expect_equal(nrow(out), 20L)
  expect_false(anyDuplicated(out$cell) > 0L)
  expect_false(any(out$cell %in% d$presences$cell))
  expect_true(all(out$pseudo))
  expect_equal(unique(out$set), 1L)
  expect_equal(attr(out, "candidates"), 96L)
  # Without an id nothing is excluded.
  expect_equal(attr(pseudo_absences(d$pool, d$presences, n = 5L), "candidates"), 100L)
})

test_that("a draw is reproducible, and each repeat is its own", {
  d <- pa_tables()
  a <- pseudo_absences(d$pool, d$presences, n = 10L, id = "cell", seed = 4L)
  b <- pseudo_absences(d$pool, d$presences, n = 10L, id = "cell", seed = 4L)
  expect_identical(a$cell, b$cell)
  many <- pseudo_absences(d$pool, d$presences, n = 10L, id = "cell", repeats = 3L, seed = 4L)
  expect_equal(nrow(many), 30L)
  expect_equal(sort(unique(many$set)), 1:3)
  expect_identical(many$cell[many$set == 1L], a$cell)
  expect_false(identical(many$cell[many$set == 1L], many$cell[many$set == 2L]))
  # The session's own stream is left where it was.
  set.seed(9L)
  before <- stats::runif(1L)
  set.seed(9L)
  invisible(pseudo_absences(d$pool, d$presences, n = 10L, id = "cell", repeats = 2L))
  expect_equal(stats::runif(1L), before)
})

test_that("the disk strategy keeps the distance to the nearest presence in its band", {
  d <- pa_tables()
  out <- pseudo_absences(d$pool, d$presences, n = 15L, strategy = "disk", id = "cell",
                         coords = c("x", "y"), dist_min = 2, dist_max = 3)
  near <- vapply(seq_len(nrow(out)), function(i) {
    min(sqrt((d$presences$x - out$x[i])^2 + (d$presences$y - out$y[i])^2))
  }, numeric(1L))
  expect_true(all(near >= 2 & near <= 3))
  # A band nothing falls in cannot be drawn from.
  expect_error(pseudo_absences(d$pool, d$presences, n = 1L, strategy = "disk",
                               coords = c("x", "y"), dist_min = 50, dist_max = 60), "admits 0")
  expect_error(pseudo_absences(d$pool, d$presences, n = 1L, strategy = "disk",
                               coords = c("x", "y"), dist_min = 3, dist_max = 2), "0 <= dist_min")
  expect_error(pseudo_absences(d$pool, d$presences, n = 1L, strategy = "disk"), "coordinate")
})

test_that("the sre strategy keeps the units outside the envelope of the presences", {
  d <- pa_tables()
  out <- pseudo_absences(d$pool, d$presences, n = 10L, strategy = "sre", id = "cell",
                         env = c("e1", "e2"), quantile = 0.1)
  lo <- vapply(c("e1", "e2"), function(v) stats::quantile(d$presences[[v]], 0.1), numeric(1L))
  hi <- vapply(c("e1", "e2"), function(v) stats::quantile(d$presences[[v]], 0.9), numeric(1L))
  inside <- out$e1 >= lo[["e1"]] & out$e1 <= hi[["e1"]] & out$e2 >= lo[["e2"]] &
    out$e2 <= hi[["e2"]]
  expect_false(any(inside))
  expect_error(pseudo_absences(d$pool, d$presences, n = 1L, strategy = "sre"), "env")
  expect_error(pseudo_absences(d$pool, d$presences, n = 1L, strategy = "sre", env = "e1",
                               quantile = 0.9), "in \\[0, 0.5\\]")
})

test_that("a draw says what it cannot do", {
  d <- pa_tables()
  expect_error(pseudo_absences(d$pool, d$presences, n = 99L, id = "cell"), "admits 96")
  expect_error(pseudo_absences(d$pool, d$presences, n = 0L), "1 or more")
  expect_error(pseudo_absences(d$pool, d$presences[0L, ], n = 2L), "at least one")
  expect_error(pseudo_absences(d$pool, d$presences, n = 2L, id = "nope"), "both tables")
  expect_output(print(pseudo_absences(d$pool, d$presences, n = 3L)), "pseudo-absences")
})
