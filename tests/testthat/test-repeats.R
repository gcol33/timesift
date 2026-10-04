repeat_learner <- function(name = "rl") {
  learner(
    name, multi = "joint", reads = "tabular",
    fit = function(x, y, ...) {
      mu <- apply(x, 1L, mean)
      s <- stats::sd(mu)
      if (!is.finite(s) || s == 0) s <- 1
      z <- (mu - mean(mu)) / s
      beta <- vapply(seq_len(ncol(y)), function(j) {
        if (length(unique(y[, j])) < 2L) 0 else unname(stats::cov(z, y[, j]))
      }, numeric(1L))
      list(centre = mean(mu), scale = s, beta = beta, prevalence = colMeans(y))
    },
    predict = function(model, x) {
      z <- (apply(x, 1L, mean) - model$centre) / model$scale
      stats::plogis(outer(z, 3 * model$beta) +
                      rep(stats::qlogis(pmin(pmax(model$prevalence, 0.05), 0.95)),
                          each = length(z)))
    })
}

repeat_case <- function(n_unit = 30L, days = 60L) {
  sim <- sim_series(n_unit = n_unit, days = days, seed = 17L)
  y <- sim_response(sim, n_var = 2L)
  targets <- cbind(data.frame(plot = sim$units, group = rep(seq_len(n_unit / 3L), each = 3L),
                              stringsAsFactors = FALSE), as.data.frame(y))
  list(targets = targets, series = sim$readings)
}

run_repeat <- function(case, resampling, ensemble = FALSE, n_inner = NULL, ...) {
  timesift(case$targets, case$series, y = starts_with("sp"), id = plot, time = t,
           learners = list(a = repeat_learner("a"), b = repeat_learner("b")),
           sift = grains("week", "month"), ensemble = ensemble, resampling = resampling,
           n_inner = n_inner, control = NULL, verbose = FALSE, ...)
}

test_that("a repeated resampling says how many times it is drawn", {
  spec <- cv(v = 4L, seed = 3L, repeats = 3L)
  expect_equal(spec$repeats, 3L)
  expect_output(print(spec), "repeated 3 times")
  one <- timesift:::.nth_repeat(spec, 2L)
  expect_equal(c(one$seed, one$repeats), c(4L, 1L))
  expect_equal(grouped_cv("site", repeats = 2L)$repeats, 2L)
  expect_equal(cv()$repeats, 1L)
  expect_error(cv(repeats = 0L), "1 or more")
  expect_error(grouped_cv("site", repeats = 0L), "1 or more")
})

test_that("each repeat is a run on its own fold map, scored as the run alone would be", {
  case <- repeat_case()
  rep3 <- run_repeat(case, cv(v = 3L, seed = 5L, repeats = 3L))
  expect_s3_class(rep3, "timesift")
  expect_equal(rep3$repeats, 3L)
  expect_true("repeat" %in% names(rep3$scores))
  expect_equal(sort(unique(rep3$scores[["repeat"]])), 1:3)
  expect_equal(sort(unique(rep3$scores$fold)), 1:9)
  expect_equal(sort(unique(rep3$cells$fold)), 1:9)
  for (r in 1:3) {
    alone <- run_repeat(case, cv(v = 3L, seed = 5L + r - 1L))
    got <- rep3$scores[rep3$scores[["repeat"]] == r, ]
    got$fold <- got$fold - (r - 1L) * 3L
    want <- alone$scores
    key <- c("candidate", "variable", "fold")
    got <- got[do.call(order, got[key]), ]
    want <- want[do.call(order, want[key]), ]
    expect_equal(got$score, want$score, info = paste("repeat", r))
    expect_equal(got$scorable, want$scorable, info = paste("repeat", r))
  }
  # The out-of-fold prediction of a target is its mean over the repeats.
  singles <- lapply(1:3, function(r) run_repeat(case, cv(v = 3L, seed = 5L + r - 1L)))
  want <- Reduce(`+`, lapply(singles, function(s) s$oof$`a / week`[rownames(rep3$y), ])) / 3
  expect_equal(rep3$oof$`a / week`, want)
})

test_that("a response is averaged over its folds and its repeats", {
  case <- repeat_case()
  fit <- run_repeat(case, cv(v = 3L, seed = 5L, repeats = 2L))
  per_response <- timesift:::.per_response(fit)
  scored <- fit$scores[fit$scores$scorable & !is.na(fit$scores$score), ]
  want <- stats::aggregate(score ~ candidate + variable, scored, mean)
  got <- merge(per_response, want, by = c("candidate", "variable"))
  expect_equal(got$score.x, got$score.y)
  expect_output(print(fit), "repeated 2 times")
})

test_that("the stack is fitted on the out-of-fold predictions of all the repeats together", {
  case <- repeat_case()
  fit <- run_repeat(case, cv(v = 3L, seed = 5L, repeats = 2L), ensemble = ensemble("stack"))
  expect_s3_class(fit$stack, "timesift_stack")
  expect_equal(sum(fit$stack$weights), 1)
  single <- run_repeat(case, cv(v = 3L, seed = 5L), ensemble = ensemble("stack"))
  # Twice the cells, so the combiner saw more than a run alone gives it.
  expect_gt(fit$stack$n_cell, single$stack$n_cell)
  p <- stats::predict(fit, case$targets, case$series)
  expect_equal(dim(p), c(nrow(case$targets), 2L))
})

test_that("a nested estimate is read off all the repeats", {
  case <- repeat_case()
  fit <- run_repeat(case, cv(v = 3L, seed = 5L, repeats = 2L), ensemble = ensemble("stack"), n_inner = 2L)
  expect_true(all(c("selected", "ensemble") %in% fit$estimate$arm))
  expect_true(is.finite(fit$estimate$score[fit$estimate$arm == "selected" &
                                              fit$estimate$metric == "roc_auc"]))
  expect_equal(sort(unique(fit$selected[["repeat"]])), 1:2)
  expect_equal(sort(unique(fit$selected$fold)), 1:6)
  expect_equal(dim(fit$predictions$selected), dim(fit$y))
})

test_that("a grouped split repeats too, and the models are the first repeat's", {
  case <- repeat_case()
  fit <- run_repeat(case, grouped_cv("group", v = 3L, seed = 2L, repeats = 2L))
  expect_equal(fit$repeats, 2L)
  expect_equal(sort(unique(fit$scores[["repeat"]])), 1:2)
  expect_true(all(c("a / week", "b / month") %in% names(fit$models)))
  alone <- run_repeat(case, grouped_cv("group", v = 3L, seed = 2L))
  expect_equal(unclass(fit$folds)[names(unclass(alone$folds))], unclass(alone$folds)[
    names(unclass(alone$folds))], ignore_attr = TRUE)
})

test_that("a run says which repeat it is on", {
  case <- repeat_case()
  expect_message(timesift(case$targets, case$series, y = starts_with("sp"), id = plot, time = t,
                          learners = list(a = repeat_learner("a")), sift = grains("month"),
                          ensemble = FALSE, resampling = cv(v = 3L, repeats = 2L), n_inner = NULL,
                          control = NULL, verbose = TRUE), "repeat 2 of 2")
})
