# A hinge basis as the MARS and discriminant fixtures write it. A fit's terms are its factors,
# `column:direction` in column order with the column one-based and the intercept `1`, and the cuts
# apart so they are compared as numbers; a fixture's are the same terms written
# `column:direction:cut`, joined by `*` within a term and by a space between them.

hinge_terms <- function(f) {
  lapply(seq_len(length(f$factor_start) - 1L), function(t) {
    idx <- f$factor_start[t] + seq_len(f$factor_start[t + 1L] - f$factor_start[t])
    list(key = if (length(idx)) paste(sprintf("%d:%d", f$factor_column[idx] + 1L,
                                              f$factor_dir[idx]), collapse = "*") else "1",
         cut = f$factor_cut[idx])
  })
}

fixture_hinge_terms <- function(terms) {
  lapply(strsplit(terms, " ", fixed = TRUE)[[1L]], function(term) {
    if (term == "1") return(list(key = "1", cut = numeric()))
    parts <- strsplit(strsplit(term, "*", fixed = TRUE)[[1L]], ":", fixed = TRUE)
    list(key = paste(vapply(parts, function(p) paste(p[1:2], collapse = ":"), character(1L)),
                     collapse = "*"),
         cut = as.numeric(vapply(parts, `[`, character(1L), 3L)))
  })
}

expect_hinge_terms <- function(f, terms, tolerance, info) {
  got <- hinge_terms(f)
  want <- fixture_hinge_terms(terms)
  expect_identical(vapply(got, `[[`, character(1L), "key"),
                   vapply(want, `[[`, character(1L), "key"), info = info)
  expect_equal(unlist(lapply(got, `[[`, "cut")), unlist(lapply(want, `[[`, "cut")),
               tolerance = tolerance, info = info)
}
