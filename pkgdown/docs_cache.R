# Chunk caching for the site build, switched on by build_site.R through TIMESIFT_DOCS_CACHE.
# A cached chunk is replayed until its own code changes or the installed package does, so the
# key carries the compiled library and the R code the articles run against.
installed <- find.package("timesift")
code <- list.files(file.path(installed, c("R", "libs")), "[.](rdb|dll|so)$",
                   recursive = TRUE, full.names = TRUE)
knitr::opts_chunk$set(cache = TRUE, autodep = TRUE,
                      cache.extra = list(unname(tools::md5sum(code)), getRversion()))
knitr::dep_auto()
