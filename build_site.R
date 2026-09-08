source("~/.R/build_pkgdown.R")

system2("python", c("tools/python_reference.py"))
pkgdown::clean_site(quiet = TRUE)
build_pkgdown_site()
# The digest the site workflow checks the committed build against.
system2("Rscript", c("tools/site_digest.R", "--write"))
