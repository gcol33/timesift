source("~/.R/build_pkgdown.R")

# The Python reference is read from the sources, and the Python articles are run against the
# installed package, so the interpreter has to be the one timesift is installed in. Under R on
# Windows `python` can resolve to Rtools' own, which has neither; `PYTHON` names the right one.
python <- Sys.getenv("PYTHON", unset = "python")
for (tool in c("tools/python_reference.py", "tools/python_articles.py")) {
  if (system2(python, tool) != 0L) {
    stop(tool, " failed under ", python, ". Set PYTHON to the interpreter timesift is installed in.",
         call. = FALSE)
  }
}
pkgdown::clean_site(quiet = TRUE)
build_pkgdown_site()
# The digest the site workflow checks the committed build against.
system2("Rscript", c("tools/site_digest.R", "--write"))
