# One plot's hourly soil temperature from the Schrankogel deposit (doi:10.5281/zenodo.17047026),
# the record the logo and the README figure are drawn from. TIMESIFT_DEPOSIT names the folder
# holding logger_data.csv. A plot's readings are spread over the file, one block per logger it
# carried, so the whole file is read and the plot's 26304 hours gathered from it.
deposit_record <- function(plot = 7L) {
  folder <- Sys.getenv(
    "TIMESIFT_DEPOSIT",
    "E:/research/vault/data/sources/schrankogel/zenodo_17047026/data"
  )
  x <- data.table::fread(file.path(folder, "logger_data.csv"),
                         select = c("logger_ID", "date", "temp"))
  x <- x[x$logger_ID == plot, ]
  x <- x[order(x$date), ]
  stopifnot(nrow(x) == 26304L, !anyDuplicated(x$date), !anyNA(x$temp))
  data.frame(
    plot = x$logger_ID,
    time = as.POSIXct(x$date, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
    temp = x$temp
  )
}

# Bin starts of a grain_matrix() result as instants.
bin_starts <- function(m) {
  as.POSIXct(dimnames(m)[[2L]], format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
}
