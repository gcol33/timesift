# The package logo: one year of one plot's soil temperature, its daily range in grey, and a blue
# line that coarsens from the hourly readings through the half day and the day to the weekly means
# and back, over a sieve mesh drifting one cell per cycle. Readers who ask for reduced motion see
# the weekly means. Writes man/figures/logo.svg.
library(timesift)
source("data-raw/deposit.R")

from <- as.POSIXct("2023-03-01", tz = "UTC")
to <- as.POSIXct("2024-03-01", tz = "UTC")
record <- deposit_record()
year <- record[record$time >= from & record$time < to, ]

border <- "#2B3A4A"
paper <- "#F5F6F8"
blue <- "#2E6F9E"
grey <- "#5C6166"
mesh <- "#D9DEE4"

cx <- 260
cy <- 300
hex <- function(r) {
  a <- (90 + 60 * 0:5) * pi / 180
  paste(sprintf("%.1f,%.1f", cx + r * cos(a), cy - r * sin(a)), collapse = " ")
}

px <- function(t) -20 + 560 * as.numeric(t - from, units = "secs") /
  as.numeric(to - from, units = "secs")
temp_lim <- c(-2, 22)
py <- function(v) 360 - 230 * (v - temp_lim[1]) / diff(temp_lim)
pt <- function(x, y) sprintf("%.1f,%.0f", x, y)

daily <- grain_matrix(year, plot, time, temp, grain = "day", stats = c("min", "max"))
day_mid <- bin_starts(daily) + 43200
band <- paste0(
  "M", paste(pt(px(day_mid), py(daily[1L, , "max"])), collapse = " L"),
  " L", paste(pt(rev(px(day_mid)), rev(py(daily[1L, , "min"]))), collapse = " L"), " Z"
)

# Every frame of the moving line is drawn on one grid of half-day cells, two points per cell, so
# all frames carry the same points and SMIL can interpolate between them. A coarse grain holds one
# value across its cells and steps at its bin edges, which fall on cell edges; the hourly frame
# runs through each half day's coldest and warmest reading in the order they were read.
cell_start <- seq(from, to - 43200, by = 43200)
cell_x <- as.vector(rbind(px(cell_start), px(cell_start + 43200)))
cell <- findInterval(year$time, cell_start)

hourly <- unlist(lapply(split(year$temp, cell), function(v) {
  if (which.min(v) < which.max(v)) c(min(v), max(v)) else c(max(v), min(v))
}), use.names = FALSE)

coarse <- function(g) {
  m <- grain_matrix(year, plot, time, temp, grain = g, year_start = "03-01")
  v <- m[1L, findInterval(cell_start, bin_starts(m)), 1L]
  rep(v, each = 2L)
}

grains <- c("halfday", "day", "week")
frames <- c(list(hourly), lapply(grains, coarse))
stopifnot(all(lengths(frames) == length(cell_x)))
frame_d <- vapply(frames, function(v) {
  paste0("M", paste(sprintf("%.0f,%.0f", cell_x, py(v)), collapse = " L"))
}, character(1))
widths <- c(3, 4, 6, 9)

# There and back, easing into each grain so the eye can rest on it.
order <- c(seq_along(frames), rev(seq_along(frames))[-1L])
n <- length(order)
key_times <- paste(sprintf("%.4f", (seq_len(n) - 1L) / (n - 1L)), collapse = ";")
splines <- paste(rep("0.45 0 0.55 1", n - 1L), collapse = ";")
morph <- c(
  sprintf('<path d="%s" fill="none" stroke="%s" stroke-width="%g" stroke-linejoin="miter">',
          frame_d[1L], blue, widths[1L]),
  sprintf(paste0('<animate attributeName="d" dur="12s" repeatCount="indefinite" ',
                 'calcMode="spline" keyTimes="%s" keySplines="%s" values="%s"/>'),
          key_times, splines, paste(frame_d[order], collapse = ";")),
  sprintf(paste0('<animate attributeName="stroke-width" dur="12s" repeatCount="indefinite" ',
                 'calcMode="spline" keyTimes="%s" keySplines="%s" values="%s"/>'),
          key_times, splines, paste(widths[order], collapse = ";")),
  "</path>"
)

weekly <- grain_matrix(year, plot, time, temp, grain = "week")
week_start <- bin_starts(weekly)
week_end <- c(week_start[-1L], to)
wv <- weekly[1L, , 1L]
steps <- paste0("M", paste(pt(px(week_start), py(wv)), pt(px(week_end), py(wv)),
                           sep = " L", collapse = " L"))

# The mesh: lines at plus and minus 45 degrees, `cell` apart along the horizontal, so a shift of
# one cell to the right maps it onto itself and the loop closes.
mesh_cell <- 44
offsets <- seq(-640, 1200, by = mesh_cell)
mesh_path <- paste(c(
  sprintf("M%d,-40 L%d,640", offsets, offsets - 680),
  sprintf("M%d,-40 L%d,640", offsets - 680, offsets)
), collapse = " ")
mesh_layer <- function(class, animated) {
  c(sprintf('<g class="%s">', class),
    sprintf('<path d="%s" fill="none" stroke="%s" stroke-width="2"/>', mesh_path, mesh),
    if (animated) sprintf(paste0('<animateTransform attributeName="transform" type="translate" ',
                                 'from="0 0" to="%d 0" dur="8s" repeatCount="indefinite"/>'),
                          mesh_cell),
    "</g>")
}

svg <- c(
  '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 600" role="img" aria-label="timesift logo">',
  "<title>timesift logo</title>",
  "<style>",
  ".still { display: none; }",
  "@media (prefers-reduced-motion: reduce) { .moving { display: none; } .still { display: inline; } }",
  "</style>",
  sprintf('<defs><clipPath id="hex"><polygon points="%s"/></clipPath></defs>', hex(288)),
  sprintf('<polygon points="%s" fill="%s"/>', hex(288), paper),
  '<g clip-path="url(#hex)">',
  mesh_layer("moving", TRUE),
  mesh_layer("still", FALSE),
  sprintf('<path d="%s" fill="%s" fill-opacity="0.6" stroke="none"/>', band, grey),
  '<g class="moving">', morph, "</g>",
  sprintf(paste0('<path class="still" d="%s" fill="none" stroke="%s" stroke-width="9" ',
                 'stroke-linejoin="miter"/>'), steps, blue),
  "</g>",
  sprintf(paste0('<text x="%d" y="472" text-anchor="middle" font-family="Georgia, \'Times New Roman\', serif" ',
                 'font-weight="bold" font-style="italic" font-size="78" fill="%s" stroke="%s" ',
                 'stroke-width="10" paint-order="stroke" stroke-linejoin="round">timesift</text>'),
          cx, blue, paper),
  sprintf('<polygon points="%s" fill="none" stroke="%s" stroke-width="18" stroke-linejoin="round"/>',
          hex(291), border),
  "</svg>"
)

dir.create("man/figures", showWarnings = FALSE, recursive = TRUE)
con <- file("man/figures/logo.svg", "wb")
writeLines(svg, con, useBytes = TRUE)
close(con)
