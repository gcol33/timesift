# The package logo: one year of one plot's soil temperature, its daily range in grey and its
# weekly means in blue, over a sieve mesh drifting one cell per cycle. Writes man/figures/logo.svg.
library(timesift)
source("data-raw/deposit.R")

record <- deposit_record()
year <- record[record$time >= as.POSIXct("2023-03-01", tz = "UTC") &
                 record$time < as.POSIXct("2024-03-01", tz = "UTC"), ]

daily <- grain_matrix(year, plot, time, temp, grain = "day", stats = c("min", "max"))
weekly <- grain_matrix(year, plot, time, temp, grain = "week")

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

from <- as.POSIXct("2023-03-01", tz = "UTC")
to <- as.POSIXct("2024-03-01", tz = "UTC")
px <- function(t) -20 + 560 * as.numeric(t - from, units = "secs") /
  as.numeric(to - from, units = "secs")
temp_lim <- c(-2, 22)
py <- function(v) 360 - 230 * (v - temp_lim[1]) / diff(temp_lim)

day_start <- bin_starts(daily)
day_mid <- day_start + 43200
lo <- daily[1L, , "min"]
hi <- daily[1L, , "max"]
band <- paste0(
  "M", paste(sprintf("%.1f,%.1f", px(day_mid), py(hi)), collapse = " L"),
  " L", paste(sprintf("%.1f,%.1f", rev(px(day_mid)), rev(py(lo))), collapse = " L"), " Z"
)

week_start <- bin_starts(weekly)
week_end <- c(week_start[-1L], to)
wv <- weekly[1L, , 1L]
steps <- paste0("M", paste(sprintf("%.1f,%.1f %.1f,%.1f", px(week_start), py(wv),
                                     px(week_end), py(wv)), collapse = " L"))

# The mesh: lines at plus and minus 45 degrees, `cell` apart along the horizontal, so a shift of
# one cell to the right maps it onto itself and the loop closes.
cell <- 44
offsets <- seq(-640, 1200, by = cell)
mesh_path <- paste(c(
  sprintf("M%d,-40 L%d,640", offsets, offsets - 680),
  sprintf("M%d,-40 L%d,640", offsets - 680, offsets)
), collapse = " ")
mesh_layer <- function(class, animated) {
  c(sprintf('<g class="%s">', class),
    sprintf('<path d="%s" fill="none" stroke="%s" stroke-width="2"/>', mesh_path, mesh),
    if (animated) sprintf(paste0('<animateTransform attributeName="transform" type="translate" ',
                                 'from="0 0" to="%d 0" dur="8s" repeatCount="indefinite"/>'), cell),
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
  sprintf('<path d="%s" fill="none" stroke="%s" stroke-width="9" stroke-linejoin="miter"/>',
          steps, blue),
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
writeLines(svg, "man/figures/logo.svg", useBytes = TRUE)
