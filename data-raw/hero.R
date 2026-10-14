# The README's figure: three years of one plot's hourly soil temperature read at the seven grains
# of the Schrankogel study, beside the true skill statistic the convolutional network reached on
# each. Writes man/figures/timesift-hero.png.
library(timesift)
library(ggplot2)
source("data-raw/deposit.R")

record <- deposit_record()

grains <- c("native", "halfday", "day", "week", "month", "season", "year")
labels <- c("hour", "half day", "day", "week", "month", "season", "year")

# Mean TSS over 101 species of the convolutional network at each grain, as published in the
# study (table S12) and reproduced by inst/reproduce/schrankogel.R.
tss <- c(0.658, 0.682, 0.689, 0.706, 0.691, 0.670, 0.626)
best <- which.max(tss)

blue <- "#2E6F9E"
grey <- "#5C6166"
light <- "#8A9097"

m <- grain_matrix(record, plot, time, temp, grain = grains, year_start = "09-01")

from <- min(record$time)
to <- max(record$time) + 3600
temp_range <- range(record$temp)

# Columns of the frame, in its own units: labels, record, scores.
x_label <- 14.5
x0 <- 16.5
x1 <- 63
s0 <- 70
s1 <- 88
tss_lim <- c(0.62, 0.71)
row_h <- 1
band <- 0.40

row_y <- function(i) (length(grains) - i) * row_h
time_x <- function(t) x0 + (x1 - x0) * as.numeric(t - from, units = "secs") /
  as.numeric(to - from, units = "secs")
temp_y <- function(v, i) row_y(i) - band + 2 * band * (v - temp_range[1]) / diff(temp_range)
tss_x <- function(v) s0 + (s1 - s0) * (v - tss_lim[1]) / diff(tss_lim)

steps <- do.call(rbind, lapply(seq_along(grains), function(i) {
  g <- m[[grains[i]]]
  start <- bin_starts(g)
  end <- c(start[-1L], to)
  v <- g[1L, , 1L]
  data.frame(
    grain = grains[i],
    x = as.vector(rbind(time_x(start), time_x(end))),
    y = temp_y(rep(v, each = 2L), i),
    colour = if (i == best) blue else grey
  )
}))

rows <- data.frame(
  label = labels,
  y = row_y(seq_along(grains)),
  tss = tss,
  colour = ifelse(seq_along(grains) == best, blue, grey),
  face = ifelse(seq_along(grains) == best, "bold", "plain")
)

hourly <- steps$grain == "native"

p <- ggplot() +
  geom_line(data = steps[hourly, ], aes(x, y), colour = grey, linewidth = 0.2) +
  geom_path(data = steps[!hourly, ], aes(x, y, group = grain, colour = colour),
            linewidth = 0.7, lineend = "butt", linejoin = "mitre") +
  geom_segment(data = rows, aes(x = s0, xend = s1, y = y, yend = y),
               colour = "#E3E6EA", linewidth = 0.5) +
  geom_path(data = rows, aes(tss_x(tss), y), colour = light, linewidth = 0.6) +
  geom_point(data = rows, aes(tss_x(tss), y, colour = colour),
             size = ifelse(seq_along(grains) == best, 7, 5)) +
  geom_text(data = rows, aes(x_label, y, label = label, colour = colour, fontface = face),
            hjust = 1, size = 12.5) +
  geom_text(data = rows, aes(tss_x(tss) + 2, y, label = sprintf("%.3f", tss),
                             colour = colour, fontface = face),
            hjust = 0, size = 11.5) +
  annotate("text", x = (x0 + x1) / 2, y = row_y(0) - 0.05, label = "soil temperature",
           colour = grey, size = 9.5, vjust = 0.5) +
  annotate("text", x = tss_x(0.666), y = row_y(0) - 0.05, label = "TSS",
           colour = grey, size = 9.5, vjust = 0.5) +
  scale_colour_identity() +
  coord_cartesian(xlim = c(0, 100), ylim = c(row_y(length(grains)) - 0.55, row_y(0) + 0.35),
                  expand = FALSE, clip = "off") +
  theme_void() +
  theme(plot.background = element_rect(fill = "white", colour = NA),
        plot.margin = margin(6, 10, 6, 6))

dir.create("man/figures", showWarnings = FALSE, recursive = TRUE)
ggsave("man/figures/timesift-hero.png", p, width = 6 * 2.26, height = 6, dpi = 120,
       device = ragg::agg_png)
