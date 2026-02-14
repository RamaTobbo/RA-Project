# Code/SummaryVis_one.R
# Usage:
#   Rscript Code/SummaryVis_one.R "<RUN_DIR>" "<ALGO>" "<METRIC>"

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("Usage: SummaryVis_one.R <RUN_DIR> <ALGO> <METRIC>")

run_dir <- args[1]
algo <- tolower(args[2])     # wolf|bee|bat|fish
metric <- tolower(args[3])   # best|average

suppressPackageStartupMessages({
  library(ggplot2)
  library(gganimate)
  library(gifski)
})

read_required <- function(path) {
  if (!file.exists(path)) stop(paste("Missing file:", path))
  df <- read.csv(path, stringsAsFactors = FALSE)

  df$Iteration <- suppressWarnings(as.numeric(df$Iteration))
  if ("Best" %in% names(df))    df$Best    <- suppressWarnings(as.numeric(df$Best))
  if ("Average" %in% names(df)) df$Average <- suppressWarnings(as.numeric(df$Average))

  df <- df[!is.na(df$Iteration), ]
  df
}

# pick dataset based on algo
csv_map <- list(
  wolf = "datawolf.csv",
  bee  = "databee.csv",
  bat  = "databat.csv",
  fish = "datafish.csv"
)

if (!(algo %in% names(csv_map))) stop(paste("Invalid algo:", algo))
df <- read_required(file.path(run_dir, csv_map[[algo]]))

# choose column
ycol <- if (metric == "best") "Best" else if (metric == "average") "Average" else stop("Invalid metric")

# compute limits using only this algo (faster)
vals <- df[[ycol]]
ymin <- min(vals, na.rm = TRUE)
ymax <- max(vals, na.rm = TRUE)

pad_range <- function(a, b) {
  if (is.infinite(a) || is.infinite(b)) return(c(0, 1))
  if (a == b) return(c(a - 1, b + 1))
  pad <- 0.06 * (b - a)
  c(a - pad, b + pad)
}
ylim <- pad_range(ymin, ymax)

max_iter <- max(df$Iteration, na.rm = TRUE)

# colors + titles
col_map <- list(wolf="royalblue", bee="orange", bat="purple", fish="brown1")
title_map <- list(
  best = "Best Fitness",
  average = "Average Fitness"
)

out_map <- list(
  wolf = list(best="Wolf_Fun_Best.gif", average="Wolf_Fun_Average.gif"),
  bee  = list(best="Bee_Fun_Best.gif",  average="Bee_Fun_Average.gif"),
  bat  = list(best="Bat_Fun_Best.gif",  average="Bat_Fun_Average.gif"),
  fish = list(best="Fish_Fun_Best.gif", average="Fish_Fun_Average.gif")
)

outfile <- file.path(run_dir, out_map[[algo]][[metric]])
plot_title <- paste0(toupper(substr(algo,1,1)), substr(algo,2,nchar(algo)), " — ", title_map[[metric]])

p <- ggplot(df, aes(x = Iteration, y = .data[[ycol]])) +
  geom_line(linewidth = 1.2, color = col_map[[algo]]) +
  geom_point(aes(group = seq_along(Iteration)), size = 2.0, color = col_map[[algo]]) +
  scale_x_continuous(limits = c(0, max_iter), expand = c(0, 0)) +
  scale_y_continuous(limits = ylim, expand = c(0, 0)) +
  labs(title = plot_title, x = "Iteration", y = ycol) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(size = 13, face = "bold"),
    panel.grid.minor = element_blank(),
    plot.margin = margin(8, 8, 8, 8)
  ) +
  transition_reveal(Iteration, keep_last = TRUE)

step <- max(1, floor(nrow(df)/150))
df <- df[seq(1, nrow(df), by = step), ]

anim <- animate(
  p,
  renderer = gifski_renderer(),
  width = 260,
  height = 260,
  fps = 4,
  end_pause = 0,
  nframes = 60
)


anim_save(outfile, animation = anim)
cat("One GIF done:", outfile, "\n")
