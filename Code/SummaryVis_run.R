# Code/SummaryVis_run.R
# Usage:
#   Rscript Code/SummaryVis_run.R "<RUN_DIR>"

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Missing RUN_DIR argument.")

run_dir <- args[1]

suppressPackageStartupMessages({
  library(ggplot2)
  library(gganimate)
  library(gifski)
})

read_required <- function(path) {
  if (!file.exists(path)) stop(paste("Missing file:", path))
  df <- read.csv(path, stringsAsFactors = FALSE)

  # Force numeric (important if they come as text)
  df$Iteration <- suppressWarnings(as.numeric(df$Iteration))
  if ("Best" %in% names(df))    df$Best    <- suppressWarnings(as.numeric(df$Best))
  if ("Average" %in% names(df)) df$Average <- suppressWarnings(as.numeric(df$Average))

  # Remove bad rows
  df <- df[!is.na(df$Iteration), ]
  df
}

# ---- Read summarized curves (Iteration, Best, Average)
datafish <- read_required(file.path(run_dir, "datafish.csv"))
databat  <- read_required(file.path(run_dir, "databat.csv"))
datawolf <- read_required(file.path(run_dir, "datawolf.csv"))
databee  <- read_required(file.path(run_dir, "databee.csv"))

# ---- Global y-limits so all algos share same scale
best_all <- c(datafish$Best, databat$Best, datawolf$Best, databee$Best)
avg_all  <- c(datafish$Average, databat$Average, datawolf$Average, databee$Average)

best_min <- min(best_all, na.rm = TRUE)
best_max <- max(best_all, na.rm = TRUE)

avg_min  <- min(avg_all, na.rm = TRUE)
avg_max  <- max(avg_all, na.rm = TRUE)

max_iter <- max(c(datafish$Iteration, databat$Iteration, datawolf$Iteration, databee$Iteration), na.rm = TRUE)

# Small padding so lines don’t touch border
pad_range <- function(a, b) {
  if (is.infinite(a) || is.infinite(b)) return(c(0, 1))
  if (a == b) return(c(a - 1, b + 1))
  pad <- 0.06 * (b - a)
  c(a - pad, b + pad)
}

best_lim <- pad_range(best_min, best_max)
avg_lim  <- pad_range(avg_min, avg_max)

# ---- Build + animate plot
make_anim <- function(df, ycol, title_txt, colr, y_lim) {
  p <- ggplot(df, aes(x = Iteration, y = .data[[ycol]])) +
    geom_line(linewidth = 1.6, color = colr) +
    geom_point(aes(group = seq_along(Iteration)), size = 2.6, color = colr) +
    scale_x_continuous(limits = c(0, max_iter), expand = c(0, 0)) +
    scale_y_continuous(limits = y_lim, expand = c(0, 0)) +
    labs(title = title_txt, x = "Iteration", y = ycol) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      panel.grid.minor = element_blank(),
      plot.margin = margin(10, 10, 10, 10)
    ) +
    transition_reveal(Iteration)

  animate(
    p,
    renderer = gifski_renderer(),
    width = 460, height = 460, fps = 12,
    end_pause = 10
  )
}

# ---- Best GIFs
anim_save(file.path(run_dir, "Fish_Fun_Best.gif"),
          animation = make_anim(datafish, "Best", "Fish — Best Fitness", "brown1", best_lim))

anim_save(file.path(run_dir, "Bat_Fun_Best.gif"),
          animation = make_anim(databat, "Best", "Bat — Best Fitness", "purple", best_lim))

anim_save(file.path(run_dir, "Wolf_Fun_Best.gif"),
          animation = make_anim(datawolf, "Best", "Wolf — Best Fitness", "royalblue", best_lim))

anim_save(file.path(run_dir, "Bee_Fun_Best.gif"),
          animation = make_anim(databee, "Best", "Bee — Best Fitness", "orange", best_lim))

# ---- Average GIFs
anim_save(file.path(run_dir, "Fish_Fun_Average.gif"),
          animation = make_anim(datafish, "Average", "Fish — Average Fitness", "brown1", avg_lim))

anim_save(file.path(run_dir, "Bat_Fun_Average.gif"),
          animation = make_anim(databat, "Average", "Bat — Average Fitness", "purple", avg_lim))

anim_save(file.path(run_dir, "Wolf_Fun_Average.gif"),
          animation = make_anim(datawolf, "Average", "Wolf — Average Fitness", "royalblue", avg_lim))

anim_save(file.path(run_dir, "Bee_Fun_Average.gif"),
          animation = make_anim(databee, "Average", "Bee — Average Fitness", "orange", avg_lim))

cat("SummaryVis done. GIFs saved in:", run_dir, "\n")
