# Code/generate_gif.R
args <- commandArgs(trailingOnly=TRUE)
# args: input_csv output_gif ycol title color
in_csv  <- args[1]
out_gif <- args[2]
ycol    <- args[3]   # "Best" or "Average"
ttl     <- args[4]
colr    <- args[5]

suppressPackageStartupMessages({
  library(ggplot2)
  library(gganimate)
  library(gifski)
})

df <- read.csv(in_csv)

p <- ggplot(df, aes(x = Iteration, y = .data[[ycol]])) +
  geom_line(size=1.6, color=colr) +
  geom_point(aes(group = seq_along(Iteration)), size=1.2, color=colr) +
  theme_gray(base_size = 12) +
  labs(title = ttl, x="Iteration", y=ycol) +
  transition_reveal(Iteration)

anim <- animate(p, renderer = gifski_renderer(), width=520, height=520, fps=12)
anim_save(out_gif, animation = anim)
