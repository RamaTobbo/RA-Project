# Code/Preprocessing_run.R
# Usage:
#   Rscript Code/Preprocessing_run.R "<RUN_DIR>"

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Missing RUN_DIR argument.")

run_dir <- args[1]

read_required <- function(path) {
  if (!file.exists(path)) stop(paste("Missing file:", path))

  # Your Python writes header=False so read with header=FALSE
  df <- read.csv(path, header = FALSE, stringsAsFactors = FALSE)

  if (nrow(df) < 2) stop(paste("File has no data:", path))

  # First row is actually the header (strings)
  hdr <- as.character(df[1, ])
  df  <- df[-1, , drop = FALSE]
  colnames(df) <- hdr

  df
}

# Reads raw fitness logs from run folder
fish_raw <- read_required(file.path(run_dir, "fish_fun_fitness.csv"))
bat_raw  <- read_required(file.path(run_dir, "bat_fun_fitness.csv"))
wolf_raw <- read_required(file.path(run_dir, "wolf_fun_fitness.csv"))
bee_raw  <- read_required(file.path(run_dir, "bee_fun_fitness.csv"))

# ---------------------------------------------------------
# Helper: summarize one raw CSV into (Iteration, Best, Average)
# Works even if df has extra columns like Phase/Id.
# ---------------------------------------------------------
summarize_fitness <- function(df) {
  if (!("Iteration" %in% names(df))) stop("CSV missing Iteration column.")
  if (!("Fitness" %in% names(df)))   stop("CSV missing Fitness column.")

  # Keep only needed columns
  df <- df[, c("Iteration", "Fitness")]

  # Convert to numeric safely
  df$Iteration <- suppressWarnings(as.numeric(df$Iteration))
  df$Fitness   <- suppressWarnings(as.numeric(df$Fitness))

  # Remove any bad rows (NAs)
  df <- df[!is.na(df$Iteration) & !is.na(df$Fitness), ]

  # Ensure ordered
  df <- df[order(df$Iteration), ]

  # Group by Iteration and compute min + mean
  out <- aggregate(
    Fitness ~ Iteration,
    df,
    function(x) c(Best = min(x), Average = mean(x))
  )

  # Unpack aggregate matrix column
  data.frame(
    Iteration = out$Iteration,
    Best      = out$Fitness[, "Best"],
    Average   = out$Fitness[, "Average"]
  )
}

# Summarize each algorithm
datafish <- summarize_fitness(fish_raw)
databat  <- summarize_fitness(bat_raw)
datawolf <- summarize_fitness(wolf_raw)
databee  <- summarize_fitness(bee_raw)

# Save in same run folder
write.csv(datafish, file.path(run_dir, "datafish.csv"), row.names = FALSE)
write.csv(databat,  file.path(run_dir, "databat.csv"),  row.names = FALSE)
write.csv(datawolf, file.path(run_dir, "datawolf.csv"), row.names = FALSE)
write.csv(databee,  file.path(run_dir, "databee.csv"),  row.names = FALSE)

cat("Preprocessing done. Saved datafish/databat/datawolf/databee in:", run_dir, "\n")
