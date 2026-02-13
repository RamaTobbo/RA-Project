FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# --- System deps (Python build tools + R + geospatial deps needed by sf/s2/units) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config \
    curl ca-certificates \
    rustc cargo \
    r-base r-base-dev \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    libfontconfig1-dev libharfbuzz-dev libfribidi-dev \
    libpng-dev libjpeg-dev libtiff5-dev \
    libudunits2-dev \
    libgdal-dev \
    libgeos-dev \
    libproj-dev proj-data \
    && rm -rf /var/lib/apt/lists/*

# --- Install R packages (with explicit repo) ---
RUN R -e "options(repos=c(CRAN='https://cloud.r-project.org')); install.packages(c('ggplot2','tweenr','transformr','gifski','gganimate'))"

# --- Hard check: FAIL build if packages not available ---
RUN R -e "library(ggplot2); library(gifski); library(transformr); library(tweenr); library(gganimate); cat('✅ R packages OK\\n')"

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Start server
CMD ["bash", "-lc", "gunicorn app:app --bind 0.0.0.0:$PORT"]
