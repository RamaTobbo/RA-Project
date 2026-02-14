# --------------------------------------------------
# Base Image (R preinstalled)
# --------------------------------------------------
FROM rocker/r-ver:4.3.2

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# --------------------------------------------------
# System Dependencies
# --------------------------------------------------
# --------------------------------------------------
# Make APT reliable
# --------------------------------------------------
RUN set -eux; \
    sed -i 's|http://archive.ubuntu.com|https://archive.ubuntu.com|g' /etc/apt/sources.list; \
    printf 'Acquire::Retries "10";\nAcquire::http::Timeout "120";\nAcquire::https::Timeout "120";\nAcquire::ForceIPv4 "true";\n' > /etc/apt/apt.conf.d/99retries

# --------------------------------------------------
# System Dependencies (with retry loop)
# --------------------------------------------------
RUN set -eux; \
    for i in 1 2 3 4 5; do \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            ca-certificates \
            curl \
            git \
            python3 \
            python3-pip \
            python3-venv \
            build-essential \
            cmake \
            pkg-config \
            zlib1g-dev \
            libgit2-dev \
            ffmpeg \
            libavfilter-dev \
            libavformat-dev \
            libavcodec-dev \
            libswscale-dev \
            libmagick++-dev \
            libcurl4-openssl-dev \
            libssl-dev \
            libxml2-dev \
            libfontconfig1-dev \
            libharfbuzz-dev \
            libfribidi-dev \
            libpng-dev \
            libjpeg-dev \
            libtiff5-dev \
            libgdal-dev \
            libgeos-dev \
            libproj-dev \
            proj-data \
            libudunits2-dev \
            cargo \
            rustc \
        && break || (echo "APT failed, retry $i/5..." && sleep 10); \
    done; \
    rm -rf /var/lib/apt/lists/*


# --------------------------------------------------
# Install R Packages (order matters for stability)
# --------------------------------------------------
RUN R -e "options(repos='https://cloud.r-project.org'); \
          install.packages(c('av','magick'), Ncpus=parallel::detectCores())"

RUN R -e "options(repos='https://cloud.r-project.org'); \
          install.packages('gifski', Ncpus=parallel::detectCores())"

RUN R -e "options(repos='https://cloud.r-project.org'); \
          install.packages(c('ggplot2','tweenr','transformr','gganimate'), \
                           Ncpus=parallel::detectCores())"

# --------------------------------------------------
# Verify R Packages (fails fast if something wrong)
# --------------------------------------------------
RUN R -e "pkgs <- c('ggplot2','av','magick','gifski','transformr','tweenr','gganimate'); \
          for (p in pkgs) { \
            cat('Loading', p, '... '); \
            suppressPackageStartupMessages(library(p, character.only=TRUE)); \
            cat('OK\\n') \
          }"

# --------------------------------------------------
# Python Dependencies
# --------------------------------------------------
WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# --------------------------------------------------
# Copy App
# --------------------------------------------------
COPY . .

# --------------------------------------------------
# Start App (Render uses $PORT)
# --------------------------------------------------
CMD ["bash", "-lc", "gunicorn app:app --bind 0.0.0.0:${PORT} --timeout 600 --workers 1"]

