# Use a slim Python image
FROM python:3.10-slim

# Install system dependencies + R
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base r-base-dev \
    build-essential \
    cmake pkg-config \
    rustc cargo \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    libfontconfig1-dev libharfbuzz-dev libfribidi-dev \
    libpng-dev libjpeg-dev libtiff5-dev \
    libudunits2-dev \
    libgdal-dev \
    libgeos-dev \
    libproj-dev proj-data \
    && rm -rf /var/lib/apt/lists/*


RUN R -e "install.packages(c('ggplot2','gganimate','gifski','transformr','tweenr'), repos='https://cloud.r-project.org')"


WORKDIR /app

# Copy and install Python dependencies first (better caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . /app

# Render uses PORT env var
ENV PYTHONUNBUFFERED=1

# Start with gunicorn (recommended for Render)
CMD ["bash", "-lc", "gunicorn app:app --bind 0.0.0.0:$PORT"]
