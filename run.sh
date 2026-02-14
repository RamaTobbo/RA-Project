#!/bin/bash
# Start R script in background


#!/bin/sh
gunicorn app:app --bind 0.0.0.0:$PORT --timeout 1200 --workers 1 --threads 2

