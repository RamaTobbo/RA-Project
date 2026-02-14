#!/bin/bash
# Start R script in background


# Start Python app
gunicorn app:app --bind 0.0.0.0:$PORT --timeout 600 --workers 1
