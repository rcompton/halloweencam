#!/bin/bash

# --- Setup Logging ---
# Redirect all output (stdout and stderr) to a log file.
LOG_FILE="$HOME/halloweencam_startup.log"
exec &>> "$LOG_FILE"

# --- Script Start ---
echo "============================================="
echo "Starting Halloween Cam script at $(date)"
echo "============================================="

PROJECT_DIR="$HOME/halloweencam"

echo "Changing directory to $PROJECT_DIR"
cd "$PROJECT_DIR" || { echo "ERROR: Failed to cd into project directory. Exiting."; exit 1; }

echo "Waiting 5 seconds for system to be ready..."
sleep 5

echo "Running git pull..."
git pull

echo "Activating virtual environment..."
source "$PROJECT_DIR/venv/bin/activate"

echo "Running pip install..."
pip install -e . --quiet

WIDTH=640
HEIGHT=480

# Use MJPEG @ WIDTHxHEIGHT
v4l2-ctl --set-fmt-video=width=${WIDTH},height=${HEIGHT},pixelformat=MJPG


echo "Launching ghoulfluids GUI..."
# Explicitly set the display for GUI applications
export DISPLAY=:0
ghoulfluids --fullscreen --debug --segmenter='yolo' --log-file="$LOG_FILE" --seg-width=${WIDTH} --seg-height=${HEIGHT} &

echo "Script finished executing ghoulfluids command at $(date)."