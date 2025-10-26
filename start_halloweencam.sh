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

echo "Waiting 10 seconds for system to be ready..."
sleep 2

echo "Running git pull..."
git pull

echo "Activating virtual environment..."
source "$PROJECT_DIR/venv/bin/activate"

echo "Running pip install..."
pip install -e .

# Use MJPEG @ 640x360 60 FPS (supported by your cam)
v4l2-ctl --set-fmt-video=width=640,height=480,pixelformat=MJPG
v4l2-ctl --set-parm=30

echo "Locking camera exposure settings..."
# Set Auto Exposure to Manual Mode (value 1)
v4l2-ctl -c auto_exposure=1
# Set Absolute Exposure Time
v4l2-ctl -c exposure_time_absolute=666
# Set Gain
v4l2-ctl -c gain=222
# Disable Continuous Auto Focus
v4l2-ctl -c focus_automatic_continuous=0
# Disable Auto White Balance
v4l2-ctl -c white_balance_automatic=0


echo "Launching ghoulfluids GUI..."
# Explicitly set the display for GUI applications
export DISPLAY=:0
ghoulfluids --fullscreen --debug --segmenter='yolo' --log-file="$LOG_FILE" --seg-width=640 --seg-height=480 &

echo "Script finished executing ghoulfluids command at $(date)."
