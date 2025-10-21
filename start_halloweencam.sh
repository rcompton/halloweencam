#!/bin/bash

# --- Setup Logging ---
LOG_FILE="$HOME/halloweencam_startup.log"
exec &>> "$LOG_FILE"

echo "============================================="
echo "Starting Halloween Cam script at $(date)"
echo "============================================="

PROJECT_DIR="$HOME/halloweencam"

# --- Configure Razer Kiyo Pro ---
echo "Configuring Razer Kiyo Pro..."
# Use MJPEG @ 640x384 60 FPS (stable, fast, low USB bandwidth)
v4l2-ctl --set-fmt-video=width=640,height=384,pixelformat=MJPG
v4l2-ctl --set-parm=60

# Manual exposure and stability settings
v4l2-ctl --set-ctrl=exposure_auto=1              # manual
v4l2-ctl --set-ctrl=exposure_auto_priority=0     # don't slow FPS for brightness
v4l2-ctl --set-ctrl=exposure_absolute=200        # ~3–4 ms; adjust if too dark
v4l2-ctl --set-ctrl=power_line_frequency=2       # 60 Hz (US)
v4l2-ctl --set-ctrl=focus_auto=0                 # optional
v4l2-ctl --set-ctrl=white_balance_automatic=0    # optional

# --- Project Setup ---
echo "Changing directory to $PROJECT_DIR"
cd "$PROJECT_DIR" || { echo "ERROR: Failed to cd into project directory. Exiting."; exit 1; }

echo "Waiting for system to be ready..."
sleep 2

echo "Running git pull..."
git pull

echo "Activating virtual environment..."
source "$PROJECT_DIR/venv/bin/activate"

echo "Installing/updating dependencies..."
pip install -e .

# --- Launch App ---
echo "Launching ghoulfluids GUI..."
export DISPLAY=:0
ghoulfluids --fullscreen --debug \
            --segmenter='yolo' \
            --log-file="$LOG_FILE" \
            --seg-width=640 --seg-height=384

echo "Script finished at $(date)"
