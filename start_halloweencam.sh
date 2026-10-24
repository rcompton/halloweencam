#!/bin/bash
# Launch the Halloween cam at desktop login and keep it running.
#
# Started by ~/.config/autostart/halloweencam.desktop (see README).
# Boot does NOT git pull or pip install: an update that fails at boot
# should never take the display down. To update, run with --update.

PROJECT_DIR="$HOME/halloweencam"
LOG_FILE="$HOME/halloweencam_startup.log"
WIDTH=640
HEIGHT=480
CAMERA_MATCH="Razer"     # substring of the camera's /dev/v4l/by-id name
CAMERA_WAIT_SECS=120

# Keep the log from growing forever.
if [ -f "$LOG_FILE" ] && [ "$(stat -c %s "$LOG_FILE")" -gt 5000000 ]; then
    mv -f "$LOG_FILE" "$LOG_FILE.1"
fi
exec &>> "$LOG_FILE"

log() { echo "$(date '+%F %T') [start] $*"; }

# Only one copy at a time (autostart + manual run, etc.).
exec 9> "/tmp/halloweencam.$USER.lock"
if ! flock -n 9; then
    log "Already running; exiting."
    exit 0
fi

cd "$PROJECT_DIR" || { log "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }

if [ "$1" = "--update" ]; then
    log "Updating: git pull + pip install"
    git pull && "$PROJECT_DIR/venv/bin/pip" install -e . --quiet \
        || log "WARNING: update failed; running existing install"
fi

# The desktop session normally provides DISPLAY / WAYLAND_DISPLAY.
# Only fall back to :0 when launched without one.
[ -z "$DISPLAY" ] && [ -z "$WAYLAND_DISPLAY" ] && export DISPLAY=:0

# Resolve the camera by its stable USB name rather than assuming video0,
# which can move around when other video devices enumerate first.
find_camera() {
    local dev
    dev=$(ls /dev/v4l/by-id/*"${CAMERA_MATCH}"*video-index0 2>/dev/null | head -1)
    [ -n "$dev" ] && dev=$(readlink -f "$dev")
    [ -z "$dev" ] && [ -e /dev/video0 ] && dev=/dev/video0
    echo "$dev"
}

log "============================================="
log "Halloween cam starting"

while true; do
    CAM=""
    for _ in $(seq "$CAMERA_WAIT_SECS"); do
        CAM=$(find_camera)
        [ -n "$CAM" ] && break
        sleep 1
    done

    if [ -z "$CAM" ]; then
        log "No camera after ${CAMERA_WAIT_SECS}s (check USB / lsusb); retrying."
        continue
    fi

    INDEX="${CAM#/dev/video}"
    log "Using camera $CAM (index $INDEX)"
    v4l2-ctl -d "$CAM" --set-fmt-video=width=${WIDTH},height=${HEIGHT},pixelformat=MJPG \
        || log "WARNING: could not set MJPEG format"

    "$PROJECT_DIR/venv/bin/ghoulfluids" \
        --fullscreen --segmenter=yolo \
        --camera-id="$INDEX" \
        --seg-width=${WIDTH} --seg-height=${HEIGHT} \
        --log-file="$LOG_FILE"
    log "ghoulfluids exited with status $?; restarting in 5s"
    sleep 5
done
