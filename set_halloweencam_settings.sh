#!/bin/bash

#(venv) rpp@pop-os:~/halloweencam$ lsusb
#Bus 001 Device 006: ID 1532:0e05 Razer USA, Ltd Razer Kiyo Pro

#udev rule to add to /etc/udev/rules.d/99-halloweencam.rules:
#ACTION=="add", SUBSYSTEM=="video4linux", ATTRS{idVendor}=="1532", ATTRS{idProduct}=="0e05", RUN+="/usr/local/bin/set_halloweencam_settings.sh %N"

# This script is run by udev when the camera is plugged in.
# It receives the device node (e.g., "video0") as the first argument.
DEVICE_NODE="/dev/$1"

# Log to your user's log file for debugging
LOG_FILE="/home/rpp/halloweencam_startup.log"

# Wait for the device to be fully ready
sleep 1

echo "--- udev rule triggered for $DEVICE_NODE ---" >> $LOG_FILE

/usr/bin/v4l2-ctl -d $DEVICE_NODE -c auto_exposure=1 >> $LOG_FILE 2>&1
/usr/bin/v4l2-ctl -d $DEVICE_NODE -c exposure_dynamic_framerate=0 >> $LOG_FILE 2>&1
/usr/bin/v4l2-ctl -d $DEVICE_NODE -c exposure_time_absolute=700 >> $LOG_FILE 2>&1
/usr/bin/v4l2-ctl -d $DEVICE_NODE -c gain=200 >> $LOG_FILE 2>&1
/usr/bin/v4l2-ctl -d $DEVICE_NODE -c focus_automatic_continuous=0 >> $LOG_FILE 2>&1
/usr/bin/v4l2-ctl -d $DEVICE_NODE -c white_balance_automatic=0 >> $LOG_FILE 2>&1

echo "--- udev settings applied successfully ---" >> $LOG_FILE

