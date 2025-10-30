#!/bin/bash

#udev rule to add to /etc/udev/rules.d/99-halloweencam.rules:
#ACTION=="add", SUBSYSTEM=="video4linux", ATTRS{idVendor}=="1532", ATTRS{idProduct}=="0e05", RUN+="/usr/local/bin/set_halloweencam_settings.sh %N"

#(venv) rpp@pop-os:~/halloweencam$ lsusb
#Bus 001 Device 006: ID 1532:0e05 Razer USA, Ltd Razer Kiyo Pro

DEVICE_NODE="/dev/$1"
LOG_FILE="/home/rpp/halloweencam_startup.log"

# Wait for the device to be fully ready
sleep 1

echo "--- udev rule triggered for $DEVICE_NODE (Manual Low-Light Mode) ---" >> $LOG_FILE

v4l2-ctl -d $DEVICE_NODE -c auto_exposure=3
v4l2-ctl -d $DEVICE_NODE -c exposure_dynamic_framerate=1
#v4l2-ctl -d $DEVICE_NODE -c focus_automatic_continuous=1
/usr/bin/v4l2-ctl -d $DEVICE_NODE -c focus_absolute=20 >> $LOG_FILE 2>&1
v4l2-ctl -d $DEVICE_NODE -c white_balance_automatic=1

# --- Set Manual Mode ---
## 1 = Manual Mode, 3 = Auto Mode
#/usr/bin/v4l2-ctl -d $DEVICE_NODE -c auto_exposure=1 >> $LOG_FILE 2>&1
## 0 = Disable dynamic framerate (Locks FPS)
#/usr/bin/v4l2-ctl -d $DEVICE_NODE -c exposure_dynamic_framerate=0 >> $LOG_FILE 2>&1

## --- Disable Auto Focus/White Balance (CRITICAL for YOLO) ---
## 0 = Auto-focus OFF
#/usr/bin/v4l2-ctl -d $DEVICE_NODE -c focus_automatic_continuous=0 >> $LOG_FILE 2>&1
## Set a fixed focus distance (TUNE THIS MANUALLY)
#/usr/bin/v4l2-ctl -d $DEVICE_NODE -c focus_absolute=20 >> $LOG_FILE 2>&1
## 0 = Auto-white-balance OFF
#/usr/bin/v4l2-ctl -d $DEVICE_NODE -c white_balance_automatic=0 >> $LOG_FILE 2>&1
## Set a fixed white balance (TUNE THIS)
#/usr/bin/v4l2-ctl -d $DEVICE_NODE -c white_balance_temperature=4000 >> $LOG_FILE 2>&1


## --- TUNING PARAMETERS ---
## These are the two settings you must balance.
##
## exposure_time_absolute: (Unit: 100µs)
## This controls your MAX FPS. 1 / (Value * 0.0001) = Max FPS
##   - Value 333 = 0.0333s = Max 30 FPS
##   - Value 500 = 0.0500s = Max 20 FPS
##   - Value 700 = 0.0700s = Max 14 FPS (Your old setting)
#/usr/bin/v4l2-ctl -d $DEVICE_NODE -c exposure_time_absolute=333 >> $LOG_FILE 2>&1

## gain: (Unit: camera-specific)
## This is digital brightness. Higher = brighter, but more noise.
## Tune this *after* setting your exposure.
#/usr/bin/v4l2-ctl -d $DEVICE_NODE -c gain=250 >> $LOG_FILE 2>&1


echo "--- udev manual settings applied ---" >> $LOG_FILE