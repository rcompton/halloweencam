#!/bin/bash
# One-time setup: launch the Halloween cam when the desktop session starts.
# Run from the repo:  ./install_autostart.sh
set -e
cd "$(dirname "$0")"
REPO="$(pwd)"

chmod +x start_halloweencam.sh

mkdir -p "$HOME/.config/autostart"
cat > "$HOME/.config/autostart/halloweencam.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Halloween Cam
Exec=$REPO/start_halloweencam.sh
X-GNOME-Autostart-enabled=true
NoDisplay=true
EOF
echo "Installed ~/.config/autostart/halloweencam.desktop"

# Camera settings rule (Razer Kiyo Pro). Needs sudo.
sudo install -m 755 set_halloweencam_settings.sh /usr/local/bin/set_halloweencam_settings.sh
echo 'ACTION=="add", SUBSYSTEM=="video4linux", ATTRS{idVendor}=="1532", ATTRS{idProduct}=="0e05", RUN+="/usr/local/bin/set_halloweencam_settings.sh %k"' \
    | sudo tee /etc/udev/rules.d/99-halloweencam.rules >/dev/null
sudo udevadm control --reload-rules
echo "Installed udev rule for the Kiyo Pro"

# Camera access needs the video group.
if ! id -nG | grep -qw video; then
    sudo usermod -aG video "$USER"
    echo "Added $USER to the video group (takes effect after logout/reboot)"
fi

echo "Done. Other autostart entries that also launch the cam should be removed;"
echo "check: grep -rl halloween ~/.config/autostart ~/.config/systemd/user /etc/systemd/system 2>/dev/null"
