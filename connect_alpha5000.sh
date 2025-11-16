#!/bin/bash
# connects to Sony A500 Wifi

CAMERA_SSID="$1"
CAMERA_PASSWORD="$2"
CAMERA_IP="192.168.122.1"

if [ $# -ne 2 ]; then
    echo "Usage $0 <SSID> <PASSWORD>"
    exit 1
fi

test_camera_connection() {
    curl -s -m 3 -H "Content-Type: application/json" \
         -d '{"method":"getVersions","params":[],"id":1,"version":"1.0"}' \
         "http://$CAMERA_IP:$CAMERA_PORT/sony/camera" > /dev/null 2>&1
    return $?
}

if test_camera_connection; then
    echo "Already connected to wifi, skipping."
    exit 0
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
    WIFI_INTERFACE=$(networksetup -listallhardwareports | grep -A 1 "Wi-Fi" | grep "Device:" | awk '{print $2}')
    # Get current network
    CURRENT_NETWORK=$(networksetup -getairportnetwork "$WIFI_INTERFACE" | cut -d: -f2 | xargs)
    
    if [ -n "$CURRENT_NETWORK" ] && [ "$CURRENT_NETWORK" != "You are not associated with an AirPort network." ]; then
        echo "Disconnecting from: $CURRENT_NETWORK"
        # Use airport utility to disconnect (keeps WiFi on)
        sudo /System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport -z
        sleep 1
    fi
    networksetup -setairportnetwork "$WIFI_INTERFACE" "$CAMERA_SSID" "$CAMERA_PASSWORD"
    echo "Connected to $CAMERA_SSID"
else
    nmcli dev wifi connect "$CAMERA_SSID" password "$CAMERA_PASSWORD"
    echo "Connected to $CAMERA_SSID"
fi

echo "Camera ready. Press Enter to disconnect"
