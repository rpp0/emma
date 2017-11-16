#!/bin/zsh

echo "[+] Killing previous instances"
#sudo screen -S wpa -X quit # Old. Only kills screen but keeps apps alive
ps aux | grep wpa_supp_leak | grep -v grep | cut -d ' ' -f 6 | xargs -I {} sudo kill {}
echo "[+] Restarting wlp2s0"
sudo ifconfig wlp2s0 down
sudo ifconfig wlp2s0 up
echo "[+] Restarting wlp0s20u1"
sudo ifconfig wlp0s20u1 down
sudo ifconfig wlp0s20u1 up
echo "[+] Starting hostapd and wpa_supplicant in screen"
sudo screen -S wpa -c wpa_supp_screenrc
