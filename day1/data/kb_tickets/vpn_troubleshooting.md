# VPN Connection Issues and Fixes

## Common symptoms
- VPN connects but drops after a few minutes
- "Connection to gateway failed" error
- Slow performance over VPN
- Cannot access internal resources after connecting

## Troubleshooting steps

### 1. Check the VPN client version
- Ensure the VPN client is up to date (FortiClient, Cisco AnyConnect, etc.)
- Older versions may have compatibility issues with current server configuration

### 2. Check network connectivity
- Verify the user has a stable internet connection
- Try connecting from a different network (mobile hotspot) to isolate the issue
- Hotel and public Wi-Fi networks often block VPN traffic — try IPSec instead of SSL

### 3. Check firewall/proxy settings
- Corporate firewall may block VPN ports (443, 1194, etc.)
- Local firewall or antivirus may interfere with VPN client

### 4. Reconfigure the VPN profile
- Delete and re-create the VPN connection profile
- Verify server address, port, and protocol settings

### 5. Check split tunneling
- If only some traffic should route through VPN, verify split tunneling configuration
- Full tunnel may cause bandwidth issues on slow connections

## Escalation
If the issue persists after these steps, escalate to network engineering with:
- VPN client version and OS version
- Error messages and timestamps
- Network type (home, hotel, mobile hotspot)