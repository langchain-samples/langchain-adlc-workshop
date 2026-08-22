# IT Setup for Remote Work

## Before you start

### Required equipment
- Acme-issued laptop with current security patches
- Stable internet connection (minimum 25 Mbps download)
- Quiet workspace with privacy for calls

### Required software
- VPN client (FortiClient) — pre-installed on Acme laptops
- Microsoft 365 apps (Outlook, Teams, Word, Excel, PowerPoint)
- MFA app on your phone (Microsoft Authenticator)

## Getting connected

1. **Connect to the internet** — home Wi-Fi, Ethernet, or mobile hotspot
2. **Connect to VPN** — Open FortiClient and connect to the Acme gateway
3. **Verify access** — Open Outlook and confirm email syncs; open an internal site

## Home network recommendations
- Use WPA2 or WPA3 encryption on your home Wi-Fi
- Change default router admin password
- Keep router firmware up to date
- Use a wired Ethernet connection when possible for stability

## Security reminders
- Always use VPN when accessing internal resources
- Lock your screen when stepping away (Windows+L or Cmd+Ctrl+Q)
- Don't use public Wi-Fi without VPN
- Keep your laptop physically secure — don't leave it in a car or unattended

## Troubleshooting
- VPN won't connect → see `vpn_troubleshooting.md`
- Email not syncing → see `email_configuration.md`
- Can't access internal sites → verify VPN is connected and try clearing browser cache