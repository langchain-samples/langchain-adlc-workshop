# Setting Up Email on Devices

## Desktop (Windows/Mac)

### Automatic setup (recommended)
1. Open Outlook
2. Enter your Acme email address
3. Click "Connect" — Outlook will auto-configure using Autodiscover

### Manual setup (if auto-discovery fails)
- **Server:** outlook.office365.com
- **Port:** 443 (HTTPS)
- **Encryption:** TLS
- **Authentication:** OAuth2 / Modern Auth

## Mobile (iOS/Android)

### iOS (iPhone/iPad)
1. Open Settings → Mail → Accounts → Add Account
2. Select "Microsoft Exchange"
3. Enter your Acme email and password
4. If prompted, enter server: outlook.office365.com

### Android
1. Open Gmail or Outlook app
2. Add account → Exchange / Office 365
3. Enter your Acme email and password
4. Complete the sign-in flow

## BYOD (personal devices)
- Personal devices must enroll in Intune (mobile device management)
- Work email will be in a separate container — no personal data is accessed
- Unenrolling removes all work data from the device

## Troubleshooting
- **"Cannot connect to server"** — Check internet connection and server address
- **"Authentication failed"** — Verify password; try signing in via web first
- **"Sync issues"** — Remove and re-add the account; check storage space