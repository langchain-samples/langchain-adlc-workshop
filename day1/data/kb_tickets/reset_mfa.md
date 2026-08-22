# How to Reset Multi-Factor Authentication (MFA)

## When to use this guide
- User replaced their phone and lost MFA access
- MFA app not generating codes or receiving push notifications
- User is locked out of all systems due to MFA failure

## Steps

1. **Verify identity** — Confirm the user's identity via a secondary channel (manager approval, badge number, or in-person verification)
2. **Access the admin console** — Navigate to the identity provider admin console (Okta, Azure AD, etc.)
3. **Locate the user** — Search by email or employee ID
4. **Reset MFA** — Click "Reset MFA" or "Revoke all sessions and MFA methods"
5. **Notify user** — Tell the user to re-register MFA on their new device at the next login

## Important notes
- MFA reset is a **sensitive action** — it temporarily disables the second factor
- Always verify identity before resetting MFA to prevent social engineering attacks
- The user will need to re-enroll their MFA device on next login

## Related articles
- `account_lockout.md` — What to do when locked out of your account
- `password_policy.md` — Password requirements and reset process