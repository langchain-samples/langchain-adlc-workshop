# Printer Setup and Troubleshooting

## Adding a printer

### Windows
1. Open Settings → Bluetooth & devices → Printers & scanners
2. Click "Add device"
3. Select the printer from the list (or add by IP address)
4. Install drivers if prompted

### Mac
1. Open System Settings → Printers & Scanners
2. Click "+" to add a printer
3. Select from the list or enter IP address
4. Choose the appropriate driver

## Common issues

### Printer shows "Offline"
- Check if the printer is powered on
- Restart the print spooler service (Windows) or CUPS (Mac)
- Remove and re-add the printer

### Print jobs stuck in queue
- Clear the print queue
- Restart the printer
- Check for paper jams or low toner

### Poor print quality
- Run the printer's built-in cleaning cycle
- Check toner/ink levels
- Verify paper type settings match the loaded paper

### Network printer not found
- Verify the printer is on the same network
- Check the printer's IP address (print a network config page)
- Try adding by IP address instead of auto-discovery

## Acme print policy
- All print jobs are logged for audit purposes
- Secure print (PIN release) is required for sensitive documents
- Color printing is restricted to authorized users