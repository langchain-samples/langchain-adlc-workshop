#!/usr/bin/env python3
"""Generate synthetic PDF vendor documents for the workshop.

Creates realistic-looking PDF documents (vendor capability statements) so participants can see
how an agent handles document parsing — not just structured JSON or Markdown.

Not a workshop step: the PDFs it produces are committed, so participants never run this. It exists
for whoever edits `vendors.json` — the documents are derived entirely from it — and as the record of
how the synthetic documents were produced.

Run from the repo root:
    uv run python day1/data/generate_pdfs.py
"""

from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent


def generate_vendor_pdf(vendor_id: str, output_path: Path) -> None:
    """Generate a synthetic vendor capability statement PDF using reportlab."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas

    vendors = json.loads((DATA_DIR / "vendors.json").read_text())
    vendor = vendors.get(vendor_id)
    if not vendor:
        print(f"Vendor {vendor_id} not found")
        return

    # invariant=1 strips the creation timestamp and document ID that reportlab embeds by default.
    # These PDFs are committed, so without it every regeneration produces a 13-file binary diff with
    # identical content — noise in review, and a moving hash for anything that fingerprints the data.
    c = canvas.Canvas(str(output_path), pagesize=letter, invariant=1)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(1 * inch, height - 1 * inch, "VENDOR CAPABILITY STATEMENT")

    # Vendor info
    c.setFont("Helvetica", 12)
    y = height - 1.5 * inch
    lines = [
        f"Company: {vendor['name']}",
        f"Vendor ID: {vendor['vendor_id']}",
        f"Category: {vendor['category']}",
        f"Country: {vendor['country']}",
        f"Size: {vendor['size']}",
        f"Risk Level: {vendor['risk_level']}",
        "",
        "CERTIFICATIONS",
    ]
    for cert in vendor.get("certifications", []):
        lines.append(f"  • {cert}")

    lines.append("")
    lines.append("COMPLIANCE FLAGS")
    for flag in vendor.get("compliance_flags", []):
        lines.append(f"  • {flag}")

    lines.append("")
    lines.append("CONTRACT HISTORY")
    for contract in vendor.get("contract_history", []):
        lines.append(
            f"  • {contract['buyer']} ({contract['year']}): "
            f"EUR {contract['value_eur']:,} — {contract['scope']}"
        )

    lines.append("")
    lines.append("HIGHLIGHTS")
    lines.append(f"  {vendor.get('highlights', 'N/A')}")
    lines.append("")
    lines.append("—")
    lines.append("This is a SYNTHETIC document created for the workshop.")
    lines.append("It does not represent any real organization.")

    for line in lines:
        c.drawString(1 * inch, y, line)
        y -= 0.25 * inch
        if y < 1 * inch:
            c.showPage()
            c.setFont("Helvetica", 12)
            y = height - 1 * inch

    c.save()
    print(f"Generated {output_path.name}")


def generate_all_pdfs() -> None:
    """Generate PDFs for all vendors."""
    pdf_dir = DATA_DIR / "pdfs"
    pdf_dir.mkdir(exist_ok=True)

    vendors = json.loads((DATA_DIR / "vendors.json").read_text())
    for vid in vendors:
        if vid.startswith("_"):
            continue
        vendor = vendors[vid]
        safe_name = vendor["name"].lower().replace(" ", "_").replace("&", "and")
        output_path = pdf_dir / f"{safe_name}_capability_statement.pdf"
        generate_vendor_pdf(vid, output_path)

    print(f"\nGenerated {len(list(pdf_dir.glob('*.pdf')))} PDFs in {pdf_dir}")


if __name__ == "__main__":
    generate_all_pdfs()
