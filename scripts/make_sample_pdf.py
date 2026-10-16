from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = Path('data/sample/acme_msa.pdf')
OUT.parent.mkdir(parents=True, exist_ok=True)
styles = getSampleStyleSheet()
H1, H2, BODY = styles['Heading1'], styles['Heading2'], styles['BodyText']
story = []

def section(title, paragraphs):
    story.append(Paragraph(title, H2))
    for text in paragraphs:
        story.append(Paragraph(text, BODY))
    story.append(Spacer(1, 8))

story.append(Paragraph('ACME CLOUD MASTER SERVICES AGREEMENT', H1))
story.append(Paragraph('Between Acme Cloud Ltd (the Supplier) and the customer named in the Order Form (the Customer).', BODY))
story.append(Spacer(1, 12))
section('1. Term and Renewal', [
    'The initial term of this Agreement is 24 months starting on the Effective Date of 1 March 2025.',
    'After the initial term, this Agreement renews automatically for successive renewal periods of 12 months unless a party gives notice of non-renewal in accordance with Section 7.',
])
story.append(PageBreak())
section('2. Fees and Payment', [
    'The Customer shall pay all undisputed invoices within 45 days of the invoice date.',
    'Late payments accrue interest at 1.5% per month until paid in full.',
    'The Supplier may suspend the Services if an undisputed invoice remains unpaid 30 days after a written reminder.',
])
story.append(PageBreak())
section('3. Service Levels', [
    'The Supplier shall make the Services available according to the monthly uptime commitments below. If actual uptime falls short, the Customer receives the service credit shown in the table.'
])
table = Table(
    [
        ['Monthly uptime', 'Service credit'],
        ['99.9% or higher', '0%'],
        ['99.0% to 99.89%', '10%'],
        ['95.0% to 98.99%', '25%'],
        ['Below 95.0%', '50%'],
    ],
    colWidths=[200, 200],
)
table.setStyle(TableStyle([
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
]))
story.append(table)
story.append(Spacer(1, 14))
section('4. Limitation of Liability', [
    'Total aggregate liability of the Supplier under this Agreement is limited to the fees paid by the Customer in the 12 months before the claim arose.',
    'Neither party is liable for indirect or consequential damages.',
])
story.append(PageBreak())
section('7. Termination and Non-Renewal', [
    'Either party may give notice of non-renewal, or terminate for convenience, by providing 90 days of written notice before the end of the then-current term.',
    'Either party may terminate immediately by written notice if the other party commits a material breach that is not cured within 30 days of notice.',
    'Termination does not relieve the Customer of fees accrued before the termination date.',
])
section('9. Governing Law', ['This Agreement is governed by the laws of England and Wales.'])
SimpleDocTemplate(str(OUT), pagesize=A4).build(story)
print('Created', OUT)
