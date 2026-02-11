import os
import markdown
from xhtml2pdf import pisa

# Define input and output paths
input_md_path = "PancreScan_Training_Report.md"
output_pdf_path = "PancreScan_Training_Report.pdf"

# Read the markdown file
with open(input_md_path, "r", encoding="utf-8") as handle:
    text = handle.read()

# Convert markdown to HTML
html_text = markdown.markdown(text, extensions=["tables"])

# Add basic styling
html_content = f"""
<html>
<head>
<style>
    body {{
        font-family: 'Helvetica', 'Arial', sans-serif;
        font-size: 11pt;
        line-height: 1.6;
        color: #2c3e50;
    }}
    h1 {{
        color: #2c3e50;
        border-bottom: 2px solid #2c3e50;
        padding-bottom: 10px;
        font-size: 22pt;
        margin-top: 0;
    }}
    h2 {{
        color: #34495e;
        margin-top: 22px;
        font-size: 16pt;
        border-bottom: 1px solid #eee;
        padding-bottom: 5px;
    }}
    table {{
        border-collapse: collapse;
        width: 100%;
        margin: 16px 0;
        font-size: 10pt;
    }}
    th, td {{
        border: 1px solid #BDC3C7;
        padding: 8px;
        text-align: left;
    }}
    th {{
        background-color: #ECF0F1;
        color: #2c3e50;
        font-weight: bold;
    }}
    tr:nth-child(even) {{
        background-color: #f9f9f9;
    }}
    code {{
        background-color: #f4f4f4;
        padding: 2px 4px;
        border-radius: 3px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 10pt;
        color: #c0392b;
    }}
</style>
</head>
<body>
{html_text}
</body>
</html>
"""

# Convert HTML to PDF
with open(output_pdf_path, "wb") as pdf_file:
    status = pisa.CreatePDF(html_content, dest=pdf_file)

if status.err:
    print(f"Error creating PDF: {status.err}")
else:
    print(f"Successfully created: {os.path.abspath(output_pdf_path)}")
