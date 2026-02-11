import markdown
from xhtml2pdf import pisa
import os

# Define input and output paths
input_md_path = 'PancreScan_Project_Report_v2.md'
output_pdf_path = 'PancreScan_Project_Report_v2.pdf'

# Read the markdown file
with open(input_md_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Convert markdown to html
html_text = markdown.markdown(text, extensions=['tables'])

# Add some CSS for styling
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
        color: #e74c3c;
        border-bottom: 2px solid #e74c3c;
        padding-bottom: 10px;
        font-size: 24pt;
        margin-top: 0;
    }}
    h2 {{
        color: #2980b9;
        margin-top: 25px;
        font-size: 18pt;
        border-bottom: 1px solid #eee;
        padding-bottom: 5px;
    }}
    h3 {{
        color: #34495e;
        margin-top: 20px;
        font-size: 14pt;
        font-weight: bold;
    }}
    table {{
        border-collapse: collapse;
        width: 100%;
        margin: 20px 0;
        font-size: 10pt;
    }}
    th, td {{
        border: 1px solid #BDC3C7;
        padding: 12px;
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
        padding: 2px 5px;
        border-radius: 3px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 10pt;
        color: #c0392b;
    }}
    .mermaid {{
        background-color: #f8f9fa;
        border: 1px dashed #ccc;
        padding: 10px;
        font-family: monospace;
        white-space: pre-wrap;
    }}
    blockquote {{
        border-left: 5px solid #3498db;
        margin: 15px 0;
        padding-left: 15px;
        color: #555;
        font-style: italic;
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
    pisa_status = pisa.CreatePDF(html_content, dest=pdf_file)

if pisa_status.err:
    print(f"Error creating PDF: {pisa_status.err}")
else:
    print(f"Successfully created: {os.path.abspath(output_pdf_path)}")
