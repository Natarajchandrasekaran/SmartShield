"""
SmartShield — Technical Report to PDF Converter
Converts docs/Technical_Report.md into a high-quality PDF report using headless browser printing.
"""

import os
import subprocess
import markdown

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MD_PATH = os.path.join(PROJECT_ROOT, "docs", "Technical_Report.md")
HTML_PATH = os.path.join(PROJECT_ROOT, "docs", "Technical_Report.html")
PDF_PATH = os.path.join(PROJECT_ROOT, "docs", "SmartShield_Technical_Report.pdf")

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not os.path.exists(EDGE_PATH):
    EDGE_PATH = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"


def convert():
    print(f"Reading markdown from: {MD_PATH}")
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Convert to HTML
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc"]
    )

    styled_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SmartShield — Technical Proposal Report</title>
<style>
    @page {{
        size: A4;
        margin: 20mm 15mm 20mm 15mm;
        @bottom-right {{
            content: counter(page);
        }}
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 10.5pt;
        line-height: 1.6;
        color: #24292e;
        max-width: 900px;
        margin: auto;
        padding: 10px;
    }}
    h1 {{
        font-size: 22pt;
        border-bottom: 2px solid #0366d6;
        padding-bottom: 8px;
        color: #0366d6;
        margin-top: 20px;
    }}
    h2 {{
        font-size: 15pt;
        border-bottom: 1px solid #eaecef;
        padding-bottom: 6px;
        color: #1a1f2c;
        margin-top: 24px;
        page-break-after: avoid;
    }}
    h3 {{
        font-size: 12pt;
        color: #2b3137;
        margin-top: 18px;
        page-break-after: avoid;
    }}
    table {{
        border-collapse: collapse;
        width: 100%;
        margin: 15px 0;
        font-size: 9.5pt;
    }}
    th, td {{
        border: 1px solid #dfe2e5;
        padding: 7px 12px;
        text-align: left;
    }}
    th {{
        background-color: #f6f8fa;
        font-weight: 600;
    }}
    tr:nth-child(even) {{
        background-color: #fafbfc;
    }}
    code {{
        background-color: #f0f3f6;
        padding: 2px 5px;
        border-radius: 3px;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 9pt;
    }}
    pre {{
        background-color: #f6f8fa;
        border: 1px solid #e1e4e8;
        border-radius: 6px;
        padding: 12px;
        overflow: auto;
        font-size: 9pt;
        line-height: 1.45;
    }}
    pre code {{
        background-color: transparent;
        padding: 0;
    }}
    blockquote {{
        border-left: 4px solid #0366d6;
        margin: 0;
        padding-left: 15px;
        color: #586069;
    }}
    hr {{
        border: 0;
        border-top: 1px solid #e1e4e8;
        margin: 20px 0;
    }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(styled_html)
    print(f"HTML generated at: {HTML_PATH}")

    # Print to PDF using Headless Edge
    cmd = [
        EDGE_PATH,
        "--headless",
        "--disable-gpu",
        f"--print-to-pdf={PDF_PATH}",
        HTML_PATH
    ]
    print("Generating PDF...")
    subprocess.run(cmd, check=True)

    if os.path.exists(PDF_PATH):
        size_kb = os.path.getsize(PDF_PATH) / 1024
        print(f"PDF generated successfully at: {PDF_PATH} ({size_kb:.1f} KB)")
    else:
        print("Failed to generate PDF.")


if __name__ == "__main__":
    convert()
