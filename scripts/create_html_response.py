import re
from pathlib import Path

text = Path(r"J:\OneDrive\WORK\RECHARCH TEAM\MRICovNetX-main\Paper\Response_to_Reviewers.md").read_text(encoding="utf-8")

html_lines = []
in_table = False
for line in text.splitlines():
    if line.startswith("# "):
        html_lines.append(f"<h1>{line[2:]}</h1>")
    elif line.startswith("## "):
        html_lines.append(f"<h2>{line[3:]}</h2>")
    elif line.startswith("### "):
        html_lines.append(f"<h3>{line[4:]}</h3>")
    elif line.startswith("> "):
        html_lines.append(f"<blockquote><p>{line[2:]}</p></blockquote>")
    elif "|" in line and not line.strip().startswith(">"):
        if not in_table:
            in_table = True
            html_lines.append("<table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse;width:100%;margin:16px 0;'>")
        parts = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(set(c).issubset({"-", ":", " "}) for c in parts):
            continue
        row = "<tr>" + "".join(f"<td>{p}</td>" for p in parts) + "</tr>"
        html_lines.append(row)
    else:
        if in_table:
            in_table = False
            html_lines.append("</table>")
        if line.strip():
            html_lines.append(f"<p>{line.strip()}</p>")

if in_table:
    html_lines.append("</table>")

body = "\n".join(html_lines)
body = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", body)

doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Response to Reviewers - MRICovNetX</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }}
blockquote {{ background: #f8f9fa; border-left: 4px solid #007bff; padding: 10px 15px; margin: 15px 0; color: #555; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 14px; }}
th, td {{ border: 1px solid #dee2e6; padding: 8px 12px; text-align: left; }}
tr:nth-child(even) {{ background: #fdfdfe; }}
h1, h2, h3 {{ color: #1a202c; border-bottom: 1px solid #eaecef; padding-bottom: 6px; }}
@media print {{ body {{ max-width: 100%; margin: 15mm; }} h2 {{ page-break-before: always; }} }}
</style>
</head>
<body>
{body}
</body>
</html>"""

Path(r"J:\OneDrive\WORK\RECHARCH TEAM\MRICovNetX-main\Paper\Response_to_Reviewers.html").write_text(doc, encoding="utf-8")
print("Response_to_Reviewers.html successfully generated!")
