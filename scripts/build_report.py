"""Convierte report/informe.md en report/Proyecto_Final_[NombreGrupo].pdf.

Markdown -> HTML (con estilos de impresión A4) -> PDF con Chromium sin interfaz.
    python scripts/build_report.py [--group NombreGrupo] [--chrome /ruta/a/chrome]
"""
import argparse
import os
import shutil
import subprocess
import tempfile

import markdown

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSS = """
@page { size: A4; margin: 15mm 15mm 15mm 15mm; }
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 9pt; line-height: 1.32; color: #111; }
h1 { font-size: 15pt; margin: 0 0 2pt 0; }
h2 { break-after: avoid; font-size: 11.5pt; margin: 9pt 0 3pt 0; border-bottom: 1px solid #bbb; padding-bottom: 1pt; }
h3 { break-after: avoid; font-size: 10pt; margin: 6pt 0 2pt 0; }
p { margin: 2pt 0 4pt 0; text-align: justify; }
ul, ol { margin: 2pt 0 4pt 0; padding-left: 15pt; }
li { margin: 0 0 1pt 0; }
table { border-collapse: collapse; width: 100%; margin: 3pt 0 6pt 0; font-size: 7.8pt; }
th, td { border: 1px solid #c8c8c8; padding: 1.5pt 3pt; text-align: left; vertical-align: top; }
th { background: #f0f0ee; }
img { width: 100%; margin: 2pt 0; }
figure { margin: 3pt 0; page-break-inside: avoid; }
figcaption { font-size: 7.8pt; color: #444; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8pt; }
pre { font-size: 7.5pt; background: #f6f6f4; padding: 4pt; white-space: pre-wrap; }
.meta { color: #444; font-size: 8.5pt; margin-bottom: 6pt; }
.refs p { font-size: 7.8pt; margin: 0 0 2pt 0; text-align: left; }
"""


def find_chrome(explicit=None):
    cands = [explicit, shutil.which("chromium"), shutil.which("google-chrome"), shutil.which("chrome"),
             "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"]
    for c in cands:
        if c and os.path.exists(c):
            return c
    raise SystemExit("No se encontró Chromium/Chrome; pasa --chrome")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="GrupoEDA")
    ap.add_argument("--chrome", default=None)
    a = ap.parse_args()
    src = os.path.join(ROOT, "report", "informe.md")
    with open(src, encoding="utf-8") as f:
        body = markdown.markdown(f.read(), extensions=["tables", "fenced_code", "attr_list", "md_in_html"])
    html = f"<!doctype html><html lang='es'><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    html_path = os.path.join(ROOT, "report", "informe.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    out = os.path.join(ROOT, "report", f"Proyecto_Final_{a.group}.pdf")
    with tempfile.TemporaryDirectory() as prof:
        subprocess.run([find_chrome(a.chrome), "--headless=new", "--no-sandbox", "--disable-gpu",
                        f"--user-data-dir={prof}", "--no-pdf-header-footer", f"--print-to-pdf={out}",
                        "file://" + html_path], check=True, capture_output=True)
    print("PDF:", out)


if __name__ == "__main__":
    main()
