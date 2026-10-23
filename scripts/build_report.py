"""Compila report/informe.tex en report/Proyecto_Final_[NombreGrupo].pdf.

Requiere una distribución LaTeX con pdflatex (TeX Live o MiKTeX); en Debian/Ubuntu:
    apt install texlive-latex-recommended texlive-latex-extra texlive-lang-spanish texlive-fonts-recommended

    python scripts/build_report.py [--group NombreGrupo]
"""
import argparse
import glob
import os
import shutil
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORT = os.path.join(ROOT, "report")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="Grupo2")
    a = ap.parse_args()
    if shutil.which("pdflatex") is None:
        raise SystemExit("No se encontró pdflatex; instala TeX Live o MiKTeX")
    # dos pasadas para el índice y las referencias cruzadas
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "informe.tex"],
                           cwd=REPORT, capture_output=True, text=True, errors="replace")
        if r.returncode != 0:
            print(r.stdout[-3000:])
            raise SystemExit("Falló la compilación; revisa report/informe.log")
    out = os.path.join(REPORT, f"Proyecto_Final_{a.group}.pdf")
    shutil.move(os.path.join(REPORT, "informe.pdf"), out)
    for ext in ("aux", "log", "out", "toc"):
        for f in glob.glob(os.path.join(REPORT, f"informe.{ext}")):
            os.remove(f)
    print("PDF:", out)


if __name__ == "__main__":
    main()
