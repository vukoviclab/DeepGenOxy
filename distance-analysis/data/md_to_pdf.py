#!/usr/bin/env python
"""
Render a Markdown file to a print-ready PDF.

Markdown -> HTML (mistune) -> PDF (headless Chromium). No pandoc/LaTeX or
weasyprint on this machine, and Chromium's print pipeline handles the wide
tables in these READMEs cleanly.

    usage: python md_to_pdf.py <file.md> [more.md ...]

Writes <file>.pdf beside each input.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

import mistune

CSS = """
@page { size: A4; margin: 17mm 15mm 16mm 15mm; }
* { box-sizing: border-box; }
body {
  font: 10pt/1.5 "Nimbus Sans", "Liberation Sans", Arial, Helvetica, sans-serif;
  color: #111; margin: 0; -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
h1 { font-size: 18pt; margin: 0 0 4pt; letter-spacing: -0.01em; }
h2 { font-size: 13pt; margin: 18pt 0 6pt; padding-bottom: 3pt;
     border-bottom: 1.2pt solid #333; break-after: avoid; }
h3 { font-size: 11pt; margin: 13pt 0 4pt; break-after: avoid; }
p  { margin: 5pt 0; orphans: 3; widows: 3; }
a  { color: #1a4f8a; text-decoration: none; }
hr { border: 0; border-top: 0.6pt solid #bbb; margin: 14pt 0; }
strong { font-weight: 700; }
code {
  font: 9pt/1.4 "DejaVu Sans Mono", Consolas, monospace;
  background: #f2f2f0; border: 0.4pt solid #ddd; border-radius: 2pt;
  padding: 0.5pt 2.5pt; white-space: nowrap;
}
pre {
  background: #f7f7f5; border: 0.5pt solid #ddd; border-radius: 3pt;
  padding: 6pt 8pt; overflow-x: auto; break-inside: avoid;
}
pre code { background: none; border: 0; padding: 0; white-space: pre; font-size: 8.5pt; }
table {
  border-collapse: collapse; width: 100%; margin: 7pt 0;
  font-size: 8.6pt; break-inside: avoid;
}
th, td { border: 0.5pt solid #c8c8c4; padding: 3pt 5pt; text-align: left;
         vertical-align: top; }
th { background: #eceff3; font-weight: 700; }
tr:nth-child(even) td { background: #fafafa; }
td code, th code { font-size: 8pt; }
ul, ol { padding-left: 15pt; margin: 5pt 0; }
li { margin: 2.5pt 0; }
blockquote { margin: 8pt 0; padding: 5pt 9pt; background: #f4f7fb;
             border-left: 2.5pt solid #4a7ab5; break-inside: avoid; }
blockquote p { margin: 2pt 0; }
"""


def find_chrome():
    for exe in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        p = shutil.which(exe)
        if p:
            return p
    raise SystemExit("no Chromium/Chrome binary found")


def staging_dir():
    """Where to put the HTML input and PDF output.

    Chromium here is a confined snap: it cannot read or write /tmp, $HOME/.cache
    or /media, so staging anywhere else fails with a bare 'Permission denied'.
    Its own snap area works, so prefer that and fall back to a normal temp dir
    for an unconfined Chrome.
    """
    snap = os.path.expanduser("~/snap/chromium/common")
    if os.path.isdir(snap):
        d = os.path.join(snap, ".md2pdf")
        os.makedirs(d, exist_ok=True)
        return d, False          # (path, delete_afterwards)
    return tempfile.mkdtemp(), True


def convert(md_path):
    md_path = os.path.abspath(md_path)
    out_pdf = os.path.splitext(md_path)[0] + ".pdf"
    text = open(md_path, encoding="utf-8").read()

    body = mistune.create_markdown(plugins=["table", "strikethrough"],
                                   escape=False)(text)
    title = os.path.basename(os.path.dirname(md_path)) or "README"
    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{title}</title><style>{CSS}</style></head>"
            f"<body>{body}</body></html>")

    stage, cleanup = staging_dir()
    src = os.path.join(stage, "doc.html")
    tmp_pdf = os.path.join(stage, "out.pdf")
    try:
        open(src, "w", encoding="utf-8").write(html)
        if os.path.exists(tmp_pdf):
            os.remove(tmp_pdf)
        cmd = [find_chrome(), "--headless=new", "--disable-gpu", "--no-sandbox",
               "--no-pdf-header-footer", "--virtual-time-budget=4000",
               f"--print-to-pdf={tmp_pdf}", f"file://{src}"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if not os.path.exists(tmp_pdf):
            cmd[1] = "--headless"        # older builds want the legacy spelling
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if not os.path.exists(tmp_pdf):
            raise SystemExit(f"chromium produced no PDF\n{r.stdout}\n{r.stderr}")
        shutil.copyfile(tmp_pdf, out_pdf)   # python is unconfined, chromium is not
    finally:
        if cleanup:
            shutil.rmtree(stage, ignore_errors=True)
        else:
            for f in (src, tmp_pdf):
                if os.path.exists(f):
                    os.remove(f)

    pages = len(re.findall(rb"/Type\s*/Page[^s]", open(out_pdf, "rb").read()))
    print(f"{os.path.relpath(md_path):<40} -> {os.path.basename(out_pdf)}  "
          f"({os.path.getsize(out_pdf) / 1024:.0f} KB, {pages} pages)")
    return out_pdf


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for f in sys.argv[1:]:
        convert(f)
