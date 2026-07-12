# scripts/reporting/convert_to_ieeetran.py
"""
Convert the Springer (sn-jnl) manuscript into an IEEEtran journal manuscript
for submission to IEEE Transactions on Quantum Engineering.

The body content is transplanted verbatim; only structural elements change
(preamble, title/author block, abstract/keywords environments, appendix
markup, backmatter headings, wide floats promoted to starred versions).
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

SRC = Path("manuscript/sn-article.tex")
DST_DIR = Path("manuscript_tqe")
DST = DST_DIR / "tqe-article.tex"

PREAMBLE = r"""\documentclass[journal]{IEEEtran}
\usepackage{graphicx}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{url}
\usepackage[hidelinks]{hyperref}

\begin{document}

\title{Kernel Geometry Governs Robustness under Distribution Shift:\\
A Controlled Multi-Dataset Study of Quantum and Classical Kernels}

\author{Roberto~Fern\'andez-Barrios,
        Iker~Pastor-L\'opez,
        Asier~Gonz\'alez-Santocildes,
        and~Pablo~Garc\'ia~Bringas%
\thanks{The authors are with the Faculty of Engineering, University of Deusto,
48007 Bilbao, Spain (e-mail: roberto.fernandez.b@deusto.es;
iker.pastor@deusto.es; gonzalez.asier@deusto.es;
pablo.garcia.bringas@deusto.es).}}

\markboth{IEEE Transactions on Quantum Engineering}%
{Fern\'andez-Barrios \MakeLowercase{\textit{et al.}}: Kernel Geometry Governs Robustness under Distribution Shift}

\maketitle
"""


def extract_braced(src: str, macro: str) -> str:
    """Return the balanced-brace argument of \\macro{...}."""
    i = src.index("\\" + macro + "{") + len(macro) + 2
    depth = 1
    j = i
    while depth:
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
        j += 1
    return src[i:j - 1]


def main() -> None:
    src = SRC.read_text(encoding="utf-8")

    abstract = extract_braced(src, "abstract")
    keywords = extract_braced(src, "keywords")

    body_start = src.index("\\section{Introduction}")
    body_end = src.index("\\clearpage\n\\bibliography")
    body = src[body_start:body_end]

    # Appendix markup: appendices environment -> IEEEtran \appendices
    body = body.replace("\\begin{appendices}\n", "\\appendices\n")
    body = body.replace("\\end{appendices}\n", "")

    # Backmatter headings
    body = body.replace("\\clearpage\n\\backmatter\n", "")
    body = body.replace("\\bmhead{Statements and Declarations}\n", "")
    body = re.sub(r"\\bmhead\{([^}]*)\}", r"\\section*{\1}", body)

    # Promote wide floats to starred versions (all tables except the
    # single-column dose-response table; figures are already correct).
    parts = re.split(r"(\\begin\{table\}\[[!a-z]*\].*?\\end\{table\})", body, flags=re.S)
    out_parts = []
    for p in parts:
        if p.startswith("\\begin{table}") and "tab:dose_response" not in p:
            p = re.sub(r"\\begin\{table\}\[[!a-z]*\]", "\\\\begin{table*}[!t]", p)
            p = p.replace("\\end{table}", "\\end{table*}")
        out_parts.append(p)
    body = "".join(out_parts)

    tail = r"""
\bibliographystyle{IEEEtran}
\bibliography{sn-bibliography}

\end{document}
"""

    doc = (
        PREAMBLE
        + "\n\\begin{abstract}\n" + abstract + "\n\\end{abstract}\n"
        + "\n\\begin{IEEEkeywords}\n" + keywords.replace(";", ",") + "\n\\end{IEEEkeywords}\n\n"
        + body
        + tail
    )

    DST_DIR.mkdir(exist_ok=True)
    DST.write_text(doc, encoding="utf-8")
    for f in ["sn-bibliography.bib", "fig_v2_arc.pdf", "fig_v2_dose_response.pdf", "fig_v2_mechanism_law.pdf"]:
        shutil.copy(Path("manuscript") / f, DST_DIR / f)
    print(f"[✓] Wrote {DST} ({len(doc)} chars) + bib + figures")


if __name__ == "__main__":
    main()
