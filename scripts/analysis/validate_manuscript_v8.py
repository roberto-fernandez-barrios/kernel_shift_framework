"""Fail-fast npj Quantum Information format gates for the v0.8 manuscript."""
from __future__ import annotations

import re
from pathlib import Path


MAIN = Path("manuscript/sn-article.tex")
SUPPLEMENT = Path("manuscript/supplementary.tex")
GENERATED_TABLE_ROWS = Path("manuscript/generated")


def uncommented(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("%")
    )


def tex_words(text: str) -> list[str]:
    text = re.sub(r"\\(?:texttt|emph|textbf|mathrm)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[A-Za-z@]+(?:\[[^\]]*\])?", " ", text)
    text = text.replace("\\%", " percent ")
    text = re.sub(r"[{}$\\]", " ", text)
    return re.findall(
        r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)*",
        text,
    )


def extract_one(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"missing {label}")
    return match.group(1).strip()


def validate_main(text: str) -> None:
    active = uncommented(text)
    documentclass = extract_one(
        r"\\documentclass\[([^\]]+)\]\{sn-jnl\}",
        active,
        "active document class",
    )
    if "sn-nature" not in documentclass:
        raise ValueError("active document class must use sn-nature")

    title = extract_one(
        r"\\title(?:\[[^\]]*\])?\{([^{}]+)\}",
        active,
        "article title",
    )
    title_words = tex_words(title)
    if len(title_words) > 15:
        raise ValueError(f"title has {len(title_words)} words, limit is 15")
    if re.search(r"[:;,.!?]", title):
        raise ValueError("article title contains prohibited punctuation")
    pdf_title = extract_one(
        r"pdftitle=\{([^{}]+)\}",
        active,
        "PDF metadata title",
    )
    if pdf_title != title:
        raise ValueError("PDF metadata title differs from article title")

    abstract = extract_one(
        r"\\abstract\{(.*?)\}\s*\\keywords",
        active,
        "unstructured abstract",
    )
    abstract_words = tex_words(abstract)
    if len(abstract_words) > 150:
        raise ValueError(
            f"abstract has {len(abstract_words)} words, limit is 150"
        )
    if re.search(r"\\(?:cite|section|subsection)\b", abstract):
        raise ValueError("abstract contains a citation or subheading command")

    display_items = len(
        re.findall(r"\\begin\{(?:figure\*?|table\*?)\}", active)
    )
    if display_items > 10:
        raise ValueError(
            f"main manuscript has {display_items} display items, limit is 10"
        )

    for prohibited in ("Conclusion", "Conclusions", "Limitations"):
        if re.search(
            rf"\\(?:section|subsection)\*?\{{{prohibited}\}}",
            active,
            flags=re.IGNORECASE,
        ):
            raise ValueError(f"prohibited separate {prohibited} section")

    discussion_start = active.find(r"\section{Discussion}")
    methods_start = active.find(r"\section{Methods}")
    data_start = active.find(r"\bmhead{Data Availability}")
    if not (0 <= discussion_start < methods_start < data_start):
        raise ValueError("Discussion, Methods, and Data Availability order is invalid")
    discussion = active[discussion_start:methods_start]
    if re.search(r"\\subsection\*?\{", discussion):
        raise ValueError("Discussion contains a prohibited subsection")
    if r"\subsection{Use of generative AI}" not in active:
        raise ValueError("generative-AI disclosure is missing from Methods")
    if r"\bmhead{Code Availability}" not in active:
        raise ValueError("Code Availability statement is missing")
    if r"\bmhead{Ethics approval and consent to participate}" not in active:
        raise ValueError("public-data ethics statement is missing")

    print(
        "[ok] npj main format: "
        f"title={len(title_words)} words, abstract={len(abstract_words)} words, "
        f"display_items={display_items}"
    )


def validate_supplement(text: str) -> None:
    active = uncommented(text)
    if re.search(r"supplementary\s+methods", active, flags=re.IGNORECASE):
        raise ValueError("Supplementary Information contains Supplementary Methods")
    if re.search(
        r"\\section\*?\{[^{}]*methods[^{}]*\}",
        active,
        flags=re.IGNORECASE,
    ):
        raise ValueError("Supplementary Information contains a Methods section")
    if r"\section*{Supplementary Results:" not in active:
        raise ValueError("Supplementary Results structure is missing")
    generated_files = sorted(GENERATED_TABLE_ROWS.glob("v8_*_rows.tex"))
    if not generated_files:
        raise ValueError("generated v0.8 table rows are missing")
    for path in generated_files:
        for row in path.read_text(encoding="utf-8").splitlines():
            if not row.strip() or row.strip() in {r"\addlinespace", r"\midrule"}:
                continue
            if row not in active:
                raise ValueError(
                    f"{path}: generated row is not represented verbatim in "
                    "Supplementary Information"
                )
    print("[ok] Supplementary Information contains results/diagnostics only")
    print(
        "[ok] Supplementary v0.8 tables match "
        f"{len(generated_files)} generated row files"
    )


def main() -> None:
    if not MAIN.is_file() or not SUPPLEMENT.is_file():
        raise FileNotFoundError("main or Supplementary LaTeX source is missing")
    validate_main(MAIN.read_text(encoding="utf-8"))
    validate_supplement(SUPPLEMENT.read_text(encoding="utf-8"))
    print("[ok] all npj manuscript-format gates passed")


if __name__ == "__main__":
    main()
