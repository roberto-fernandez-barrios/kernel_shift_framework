# scripts/reporting/make_readme_assets.py
"""Export PNG copies of the key manuscript figures for the README."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

import matplotlib.pyplot as plt

import scripts.reporting.make_v3_figures as figs

OUT = Path("docs/assets")
OUT.mkdir(parents=True, exist_ok=True)
figs.OUT = OUT

_orig = plt.Figure.savefig


def _png(self, path, **kw):
    kw["dpi"] = 160
    return _orig(self, str(path).replace(".pdf", ".png"), **kw)


plt.Figure.savefig = _png
figs.fig_protocol()
figs.fig_reversal()
figs.fig_continuum()
print("[ok] README assets in docs/assets/")
