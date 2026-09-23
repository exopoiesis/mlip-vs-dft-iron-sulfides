#!/usr/bin/env python3
"""Figure 3 — cross-mineral foundation-MLIP versus DFT benchmark (visualisation of Table 2).

Regenerated 2026-09-23. Changes against the previous version:
  * the pentlandite column is removed — that model did not reproduce the pentlandite
    structure and the whole section was withdrawn (main text §3.3);
  * the pyrite V_S coordinate is relabelled: it is an Fe-to-Fe hydrogen transfer in the
    sulfur-vacancy pocket, not a "V_S2 dimer" S–H hop (§2.4);
  * marcasite and greigite are included, so the figure now shows every coordinate of Table 2;
  * MLIP values are the self-consistent-endpoint ones except greigite, which is a single
    point on the DFT band (its self-consistent value was measured on the retracted path);
  * rendered at 600 dpi for the journal.

Values are taken from Table 2 of the manuscript. Output: fig3_cross_mineral_benchmark.{pdf,png,svg}
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent

# label, DFT eV, MACE eV (None = no defined barrier), CHGNet eV, note
ROWS = [
    ("Mackinawite\nV_Fe + S–H", 0.0429, 0.108, 0.183, ""),
    ("Pyrite\nV_S pocket, Fe→Fe", 0.0946, None, None, "endpoints merge"),
    ("Marcasite\nV_Fe + S–H", 0.208, 0.208, 0.318, ""),
    ("Greigite\nV_Fe + S–H (channel)", 0.236, 0.221, 0.254, "single point"),
    ("Pyrite\nV_Fe + S–H", 0.268, 0.256, 0.387, ""),
]

DFT_C, MACE_C, CHG_C = "#2c3e50", "#e67e22", "#3498db"


def main():
    labels = [r[0] for r in ROWS]
    dft = np.array([r[1] for r in ROWS])
    mace = np.array([np.nan if r[2] is None else r[2] for r in ROWS])
    chg = np.array([np.nan if r[3] is None else r[3] for r in ROWS])

    x = np.arange(len(ROWS))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9.2, 4.6))

    ax.bar(x - w, dft, w, label="DFT (PBE, U = 0)", color=DFT_C)
    ax.bar(x, mace, w, label="MACE-MP-0 (large)", color=MACE_C)
    ax.bar(x + w, chg, w, label="CHGNet v0.3.0", color=CHG_C)

    for i, (lab, d, m, c, note) in enumerate(ROWS):
        ax.text(i - w, d + 0.008, f"{d*1000:.0f}", ha="center", fontsize=8, color=DFT_C)
        if m is not None:
            ax.text(i, m + 0.008, f"{m*1000:.0f}", ha="center", fontsize=8, color=MACE_C)
        if c is not None:
            ax.text(i + w, c + 0.008, f"{c*1000:.0f}", ha="center", fontsize=8, color=CHG_C)
        if note:
            ax.annotate(note, xy=(i + w / 2, 0), xytext=(0, -34),
                        textcoords="offset points", ha="center", fontsize=7.5,
                        style="italic", color="#7f8c8d", annotation_clip=False)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Migration barrier $E_a$ (eV)", fontsize=11)
    ax.set_title("Foundation-MLIP versus DFT on the converged reaction coordinates",
                 fontsize=12)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.set_ylim(0, 0.44)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(0.01, 0.015,
             "MLIP bars use endpoints relaxed on the model's own surface, except greigite "
             "(single point on the DFT band).\nNo MLIP bar for the pyrite V_S pocket: both "
             "potentials merge the two Fe minima, so no barrier is defined.",
             fontsize=7.5, color="#555555")

    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"fig3_cross_mineral_benchmark.{ext}", dpi=600,
                    bbox_inches="tight")
    print("written:", ", ".join(f"fig3_cross_mineral_benchmark.{e}" for e in ("pdf", "png", "svg")))


if __name__ == "__main__":
    main()
