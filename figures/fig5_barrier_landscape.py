"""
Figure 5 — Fe-S series DFT barrier landscape (log scale, meV).

Vacancy-anchored proton-migration barriers, sorted by ascending E_a.
Horizontal threshold at 25*k_BT@298K ~= 642 meV: the entire series lies below it.

Data (all PBE, U=0; provenance in main text Table 2 / Table 3):
  Mackinawite V_Fe+S-H         42.9 meV   (Table 2)
  Pyrite V_S pocket, Fe-H swing 94.6 meV  (Table 2; a different channel, plotted for scale)
  Marcasite V_Fe+S-H          208.2 meV   (Table 2)
  Greigite channel edge       236.0 meV   (Table 3, 235.97)
  Pyrite V_Fe+S-H             268.0 meV   (Table 2, 0.268 eV)
  Greigite cation edge        566.8 meV   (Table 3, 566.83)

MLIP overlays on the pyrite V_Fe coordinate only, endpoints relaxed on each
model's own surface (the self-consistent convention of Table 2):
  MACE-MP-0 (large) 256 meV, CHGNet-v0.3.0 387 meV.

History: this figure previously carried the greigite barrier as a single
1860 meV bar with a "kinetically forbidden" zone. That band was retracted --
its midpoint sat on the vacant Fe site, a special position where the force
vanishes by symmetry -- and replaced by the two symmetry-distinct edge classes
computed in 2026-09. Nothing in the series is kinetically forbidden.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# --- Verified DFT data, ascending ---
LABELS = [
    "Mackinawite\n$V_{Fe}$+S–H",
    "Pyrite\n$V_S$ pocket",
    "Marcasite\n$V_{Fe}$+S–H",
    "Greigite\nchannel edge",
    "Pyrite\n$V_{Fe}$+S–H",
    "Greigite\ncation edge",
]
EA_DFT = np.array([42.9, 94.6, 208.2, 236.0, 268.0, 566.8])  # meV
IS_GREIGITE = [False, False, False, True, False, True]

# kinetic accessibility threshold: 25 * k_B * 298 K
K_BT_298 = 25.69             # meV
THRESHOLD = 25 * K_BT_298    # ~642 meV

C_SERIES = "#4393c3"
C_GREIGITE = "#7b3294"
COLORS_BAR = [C_GREIGITE if g else C_SERIES for g in IS_GREIGITE]

# MLIP overlays for pyrite V_Fe (index 4), self-consistent endpoints
MACE_PYR_VFE = 256.0    # meV
CHGNET_PYR_VFE = 387.0  # meV

# the analytical point of the panel
GREIGITE_SPREAD = EA_DFT[5] - EA_DFT[3]           # 330.8 meV
SERIES_SPREAD = EA_DFT[4] - EA_DFT[0]             # 225.1 meV


def make_figure(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.0, 5.5))

    x = np.arange(len(LABELS))
    width = 0.55

    bars = ax.bar(
        x, EA_DFT, width,
        color=COLORS_BAR, edgecolor="black", linewidth=0.6, zorder=3,
        label="DFT QE PBE (this work, $U=0$)",
    )

    for bar, val in zip(bars, EA_DFT):
        ax.text(
            bar.get_x() + bar.get_width() / 2, val * 1.10,
            f"{val:.1f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="black", zorder=7,
        )

    # --- the two greigite edge classes belong to ONE vacancy octahedron ---
    ax.plot([2.72, 5.28], [EA_DFT[3], EA_DFT[3]], ":", color=C_GREIGITE,
            lw=1.0, alpha=0.8, zorder=2)
    ax.plot([2.72, 5.28], [EA_DFT[5], EA_DFT[5]], ":", color=C_GREIGITE,
            lw=1.0, alpha=0.8, zorder=2)
    ax.annotate(
        "", xy=(5.48, EA_DFT[5]), xytext=(5.48, EA_DFT[3]),
        arrowprops=dict(arrowstyle="<->", color=C_GREIGITE, lw=1.3),
    )
    ax.text(
        5.60, np.sqrt(EA_DFT[3] * EA_DFT[5]),
        f"one octahedron,\ntwo edge classes:\n{GREIGITE_SPREAD:.0f} meV apart\n"
        f"(vs {SERIES_SPREAD:.0f} meV across\nthe other minerals)",
        ha="left", va="center", fontsize=8, color=C_GREIGITE, fontweight="bold",
    )

    # --- MLIP overlay points for pyrite V_Fe (index 4) ---
    pyr_x = x[4]
    ax.scatter(
        [pyr_x - 0.14, pyr_x + 0.14],
        [MACE_PYR_VFE, CHGNET_PYR_VFE],
        s=70, marker="D", color=["#d95f02", "#1b9e77"],
        edgecolors="black", linewidths=0.6, zorder=6,
        label="MACE-MP-0 (large) / CHGNet on pyrite $V_{Fe}$, self-consistent",
    )
    ax.annotate(
        f"MACE {MACE_PYR_VFE:.0f}", xy=(pyr_x - 0.14, MACE_PYR_VFE),
        xytext=(pyr_x - 0.95, MACE_PYR_VFE * 0.58),
        fontsize=7.5, color="#d95f02",
        arrowprops=dict(arrowstyle="->", color="#d95f02", lw=0.7),
    )
    ax.annotate(
        f"CHGNet {CHGNET_PYR_VFE:.0f}", xy=(pyr_x + 0.14, CHGNET_PYR_VFE),
        xytext=(pyr_x - 0.55, CHGNET_PYR_VFE * 1.45),
        fontsize=7.5, color="#1b9e77",
        arrowprops=dict(arrowstyle="->", color="#1b9e77", lw=0.7),
    )

    # --- illustrative energy scale, not a kinetic threshold ---
    ax.axhline(
        THRESHOLD, color="#b2182b", linewidth=1.5, linestyle="--", zorder=4,
        label=f"Illustrative 25$k_BT$ at 298 K ≈ {THRESHOLD:.0f} meV",
    )
    ax.text(
        -0.42, THRESHOLD * 1.06,
        f"Illustrative energy scale: 25$k_BT$ ≈ {THRESHOLD:.0f} meV",
        ha="left", va="bottom", fontsize=8.5, color="#b2182b",
    )

    ax.set_yscale("log")
    ax.set_ylim(25, 1400)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylabel("Migration barrier $E_a$ (meV, log scale)", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_xlim(-0.5, len(LABELS) + 1.25)

    ax.set_title(
        "DFT vacancy-anchored hydrogen-migration barriers across the iron-sulfide series\n"
        "(all PBE, $U=0$; spin setting per mineral)",
        fontsize=11,
    )
    ax.grid(True, axis="y", which="both", alpha=0.25, linestyle=":")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    caption = (
        "All barriers: DFT QE pw.x, PBE, U=0\n"
        "Greigite: two symmetry-distinct S–S edges of one $V_{Fe}$ octahedron\n"
        "Pyrite $V_S$ pocket is a different channel, plotted for scale"
    )
    fig.tight_layout()
    fig.text(
        0.5, -0.045, caption, ha="center", va="top", fontsize=7.5,
        bbox=dict(facecolor="white", edgecolor="grey", alpha=0.9,
                  boxstyle="round,pad=0.4"),
    )
    for ext in ("pdf", "png", "svg"):
        kw = {"dpi": 600} if ext == "png" else {}
        fig.savefig(out_dir / f"fig5_barrier_landscape.{ext}",
                    bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"Wrote fig5_barrier_landscape.{{pdf,png,svg}} to {out_dir}/")


if __name__ == "__main__":
    make_figure(Path(__file__).parent)
