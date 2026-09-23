"""
Figure 6 — Fe-S series barrier landscape (log-scale, meV).
DFT V-anchored proton-migration barriers sorted by ascending E_a.
Horizontal threshold line at 25·k_BT@298K ≈ 642 meV.

Data (verified, all PBE U=0):
  Mackinawite V_Fe+S-H:            43 meV
  Pyrite V_S pocket, Fe-Fe hydride: 95 meV
  Marcasite V_Fe+S-H:             208 meV
  Greigite V_Fe+S-H, channel edge: 236 meV
  Pyrite V_Fe+S-H:                268 meV
  Greigite V_Fe+S-H, cation edge:  567 meV

Optional MLIP overlays (where available):
  Pyrite V_Fe: MACE 182 meV, CHGNet 223 meV

CORRECTED 2026-09-22. This figure previously plotted a single greigite bar at 1860 meV, the only
point above the 25 kT threshold, and annotated a "kinetically forbidden" zone around it. That
barrier is RETRACTED: the band ran along a trans axis of the vacancy octahedron whose midpoint is
the vacant Fe site, a special position where the force vanishes by symmetry (three imaginary modes,
not one). It is replaced by the two symmetry-distinct edge classes of the same octahedron, both of
which lie BELOW the threshold. Every barrier in the series is now kinetically accessible, so the
forbidden-zone annotation is removed rather than moved.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# --- Verified DFT data (ascending order) ---
LABELS = [
    "Mackinawite\n$V_{Fe}$+S–H",
    "Pyrite\n$V_S$ hydride",
    "Marcasite\n$V_{Fe}$+S–H",
    "Greigite\n$V_{Fe}$+S–H\n(channel edge)",
    "Pyrite\n$V_{Fe}$+S–H",
    "Greigite\n$V_{Fe}$+S–H\n(cation edge)",
]
EA_DFT = np.array([43.0, 95.0, 208.0, 236.0, 268.0, 567.0])  # meV

# kinetic accessibility threshold: 25 * k_B * 298 K = 25 * 25.69 meV ≈ 642 meV
K_BT_298 = 25.69       # meV
THRESHOLD = 25 * K_BT_298  # ~642 meV

# The two greigite bars are the two edge classes of ONE vacancy octahedron, highlighted because
# the spread between them (331 meV) exceeds the spread across the other three minerals (225 meV):
# path choice inside one mineral outweighs mineral choice.
COLORS_BAR = ["#4393c3", "#4393c3", "#4393c3", "#7b3294", "#4393c3", "#7b3294"]

# MLIP overlays for Pyrite V_Fe (index 3)
MACE_PYR_VFE    = 182.0  # meV
CHGNET_PYR_VFE  = 223.0  # meV


def make_figure(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    x = np.arange(len(LABELS))
    width = 0.55

    bars = ax.bar(
        x,
        EA_DFT,
        width,
        color=COLORS_BAR,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
        label="DFT QE PBE (this work, $U=0$)",
    )

    # Annotate each bar with its value
    for i, (bar, val) in enumerate(zip(bars, EA_DFT)):
        label_str = f"{val:.0f} meV"
        # For greigite, place label inside bar near top since it's very tall
        if val > THRESHOLD:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val * 0.55,  # mid-log position
                label_str,
                ha="center", va="center", fontsize=9,
                fontweight="bold", color="white",
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val * 1.25,
                label_str,
                ha="center", va="bottom", fontsize=9,
                fontweight="bold", color="black",
            )

    # MLIP overlay points for Pyrite V_Fe (index 4 after the greigite channel bar was inserted)
    pyr_x = x[LABELS.index("Pyrite\n$V_{Fe}$+S–H")]
    ax.scatter(
        [pyr_x - 0.12, pyr_x + 0.12],
        [MACE_PYR_VFE, CHGNET_PYR_VFE],
        s=70,
        marker="D",
        color=["#d95f02", "#1b9e77"],
        edgecolors="black",
        linewidths=0.6,
        zorder=6,
        label="MACE-MP-0 / CHGNet $V_{Fe}$ (Pyrite)",
    )
    ax.annotate(
        f"MACE {MACE_PYR_VFE:.0f}",
        xy=(pyr_x - 0.12, MACE_PYR_VFE),
        xytext=(pyr_x - 0.42, MACE_PYR_VFE * 1.5),
        fontsize=7.5, color="#d95f02",
        arrowprops=dict(arrowstyle="->", color="#d95f02", lw=0.7),
    )
    ax.annotate(
        f"CHGNet {CHGNET_PYR_VFE:.0f}",
        xy=(pyr_x + 0.12, CHGNET_PYR_VFE),
        xytext=(pyr_x + 0.22, CHGNET_PYR_VFE * 1.8),
        fontsize=7.5, color="#1b9e77",
        arrowprops=dict(arrowstyle="->", color="#1b9e77", lw=0.7),
    )

    # Kinetic accessibility threshold line
    ax.axhline(
        THRESHOLD, color="#b2182b", linewidth=1.5, linestyle="--", zorder=4,
        label=f"Kinetic accessibility threshold (25$k_BT$@298 K ≈ {THRESHOLD:.0f} meV)",
    )
    # Every barrier in the corrected series lies below the threshold, so there is no
    # "forbidden" zone to annotate. The one substantive point the figure carries beyond the bars
    # -- that the largest gap is between two paths inside ONE mineral -- goes in the caption
    # under the axes rather than as an in-axes bracket, which collided with the bars at every
    # placement tried.
    # No in-axes label for the threshold: the legend already names it, and at every placement
    # tried the text collided with the tallest bar's value label.

    # Log scale y-axis
    ax.set_yscale("log")
    ax.set_ylim(25, 1600)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylabel("Migration barrier $E_a$ (meV, log scale)", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=9.5)
    ax.set_xlim(-0.5, len(LABELS) - 0.5)

    ax.set_title(
        "DFT $V$-anchored proton-migration barriers across the iron-sulfide series\n"
        "(all PBE, $U=0$, vacancy-anchored RC)",
        fontsize=11,
    )
    ax.grid(True, axis="y", which="both", alpha=0.25, linestyle=":")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    # Caption goes UNDER the axes, so it cannot collide with a bar at any data range.
    caption = (
        "All barriers: DFT QE pw.x PBE $U=0$, spin treatment per mineral.  "
        "MLIP diamonds: MACE-MP-0 large / CHGNet-v0.3.0, pyrite $V_{Fe}$ coordinate only.  "
        "Threshold = 25 $k_BT$ at 298 K ≈ 642 meV — every barrier in the corrected series lies below it.\n"
        "The two greigite bars are the two symmetry-distinct S–S edge classes of the SAME vacancy "
        "octahedron, computed from a common endpoint. The 331 meV between them exceeds the 225 meV\n"
        "spread across mackinawite, marcasite and pyrite $V_{Fe}$ combined: inside this structure "
        "type, which path you pick matters more than which mineral you pick."
    )
    fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=7.8)

    fig.tight_layout()
    fig.savefig(out_dir / "fig6_barrier_landscape.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig6_barrier_landscape.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "fig6_barrier_landscape.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote 3 files (pdf+png+svg) to {out_dir}/")


if __name__ == "__main__":
    out_dir = Path(__file__).parent
    make_figure(out_dir)
