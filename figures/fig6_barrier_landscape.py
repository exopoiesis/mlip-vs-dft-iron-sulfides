"""
Figure 6 — Fe-S series barrier landscape (log-scale, meV).
DFT V-anchored proton-migration barriers sorted by ascending E_a.
Horizontal threshold line at 25·k_BT@298K ≈ 642 meV.

Data (verified, all PBE U=0):
  Mackinawite V_Fe+S-H:  43 meV
  Pyrite V_S2 dimer:     95 meV
  Marcasite V_Fe+S-H:   208 meV
  Pyrite V_Fe+S-H:      268 meV
  Greigite V_Fe+S-H:   1860 meV

Optional MLIP overlays (where available):
  Pyrite V_Fe: MACE 182 meV, CHGNet 223 meV
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# --- Verified DFT data (ascending order) ---
LABELS = [
    "Mackinawite\n$V_{Fe}$+S–H",
    "Pyrite\n$V_{S_2}$ dimer",
    "Marcasite\n$V_{Fe}$+S–H",
    "Pyrite\n$V_{Fe}$+S–H",
    "Greigite\n$V_{Fe}$+S–H",
]
EA_DFT = np.array([43.0, 95.0, 208.0, 268.0, 1860.0])  # meV

# kinetic accessibility threshold: 25 * k_B * 298 K = 25 * 25.69 meV ≈ 642 meV
K_BT_298 = 25.69       # meV
THRESHOLD = 25 * K_BT_298  # ~642 meV

# bar colors: accessible (< threshold) = blue family, inaccessible (> threshold) = red
COLORS_BAR = ["#4393c3", "#4393c3", "#4393c3", "#4393c3", "#d6604d"]

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

    # MLIP overlay points for Pyrite V_Fe (index 3)
    pyr_x = x[3]
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
    ax.text(
        len(LABELS) - 0.52, THRESHOLD * 1.08,
        f"25$k_BT$ ≈ {THRESHOLD:.0f} meV\n(kinetic accessibility threshold)",
        ha="right", va="bottom", fontsize=8.5, color="#b2182b",
    )

    # "accessible" / "forbidden" zone annotations
    ax.text(
        1.5, THRESHOLD * 0.35,
        "accessible\n(below threshold)",
        ha="center", va="center", fontsize=8,
        color="#2166ac", alpha=0.7, fontstyle="italic",
    )
    ax.text(
        4.0, THRESHOLD * 2.5,
        "kinetically\nforbidden",
        ha="center", va="center", fontsize=8,
        color="#b2182b", alpha=0.7, fontstyle="italic",
    )

    # Log scale y-axis
    ax.set_yscale("log")
    ax.set_ylim(20, 5000)
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

    # caption box
    caption = (
        "All barriers: DFT QE pw.x PBE U=0, spin per mineral\n"
        "MLIP diamonds: MACE-MP-0 medium / CHGNet-v0.3.0 on Pyrite V_Fe RC only\n"
        "Threshold = 25 k_BT at 298 K ≈ 642 meV (first-passage time ~1 ns)"
    )
    ax.text(
        0.98, 0.02, caption,
        transform=ax.transAxes,
        fontsize=7.5, va="bottom", ha="right",
        bbox=dict(facecolor="white", edgecolor="grey", alpha=0.88,
                  boxstyle="round,pad=0.4"),
    )

    fig.tight_layout()
    fig.savefig(out_dir / "fig6_barrier_landscape.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig6_barrier_landscape.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "fig6_barrier_landscape.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote 3 files (pdf+png+svg) to {out_dir}/")


if __name__ == "__main__":
    out_dir = Path(__file__).parent
    make_figure(out_dir)
