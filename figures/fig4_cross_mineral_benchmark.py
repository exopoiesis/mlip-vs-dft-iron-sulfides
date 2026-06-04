"""
Figure 4 — Cross-mineral foundation-MLIP vs DFT benchmark (T6 paper claim).
Per main-text §3.4 Table 1 + SI §J data table:
- Pyrite: DFT 94.6 meV, MACE 0 meV (flat), CHGNet 27.6 meV (topologically distorted band, spurious sub-endpoint wells, saddle 3.4x below DFT)
- Mackinawite: DFT 42.9 meV (V_Fe + S-H), MACE/CHGNet flat PES on V_Fe pathway
- Pentlandite: cubane structural-motif MLIP failure (§3.3.2); V_Fe DFT barrier deferred to companion MagNEB study (nspin=2)
Output: fig4_cross_mineral_benchmark.{pdf, png, svg}
Note: AFM+U Tier 1 mack bracket [-85, +171] meV (U=2) shown as error bar
on the mackinawite DFT bar (per SI §H methodology-induced sensitivity range).
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Data per SI §J.1 table + main-text §3.4 Table 1
MINERALS = ["Pyrite\n(V_S₂ dimer)", "Mackinawite\n(V_Fe + S–H)", "Pentlandite\n(V_Fe + S–H)"]
DFT_VALUES = [0.0946, 0.0429, np.nan]   # eV; pent endpoint-only
MACE_VALUES = [0.000, 0.000, 0.000]     # eV; effectively flat PES across all 3
CHGNET_VALUES = [0.0276, 0.000, 0.000]  # eV; pyr V_S2 production band (saddle 27.6 meV, spurious -24 meV sub-endpoint wells, wrong topology)
# AFM+U Tier 1 mack bracket (SI §H.3.2): [-85, +171] meV at U=2
AFMU_LOWER_U2 = -0.085  # eV
AFMU_UPPER_U2 = 0.171   # eV
AFMU_LOWER_U4 = -0.159  # eV
AFMU_UPPER_U4 = 0.245   # eV


def make_figure(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x = np.arange(len(MINERALS))
    width = 0.27

    # DFT bars (paper-grade anchors; pent annotated separately)
    dft_bars = ax.bar(
        x - width,
        [v if not np.isnan(v) else 0.001 for v in DFT_VALUES],  # tiny stub for pent endpoint-only
        width,
        label="DFT QE PBE (this work)",
        color="#5b3a8c",
        edgecolor="black",
        linewidth=0.5,
    )
    # MACE-MP-0 bars (zero-shot failure -- effectively flat)
    mace_bars = ax.bar(
        x,
        [max(v, 0.0005) for v in MACE_VALUES],  # tiny stub so bars visible
        width,
        label="MACE-MP-0 medium",
        color="#d95f02",
        edgecolor="black",
        linewidth=0.5,
        hatch="///",
    )
    # CHGNet bars
    chgnet_bars = ax.bar(
        x + width,
        [max(v, 0.0005) for v in CHGNET_VALUES],
        width,
        label="CHGNet-v0.3.0",
        color="#1b9e77",
        edgecolor="black",
        linewidth=0.5,
        hatch="\\\\\\",
    )

    # Annotate DFT values
    for i, v in enumerate(DFT_VALUES):
        if np.isnan(v):
            ax.text(
                x[i] - width, 0.020,
                "V_Fe DFT:\nMagNEB\n(nspin = 2)",
                ha="center", va="bottom", fontsize=8,
                fontstyle="italic", color="#5b3a8c",
            )
        else:
            ax.text(
                x[i] - width, v + 0.008,
                f"{v * 1000:.1f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    # Annotate MLIP values (since they're tiny)
    for i, (mv, cv) in enumerate(zip(MACE_VALUES, CHGNET_VALUES)):
        if i < 2:  # pyr + mack have measured MLIP values
            mace_label = "≈ 0" if mv < 0.001 else f"{mv * 1000:.0f}"
            chgnet_label = "≈ 0" if cv < 0.001 else f"{cv * 1000:.1f}"
            ax.text(x[i], 0.008, mace_label, ha="center", va="bottom", fontsize=8, color="#d95f02")
            ax.text(x[i] + width, 0.008, chgnet_label, ha="center", va="bottom", fontsize=8, color="#1b9e77")

    # AFM+U bracket on mack DFT bar (methodology-induced sensitivity range)
    mack_x = x[1] - width
    mack_dft = DFT_VALUES[1]
    afmu_u2_half = (AFMU_UPPER_U2 - AFMU_LOWER_U2) / 2
    ax.errorbar(
        mack_x, mack_dft,
        yerr=[[afmu_u2_half], [afmu_u2_half]],
        fmt="none", ecolor="#5b3a8c", capsize=4, capthick=1.5, linewidth=1.5,
        label="AFM+U Tier 1 sensitivity range (SI §H)",
    )
    ax.annotate(
        "U=2 bracket\n[−85, +171] meV",
        xy=(mack_x, mack_dft + afmu_u2_half),
        xytext=(mack_x - 0.18, 0.22),
        fontsize=7.5, ha="left", color="#5b3a8c",
        arrowprops=dict(arrowstyle="-", color="#5b3a8c", linewidth=0.7),
    )

    # Axes
    ax.set_xticks(x)
    ax.set_xticklabels(MINERALS, fontsize=10)
    ax.set_ylabel("Migration barrier E$_a$ (eV)", fontsize=11)
    ax.set_title(
        "Cross-mineral foundation-MLIP vs DFT benchmark\n"
        + "vacancy-anchored proton migration on canonical RC (V_S₂ pyr, V_Fe+S–H mack/pent)",
        fontsize=11,
    )
    ax.set_ylim(-0.020, 0.32)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    # Add interpretive caption text box
    caption = (
        "Two distinct foundation-MLIP failure modes + a pathway pitfall (SI §J.2):\n"
        "(1) Zero-shot PES failure — pyr V_S₂ dimer hop (MACE 0 flat, CHGNet 27.6 meV distorted band vs DFT 95 meV);\n"
        "(2) Structural-motif failure — pent cubane [Fe₄S₄] 3+3 distortion (§3.3.2; not shown);\n"
        "(3) V_S+H pathway-selection pitfall — universally broken for fcc/layered Fe-S (SI §C)."
    )
    ax.text(
        0.02, 0.98, caption,
        transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
        bbox=dict(facecolor="white", edgecolor="grey", alpha=0.9, boxstyle="round,pad=0.4"),
    )

    fig.tight_layout()
    fig.savefig(out_dir / "fig4_cross_mineral_benchmark.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig4_cross_mineral_benchmark.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "fig4_cross_mineral_benchmark.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote 3 files (pdf+png+svg) to {out_dir}/")


if __name__ == "__main__":
    out_dir = Path(__file__).parent
    make_figure(out_dir)
