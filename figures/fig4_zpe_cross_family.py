"""
Figure 4 — ZPE cross-family trend (two-panel).
Panel (a): scatter — reactant S-H stretch frequency vs |ΔZPE‡|
Panel (b): grouped bar — E_a electronic vs E_a ZPE-corrected

Data (verified, main-text Table 4; SI §N.8):
  Mackinawite: nu=1721 cm-1, |ΔZPE‡|=66 meV, E_a elec=42.9 meV, E_a ZPE-corr≈0 (-23±15)
  Marcasite:   nu=2234 cm-1, |ΔZPE‡|=85 meV, E_a elec=208 meV,  E_a ZPE-corr=123 meV
  Pyrite:      nu=2248 cm-1, |ΔZPE‡|=94 meV, E_a elec=268 meV,  E_a ZPE-corr=173 meV
  Greigite:    nu=2439 cm-1, |ΔZPE‡|=103 meV, E_a elec=236 meV, E_a ZPE-corr=133 meV
               (channel edge; ΔZPE‡ = -103.0 meV, of which -109.8 proton / +6.8 framework)

The greigite row was added in 2026-09 with the corrected channel-edge band. The
generator previously plotted three minerals while the caption already claimed
four, and wrote its output under the pre-pentlandite-removal figure number.

|ΔZPE‡| as a fraction of the bare zero-point quantum ½hν_S–H:
  mackinawite 0.62, marcasite 0.61, pyrite 0.67, greigite 0.68 — the 0.61–0.68
  band quoted in the caption (1 cm-1 = 0.1239842 meV).
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# --- Verified data ---
MINERALS = ["Mackinawite", "Marcasite", "Pyrite", "Greigite"]
COLORS   = ["#1b9e77", "#d95f02", "#7570b3", "#7b3294"]

NU_SH    = np.array([1721, 2234, 2248, 2439])   # cm-1, reactant S-H stretch
DZPE     = np.array([66,   85,   94,   103])    # meV, |ΔZPE‡|

EA_ELEC  = np.array([42.9, 208.0, 268.0, 236.0])  # meV, electronic
EA_ZPE   = np.array([-23.0, 123.0, 173.0, 133.0]) # meV, harmonic estimate; no clipping
EA_ZPE_ERR = np.array([15.0,  8.0,   0.0,   0.0]) # meV, uncertainty (mack ±15, marc ±8)


def make_figure(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.0, 5.0))

    # ------------------------------------------------------------------ #
    # Panel (a) — scatter: nu_SH vs |ΔZPE‡|
    # ------------------------------------------------------------------ #
    for i, (mineral, nu, dz, col) in enumerate(zip(MINERALS, NU_SH, DZPE, COLORS)):
        ax_a.scatter(nu, dz, s=90, color=col, edgecolors="black", linewidths=0.6,
                     zorder=5, label=mineral)
        # offset labels to avoid overlap
        offsets = [(-80, 5), (10, -8), (10, 4), (-60, 6)]
        ax_a.annotate(
            mineral,
            xy=(nu, dz),
            xytext=(nu + offsets[i][0], dz + offsets[i][1]),
            fontsize=9,
            color=col,
            fontweight="bold",
        )

    ax_a.set_xlabel(r"Reactant S–H stretch frequency $\nu$ (cm$^{-1}$)", fontsize=10)
    ax_a.set_ylabel(r"|ΔZPE$^{\ddagger}$| (meV)", fontsize=10)
    ax_a.set_title("(a) Harmonic zero-point correction\nand reactant S–H stretch", fontsize=10)
    ax_a.set_xlim(1550, 2560)
    ax_a.set_ylim(50, 120)
    ax_a.grid(True, alpha=0.3, linestyle=":")
    ax_a.legend(fontsize=8.5, framealpha=0.95)

    # annotation arrow label
    ax_a.annotate(
        "|ΔZPE‡| grows with\nreactant S–H stretch",
        xy=(2234, 85),
        xytext=(1800, 100),
        fontsize=8,
        color="grey",
        arrowprops=dict(arrowstyle="->", color="grey", lw=0.8),
    )

    # ------------------------------------------------------------------ #
    # Panel (b) — grouped bar: E_a electronic vs ZPE-corrected
    # ------------------------------------------------------------------ #
    x = np.arange(len(MINERALS))
    width = 0.35

    bars_elec = ax_b.bar(
        x - width / 2,
        EA_ELEC,
        width,
        label="Electronic $E_a$",
        color="#5b3a8c",
        edgecolor="black",
        linewidth=0.5,
    )
    bars_zpe = ax_b.bar(
        x + width / 2,
        EA_ZPE,
        width,
        label="Harmonic ZPE-corrected",
        color="#e6ab02",
        edgecolor="black",
        linewidth=0.5,
        hatch="///",
    )

    # Error bars on ZPE-corrected where uncertainty given
    for i, (xi, yerr) in enumerate(zip(x + width / 2, EA_ZPE_ERR)):
        if yerr > 0:
            ax_b.errorbar(xi, EA_ZPE[i], yerr=yerr,
                          fmt="none", ecolor="black", capsize=4,
                          capthick=1.2, linewidth=1.2)

    # Annotate bar values
    for i, rect in enumerate(bars_elec):
        h = rect.get_height()
        ax_b.text(rect.get_x() + rect.get_width() / 2, h + 3,
                  f"{EA_ELEC[i]:.1f}", ha="center", va="bottom", fontsize=8.5,
                  fontweight="bold", color="#5b3a8c")

    for i, rect in enumerate(bars_zpe):
        h = rect.get_height()
        label_y = h + 3
        if i == 0:  # mackinawite special label — clear of the ±15 meV error bar
            ax_b.text(rect.get_x() + rect.get_width() / 2, 70,
                      "−23±15 meV\n(harmonic estimate)",
                      ha="center", va="bottom", fontsize=7.5,
                      fontstyle="italic", color="#705400",
                      bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=1.5))
        else:
            ax_b.text(rect.get_x() + rect.get_width() / 2, label_y,
                      f"{EA_ZPE[i]:.0f}", ha="center", va="bottom", fontsize=8.5,
                      fontweight="bold", color="#9b7400")

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(MINERALS, fontsize=10)
    ax_b.set_ylabel("Barrier $E_a$ (meV)", fontsize=10)
    ax_b.set_title("(b) Electronic and harmonic\nZPE-corrected barriers", fontsize=10)
    ax_b.set_ylim(-50, 320)
    ax_b.axhline(0, color="black", linewidth=0.5)
    ax_b.grid(True, axis="y", alpha=0.3, linestyle=":")
    ax_b.legend(loc="upper left", fontsize=9, framealpha=0.95)

    # panel (b) caption box
    caption_b = (
        "All U=0, PBE; harmonic partial-Hessian ZPE\n"
        "ΔZPE‡ = ZPE_saddle − ZPE_endpoint (SI §N.8)\n"
        "Mack: correction exceeds electronic barrier; quantum rate undetermined"
    )
    fig.suptitle(
        "Harmonic zero-point corrections to Fe–S hydrogen-migration barriers",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.text(
        0.75, -0.02, caption_b, ha="center", va="top", fontsize=7.5,
        bbox=dict(facecolor="white", edgecolor="grey", alpha=0.88,
                  boxstyle="round,pad=0.4"),
    )
    for ext in ("pdf", "png", "svg"):
        kw = {"dpi": 600} if ext == "png" else {}
        fig.savefig(out_dir / f"fig4_zpe_cross_family.{ext}",
                    bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"Wrote fig4_zpe_cross_family.{{pdf,png,svg}} to {out_dir}/")


if __name__ == "__main__":
    out_dir = Path(__file__).parent
    make_figure(out_dir)
