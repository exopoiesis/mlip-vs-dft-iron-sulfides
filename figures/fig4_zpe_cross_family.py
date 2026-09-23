"""
Figure 5 — ZPE cross-family trend (two-panel).
Panel (a): scatter — reactant S-H stretch frequency vs |ΔZPE‡|
Panel (b): grouped bar — E_a electronic vs E_a ZPE-corrected

Data (verified, Paper #1 SI §N.8 and §K.13):
  Mackinawite: nu=1721 cm-1, |ΔZPE‡|=66 meV,  E_a elec=42.9 meV, E_a ZPE-corr≈0 (-23±15)
  Marcasite:   nu=2234 cm-1, |ΔZPE‡|=85 meV,  E_a elec=208 meV,  E_a ZPE-corr=123 meV
  Pyrite:      nu=2248 cm-1, |ΔZPE‡|=94 meV,  E_a elec=268 meV,  E_a ZPE-corr=173 meV
  Greigite:    nu=2439 cm-1, |ΔZPE‡|=103 meV, E_a elec=236 meV,  E_a ZPE-corr=133 meV

GREIGITE ADDED 2026-09-22. The submitted version declined this calculation, on the argument that a
correction of order 0.1 eV would be negligible against a 1.86 eV barrier. Both premises are gone:
that barrier is retracted (the band ran along a trans axis whose midpoint is the vacant Fe site, a
special position where the force vanishes by symmetry), and against the corrected 236 meV the
correction is the second-largest in the series in relative terms, 44 per cent. It also extends the
trend from three points to four, monotonically, at a near-constant fraction of the bare zero-point
quantum: |dZPE| / (0.5*h*nu) = 0.62, 0.61, 0.68, 0.68.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# --- Verified data (ascending in nu_SH) ---
MINERALS = ["Mackinawite", "Marcasite", "Pyrite", "Greigite"]
COLORS   = ["#1b9e77", "#d95f02", "#7570b3", "#7b3294"]

NU_SH    = np.array([1721, 2234, 2248, 2439])   # cm-1, reactant S-H stretch
DZPE     = np.array([66,   85,   94,   103])    # meV, |ΔZPE‡|

EA_ELEC  = np.array([42.9, 208.0, 268.0, 236.0])  # meV, electronic
EA_ZPE   = np.array([  0.0, 123.0, 173.0, 133.0]) # meV, ZPE-corrected (mack shown as 0)
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
        # one per mineral; marcasite and pyrite nearly coincide in nu, so they are pushed apart
        offsets = [(-80, 5), (-40, -11), (10, 3), (-90, 5)]
        ax_a.annotate(
            mineral,
            xy=(nu, dz),
            xytext=(nu + offsets[i][0], dz + offsets[i][1]),
            fontsize=9,
            color=col,
            fontweight="bold",
        )

    # linear trend line through all four points
    coeffs = np.polyfit(NU_SH, DZPE, 1)
    nu_fit = np.linspace(1600, 2520, 200)
    ax_a.plot(nu_fit, np.polyval(coeffs, nu_fit),
              "--", color="grey", linewidth=1.0, alpha=0.7,
              label=f"Linear fit, n = {len(NU_SH)} (R²={np.corrcoef(NU_SH, DZPE)[0,1]**2:.2f})")

    ax_a.set_xlabel(r"Reactant S–H stretch frequency $\nu$ (cm$^{-1}$)", fontsize=10)
    ax_a.set_ylabel(r"|ΔZPE$^{\ddagger}$| (meV)", fontsize=10)
    ax_a.set_title("(a) Nuclear quantum effect grows\nwith reactant S–H stretch", fontsize=10)
    ax_a.set_xlim(1550, 2560)          # greigite sits at 2439 and was off-scale before
    ax_a.set_ylim(50, 115)
    ax_a.grid(True, alpha=0.3, linestyle=":")
    ax_a.legend(fontsize=8.5, framealpha=0.95, loc="lower right")

    # The constant-fraction statement is the substantive one: the correction is a near-constant
    # share of the bare zero-point quantum, not merely correlated with it.
    frac = DZPE / (NU_SH * 0.0619926)      # |dZPE| / (half h nu), meV per cm-1
    ax_a.text(
        1600, 108,
        f"|ΔZPE‡| is {frac.min():.2f}–{frac.max():.2f} of the bare ½$h\\nu_{{S-H}}$\nacross all four minerals",
        fontsize=8, color="grey", va="top",
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
        label="ZPE-corrected (rate-relevant)",
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
        if i == 0:  # mackinawite special label
            ax_b.text(rect.get_x() + rect.get_width() / 2, 6,
                      "effectively\nbarrierless\n(−23±15 meV)",
                      ha="center", va="bottom", fontsize=7.5,
                      fontstyle="italic", color="#e6ab02")
        else:
            ax_b.text(rect.get_x() + rect.get_width() / 2, label_y,
                      f"{EA_ZPE[i]:.0f}", ha="center", va="bottom", fontsize=8.5,
                      fontweight="bold", color="#9b7400")

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(MINERALS, fontsize=10)
    ax_b.set_ylabel("Barrier $E_a$ (meV)", fontsize=10)
    ax_b.set_title("(b) Nuclear quantum effect lowers barriers:\nelectronic vs ZPE-corrected", fontsize=10)
    ax_b.set_ylim(0, 360)          # headroom so the legend clears the tallest bar's label
    ax_b.axhline(0, color="black", linewidth=0.5)
    ax_b.grid(True, axis="y", alpha=0.3, linestyle=":")
    ax_b.legend(loc="upper left", fontsize=9, framealpha=0.95)

    # Caption goes under the axes rather than inside them: in-panel it collided with the legend
    # and hid the tallest bar's value label.
    caption_b = (
        "All PBE, $U=0$; harmonic partial-Hessian ZPE, ΔZPE$^{\\ddagger}$ = ZPE$_{saddle}$ − "
        "ZPE$_{endpoint}$ on identical reactive subsystems so distant modes cancel (SI §N.8, §K.13).  "
        "Mackinawite ZPE-corrected: effectively barrierless (−23 ± 15 meV).\n"
        "Greigite is the channel edge of its vacancy octahedron; its imaginary frequency is the only "
        "one here measured in the full 168 degrees of freedom rather than in a subsystem."
    )
    fig.text(0.5, -0.03, caption_b, ha="center", va="top", fontsize=7.8)

    fig.suptitle(
        "ZPE cross-family trend: nuclear quantum effects on Fe–S proton-migration barriers",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_zpe_cross_family.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig5_zpe_cross_family.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "fig5_zpe_cross_family.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote 3 files (pdf+png+svg) to {out_dir}/")


if __name__ == "__main__":
    out_dir = Path(__file__).parent
    make_figure(out_dir)
