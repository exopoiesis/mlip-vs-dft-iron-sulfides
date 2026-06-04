"""
Figure S (SI §O) -- per-image foundation-MLIP vs DFT NEB band correlation.

Stanford review Q5: shows WHAT TYPE of failure the foundation MLIPs exhibit
along the canonical pyrite V_S2 H-hop path (9 images, symmetric, DFT saddle
img4 = 94.6 meV, QE PBE).

Panels:
  (a) per-image band overlay: DFT vs MACE-MP-0 large vs CHGNet v0.3.0 (meV)
  (b) parity inset: MLIP per-image rel-energy vs DFT per-image rel-energy

Data source: tmp/q5_summary.json (produced by tmp/q5_correlation.py from
results/mlip_canonical/*.json + neb.traj-extracted DFT band). All numbers
are read from the summary JSON -- no hard-coded band values here.

Output: figS_mlip_dft_band_correlation.{pdf, png, svg}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
# project root = .../project-third-matter ; figures live at paper/MLIPvsDFT/figures
ROOT = HERE.parents[2]
SUMMARY = ROOT / "tmp" / "q5_summary.json"


def load_pyrite_case():
    with open(SUMMARY) as fh:
        summary = json.load(fh)
    for c in summary["cases"]:
        if c["case"] == "pyrite_V_S2":
            return c
    raise RuntimeError("pyrite_V_S2 case not found in q5_summary.json")


def make_figure(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    c = load_pyrite_case()
    n = c["n_images"]
    img = np.arange(n)
    dft = np.array(c["dft_band_meV"], float)
    mace = np.array(c["models"]["mace_mp0_large"]["mlip_band_meV"], float)
    chg = np.array(c["models"]["chgnet_v030"]["mlip_band_meV"], float)

    R_mace = c["models"]["mace_mp0_large"]["pearson_R"]
    R_chg = c["models"]["chgnet_v030"]["pearson_R"]
    rmse_mace = c["models"]["mace_mp0_large"]["rmse_meV"]
    rmse_chg = c["models"]["chgnet_v030"]["rmse_meV"]

    fig, (ax, axp) = plt.subplots(
        1, 2, figsize=(11.0, 4.6), gridspec_kw={"width_ratios": [1.55, 1.0]}
    )

    # ---- panel (a): band overlay ----
    ax.plot(img, dft, "o-", color="#5b3a8c", lw=2.2, ms=7,
            label="DFT (QE PBE), 94.6 meV", zorder=5)
    ax.plot(img, mace, "s--", color="#d95f02", lw=1.8, ms=6,
            label=f"MACE-MP-0 large (R={R_mace})")
    ax.plot(img, chg, "^--", color="#1b9e77", lw=1.8, ms=6,
            label=f"CHGNet v0.3.0 (R={R_chg})")
    ax.axhline(0.0, color="gray", lw=0.6, ls=":")
    ax.axvline(c["dft_saddle_index"], color="#5b3a8c", lw=0.8, ls=":", alpha=0.6)
    ax.set_xlabel("NEB image index")
    ax.set_ylabel("Relative energy (meV, ref = endpoint A)")
    ax.set_title("(a) Pyrite V$_{S_2}$ H-hop band: DFT vs foundation MLIP")
    ax.set_xticks(img)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.text(0.02, 0.95,
            "MACE: flat (no barrier)\nCHGNet: false wells mid-path",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

    # ---- panel (b): parity ----
    lim = [min(dft.min(), mace.min(), chg.min()) - 8,
           dft.max() + 8]
    axp.plot(lim, lim, color="gray", lw=0.8, ls="--", label="ideal (y=x)")
    axp.scatter(dft, mace, color="#d95f02", marker="s", s=42,
                label=f"MACE (RMSE {rmse_mace:.0f} meV)")
    axp.scatter(dft, chg, color="#1b9e77", marker="^", s=42,
                label=f"CHGNet (RMSE {rmse_chg:.0f} meV)")
    axp.set_xlabel("DFT rel-energy (meV)")
    axp.set_ylabel("MLIP rel-energy (meV)")
    axp.set_title("(b) Per-image parity")
    axp.set_xlim(lim)
    axp.set_ylim(lim)
    axp.set_aspect("equal", adjustable="box")
    axp.legend(frameon=False, fontsize=8.5, loc="upper left")

    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(out_dir / f"figS_mlip_dft_band_correlation.{ext}",
                    dpi=300, bbox_inches="tight")
    print(f"wrote figS_mlip_dft_band_correlation.* to {out_dir}")


if __name__ == "__main__":
    make_figure(HERE)
