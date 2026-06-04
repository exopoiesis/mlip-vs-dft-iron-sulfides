#!/usr/bin/env python3
"""Figure 2 — Mackinawite V_Fe + S–H lateral hop.

Panel (a): V_Fe DFT NEB profile (paper-quotable 0.043 eV) with annotated
           saddle and symmetric endpoints.
Panel (b): Cross-method E_a comparison bar chart — V_Fe DFT (this work) vs.
           literature V_Fe surface anchor (Liu 2021) vs. legacy V_S+H DFT and
           foundation-MLIP values.

Output: paper/MLIPvsDFT/figures/fig2_mackinawite_vfe.{png,pdf,svg}

Run: python paper/MLIPvsDFT/figures/fig2_mackinawite_vfe.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# =============================================================
# Project root + data paths
# =============================================================
ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
FIGDIR = Path(__file__).resolve().parent

VFE_JSON = (
    RESULTS / "dft_datasets" / "2026-05-03"
    / "mack_vfe_w3_aborted_2026-05-03" / "results"
    / "neb_canonical_mack_72at_qe_VFe.json"
)


def load_neb(path):
    with open(path) as f:
        return json.load(f)


# =============================================================
# Load V_Fe DFT NEB
# =============================================================
vfe = load_neb(VFE_JSON)
vfe_E = np.array(vfe["neb_energies_rel_eV"])
n_img = len(vfe_E)
Ea_vfe = vfe["E_a_eV"]
saddle_idx = vfe.get("saddle_image_idx", int(np.argmax(vfe_E)))

print(f"V_Fe DFT NEB: E_a = {Ea_vfe:.4f} eV ({n_img} images)")
print(f"  Saddle at image {saddle_idx}, ΔE_endpoints = {vfe['dE_endpoints_eV']*1e6:.2f} µeV")
print(f"  Test A: {vfe['test_a']['pass']}, h_displacement = {vfe['test_a']['h_displacement_A']:.3f} Å")
print(f"  Hop distance = {vfe['hop_distance_A']:.3f} Å")

# =============================================================
# Cross-method bar chart data
# =============================================================
# This work (mackinawite V_Fe DFT, paper-quotable)
# Literature anchor: Liu 2021 ACS Omega L-650 — V_Fe surface 0.26 eV
# Legacy V_S+H pathway: artifact (~0 effective, see SI forensic)
# Foundation MLIPs (legacy V_S+H intra-layer pathway from manuscript v1):
#   MACE intra 0.44 eV; GPAW intra 0.738 eV (legacy DFT cross-check, v1 mscript)
# QE cross-layer (legacy 2×2×1 supercell, neb_mackinawite_qe_result.json): 2.479 eV
methods = [
    ("This work\nDFT V_Fe\n(72 at)", 0.0429, "#9467bd"),
    ("Liu 2021\nDFT V_Fe surface\n(literature)", 0.26, "#1f77b4"),
    ("Legacy MACE\nintra V_S+H\n(v1)", 0.44, "#ff7f0e"),
    ("Legacy DFT\nintra V_S+H\n(GPAW)", 0.738, "#d62728"),
    ("Legacy DFT\ncross-layer\n(QE)", 2.479, "#7f7f7f"),
]

# =============================================================
# Figure
# =============================================================
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "axes.titlesize": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.5, 3.2), constrained_layout=True)

# ---- Panel (a): V_Fe DFT NEB profile ----
rc = np.linspace(0.0, 1.0, n_img)
energies_meV = vfe_E * 1000.0

axA.plot(rc, energies_meV,
         color="#9467bd", marker="D", markersize=5, linewidth=1.7,
         label=f"DFT V_Fe + S–H hop\n$E_a$ = {Ea_vfe*1000:.1f} meV")

# Annotate saddle
axA.annotate(f"Saddle: {energies_meV[saddle_idx]:.1f} meV",
             xy=(rc[saddle_idx], energies_meV[saddle_idx]),
             xytext=(rc[saddle_idx] + 0.15, energies_meV[saddle_idx] + 5),
             fontsize=8,
             arrowprops=dict(arrowstyle="->", color="grey", lw=0.5))

# Annotate symmetric endpoints
axA.annotate(f"$\\Delta E_\\mathrm{{endpoints}}$ = {vfe['dE_endpoints_eV']*1e6:.1f} µeV",
             xy=(0.5, -3),
             xytext=(0.5, -8),
             fontsize=7, color="grey",
             ha="center")

axA.axhline(0, color="grey", linewidth=0.5, linestyle=":")
axA.set_xlabel("Reaction coordinate")
axA.set_ylabel("Relative energy (meV)")
axA.set_title(r"(a) Mackinawite 3$\times$3$\times$2 (72 atoms): V$_\mathrm{Fe}$ + S–H hop")
axA.legend(loc="upper right", frameon=False)
axA.set_xlim(-0.02, 1.02)
axA.set_ylim(-12, 55)

# ---- Panel (b): Cross-method bar chart ----
labels = [m[0] for m in methods]
values = [m[1] for m in methods]
colors = [m[2] for m in methods]

bars = axB.bar(range(len(methods)), values, color=colors, edgecolor="black", linewidth=0.5)

# Annotate values
for i, (bar, v) in enumerate(zip(bars, values)):
    if v < 0.1:
        text = f"{v*1000:.1f} meV"
    else:
        text = f"{v:.2f} eV"
    axB.text(bar.get_x() + bar.get_width() / 2,
             v + 0.05, text,
             ha="center", va="bottom", fontsize=8)

# Highlight «paper-quotable» bar
bars[0].set_edgecolor("#9467bd")
bars[0].set_linewidth(2.5)

axB.set_xticks(range(len(methods)))
axB.set_xticklabels(labels, fontsize=7)
axB.set_ylabel("$E_a$ (eV)")
axB.set_title("(b) Mackinawite proton migration: cross-method comparison")
axB.set_ylim(0, 2.8)
axB.axhline(Ea_vfe, color="#9467bd", linewidth=0.7, linestyle="--", alpha=0.5)

# Save
for ext in ["png", "pdf", "svg"]:
    fig.savefig(FIGDIR / f"fig2_mackinawite_vfe.{ext}", bbox_inches="tight", dpi=300)
    print(f"  Saved: {FIGDIR / f'fig2_mackinawite_vfe.{ext}'}")

plt.close(fig)
print("\nDone.")
