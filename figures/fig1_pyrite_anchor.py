#!/usr/bin/env python3
"""Figure 1 — Pyrite cross-code anchor + 96-atom production NEB.

Panel A: NEB profiles 1×1×1 cross-code (GPAW + QE + ABACUS) — 5% agreement.
Panel B: 96-atom production NEB profile (QE, paper-quotable 0.0946 eV).
Panel C: V_S2 dimer hop schematic (placeholder — TODO ASE/VESTA render).

Output: paper/MLIPvsDFT/figures/fig1_pyrite_anchor.{png,pdf,svg}

Run: python paper/MLIPvsDFT/figures/fig1_pyrite_anchor.py
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

# 1×1×1 cross-code data
GPAW_JSON = RESULTS / "q071_dft_neb_pyrite.json"
QE_JSON = RESULTS / "dft_datasets" / "neb_pyrite_qe_result.json"
ABACUS_JSON = RESULTS / "dft_datasets" / "neb_pyrite_abacus_1x1x1_result.json"

# 96at production
PROD_JSON = RESULTS / "dft_datasets" / "2026-05-01" / "pyr_prod_neb_W3" / "results" / "neb_canonical_pyr_96at_qe.json"


# =============================================================
# Helpers
# =============================================================
def load_neb_profile(path, energies_key="energies_per_image"):
    """Load NEB image energies + labels. Auto-detect key fallback."""
    with open(path) as f:
        d = json.load(f)
    if energies_key in d:
        return np.array(d[energies_key]), d
    # Fallbacks
    for k in ["energies_eV", "neb_energies_rel_eV", "energies_per_image"]:
        if k in d:
            return np.array(d[k]), d
    raise KeyError(f"No NEB energies found in {path}; keys={list(d.keys())}")


def normalize_rc(n):
    """Normalize reaction coordinate to [0, 1]."""
    return np.linspace(0.0, 1.0, n)


# =============================================================
# Load data
# =============================================================
gpaw_E, gpaw_meta = load_neb_profile(GPAW_JSON)
qe_E, qe_meta = load_neb_profile(QE_JSON)
abacus_E, abacus_meta = load_neb_profile(ABACUS_JSON)
prod_E, prod_meta = load_neb_profile(PROD_JSON, energies_key="neb_energies_rel_eV")

print(f"GPAW   1×1×1: E_a = {gpaw_meta['E_a_eV']:.4f} eV  ({len(gpaw_E)} images)")
print(f"QE     1×1×1: E_a = {qe_meta['E_a_eV']:.4f} eV  ({len(qe_E)} images)")
print(f"ABACUS 1×1×1: E_a = {abacus_meta['E_a_eV']:.4f} eV  ({len(abacus_E)} images)")
print(f"QE 96at prod: E_a = {prod_meta['E_a_eV']:.4f} eV  ({len(prod_E)} images)")

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

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)

# ---- Panel A: cross-code 1×1×1 ----
colors_A = {"GPAW": "#1f77b4", "QE": "#d62728", "ABACUS": "#2ca02c"}
markers_A = {"GPAW": "o", "QE": "s", "ABACUS": "^"}

datasets_A = [
    ("GPAW", gpaw_E * 1000.0, gpaw_meta["E_a_eV"]),
    ("QE", qe_E * 1000.0, qe_meta["E_a_eV"]),
    ("ABACUS", abacus_E * 1000.0, abacus_meta["E_a_eV"]),
]

for label, energies_meV, Ea_eV in datasets_A:
    rc = normalize_rc(len(energies_meV))
    axA.plot(rc, energies_meV,
             color=colors_A[label],
             marker=markers_A[label],
             markersize=5,
             linewidth=1.5,
             label=f"{label}: $E_a$ = {Ea_eV*1000:.0f} meV")

axA.axhline(0, color="grey", linewidth=0.5, linestyle=":")
axA.set_xlabel("Reaction coordinate")
axA.set_ylabel("Relative energy (meV)")
axA.set_title(r"(a) Pyrite 1$\times$1$\times$1 (12 atoms) — cross-code anchor")
axA.legend(loc="lower center", frameon=False, ncol=1)
axA.set_xlim(-0.02, 1.02)

# ---- Panel B: 96at production ----
rc_prod = normalize_rc(len(prod_E))
prod_meV = prod_E * 1000.0
axB.plot(rc_prod, prod_meV,
         color="#9467bd",
         marker="D",
         markersize=5,
         linewidth=1.7,
         label=f"QE 96at prod NEB:\n$E_a$ = {prod_meta['E_a_eV']*1000:.1f} meV (paper-grade)")

# Annotate saddle point
i_max = int(np.argmax(prod_meV))
axB.annotate(f"{prod_meV[i_max]:.0f} meV",
             xy=(rc_prod[i_max], prod_meV[i_max]),
             xytext=(rc_prod[i_max] + 0.15, prod_meV[i_max] + 4),
             fontsize=8,
             arrowprops=dict(arrowstyle="->", color="grey", lw=0.5))

axB.axhline(0, color="grey", linewidth=0.5, linestyle=":")
axB.set_xlabel("Reaction coordinate")
axB.set_ylabel("Relative energy (meV)")
axB.set_title(r"(b) Pyrite conv$\times$2$\times$2$\times$2 (96 atoms) — production NEB")
axB.legend(loc="upper right", frameon=False)
axB.set_xlim(-0.02, 1.02)

# Suptitle (small, paper-grade)
# fig.suptitle("Pyrite vacancy-anchored proton migration: cross-code agreement and 96-atom convergence", fontsize=10)

# Save
for ext in ["png", "pdf", "svg"]:
    fig.savefig(FIGDIR / f"fig1_pyrite_anchor.{ext}", bbox_inches="tight", dpi=300)
    print(f"  Saved: {FIGDIR / f'fig1_pyrite_anchor.{ext}'}")

plt.close(fig)
print("\nDone.")
