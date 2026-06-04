"""
Figure 3 — Pentlandite foundation-MLIP cubane failure mode.

⚠️ NEEDS REDRAW (2026-06-01 pentlandite scope change). The V_Fe + S–H DFT
endpoint demonstration (old panels a/b) was REMOVED from Paper #1 and relocated
to the companion MagNEB study (nspin=2 rationale). Figure 3 is now purely the
cubane structural-motif MLIP failure (manuscript §3.3.2 caption). Target panels:
(a) the two Fe environments — octahedral Fe (Wyckoff 4b, 6-coord) vs cubane Fe
    (32f, 4-coord corners of [Fe₄S₄]);
(b) V_Fe at cubane Fe under MACE/CHGNet → 3+3 distortion (3 short ≈2.34 Å,
    3 long ≈3.57 Å), cube destroyed, several eV above symmetric DFT minimum;
(c) V_Fe at octahedral Fe → preserved cleanly.
The old endpoint-A/B rendering below is retained only as a code reference for the
ASE ball-and-stick machinery; the endpoint geometries themselves now live in
paper/MagNEB_Preflight/. Regenerate before Paper #1 submission.
Per main-text §3.3.2.
Output: fig3_pent_endpoints.{pdf, png, svg}
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from ase.visualize.plot import plot_atoms
ENDPOINT_A_PATH = Path("results/dft_datasets/2026-05-11/w3_pent_v3_mirror_endB/relaxed_endA.xyz")
ENDPOINT_B_PATH = Path("results/dft_datasets/2026-05-11/w3_pent_v3_mirror_endB/relaxed_endB.xyz")
# V_Fe site index per SI §G.1
V_FE_INDEX = 119 # not present in defect cell; reference for geometry
# H atom in 136-atom defect cell is the last atom
S_I = 98 # endpoint A anchor
S_K = 62 # endpoint B anchor
def find_h_atom(atoms):
 """Find H atom index in the structure."""
 for i, sym in enumerate(atoms.get_chemical_symbols):
 if sym == "H":
 return i
 raise ValueError("No H atom found in structure")
def find_v_fe_neighbours(atoms, v_fe_pos, rcut=3.0):
 """Find S atoms within rcut of the V_Fe site (4 S neighbours expected for octahedral Fe; full octa = 6 S)."""
 syms = atoms.get_chemical_symbols
 pos = atoms.get_positions
 cell = atoms.cell
 s_neighbours =
 for i, sym in enumerate(syms):
 if sym != "S":
 continue
 # Apply MIC for distance
 diff = pos[i] - v_fe_pos
 # crude MIC: assume orthorhombic-ish cell (good enough for visualisation)
 for j in range(3):
 cell_len = np.linalg.norm(cell[j])
 while diff[j] > cell_len / 2:
 diff[j] -= cell_len
 while diff[j] < -cell_len / 2:
 diff[j] += cell_len
 d = np.linalg.norm(diff)
 if d < rcut:
 s_neighbours.append((i, d, diff))
 return sorted(s_neighbours, key=lambda x: x[1])
def make_figure(out_dir: Path) -> None:
 out_dir.mkdir(parents=True, exist_ok=True)
 try:
 atoms_a = read(ENDPOINT_A_PATH)
 atoms_b = read(ENDPOINT_B_PATH)
 except Exception as e:
 print(f"WARN: Could not load .xyz files ({e}); falling back to schematic-only figure")
 atoms_a = None
 atoms_b = None
 # 3-panel figure: (a) endA, (b) endB, (c) cubane vs octahedral schematic
 fig = plt.figure(figsize=(11.0, 5.0))
 gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1], wspace=0.15)
 ax_a = fig.add_subplot(gs[0, 0])
 ax_b = fig.add_subplot(gs[0, 1])
 ax_c = fig.add_subplot(gs[0, 2])
 # ---------- Panel (a) Endpoint A ----------
 if atoms_a is not None:
 # Identify H atom + nearest S to centre the view
 h_idx_a = find_h_atom(atoms_a)
 h_pos_a = atoms_a.get_positions[h_idx_a]
 # Crop a 6 Å radius region around H for visibility
 # We'll just plot the full structure; ASE will handle PBC
 plot_atoms(
 atoms_a,
 ax_a,
 radii=0.35,
 rotation="0x,0y,0z",
 show_unit_cell=2,
 )
 # Add annotation pointing to H atom
 # Project H position to 2D (default rotation = xy-plane projection)
 ax_a.annotate(
 f"H@S_i\n(d_H-S = 1.41 Å)",
 xy=(h_pos_a[0], h_pos_a[1]),
 xytext=(h_pos_a[0] + 3, h_pos_a[1] + 3),
 fontsize=9,
 color="darkred",
 arrowprops=dict(arrowstyle="->", color="darkred", linewidth=1.0),
 bbox=dict(facecolor="white", edgecolor="darkred", alpha=0.85, boxstyle="round,pad=0.3"),
 )
 ax_a.set_title(
 "(a) Endpoint A — paper-grade\n141 BFGS iters, fmax = 0.026 eV/Å",
 fontsize=10,
 fontweight="bold",
 )
 else:
 ax_a.text(0.5, 0.5, "Endpoint A\n(structure load failed)", ha="center", va="center", fontsize=12, transform=ax_a.transAxes)
 ax_a.set_title("(a) Endpoint A", fontsize=10)
 ax_a.set_xticks
 ax_a.set_yticks
 # ---------- Panel (b) Endpoint B ----------
 if atoms_b is not None:
 h_idx_b = find_h_atom(atoms_b)
 h_pos_b = atoms_b.get_positions[h_idx_b]
 plot_atoms(
 atoms_b,
 ax_b,
 radii=0.35,
 rotation="0x,0y,0z",
 show_unit_cell=2,
 )
 ax_b.annotate(
 f"H@S_k\n(mirror construction)",
 xy=(h_pos_b[0], h_pos_b[1]),
 xytext=(h_pos_b[0] + 3, h_pos_b[1] + 3),
 fontsize=9,
 color="darkred",
 arrowprops=dict(arrowstyle="->", color="darkred", linewidth=1.0),
 bbox=dict(facecolor="white", edgecolor="darkred", alpha=0.85, boxstyle="round,pad=0.3"),
 )
 ax_b.set_title(
 "(b) Endpoint B — smoke-level\n20 BFGS iters, fmax = 1.77 eV/Å (SI §G.2)",
 fontsize=10,
 fontweight="bold",
 )
 else:
 ax_b.text(0.5, 0.5, "Endpoint B\n(structure load failed)", ha="center", va="center", fontsize=12, transform=ax_b.transAxes)
 ax_b.set_title("(b) Endpoint B", fontsize=10)
 ax_b.set_xticks
 ax_b.set_yticks
 # ---------- Panel (c) Cubane vs octahedral Fe schematic ----------
 # Draw a simple 2D schematic:
 # Top: cubane Fe4S4 motif (4 Fe at corners, 4 S at faces of a cube)
 # Bottom: octahedral Fe (1 Fe centre, 6 S vertices)
 # With V_Fe annotation: cubane V_Fe → 3+3 distortion (MLIP fail); octahedral V_Fe → preserved (OK)
 # Cubane motif (top half)
 cube_centre = np.array([0.5, 0.7])
 cube_size = 0.18
 fe_positions = np.array(
 [
 [-1, -1, -1],
 [+1, -1, -1],
 [-1, +1, -1],
 [+1, +1, -1],
 [-1, -1, +1],
 [+1, -1, +1],
 [-1, +1, +1],
 [+1, +1, +1],
 ]
 )
 # Project to 2D (perspective view)
 for fe in fe_positions:
 # tiny perspective shift on z
 x = cube_centre[0] + cube_size * (fe[0] + 0.15 * fe[2])
 y = cube_centre[1] + cube_size * (fe[1] + 0.15 * fe[2])
 # In cubane structure: 4 Fe + 4 S alternate; here we colour-code by index parity
 is_fe = fe[0] * fe[1] * fe[2] > 0 if any(fe == -1) else True
 color = "darkorange" if (fe[0] + fe[1] + fe[2]) % 4 == 1 else "gold" # Fe = orange, S = yellow
 ax_c.add_patch(plt.Circle((x, y), 0.025, color=color, ec="black", zorder=3))
 # Annotate
 ax_c.text(
 0.5,
 0.95,
 "[Fe₄S₄] cubane Fe (Wyckoff 32f)",
 ha="center",
 va="center",
 fontsize=9,
 fontweight="bold",
 )
 ax_c.text(
 0.5,
 0.47,
 "V_Fe → MLIP 3+3 distortion\n(structural-motif failure, §3.3.2)",
 ha="center",
 va="center",
 fontsize=8,
 color="red",
 bbox=dict(facecolor="white", edgecolor="red", alpha=0.85, boxstyle="round,pad=0.3"),
 )
 # Octahedral motif (bottom half)
 oct_centre = np.array([0.5, 0.20])
 oct_size = 0.10
 # 1 Fe centre + 6 S at octahedron vertices
 ax_c.add_patch(plt.Circle(oct_centre, 0.025, color="darkorange", ec="black", zorder=3))
 # 6 vertices of octahedron projected to 2D
 oct_vertices = np.array(
 [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
 )
 for v in oct_vertices:
 x = oct_centre[0] + oct_size * (v[0] + 0.15 * v[2])
 y = oct_centre[1] + oct_size * (v[1] + 0.15 * v[2])
 ax_c.add_patch(plt.Circle((x, y), 0.022, color="gold", ec="black", zorder=3))
 # Draw bond
 ax_c.plot([oct_centre[0], x], [oct_centre[1], y], color="grey", linewidth=0.8, zorder=2)
 ax_c.text(
 0.5,
 0.05,
 "Octahedral Fe (Wyckoff 4b/4a) — PRESERVED ✓",
 ha="center",
 va="center",
 fontsize=9,
 color="darkgreen",
 fontweight="bold",
 )
 ax_c.set_xlim(-0.05, 1.05)
 ax_c.set_ylim(-0.05, 1.05)
 ax_c.set_xticks
 ax_c.set_yticks
 ax_c.set_aspect("equal")
 ax_c.set_title(
 "(c) Pent Fe environments\ncubane (32f) vs octahedral (4b/4a)",
 fontsize=10,
 fontweight="bold",
 )
 # Legend for panel c
 fe_patch = plt.Circle((0, 0), 1, color="darkorange", ec="black", label="Fe")
 s_patch = plt.Circle((0, 0), 1, color="gold", ec="black", label="S")
 ax_c.legend(
 [fe_patch, s_patch],
 ["Fe", "S"],
 loc="upper right",
 fontsize=8,
 framealpha=0.9,
 )
 fig.suptitle(
 "Figure 3 — Pentlandite V_Fe + S–H endpoint geometries + cubane-vs-octahedral Fe schematic",
 fontsize=11,
 y=1.02,
 )
 fig.tight_layout
 fig.savefig(out_dir / "fig3_pent_endpoints.pdf", bbox_inches="tight")
 fig.savefig(out_dir / "fig3_pent_endpoints.png", dpi=200, bbox_inches="tight")
 fig.savefig(out_dir / "fig3_pent_endpoints.svg", bbox_inches="tight")
 plt.close(fig)
 print(f"Wrote 3 files (pdf+png+svg) to {out_dir}/")
if __name__ == "__main__":
 out_dir = Path(__file__).parent
 make_figure(out_dir)
