# DFT NEB Structure Dataset — Iron-Sulfide Defect Diffusion

Extended-XYZ files of converged DFT NEB bands and relaxed endpoint/saddle structures
for proton/vacancy diffusion in Fe-S minerals. All DFT: PBE functional, U=0 (no Hubbard
correction), ONCV pseudopotentials, Quantum ESPRESSO 7.x.

Units: energies in **eV**, forces in **eV/Å**, positions in **Å**.

Per-frame info fields (in every extxyz):
- `mineral` — mineral name
- `rc` — reaction coordinate [0..1], endA=0, endB=1
- `image_index` — NEB image index (0-based)
- `U_eV` — Hubbard U (0.0 for all files here)
- `nspin` — number of spin channels used in DFT
- `source_file` — relative path to original source file
- `method` — DFT functional or MLIP label

---

## Files

### Marcasite (FeS₂, Pnnm)

**`marcasite_VFe_band.extxyz`**
- Mineral: marcasite (orthorhombic FeS₂, space group Pnnm)
- Reaction: V_Fe iron-vacancy S→S proton hop, Fe₃₁S₆₄H₁ (96 atoms)
- Method: PBE U=0, nspin=2, ONCV, kpts 2×2×2, QE 7.x, warm-restart NEB
- n_atoms per frame: 96
- n_frames: 9 (endA, 7 intermediate, endB)
- Barrier: **208.2 meV** (image index 4 = saddle)
- Energies+forces: present for all frames (from DFT NEB calculator)
- Note: image index 2 carries `energy_stale_dyneb_skip=True` — a CRASH file
  was found in that image directory; the DyNEB optimizer skipped this image
  and its energy may not be fully converged. Geometry is valid.
- Source: `results/dft_datasets/2026-06-03/marc_warm_neb_W2/prod_essentials/neb_warm.traj`

**`marcasite_endA.extxyz`** / **`marcasite_endB.extxyz`**
- DFT-relaxed endpoints for the marcasite V_Fe hop (96 atoms each)
- Geometry: dft_relaxed (eigenvalues stored in file; no total energy in xyz header — these
  are GPAW/QE geometry-only outputs; endpoint energies recoverable from the NEB band frames)
- Source: `relaxed_endA_warm.xyz`, `relaxed_endB_warm.xyz` (same directory as band)

---

### Greigite (Fe₃S₄, Fd-3m)

**`greigite_VFe_band.extxyz`**
- Mineral: greigite (cubic, space group Fd-3m, inverse spinel)
- Reaction: V_Fe iron-vacancy S→S proton hop, Fe₂₃S₃₂H₁ (56 atoms)
- Method: PBE U=0, nspin=2 (ferrimagnetic), ONCV, kpts 2×2×2, QE 7.x
- n_atoms per frame: 56
- n_frames: 9
- Barrier: **1860.7 meV** (symmetric band; endA ≈ endB energy)
- Energies+forces: present for all frames
- Note: symmetric band (endA ≈ endB within 0.1 meV); high barrier reflects
  S→S hop geometry in the spinel V_Fe pocket
- Source: `results/dft_datasets/2026-05-27/w2_greigite_full_neb/greig_neb_full_s150/neb.traj`

**`greigite_endA.extxyz`** / **`greigite_endB.extxyz`**
- DFT-relaxed ferrimagnetic endpoints (56 atoms each)
- Has energy+forces from DFT singlepoint
- Source: `relaxed_endA.xyz`, `relaxed_endB.xyz` (same directory as band)

---

### Pyrite (FeS₂, Pa-3) — V_S2 dimer hop

**`pyrite_VS2_band.extxyz`**
- Mineral: pyrite (cubic, space group Pa-3)
- Reaction: S₂ dimer vacancy hop (V_S2), FeS₂ 96 atoms, nspin=1 (non-magnetic)
- Method: PBE U=0, nspin=1, ONCV, kpts 2×2×2, QE 7.x
- n_atoms per frame: 96
- n_frames: 9
- Barrier: **94.6 meV** (image index 4 = saddle); endA ≈ endB within 0.2 meV
- Energies+forces: present for all frames
- Note: the "hero anchor" for MLIP benchmarking — near-symmetric band with
  clean convergence; this is the primary DFT reference for MLIP comparison
- Source: `results/dft_datasets/2026-05-01/pyr_prod_neb_W3/prod_essentials/neb.traj`

**`pyrite_VS2_endA.extxyz`** / **`pyrite_VS2_endB.extxyz`**
- DFT-relaxed endpoints (96 atoms each), has energy+forces
- Source: `relaxed_endA.xyz`, `relaxed_endB.xyz` (same directory)

---

### Pyrite (FeS₂, Pa-3) — V_Fe + S-H dimer saddle

**`pyrite_VFe_dimer_saddle.extxyz`**
- Mineral: pyrite (cubic, space group Pa-3)
- Reaction: V_Fe iron-vacancy S-H transfer, FeS₂ + H (96 atoms), nspin=1
- Method: PBE U=0, nspin=1, ONCV, kpts 2×2×2, QE 7.x, unconstrained ASE Dimer method
- n_atoms: 96
- n_frames: 1 (saddle geometry only)
- Barrier: **268 meV** (paper value, E_saddle − E_endA; endA ref = −124947.942 eV)
- Energies+forces: **NOT stored in this file** — the absolute DFT energy was not
  written to the geometry file. The barrier is known from the dimer run result JSON.
  The geometry is the converged dimer saddle (fmax = 0.0165 eV/Å, converged=True).
- Note: geometry_only; for MLIP benchmarking use as saddle seed for frequency
  or constrained relaxation; absolute energy reconstruction requires a DFT singlepoint
- Source: `results/dft_datasets/2026-05-31/pyr_v6_dimer_complete/results/v6_saddle_final.xyz`

---

### Mackinawite (FeS, P4/nmm)

**`mackinawite_VFe_band.extxyz`**
- Mineral: mackinawite (tetragonal, space group P4/nmm)
- Reaction: V_Fe iron-vacancy S-H transfer, Fe₃₁S₄₀H₁ (72 atoms), nspin=1
- Method: PBE U=0, nspin=1, ONCV, QE 7.x
- n_atoms per frame: 72
- n_frames: 9
- Barrier: **42.9 meV** (image index 4 = saddle); symmetric near-zero asymmetry
- Energies+forces: present for all frames
- Note: the source directory is labeled "aborted" (the run was stopped after convergence);
  the band is physically complete — the saddle is well-defined and energies match the
  paper value (42.9 meV verified against expected). The run reached a fully converged
  NEB band before being stopped.
- Source: `results/dft_datasets/2026-05-03/mack_vfe_w3_aborted_2026-05-03/neb_canonical_mack_72at_qe_VFe/neb.traj`

**`mackinawite_endA.extxyz`** / **`mackinawite_endB.extxyz`**
- DFT-relaxed endpoints (72 atoms each), has energy+forces
- Source: `relaxed_endA.xyz`, `relaxed_endB.xyz` (same directory as band)

---

### MLIP Reference Bands (pyrite V_Fe)

These files contain geometries produced by MLIP NEB (MACE-MP-0 and CHGNet),
for direct comparison with DFT NEB bands (pyrite V_Fe reaction coordinate).

**`pyrite_VFe_band_MACE-MP-0.extxyz`**
- Method: MACE-MP-0 (universal MLIP, Materials Project 2023)
- Mineral: pyrite (Pa-3, 96 atoms, nspin=1 in DFT context)
- n_frames: 9 (MACE NEB band images)
- Energies: present (MACE-MP-0 total energies, eV); forces present
- Band (meV relative to endA): 0, -58, -20, +82, +182, +113, 0, -55, 0
- Max: 181.7 meV (vs DFT ~268 meV for V_Fe; note: different reaction — see note)
- Note: `method=MACE-MP-0`; pyrite V_Fe path, same reaction coordinate
  as DFT dimer saddle. MACE underestimates and shifts the barrier.
- Source: `results/dft_datasets/2026-05-29/pyr_vfe_mlip_band/pyr_vfe_mlip_band_mace.xyz`

**`pyrite_VFe_band_CHGNet.extxyz`**
- Method: CHGNet (universal MLIP, Deng et al. 2023)
- Mineral: pyrite (Pa-3, 96 atoms)
- n_frames: 9 (CHGNet NEB band images, geometry only)
- Energies: **NOT stored** — all frames in the source file carry an identical
  energy value (−599.374 eV), which is a known artifact of how this file was
  produced. Geometries are valid CHGNet NEB images.
- Forces: present in source (stored in arrays)
- Note: `method=CHGNet`, `note=geometry_only`; use geometries only, not energies
- Source: `results/dft_datasets/2026-05-29/pyr_vfe_mlip_band/pyr_vfe_mlip_band_chgnet.xyz`

---

## Who Can Use This and How

### (a) MLIP training and benchmarking

Complete DFT NEB bands with per-image energy and forces are rare in public datasets.
Most MLIP training sets contain only stable configurations; transition-state frames and
NEB intermediate geometries are systematically underrepresented.

These files can be used as:
- Transition-state reference structures for benchmarking universal MLIPs (MACE-MP-0,
  CHGNet, M3GNet, SevenNet, etc.) on Fe-S defect diffusion barriers
- Fine-tuning seed data: the 9-image NEB bands (with energies+forces) can be directly
  appended to MLIP training sets to improve TS coverage in iron-sulfide chemistry
- Evaluation of whether an MLIP reproduces barrier heights vs DFT at the PBE U=0 level

The pyrite VS2 band (94.6 meV) and mackinawite V_Fe band (42.9 meV) are particularly
useful — clean, symmetric, fully converged, with energies and forces on every image.

### (b) NEB method developers

Real converged DFT NEB bands to test path optimizers, climbing image algorithms, or
adaptive NEB schemes. The bands span a 40× range of barrier heights (42.9 meV to
1860.7 meV), with different symmetries (symmetric / asymmetric) and magnetic complexity
(nspin=1 / nspin=2 ferrimagnetic). The marcasite band has a stale intermediate image
(image_02, flagged) that can serve as a test for NEB robustness.

### (c) Fe-S defect and diffusion researchers

Relaxed endpoint geometries and saddle structures to build on for:
- Computing diffusion coefficients or rates via harmonic transition state theory
- Running NEB on a different level of theory (GGA+U, hybrid, etc.) starting from
  these geometries
- Setting up AIMD or metadynamics runs near the saddle

The pyrite V_Fe dimer saddle (268 meV, fmax=0.016 eV/Å) is a converged saddle-point
geometry that can be re-used as input to frequency calculations or as a starting point
for constrained relaxations.

### (d) DFT reproducers

All geometries were produced with Quantum ESPRESSO 7.x, PBE functional, ONCV
pseudopotentials (SG15 library), kpts 2×2×2 Monkhorst-Pack (Gamma-centered),
smearing gaussian/Marzari-Vanderbilt, plane-wave cutoffs 60/480 Ry. Geometries
in extxyz format are readable by ASE and OVITO. The `source_file` info field
points to the original traj/xyz in the project data archive.

---

## Notes on Missing or Flagged Data

- **marcasite image_02**: `energy_stale_dyneb_skip=True` — CRASH file found in
  image directory; the DyNEB optimizer deprioritized this image. Geometry is from
  the converged NEB band; energy reflects a partial SCF step.
- **pyrite_VFe_dimer_saddle**: geometry only, no absolute energy in file.
  Barrier (268 meV) is from the dimer calculation result (stored separately as JSON).
- **pyrite_VFe_band_CHGNet**: geometry only (energies corrupted in source file, identical
  across all frames). Forces from MLIP are stored in arrays but energies should not be used.


## Pentlandite (Fe72S64 pure-Fe model, Fm-3m, 136-atom) -- endpoints only

`pentlandite_endA.extxyz`, `pentlandite_endB.extxyz` -- DFT-relaxed V_Fe + S-H endpoints
(Fe71S64H1, nspin=1, PBE U=0). **No NEB band / barrier is provided**: the production
V_Fe + S-H NEB requires a spin-polarized (nspin=2) protocol and is deferred to a companion
magnetic-framework study. In Paper #1, pentlandite serves only the two foundation-MLIP
structural-motif failure diagnostics (V_S+H Fe-cluster collapse; [Fe4S4] cubane 3+3 collapse).
Earlier V_S+H values (foundation-MLIP 1.43 eV; ABACUS LCAO 0.55 eV) are NOT reliable barriers
under the corrected methodology. Natural pentlandite is (Fe,Ni)9S8; this is the pure-Fe endmember.
