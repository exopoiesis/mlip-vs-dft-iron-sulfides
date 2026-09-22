# DFT NEB Structure Dataset — Iron-Sulfide Defect Diffusion

Extended-XYZ files of converged DFT NEB bands and relaxed endpoint/saddle structures
for proton/vacancy diffusion in Fe-S minerals. All DFT: PBE functional, U=0 (no Hubbard
correction), ONCV pseudopotentials, Quantum ESPRESSO 7.x.

Units: energies in **eV**, forces in **eV/Å**, positions in **Å**.

Per-frame info fields (in every extxyz) — all provenance keys carry a `meta_` prefix:
- `meta_mineral` — mineral name
- `meta_rc` — reaction coordinate [0..1], endA=0, endB=1
- `meta_image_index` — NEB image index (0-based)
- `meta_U_eV` — Hubbard U (0.0 for all files here)
- `meta_nspin` — number of spin channels used in DFT
- `meta_source_file` — relative path to original source file
- `meta_method` — DFT functional or MLIP label
- plus per-file DFT settings (`meta_nspins`, `meta_nkpts`, `meta_nbands`, `meta_fermi_level`, …)

`energy` (and, where present, `forces` in the `Properties` spec) keep their standard names so that
ASE populates the calculator normally.

> **Header-compatibility note (2026-09-14).** These files previously wrote DFT settings under bare
> names — `nspins`, `nkpts`, `nbands`, `fermi_level`, `kpoint_weights`. Those collide with reserved
> attribute names of ASE's `SinglePointDFTCalculator`: on read, ASE feeds them back as calculator
> *properties* and raises `AssertionError: nspins` (and so on), so **12 of the 17 files could not be
> opened with a plain `ase.io.read`**. The same failure aborted one of our own production NEB runs
> (`neb_canonical_pyr_96at_qe_VFe_v5`, 2026-05-30) at the warm-start step. All provenance keys are now
> prefixed with `meta_`, which cannot collide with any ASE property name. Only key *names* changed —
> positions, forces, energies and every value are bit-identical to the previous release. Verified:
> all 17 files load with `ase.io.read(path, index=":")` under ASE 3.23.0.

```python
from ase.io import read
band = read("marcasite_VFe_band.extxyz", index=":")     # 9 images
print(band[4].info["meta_rc"], band[4].get_potential_energy())
```

---

## Files

### Marcasite (FeS₂, Pnnm)

> ⚠️ **Image 1 carries an unreliable energy — corrected flag, 2026-09-22.**
> The band is non-monotone before the saddle (0 → **151.3** → 42.0 → 104.5 → 208.2 meV) while its
> geometry is smooth. Two independent checks indict **image 1**: dE/ds from the stored forces is
> **+0.134** eV/Å against **+0.685** from the energies, and the d(S–H) stretch from image 0 is only
> **0.0040 Å**, which is ≤10 meV harmonically, not 151.3. Substituting a force-consistent ~30 meV
> gives dE/ds = 0.136 against 0.134 from forces.
>
> The original deposit flagged **image 2** (`energy_stale_dyneb_skip`). That was an off-by-one: one
> bad E₁ corrupts both (E₁−E₀) and (E₂−E₁), which makes the inconsistency surface at image 2.
> Frames now carry `meta_energy_unreliable` on image 1 and a history note on image 2.
>
> **The barrier (208.2 meV = image 4 − image 0) is unaffected.** Per-image analyses are: exclude
> image 1.

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

> ### 🔴 RETRACTED 2026-09-15 — `greigite_VFe_band.extxyz` is not a barrier
>
> The path runs along a **trans axis** of the vacancy octahedron, so its midpoint is the vacant Fe
> site itself: a special position where the force vanishes **by symmetry**. A band therefore
> "converges" at any criterion, and the proton Hessian there carries **three imaginary modes** —
> a third-order saddle. The quoted 1860.7 meV is the energy of an interstitial proton, not a
> migration barrier.
>
> The file is **kept, not deleted**, because it may already have been cited; every frame now
> carries `meta_RETRACTED=True` with the reason. **Do not use it.**
>
> **Superseded by `greigite_VFe_channel_band.extxyz` (235.97 meV).**

**`greigite_VFe_channel_band.extxyz`** ⭐ current
- Mineral: greigite (cubic, Fd-3m, inverse spinel)
- Reaction: V_Fe(16d) + S–H hop along the octahedron edge facing the **empty 16c site**,
  Fe₂₃S₃₂H₁ (56 atoms)
- Method: PBE U=0, nspin=2 (ferrimagnetic A↑↓B), ONCV, kpts 2×2×2, ecutwfc 80 Ry, QE 7.5
- n_frames: 9 · Barrier: **235.97 meV** · E_rxn ≈ 0 (endpoints symmetry-equivalent)
- Saddle verified: exactly one imaginary mode, **829.6i cm⁻¹**, μ_eff 1.054 amu, measured in the
  **full 168 DOF** (Lanczos on the mass-weighted Hessian), zero imaginary modes at the endpoint
- Energies+forces present for all frames
- Source: `results/dft_datasets/2026-09-21_greigite_two_stream/greig_channel/final_0*.xyz`

**`greigite_VFe_cation_band.extxyz`**
- Same cell and method; the octahedron edge **shared with an occupied Fe_oct**
- Barrier: **566.83 meV** — contrast case, 2.40× the channel edge
- The two bands start from the **same** endpoint (energies identical), so the 331 meV gap is a
  pure path difference

⚠️ Energies are **free energies F** (QE's `!` line, gaussian smearing 0.005 Ry). Rebuilding the
barrier with the internal energy E = F+TS or the 0 K extrapolation gives 227.89 / 231.90 meV.

**`greigite_endA.extxyz`** / **`greigite_endB.extxyz`**
- DFT-relaxed ferrimagnetic endpoints (56 atoms each)
- Has energy+forces from DFT singlepoint
- Source: `relaxed_endA.xyz`, `relaxed_endB.xyz` (same directory as band)

---

### Pyrite (FeS₂, Pa-3) — ⚠️ label corrected 2026-09-22

> The filename says `VS2` and the old description said "S₂ dimer hop". **Both are wrong.**
> The cell is **HFe₃₂S₆₃**: one sulfur removed, not a dimer — so this is **V_S**, not V_S₂.
> And the hydrogen is **Fe-bound in all nine frames** (Fe71 → Fe37, d(H–Fe) 1.62–1.97 Å); the
> nearest sulfur is never closer than **2.50 Å**. It is a **hydride hop**, not an S–H transfer.
>
> This was established in the 2026-09-14 revision audit; the correction had not reached this
> deposit. **The barrier value is unaffected** — this file reproduces 94.59 meV. The filename is
> kept for citation stability; `meta_reaction` now reads `V_S_Fe_bound_hydride_hop`.

**`pyrite_VS2_band.extxyz`**
- Mineral: pyrite (cubic, space group Pa-3)
- Reaction: **V_S + Fe-bound hydride hop** (previously mislabelled "S₂ dimer hop, V_S2"),
  HFe₃₂S₆₃, 96 atoms, nspin=1 (non-magnetic)
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
- Reaction: V_Fe iron-vacancy S-H transfer, **Fe₃₅S₃₆H₁** (72 atoms), nspin=1
  <br>⚠️ corrected 2026-09-22 — previously written Fe₃₁S₄₀H₁. The atom count (72) is the same in
  both, so a "how many atoms" check does not catch it; the composition read from the file is
  35 Fe / 36 S / 1 H.
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
