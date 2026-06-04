# Harvested result data

Each JSON holds the energies/frequencies/magnetizations behind a table or figure in the paper.
All DFT is PBE, U = 0, ONCV norm-conserving pseudopotentials, Quantum ESPRESSO 7.5. Raw wavefunction /
charge-density dumps are **not** archived (multi-GB, regenerable from the `scripts/` inputs).

## DFT barriers (`data/`)

| File | Paper number | Notes |
|---|---|---|
| `marc_VFe_warm_neb_208meV.json` | Marcasite V_Fe + S–H, **208 meV** (Table 1/2, §3.5.2) | endpoint-relaxed warm CI-NEB; nspin=2, `tot_magnetization` 1.1 pin, single magnetic sheet M_abs≈2.5; band + per-image M. |
| `marc_VFe_conv_scan_Q1.json` | Marcasite Q1 convergence (SI §M.4) | k/ecut/smearing single-points on the relaxed endA+saddle: Δk(2,2,3→3,3,4)=+0.4, Δecut(60→80)=−1.7 meV; mv vs gaussian sheet term ≈15 meV. |
| `greigite_VFe_convergence_Q1.json` | Greigite Q1 convergence (SI §K.7) | U=0 k/ecut: Δk(2,2,2→3,3,3)=0.0, Δecut(80→100)=+0.3 meV. Baseline E_a≈1.84 eV. |
| `greigite_VFe_Uscan_Q2.json` | Greigite Q2 U-sensitivity (SI §K.7) | warm-start chain U=1/2/3 eV → E_a = 1.815/1.806/1.478 eV (all ≫ forbidden threshold). U=3 saddle has a residual occupation-basin offset (see SI). |
| `pyrite_VFe_dimer_268meV.json` | Pyrite V_Fe + S–H, **268 meV** (Table 1, §3.5.5) | unconstrained ASE-Dimer (the band-NEB fails at the m-3̄ degenerate saddle). |

## Zero-point frequencies (`data/zpe/`)

Partial-Hessian (9–10-atom reactive subsystem) saddle + endpoint frequency runs; ΔZPE‡ in Table 2 / SI §N.

| File | Mineral / state |
|---|---|
| `pyrite_saddle_freq.json` / `pyrite_endA_freq.json` | Pyrite V_Fe (ΔZPE‡ = −94 meV; 268→173 meV) |
| `mackinawite_saddle_freq.json` / `mackinawite_endA_freq.json` | Mackinawite V_Fe (ΔZPE‡ = −66 meV; 43→≈0) |
| `marcasite_saddle_freq.json` / `marcasite_endA_freq.json` | Marcasite V_Fe (ΔZPE‡ = −85 meV; 208→123 meV; same M_abs≈2.5 sheet as the barrier) |

## Provenance / omissions

- The **mackinawite V_Fe 42.9 meV** electronic barrier and the **pyrite V_S₂ 94.6 meV** dimer value are quoted
  in the paper from earlier production runs; their canonical NEB band data are summarised in the SI rather than
  re-archived here as standalone JSON (the `scripts/` drivers regenerate them).
- Foundation-MLIP topology-scan / band data (MACE-MP-0, CHGNet) are produced by the `mlip/` drivers; the CHGNet
  magnitude-only magmom control (mackinawite V_Fe, AFM-init vs zero → 0.0-pp shift, §4.1) is reproduced by
  `mlip/compare_chgnet_default_vs_none.py`.
- Marcasite mv-branch k(3,3,4) transferability single-point (`scripts/marc_mv_k334_sp.py`) result is added on
  completion.
