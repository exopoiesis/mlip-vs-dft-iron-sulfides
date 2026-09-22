# MLIP vs DFT for iron-sulfide defect kinetics

Code and data accompanying the manuscript:

> **Where Foundation Machine-Learning Potentials Fail for Iron-Sulfide Defect Kinetics: Failure Modes,
> Pre-Flight Checks, and a DFT Benchmark of Vacancy-Anchored Hydrogen Migration in Pyrite, Marcasite,
> Mackinawite, and Greigite**
>
> Igor N. Morozov. Independent Researcher, Ukraine.
> ORCID: [0009-0007-3863-1747](https://orcid.org/0009-0007-3863-1747) · igor@exopoiesis.space · exopoiesis.space

> ### ⚠️ Corrections in release v1.1 and later — read before using any greigite number
>
> **The greigite barrier of 1.86 eV published in v1.0 is retracted.** That NEB band ran along a *trans axis*
> of the vacancy octahedron, and the midpoint of a trans axis is the vacant Fe site itself — a special
> position where the force vanishes **by symmetry rather than by convergence**. The band therefore satisfied
> any force criterion it was given, and its midpoint carries **three** imaginary modes, so it is not a
> first-order saddle. The value is the energy of a proton at the interstitial vacant site, not a migration
> barrier. It is superseded by **235.97 meV** (`greigite_VFe_channel_band.extxyz`) and **566.83 meV**
> (`greigite_VFe_cation_band.extxyz`). The retracted band is **kept, not deleted** — it may already have been
> cited — and every frame carries `meta_RETRACTED`, as do the two JSON files derived from it.
>
> Two further labels were corrected: the pyrite anchor is **V_S with an Fe-bound hydride hop**, not a
> "V_S₂ dimer hop" or an S–H transfer (the cell removes one sulfur and the hydrogen is Fe-bound in all nine
> images); and the marcasite unreliable-energy flag moved from image 2 to **image 1**. Neither correction
> changes a barrier. Full detail in `data/structures/README.md` and `data/README.md`.

We benchmark foundation machine-learning interatomic potentials, zero-shot, against plane-wave DFT (Quantum
ESPRESSO) for **vacancy-anchored proton (neutral H⁰ proxy) migration** across four iron sulfides spanning the
natural diagenetic series, plus a pentlandite structural-motif diagnostic. The core comparison uses
**MACE-MP-0 (large)** and **CHGNet-v0.3.0**; a second pass extends it to **nine checkpoints across five DFT
bands** (45 model×band measurements). We establish a unified PBE (U = 0) DFT reference landscape, report the
first harmonic zero-point corrections for *bulk* vacancy-anchored proton migration in iron sulfides, and
document three distinct, diagnosable foundation-MLIP failure modes.

## Headline results (DFT, PBE U = 0)

| Mineral | Reaction coordinate | E_a (electronic) | E_a (ZPE-corr.) | Foundation-MLIP outcome |
|---|---|---|---|---|
| Pyrite (Pa-3̄) | V_S pocket, Fe→Fe hydride hop | 94.6 meV | — | both potentials merge the two Fe minima when endpoints are seeded from sulfur, so no barrier is defined |
| Pyrite (Pa-3̄) | V_Fe + S–H | 268 meV | 173 meV | with own-surface endpoints: MACE within ~12 meV, CHGNet +119 meV |
| Mackinawite (P4/nmm) | V_Fe + S–H | 42.9 meV | ≈ 0 (barrierless) | **all nine** models overestimate the smallest barrier in the series, by 1.75–9.66× |
| Marcasite (Pnnm) | V_Fe + S–H | 208 meV | 123 meV | MACE reproduces barrier and reaction energy inside the reference's own ±15 meV magnetic-sheet uncertainty |
| Greigite (Fd-3̄m), **channel** edge | V_Fe + S–H | **236 meV** | 133 meV | pristine cell collapses under relaxation (S–S → 0.072 Å); on DFT geometries the same models evaluate it normally |
| Greigite (Fd-3̄m), **cation** edge | V_Fe + S–H | **567 meV** | 459 meV | contrast path; **all nine** models rank it above the channel edge |

The two greigite rows are the two symmetry-distinct S–S edge classes of the *same* vacancy octahedron,
computed from a common endpoint. The **shorter** edge (3.369 Å, shared with an occupied cation) carries the
**higher** barrier: by Pauling's third rule a shared edge contracts precisely because a cation sits across it,
and it is expensive for that same reason. Selecting a migration path by hop distance therefore picks the worst
one systematically in this structure type.

(Pentlandite V_Fe barrier requires nspin = 2 and is developed in a companion magnetic-framework study; the
[Fe₄S₄] cubane structural-motif MLIP failure is documented here.)

## Repository layout

```
scripts/   QE DFT/NEB + dimer + convergence single-point drivers (ASE-driven), plus per_image_stats.py
zpe/       partial-Hessian zero-point frequency drivers (saddle + endpoint)
mlip/      MACE-MP-0 / CHGNet topology-scan, NEB-probe, and band-correlation drivers
mlip/r22_2026-09/    nine-model x five-band benchmark: per-image profiles, forces, barrier matrix
mlip/environments/   three pip freezes (the nine models do not fit in one environment) + weight provenance
figures/   figure-generation scripts (Table 1 / ZPE trend / barrier landscape / band correlation)
data/      harvested result JSON (the numbers behind every table and figure)
data/structures/  complete DFT NEB bands + relaxed endpoints/saddles as extended-XYZ (energy+forces)
tm-spec/   TM-Spec v0.3 declarative records of each result (machine-readable + format examples)
```

**Reusable structure data.** `data/structures/` ships the *complete* DFT NEB bands (all images, with energies
and forces) plus relaxed endpoints and saddle geometries as extended-XYZ — the kind of complete iron-sulfide
defect-migration reference that is otherwise hard to find. Useful as MLIP training/benchmark reference data,
NEB-method test cases, or starting geometries for further Fe–S defect studies. See `data/structures/README.md`.

**Three environments, not one.** The nine checkpoints cannot be installed together: orb-models 0.7.0 removed
its ASE calculator (we used 0.5.5, which pins its own torch), and GRACE runs on TensorFlow. All three
`pip freeze` files are deposited verbatim in `mlip/environments/`, with weight provenance and checksums.

**TM-Spec records.** `tm-spec/` holds [TM-Spec v0.3](https://github.com/exopoiesis/tm-spec) declarative records
of every result (successes *and* failures), both as a validate-able provenance trail and as worked examples of
the format. See `tm-spec/README.md`.

## Methods (summary)

- **DFT:** Quantum ESPRESSO PWSCF 7.5 (GPU), ONCV PBE norm-conserving pseudopotentials (Fe, S, H), **U = 0**
  (a single unified protocol across all minerals). nspin = 1 for diamagnetic pyrite / non-magnetic mackinawite;
  nspin = 2 (cold/`mv` smearing, `local-TF` mixing, `tot_magnetization` pinned) for the magnetic marcasite and
  ferrimagnetic greigite V_Fe defect cells. The U = 0 label is verifiable from the deposit rather than taken on
  trust: no HUBBARD card appears in any production input, and no Hubbard energy term in any output.
- **NEB:** 9-image CI-NEB, FIRE + DyNEB, with the **ASE IDPP minimum-image-convention prewrap** fix
  (ASE GitLab issue #1130 → upstream merge request **MR !4091**, commit `8d9b69bf5`). Pyrite V_Fe used an
  unconstrained ASE-Dimer search (the band-NEB fails at the m-3̄ degenerate saddle).
- **Saddle verification (greigite):** Lanczos on the mass-weighted Hessian via finite-difference
  Hessian-vector products over the **full 168 degrees of freedom** — one negative eigenvalue,
  ν‡ = 829.6i cm⁻¹ at effective mass 1.054 amu, i.e. a purely protonic unstable mode, and zero imaginary
  modes at the endpoint. Data in `data/greigite_VFe_saddle_lanczos_2026-09-21.json`.
- **ZPE:** harmonic partial-Hessian on a 9–10-atom reactive subsystem; ΔZPE‡ = ZPE_saddle − ZPE_endpoint on an
  identical atom set so distant modes cancel; Wigner tunneling correction from the imaginary saddle frequency
  (for greigite, a numerical symmetric-Eckart correction: κ = 2.02 at 298 K, crossover T_c = 187 K).
- **MLIP:** the core comparison is MACE-MP-0 (**large** checkpoint, `MACE_MPtrj_2022.9.model`) and
  CHGNet-v0.3.0, zero-shot (pretrained), via ASE. CHGNet uses a magnitude-only magnetic representation
  (Jiang/Xu, *PNAS* 2025); initialized magmoms do not enter its energy prediction (verified control included).
  The nine-model extension adds SevenNet 7net-0, UMA-s-1p2, Orb-v2, Orb-v3-conservative-inf-omat, and
  GRACE-1L-OMAT / 2L-OMAT / 2L-OAM, evaluated as **single points on the converged DFT geometries**. That
  protocol removes the optimizer as a confounder but is biased upward by construction, since a barrier along a
  prescribed path is at least the barrier along the model's own minimum-energy path.

## Energy convention

Barriers are differences of the free energy F (the `!` line of the Quantum ESPRESSO output). Rebuilding the
greigite channel barrier from the internal energy E = F + TS, or from the 0 K extrapolation, gives 227.89 and
231.90 meV instead of 235.91 — the choice moves the number by up to 8 meV. Absolute totals additionally differ
by ≈ 7 meV between ASE versions through the CODATA value of the Rydberg; this cancels in every difference.

## Reproducing

```bash
pip install -r requirements.txt
```

The DFT drivers expect Quantum ESPRESSO 7.5 (`pw.x`, GPU build) on `PATH` and ONCV PBE pseudopotentials for
Fe, S, H (standard QE/PseudoDojo distribution; place them where `ESPRESSO_PSEUDO` points and map them as
`Fe.upf`, `S.upf`, `H.upf`). MLIP drivers require `mace-torch` and `chgnet`; for the nine-model extension use
the three environment specifications in `mlip/environments/` rather than `requirements.txt`. Figure scripts
require only `matplotlib` + `numpy` and read from `data/`. Raw DFT wavefunction/charge-density dumps are
**not** archived (multi-GB, regenerable from the provided inputs); `data/` holds the harvested
energies/frequencies behind the paper's tables and figures.

## Citing

See `CITATION.cff`. Archived release: Zenodo DOI
[10.5281/zenodo.20540264](https://doi.org/10.5281/zenodo.20540264) (v1.0). **Releases from v1.1 onward carry
the corrections described at the top of this file; cite the release that matches the numbers you use.**

## License

Code: MIT (`LICENSE`). Data (`data/`): CC-BY-4.0.

## Acknowledgements

Trelis Research; compute on Vast.ai (A100 PCIe 40 GB) and a local RTX 4070 node.
