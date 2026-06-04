# MLIP vs DFT for iron-sulfide defect kinetics

Code and data accompanying the manuscript:

> **Where Foundation Machine-Learning Potentials Fail for Iron-Sulfide Defect Kinetics: Four Failure Modes
> and a DFT Benchmark of Vacancy-Anchored Hydrogen Migration in Pyrite, Marcasite, Mackinawite, and Greigite**
>
> Igor N. Morozov. Independent Researcher, Ukraine.
> ORCID: [0009-0007-3863-1747](https://orcid.org/0009-0007-3863-1747) · igor@exopoiesis.space · exopoiesis.space

We benchmark two widely used foundation machine-learning interatomic potentials — **MACE-MP-0 (medium)** and
**CHGNet-v0.3.0**, zero-shot — against plane-wave DFT (Quantum ESPRESSO) for **vacancy-anchored proton (neutral
H⁰ proxy) migration** across four iron sulfides spanning the natural diagenetic series, plus a pentlandite
structural-motif diagnostic. We establish a unified PBE (U = 0) DFT reference landscape, report the first
harmonic zero-point corrections for *bulk* vacancy-anchored proton migration in iron sulfides, and document
four distinct, diagnosable foundation-MLIP failure modes.

## Headline results (DFT, PBE U = 0)

| Mineral | Reaction coordinate | E_a (electronic) | E_a (ZPE-corr.) | Foundation-MLIP outcome |
|---|---|---|---|---|
| Pyrite (Pa-3̄) | V_S₂ dimer hop | 94.6 meV | — | MACE flat (0); CHGNet 27.6 meV, wrong band topology |
| Pyrite (Pa-3̄) | V_Fe + S–H | 268 meV | 173 meV | MACE/CHGNet converge but underbind 1.2–1.5× |
| Mackinawite (P4/nmm) | V_Fe + S–H | 42.9 meV | ≈ 0 (barrierless) | topology-scan basin collapse |
| Marcasite (Pnnm) | V_Fe + S–H | 208 meV | 123 meV | false-positive band-collapse warning |
| Greigite (Fd-3̄m) | V_Fe + S–H | 1.86 eV (robust to U) | — | unphysical / non-convergent; bulk H-migration kinetically forbidden |

(Pentlandite V_Fe barrier requires nspin = 2 and is developed in a companion magnetic-framework study; the
[Fe₄S₄] cubane structural-motif MLIP failure is documented here.)

## Repository layout

```
scripts/   QE DFT/NEB + dimer + convergence/U-scan single-point drivers (ASE-driven)
zpe/       partial-Hessian zero-point frequency drivers (saddle + endpoint)
mlip/      MACE-MP-0 / CHGNet topology-scan, NEB-probe, and band-correlation drivers
figures/   figure-generation scripts (Table 1 / ZPE trend / barrier landscape / band correlation)
data/      harvested result JSON (the numbers behind every table and figure)
```

## Methods (summary)

- **DFT:** Quantum ESPRESSO PWSCF 7.5 (GPU), ONCV PBE norm-conserving pseudopotentials (Fe, S, H), **U = 0**
  (a single unified protocol across all minerals). nspin = 1 for diamagnetic pyrite / non-magnetic mackinawite;
  nspin = 2 (cold/`mv` smearing, `local-TF` mixing, `tot_magnetization` pinned) for the magnetic marcasite and
  ferrimagnetic greigite V_Fe defect cells.
- **NEB:** 9-image CI-NEB, FIRE + DyNEB, with the **ASE IDPP minimum-image-convention prewrap** fix
  (ASE GitLab issue #1130 → upstream merge request **MR !4091**, commit `8d9b69bf5`). Pyrite V_Fe used an
  unconstrained ASE-Dimer search (the band-NEB fails at the m-3̄ degenerate saddle).
- **ZPE:** harmonic partial-Hessian on a 9–10-atom reactive subsystem; ΔZPE‡ = ZPE_saddle − ZPE_endpoint on an
  identical atom set so distant modes cancel; Wigner tunneling correction from the imaginary saddle frequency.
- **MLIP:** MACE-MP-0 (medium) and CHGNet-v0.3.0, zero-shot (pretrained), via ASE. CHGNet uses a
  magnitude-only magnetic representation (Jiang/Xu, *PNAS* 2025); initialized magmoms do not enter its
  energy prediction (verified control included).

## Reproducing

```bash
pip install -r requirements.txt
```

The DFT drivers expect Quantum ESPRESSO 7.5 (`pw.x`, GPU build) on `PATH` and ONCV PBE pseudopotentials for
Fe, S, H (standard QE/PseudoDojo distribution; place them where `ESPRESSO_PSEUDO` points and map them as
`Fe.upf`, `S.upf`, `H.upf`). MLIP drivers require `mace-torch` and `chgnet`. Figure scripts require only
`matplotlib` + `numpy` and read from `data/`. Raw DFT wavefunction/charge-density dumps are **not** archived
(multi-GB, regenerable from the provided inputs); `data/` holds the harvested energies/frequencies behind the
paper's tables and figures.

## Citing

See `CITATION.cff`. Archived release: Zenodo DOI `10.5281/zenodo.XXXXXXX` (minted on submission).

## License

Code: MIT (`LICENSE`). Data (`data/`): CC-BY-4.0.

## Acknowledgements

Trelis Research; compute on Vast.ai (A100 PCIe 40 GB) and a local RTX 4070 node.
