# MLIP vs DFT for iron-sulfide defect kinetics

Code and data accompanying the manuscript:

> **Where Foundation Machine-Learning Potentials Fail for Iron-Sulfide Defect Kinetics: Failure Modes,
> Pre-Flight Checks, and a DFT Benchmark of Vacancy-Anchored Hydrogen Migration in Pyrite, Marcasite,
> Mackinawite, and Greigite**
>
> Igor N. Morozov. Independent Researcher, Ukraine.
> ORCID: [0009-0007-3863-1747](https://orcid.org/0009-0007-3863-1747) · igor@exopoiesis.space · exopoiesis.space

> ### Corrections to the archived results
>
> **The greigite 1.86 eV value in v1.0 is withdrawn as a migration barrier.** The trans-axis
> midpoint coincides with the vacant Fe site. Its symmetry-preserving construction can make the
> migrating H force vanish, but the partial Hessian has three negative modes, excluding an index-1
> saddle. The replacement paths give **235.97 meV** (`greigite_VFe_channel_band.extxyz`) and
> **566.83 meV** (`greigite_VFe_cation_band.extxyz`). The old band and its derived scans remain
> archived with `meta_RETRACTED`; they do not establish Hubbard-U sensitivity of the replacement paths.
>
> The pyrite 94.6 meV reference is **Fe-bound hydrogen transfer in a single V_S pocket**, not a
> V_S₂ dimer hop or S–H transfer. Host identity does not assign a formal hydride charge. The
> unreliable marcasite energy belongs to **image 1**, which is excluded from profile statistics;
> neither endpoint-to-maximum barrier changes with these label corrections. See
> [structure records](data/structures/README.md) and [result provenance](data/README.md).

This repository benchmarks vacancy-anchored hydrogen migration in four iron sulfides against plane-wave
PBE, U = 0 DFT references, with mineral-specific magnetic treatments. Hydrogen is added to neutral defect
supercells; these local dry-bulk paths do not predict hydrated proton conductivity.

Two protocols answer different questions. MACE-MP-0 large and CHGNet weights v0.3.0 are tested through
endpoint relaxation and self-consistent NEB. A separate comparison evaluates **nine checkpoints on five
fixed DFT bands**, giving 45 dependent model–band cells. The analysis separates energy-profile errors,
relaxation failures and endpoint-construction effects. Harmonic zero-point corrections and a limited
fine-tuning ladder accompany the reference calculations.

## Headline results (DFT, PBE U = 0)

| Mineral | Reaction coordinate | E_a (electronic) | Harmonic ZPE-corrected | Foundation-MLIP outcome |
|---|---|---|---|---|
| Pyrite (Pa-3̄) | V_S pocket, Fe→Fe hydrogen transfer | 94.6 meV | — | sulfur-adjacent seeds collapse to one Fe host; DFT-seeded endpoints retain distinct Fe hosts, but no validated self-consistent barrier is reported |
| Pyrite (Pa-3̄) | V_Fe + S–H | 268 meV | 173 meV | with own-surface endpoints: MACE within ~12 meV, CHGNet +119 meV |
| Mackinawite (P4/nmm) | V_Fe + S–H | 42.9 meV | −23 ± 15 meV | **all nine** models overestimate the smallest barrier in the series, by 1.75–9.66× |
| Marcasite (Pnnm) | V_Fe + S–H | 208 meV | 123 meV | self-consistent MACE 208 meV and CHGNet 318 meV; the DFT reference has about 15 meV magnetic-sheet uncertainty |
| Greigite (Fd-3̄m), **channel** edge | V_Fe + S–H | **236 meV** | 133 meV | pristine-cell MACE relaxation collapses (S–S → 0.072 Å); fixed-geometry MACE and CHGNet evaluations remain finite |
| Greigite (Fd-3̄m), **cation** edge | V_Fe + S–H | **567 meV** | 459 meV | contrast path; **all nine** models rank it above the channel edge |

The negative mackinawite harmonic estimate means that the correction exceeds the electronic barrier;
it is a limit of harmonic barrier theory, not a measured barrierless quantum rate. The table combines
the explicitly labelled self-consistent and fixed-geometry observations; the full comparison and
convergence details are in the manuscript and supporting data.

The greigite rows compare two symmetry-distinct S–S edges of the same vacancy octahedron from a common
endpoint. The shorter cation edge (3.369 Å) has the higher barrier; the channel edge is 3.493 Å. Edge
length, neighboring cation occupancy and relaxation change together, so the pair does not isolate a
unique cause or establish a rule for all spinels. It shows why both local coordination classes should
be considered. Both paths remain within one vacancy pocket and do not determine inter-pocket escape
or bulk conductivity.

## Does limited fine-tuning fix it?

The [fine-tuning experiment](mlip/r23_2026-09/README.md) uses MACE-MP-0 large and mackinawite as the
interpreted holdout (DFT barrier 42.88 meV). The four non-holdout bands supply 26 symmetry-inequivalent
configurations, split into **22 training and 4 validation configurations**. The table reports signed
barrier errors on the fixed DFT band, mean ± seed standard deviation over eight seeds; zero-shot is
one deterministic evaluation.

| Training rung | Signed barrier error (meV) |
|---|---|
| No new data (S0) | +149.1 |
| Four bands from the three other sulfides (S1) | +68.3 ± 22.7 |
| S1 plus both mackinawite endpoint basins (S2) | +68.6 ± 17.4 |
| S1 plus the full mackinawite band (S3, fitting control) | +9.1 ± 7.1 |
| Mackinawite band alone (S3solo, fitting control) | +0.5 ± 5.9 |

S1 demonstrates **partial transfer**, with residual error above the prespecified 25 meV tolerance.
Adding target endpoints changes little for this fixed-band metric. A 20× longer S1 schedule leaves
an error of +51.6 ± 18.8 meV, so undertraining alone does not explain the remaining error.

S3 and S3solo evaluate a band included in training. For S3solo, the mean signed error of +0.5 meV
reflects cancellation: its **MAE is 4.6 meV**, not sub-meV accuracy. Its nine target images are also
used for evaluation, with no independent validation set. This demonstrates target-profile fitting,
not generalization or a uniquely identified data-coverage cause. The paired S1−S3 absolute-error
reduction is 59.3 meV, with reported 95% interval [36.7, 81.8] meV and exact p = 0.0078, conditional
on eight seeds and this one holdout.

Self-consistent NEB and forgetting tests are reported separately in the experiment README. Target-only
fitting worsens other bands; the second, marcasite holdout fails its schedule gate and supports no
transfer conclusion. Active learning was not tested.

## Withdrawn in v1.4 — pentlandite

The withdrawn pentlandite calculations used a cell that reproduces the
composition of pentlandite but not its structure: the octahedral metal sits at Wyckoff 4a
instead of 4b, the tetrahedral metal at 32f with x = 0.356 instead of 0.1261, and all sulfur
on a single 32f orbit instead of the 8c and 24e sites pentlandite actually has. Composition
and site multiplicities come out right either way — 36 metals and 32 sulfurs give Fe₉S₈ — so
every composition check passed; the coordination does not (3 S at the tetrahedral metal and
8 at the octahedral one, against 4 and 6).

**Nothing is deleted.** Releases v1.0–v1.3 contain these files and may have been cited, so they
are flagged in place: `meta_RETRACTED` in the two `data/structures/pentlandite_end*.extxyz`,
a header in `mlip/mlip_pent_*.py` and `tm-spec/pent_vfe_neb.tm.yaml`, and the former Figure 3
renamed to `figures/withdrawn_pentlandite_endpoints.py`. The full account, including what the
"3 + 3 cubane collapse" actually was, is in **`mlip/PENTLANDITE_WITHDRAWN.md`**.

`mlip/pentlandite_structure_verified.py` builds the mineral correctly from the published
Wyckoff positions and refuses to return a cell whose metal coordination is wrong.

Pentlandite is excluded from the current four-mineral barrier benchmark and fine-tuning ladder.
Its withdrawal concerns the structural diagnostic; the retained reference compositions and
coordination checks are documented in [data/structures/README.md](data/structures/README.md).

## Repository layout

```
scripts/   QE DFT/NEB + dimer + convergence single-point drivers (ASE-driven), plus per_image_stats.py
zpe/       partial-Hessian zero-point frequency drivers (saddle + endpoint)
mlip/      MACE-MP-0 / CHGNet topology-scan, NEB-probe, and band-correlation drivers
mlip/r22_2026-09/    nine-model x five-band benchmark: per-image profiles, forces, barrier matrix
mlip/r23_2026-09/    few-shot fine-tuning ladder for MACE-MP-0: does limited fine-tuning recover a
                     held-out DFT barrier? ladders, blind schedule selection, convergence control
mlip/environments/   three environments used for the nine-model runs + available weight provenance
figures/   figure-generation scripts (Table 1 / ZPE trend / barrier landscape / band correlation)
data/      harvested result JSON and supporting numerical records
data/structures/  complete DFT NEB bands + relaxed endpoints/saddles as extended-XYZ (energy+forces)
tm-spec/   TM-Spec v0.3 declarative calculation records (machine-readable + format examples)
```

**Structure data.** [data/structures/](data/structures/README.md) contains DFT bands, endpoints and
saddles in extended-XYZ format, with energy/force metadata and explicit unreliable-image or withdrawal
flags. Read those flags before using the structures as training data or migration references.

**Software environments.** Three environments were used for the nine-model evaluation. Their complete
`pip freeze` files, model-source identifiers and the MACE checkpoint checksum are in
[mlip/environments/](mlip/environments/README.md). Older self-consistent calculations and fine-tuning
have documented component versions, but no contemporaneous complete package freeze.

**TM-Spec records.** `tm-spec/` holds [TM-Spec v0.3](https://github.com/exopoiesis/tm-spec) declarative records
for successful and unsuccessful calculations, as a provenance trail and as worked examples of
the format. See `tm-spec/README.md`.

## Methods (summary)

- **DFT:** Quantum ESPRESSO PWSCF 7.5, PBE and ONCV norm-conserving pseudopotentials, with U = 0
  for the reference geometries. Pyrite uses nspin = 1. Mackinawite uses a non-spin-polarized surrogate,
  not a claim of a physically nonmagnetic ground state. The marcasite defect is followed on a weakly
  magnetic sheet; greigite uses ferrimagnetic initialization. The production inputs and the manuscript
  specify mineral-dependent smearing, cutoffs and magnetic constraints. Hubbard-U sensitivity of the
  corrected greigite paths is unmeasured.
- **Paths:** nine-image CI-NEB with minimum-image prewrapping before ASE IDPP. The interpolation fix
  was contributed as ASE GitLab MR !4091 against issue #1130. Pyrite V_Fe instead uses an unconstrained
  ASE dimer saddle, with a constrained NEB as a consistency check.
- **Greigite saddle diagnostics:** 25-step mass-weighted Lanczos explores the full 168-dimensional
  space and identifies a dominant H-localized unstable mode at 829.6i cm⁻¹, effective mass 1.054 amu.
  No negative mode was detected at the endpoint. Finite-iteration Ritz values do not certify that
  exactly one negative mode exists; the cation edge has not received the same full-space test.
  See `data/greigite_VFe_saddle_lanczos_2026-09-21.json` and the [result notes](data/README.md).
- **ZPE:** harmonic partial-Hessian endpoint and saddle calculations use matched reactive atom sets.
  Distant-mode cancellation is an approximation. Tunneling estimates are conditional model
  calculations, not measured rates; their assumptions and subsystem checks are documented with
  the [result notes](data/README.md) and manuscript SI.
- **MLIP:** MACE-MP-0 large (`MACE_MPtrj_2022.9.model`) and CHGNet weights v0.3.0 are used through ASE.
  The deposited CHGNet initialization control tests insensitivity to supplied initial magnetic
  moments; it does not equate the learned surface with nspin = 1 DFT. The nine-model extension adds
  SevenNet, UMA, Orb and GRACE checkpoints at fixed DFT geometries. With both endpoints and path fixed,
  there is no general upward variational bound relative to each model's independently optimized
  barrier. It tests energy profiles, not whether the optimizer would find the same path.

## Energy convention

Electronic barriers use differences of the smeared electronic energy F reported on the `!` line
of Quantum ESPRESSO output; this is not a complete finite-temperature migration free energy. Rebuilding the
greigite channel barrier from the internal energy E = F + TS, or from the 0 K extrapolation, gives 227.89 and
231.90 meV instead of 235.91 — the choice moves the number by up to 8 meV. Absolute totals also depend
on the ASE CODATA conversion. Use the same energy convention and
conversion when regenerating energy differences; provenance and the cross-check are in the result records.

## Reproducing

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On Windows, use `.venv\Scripts\python.exe` in place of `.venv/bin/python`. These are core plotting/geometry dependencies, not a lockfile for all MLIP environments. The [figure instructions](figures/README.md) identify current and historical plots and their deposited inputs.

The DFT drivers expect Quantum ESPRESSO 7.5 (`pw.x`, GPU build) on `PATH` and ONCV PBE pseudopotentials for
Fe, S, H (standard QE/PseudoDojo distribution; place them where `ESPRESSO_PSEUDO` points and map them as
`Fe.upf`, `S.upf`, `H.upf`). MLIP drivers require `mace-torch` and `chgnet`; for the nine-model extension use
the three environment specifications in `mlip/environments/` rather than `requirements.txt`. Figure scripts
use `matplotlib` and `numpy`; their input records are documented with the scripts. Raw DFT wavefunction/charge-density dumps are
**not** archived (multi-GB, regenerable from the provided inputs); `data/` holds the harvested
energies/frequencies behind the paper's tables and figures.

The [offline reproduction checks](scripts/REPRODUCING.md) regenerate the saved profile statistics and test the fine-tuning aggregation without launching MLIP inference or DFT. Selected [primary input/output records](data/provenance/README.md) verify the corrected greigite cutoffs and pseudopotential identifiers.

## Citing

See `CITATION.cff` and the [archive description](ARCHIVE_DESCRIPTION.md).
Release **v1.5** is archived in [Zenodo record 22958549](https://zenodo.org/records/22958549),
with version-specific DOI **10.5281/zenodo.22958549**. It corrects analysis code, method
metadata and reproducibility documentation; see [CHANGELOG.md](CHANGELOG.md).
The previous archived release is **v1.4**, DOI
[10.5281/zenodo.22919796](https://doi.org/10.5281/zenodo.22919796).

**Cite the release that matches the numbers you use.** These are version-specific DOIs by choice, not
all-versions DOIs, because published numbers changed between releases. Archived predecessors:
**v1.2** ([10.5281/zenodo.22894038](https://doi.org/10.5281/zenodo.22894038)) — the state before the
fine-tuning ladder was added; **v1.0**
([10.5281/zenodo.20540264](https://doi.org/10.5281/zenodo.20540264)) — the submitted state, carrying
the greigite barrier retracted in v1.1 and the pyrite mechanism label corrected there. The retracted
band and the scans derived from it are kept in every release, flagged `meta_RETRACTED`, so that
anyone who cited an earlier release can still resolve what they cited.

## License

Code: MIT (`LICENSE`). Data (`data/`): CC-BY-4.0.

## Acknowledgements

Trelis Research; compute on Vast.ai (A100 PCIe 40 GB) and a local RTX 4070 node.
