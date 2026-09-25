This dataset and code package accompanies the study of foundation machine-learning interatomic potentials (MLIPs) for vacancy-anchored hydrogen migration in pyrite, marcasite, mackinawite and greigite.

Foundation MLIPs offer inexpensive migration-barrier calculations, but their reliability for hydrogen defects in magnetic iron sulfides remains uncertain. We establish plane-wave DFT references at PBE, U = 0 for these four minerals. The electronic barriers for sulfur-to-sulfur transfer around an Fe vacancy are 43, 208, 268 and 236 meV for mackinawite, marcasite, pyrite and the lower-barrier greigite path, respectively. A second greigite edge gives 567 meV: the shorter edge has the higher barrier, showing why path selection requires local coordination rather than distance alone. A separate pyrite sulfur-vacancy pocket supports Fe-to-Fe hydrogen transfer with a 94.6 meV reference barrier.

The study distinguishes self-consistent MLIP pathways from energy evaluations on fixed DFT geometries. Nine foundation potentials evaluated on five DFT bands give barrier errors from −132 to +371 meV. The pyrite Fe-vacancy dimer reference is not part of this five-band comparison. The models identify the highest-energy image in 43 of 45 model–band comparisons, yet all nine overestimate the 43 meV mackinawite barrier by factors of 1.75–9.66. No consistent improvement with model generation is resolved in this small benchmark. Separate relaxation tests reveal pristine-greigite instability and endpoint-seeding sensitivity, which fixed-geometry energy tests cannot diagnose.

Fine-tuning MACE-MP-0 on four bands from the other three sulfides reduces the held-out mackinawite error from 149 to 68 ± 23 meV (mean signed barrier error on the fixed DFT band ± standard deviation across eight training seeds). Fitting its own band alone permits a 4.6 meV mean absolute error, demonstrating representability without establishing transfer to unseen pathways. Harmonic zero-point corrections lower the four channel-reference barriers by 66–103 meV; mackinawite reaches the limit of harmonic barrier theory.

## Contents of this archive

- [Repository documentation](https://github.com/exopoiesis/mlip-vs-dft-iron-sulfides/tree/v1.5/README.md), DFT/NEB and dimer drivers, and instructions for reproducing analyses from stored results without new electronic-structure calculations.
- [Structures](https://github.com/exopoiesis/mlip-vs-dft-iron-sulfides/tree/v1.5/data/structures/) and [result data](https://github.com/exopoiesis/mlip-vs-dft-iron-sulfides/tree/v1.5/data/): extended-XYZ bands, endpoints and saddles, harvested energies and frequencies, selected production input/output records, and explicit unreliable-image and withdrawal flags.
- [Nine-model benchmark records](https://github.com/exopoiesis/mlip-vs-dft-iron-sulfides/tree/v1.5/mlip/r22_2026-09/), including per-image profiles, forces and barrier comparisons, and the [fine-tuning ladder](https://github.com/exopoiesis/mlip-vs-dft-iron-sulfides/tree/v1.5/mlip/r23_2026-09/), with dataset specifications, per-seed evaluations, schedule selection records and training logs.
- [Figure generators](https://github.com/exopoiesis/mlip-vs-dft-iron-sulfides/tree/v1.5/figures/) with deposited plotting inputs and provenance hashes; [TM-Spec records](https://github.com/exopoiesis/mlip-vs-dft-iron-sulfides/tree/v1.5/tm-spec/) describe calculation settings, results and their stated limitations in machine-readable form.

## Scope and limitations

These are local neutral-H defect calculations in dry bulk, with mineral-specific magnetic reference choices, not predictions of hydrated proton conductivity. The 45 model–band comparisons concern five pathways and are not independent observations. Target-band fitting controls are not validation on unseen paths, and a negative harmonic corrected barrier does not establish a barrierless quantum rate. Finite-iteration greigite Lanczos diagnostics do not certify the complete saddle index.

[Software manifests](https://github.com/exopoiesis/mlip-vs-dft-iron-sulfides/tree/v1.5/mlip/environments/) cover the three environments used for the nine-model evaluation. Older calculations have component-version records rather than complete contemporaneous package snapshots. A checkpoint checksum is supplied for MACE; model weights, immutable container images, and raw wavefunction/charge-density dumps are not included.

## Corrections and version 1.5

The historical greigite 1.86 eV trans-axis result is withdrawn as a migration barrier. Pentlandite diagnostics based on an incorrect crystal model are also withdrawn. Their marked records remain available for traceability and must not be treated as current reference results.

Version 1.5 aligns documentation and structured metadata with these interpretations, corrects aggregation of the three available self-consistent NEB seed pairs, and strengthens analysis-input checks. It also supplies standalone figure inputs and selected method-provenance files. These changes preserve the existing numerical datasets and do not introduce new DFT or MLIP calculations.
