# R2.3 — Few-shot fine-tuning ladder for MACE-MP-0

Data behind the fine-tuning experiment added during the Digital Discovery revision, in response to
referee 2, comment 3:

> *It would considerably strengthen the work if the authors examined whether limited fine-tuning or
> active learning can recover the DFT barriers for at least one representative system. This would
> distinguish zero-shot failure from a more fundamental limitation of the MLIP approach.*

**This is fine-tuning, not active learning.** There is no acquisition function, no uncertainty model
and no loop: every additional label here would cost a DFT-NEB calculation. We call it a *few-shot
fine-tuning ladder* and make no claim about active learning.

---

## 1. What the ladder is

One foundation model (**MACE-MP-0 large**, checkpoint `MACE_MPtrj_2022.9.model` — the same
checkpoint benchmarked in the paper) is fine-tuned on DFT energies and forces taken from the
NEB bands already deposited in `data/structures/`. One mineral is held out; the amount of
information the model is given about that mineral is increased rung by rung.

| rung | training set | question it answers |
|---|---|---|
| **S0** | nothing (zero-shot) | the paper's baseline |
| **S1** | the four bands of the three *other* iron sulfides | does the failure transfer away? |
| **S2** | S1 + both endpoint basins of the held-out mineral | do the basins suffice? |
| **S3** | S1 + the full band of the held-out mineral | representability ceiling (in-domain) |
| **S3solo** | *only* the held-out band | capacity control, free of cross-mineral conflict |

**The discriminator is the paired difference S3 − S1**, not "did S1 recover". S3 is deliberately
tautological: it works as a ceiling, not as a result. S3solo was added because a negative S3 would
otherwise be indistinguishable from a third possibility — conflicting supervision between minerals,
which here require corrections of opposite sign (+149, +159, −15, −132 meV).

Eight seeds per rung. The validation split is drawn only from non-holdout minerals and is
**identical across rungs at a given seed**, so the design is paired.

## 2. How little data this is

For the mackinawite holdout the training pool is 35 band images, but:

* one pair is a **bit-identical duplicate** (`greigite_channel:0` = `greigite_cation:0`) → 34 distinct;
* both greigite bands are **exactly mirror-symmetric**: images *i* and *8−i* are isometric
  (interatomic-distance spectra agree to 1.3·10⁻⁵–8.4·10⁻⁵ Å), so for an E(3)-equivariant model the
  second half of each greigite band carries no new information. MACE reproduces the mirror pairs to
  0.07–0.10 meV, i.e. numerical noise;
* mackinawite, pyrite and marcasite are **not** mirror-symmetric.

⇒ **26 symmetry-inequivalent configurations**, and an **effective n ≈ 4 paths**, because the nine
images of a band are points on one trajectory. All three numbers (35 / 34 / 26) are stated because
they are the honest measure of the data budget. Training uses only the 26.

The energy targets are referenced per band:
`target_E(b,i) = E_DFT(b,i) − E_DFT(b,0) + E_MACE0(b,0)`, forces are the DFT forces unchanged, and
`E0s` are the foundation model's own atomic energies. Absolute DFT totals here are ONCV
pseudopotential energies (−1300…−1800 eV/atom) and are incompatible with the foundation scale
(≈ −5.9 eV/atom). A consequence worth stating: the four training bands span only **three distinct
stoichiometries** over three elements — Fe23S32H for both greigite bands, Fe32S63H for pyrite,
Fe31S64H for marcasite (determinant of the composition matrix = 40 ≠ 0) — so a linear
composition reference is exactly determined and **inter-band energy information is identically zero
by construction**. The model learns intra-band shape plus forces, nothing else.

## 3. Result — mackinawite holdout (DFT barrier 42.88 meV)

Errors in meV, mean ± s.d. over 8 seeds. "Single-point" = model energies on the fixed DFT band
(the R2.2 protocol); "self-consistent NEB" = endpoints relaxed by the model, then its own NEB
(the convention of Table 2 in the paper), computed for 3 seeds per rung.

| rung | single-point | self-consistent NEB | recovers? |
|---|---|---|---|
| S0 | +149.12 | +65.60 | no |
| S1 | +68.33 ± 22.70 | +33.06 ± 24.56 | no |
| S2 | +68.55 ± 17.38 | +16.99 ± 13.76 | no / NEB yes |
| **S3** | **+9.05 ± 7.13** | **+9.02 ± 11.45** | **yes** |
| **S3solo** | **+0.52 ± 5.92** | **+2.66 ± 9.70** | **yes** |

Discriminator, paired over seeds, exact sign-permutation test:

* single-point, **8 pairs**: Δ = |S1| − |S3| = **+59.28 meV**, 95 % CI [+36.72, +81.84],
  **p = 0.0078** — the smallest value 8 pairs can produce (2/2⁸), stated so that it is read as a
  design limit rather than as strength of evidence;
* self-consistent NEB, **3 pairs**: Δ = +24.03 meV, 95 % CI [−65.39, +113.46], p = 0.5. The
  smallest attainable two-sided p at 3 pairs is 0.25, so **this convention is descriptive only**
  and establishes nothing on its own. The single-point result carries the claim.

> ⚠️ **Corrected 2026-09-23.** The NEB discriminator was first reported as +44.81 meV with
> p = 0.0156. That was wrong: the paired difference was computed over all eight seeds although the
> self-consistent convention was evaluated for only three, and the remaining seeds silently fell
> back to the single-point value, mixing two conventions. `ladder_*.json` now carries the corrected
> figure and a `correction_note`. The single-point discriminator was never affected.

**Reading.** The zero-shot failure is a **data-coverage** failure, not a limitation of the MLIP
approach: the architecture represents this barrier accurately once it has seen the reaction
coordinate of *that* mineral (S3solo lands within 0.5 meV). What does not happen is transfer —
fine-tuning on three other iron sulfides leaves the error essentially where zero-shot left it.

Three secondary observations:

* **S2 ≈ S1.** Giving the model both endpoint basins of the target mineral changes nothing in the
  single-point convention. The saddle region is what is needed, and that is exactly what cannot be
  obtained without DFT.
* A model that fits the barriers of its own four training minerals to **5.4 meV** still misses the
  fifth by ~70 meV (see `control_schedule_length.json`). Good training-set fit says nothing about
  transfer.
* **The in-domain rung buys its accuracy by forgetting.** S3solo reaches +0.5 meV on mackinawite
  while its barrier error on the four minerals it was *not* trained on degrades by +38 to +186 meV
  relative to zero-shot, and its force error on structures outside the training set (pentlandite
  V_Fe + H) drifts by 0.22–0.28 eV/Å. Every rung shows out-of-domain force drift of 0.25–0.45 eV/Å
  on pentlandite, above the 0.1 eV/Å threshold declared in advance. Single-head fine-tuning
  produces a specialist, not an improved general potential; replay-based multi-head fine-tuning is
  the established mitigation and was not available for this checkpoint (see §7). Per-rung figures
  are in the `F1_train_bands` and `F2_force_drift` fields of `ladder_*.json`.

## 4. Control: is this an artefact of a short schedule?

The ladders were trained for 200 epochs, which **fails** the preregistered schedule gate (see
`schedule_sweep.json`): at 200 epochs the best configuration reproduces the training bands' own
barriers only to MAE 31.6 meV, against a 10 meV threshold. The identical hyperparameters run to
convergence (4000 epochs) pass the gate at 5.4 meV.

S1 was therefore re-run to convergence, 8 seeds:

| schedule | single-point | self-consistent NEB |
|---|---|---|
| 200 epochs | +68.33 ± 22.70 (n = 8) | +33.06 ± 24.56 (n = 3) |
| 4000 epochs | +51.56 ± 18.76 (n = 8) | **−25.41 ± 18.25** (n = 8) |

A twentyfold longer schedule moves the single-point error by 7–17 meV and leaves S1 far outside the
25 meV recovery tolerance. The conclusion is not an artefact of training length.

In the self-consistent convention the longer schedule **reverses the sign of the error** and inflates
the seed-to-seed spread: per-seed NEB errors are
−42.9, −41.6, −40.8, −34.5, −28.3, −15.3, −4.4, +4.5 meV. The spread (47 meV) **exceeds the DFT
barrier itself** (42.88 meV), and in one seed of eight the barrier vanishes entirely (E_a = 0.00 meV).

Because S3 and S3solo *already* recover at the short schedule, and longer training can only improve
an in-domain fit, the ceiling conclusion holds a fortiori and the full ladder was not re-run.

## 5. Second holdout (marcasite) — reported, not interpreted

A second ladder was run with marcasite held out, at the same 200-epoch schedule
(`ladder_marcasite.json`, `evaluations_marcasite.json`). **It does not support an interpretation,
and we say so rather than quietly dropping it.**

* S3solo gives −56.06 ± 22.71 meV, i.e. the model fails to reproduce the barrier of the very band it
  was trained on. For mackinawite the same rung landed at +0.52 meV. 200 epochs is not enough for a
  208 meV barrier although it sufficed for a 42.88 meV one. This is precisely the schedule-gate
  failure described above, and the preregistered rule says such a ladder cannot be read as a ceiling.
* The discriminator is flat: Δ = −6.61 meV, 95 % CI [−36.07, +22.84], p = 0.625 — indistinguishable.

One result from this ladder *is* usable and is reported: **zero-shot MACE-MP-0 reproduces the
marcasite barrier to −1.44 meV in the self-consistent convention** (−36.29 meV single-point).
That is the best agreement anywhere in this work, and it is why marcasite is a poor choice of
holdout for a recovery experiment — there is almost nothing to recover. The holdout was moved to
mackinawite, where every one of the nine models benchmarked in §3.6 overestimates the barrier
(MACE-MP-0 by 4.5×), on the basis of that published fact and not of any fine-tuning outcome.

Bringing the marcasite ladder to a converged schedule would cost roughly 35 GPU-hours; it was judged
not worth it, since the mackinawite ladder answers the referee's question and the marcasite ladder's
only interpretable number is already reported.

## 6. Files

| file | content |
|---|---|
| `zeroshot.json` | S0 baseline: MACE-MP-0 large on all five DFT bands, energies, forces, barriers |
| `datasets_{mineral}.json` | exactly which band images are in train/validation for every rung and seed |
| `schedule_sweep.json` | the blind schedule selection: five configurations, the gate, the outcome |
| `evaluations_{mineral}.json` | per-model results for every rung and seed: profiles, barriers, reaction energies, per-image errors, force RMS |
| `ladder_{mineral}.json` | aggregated ladder, discriminator, CI, exact p |
| `control_schedule_length.json` | 200 vs 4000 epochs for S1, both conventions |
| `training_logs.tar.gz` | 78 MACE training logs (convergence traces behind the schedule argument) |

Pipeline in `../../scripts/`: `zeroshot.py` → `build_sets3.py` → `verify_sets3.py` →
`train4.sh` → `evaluate2.py` → `aggregate2.py`; `tune_report2.py` and `pick_schedule.py` implement
the blind schedule selection.

**Model weights are deliberately not deposited** (77 checkpoints, 1.7 GB). They are reproducible
from the deposited bands, dataset specifications, seeds and scripts; note that MACE training on GPU
is not bit-reproducible, so barriers will differ at the few-meV level. Optimiser states and
duplicate checkpoint copies (a further 8.8 GB) are likewise omitted as carrying no scientific
information.

## 7. Environment

Container `exopoiesis/infra-mace-gpu`, NVIDIA A100-SXM4-80GB: MACE 0.3.15, PyTorch 2.5.1+cu124,
ASE 3.23.0, Python 3.10.12, `--default_dtype float64` throughout. Fine-tuning is single-head
(`--multiheads_finetuning False`): MACE does not recognise this checkpoint as a Materials Project
model and requires an explicit replay set, which we do not have. Single-head fine-tuning converges
*better* on one target system, so this is the regime most favourable to recovery — the caveat is
that out-of-domain retention is not protected, which is why drift on structures outside the training
set is reported in the evaluation files.

The evaluator reproduces the published R2.2 zero-shot barriers to within 0.05 meV on all five bands
(171.86 / 221.14 / 435.03 / 192.00 / 253.35 meV), which is checked in code before any ladder runs.
