# Changelog

## Metadata update after v1.5 — 2026-09-25

Added a study description, the manuscript abstract in `CITATION.cff`, and the
Zenodo-assigned v1.5 DOI. The GitHub release description was expanded for readers
of the paper. These metadata changes do not move tag v1.5 or replace its archived
files; the Zenodo landing-page description can be edited separately without a new DOI.

## v1.5 — 2026-09-25

Archived as [Zenodo record 22958549](https://zenodo.org/records/22958549),
DOI **10.5281/zenodo.22958549**. The previous archive is
v1.4, DOI [10.5281/zenodo.22919796](https://doi.org/10.5281/zenodo.22919796).

This release corrects analysis code, interpretation, method descriptions and reproducibility
instructions. No new DFT, MLIP inference or fine-tuning calculations were performed for these
changes. The reported electronic reference barriers are not being replaced by newly computed
values. Comparison with tag v1.4 confirms unchanged numerical values in all 39 pre-existing JSON files
and unchanged content in all 19 structure files (apart from Git's LF/CRLF checkout convention).

### Analysis code

- Fixed `scripts/aggregate2.py`: paired self-consistent NEB statistics now use only seeds with
  NEB results in both rungs. Missing NEBs are never replaced by fixed-band evaluations. The
  previous code incorrectly regenerated eight mixed pairs for the NEB contrast; the corrected
  code reproduces the three-pair statistics already present in the deposited ladder JSON files.
  For mackinawite this is 24.03 meV, p = 0.5, rather than the invalid mixed-protocol
  44.81 meV, p = 0.0156. Fixed-band statistics remain separate.
- Replaced automatic causal and equivalence labels in the aggregation output with descriptive
  statements conditional on the selected band and seeds. A schedule-selection record does not
  certify which schedule trained an existing model; fitting success does not establish a passed
  training gate or a general representation limit.
- Fixed `scripts/pick_schedule.py` to consider all five recorded schedules, including the
  4000-epoch tD and tE controls, and to select among gate-passing schedules using validation
  force RMSE. Missing validation data for a passing schedule now prevents selection.
- Fixed `scripts/train4.sh` to return the training process's exit status. Added configurable
  working-directory, Python and checkpoint paths without changing the recorded training data.
- Made `scripts/per_image_stats.py` use files inside the checkout by default, reject incomplete
  nine-model profile sets and write regenerated statistics under `tmp/` rather than overwrite
  the archived result file.
- Added offline regression tests for the two deposited ladders, missing NEB pairs, schedule
  selection from archived logs, failure-status propagation and the five-band profile inventory.
  Missing or incomplete reference self-tests, malformed/non-finite profiles and stale successful
  schedule selections are rejected; all 10 offline tests pass.
- Made incomplete or absent reference self-tests fail before ladder aggregation; malformed
  per-image profiles can no longer produce a successful 44-cell result. Failed schedule
  selection atomically replaces a previous passing `chosen.json` with an explicit invalid record.
  The ten offline regressions include these negative cases.

### Interpretation and method documentation

- Distinguished self-consistent MLIP pathways from energy evaluations on fixed DFT bands.
  Removed the claim that the fixed-band protocol must bias barrier estimates upward.
- Described pyrite V_S endpoint-seeding sensitivity without claiming that the potentials lack
  two Fe-bound minima. Fe-bound hydrogen is not assigned a formal hydride charge from geometry.
- Reported the negative mackinawite harmonic estimate as a limitation of harmonic barrier
  theory, not a measured barrierless quantum rate. Finite-iteration greigite Lanczos results
  are evidence for a dominant unstable mode, not a strict full-space index-1 certificate.
- Distinguished partial transfer from fitting the target band. The S3solo signed mean of
  0.5 meV is not its accuracy: the mean absolute error across seeds is 4.6 meV. The results
  do not establish training coverage as the sole cause of every zero-shot failure.
- Corrected descriptions of the accepted greigite calculations to the recorded 80/320 Ry
  wavefunction/charge-density cutoffs. This corrects method metadata; it is not a new calculation.
- Limited full package-freeze claims to the nine-model evaluation. Older runs have recorded
  component versions but no contemporaneous complete environment snapshot. Bulk eSEN and
  EquiformerV2 were untested, rather than unavailable. Checkpoint hashes, mutable container
  tags and unavailable build definitions are described with their actual coverage.

### Provenance and figures

- Recovered the historical three-model GRACE driver as
  `scripts/grace_band_multi_historical.py`, with the source-recovery record and SHA-256 in
  `scripts/REPRODUCING.md`. It retains its documented historical staging layout; no new
  inference was run to validate the restored file.
- Added accepted-greigite method evidence under `data/provenance/` and local figure input
  JSON files with SHA-256 provenance under `data/figure_inputs/`.
- Made figure generators use the included inputs and documented their mapping to the current
  main-text and SI figures. Historical degenerate-endpoint comparisons are labelled as withdrawn
  interpretations. The damaged pentlandite figure source is retained verbatim as `.py.txt`,
  while its former Python entrypoint explains the withdrawal.
- Corrected the current archived-version label from v1.3 to v1.4 without changing the v1.4 DOI.
  Previously withdrawn calculations remain available as historical records.
- Synchronized all nine TM-Spec records with current protocols and withdrawal status. Historical
  values remain labelled separately from current results; missing provenance is stated rather
  than filled with placeholder hashes. All nine records pass the reference schema and rules.

### Verification

- Ten offline regression tests passed; an independent replay closed all three identified
  input-validation issues. No MLIP inference or DFT was required.
- Six figure generators produced PDF/PNG/SVG in an isolated copy of the checkout. Five added
  figure-input JSON files and three primary greigite records match their retained source bytes.
- Structure and existing JSON numerical comparisons against v1.4 passed. Changes to scientific
  descriptions do not silently replace raw observations.
- Historical releases are not retagged.

## v1.4 — 2026-09-23

Archived release: [10.5281/zenodo.22919796](https://doi.org/10.5281/zenodo.22919796).
Withdrew the pentlandite structural diagnostic because the constructed cell did not reproduce
the intended crystal structure; retained and flagged the historical artifacts. The fine-tuning
composition-matrix determinant was corrected to 40. See the release's withdrawal notice and
`mlip/PENTLANDITE_WITHDRAWN.md` for the historical record.
