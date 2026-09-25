# TM-Spec v0.3 records

Declarative, machine-readable records of this paper's DFT/MLIP results in **TM-Spec v0.3** — a YAML
metalanguage for atomistic calculations (structure / defects / magnetic / environment / calculation /
workflow / results / sanity gates / provenance). These double as **worked examples** of the format.

Spec + reference validator: **https://github.com/exopoiesis/tm-spec** (`schemas/0.3.json`).
All nine records were checked on 2026-09-25 against the local reference implementation's
`schemas/0.3.json` and additional validator rules. Schema validation checks representation,
not scientific correctness or proof of every sanity gate:

```
python -m tm_spec.cli validate tm-spec/<file>.tm.yaml
```

| File | kind | Captures |
|---|---|---|
| `marc_vfe_neb.tm.yaml` | NEBCalculation (PASS) | Marcasite V_Fe + S–H, 208 meV (endpoint-relaxed, single sheet) |
| `greig_vfe_neb.tm.yaml` | NEBCalculation (**RETRACTED**) | Greigite trans-axis band, 1.86 eV — **not a migration barrier**; kept for citation trace, superseded by the two records below |
| `greig_vfe_channel_neb.tm.yaml` | NEBCalculation (PASS, qualified) | Greigite channel-edge path maximum **235.97 meV**; dominant negative Lanczos mode, no strict full-space index-1 certificate |
| `greig_vfe_cation_neb.tm.yaml` | NEBCalculation (PASS) | Greigite V_Fe + S–H along a **cation** edge, **566.83 meV**; the expensive contrast path, same endpoint |
| `mack_vfe_neb.tm.yaml` | NEBCalculation (PASS) | Mackinawite V_Fe + S–H, 43 meV; negative harmonic corrected estimate marks a limitation of harmonic barrier theory |
| `pyr_vs2_neb.tm.yaml` | NEBCalculation (PASS) | Pyrite **single V_S pocket, Fe→Fe hydrogen transfer**, 94.6 meV; filename/id retain `vs2` as historical aliases, not a dimer-vacancy or formal hydride-charge claim |
| `pyr_vfe_bandfail.tm.yaml` | NEBCalculation (**FAIL**) | Pyrite V_Fe band-NEB failure (degenerate ridge) + dimer resolution (268 meV) |
| `pent_vfe_neb.tm.yaml` | NEBCalculation (**RETRACTED**) | Endpoints from an incorrect pentlandite structure; no accepted barrier or motif-failure diagnosis |
| `mlip_cross_mineral_benchmark.tm.yaml` | MLIPBenchmark (PASS, qualified) | Current two-model comparisons with explicit self-consistent protocol and historical withdrawn interpretations |

### Corrections, and why the superseded records are still here

Two entries above carry corrections made on **2026-09-22**, and both are kept rather than rewritten
away, because a provenance trail that silently edits itself is not a provenance trail.

- **`greig_vfe_neb.tm.yaml` is retracted.** Its band ran along a *trans axis* of the V_Fe octahedron,
  and the midpoint of a trans axis is the vacant Fe site — a special position at which the force
  vanishes **by symmetry rather than by convergence**. The band therefore satisfied any force
  criterion it was given, and a partial Hessian there carries **three** imaginary modes, so it is not
  a first-order saddle. Everything derived from it is withdrawn too: the U scan, the k/cutoff scan and
  the "kinetically forbidden, τ ≈ 90 Gyr" conclusion. The record now states this in its
  `retraction_reason` field and points at its successors.
- **`pyr_vs2_neb.tm.yaml` was relabelled, not retracted.** The cell is `HFe₃₂S₆₃` — one sulfur
  removed, not a dimer — and the hydrogen is Fe-bound in all nine images, so the migration is a
  hydrogen transfer between two adjacent Fe sites rather than S–H transfer; no formal H charge is inferred. The composition and geometry
  were always what they are; only the name was wrong, so **the 94.6 meV barrier is unaffected**.

The barrier/structure numbers here mirror `../data/` (result JSON) and `../data/structures/` (geometries);
DFT/MLIP drivers are in `../scripts/`, `../zpe/`, `../mlip/`. Both successes and failures are recorded — a
provenance record with the limits below, retractions included.

### Metadata corrections and reference resolution (2026-09-25)

- `structure.ref` and `workflow.endpoints.*.ref` now resolve **relative to the YAML file**;
  they point to existing files under `../data/structures/`. A band reference identifies the
  associated defect structure/path, not a missing pristine parent. IDs retain historical aliases.
- The failed pyrite V_Fe record's old standalone endpoint references are not deposited. Their
  original names are retained in `results.missing_legacy_geometry_refs`; the separately linked
  dimer structure is the subsequent resolution, not the missing failed band.
- `results.historical_interpretation` retains superseded descriptions, internal review labels and,
  where applicable, old numerical summaries. It is an audit trail, **not current evidence**. The
  current two-model pyrite V_Fe fields are 256/387 meV; 182/223 are retained only as historical
  frozen-endpoint results. Pyrite V_S self-consistent values are null because no validated barrier
  was obtained under the tested construction. Raw data files and numerical arrays are unchanged.
- `PASS` with a `quote_scope` means that the reported path-energy result is retained within that
  scope; it is not a global validation of rates or transport. Both corrected greigite records have
  a saddle-index gate of `warn`: finite Lanczos evidence for the channel, no equivalent full-space
  check for the cation edge. Rate entries remain conditional harmonic estimates.
- Legacy input/output checksum placeholders were removed and replaced by `provenance.hash_status`.
  They were not computed hashes. New greigite records now include the required provenance block;
  former `FINAL` values were replaced by schema-supported `PASS` with explicit scope, and the
  invalid null gate value became `warn`. These changes repair five pre-existing schema errors.
- Mackinawite's Gaussian width is 0.01 Ry, verified from the production input; the old 0.005 label
  was incorrect. Corrected greigite edges use 80/320 Ry, independently verified in their production
  inputs and outputs. Method assumptions do not override source artifacts.
