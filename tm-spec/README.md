# TM-Spec v0.3 records

Declarative, machine-readable records of this paper's DFT/MLIP results in **TM-Spec v0.3** — a YAML
metalanguage for atomistic calculations (structure / defects / magnetic / environment / calculation /
workflow / results / sanity gates / provenance). These double as **worked examples** of the format.

Spec + reference validator: **https://github.com/exopoiesis/tm-spec** (`schemas/0.3.json`).
All files here validate against `tm-spec/0.3.json`:

```
python -m tm_spec.cli validate tm-spec/<file>.tm.yaml
```

| File | kind | Captures |
|---|---|---|
| `marc_vfe_neb.tm.yaml` | NEBCalculation (PASS) | Marcasite V_Fe + S–H, 208 meV (endpoint-relaxed, single sheet) |
| `greig_vfe_neb.tm.yaml` | NEBCalculation (**RETRACTED**) | Greigite trans-axis band, 1.86 eV — **not a migration barrier**; kept for citation trace, superseded by the two records below |
| `greig_vfe_channel_neb.tm.yaml` | NEBCalculation (PASS) | Greigite V_Fe + S–H along a **channel** edge, **235.97 meV**; index-1 saddle verified in the full 168 DOF |
| `greig_vfe_cation_neb.tm.yaml` | NEBCalculation (PASS) | Greigite V_Fe + S–H along a **cation** edge, **566.83 meV**; the expensive contrast path, same endpoint |
| `mack_vfe_neb.tm.yaml` | NEBCalculation (PASS) | Mackinawite V_Fe + S–H, 43 meV (ZPE → barrierless) |
| `pyr_vs2_neb.tm.yaml` | NEBCalculation (PASS) | Pyrite **V_S pocket, Fe→Fe hydride hop**, 94.6 meV (hero anchor). Filename and id retain `vs2` for citation stability; the *mechanism label* was corrected 2026-09-22 — it is not a dimer vacancy and not an S–H hop |
| `pyr_vfe_bandfail.tm.yaml` | NEBCalculation (**FAIL**) | Pyrite V_Fe band-NEB failure (degenerate ridge) + dimer resolution (268 meV) |
| `pent_vfe_neb.tm.yaml` | NEBCalculation (PRELIMINARY) | Pentlandite V_Fe endpoints; barrier deferred to companion nspin=2 study |
| `mlip_cross_mineral_benchmark.tm.yaml` | MLIPBenchmark (PASS) | Foundation-MLIP failure modes (MACE-MP-0 large / CHGNet vs DFT) |

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
  hydride hop between two adjacent Fe sites rather than an S–H transfer. The composition and geometry
  were always what they are; only the name was wrong, so **the 94.6 meV barrier is unaffected**.

The barrier/structure numbers here mirror `../data/` (result JSON) and `../data/structures/` (geometries);
DFT/MLIP drivers are in `../scripts/`, `../zpe/`, `../mlip/`. Both successes and failures are recorded — a
complete, validate-able provenance trail for the paper, retractions included.
