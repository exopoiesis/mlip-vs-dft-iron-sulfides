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
| `greig_vfe_neb.tm.yaml` | NEBCalculation (PRELIMINARY) | Greigite V_Fe + S–H, 1.86 eV, robust to U (1.48–1.84 eV) |
| `mack_vfe_neb.tm.yaml` | NEBCalculation (PASS) | Mackinawite V_Fe + S–H, 43 meV (ZPE → barrierless) |
| `pyr_vs2_neb.tm.yaml` | NEBCalculation (PASS) | Pyrite V_S₂ dimer hop, 94.6 meV (hero anchor) |
| `pyr_vfe_bandfail.tm.yaml` | NEBCalculation (**FAIL**) | Pyrite V_Fe band-NEB failure (degenerate ridge) + dimer resolution (268 meV) |
| `pent_vfe_neb.tm.yaml` | NEBCalculation (PRELIMINARY) | Pentlandite V_Fe endpoints; barrier deferred to companion nspin=2 study |
| `mlip_cross_mineral_benchmark.tm.yaml` | MLIPBenchmark (PASS) | The four foundation-MLIP failure modes (MACE-MP-0 / CHGNet vs DFT) |

The barrier/structure numbers here mirror `../data/` (result JSON) and `../data/structures/` (geometries);
DFT/MLIP drivers are in `../scripts/`, `../zpe/`, `../mlip/`. Both successes and failures are recorded — a
complete, validate-able provenance trail for the paper.
