# Software environments for the nine-model MLIP benchmark (2026-09)

Exact package versions for every foundation-potential number in the paper, recorded **from the runs
themselves** rather than assembled afterwards. Each `env_*.txt` is a verbatim `pip freeze` taken on the
machine that produced the corresponding results.

---

## Three environments, not one — and that is itself a finding

The nine checkpoints benchmarked here **cannot be installed together**. The obstruction is not
carelessness on our part but genuine incompatibility between the stacks:

| file | environment | results obtained in it | why it has to be separate |
|---|---|---|---|
| `env_main.txt` (204 packages) | torch 2.8.0+cu126, Python 3.12 | MACE-MP-0 (large), CHGNet v0.3.0, SevenNet 7net-0, UMA-s-1p2 | the base stack |
| `env_orb.txt` (84) | **orb-models 0.5.5, torch 2.14.0** | Orb-v2, Orb-v3-conservative-inf-omat | orb-models **0.7.0 removed the ASE calculator**; 0.5.5 pulls its own torch, which is incompatible with fairchem |
| `env_grace.txt` (77) | **tensorpotential 0.6.1, TensorFlow 2.20.0** | GRACE-1L-OMAT, GRACE-2L-OMAT, GRACE-2L-OAM | GRACE runs on TensorFlow; two CUDA runtimes do not coexist in one container |

Anyone assembling a multi-model benchmark of this kind should plan for the same split.

### Two honest notes about `env_main.txt`

**The `orb-models==0.7.0` pin in that freeze produced no results.** Version 0.7.0 ships no ASE
calculator, and an attempt to use it fails with `no ORBCalculator found`. Every Orb number in the paper
comes from `env_orb.txt` (orb-models 0.5.5). The pin in the main freeze is a by-product of installation,
not a source of results, and we leave it visible rather than editing the freeze.

**pip reports a dependency conflict** in that environment: `fairchem-core 2.20.0 requires e3nn>=0.5,
but you have e3nn 0.4.4` (pulled in by mace-torch). We checked that it does not affect the numbers —
MACE on the greigite channel band reproduced the control value of **221.1 meV** obtained independently
on a different machine.

## Models and weight provenance

| model | source | licence |
|---|---|---|
| MACE-MP-0 (large) | `github.com/ACEsuit/mace-mp` → `MACE_MPtrj_2022.9.model`, 133 803 220 B, `sha256 f80e992b65ab8f88fdf26964511357c022e92704e4d9bcd086652635a8495b32` | open |
| CHGNet v0.3.0 | weights bundled with the `chgnet` package | open |
| SevenNet `7net-0` | bundled with the `sevenn` package | open |
| UMA-s-1p2 | HuggingFace `facebook/UMA` (gated; licence accepted) | Meta — **not redistributable** |
| Orb-v2, Orb-v3-conservative-inf-omat | `orbital-materials`, downloaded by the package | open |
| GRACE-1L-OMAT, GRACE-2L-OMAT, GRACE-2L-OAM | HuggingFace `AMS-ICAMS-RUB/grace-foundation-models` | Academic Software License |

**Weights are not archived here and were not baked into the container images**: the Meta checkpoint is
gated and not redistributable, and GRACE is under an academic licence. The freezes plus the sources
above are sufficient to reconstruct every environment.

## Why eSEN and eqV2 are absent

`fairchem_available_models.json` is a verbatim listing of `pretrained_mlip.available_models` from
`fairchem-core 2.20.0`, deposited as evidence:

```
uma-s-1p2, uma-s-1p1, uma-m-1p1,
esen-md-direct-all-omol, esen-sm-conserving-all-omol, esen-sm-direct-all-omol,
allscaip-md-conserving-all-omol, allscaip-md-direct-all-omol,
esen-sm-conserving-all-oc25, esen-md-direct-all-oc25,
esen-sm-filtered-odac25, esen-sm-full-odac25
```

Every eSEN checkpoint exposed by that API is trained on **OMol** (molecules), **OC25** (catalysis) or
**ODAC25** (metal-organic frameworks); there is no materials variant in the registry, and **eqV2 is
absent entirely**. OMat24 checkpoints do exist as raw `.pt` files under `facebook/OMAT24`, loadable
through the previous-generation API.

We chose not to pull those raw checkpoints, and the reason is the reference rather than the domain:
the available eSEN models are trained against ωB97M-V or RPBE, whereas every barrier in this study is
PBE. The functional difference exceeds any barrier we report, so a comparison would not have measured
what it appeared to measure.

## Container images

| image | purpose |
|---|---|
| `exopoiesis/infra-omat-gpu:latest` | the torch stack |
| `exopoiesis/infra-grace-gpu:latest` | the TensorFlow stack for GRACE |

Both build from the Dockerfiles in the project's `infra/docker/` with an empty build context.
