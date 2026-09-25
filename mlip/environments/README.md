# Software environments for the nine-model MLIP benchmark (2026-09)

Package manifests for the **nine-model fixed-geometry benchmark** (paper §3.4, Table 3).
Each `env_*.txt` records the `pip freeze` from the environment used for its model evaluations.

The older two-model self-consistent calculations (paper §3.3, Table 2) and the fine-tuning ladder
(§3.5) used `mace-torch` 0.3.15, `chgnet` 0.4.2, PyTorch 2.5.1+cu124, ASE 3.23.0,
Python 3.10.12 and CUDA 12.4. Their recorded containers were
`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` on the RTX 4070 and `exopoiesis/infra-mace-gpu`
on the A100. These component versions are documented in the scripts, fine-tuning README and
paper §2.3; **no contemporaneous complete package freeze exists for those older calculations**.
In particular, `mace-torch` is 0.3.15 there and 0.3.16 in `env_main.txt`.

## Three environments used for these runs

The runner used separate PyTorch, Orb and TensorFlow environments to handle the installed versions
and interfaces. This records the setup used, not a claim that all nine models can never coexist
in another environment.

| File | Environment | Results obtained in it |
|---|---|---|
| `env_main.txt` (204 packages) | torch 2.8.0+cu126, Python 3.12 | MACE-MP-0 large, CHGNet v0.3.0, SevenNet 7net-0, UMA-s-1p2 |
| `env_orb.txt` (84) | orb-models 0.5.5, torch 2.14.0 | Orb-v2, Orb-v3-conservative-inf-omat |
| `env_grace.txt` (77) | tensorpotential 0.6.1, TensorFlow 2.20.0 | GRACE-1L-OMAT, GRACE-2L-OMAT, GRACE-2L-OAM |

### Recorded installation limitations

The `orb-models==0.7.0` pin in `env_main.txt` produced no results: the runner could not find
`ORBCalculator` with that installation. All reported Orb evaluations used `env_orb.txt` and
orb-models 0.5.5. The unused pin remains in the unedited package manifest.

`pip` reported a conflict in `env_main.txt`: `fairchem-core 2.20.0 requires e3nn>=0.5`, while
e3nn 0.4.4 was installed. A MACE control on the greigite channel reproduced the independently
obtained 221.1 meV value. This checks one checkpoint and band; it does not establish that the
dependency conflict has no effect on every model or result.

## Models and weight provenance

| Model | Recorded source |
|---|---|
| MACE-MP-0 (large) | `github.com/ACEsuit/mace-mp` → `MACE_MPtrj_2022.9.model`, 133 803 220 B, `sha256 f80e992b65ab8f88fdf26964511357c022e92704e4d9bcd086652635a8495b32` |
| CHGNet v0.3.0 | weights bundled with the `chgnet` package |
| SevenNet `7net-0` | bundled with the `sevenn` package |
| UMA-s-1p2 | HuggingFace `facebook/UMA` (gated; licence accepted for these runs) |
| Orb-v2, Orb-v3-conservative-inf-omat | `orbital-materials`, downloaded by the package |
| GRACE-1L-OMAT, GRACE-2L-OMAT, GRACE-2L-OAM | HuggingFace `AMS-ICAMS-RUB/grace-foundation-models` |

Weights are not archived here or baked into the container images. The table records the model
sources and identifiers used; **a checkpoint checksum is provided only for MACE-MP-0**. The package
freezes and sources support reconstruction, but they are not a hash-complete lockfile for all model
weights, external downloads and container images. Follow the model providers' access and licence
terms when obtaining weights.

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

Every eSEN checkpoint in this particular API listing is associated with **OMol**, **OC25** or
**ODAC25**, and the listing contains no eqV2 entry. This describes the `fairchem-core 2.20.0`
registry used by our runner, not the availability of those architectures for bulk materials.

**Correction (2026-09-25):** bulk eSEN and EquiformerV2 checkpoints are available in the
[official OMat24 model repository](https://huggingface.co/facebook/OMAT24), including
eSEN-30M-OMat, eSEN-30M-OAM and eSEN-30M-MP, with loading instructions through the earlier
`OCPCalculator` interface. The [official eSEN release](https://huggingface.co/facebook/OMAT24/discussions/7)
was merged on 14 April 2025. Our previous explanation incorrectly generalized the contents of one
registry into a lack of bulk checkpoints and incorrectly generalized the reference functionals of
other domain-specific eSEN models. Those explanations are withdrawn. We did not evaluate bulk eSEN
or EquiformerV2 in this benchmark, and make no claim about their barrier accuracy or relaxation
reliability. The nine-model comparison is not an exhaustive evaluation of contemporary models.

## Container images

| image | purpose |
|---|---|
| `exopoiesis/infra-omat-gpu:latest` | the torch stack |
| `exopoiesis/infra-grace-gpu:latest` | the TensorFlow stack for GRACE |

These are recorded image names; the `latest` tags do not provide immutable image digests.
The project used Dockerfiles under `infra/docker/`; those definitions are not part of this
environment directory.
