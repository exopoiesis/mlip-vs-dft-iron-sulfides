"""R2.3 step A: zero-shot MACE-MP-0 large on all five DFT bands.

Gives (1) the S0 rung, measured by the SAME code as the fine-tuned rungs, and
(2) anchors E_MACE0(b, 0) for pinning the energy scale of the training targets.

The model-loading path is copied verbatim from R2.2 band_runner.py, otherwise S0 would be
measured with a different harness than S1-S3.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from ase.io import read

BANDS = {
    "greigite_channel": "greigite_VFe_channel_band.extxyz",
    "greigite_cation": "greigite_VFe_cation_band.extxyz",
    "mackinawite": "mackinawite_VFe_band.extxyz",
    "pyrite_VS2": "pyrite_VS2_band.extxyz",
    "marcasite": "marcasite_VFe_band.extxyz",
}

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "/work/structures")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/work/out/zeroshot.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

from mace.calculators import mace_mp  # noqa: E402

t0 = time.time()
calc = mace_mp(model="large", dispersion=False, default_dtype="float64", device="cuda")
print(f"MACE-MP-0 large loaded in {time.time()-t0:.1f} s", flush=True)

result = {}
for band, fname in BANDS.items():
    imgs = read(SRC / fname, ":")
    e_mlip, f_mlip, e_dft, f_dft = [], [], [], []
    for im in imgs:
        e_dft.append(float(im.get_potential_energy()))
        f_dft.append(im.get_forces().tolist())
        a = im.copy()
        a.calc = calc
        e_mlip.append(float(a.get_potential_energy()))
        f_mlip.append(a.get_forces().tolist())

    e_mlip = np.array(e_mlip)
    e_dft_a = np.array(e_dft)
    prof_mlip = (e_mlip - e_mlip[0]) * 1000.0
    prof_dft = (e_dft_a - e_dft_a[0]) * 1000.0
    frms = [float(np.sqrt(np.mean((np.array(a) - np.array(b)) ** 2)))
            for a, b in zip(f_mlip, f_dft)]

    result[band] = {
        "n_images": len(imgs),
        "n_atoms": len(imgs[0]),
        "E_mlip_abs_eV": e_mlip.tolist(),
        "E_dft_abs_eV": e_dft_a.tolist(),
        "profile_mlip_meV": [round(x, 3) for x in prof_mlip],
        "profile_dft_meV": [round(x, 3) for x in prof_dft],
        "E_a_mlip_meV": round(float(prof_mlip.max()), 2),
        "E_a_dft_meV": round(float(prof_dft.max()), 2),
        "saddle_image_mlip": int(np.argmax(prof_mlip)),
        "saddle_image_dft": int(np.argmax(prof_dft)),
        "force_rms_vs_dft_eVA": [round(x, 4) for x in frms],
        "anchor_E_mlip_image0_eV": float(e_mlip[0]),
    }
    print(f"{band:18s} E_a(MACE)={result[band]['E_a_mlip_meV']:8.2f}  "
          f"E_a(DFT)={result[band]['E_a_dft_meV']:8.2f}  "
          f"saddle {result[band]['saddle_image_mlip']}/{result[band]['saddle_image_dft']}",
          flush=True)

OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

# --- gate G5: do we reproduce the published R2.2 number on the holdout
published = 171.86
got = result["marcasite"]["E_a_mlip_meV"]
delta = abs(got - published)
print(f"\nGATE G5: marcasite S0 = {got:.2f} meV, published {published:.2f}, "
      f"|Δ| = {delta:.2f} meV -> {'PASS' if delta <= 2.0 else 'FAIL'}")
if delta > 2.0:
    sys.exit(3)
