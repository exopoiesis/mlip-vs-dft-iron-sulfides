"""R2.3 v2: model evaluation. Fixes the G1 criterion and adds self-consistent NEB.

Call: evaluate2.py <model|FOUNDATION> <tag> <holdout> [--neb]
"""
import json
import sys
from pathlib import Path

import numpy as np
from ase.io import read
from ase.optimize import FIRE


class SafeJSONEncoder(json.JSONEncoder):
    """ASE returns numpy.bool_, not bool; json chokes on this right at the end of the run."""

    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


import os  # noqa: E402

SRC = Path(os.environ.get("R23_SRC", "/work/structures"))
OUTD = Path(os.environ.get("R23_OUT", "/work/out"))
OUTD.mkdir(parents=True, exist_ok=True)

BANDS = {
    "greigite_channel": "greigite_VFe_channel_band.extxyz",
    "greigite_cation": "greigite_VFe_cation_band.extxyz",
    "mackinawite": "mackinawite_VFe_band.extxyz",
    "pyrite_VS2": "pyrite_VS2_band.extxyz",
    "marcasite": "marcasite_VFe_band.extxyz",
}
USABLE = {b: list(range(9)) for b in BANDS}
USABLE["marcasite"] = [0, 2, 3, 4, 5, 6, 7, 8]
OOD = ["pentlandite_endA.extxyz", "pentlandite_endB.extxyz", "pyrite_VFe_dimer_saddle.extxyz"]

MODEL = sys.argv[1]
TAG = sys.argv[2]
HOLD = sys.argv[3]
DO_NEB = "--neb" in sys.argv

ZS = json.loads(Path(os.environ.get("R23_ZS", str(OUTD / "zeroshot.json"))).read_text())
if MODEL == "FOUNDATION":
    MODEL = "/root/.cache/mace/MACE_MPtrj_20229model"

TOL_DEPTH = 25.0     # meV below the minimum of the DFT profile


def check_P1_shape(prof, dft, saddle_dft):
    """Points 2 and 3 of the G1 criterion. Formulated RELATIVE TO THE REFERENCE."""
    return {
        "saddle_ok": bool(int(np.argmax(prof)) == saddle_dft),
        "no_spurious_basin": bool(prof.min() >= dft.min() - TOL_DEPTH),
        "min_profile_meV": round(float(prof.min()), 2),
        "min_dft_meV": round(float(dft.min()), 2),
    }


# ---- SELF-TEST: the criterion must pass on the DFT itself ----
selftest = {}
for band in BANDS:
    keep = USABLE[band]
    d = np.array(ZS[band]["profile_dft_meV"])[keep]
    d = d - d[0]
    selftest[band] = check_P1_shape(d, d, int(np.argmax(d)))
bad = [b for b, v in selftest.items() if not (v["saddle_ok"] and v["no_spurious_basin"])]
print(f"SELF-TEST G1 on DFT: {'PASS' if not bad else 'FAIL ' + str(bad)}")
if bad:
    sys.exit(5)

from mace.calculators import MACECalculator  # noqa: E402

calc = MACECalculator(model_paths=MODEL, device="cuda", default_dtype="float64")

out = {"model": MODEL, "tag": TAG, "holdout": HOLD, "P1_selftest": selftest,
       "bands": {}, "ood": {}}

for band in BANDS:
    imgs = read(SRC / BANDS[band], ":")
    keep = USABLE[band]
    e, frms = [], []
    for i in keep:
        a = imgs[i].copy()
        a.calc = calc
        e.append(a.get_potential_energy())
        frms.append(float(np.sqrt(np.mean((a.get_forces() - imgs[i].get_forces()) ** 2))))
    e = np.array(e)
    prof = (e - e[0]) * 1000.0
    dall = np.array(ZS[band]["profile_dft_meV"])
    dft = dall[keep] - dall[keep[0]]
    saddle_dft = int(np.argmax(dft))

    rec = {
        "images_used": keep,
        "profile_meV": [round(x, 3) for x in prof],
        "profile_dft_meV": [round(x, 3) for x in dft],
        "E_a_meV": round(float(prof.max()), 2),
        "E_a_dft_meV": round(float(dft.max()), 2),
        "err_meV": round(float(prof.max() - dft.max()), 2),
        "dE_rxn_meV": round(float(prof[-1] - prof[0]), 2),
        "dE_rxn_dft_meV": round(float(dft[-1] - dft[0]), 2),
        "dE_rxn_err_meV": round(float((prof[-1] - prof[0]) - (dft[-1] - dft[0])), 2),
        "saddle_image_dft": int(keep[saddle_dft]),
        "force_rms_vs_dft_eVA": [round(x, 4) for x in frms],
    }
    rec.update(check_P1_shape(prof, dft, saddle_dft))
    out["bands"][band] = rec

# ---- self-consistent NEB on the holdout: Table 2 convention from the paper ----
if DO_NEB:
    from ase.mep import NEB
    imgs = read(SRC / BANDS[HOLD], ":")
    a0, a1 = imgs[0].copy(), imgs[USABLE[HOLD][-1]].copy()
    e_ref = [imgs[0].get_potential_energy(),
             imgs[USABLE[HOLD][-1]].get_potential_energy()]
    relaxed = []
    for a in (a0, a1):
        a.calc = calc
        FIRE(a, logfile=None).run(fmax=0.03, steps=400)
        relaxed.append(a)
    bandims = [relaxed[0]] + [relaxed[0].copy() for _ in range(7)] + [relaxed[1]]
    neb = NEB(bandims, climb=True, k=0.3)
    # mic=True is mandatory: the cell is periodic, the hop can cross the boundary
    neb.interpolate("idpp", mic=True)
    # EACH image must have its OWN calculator - ASE checks this and fails on a shared one.
    # MACE weights are ~40 MB, nine instances cost nothing; memory is eaten by the forward pass, which is shared.
    for im in bandims:
        im.calc = MACECalculator(model_paths=MODEL, device="cuda", default_dtype="float64")
    conv = bool(FIRE(neb, logfile=None).run(fmax=0.05, steps=400))
    en = np.array([im.get_potential_energy() for im in bandims])
    p = (en - en[0]) * 1000.0
    out["neb_selfconsistent"] = {
        "converged": conv,
        "profile_meV": [round(float(x), 3) for x in p],
        "E_a_meV": round(float(p.max()), 2),
        "E_a_dft_meV": out["bands"][HOLD]["E_a_dft_meV"],
        "err_meV": round(float(p.max() - out["bands"][HOLD]["E_a_dft_meV"]), 2),
        "saddle_image": int(np.argmax(p)),
        "endpoint_relax_meV": [
            round(float(relaxed[0].get_potential_energy()) * 1000
                  - float(e_ref[0]) * 1000, 2),
            round(float(relaxed[1].get_potential_energy()) * 1000
                  - float(e_ref[1]) * 1000, 2)],
    }

for fname in OOD:
    p = SRC / fname
    if not p.exists():
        continue
    a = read(p)
    a.calc = calc
    out["ood"][fname] = {"E_eV": float(a.get_potential_energy()),
                         "n_atoms": len(a), "forces": a.get_forces().tolist()}
if all(f"pentlandite_end{x}.extxyz" in out["ood"] for x in "AB"):
    out["ood"]["pentlandite_dE_BA_meV"] = round(
        (out["ood"]["pentlandite_endB.extxyz"]["E_eV"]
         - out["ood"]["pentlandite_endA.extxyz"]["E_eV"]) * 1000, 2)

(OUTD / f"eval_{TAG}.json").write_text(
    json.dumps(out, indent=2, cls=SafeJSONEncoder), encoding="utf-8")

h = out["bands"][HOLD]
print(f"[{TAG}] HOLDOUT {HOLD}: single-point E_a={h['E_a_meV']:.2f} "
      f"(DFT {h['E_a_dft_meV']:.2f}, error {h['err_meV']:+.2f})  "
      f"saddle {'ok' if h['saddle_ok'] else 'NO'}  "
      f"spurious basins {'none' if h['no_spurious_basin'] else 'PRESENT'}  "
      f"ΔE_rxn error {h['dE_rxn_err_meV']:+.2f}")
if DO_NEB:
    n = out["neb_selfconsistent"]
    print(f"        self-consistent NEB: E_a={n['E_a_meV']:.2f} "
          f"error {n['err_meV']:+.2f}  converged={n['converged']}")
