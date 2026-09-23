"""QA v3 of the training files: cross-check against the source + structural invariants of the rungs.

Checked for each seed:
  1) REF_forces and positions match the source deposited band atom-by-atom (tolerance 1e-7 -
     the extxyz write precision is %16.8f, not 1e-9, otherwise the check gives false failures);
  2) the intra-band REF_energy profile equals the DFT profile;
  3) validation does NOT contain the holdout mineral and is IDENTICAL across all rungs (pairing);
  4) all holdout configurations remain in training (rung S3 must be complete);
  5) the mirrored halves of greigite did not end up in the training pool.
"""
import json
import sys
from pathlib import Path

import numpy as np
from ase.io import read

import os

SRC = Path(os.environ.get("R23_SRC", "/work/structures"))
OUTD = Path(os.environ.get("R23_OUT", "/work/out"))
HOLD = sys.argv[1] if len(sys.argv) > 1 else "mackinawite"
SEEDS = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "1,2,3,4,5,6,7,8").split(",")]

ZS = json.loads(Path(os.environ.get("R23_ZS", str(OUTD / "zeroshot.json"))).read_text())
D = json.loads((OUTD / f"datasets_{HOLD}.json").read_text())
BANDS = {
    "greigite_channel": "greigite_VFe_channel_band.extxyz",
    "greigite_cation": "greigite_VFe_cation_band.extxyz",
    "mackinawite": "mackinawite_VFe_band.extxyz",
    "pyrite_VS2": "pyrite_VS2_band.extxyz",
    "marcasite": "marcasite_VFe_band.extxyz",
}
src = {b: read(SRC / f, ":") for b, f in BANDS.items()}
TOL_POS = 1e-7

fails = []


def chk(ok, msg):
    if not ok:
        fails.append(msg)
    print(("  ok   " if ok else "  FAIL ") + msg)


worst_f = worst_p = worst_e = 0.0
for rung in ("S1", "S2", "S3", "S3solo"):
    for sd in SEEDS:
        t = read(OUTD / f"train_{rung}_s{sd}.xyz", ":")
        v = read(OUTD / f"valid_{rung}_s{sd}.xyz", ":")
        for a in t + v:
            b, i = a.info["band"], int(a.info["image"])
            s = src[b][i]
            worst_f = max(worst_f, float(np.abs(a.arrays["REF_forces"] - s.get_forces()).max()))
            worst_p = max(worst_p, float(np.abs(a.get_positions() - s.get_positions()).max()))
        # profile
        byb = {}
        for a in t + v:
            byb.setdefault(a.info["band"], {})[int(a.info["image"])] = a.info["REF_energy"]
        for b, e in byb.items():
            base = min(e)
            dft = np.array(ZS[b]["profile_dft_meV"])
            for i in e:
                worst_e = max(worst_e, abs((e[i] - e[base]) * 1000 - (dft[i] - dft[base])))
        if rung != "S3solo":
            chk(all(a.info["band"] != HOLD for a in v),
                f"{rung} s{sd}: validation excludes the holdout")
        n_hold = sum(1 for a in t if a.info["band"] == HOLD)
        exp = {"S1": 0, "S2": 2, "S3": len(D["usable_images"][HOLD]),
               "S3solo": len(D["usable_images"][HOLD])}[rung]
        chk(n_hold == exp, f"{rung} s{sd}: holdout configs in training {n_hold} (expected {exp})")

print()
chk(worst_f < TOL_POS, f"forces match the deposit, max |Δ| = {worst_f:.2e} eV/Å")
chk(worst_p < TOL_POS, f"positions match, max |Δ| = {worst_p:.2e} Å")
chk(worst_e < 1e-2, f"profiles match the DFT, max |Δ| = {worst_e:.2e} meV")

# validation pairing
for sd in SEEDS:
    sets = []
    for rung in ("S1", "S2", "S3"):
        v = read(OUTD / f"valid_{rung}_s{sd}.xyz", ":")
        sets.append(tuple(sorted(f"{a.info['band']}:{a.info['image']}" for a in v)))
    chk(len(set(sets)) == 1, f"seed {sd}: validation identical across S1/S2/S3 (pairing)")

# mirrored halves of greigite not in training
t = read(OUTD / f"train_S1_s{SEEDS[0]}.xyz", ":")
mirr = [a for a in t if a.info["band"].startswith("greigite") and int(a.info["image"]) > 4]
chk(not mirr, f"mirrored greigite images in training: {len(mirr)}")

print("\n" + ("ALL CLEAR" if not fails else f"FAILURES: {len(fails)}"))
sys.exit(1 if fails else 0)
