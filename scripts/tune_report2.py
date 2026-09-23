"""Tuning report. BLIND TO THE HOLDOUT: the holdout band is not read here.

Criterion, declared in advance: a scheme is acceptable only if it fits the barriers of the
TRAINING bands to MAE <= 10 meV. If it cannot reproduce what it has seen — the S3 ceiling is
meaningless.

Call: tune_report2.py <holdout> <tag> [tag...]
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from ase.io import read

SRC = Path("/work/structures")
OUTD = Path("/work/out")
ZS = json.loads((OUTD / "zeroshot.json").read_text())

BANDS = {
    "greigite_channel": "greigite_VFe_channel_band.extxyz",
    "greigite_cation": "greigite_VFe_cation_band.extxyz",
    "mackinawite": "mackinawite_VFe_band.extxyz",
    "pyrite_VS2": "pyrite_VS2_band.extxyz",
    "marcasite": "marcasite_VFe_band.extxyz",
}
USABLE = {b: list(range(9)) for b in BANDS}
USABLE["marcasite"] = [0, 2, 3, 4, 5, 6, 7, 8]

HOLD = sys.argv[1]
TAGS = sys.argv[2:]
TRAIN = [b for b in BANDS if b != HOLD]

from mace.calculators import MACECalculator  # noqa: E402

THRESH = 10.0
rows = []
for tag in TAGS:
    run = Path(f"/work/runs/{tag}")
    model = run / f"{tag}.model"
    if not model.exists():
        print(f"{tag}: no model")
        continue
    log = (run / "train.log").read_text(errors="replace")
    ep = re.findall(r"Epoch (\d+):.*RMSE_E_per_atom=\s*([\d.]+) meV, RMSE_F=\s*([\d.]+)", log)
    last = ep[-1] if ep else ("-", "-", "-")

    calc = MACECalculator(model_paths=str(model), device="cuda", default_dtype="float64")
    errs, frms = {}, []
    for band in TRAIN:
        imgs = read(SRC / BANDS[band], ":")
        keep = USABLE[band]
        e = []
        for i in keep:
            a = imgs[i].copy()
            a.calc = calc
            e.append(a.get_potential_energy())
            frms.append(float(np.sqrt(np.mean((a.get_forces() - imgs[i].get_forces()) ** 2))))
        e = np.array(e)
        prof = (e - e[0]) * 1000.0
        dftall = np.array(ZS[band]["profile_dft_meV"])
        dft = dftall[keep] - dftall[keep[0]]
        errs[band] = float(prof.max() - dft.max())

    mae = float(np.mean([abs(v) for v in errs.values()]))
    rows.append((tag, mae, errs, float(np.mean(frms)), last))
    ok = "ACCEPTABLE" if mae <= THRESH else "not acceptable"
    print(f"\n### {tag}   epoch {last[0]}: valid RMSE_E {last[1]} meV/atom, RMSE_F {last[2]} meV/Å")
    for b, v in errs.items():
        print(f"    {b:18s} barrier error {v:+8.2f} meV")
    print(f"    MAE over training bands: {mae:7.2f} meV   (threshold {THRESH}) -> {ok}")
    print(f"    mean force RMS:          {rows[-1][3]:.4f} eV/Å")

if rows:
    best = min(rows, key=lambda r: r[1])
    print(f"\nBest: {best[0]}, MAE {best[1]:.2f} meV")
    print(f"Holdout ({HOLD}) was not read in this report.")
    json.dump({t: {"train_band_barrier_MAE_meV": m, "errs": e, "force_rms_eVA": f}
               for t, m, e, f, _l in rows},
              open(OUTD / "tune_v2.json", "w"), indent=2, ensure_ascii=False)
