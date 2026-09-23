"""QA обучающих файлов v3: сверка с источником + структурные инварианты ступеней.

Проверяем на каждом сиде:
  1) REF_forces и позиции поатомно совпадают с исходной полосой депозита (порог 1e-7 -
     точность записи extxyz %16.8f, а не 1e-9, иначе ложные провалы);
  2) внутриполосный профиль REF_energy равен DFT-профилю;
  3) валидация НЕ содержит холдаутного минерала и ОДИНАКОВА на всех ступенях (парность);
  4) все конфигурации холдаута остались в обучении (ступень S3 обязана быть полной);
  5) зеркальные половины греигита в обучающий пул не попали.
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
        # профиль
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
                f"{rung} s{sd}: валидация без холдаута")
        n_hold = sum(1 for a in t if a.info["band"] == HOLD)
        exp = {"S1": 0, "S2": 2, "S3": len(D["usable_images"][HOLD]),
               "S3solo": len(D["usable_images"][HOLD])}[rung]
        chk(n_hold == exp, f"{rung} s{sd}: конфигураций холдаута в обучении {n_hold} (ждали {exp})")

print()
chk(worst_f < TOL_POS, f"силы совпадают с депозитом, макс |Δ| = {worst_f:.2e} эВ/Å")
chk(worst_p < TOL_POS, f"позиции совпадают, макс |Δ| = {worst_p:.2e} Å")
chk(worst_e < 1e-2, f"профили совпадают с DFT, макс |Δ| = {worst_e:.2e} мэВ")

# парность валидации
for sd in SEEDS:
    sets = []
    for rung in ("S1", "S2", "S3"):
        v = read(OUTD / f"valid_{rung}_s{sd}.xyz", ":")
        sets.append(tuple(sorted(f"{a.info['band']}:{a.info['image']}" for a in v)))
    chk(len(set(sets)) == 1, f"сид {sd}: валидация одинакова на S1/S2/S3 (парность)")

# зеркальные половины греигита не в обучении
t = read(OUTD / f"train_S1_s{SEEDS[0]}.xyz", ":")
mirr = [a for a in t if a.info["band"].startswith("greigite") and int(a.info["image"]) > 4]
chk(not mirr, f"зеркальных греигитовых образов в обучении: {len(mirr)}")

print("\n" + ("ВСЁ ЧИСТО" if not fails else f"ПРОВАЛОВ: {len(fails)}"))
sys.exit(1 if fails else 0)
