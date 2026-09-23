"""R2.3 v2-final: выборки на СИММЕТРИЙНО-НЕЭКВИВАЛЕНТНЫХ конфигурациях.

Что изменено против build_sets2 и почему:
  * из обучающего пула убраны зеркальные половины обеих греигитовых полос. Образы i и 8-i там
    ИЗОМЕТРИЧНЫ (спектры расстояний совпадают до 1e-5 A), MACE воспроизводит их до 0.07-0.10 мэВ
    (2e-7 от абсолютной энергии = численный шум). Для эквивариантной модели это одно свидетельство.
    Держать их в пуле нельзя: валидация тогда почти всегда содержит двойника обучающей точки.
  * убран побитовый дубликат: греигит-канал:0 == греигит-катион:0.
  * валидация СТРАТИФИЦИРОВАНА - ровно по одной конфигурации с каждой обучающей полосы.
  * валидация одинакова на всех ступенях при данном сиде => дизайн ПАРНЫЙ.

Вызов: build_sets3.py <holdout> <seeds>
"""
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

import os

SRC = Path(os.environ.get("R23_SRC", "/work/structures"))
OUTD = Path(os.environ.get("R23_OUT", "/work/out"))
OUTD.mkdir(parents=True, exist_ok=True)
# нуль-шот общий для обеих лестниц, он не зависит от выбора холдаута
ZS = json.loads(Path(os.environ.get("R23_ZS", str(OUTD / "zeroshot.json"))).read_text())

HOLDOUT = sys.argv[1] if len(sys.argv) > 1 else "mackinawite"
SEEDS = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "1,2,3,4,5,6,7,8").split(",")]

BANDS = {
    "greigite_channel": "greigite_VFe_channel_band.extxyz",
    "greigite_cation": "greigite_VFe_cation_band.extxyz",
    "mackinawite": "mackinawite_VFe_band.extxyz",
    "pyrite_VS2": "pyrite_VS2_band.extxyz",
    "marcasite": "marcasite_VFe_band.extxyz",
}

# полный набор пригодных образов (маркаситовый образ 1 дефектен)
USABLE = {b: list(range(9)) for b in BANDS}
USABLE["marcasite"] = [0, 2, 3, 4, 5, 6, 7, 8]

# симметрийно-неэквивалентное подмножество для ОБУЧЕНИЯ
SYM_UNIQUE = {
    "greigite_channel": [0, 1, 2, 3, 4],   # 5..8 - зеркала 3..0
    "greigite_cation": [1, 2, 3, 4],       # 0 совпадает с greigite_channel:0
    "mackinawite": list(range(9)),
    "pyrite_VS2": list(range(9)),
    "marcasite": [0, 2, 3, 4, 5, 6, 7, 8],
}


def targets(band, images):
    imgs = read(SRC / BANDS[band], ":")
    e_dft = np.array([im.get_potential_energy() for im in imgs])
    anchor = ZS[band]["anchor_E_mlip_image0_eV"]
    out = {}
    for i in images:
        im = imgs[i]
        a = Atoms(symbols=im.get_chemical_symbols(), positions=im.get_positions(),
                  cell=im.get_cell(), pbc=True)
        a.info["REF_energy"] = float(e_dft[i] - e_dft[0] + anchor)
        a.info["band"] = band
        a.info["image"] = i
        a.arrays["REF_forces"] = im.get_forces().copy()
        out[i] = a
    return out


others = [b for b in BANDS if b != HOLDOUT]
by_band = {b: targets(b, SYM_UNIQUE[b]) for b in others}
hold = targets(HOLDOUT, USABLE[HOLDOUT])

n_raw = sum(len(USABLE[b]) for b in others)
n_uni = sum(len(SYM_UNIQUE[b]) for b in others)
print(f"холдаут: {HOLDOUT} ({len(hold)} образов)")
print(f"обучающий пул: {n_raw} образов -> {n_uni} симметрийно-неэквивалентных")
for b in others:
    print(f"    {b:18s} {len(USABLE[b])} -> {len(SYM_UNIQUE[b])}  образы {SYM_UNIQUE[b]}")

ends = [USABLE[HOLDOUT][0], USABLE[HOLDOUT][-1]]
RUNGS = {
    "S1": [],
    "S2": [hold[i] for i in ends],
    "S3": [hold[i] for i in USABLE[HOLDOUT]],
}

summary = {"holdout": HOLDOUT, "seeds": SEEDS,
           "n_raw_training_images": n_raw, "n_symmetry_unique": n_uni,
           "sym_unique_images": SYM_UNIQUE, "usable_images": USABLE, "rungs": {}}

for rung, extra in RUNGS.items():
    summary["rungs"][rung] = {"n_holdout_configs": len(extra), "per_seed": {}}
    for sd in SEEDS:
        rng = np.random.default_rng(1000 + sd)
        vset, tset = [], []
        for b in others:                      # ровно по одной валидационной с каждой полосы
            idxs = list(by_band[b])
            pick = int(rng.choice(idxs))
            for i in idxs:
                (vset if i == pick else tset).append(by_band[b][i])
        tset += extra
        write(OUTD / f"train_{rung}_s{sd}.xyz", tset, format="extxyz")
        write(OUTD / f"valid_{rung}_s{sd}.xyz", vset, format="extxyz")
        assert all(c.info["band"] != HOLDOUT for c in vset)
        assert sum(1 for c in tset if c.info["band"] == HOLDOUT) == len(extra)
        summary["rungs"][rung]["per_seed"][str(sd)] = {
            "n_train": len(tset), "n_valid": len(vset),
            "valid_configs": [f"{c.info['band']}:{c.info['image']}" for c in vset]}
    n = summary["rungs"][rung]["per_seed"][str(SEEDS[0])]["n_train"]
    print(f"  {rung}: обучение {n}, валидация 4, из холдаута {len(extra)}")

for sd in SEEDS:
    cfgs = [hold[i] for i in USABLE[HOLDOUT]]
    write(OUTD / f"train_S3solo_s{sd}.xyz", cfgs, format="extxyz")
    write(OUTD / f"valid_S3solo_s{sd}.xyz", cfgs, format="extxyz")
summary["rungs"]["S3solo"] = {
    "n_holdout_configs": len(hold),
    "note": "валидация = обучение; это ПОДГОНКА, контроль ёмкости, не обобщение",
    "per_seed": {str(s): {"n_train": len(hold), "n_valid": len(hold)} for s in SEEDS}}
print(f"  S3solo: обучение {len(hold)} (валидация = те же)")

(OUTD / f"datasets_{HOLDOUT}.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nзаписано datasets_{HOLDOUT}.json")
