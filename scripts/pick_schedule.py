"""Выбор графика обучения по ПРЕДРЕГИСТРИРОВАННОМУ правилу (§4 предрегистрации v2).

Двухступенчато, чтобы не выбирать по in-sample:
  1) ГЕЙТ (in-sample): MAE барьеров на обучающих полосах <= 10 мэВ. Это условие осмысленности:
     модель, не воспроизводящая виденное, потолком быть не может.
  2) ОТБОР среди прошедших (out-of-sample): по RMS сил на ВАЛИДАЦИОННОМ файле из лога обучения.

Холдаут здесь не читается.
"""
import json
import re
from pathlib import Path

OUTD = Path("/work/out")
GATE_MAE = 10.0

CFG = {"tA": {"energy_weight": 100, "forces_weight": 100, "lr": 0.0001},
       "tB": {"energy_weight": 10000, "forces_weight": 100, "lr": 0.0001},
       "tC": {"energy_weight": 10000, "forces_weight": 100, "lr": 0.001}}

tune = json.loads((OUTD / "tune_v2.json").read_text())

rows = []
for tag, cfg in CFG.items():
    if tag not in tune:
        print(f"  {tag}: нет результата")
        continue
    mae = tune[tag]["train_band_barrier_MAE_meV"]
    log = Path(f"/work/runs/{tag}/train.log")
    vf = None
    if log.exists():
        ep = re.findall(r"Epoch \d+:.*RMSE_F=\s*([\d.]+)", log.read_text(errors="replace"))
        if ep:
            vf = float(ep[-1])
    passed = mae <= GATE_MAE
    rows.append((tag, cfg, mae, vf, passed))
    print(f"  {tag}: MAE барьеров {mae:7.2f} мэВ (гейт {GATE_MAE}) -> "
          f"{'прошёл' if passed else 'НЕ прошёл'}; валид. RMS сил {vf} мэВ/Å")

ok = [r for r in rows if r[4] and r[3] is not None]
if ok:
    best = min(ok, key=lambda r: r[3])
    reason = "прошёл гейт, минимальная валидационная RMS сил"
else:
    # ни одна не прошла гейт: берём наименьший MAE и ЯВНО помечаем это как нарушение П5
    best = min(rows, key=lambda r: r[2])
    reason = ("ГЕЙТ НЕ ПРОЙДЕН НИ ОДНОЙ СХЕМОЙ: по П5 это провал обвязки, лестницу нельзя "
              "трактовать как потолок; выбран наименьший MAE, результат помечается")
    print("\n  ⚠️ ВНИМАНИЕ: " + reason)

chosen = dict(best[1])
chosen.update({"tag": best[0], "train_band_MAE_meV": best[2],
               "valid_force_rmse_meV_A": best[3],
               "gate_passed": bool(best[4]), "gate_threshold_meV": GATE_MAE,
               "reason": reason})
(OUTD / "chosen.json").write_text(json.dumps(chosen, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
print(f"\nВЫБРАНО: {best[0]}  w_E={chosen['energy_weight']} "
      f"w_F={chosen['forces_weight']} lr={chosen['lr']}")
print(f"  причина: {reason}")
