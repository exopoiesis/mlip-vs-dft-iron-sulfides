"""Training-schedule selection by the PREREGISTERED rule (preregistration v2, §4).

Two-stage, so as not to select on in-sample:
  1) GATE (in-sample): barrier MAE on the training bands <= 10 meV. This is a sanity
     condition: a model that cannot reproduce what it has seen cannot be a ceiling.
  2) SELECTION among those that pass (out-of-sample): by force RMS on the VALIDATION
     file from the training log.

The holdout is not read here.
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
        print(f"  {tag}: no result")
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
    print(f"  {tag}: barrier MAE {mae:7.2f} meV (gate {GATE_MAE}) -> "
          f"{'passed' if passed else 'FAILED'}; valid. force RMS {vf} meV/Å")

ok = [r for r in rows if r[4] and r[3] is not None]
if ok:
    best = min(ok, key=lambda r: r[3])
    reason = "passed the gate, minimum validation force RMS"
else:
    # no scheme passed the gate: take the smallest MAE and EXPLICITLY flag this as a G5 violation
    best = min(rows, key=lambda r: r[2])
    reason = ("GATE PASSED BY NO SCHEME: per G5 this is a harness failure, the ladder cannot "
              "be treated as a ceiling; the smallest MAE is chosen, the result is flagged")
    print("\n  ⚠️ WARNING: " + reason)

chosen = dict(best[1])
chosen.update({"tag": best[0], "train_band_MAE_meV": best[2],
               "valid_force_rmse_meV_A": best[3],
               "gate_passed": bool(best[4]), "gate_threshold_meV": GATE_MAE,
               "reason": reason})
(OUTD / "chosen.json").write_text(json.dumps(chosen, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
print(f"\nCHOSEN: {best[0]}  w_E={chosen['energy_weight']} "
      f"w_F={chosen['forces_weight']} lr={chosen['lr']}")
print(f"  reason: {reason}")
