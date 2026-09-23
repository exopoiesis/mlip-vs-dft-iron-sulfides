"""R2.3 v2: ladder aggregation. Paired analysis, all four rows of G2, exact test.

Rules taken from PREREGISTRATION_R23_finetune_v2_2026-09-22.md and are not invented here.
Call: aggregate2.py <holdout>
"""
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

import os

OUTD = Path(os.environ.get("R23_OUT", "/work/out"))
HOLD = sys.argv[1] if len(sys.argv) > 1 else "mackinawite"

TOL = 25.0            # G1-1: barrier tolerance, meV (kT at 290 K)
FORGET_BARRIER = 50.0  # G3 F1
FORGET_FORCE = 0.1     # G3 F2, eV/Å
RUNGS = ("S1", "S2", "S3", "S3solo")

try:
    from scipy.stats import t as tdist

    def tcrit(df):
        return float(tdist.ppf(0.975, df))
except Exception:                                   # table in case scipy is unavailable
    _T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
          7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086}

    def tcrit(df):
        return _T.get(df, 1.96)


evals = {}
for p in sorted(OUTD.glob("eval_*.json")):
    evals[p.stem[len("eval_"):]] = json.loads(p.read_text())
if "S0" not in evals:
    raise SystemExit("eval_S0.json missing")
S0 = evals["S0"]

# the criterion's self-test must pass
st = S0.get("P1_selftest", {})
bad = [b for b, v in st.items() if not (v["saddle_ok"] and v["no_spurious_basin"])]
print(f"SELF-TEST G1 on DFT: {'PASS' if not bad else 'FAIL ' + str(bad)}")
if bad:
    raise SystemExit("the criterion fails its own reference — the ladder cannot be interpreted")

by_rung = {}
for tag, d in evals.items():
    m = re.fullmatch(r"(S1|S2|S3|S3solo)_s(\d+)", tag)
    if m:
        by_rung.setdefault(m.group(1), {})[int(m.group(2))] = d

E_DFT = S0["bands"][HOLD]["E_a_dft_meV"]
report = {"holdout": HOLD, "E_a_dft_meV": E_DFT,
          "thresholds": {"tol_meV": TOL, "forget_barrier_meV": FORGET_BARRIER,
                         "forget_force_eVA": FORGET_FORCE},
          "rungs": {}}


def metric(d, kind):
    if kind == "neb" and "neb_selfconsistent" in d:
        return d["neb_selfconsistent"]
    return d["bands"][HOLD]


def drift(d):
    o = {}
    for k, v in d["ood"].items():
        if isinstance(v, dict) and "forces" in v:
            o[k] = float(np.sqrt(np.mean((np.array(v["forces"])
                                          - np.array(S0["ood"][k]["forces"])) ** 2)))
    return o


print(f"\nholdout {HOLD}, DFT barrier {E_DFT:.2f} meV\n")
hdr = f"{'rung':9s} {'n':>2s} {'E_a single-point':>22s} {'E_a self-consist. NEB':>22s} {'recov.':>7s}"
print(hdr)

for rung in ("S0",) + RUNGS:
    if rung == "S0":
        runs = {0: S0}
    else:
        runs = by_rung.get(rung, {})
        if not runs:
            continue
    seeds = sorted(runs)
    row = {"n_seeds": len(seeds), "seeds": seeds}

    for kind in ("sp", "neb"):
        # self-consistent NEB is not computed for every seed (it is expensive) - take the
        # subset where it exists, and honestly print its size rather than dropping the whole rung
        if kind == "neb":
            sub = [s for s in seeds if "neb_selfconsistent" in runs[s]]
            if not sub:
                row[kind] = None
                continue
            vals = [runs[s]["neb_selfconsistent"] for s in sub]
        else:
            sub = seeds
            vals = [metric(runs[s], kind) for s in sub]
        ea = np.array([v["E_a_meV"] for v in vals])
        err = np.array([v["err_meV"] for v in vals])
        if kind == "sp":
            shape_ok = np.mean([v["saddle_ok"] and v["no_spurious_basin"] for v in vals])
        else:
            shape_ok = np.mean([v["converged"] for v in vals])
        row[kind] = {
            "n": len(sub), "seeds_used": list(sub),
            "E_a_mean": round(float(ea.mean()), 2),
            "E_a_sd": round(float(ea.std(ddof=1)) if len(ea) > 1 else 0.0, 2),
            "err_mean": round(float(err.mean()), 2),
            "err_sd": round(float(err.std(ddof=1)) if len(err) > 1 else 0.0, 2),
            "err_per_seed": [round(float(x), 2) for x in err],
            "shape_ok_frac": float(shape_ok),
            "recovers": bool(abs(err.mean()) <= TOL and shape_ok == 1.0),
        }

    if rung != "S0":
        tb = {}
        for b in S0["bands"]:
            if b == HOLD:
                continue
            e = np.array([runs[s]["bands"][b]["err_meV"] for s in seeds])
            tb[b] = {"err_mean": round(float(e.mean()), 2),
                     "delta_vs_S0": round(float(abs(e.mean())
                                                - abs(S0["bands"][b]["err_meV"])), 2)}
        row["F1_train_bands"] = tb
        row["F1_flagged"] = [b for b, v in tb.items() if v["delta_vs_S0"] > FORGET_BARRIER]
        dr = [drift(runs[s]) for s in seeds]
        row["F2_force_drift"] = {k: round(float(np.mean([x[k] for x in dr])), 4) for k in dr[0]}
        row["F2_flagged"] = [k for k, v in row["F2_force_drift"].items() if v > FORGET_FORCE]

    report["rungs"][rung] = row
    sp, nb = row.get("sp"), row.get("neb")
    s_sp = f"{sp['E_a_mean']:8.2f}±{sp['E_a_sd']:<5.2f}({sp['err_mean']:+7.2f})" if sp else "-"
    s_nb = f"{nb['E_a_mean']:8.2f}±{nb['E_a_sd']:<5.2f}({nb['err_mean']:+7.2f})" if nb else "-"
    rec = "YES" if (nb or sp or {}).get("recovers") else "no"
    print(f"{rung:9s} {len(seeds):2d} {s_sp:>22s} {s_nb:>22s} {rec:>7s}")


# ---------- G2: paired discriminator ----------
def paired(a_rung, b_rung, kind):
    A, B = by_rung.get(a_rung, {}), by_rung.get(b_rung, {})
    seeds = sorted(set(A) & set(B))
    if len(seeds) < 2:
        return None
    da = np.array([abs(metric(A[s], kind)["err_meV"]) for s in seeds])
    db = np.array([abs(metric(B[s], kind)["err_meV"]) for s in seeds])
    d = da - db
    n = len(d)
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    half = tcrit(n - 1) * se
    # exact sign-permutation test
    obs = abs(mean)
    cnt = sum(1 for signs in itertools.product([1, -1], repeat=n)
              if abs(float(np.mean(d * np.array(signs)))) >= obs - 1e-12)
    p = cnt / 2 ** n
    return {"n_pairs": n, "seeds": seeds,
            "delta_per_seed": [round(float(x), 2) for x in d],
            "delta_mean": round(mean, 2), "delta_sd": round(sd, 2),
            "ci95": [round(mean - half, 2), round(mean + half, 2)],
            "p_exact": round(p, 4), "p_min_attainable": round(2 / 2 ** n, 4),
            "distinguishable": bool((mean - half) * (mean + half) > 0),
            "equivalent_TOST_25meV": bool(mean - half > -TOL and mean + half < TOL)}


report["discriminator"] = {}
for kind in ("sp", "neb"):
    r = paired("S1", "S3", kind)
    if not r:
        continue
    report["discriminator"][kind] = r
    print(f"\nG2 ({'single-point' if kind == 'sp' else 'self-consist. NEB'}): "
          f"Δ = |S1|−|S3| = {r['delta_mean']:+.2f} meV, "
          f"95% CI [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}], "
          f"p = {r['p_exact']} (minimum attainable {r['p_min_attainable']})")

    s1 = report["rungs"].get("S1", {}).get(kind) or {}
    s3 = report["rungs"].get("S3", {}).get(kind) or {}
    solo = report["rungs"].get("S3solo", {}).get(kind) or {}
    if s1.get("recovers") and s3.get("recovers"):
        v = "both recover: the failure is cured even without mineral-specific data"
    elif s3.get("recovers") and not s1.get("recovers"):
        v = "S3 recovers, S1 does not: DATA-COVERAGE DEFICIT, not a limit of the approach"
    elif s1.get("recovers") and not s3.get("recovers"):
        v = ("S1 recovers, S3 does not — per the preregistration this is a sign of a BROKEN "
             "HARNESS, not a result; the ladder cannot be interpreted")
    elif solo and solo.get("recovers"):
        v = "neither S1 nor S3, but S3solo does: SUPERVISION CONFLICT between minerals, not a capacity limit"
    else:
        v = "no rung recovers: LIMIT OF REPRESENTATION/CAPACITY at this budget"
    if not r["distinguishable"]:
        v += "  [95% CI contains zero -> rungs are INDISTINGUISHABLE, state it as such]"
    report["discriminator"][kind]["verdict"] = v
    print(f"VERDICT: {v}")

(OUTD / f"ladder_{HOLD}.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nwrote ladder_{HOLD}.json")
