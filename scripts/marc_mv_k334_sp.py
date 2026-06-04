#!/usr/bin/env python3
"""
Marcasite V_Fe -- MV-BRANCH k-mesh transferability SP (Stanford Q1 hedge removal).

The marc_conv_scan ran the k(2,2,3)->(3,3,4) delta in GAUSSIAN smearing (robust),
because mv (cold) smearing re-triggers the Fermi-finder crash at dense k. The quoted
primary 208.3 meV is on the mv/Mabs~2.5 sheet, so the manuscript currently says the
+0.4 meV k-delta is "assumed transferable" to the mv sheet (SI M.4, RESPONSE Q1).

This job removes the "assumed": run mv/0.015 at BOTH k(2,2,3) [reproduce the 208 anchor
in-job] and k(3,3,4) on the SAME endA + saddle geometries, and report
  dEa_k_mv = E_a(mv,k334) - E_a(mv,k223).
If |dEa_k_mv| is small (like the gaussian +0.4 meV) -> transferability VERIFIED, drop
"assumed". If the mv/k334 SCF hits the Fermi-finder crash on endA or saddle -> that
itself documents WHY the scan was done in gaussian, and the hedge stays (honest).

Methodology IDENTICAL to marc_conv_scan.py (apples-to-apples with the 208 number):
  PBE U=0, nspin=2, tot_magnetization=1.1 pin, uniform Yang 0.065 muB/Fe init,
  local-TF beta=0.1, mixing_ndim=12, diago_david_ndim=4, diago_full_acc=True,
  conv_thr=1e-8, disk_io=nowf, ecutrho=4*ecutwfc.
"""
import os
import re
import sys
import time
import json
from pathlib import Path

import numpy as np
from ase.io import read
from ase.calculators.espresso import Espresso, EspressoProfile

WORK_ROOT  = Path("/workspace/marc_mv_k334")
OUTPUT_DIR = Path("/workspace/results/marc_mv_k334")
ENDA_XYZ   = Path("/workspace/marc_warm_restart/relaxed_endA_warm.xyz")
SADDLE_PWO = Path("/workspace/marc_warm_restart/image_04/espresso.pwo")
PW_BIN     = os.environ.get("PW_BIN", "pw.x")
PSEUDO_DIR = os.environ.get("ESPRESSO_PSEUDO",
             os.environ.get("PSEUDO_DIR", "/opt/pp/oncv_pbe"))
LOCK_FILE  = Path("/workspace/.marc_mv_k334.lock")
MPI_NP     = int(os.environ.get("MPI_NP", "1"))

EXPECT_COUNTS = {"Fe": 31, "S": 64, "H": 1}   # marc V_Fe + H defect, 96 atoms

# (label, kpts, ecutwfc, ecutrho, smearing, degauss) -- BOTH mv to isolate the k-delta on the mv sheet
CONFIGS = [
    ("mv_base_k223_e60", (2, 2, 3), 60.0, 240.0, "mv", 0.015),  # reproduce 208 anchor in-job
    ("mv_kdns_k334_e60", (3, 3, 4), 60.0, 240.0, "mv", 0.015),  # the transferability test point
]


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)


def acquire_singleton():
    import subprocess
    try:
        out = subprocess.check_output(["ps", "-C", "python3", "-o", "pid=,args="], text=True)
        my = os.getpid()
        for line in out.strip().splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) < 2: continue
            pid_str, args_str = parts
            if "marc_mv_k334_sp.py" not in args_str: continue
            try: pid = int(pid_str)
            except ValueError: continue
            if pid == my: continue
            print(f"[SINGLETON] another instance pid={pid} running, exit", flush=True)
            sys.exit(2)
    except subprocess.CalledProcessError:
        pass
    if LOCK_FILE.exists():
        try:
            old = int(LOCK_FILE.read_text().strip()); os.kill(old, 0)
            print(f"[SINGLETON] lock held by pid={old}, exit", flush=True); sys.exit(2)
        except (OSError, ValueError):
            print("[SINGLETON] stale lock, overwriting", flush=True)
    LOCK_FILE.write_text(str(os.getpid()))


def release_singleton():
    try:
        if LOCK_FILE.exists(): LOCK_FILE.unlink()
    except Exception: pass


def load_geom_clean(path, fmt=None):
    """Read geometry, strip calculator (avoid SinglePointCalculator nspins clash)."""
    import ase.calculators.singlepoint as _sp
    from ase.calculators.calculator import all_properties as _allp
    _orig = _sp.SinglePointCalculator.__init__
    def _patched(self, obj, **res):
        _orig(self, obj, **{k: v for k, v in res.items() if k in _allp})
    _sp.SinglePointCalculator.__init__ = _patched
    try:
        a = read(str(path), format=fmt) if fmt else read(str(path))
    finally:
        _sp.SinglePointCalculator.__init__ = _orig
    a.calc = None
    return a


def assert_composition(atoms, name):
    from collections import Counter
    c = Counter(atoms.get_chemical_symbols())
    got = {k: int(c.get(k, 0)) for k in EXPECT_COUNTS}
    if got != EXPECT_COUNTS or len(atoms) != sum(EXPECT_COUNTS.values()):
        print(f"[FATAL] {name} composition {dict(c)} (n={len(atoms)}) != expected "
              f"{EXPECT_COUNTS} (n={sum(EXPECT_COUNTS.values())}) -- wrong geometry file?",
              flush=True)
        sys.exit(5)
    print(f"[geom] {name}: {atoms.get_chemical_formula()} ({len(atoms)} at) OK", flush=True)


def apply_uniform_init_marc(atoms, magmom_per_fe=0.065):
    syms = atoms.get_chemical_symbols()
    m = np.zeros(len(atoms))
    nfe = 0
    for i, s in enumerate(syms):
        if s == "Fe":
            m[i] = magmom_per_fe; nfe += 1
    atoms.set_initial_magnetic_moments(m)
    print(f"  [init] {nfe} Fe x {magmom_per_fe} = net {m.sum():.3f} muB", flush=True)
    return atoms


def parse_mag(pwo_path):
    try:
        txt = Path(pwo_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None, None
    mt = re.findall(r"total magnetization\s*=\s*([-\d.]+)", txt)
    ma = re.findall(r"absolute magnetization\s*=\s*([-\d.]+)", txt)
    return (float(mt[-1]) if mt else None, float(ma[-1]) if ma else None)


def make_calc(label, work_subdir, kpts, ecutwfc, ecutrho, smearing, degauss):
    work_subdir.mkdir(parents=True, exist_ok=True)
    cmd = f"mpirun --allow-run-as-root --bind-to none -np {MPI_NP} {PW_BIN}"
    profile = EspressoProfile(command=cmd, pseudo_dir=PSEUDO_DIR)
    input_data = {
        "control": {"calculation": "scf", "restart_mode": "from_scratch",
                    "tprnfor": True, "tstress": False, "verbosity": "high",
                    "disk_io": "nowf", "outdir": str(work_subdir / "tmp"),
                    "prefix": label},
        "system": {"ecutwfc": ecutwfc, "ecutrho": ecutrho,
                   "occupations": "smearing", "smearing": smearing, "degauss": degauss,
                   "nspin": 2, "tot_magnetization": 1.1},
        "electrons": {"conv_thr": 1e-8, "mixing_mode": "local-TF",
                      "mixing_beta": 0.1, "mixing_ndim": 12,
                      "electron_maxstep": 600, "diagonalization": "david",
                      "diago_david_ndim": 4, "diago_full_acc": True},
    }
    return Espresso(profile=profile, directory=str(work_subdir),
                    input_data=input_data,
                    pseudopotentials={"Fe": "Fe.upf", "S": "S.upf", "H": "H.upf"},
                    kpts=kpts, koffset=(0, 0, 0))


def run_sp(geom, label, work_subdir, kpts, ecutwfc, ecutrho, smearing, degauss):
    a = geom.copy()
    a.calc = None
    apply_uniform_init_marc(a)
    a.calc = make_calc(label, work_subdir, kpts, ecutwfc, ecutrho, smearing, degauss)
    t = time.time()
    try:
        e = float(a.get_potential_energy())
        mt, ma = parse_mag(work_subdir / "espresso.pwo")
        print(f"  [{label}] E={e:.6f} eV  Mtot={mt} Mabs={ma}  ({time.time()-t:.0f}s)", flush=True)
        return {"E_eV": e, "Mtot_uB": mt, "Mabs_uB": ma}
    except Exception as exc:
        print(f"  [{label}] SCF FAILED: {exc}", flush=True)
        mt, ma = parse_mag(work_subdir / "espresso.pwo")
        return {"E_eV": None, "Mtot_uB": mt, "Mabs_uB": ma, "error": str(exc)}


def main():
    t0 = time.time()
    acquire_singleton()
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for f in ("Fe.upf", "S.upf", "H.upf"):
        if not (Path(PSEUDO_DIR) / f).exists():
            print(f"[FATAL] Missing PP: {Path(PSEUDO_DIR)/f}", flush=True); sys.exit(3)
    if not ENDA_XYZ.exists():
        print(f"[FATAL] endA geom not found: {ENDA_XYZ}", flush=True); sys.exit(4)
    if not SADDLE_PWO.exists():
        print(f"[FATAL] saddle pwo not found: {SADDLE_PWO}", flush=True); sys.exit(4)

    endA = load_geom_clean(ENDA_XYZ)
    saddle = load_geom_clean(SADDLE_PWO, fmt="espresso-out")
    assert_composition(endA, "endA")
    assert_composition(saddle, "saddle")

    results = {"configs": {}, "geom": {"endA": str(ENDA_XYZ), "saddle": str(SADDLE_PWO)},
               "expect_counts": EXPECT_COUNTS}
    for label, kpts, ew, er, sm, dg in CONFIGS:
        print(f"\n=== {label}: kpts={kpts} ecutwfc={ew} ecutrho={er} smearing={sm} degauss={dg} ===",
              flush=True)
        rA = run_sp(endA, f"{label}_endA", WORK_ROOT / f"{label}_endA", kpts, ew, er, sm, dg)
        rS = run_sp(saddle, f"{label}_saddle", WORK_ROOT / f"{label}_saddle", kpts, ew, er, sm, dg)
        eA, eS = rA["E_eV"], rS["E_eV"]
        ea = (eS - eA) if (eA is not None and eS is not None) else None
        results["configs"][label] = {
            "kpts": list(kpts), "ecutwfc": ew, "ecutrho": er, "smearing": sm, "degauss": dg,
            "endA": rA, "saddle": rS,
            "E_a_eV": ea, "E_a_meV": (round(ea * 1000, 1) if ea is not None else None),
        }
        (OUTPUT_DIR / "marc_mv_k334_result.json").write_text(
            json.dumps(results, indent=2, cls=NumpyEncoder))  # SIGKILL-safe incremental

    def ea(lbl): return results["configs"].get(lbl, {}).get("E_a_meV")
    mv_b  = ea("mv_base_k223_e60")
    mv_kd = ea("mv_kdns_k334_e60")
    def d(a, b): return (round(a - b, 1) if (a is not None and b is not None) else None)
    results["mv_anchor_k223_Ea_meV"] = mv_b
    results["mv_kdns_k334_Ea_meV"]   = mv_kd
    results["dEa_kmesh_mv_meV"]      = d(mv_kd, mv_b)   # the transferability number
    results["status"] = "completed"
    results["t_total_s"] = time.time() - t0
    (OUTPUT_DIR / "marc_mv_k334_result.json").write_text(
        json.dumps(results, indent=2, cls=NumpyEncoder))
    print(f"\n[RESULT] mv k223 E_a={mv_b} meV | mv k334 E_a={mv_kd} meV | "
          f"dEa(k 223->334, MV branch)={results['dEa_kmesh_mv_meV']} meV "
          f"(gaussian branch gave +0.4 meV -- compare for transferability)", flush=True)
    print(f"[saved] {OUTPUT_DIR/'marc_mv_k334_result.json'}", flush=True)
    release_singleton()


if __name__ == "__main__":
    main()
