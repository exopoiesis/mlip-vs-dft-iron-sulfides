#!/usr/bin/env python3
"""
Marcasite V_Fe barrier -- k-mesh & ecut CONVERGENCE single-points (Stanford Q1).

Runs SPs on the EXISTING converged warm-NEB geometries (no new NEB):
  endA  = /workspace/marc_warm_restart/relaxed_endA_warm.xyz
  saddle= /workspace/marc_warm_restart/image_04/espresso.pwo  (img4, the 208 meV CI)

ROBUSTNESS (consilium 2026-06-03): the 208-meV warm run crashed on an INTERMEDIATE
image (img2) via the mv-smearing Fermi-finder ("smearing larger than band-gap").
endA and saddle (img4) themselves converged cleanly, but denser k(3,3,4) can
re-trigger the Fermi-finder fragility. Per chemist option (B): anchor ONE point in
mv/0.015 (apples-to-apples with the 208 prod number), run the k/ecut scans in the
ROBUST gaussian/0.010 scheme, and report the mv->gaussian bridge so the prod number
stays connected. Gaussian Fermi-finding is monotone -> no Fermi-finder crash.

Configs (E_a = E_saddle - E_endA at each setting):
  mv_base    : (2,2,3) e60  mv/0.015     -> ANCHOR to prod 208 meV
  gauss_base : (2,2,3) e60  gaussian/0.01 -> robust baseline; dEa(smearing)=gauss_base-mv_base
  gauss_kdns : (3,3,4) e60  gaussian/0.01 -> dEa(k)   = gauss_kdns - gauss_base
  gauss_ecut : (2,2,3) e80  gaussian/0.01 -> dEa(ecut)= gauss_ecut - gauss_base

Q1 uncertainty band = combine dEa(k), dEa(ecut), dEa(smearing) -> replaces the bare
"<=100 meV" in SI §M.7 with measured numbers.

Methodology otherwise IDENTICAL to the 208-meV warm run (apples-to-apples):
  PBE U=0, nspin=2, tot_magnetization=1.1 pin, uniform Yang 0.065 muB/Fe init,
  mixing local-TF beta=0.1, diago_david_ndim=4, conv_thr=1e-8. ecutrho = 4*ecutwfc.
Per-SP magnetization (Mtot/Mabs) is parsed from each pwo and stored -> verifies the
single-sheet pin holds physically (consilium: magnetic sheet = dominant uncertainty).
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

WORK_ROOT  = Path("/workspace/marc_conv_scan")
OUTPUT_DIR = Path("/workspace/results/marc_conv_scan")
ENDA_XYZ   = Path("/workspace/marc_warm_restart/relaxed_endA_warm.xyz")
SADDLE_PWO = Path("/workspace/marc_warm_restart/image_04/espresso.pwo")
PW_BIN     = os.environ.get("PW_BIN", "pw.x")
PSEUDO_DIR = os.environ.get("ESPRESSO_PSEUDO",
             os.environ.get("PSEUDO_DIR", "/opt/pp/oncv_pbe"))
LOCK_FILE  = Path("/workspace/.marc_conv_scan.lock")
MPI_NP     = int(os.environ.get("MPI_NP", "1"))

EXPECT_COUNTS = {"Fe": 31, "S": 64, "H": 1}   # marc V_Fe + H defect, 96 atoms

# (label, kpts, ecutwfc, ecutrho, smearing, degauss)
CONFIGS = [
    ("mv_base_k223_e60",    (2, 2, 3), 60.0, 240.0, "mv",       0.015),
    ("gauss_base_k223_e60", (2, 2, 3), 60.0, 240.0, "gaussian", 0.010),
    ("gauss_kdns_k334_e60", (3, 3, 4), 60.0, 240.0, "gaussian", 0.010),
    ("gauss_ecut_k223_e80", (2, 2, 3), 80.0, 320.0, "gaussian", 0.010),
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
            if "marc_conv_scan.py" not in args_str: continue
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
    """Return (Mtot, Mabs) from the LAST occurrence in a QE pwo, else (None, None)."""
    try:
        txt = Path(pwo_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None, None
    mt = re.findall(r"total magnetization\s*=\s*([-\d.]+)", txt)
    ma = re.findall(r"absolute magnetization\s*=\s*([-\d.]+)", txt)
    return (float(mt[-1]) if mt else None, float(ma[-1]) if ma else None)


def make_calc(label, work_subdir, kpts, ecutwfc, ecutrho, smearing, degauss):
    """U=0 PBE, nspin=2, tot_magnetization=1.1 pin, local-TF beta=0.1,
    diago_david_ndim=4. smearing/degauss + kpts/ecut vary per config."""
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
        print(f"  [{label}] E={e:.6f} eV  Mtot={mt} Mabs={ma}  ({time.time()-t:.0f}s)",
              flush=True)
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
        (OUTPUT_DIR / "marc_conv_scan_result.json").write_text(
            json.dumps(results, indent=2, cls=NumpyEncoder))  # SIGKILL-safe incremental

    def ea(lbl): return results["configs"].get(lbl, {}).get("E_a_meV")
    mv_b   = ea("mv_base_k223_e60")
    g_b    = ea("gauss_base_k223_e60")
    g_kd   = ea("gauss_kdns_k334_e60")
    g_ec   = ea("gauss_ecut_k223_e80")
    def d(a, b): return (round(a - b, 1) if (a is not None and b is not None) else None)
    results["anchor_mv_Ea_meV"]   = mv_b
    results["dEa_smearing_meV"]   = d(g_b, mv_b)    # gaussian - mv (baseline)
    results["dEa_kmesh_meV"]      = d(g_kd, g_b)    # 223 -> 334 (gaussian)
    results["dEa_ecut_meV"]       = d(g_ec, g_b)    # e60 -> e80 (gaussian)
    results["status"] = "completed"
    results["t_total_s"] = time.time() - t0
    (OUTPUT_DIR / "marc_conv_scan_result.json").write_text(
        json.dumps(results, indent=2, cls=NumpyEncoder))
    print(f"\n[RESULT] anchor mv E_a={mv_b} meV | dEa(smearing mv->gauss)={results['dEa_smearing_meV']} "
          f"| dEa(k 223->334)={results['dEa_kmesh_meV']} | dEa(ecut 60->80)={results['dEa_ecut_meV']} meV",
          flush=True)
    print(f"[saved] {OUTPUT_DIR/'marc_conv_scan_result.json'}", flush=True)
    release_singleton()


if __name__ == "__main__":
    main()
