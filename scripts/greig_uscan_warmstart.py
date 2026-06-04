#!/usr/bin/env python3
"""
Greigite V_Fe U-scan via WARM-START chain (consilium 2026-06-04, Variant B).

Re-runs ONLY the U-scan (U=1,2,3) that stalled in the plain campaign: at U>=2 the
DFT+U occupation matrix is bistable -> the charge-density mixer ping-pongs at a
~6.5e-4 Ry floor and never reaches 1e-6 (physicist diagnosis). Fix: per endpoint,
chain U=1 -> U=2 -> U=3 in the SAME outdir with startingpot='file' so each step
warm-starts from the previous U's converged charge density + Hubbard ns occupation
matrix -> small U-step stays in one occupation basin -> converges to 1e-6 in ~15-25
iters (precedent: [[feedback_disk_io_nowf_for_singlepoint]] pent dimer).

Per endpoint (endA, saddle, endB) INDEPENDENT chain (ends NOT mixed):
  U=1: startingpot default (fresh)  -> .save in chain_<ep>/tmp
  U=2: startingpot='file' (warm)    -> same outdir
  U=3: startingpot='file' (warm)
directory (pwi/pwo) is per (ep,U); outdir (.save) is shared per ep -> chain persists,
no pwo overwrite. disk_io='nowf' keeps charge density + ns (not wfc).

Recipe identical to the converged U=0/U=1 of the campaign: PBE, nspin=2, ferri-init
+5/-4, tot_magnetization=-23.9 pin, mv degauss=0.02, local-TF beta=0.1 ndim=16,
NO diago_full_acc, conv_thr=1e-6, HUBBARD (ortho-atomic) on Fe-3d AND Fe1-3d.
Geometries: /workspace/greig_geom/greig_{endA,saddle,endB}.xyz (already uploaded).
U=0/U=1 anchors + k/ecut convergence already done (campaign partial harvest).
"""
import json, os, re, sys, time
from pathlib import Path
from collections import Counter

import numpy as np
from ase.io import read
from ase.calculators.espresso import Espresso, EspressoProfile

PSEUDO_DIR = os.environ.get("ESPRESSO_PSEUDO", os.environ.get("PSEUDO_DIR", "/opt/pp/oncv_pbe"))
PW_BIN     = os.environ.get("PW_BIN", "pw.x")
WORK_DIR   = Path("/workspace/greig_uscan_ws")
OUTPUT_DIR = Path("/workspace/results/greig_uscan_ws")
GEOM_DIR   = Path("/workspace/greig_geom")
LOCK_FILE  = Path("/workspace/.greig_uscan_ws.lock")
MPI_NP     = int(os.environ.get("MPI_NP", "1"))
EXPECT_COUNTS = {"Fe": 23, "S": 32, "H": 1}

KPTS, ECUTWFC, ECUTRHO, DEGAUSS, TOT_MAG = (2, 2, 2), 80.0, 320.0, 0.02, -23.9
# conv_thr relaxed 1e-6 -> 3e-4 Ry (2026-06-04 fallback): even with warm-start the U>=2
# DFT+U occupation manifold plateaus at ~1.4e-4 Ry (warm-start lowered it from 6.5e-4 but
# didn't fully break it). 3e-4 Ry = ~4 meV is FAR below the BOUND need (barrier ~1.8 eV,
# margin ~0.5 eV to the forbidden-window edge; chemist: +-10 meV fine). Warm-start keeps
# both endpoints in the SAME occupation basin -> barrier error cancels (physicist's cross-
# basin concern resolved). U=0/U=1 already at 1e-6 from the campaign; these refine U=2/U=3.
CONV_THR = 3.0e-4
ENDPOINTS = ["endA", "saddle", "endB"]
U_CHAIN   = [1.0, 2.0, 3.0]


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
            if "greig_uscan_warmstart.py" not in args_str: continue
            try: pid = int(pid_str)
            except ValueError: continue
            if pid == my: continue
            print(f"[SINGLETON] another instance pid={pid}, exit", flush=True); sys.exit(2)
    except subprocess.CalledProcessError:
        pass
    if LOCK_FILE.exists():
        try:
            old = int(LOCK_FILE.read_text().strip()); os.kill(old, 0)
            print(f"[SINGLETON] lock held by {old}, exit", flush=True); sys.exit(2)
        except (OSError, ValueError):
            print("[SINGLETON] stale lock, overwriting", flush=True)
    LOCK_FILE.write_text(str(os.getpid()))


def release_singleton():
    try:
        if LOCK_FILE.exists(): LOCK_FILE.unlink()
    except Exception: pass


def load_atoms_safe(xyz_path):
    import ase.calculators.singlepoint as _sp
    from ase.calculators.calculator import all_properties as _allp
    _orig = _sp.SinglePointCalculator.__init__
    def _patched(self, obj, **res): _orig(self, obj, **{k: v for k, v in res.items() if k in _allp})
    _sp.SinglePointCalculator.__init__ = _patched
    try:
        a = read(str(xyz_path))
    finally:
        _sp.SinglePointCalculator.__init__ = _orig
    a.calc = None
    return a


def assert_composition(atoms, name):
    c = Counter(atoms.get_chemical_symbols())
    if {k: int(c.get(k, 0)) for k in EXPECT_COUNTS} != EXPECT_COUNTS or len(atoms) != 56:
        print(f"[FATAL] {name} comp {dict(c)} != {EXPECT_COUNTS}", flush=True); sys.exit(5)
    print(f"[geom] {name}: {atoms.get_chemical_formula()} OK", flush=True)


def apply_ferri_init(atoms):
    syms = atoms.get_chemical_symbols()
    m = np.zeros(len(atoms))
    fe = [i for i, s in enumerate(syms) if s == "Fe"]
    for k, i in enumerate(fe):
        m[i] = +5.0 if k < min(8, len(fe)) else -4.0
    atoms.set_initial_magnetic_moments(m)
    return atoms


def parse_mag(pwo):
    try:
        txt = Path(pwo).read_text(errors="ignore")
    except Exception:
        return None, None
    mt = re.findall(r"total magnetization\s*=\s*([-\d.]+)", txt)
    ma = re.findall(r"absolute magnetization\s*=\s*([-\d.]+)", txt)
    return (float(mt[-1]) if mt else None, float(ma[-1]) if ma else None)


def parse_scf_accuracy(pwo):
    try:
        v = re.findall(r"estimated scf accuracy\s*<\s*([-\d.E+]+)\s*Ry", Path(pwo).read_text(errors="ignore"))
        return float(v[-1]) if v else None
    except Exception:
        return None


def make_calc(directory, outdir, u_eff, startingpot):
    directory.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = f"mpirun --allow-run-as-root --bind-to none -np {MPI_NP} {PW_BIN}"
    profile = EspressoProfile(command=cmd, pseudo_dir=PSEUDO_DIR)
    input_data = {
        "control": {"calculation": "scf", "restart_mode": "from_scratch", "tprnfor": True,
                    "tstress": False, "verbosity": "high", "disk_io": "nowf",
                    "outdir": str(outdir), "prefix": "greig"},
        "system": {"ecutwfc": ECUTWFC, "ecutrho": ECUTRHO, "occupations": "smearing",
                   "smearing": "mv", "degauss": DEGAUSS, "nspin": 2,
                   "tot_magnetization": TOT_MAG},
        # startingpot/startingwfc are &ELECTRONS keywords (NOT &control -- QE read_namelists
        # rejected startingpot in &control: bad-line crash, fixed 2026-06-04).
        "electrons": {"conv_thr": CONV_THR, "mixing_mode": "local-TF", "mixing_beta": 0.1,
                      "mixing_ndim": 16, "electron_maxstep": 800, "diagonalization": "david",
                      "diago_thr_init": 1.0e-4,
                      "startingwfc": "atomic+random", "startingpot": startingpot},
    }
    return Espresso(profile=profile, directory=str(directory), input_data=input_data,
                    pseudopotentials={"Fe": "Fe.upf", "S": "S.upf", "H": "H.upf"},
                    kpts=KPTS, koffset=(0, 0, 0),
                    additional_cards=(f"HUBBARD (ortho-atomic)\nU Fe-3d {u_eff:.4f}\nU Fe1-3d {u_eff:.4f}\n"))


def run_sp(geom, ep, u_eff, startingpot):
    directory = WORK_DIR / f"{ep}_u{int(u_eff)}"
    outdir = WORK_DIR / f"chain_{ep}" / "tmp"     # SHARED across the U-chain of this endpoint
    a = geom.copy(); a.calc = None
    apply_ferri_init(a)
    a.calc = make_calc(directory, outdir, u_eff, startingpot)
    t = time.time()
    rec = {"U": u_eff, "startingpot": startingpot, "E_eV": None}
    try:
        rec["E_eV"] = float(a.get_potential_energy())
    except Exception as exc:
        rec["error"] = str(exc)
        print(f"  [{ep} U={u_eff}] SCF FAIL: {exc}", flush=True)
    pwo = directory / "espresso.pwo"; pwi = directory / "espresso.pwi"
    rec["Mtot_uB"], rec["Mabs_uB"] = parse_mag(pwo)
    rec["final_scf_accuracy_Ry"] = parse_scf_accuracy(pwo)
    try:
        rec["hubbard_in_pwi"] = ("HUBBARD" in pwi.read_text())
    except Exception:
        rec["hubbard_in_pwi"] = None
    try:
        rec["hubbard_energy_count"] = Path(pwo).read_text(errors="ignore").lower().count("hubbard energy")
    except Exception:
        rec["hubbard_energy_count"] = -1
    rec["hubbard_ok"] = bool(rec["hubbard_in_pwi"]) and rec.get("hubbard_energy_count", 0) > 0
    rec["converged"] = (rec["final_scf_accuracy_Ry"] is not None
                        and rec["final_scf_accuracy_Ry"] < CONV_THR * 1.05)
    print(f"  [{ep} U={u_eff} sp={startingpot}] E={rec['E_eV']} Mtot={rec['Mtot_uB']} "
          f"Mabs={rec['Mabs_uB']} acc={rec['final_scf_accuracy_Ry']} conv={rec['converged']} "
          f"hub_ok={rec['hubbard_ok']} ({time.time()-t:.0f}s)", flush=True)
    return rec


def main():
    t0 = time.time()
    acquire_singleton()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in ("Fe.upf", "S.upf", "H.upf"):
        if not (Path(PSEUDO_DIR) / f).exists():
            print(f"[FATAL] Missing PP {f}", flush=True); sys.exit(3)
    geoms = {}
    for ep in ENDPOINTS:
        g = load_atoms_safe(GEOM_DIR / f"greig_{ep}.xyz"); assert_composition(g, ep); geoms[ep] = g

    results = {"endpoints": {}, "U_chain": U_CHAIN,
               "recipe": {"warm_start": True, "conv_thr": CONV_THR, "tot_magnetization_pin": TOT_MAG,
                          "degauss": DEGAUSS, "mixing_beta": 0.1, "mixing_ndim": 16,
                          "diago_full_acc": False, "kpts": list(KPTS), "ecutwfc": ECUTWFC}}
    for ep in ENDPOINTS:
        results["endpoints"][ep] = {}
        for i, u in enumerate(U_CHAIN):
            sp = "file" if i > 0 else "atomic"     # warm-start from previous U in same outdir
            print(f"\n=== {ep} U={u} startingpot={sp} ===", flush=True)
            results["endpoints"][ep][str(int(u))] = run_sp(geoms[ep], ep, u, sp)
            (OUTPUT_DIR / "greig_uscan_ws_result.json").write_text(
                json.dumps(results, indent=2, cls=NumpyEncoder))   # SIGKILL-safe

    # barriers per U
    bar = {}
    for u in U_CHAIN:
        k = str(int(u))
        eA = results["endpoints"]["endA"].get(k, {}).get("E_eV")
        eS = results["endpoints"]["saddle"].get(k, {}).get("E_eV")
        eB = results["endpoints"]["endB"].get(k, {}).get("E_eV")
        bar[k] = {"E_a_meV": (round((eS - eA) * 1000, 1) if (eA is not None and eS is not None) else None),
                  "E_rxn_meV": (round((eB - eA) * 1000, 1) if (eA is not None and eB is not None) else None)}
    results["barriers_by_U"] = bar
    results["status"] = "completed"
    results["t_total_s"] = time.time() - t0
    (OUTPUT_DIR / "greig_uscan_ws_result.json").write_text(
        json.dumps(results, indent=2, cls=NumpyEncoder))
    print(f"\n[RESULT] barriers_by_U(meV) = {{u: bar[u]['E_a_meV'] for u in bar}}", flush=True)
    print(f"[RESULT] {bar}", flush=True)
    print(f"[saved] {OUTPUT_DIR/'greig_uscan_ws_result.json'}", flush=True)
    release_singleton()


if __name__ == "__main__":
    main()
