#!/usr/bin/env python3
"""
Greigite V_Fe barrier -- k/ecut CONVERGENCE + U-sensitivity SP campaign (Stanford Q1+Q2).

Reuses the PROVEN greigite_u2_sp.py machinery (ferri-init Fe_tet+5/Fe_oct-4, HUBBARD
via additional_cards for ASE>=3.23, HUBBARD/Hubbard-energy guards) but:
  - mixing_beta 0.2 -> 0.1  (FIX: the killed U=2 SP sloshed at beta=0.2 on the -28 muB
    ferrimagnet; diagnosed 2026-06-04, GREIGITE_PLAN §2).
  - parameterised per config (kpts, ecutwfc, ecutrho, U_eff).
  - electron_maxstep 500 -> 800 (killed run hit 500 w/o converging; beta=0.1 is gentler).
  - reads 3 explicit geometries (greig_endA/saddle/endB.xyz uploaded next to it).

Configs (E_a = E_saddle - E_endA per config):
  u0_base  (2,2,2) e80  U=0  endA,saddle,endB   -> U=0 anchor (repro ~1.86) + E_rxn
  u0_kdns  (3,3,3) e80  U=0  endA,saddle         -> dEa(k 222->333)
  u0_ecut  (2,2,2) e100 U=0  endA,saddle         -> dEa(ecut 80->100)
  u1       (2,2,2) e80  U=1  endA,saddle,endB    -> Q2
  u2       (2,2,2) e80  U=2  endA,saddle,endB    -> Q2
  u3       (2,2,2) e80  U=3  endA,saddle,endB    -> Q2

Q1 = dEa(k), dEa(ecut) at U=0. Q2 = E_a(U=0,1,2,3) spread. Goal (GREIGITE_PLAN §0):
BOUND the barrier across U+k; "kinetically forbidden at 298 K" robust for any 1.3-2.3 eV.

Recipe (consilium 2026-06-04): PBE, nspin=2, ferri-init (+5/-4) + tot_magnetization=-23.9 pin
(production NEB sheet, all configs), mv(=cold in QE) degauss=0.02, local-TF beta=0.1 ndim=16,
conv_thr=1e-6, diago david thr_init=1e-4 (NO diago_full_acc -- s162: it diverges the Fe-S ferri
saddle), startingwfc atomic+random, electron_maxstep=800, disk_io=nowf. U via HUBBARD
(ortho-atomic) on Fe-3d AND Fe1-3d. NOTE for SI: report INTRA-campaign deltas dEa(k)/dEa(ecut)/
E_a(U) vs u0_base anchor, NOT vs the production gaussian-NEB 1.86 eV (different smearing).
"""
import json, os, re, sys, time, traceback
from pathlib import Path
from collections import Counter

import numpy as np
from ase.io import read, write
from ase.calculators.espresso import Espresso, EspressoProfile

PSEUDO_DIR = os.environ.get("ESPRESSO_PSEUDO", os.environ.get("PSEUDO_DIR", "/opt/pp/oncv_pbe"))
PW_BIN     = os.environ.get("PW_BIN", "pw.x")
WORK_DIR   = Path("/workspace/greig_campaign")
OUTPUT_DIR = Path("/workspace/results/greig_campaign")
GEOM_DIR   = Path("/workspace/greig_geom")          # greig_endA/saddle/endB.xyz uploaded here
LOCK_FILE  = Path("/workspace/.greig_campaign.lock")
MPI_NP     = int(os.environ.get("MPI_NP", "1"))
EXPECT_COUNTS = {"Fe": 23, "S": 32, "H": 1}          # greigite V_Fe + H, 56 atoms

DEGAUSS = 0.02      # C3 (physicist): 0.015->0.02, safer vs occupation-flicker on coarse k(2,2,2)
TOT_MAG = -23.9     # M1 (physicist): pin production NEB ferri sheet (-23.9 uB) for ALL configs
                    # -> apples-to-apples across k/ecut/U + damps the Mtot flicker that helped kill the U=2 run
# (label, kpts, ecutwfc, ecutrho, U_eff, points)
CONFIGS = [
    ("u0_base_k222_e80",  (2, 2, 2),  80.0, 320.0, 0.0, ["endA", "saddle", "endB"]),
    ("u0_kdns_k333_e80",  (3, 3, 3),  80.0, 320.0, 0.0, ["endA", "saddle"]),
    ("u0_ecut_k222_e100", (2, 2, 2), 100.0, 400.0, 0.0, ["endA", "saddle"]),
    ("u1_k222_e80",       (2, 2, 2),  80.0, 320.0, 1.0, ["endA", "saddle", "endB"]),
    ("u2_k222_e80",       (2, 2, 2),  80.0, 320.0, 2.0, ["endA", "saddle", "endB"]),
    ("u3_k222_e80",       (2, 2, 2),  80.0, 320.0, 3.0, ["endA", "saddle", "endB"]),
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
            if "greig_sp_campaign.py" not in args_str: continue
            try: pid = int(pid_str)
            except ValueError: continue
            if pid == my: continue
            print(f"[SINGLETON] another instance pid={pid} running, exit", flush=True); sys.exit(2)
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
    got = {k: int(c.get(k, 0)) for k in EXPECT_COUNTS}
    if got != EXPECT_COUNTS or len(atoms) != sum(EXPECT_COUNTS.values()):
        print(f"[FATAL] {name} composition {dict(c)} (n={len(atoms)}) != {EXPECT_COUNTS}", flush=True)
        sys.exit(5)
    print(f"[geom] {name}: {atoms.get_chemical_formula()} ({len(atoms)} at) OK", flush=True)


def apply_greigite_ferri_init(atoms):
    """Fe_tet (first 8 Fe) +5 muB, Fe_oct (rest) -4 muB. (greigite Fd-3m build order)."""
    syms = atoms.get_chemical_symbols()
    magmoms = np.zeros(len(atoms))
    fe_idx = [i for i, s in enumerate(syms) if s == "Fe"]
    n_tet = min(8, len(fe_idx))
    for k, i in enumerate(fe_idx):
        magmoms[i] = +5.0 if k < n_tet else -4.0
    atoms.set_initial_magnetic_moments(magmoms)
    print(f"  [FERRI init] {int((magmoms>0).sum())} Fe_tet +5, {int((magmoms<0).sum())} Fe_oct -4, "
          f"net={magmoms.sum():.1f} muB", flush=True)
    return atoms


def make_calc(label, work_subdir, kpts, ecutwfc, ecutrho, u_eff):
    work_subdir.mkdir(parents=True, exist_ok=True)
    cmd = f"mpirun --allow-run-as-root --bind-to none -np {MPI_NP} {PW_BIN}"
    profile = EspressoProfile(command=cmd, pseudo_dir=PSEUDO_DIR)
    input_data = {
        "control": {"calculation": "scf", "restart_mode": "from_scratch", "tprnfor": True,
                    "tstress": False, "verbosity": "high", "disk_io": "nowf",
                    "outdir": str(work_subdir / "tmp"), "prefix": label},
        "system": {"ecutwfc": ecutwfc, "ecutrho": ecutrho, "occupations": "smearing",
                   "smearing": "mv", "degauss": DEGAUSS, "nspin": 2,
                   "tot_magnetization": TOT_MAG},   # M1: pin ferri sheet
        # C1 (physicist, s162 lesson): NO diago_full_acc -- it forces all-band diagonalisation
        # incl. dense empty bands near E_F on the Fe-S ferri-metal -> Davidson divergence at the
        # saddle (the likely real killer, beyond beta). ndim 12->16 (M3) for stubborn sloshing.
        "electrons": {"conv_thr": 1.0e-6, "mixing_mode": "local-TF", "mixing_beta": 0.1,
                      "mixing_ndim": 16, "electron_maxstep": 800, "diagonalization": "david",
                      "diago_thr_init": 1.0e-4,
                      "startingwfc": "atomic+random"},
    }
    kw = dict(profile=profile, directory=str(work_subdir), input_data=input_data,
              pseudopotentials={"Fe": "Fe.upf", "S": "S.upf", "H": "H.upf"},
              kpts=kpts, koffset=(0, 0, 0))
    if u_eff > 1e-6:
        # ASE auto-splits Fe (+5)->"Fe", Fe (-4)->"Fe1"; U on BOTH sublattices.
        kw["additional_cards"] = (f"HUBBARD (ortho-atomic)\nU Fe-3d {u_eff:.4f}\nU Fe1-3d {u_eff:.4f}\n")
    return Espresso(**kw)


def parse_mag(pwo):
    try:
        txt = Path(pwo).read_text(errors="ignore")
    except Exception:
        return None, None
    mt = re.findall(r"total magnetization\s*=\s*([-\d.]+)", txt)
    ma = re.findall(r"absolute magnetization\s*=\s*([-\d.]+)", txt)
    return (float(mt[-1]) if mt else None, float(ma[-1]) if ma else None)


def run_sp(geom, label, work_subdir, kpts, ecutwfc, ecutrho, u_eff):
    a = geom.copy(); a.calc = None
    apply_greigite_ferri_init(a)
    a.calc = make_calc(label, work_subdir, kpts, ecutwfc, ecutrho, u_eff)
    t = time.time()
    rec = {"E_eV": None, "Mtot_uB": None, "Mabs_uB": None, "hubbard_ok": None}
    try:
        e = float(a.get_potential_energy())
        rec["E_eV"] = e
    except Exception as exc:
        print(f"  [{label}] SCF FAILED: {exc}", flush=True)
        rec["error"] = str(exc)
    pwi = work_subdir / "espresso.pwi"; pwo = work_subdir / "espresso.pwo"
    rec["Mtot_uB"], rec["Mabs_uB"] = parse_mag(pwo)
    # HUBBARD guards (only when U>0)
    if u_eff > 1e-6:
        try:
            txt = pwi.read_text(); rec["hubbard_in_pwi"] = ("HUBBARD" in txt)
        except Exception:
            rec["hubbard_in_pwi"] = None
        try:
            rec["hubbard_energy_count"] = Path(pwo).read_text(errors="ignore").lower().count("hubbard energy")
        except Exception:
            rec["hubbard_energy_count"] = -1
        rec["hubbard_ok"] = bool(rec.get("hubbard_in_pwi")) and (rec.get("hubbard_energy_count", 0) > 0)
        if rec["E_eV"] is not None and not rec["hubbard_ok"]:
            print(f"  [{label}] GUARD WARN: U={u_eff} requested but HUBBARD not confirmed "
                  f"(pwi={rec.get('hubbard_in_pwi')}, count={rec.get('hubbard_energy_count')}) "
                  f"-> energy may be PBE U=0, NOT trustworthy", flush=True)
    print(f"  [{label}] E={rec['E_eV']} Mtot={rec['Mtot_uB']} Mabs={rec['Mabs_uB']} "
          f"hub_ok={rec['hubbard_ok']} ({time.time()-t:.0f}s)", flush=True)
    return rec


def main():
    t0 = time.time()
    acquire_singleton()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in ("Fe.upf", "S.upf", "H.upf"):
        if not (Path(PSEUDO_DIR) / f).exists():
            print(f"[FATAL] Missing PP: {Path(PSEUDO_DIR)/f}", flush=True); sys.exit(3)
    geoms = {}
    for name in ("endA", "saddle", "endB"):
        p = GEOM_DIR / f"greig_{name}.xyz"
        if not p.exists():
            print(f"[FATAL] missing geom {p}", flush=True); sys.exit(4)
        g = load_atoms_safe(p); assert_composition(g, name); geoms[name] = g

    results = {"configs": {}, "geom_dir": str(GEOM_DIR), "expect_counts": EXPECT_COUNTS,
               "recipe": {"smearing": "mv", "degauss": DEGAUSS, "mixing_beta": 0.1,
                          "mixing_ndim": 16, "diago_full_acc": False, "conv_thr": 1e-6,
                          "tot_magnetization_pin": TOT_MAG, "ferri_init": "Fe_tet+5/Fe_oct-4",
                          "nspin": 2}}
    for label, kpts, ew, er, u, pts in CONFIGS:
        print(f"\n=== {label}: kpts={kpts} ecut={ew}/{er} U={u} points={pts} ===", flush=True)
        cfg = {"kpts": list(kpts), "ecutwfc": ew, "ecutrho": er, "U_eff": u, "points": {}}
        for pt in pts:
            cfg["points"][pt] = run_sp(geoms[pt], f"{label}_{pt}", WORK_DIR / f"{label}_{pt}",
                                       kpts, ew, er, u)
        eA = cfg["points"].get("endA", {}).get("E_eV")
        eS = cfg["points"].get("saddle", {}).get("E_eV")
        eB = cfg["points"].get("endB", {}).get("E_eV")
        cfg["E_a_eV"]  = (eS - eA) if (eA is not None and eS is not None) else None
        cfg["E_a_meV"] = (round(cfg["E_a_eV"] * 1000, 1) if cfg["E_a_eV"] is not None else None)
        cfg["E_rxn_eV"] = (eB - eA) if (eA is not None and eB is not None) else None
        results["configs"][label] = cfg
        (OUTPUT_DIR / "greig_campaign_result.json").write_text(
            json.dumps(results, indent=2, cls=NumpyEncoder))  # SIGKILL-safe

    def ea(l): return results["configs"].get(l, {}).get("E_a_meV")
    u0, kd, ec = ea("u0_base_k222_e80"), ea("u0_kdns_k333_e80"), ea("u0_ecut_k222_e100")
    def d(a, b): return (round(a - b, 1) if (a is not None and b is not None) else None)
    results["E_a_U0_meV"]    = u0
    results["dEa_kmesh_meV"] = d(kd, u0)
    results["dEa_ecut_meV"]  = d(ec, u0)
    results["E_a_U_scan_meV"] = {u: ea(l) for l, _, _, _, u, _ in CONFIGS if u > 0}
    results["status"] = "completed"
    results["t_total_s"] = time.time() - t0
    (OUTPUT_DIR / "greig_campaign_result.json").write_text(
        json.dumps(results, indent=2, cls=NumpyEncoder))
    print(f"\n[RESULT] E_a(U=0)={u0} meV | dEa(k 222->333)={results['dEa_kmesh_meV']} "
          f"| dEa(ecut 80->100)={results['dEa_ecut_meV']} | U-scan(meV)={results['E_a_U_scan_meV']}", flush=True)
    print(f"[saved] {OUTPUT_DIR/'greig_campaign_result.json'}", flush=True)
    release_singleton()


if __name__ == "__main__":
    main()
