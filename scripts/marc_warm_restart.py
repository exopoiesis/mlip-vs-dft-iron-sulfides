#!/usr/bin/env python3
"""
Marc V_Fe warm restart: re-relax endA+endB from v4c band-endpoints, then warm NEB.

Source: v4c neb.traj frames 0 (endA) and 8 (endB) from last NEB iteration.
Input: /workspace/marc_v4c/neb.traj  (uploaded from local v4c harvest)

Steps:
1. Read v4c neb.traj, take last full 9-image batch: frames [0]/[-1] = endA/endB
2. Apply UNIFORM Yang-inspired init: 0.065 muB/Fe on each of 31 Fe atoms
   (net ~2.015 muB, matching Yang 2019 V_Fe 1.98 muB target).
   SAME init as v4c production (--magmom-uniform 0.065) + ZPE run.
   NOT AFM stripe (which forces wrong magnetic sheet incomparable with v4c/ZPE).
3. tot_magnetization=1.1 pinned in QE input (same sheet as v4c settled to
   Mtot~1.1, Mabs~1.7; ensures barrier comparable with 177 meV + ZPE -85).
4. Re-relax endA (BFGS, fmax=0.05) with local-TF beta=0.2, mv degauss=0.015
5. Re-relax endB (BFGS, fmax=0.05)
6. Guard: after endA SCF, check Mtot~1.1 (tol 0.3), Mabs~1.7 (tol 0.5).
   If settled to stripe (Mabs >> 10) -> ABORT with message.
7. Build 9 images from v4c band (frames 1..7 inner) with new relaxed endpoints
8. Launch CI-FIRE DyNEB, k_spring=0.3, fmax=0.05

QE params:
- PBE U=0 (nspin=2 for defect cell, U=0 per v4c baseline)
- ecutwfc=60, ecutrho=240
- kpts=(2,2,3), degauss=0.015, smearing=mv
- tot_magnetization=1.1 (pin to v4c/ZPE magnetic sheet)
- mixing_mode=local-TF, mixing_beta=0.2, mixing_ndim=12
- conv_thr=1e-8, electron_maxstep=500

Singleton guard + lockfile.

Magnetic init change log:
  s155 crash: AFM stripe +-2 muB -> WRONG sheet, incomparable with v4c 177 meV
  This version (2026-06-02): Yang uniform 0.065 muB/Fe + tot_mag=1.1 pin
  Reference: v4c --magmom-uniform 0.065 + ZPE marc_vfe_saddle_freq_dft.py
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.optimize import BFGS, FIRE
from ase.mep import DyNEB
from ase.calculators.espresso import Espresso, EspressoProfile

# ---- Config ----
WORK_DIR    = Path("/workspace/marc_warm_restart")
TRAJ_SRC    = Path("/workspace/marc_v4c/neb.traj")
OUTPUT_DIR  = Path("/workspace/results/marc_warm")
PW_BIN      = os.environ.get("PW_BIN", "pw.x")
PSEUDO_DIR  = os.environ.get("ESPRESSO_PSEUDO",
              os.environ.get("PSEUDO_DIR", "/opt/pp/oncv_pbe"))
LOCK_FILE   = Path("/workspace/.marc_warm_restart.lock")

ECUTWFC  = 60.0
ECUTRHO  = 240.0
KPTS     = (2, 2, 3)
DEGAUSS  = 0.015
FMAX_EP  = 0.05
FMAX_NEB = 0.05
K_SPRING = 0.3
N_IMAGES = 9
MPI_NP   = int(os.environ.get("MPI_NP", "1"))
OMP      = int(os.environ.get("OMP_NUM_THREADS", "8"))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def acquire_singleton():
    """Prevent double-launch."""
    try:
        out = subprocess.check_output(
            ["ps", "-C", "python3", "-o", "pid=,args="], text=True)
        my_pid = os.getpid()
        for line in out.strip().splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) < 2: continue
            pid_str, args_str = parts
            if "marc_warm_restart.py" not in args_str: continue
            try:
                pid = int(pid_str)
            except ValueError:
                continue
            if pid == my_pid: continue
            print(f"[SINGLETON] another instance pid={pid} running, exit", flush=True)
            sys.exit(2)
    except subprocess.CalledProcessError:
        pass
    if LOCK_FILE.exists():
        try:
            old = int(LOCK_FILE.read_text().strip())
            os.kill(old, 0)
            print(f"[SINGLETON] lock held by pid={old}, exit", flush=True)
            sys.exit(2)
        except (OSError, ValueError):
            print("[SINGLETON] stale lock, overwriting", flush=True)
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.write_text(str(os.getpid()))


def release_singleton():
    try:
        if LOCK_FILE.exists(): LOCK_FILE.unlink()
    except Exception: pass


def apply_uniform_init_marc(atoms, magmom_per_fe=0.065):
    """Yang 2019-inspired uniform Fe magmom init for marc V_Fe defect.

    Yang 2019 Physica B 575:411739: V_Fe in m-FeS2 -> 2 holes -> 1.98 muB net.
    31 Fe * 0.065 muB = 2.015 muB ~ Yang 1.98 muB target.

    SAME init as v4c production (--magmom-uniform 0.065) and ZPE run
    (marc_vfe_saddle_freq_dft.py magmom_per_fe=0.065).

    After SCF convergence, system settles to:
      Mtot ~ 1.1 muB, Mabs ~ 1.7 muB  (v4c + ZPE observed values)
    This is the correct magnetic sheet for comparison with 177 meV and ZPE -85 meV.

    NOT AFM stripe (+-2 muB) which forces a different magnetic sheet incomparable
    with v4c/ZPE results.
    """
    syms = atoms.get_chemical_symbols()
    magmoms = np.zeros(len(atoms))
    n_fe = 0
    for i, s in enumerate(syms):
        if s == "Fe":
            magmoms[i] = magmom_per_fe
            n_fe += 1
    atoms.set_initial_magnetic_moments(magmoms)
    net = float(magmoms.sum())
    print(f"  [UNIFORM init Yang-inspired] {n_fe} Fe x {magmom_per_fe} muB = "
          f"net {net:.3f} muB (target Yang 1.98 muB)", flush=True)
    return atoms


def check_magnetic_sheet(atoms, label="", mtot_target=1.1, mabs_target=1.7,
                          mtot_tol=0.3, mabs_tol=0.5, stripe_abort_threshold=10.0):
    """After SCF: verify Mtot~1.1, Mabs~1.7 (v4c/ZPE sheet).

    Reads magmoms from atoms.calc result (set by QE via get_magnetic_moment /
    get_magnetic_moments). Falls back to None if not available.

    Returns dict with keys: mtot, mabs, ok, abort.
    abort=True means Mabs >> 10 = stripe sheet = wrong magnetic leaf.
    """
    diag = {"mtot": None, "mabs": None, "ok": False, "abort": False}
    try:
        calc = atoms.calc
        if calc is None:
            print(f"  [MAG-GUARD {label}] No calc attached, skip check.", flush=True)
            return diag
        # Try total magnetic moment
        try:
            mtot = float(calc.get_magnetic_moment(atoms))
        except Exception:
            mtot = None
        # Try per-atom to get abs sum
        try:
            moms = calc.get_magnetic_moments(atoms)
            mabs = float(np.abs(moms).sum())
        except Exception:
            mabs = None
        diag["mtot"] = mtot
        diag["mabs"] = mabs

        if mtot is None or mabs is None:
            print(f"  [MAG-GUARD {label}] Could not read magmoms from calc. Skip.", flush=True)
            return diag

        print(f"  [MAG-GUARD {label}] Mtot={mtot:.3f} muB, Mabs={mabs:.3f} muB", flush=True)
        print(f"    targets: Mtot~{mtot_target} (tol {mtot_tol}), "
              f"Mabs~{mabs_target} (tol {mabs_tol})", flush=True)

        # Stripe abort: Mabs >> stripe_abort_threshold
        if mabs > stripe_abort_threshold:
            print(f"  [MAG-GUARD {label}] ABORT: Mabs={mabs:.1f} >> {stripe_abort_threshold} "
                  f"= STRIPE sheet. Wrong magnetic leaf! "
                  f"Init (uniform 0.065) did not prevent AFM collapse. "
                  f"Check tot_magnetization constraint and degauss.", flush=True)
            diag["abort"] = True
            return diag

        # Soft checks
        mtot_ok = abs(mtot - mtot_target) <= mtot_tol
        mabs_ok = abs(mabs - mabs_target) <= mabs_tol
        diag["ok"] = mtot_ok and mabs_ok
        if diag["ok"]:
            print(f"  [MAG-GUARD {label}] PASS: correct v4c/ZPE magnetic sheet.", flush=True)
        else:
            print(f"  [MAG-GUARD {label}] WARN: Mtot={mtot:.3f} (ok={mtot_ok}), "
                  f"Mabs={mabs:.3f} (ok={mabs_ok}). "
                  f"Not exactly v4c sheet but not stripe -- continuing.", flush=True)
    except Exception as exc:
        print(f"  [MAG-GUARD {label}] Exception: {exc} -- skip.", flush=True)
    return diag


def load_atoms_clean(src):
    """Load atoms, strip calculator (avoid SinglePointCalculator clash)."""
    import ase.calculators.singlepoint as _sp
    from ase.calculators.calculator import all_properties as _all_props
    _orig = _sp.SinglePointCalculator.__init__
    def _patched(self, obj, **results):
        filtered = {k: v for k, v in results.items() if k in _all_props}
        _orig(self, obj, **filtered)
    _sp.SinglePointCalculator.__init__ = _patched
    try:
        atoms = read(str(src))
    finally:
        _sp.SinglePointCalculator.__init__ = _orig
    atoms.calc = None
    return atoms


def make_calc(label, work_subdir, disk_io="nowf", conv_thr=1e-8, is_relax=True):
    """Make QE calculator with local-TF mixing, mv smearing."""
    work_subdir.mkdir(parents=True, exist_ok=True)
    cmd = f"mpirun --allow-run-as-root --bind-to none -np {MPI_NP} {PW_BIN}"
    profile = EspressoProfile(command=cmd, pseudo_dir=PSEUDO_DIR)

    # disk_io: relax = medium (needs wfc for forces), NEB = medium
    input_data = {
        "control": {
            "calculation":  "scf",
            "restart_mode": "from_scratch",
            "tprnfor":      True,
            "tstress":      False,
            "verbosity":    "high",
            "disk_io":      disk_io,
            "outdir":       str(work_subdir / "tmp"),
            "prefix":       label,
        },
        "system": {
            "ecutwfc":          ECUTWFC,
            "ecutrho":          ECUTRHO,
            "occupations":      "smearing",
            "smearing":         "mv",
            "degauss":          DEGAUSS,
            "nspin":            2,          # defect cell always nspin=2
            "tot_magnetization": 1.1,       # pin to v4c/ZPE sheet (Mtot~1.1 muB)
                                            # prevents SCF wandering to stripe leaf
        },
        "electrons": {
            "conv_thr":         conv_thr,
            "mixing_mode":      "local-TF",
            "mixing_beta":      0.2,
            "mixing_ndim":      12,
            "electron_maxstep": 500,
            "diagonalization":  "david",
            "diago_full_acc":   True,
        },
    }

    return Espresso(
        profile=profile,
        directory=str(work_subdir),
        input_data=input_data,
        pseudopotentials={"Fe": "Fe.upf", "S": "S.upf", "H": "H.upf"},
        kpts=KPTS,
        koffset=(0, 0, 0),
    )


def main():
    t0 = time.time()

    acquire_singleton()

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[marc_warm_restart] OMP={OMP} MPI_NP={MPI_NP}", flush=True)
    print(f"[marc_warm_restart] ECUTWFC={ECUTWFC} KPTS={KPTS} DEGAUSS={DEGAUSS}", flush=True)
    print(f"[marc_warm_restart] mixing=local-TF beta=0.2 ndim=12 smearing=mv", flush=True)

    # Check PP
    for f in ("Fe.upf", "S.upf", "H.upf"):
        p = Path(PSEUDO_DIR) / f
        if not p.exists():
            print(f"[FATAL] Missing PP: {p}", flush=True)
            sys.exit(3)
    print(f"[PP] Fe+S+H pseudos found in {PSEUDO_DIR}", flush=True)

    # ---- Load v4c neb.traj ----
    print(f"\n[1/4] Loading v4c neb.traj from {TRAJ_SRC}", flush=True)
    if not TRAJ_SRC.exists():
        print(f"[FATAL] {TRAJ_SRC} not found. Upload neb.traj first.", flush=True)
        sys.exit(4)

    all_frames = read(str(TRAJ_SRC), index=":")
    n_frames = len(all_frames)
    print(f"  Total frames: {n_frames}", flush=True)

    n_images = N_IMAGES  # 9
    n_complete = n_frames // n_images
    if n_complete == 0:
        print("[FATAL] Not enough frames in neb.traj for 9-image batch", flush=True)
        sys.exit(5)

    last_start = (n_complete - 1) * n_images
    band = all_frames[last_start:last_start + n_images]
    print(f"  Using last batch {n_complete-1}: frames {last_start}..{last_start+n_images-1}", flush=True)
    print(f"  endA: {band[0].get_chemical_formula()} ({len(band[0])} at)", flush=True)
    print(f"  endB: {band[-1].get_chemical_formula()} ({len(band[-1])} at)", flush=True)

    # Clean band atoms (strip calculators)
    band_clean = []
    for fr in band:
        a = fr.copy()
        a.calc = None
        band_clean.append(a)

    endA_raw = band_clean[0]
    endB_raw = band_clean[-1]
    inner    = band_clean[1:8]  # 7 inner images

    # ---- Re-relax endA ----
    print(f"\n[2/4] Re-relax endA (fmax={FMAX_EP}, local-TF, Yang uniform init)", flush=True)
    endA = endA_raw.copy()
    apply_uniform_init_marc(endA, magmom_per_fe=0.065)
    endA_dir = WORK_DIR / "relax_endA"
    endA.calc = make_calc("relax_endA", endA_dir, disk_io="medium")
    opt_A = BFGS(endA, trajectory=str(WORK_DIR / "relax_endA.traj"),
                 logfile=str(WORK_DIR / "relax_endA.log"))
    try:
        opt_A.run(fmax=FMAX_EP, steps=100)
        print(f"  endA relaxed: E={endA.get_potential_energy():.6f} eV", flush=True)
    except Exception as exc:
        print(f"  [WARN] endA relax: {exc} -- continuing with last geometry", flush=True)

    write(str(WORK_DIR / "relaxed_endA_warm.xyz"), endA)
    eA = float(endA.get_potential_energy())

    # ---- Magnetic sheet guard (critical: verify v4c/ZPE sheet) ----
    print(f"\n[MAG-GUARD] Checking endA magnetic sheet after relax...", flush=True)
    mag_diag = check_magnetic_sheet(endA, label="endA", mtot_target=1.1, mabs_target=1.7,
                                    mtot_tol=0.3, mabs_tol=0.5, stripe_abort_threshold=10.0)
    if mag_diag["abort"]:
        print(f"[FATAL] Wrong magnetic leaf (stripe) detected on endA. "
              f"Mabs={mag_diag['mabs']:.1f} >> 10. "
              f"tot_magnetization=1.1 pin did not hold. "
              f"Adjust init or degauss. ABORTING.", flush=True)
        release_singleton()
        sys.exit(10)

    # ---- Re-relax endB ----
    print(f"\n[3/4] Re-relax endB (fmax={FMAX_EP}, local-TF, Yang uniform init)", flush=True)
    endB = endB_raw.copy()
    apply_uniform_init_marc(endB, magmom_per_fe=0.065)
    endB_dir = WORK_DIR / "relax_endB"
    endB.calc = make_calc("relax_endB", endB_dir, disk_io="medium")
    opt_B = BFGS(endB, trajectory=str(WORK_DIR / "relax_endB.traj"),
                 logfile=str(WORK_DIR / "relax_endB.log"))
    try:
        opt_B.run(fmax=FMAX_EP, steps=100)
        print(f"  endB relaxed: E={endB.get_potential_energy():.6f} eV", flush=True)
    except Exception as exc:
        print(f"  [WARN] endB relax: {exc} -- continuing with last geometry", flush=True)

    write(str(WORK_DIR / "relaxed_endB_warm.xyz"), endB)
    eB = float(endB.get_potential_energy())

    print(f"\n  Relaxed: E_endA={eA:.4f} eV, E_endB={eB:.4f} eV, dE={eB-eA:+.4f} eV",
          flush=True)

    # ---- Build NEB images: relaxed_endA + v4c inner 1..7 + relaxed_endB ----
    print(f"\n[4/4] Warm NEB: replace band endpoints, launch CI-FIRE DyNEB", flush=True)

    images = [endA.copy()]
    for im in inner:
        img = im.copy()
        apply_uniform_init_marc(img, magmom_per_fe=0.065)
        images.append(img)
    images.append(endB.copy())
    assert len(images) == N_IMAGES, f"Expected {N_IMAGES} images, got {len(images)}"

    # Assign calculators
    for k, img in enumerate(images[1:-1], start=1):
        img_dir = WORK_DIR / f"image_{k:02d}"
        img.calc = make_calc(f"image_{k:02d}", img_dir, disk_io="medium")

    images[0].calc  = make_calc("image_00", WORK_DIR / "image_00", disk_io="medium")
    images[-1].calc = make_calc("image_08", WORK_DIR / "image_08", disk_io="medium")

    neb = DyNEB(
        images,
        climb=True,
        method="improvedtangent",
        k=K_SPRING,
        fmax=FMAX_NEB,
        dynamic_relaxation=True,
        scale_fmax=0.0,
    )
    print(f"  DyNEB: k_spring={K_SPRING}, climb=True, fmax={FMAX_NEB}", flush=True)

    opt = FIRE(neb,
               trajectory=str(WORK_DIR / "neb_warm.traj"),
               logfile=str(WORK_DIR / "neb_warm.log"))

    print(f"  Starting FIRE optimization...", flush=True)
    try:
        opt.run(fmax=FMAX_NEB, steps=200)
    except Exception as exc:
        print(f"  [WARN] NEB FIRE exception: {exc}", flush=True)

    # Collect energies from inner images
    energies = []
    for k, img in enumerate(images):
        try:
            e = float(img.get_potential_energy())
            energies.append(e)
        except Exception:
            energies.append(None)

    e_barrier = None
    if energies[0] is not None and any(e is not None for e in energies[1:8]):
        valid_inner = [e for e in energies[1:8] if e is not None]
        if valid_inner:
            e_barrier = max(valid_inner) - energies[0]
            print(f"\n  [RESULT] E_a (warm) = {e_barrier:.4f} eV", flush=True)

    result = {
        "status": "completed",
        "E_a_warm_eV": e_barrier,
        "E_endA_eV": energies[0],
        "E_endB_eV": energies[-1],
        "band_energies_eV": energies,
        "n_images": N_IMAGES,
        "k_spring": K_SPRING,
        "kpts": list(KPTS),
        "ecutwfc": ECUTWFC,
        "mixing": "local-TF",
        "mixing_beta": 0.2,
        "smearing": "mv",
        "degauss": DEGAUSS,
        "conv_thr": 1e-8,
        "mag_init": "uniform_yang",
        "magmom_per_fe": 0.065,
        "tot_magnetization_pin": 1.1,
        "t_total_s": time.time() - t0,
    }

    out = OUTPUT_DIR / "marc_warm_result.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    print(f"\n[saved] {out}", flush=True)

    release_singleton()


if __name__ == "__main__":
    main()
