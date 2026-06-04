#!/usr/bin/env python3
"""
Greigite V_Fe U=2 Single-Point sensitivity scan (Option B, s162).

Reads U=0 relaxed geometries (endA, saddle from neb.traj frame 4, endB)
and runs 3 single-point QE calculations at U_eff=2.0 eV (Dudarev).

Purpose: quantify Delta_E_a(U=0 -> U=2) as U-sensitivity bound.
U=0 baseline: 1.86 eV (greigite NEB s150, PBE plain confirmed).
Compare E_a(U=2) = E_saddle(U=2) - E_endA(U=2).

CRITICAL: Uses additional_cards kwarg for HUBBARD injection
(write_input override DEAD on ASE>=3.23 per memory/feedback_qe7x_hubbard_syntax.md).
GUARD: grep pwi for HUBBARD; grep pwo for "Hubbard energy" > 0.

Geometry sources:
  endA:   /workspace/greig_neb_full_s150/relaxed_endA.xyz
  endB:   /workspace/greig_neb_full_s150/relaxed_endB.xyz
  saddle: frame 4 from /workspace/greig_neb_full_s150/neb.traj (final iteration)

Magnetic init: ferri A up/down B (Fe_tet +5 muB, Fe_oct -4 muB)
as in original neb_canonical_greigite_56at_qe_VFe.py.

Output: /workspace/results/greigite_u2_sp.json
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.calculators.espresso import Espresso, EspressoProfile

PSEUDO_DIR = os.environ.get("ESPRESSO_PSEUDO",
                            os.environ.get("PSEUDO_DIR", "/opt/pp/oncv_pbe"))
PW_BIN = os.environ.get("PW_BIN", "pw.x")

WORK_DIR   = Path("/workspace/greig_u2_sp")
OUTPUT_DIR = Path("/workspace/results")
GEOM_DIR   = Path("/workspace/greig_neb_full_s150")

U_EFF   = 2.0   # eV, Dudarev Fe-3d
KPTS    = (2, 2, 2)
ECUTWFC = 80.0
ECUTRHO = 320.0
DEGAUSS = 0.015


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def apply_greigite_ferri_init(atoms):
    """Apply ferrimagnetic init: Fe_tet (+5 muB) up, Fe_oct (-4 muB) down.

    Greigite Fd-3m build order (build_greigite_conventional from afmu module):
      atoms[0:8]   -> Fe_tet (Wyckoff 8a, tetrahedral, Fe3+)
      atoms[8:24]  -> Fe_oct (Wyckoff 16d, octahedral, Fe2+/Fe3+ mixed)
      atoms[24:56] -> S (Wyckoff 32e)

    V_Fe defect: one Fe removed from 8:24 range, H appended as last atom.
    So:
      - len=56: pristine 56-atom cell (no vacancy, no H)
      - len=56: V_Fe+H cell: Fe23 S32 H1 (1 Fe removed from oct, 1 H added)
    """
    syms = atoms.get_chemical_symbols()
    magmoms = np.zeros(len(atoms))

    # Count Fe positions
    fe_positions = [i for i, s in enumerate(syms) if s == "Fe"]
    n_fe = len(fe_positions)

    # For 56-atom pristine: first 8 = tet, next 16 = oct
    # For V_Fe+H cell (56 atoms, 23 Fe): ratio tet:oct ~ 8:15 or 8:16 depending on which removed
    # Use heuristic: first 8 Fe = tet (+5), rest = oct (-4)
    n_tet = min(8, n_fe)
    for k, i in enumerate(fe_positions):
        if k < n_tet:
            magmoms[i] = +5.0   # Fe_tet HS Fe3+
        else:
            magmoms[i] = -4.0   # Fe_oct Fe2+/Fe3+ mixed
    # H: 0, S: 0
    atoms.set_initial_magnetic_moments(magmoms)
    n_pos = int((magmoms > 0).sum())
    n_neg = int((magmoms < 0).sum())
    net = float(magmoms.sum())
    print(f"  [FERRI init] {n_pos} Fe_tet +5muB, {n_neg} Fe_oct -4muB, net={net:.1f} muB",
          flush=True)
    return atoms


def make_sp_calc(label, mpi_np=1):
    """Create Espresso SP calculator with HUBBARD card via additional_cards kwarg.

    CRITICAL: additional_cards is the ONLY working mechanism on ASE>=3.23.
    write_input override is DEAD (GenericFileIOCalculator does not call it).
    Per memory/feedback_qe7x_hubbard_syntax.md.

    Fe1 = Fe_tet sublattice (8a), Fe2 = Fe_oct sublattice (16d).
    Both get U=2.0 eV (Dudarev ortho-atomic).
    """
    work_subdir = WORK_DIR / label
    work_subdir.mkdir(parents=True, exist_ok=True)

    cmd = f"mpirun --allow-run-as-root --bind-to none -np {mpi_np} {PW_BIN}"
    profile = EspressoProfile(command=cmd, pseudo_dir=PSEUDO_DIR)

    input_data = {
        "control": {
            "calculation":  "scf",
            "restart_mode": "from_scratch",
            "tprnfor":      True,
            "tstress":      False,
            "verbosity":    "high",
            "disk_io":      "nowf",
            "outdir":       str(work_subdir / "tmp"),
            "prefix":       label,
        },
        "system": {
            "ecutwfc":     ECUTWFC,
            "ecutrho":     ECUTRHO,
            "occupations": "smearing",
            "smearing":    "mv",
            "degauss":     DEGAUSS,
            "nspin":       2,
        },
        "electrons": {
            "conv_thr":         1.0e-8,
            "mixing_mode":      "local-TF",
            "mixing_beta":      0.2,
            "mixing_ndim":      12,
            "electron_maxstep": 500,
            "diagonalization":  "david",
            "diago_thr_init":   1.0e-4,
            "diago_full_acc":   True,
            "startingwfc":      "atomic+random",
        },
    }

    # HUBBARD card via additional_cards (ASE>=3.23 compatible)
    # CRITICAL: ASE auto-splits Fe species when initial_magmoms differ:
    #   Fe (tet, +5 muB) -> species "Fe"
    #   Fe (oct, -4 muB) -> species "Fe1"
    # Both must receive U=2 (Devey 2009, Roldan 2016 both apply to all Fe).
    # GUARD: if only "Fe-3d" covered, Fe1 atoms run at PBE U=0.
    # Both Fe and Fe1 use same pseudopotential (Fe.upf).
    hubbard_card = (
        "HUBBARD (ortho-atomic)\n"
        f"U Fe-3d {U_EFF:.4f}\n"
        f"U Fe1-3d {U_EFF:.4f}\n"
    )

    calc = Espresso(
        profile=profile,
        directory=str(work_subdir),
        input_data=input_data,
        pseudopotentials={"Fe": "Fe.upf", "S": "S.upf", "H": "H.upf"},
        kpts=KPTS,
        koffset=(0, 0, 0),
        additional_cards=hubbard_card,
    )
    return calc


def verify_hubbard_in_pwi(pwi_path, label):
    """Guard: verify HUBBARD card actually written to pwi. Abort if not."""
    try:
        text = Path(pwi_path).read_text()
        if "HUBBARD" not in text:
            raise RuntimeError(
                f"GUARD FAIL [{label}]: HUBBARD card NOT found in {pwi_path}. "
                f"U={U_EFF} silently dropped. ABORT."
            )
        print(f"  [GUARD] HUBBARD card confirmed in {pwi_path}", flush=True)
    except FileNotFoundError:
        print(f"  [GUARD WARN] pwi not found yet: {pwi_path}", flush=True)


def parse_hubbard_energy_count(pwo_path):
    """Count 'Hubbard energy' occurrences in pwo (>0 means U applied in SCF)."""
    try:
        text = Path(pwo_path).read_text(errors="ignore")
        count = text.count("Hubbard energy")
        return count
    except Exception:
        return -1


def parse_final_energy(pwo_path):
    """Parse final total energy from QE pwo."""
    try:
        text = Path(pwo_path).read_text(errors="ignore")
        lines = text.splitlines()
        e = None
        for line in lines:
            if "!    total energy" in line:
                # Format: "!    total energy              =     -X.XXXXX Ry"
                parts = line.split("=")
                if len(parts) >= 2:
                    val_str = parts[-1].strip().split()[0]
                    e = float(val_str) * 13.6057  # Ry -> eV
        return e
    except Exception as exc:
        print(f"  [WARN parse_final_energy] {pwo_path}: {exc}", flush=True)
        return None


def parse_magnetization(pwo_path):
    """Parse last total/absolute magnetization from pwo."""
    try:
        text = Path(pwo_path).read_text(errors="ignore")
        total_vals, abs_vals = [], []
        for line in text.splitlines():
            if "total magnetization" in line:
                try:
                    total_vals.append(float(line.split("=")[-1].split("Bohr")[0].strip()))
                except Exception:
                    pass
            elif "absolute magnetization" in line:
                try:
                    abs_vals.append(float(line.split("=")[-1].split("Bohr")[0].strip()))
                except Exception:
                    pass
        if total_vals and abs_vals:
            return {"total_mag_uB": total_vals[-1], "abs_mag_uB": abs_vals[-1]}
        return None
    except Exception:
        return None


def load_atoms_safe(xyz_path):
    """Load atoms from xyz, clear calculator (avoid stale SinglePointCalculator clash)."""
    import ase.calculators.singlepoint as _sp
    from ase.calculators.calculator import all_properties as _all_props
    _orig = _sp.SinglePointCalculator.__init__
    def _patched(self, atoms_obj, **results):
        filtered = {k: v for k, v in results.items() if k in _all_props}
        _orig(self, atoms_obj, **filtered)
    _sp.SinglePointCalculator.__init__ = _patched
    try:
        atoms = read(str(xyz_path))
    finally:
        _sp.SinglePointCalculator.__init__ = _orig
    atoms.calc = None
    return atoms


def run_sp(atoms, label, mpi_np=1):
    """Run single-point SCF; return (E_eV, mag_dict, pwi_path, pwo_path)."""
    print(f"\n[SP {label}] Starting SCF ...", flush=True)
    t0 = time.time()

    calc = make_sp_calc(label, mpi_np=mpi_np)
    atoms.calc = calc

    # Trigger write of pwi (force ASE to write before SCF)
    # ASE writes pwi when get_potential_energy() is called; guard runs after.
    try:
        e_eV = float(atoms.get_potential_energy())
    except RuntimeError as exc:
        raise RuntimeError(f"SP [{label}] SCF failed: {exc}") from exc

    elapsed = time.time() - t0
    work_subdir = WORK_DIR / label
    pwi_path = work_subdir / "espresso.pwi"
    pwo_path = work_subdir / "espresso.pwo"

    # GUARD: verify HUBBARD in pwi
    verify_hubbard_in_pwi(pwi_path, label)

    # GUARD: count Hubbard energy lines in pwo (>0 = U applied)
    hub_count = parse_hubbard_energy_count(pwo_path)
    if hub_count == 0:
        raise RuntimeError(
            f"GUARD FAIL [{label}]: 'Hubbard energy' NOT found in {pwo_path}. "
            f"U={U_EFF} was NOT applied in SCF. Check species labels / HUBBARD card."
        )
    print(f"  [GUARD] Hubbard energy lines in pwo: {hub_count} (>0 = U applied)", flush=True)

    mag = parse_magnetization(pwo_path)
    print(f"  [SP {label}] E={e_eV:.6f} eV, elapsed={elapsed:.0f}s", flush=True)
    if mag:
        print(f"  [SP {label}] total_mag={mag['total_mag_uB']:.3f} muB, "
              f"abs_mag={mag['abs_mag_uB']:.3f} muB", flush=True)

    return e_eV, mag, str(pwi_path), str(pwo_path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpi-np", type=int, default=1)
    parser.add_argument("--saddle-frame", type=int, default=4,
                        help="Frame index in neb.traj (0-based) for saddle. "
                             "Default 4 = image_04 (last NEB iteration, 9 images: "
                             "0=endA, 1..7=intermediate, 8=endB -> image_04 is frame 4).")
    parser.add_argument("--omp", type=int, default=8)
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"]      = str(args.omp)
    os.environ["MKL_NUM_THREADS"]      = str(args.omp)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.omp)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[greigite_u2_sp] U_eff={U_EFF} eV, kpts={KPTS}, "
          f"ecutwfc={ECUTWFC}, ecutrho={ECUTRHO}", flush=True)
    print(f"[greigite_u2_sp] PSEUDO_DIR={PSEUDO_DIR}", flush=True)

    # Verify pseudo files present
    for sym, fn in (("Fe", "Fe.upf"), ("S", "S.upf"), ("H", "H.upf")):
        p = Path(PSEUDO_DIR) / fn
        if not p.exists():
            print(f"[FATAL] Missing pseudo: {p}", flush=True)
            sys.exit(3)
    print(f"  [PP] Fe.upf, S.upf, H.upf present in {PSEUDO_DIR}", flush=True)

    # ---- Load geometries ----
    print(f"\n[1/5] Loading geometries from {GEOM_DIR}", flush=True)

    endA_xyz  = GEOM_DIR / "relaxed_endA.xyz"
    endB_xyz  = GEOM_DIR / "relaxed_endB.xyz"
    neb_traj  = GEOM_DIR / "neb.traj"

    for p in (endA_xyz, endB_xyz, neb_traj):
        if not p.exists():
            print(f"[FATAL] Missing geometry file: {p}", flush=True)
            sys.exit(4)

    atoms_endA = load_atoms_safe(endA_xyz)
    atoms_endB = load_atoms_safe(endB_xyz)

    # Load saddle from neb.traj
    # neb.traj stores ALL frames from ALL NEB iterations.
    # Each iteration: 9 frames (endA + 7 intermediate + endB).
    # Last iteration's frame 4 (0-indexed) = image_04 = saddle candidate.
    # We read last full batch.
    print(f"  Loading neb.traj ...", flush=True)
    try:
        all_frames = read(str(neb_traj), index=":")
        n_frames = len(all_frames)
        print(f"  neb.traj: {n_frames} total frames", flush=True)

        n_images = 9  # endA + 7 intermediate + endB
        n_complete = n_frames // n_images
        last_batch_start = (n_complete - 1) * n_images if n_complete > 0 else 0
        saddle_abs_idx = last_batch_start + args.saddle_frame
        if saddle_abs_idx >= n_frames:
            saddle_abs_idx = n_frames - 1
            print(f"  WARN: saddle_abs_idx clamped to {saddle_abs_idx}", flush=True)
        atoms_saddle = all_frames[saddle_abs_idx]
        atoms_saddle = atoms_saddle.copy()
        atoms_saddle.calc = None
        print(f"  Saddle: batch {n_complete-1}, frame {args.saddle_frame}, "
              f"abs_idx={saddle_abs_idx}, "
              f"formula={atoms_saddle.get_chemical_formula()}", flush=True)
    except Exception as exc:
        raise RuntimeError(f"neb.traj load failed: {exc}") from exc

    print(f"  endA:   {atoms_endA.get_chemical_formula()} "
          f"({len(atoms_endA)} atoms)", flush=True)
    print(f"  saddle: {atoms_saddle.get_chemical_formula()} "
          f"({len(atoms_saddle)} atoms)", flush=True)
    print(f"  endB:   {atoms_endB.get_chemical_formula()} "
          f"({len(atoms_endB)} atoms)", flush=True)

    # Save saddle xyz for reference
    write(str(WORK_DIR / "saddle_frame4.xyz"), atoms_saddle)

    # ---- Apply ferri init ----
    print(f"\n[2/5] Applying ferrimagnetic init (Fe_tet +5, Fe_oct -4 muB)", flush=True)
    apply_greigite_ferri_init(atoms_endA)
    apply_greigite_ferri_init(atoms_saddle)
    apply_greigite_ferri_init(atoms_endB)

    result = {
        "U_eff_eV":    U_EFF,
        "kpts":        list(KPTS),
        "ecutwfc":     ECUTWFC,
        "ecutrho":     ECUTRHO,
        "degauss":     DEGAUSS,
        "saddle_frame_in_batch": args.saddle_frame,
        "saddle_abs_idx": saddle_abs_idx,
        "n_traj_frames": n_frames,
        "geom_source": str(GEOM_DIR),
    }

    t_total = time.time()

    # ---- SP endA ----
    print(f"\n[3/5] SP endA at U={U_EFF} eV", flush=True)
    try:
        e_endA, mag_endA, pwi_endA, pwo_endA = run_sp(
            atoms_endA.copy(), "sp_endA_u2", mpi_np=args.mpi_np)
        result["E_endA_eV"]  = e_endA
        result["mag_endA"]   = mag_endA
        result["pwi_endA"]   = pwi_endA
        result["pwo_endA"]   = pwo_endA
    except Exception as exc:
        result["endA_error"] = str(exc)
        traceback.print_exc()
        print(f"[FAIL] endA: {exc}", flush=True)
        e_endA = None

    # ---- SP saddle ----
    print(f"\n[4/5] SP saddle at U={U_EFF} eV", flush=True)
    try:
        e_saddle, mag_saddle, pwi_sad, pwo_sad = run_sp(
            atoms_saddle.copy(), "sp_saddle_u2", mpi_np=args.mpi_np)
        result["E_saddle_eV"] = e_saddle
        result["mag_saddle"]  = mag_saddle
        result["pwi_saddle"]  = pwi_sad
        result["pwo_saddle"]  = pwo_sad
    except Exception as exc:
        result["saddle_error"] = str(exc)
        traceback.print_exc()
        print(f"[FAIL] saddle: {exc}", flush=True)
        e_saddle = None

    # ---- SP endB ----
    print(f"\n[5/5] SP endB at U={U_EFF} eV", flush=True)
    try:
        e_endB, mag_endB, pwi_endB, pwo_endB = run_sp(
            atoms_endB.copy(), "sp_endB_u2", mpi_np=args.mpi_np)
        result["E_endB_eV"]  = e_endB
        result["mag_endB"]   = mag_endB
        result["pwi_endB"]   = pwi_endB
        result["pwo_endB"]   = pwo_endB
    except Exception as exc:
        result["endB_error"] = str(exc)
        traceback.print_exc()
        print(f"[FAIL] endB: {exc}", flush=True)
        e_endB = None

    # ---- Compute E_a(U=2) ----
    if e_endA is not None and e_saddle is not None:
        e_a_u2 = e_saddle - e_endA
        result["E_a_U2_eV"] = e_a_u2
        print(f"\n[RESULT] E_a(U={U_EFF}) = {e_a_u2:.4f} eV "
              f"(saddle={e_saddle:.4f}, endA={e_endA:.4f})", flush=True)
        # Baseline U=0 from s150
        e_a_u0 = 1.86  # eV, greigite NEB s150 PBE U=0 confirmed
        delta = e_a_u2 - e_a_u0
        result["E_a_U0_baseline_eV"] = e_a_u0
        result["Delta_E_a_U0_to_U2_eV"] = delta
        print(f"[RESULT] Delta_E_a(U=0->U=2) = {delta:+.4f} eV "
              f"(baseline U=0: {e_a_u0} eV)", flush=True)
        if e_endB is not None:
            e_rxn = e_endB - e_endA
            result["E_rxn_U2_eV"] = e_rxn
            print(f"[RESULT] E_rxn(U=2) = {e_rxn:+.4f} eV", flush=True)
    else:
        print(f"\n[RESULT] Cannot compute E_a(U=2): endA or saddle SP failed", flush=True)
        result["E_a_U2_eV"] = None

    result["t_total_s"] = time.time() - t_total
    result["status"] = "success" if e_endA is not None and e_saddle is not None else "partial_fail"

    out = OUTPUT_DIR / "greigite_u2_sp.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    print(f"\n[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
