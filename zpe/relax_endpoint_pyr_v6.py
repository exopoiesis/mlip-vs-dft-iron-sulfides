#!/usr/bin/env python3
"""Re-relax pyrite V_Fe+H endpoint (endA) at conv_thr=1e-10 for a clean minimum
before the ZPE freq (v5 endA was relaxed at conv_thr 1e-8 -> residual forces would
contaminate the Hessian). Same calc as v6/v5 (apples-to-apples). nspin=1.

Output: relaxed_endA_v6.xyz (fmax target 0.01) for the paired endpoint-freq -> dZPE#.
"""
import argparse, os
from pathlib import Path
import numpy as np
from ase.io import read, write
from ase.optimize import BFGS
from ase.calculators.espresso import Espresso, EspressoProfile

# --- tolerate extxyz files carrying QE calc results ASE rejects (e.g. 'nspins',
# 'nkpts', 'nbands' from a prior QE run -> SinglePointCalculator AssertionError).
# v5 endA was written with these; strip unknown props on read. (s158 Phase-0 fix.)
from ase.calculators.singlepoint import SinglePointCalculator as _SPC
import ase.calculators.singlepoint as _spmod
_spc_orig_init = _SPC.__init__
def _spc_tolerant_init(self, atoms, **results):
    results = {k: v for k, v in results.items() if k in _spmod.all_properties}
    _spc_orig_init(self, atoms, **results)
_SPC.__init__ = _spc_tolerant_init

PW_BIN = os.environ.get("PW_BIN", "pw.x")
PSEUDO_DIR = os.environ.get("ESPRESSO_PSEUDO", os.environ.get("PSEUDO_DIR", "/opt/pp/oncv_pbe"))


def make_calc(work_dir, label, kpts=(2, 2, 2), conv_thr=1.0e-10, mpi_np=1, nspin=1):
    pseudo_files = {"Fe": "Fe.upf", "S": "S.upf", "H": "H.upf"}
    cmd = f"mpirun --allow-run-as-root --bind-to none -np {mpi_np} {PW_BIN}"
    profile = EspressoProfile(command=cmd, pseudo_dir=PSEUDO_DIR)
    input_data = {
        "control": {"calculation": "scf", "restart_mode": "from_scratch", "tprnfor": True,
                    "tstress": False, "verbosity": "high", "disk_io": "low",
                    "outdir": str(Path(work_dir) / label / "tmp"), "prefix": label},
        "system": {"ecutwfc": 60.0, "ecutrho": 480.0, "occupations": "smearing",
                   "nspin": nspin, "smearing": "gaussian", "degauss": 0.01},
        "electrons": {"conv_thr": conv_thr, "mixing_mode": "plain", "mixing_beta": 0.3,
                      "electron_maxstep": 200, "diagonalization": "david"},
    }
    print(f"[make_calc] {label} {input_data['system']} conv_thr={conv_thr}", flush=True)
    return Espresso(profile=profile, directory=str(Path(work_dir) / label),
                    input_data=input_data, pseudopotentials=pseudo_files,
                    kpts=tuple(kpts), koffset=(0, 0, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-xyz", required=True)
    ap.add_argument("--out-xyz", required=True)
    ap.add_argument("--work-dir", default="/workspace/pyr_endA_relax_v6")
    ap.add_argument("--fmax", type=float, default=0.01)
    ap.add_argument("--conv-thr", type=float, default=1.0e-10)
    ap.add_argument("--kpts", default="2,2,2")
    ap.add_argument("--mpi-np", type=int, default=1)
    ap.add_argument("--nspin", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=60)
    args = ap.parse_args()

    wd = Path(args.work_dir); wd.mkdir(parents=True, exist_ok=True)
    kpts = tuple(int(x) for x in args.kpts.split(","))
    atoms = read(args.in_xyz)
    atoms.set_constraint()  # no constraints
    n_e = sum({"Fe": 16, "S": 6, "H": 1}[s] for s in atoms.get_chemical_symbols())
    print(f"[load] {args.in_xyz}: {len(atoms)} atoms, N_e={n_e} "
          f"({'ODD' if n_e % 2 else 'EVEN'}), nspin={args.nspin}", flush=True)
    print(f"[geom] d(H-S18)={atoms.get_distance(95,18,mic=True):.3f} "
          f"d(H-S71)={atoms.get_distance(95,71,mic=True):.3f}", flush=True)
    atoms.calc = make_calc(wd, "endA", kpts=kpts, conv_thr=args.conv_thr,
                           mpi_np=args.mpi_np, nspin=args.nspin)
    opt = BFGS(atoms, trajectory=str(wd / "endA_relax.traj"), logfile=str(wd / "endA_relax.log"))
    opt.run(fmax=args.fmax, steps=args.max_steps)
    fmax_final = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    write(args.out_xyz, atoms.copy())  # clean copy (no calc) -> freq read won't choke on 'nspins'
    print(f"[DONE] fmax_final={fmax_final:.4f} E={atoms.get_potential_energy():.5f} eV "
          f"-> {args.out_xyz}", flush=True)
    print(f"[geom] final d(H-S18)={atoms.get_distance(95,18,mic=True):.3f} "
          f"d(H-S71)={atoms.get_distance(95,71,mic=True):.3f}", flush=True)
    (wd / "DONE_endA_relax").write_text(f"fmax={fmax_final:.4f}\n")


if __name__ == "__main__":
    main()
