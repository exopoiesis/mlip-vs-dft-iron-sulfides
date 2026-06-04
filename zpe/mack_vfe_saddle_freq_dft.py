#!/usr/bin/env python3
"""DFT partial-Hessian frequency at the mackinawite V_Fe+H NEB saddle.

PURPOSE (s161 ZPE campaign; apples-to-apples with the mack V_Fe NEB):
  (a) confirm index-1 TS at DFT level (exactly 1 imaginary mode),
  (b) extract the reactive negative-mode eigenvector,
  (c) ZPE of the reactive subsystem (for DeltaZPE-double-dagger = ZPE_saddle - ZPE_endA).

APPLES-TO-APPLES with the mack V_Fe NEB (paper-quotable 42.88 meV): IDENTICAL QE
calculator -- nspin=1, ecutwfc=60 Ry, ecutrho=240 Ry (NOT 480: mackinawite ran at
240), gaussian smearing degauss=0.01 Ry, kpts 2x2x2, NO Hubbard U, david, plain
mix 0.3. One deliberate change vs the NEB SCF: tighter conv_thr (Hessian = 2nd
derivative of energy, force noise amplified).

Pseudopotentials: pass --pp-fe/--pp-s/--pp-h and --pseudo-dir to match the mack
NEB pseudo set EXACTLY (verify the UPF on the instance: ONCV vs USPP; the freq
run must use the SAME pseudo for apples-to-apples + number_of_wfc).

REGION: H + the reactive S anchors + 1st shell. Either an explicit --indices list,
or auto-selected via --auto-region-cutoff (all non-H atoms within cutoff Angstrom of
the migrating proton --h-index). The auto-mode asserts H + >=2 S + >=1 Fe and a size
in [8, 15] (memory: feedback_partial_hessian_region_too_small -- a 3-atom region
gives a spurious soft mode).

Output: <output_dir>/mack_vfe_saddle_freq_dft.json (freqs cm-1 + n_imaginary +
imaginary eigenvector as [idx,dx,dy,dz] + ZPE + real_freqs) + vib cache pckls
(full Hessian, recoverable) + mack_vfe_imag_mode.extxyz (mode vector in an array).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.vibrations import Vibrations

# tolerate extxyz carrying QE calc results ASE rejects (e.g. 'nspins' from a prior
# QE relax write -> SinglePointCalculator AssertionError). strip unknown props on read.
from ase.calculators.singlepoint import SinglePointCalculator as _SPC
import ase.calculators.singlepoint as _spmod
_spc_orig_init = _SPC.__init__
def _spc_tolerant_init(self, atoms, **results):
    results = {k: v for k, v in results.items() if k in _spmod.all_properties}
    _spc_orig_init(self, atoms, **results)
_SPC.__init__ = _spc_tolerant_init

PW_BIN = os.environ.get("PW_BIN", "pw.x")
PSEUDO_DIR = os.environ.get("ESPRESSO_PSEUDO",
                            os.environ.get("PSEUDO_DIR", "/opt/pp/oncv_pbe"))


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, complex):
            return {"re": o.real, "im": o.imag}
        return super().default(o)


def make_calc(work_dir, label, kpts=(2, 2, 2), ecutwfc=60.0, ecutrho=240.0,
              mpi_np=1, conv_thr=1.0e-10, degauss=0.01, nspin=1,
              pseudo_files=None, pseudo_dir=None):
    """IDENTICAL to mack NEB make_calc (smearing branch) except tighter conv_thr.

    ecutrho default 240 (mackinawite NEB ran at 240, NOT pyrite's 480).
    pseudo_files / pseudo_dir configurable to match the mack NEB pseudo set.
    """
    if pseudo_files is None:
        pseudo_files = {"Fe": "Fe.upf", "S": "S.upf", "H": "H.upf"}
    if pseudo_dir is None:
        pseudo_dir = PSEUDO_DIR
    cmd = f"mpirun --allow-run-as-root --bind-to none -np {mpi_np} {PW_BIN}"
    profile = EspressoProfile(command=cmd, pseudo_dir=pseudo_dir)
    input_data = {
        "control": {
            "calculation": "scf",
            "restart_mode": "from_scratch",
            "tprnfor": True,
            "tstress": False,
            "verbosity": "high",
            "disk_io": "low",  # freq displacements are independent SCFs
            "outdir": str(Path(work_dir) / label / "tmp"),
            "prefix": label,
        },
        "system": {
            "ecutwfc": ecutwfc,
            "ecutrho": ecutrho,
            "occupations": "smearing",
            "nspin": nspin,
            "smearing": "gaussian",
            "degauss": degauss,
            # NO Hubbard U (matches mack NEB, nspin=1 no U)
        },
        "electrons": {
            "conv_thr": conv_thr,
            "mixing_mode": "plain",
            "mixing_beta": 0.3,
            "electron_maxstep": 200,
            "diagonalization": "david",
        },
    }
    print(f"[make_calc] label={label} system={input_data['system']} conv_thr={conv_thr} "
          f"pseudo_dir={pseudo_dir} pseudo={pseudo_files}", flush=True)
    return Espresso(profile=profile, directory=str(Path(work_dir) / label),
                    input_data=input_data, pseudopotentials=pseudo_files,
                    kpts=tuple(kpts), koffset=(0, 0, 0))


def n_valence_electrons(atoms):
    # ONCV-SR PBE valence: Fe 16, S 6, H 1 (matches mack NEB parity logic).
    val = {"Fe": 16, "S": 6, "H": 1}
    return sum(val[s] for s in atoms.get_chemical_symbols())


def select_region_by_cutoff(atoms, h_index, cutoff):
    """Region = [h_index] + all NON-H atoms within cutoff Angstrom of H (mic=True).

    Prints every member with symbol + distance. Asserts H + >=2 S + >=1 Fe and a
    region size in [8, 15] (feedback_partial_hessian_region_too_small). Exits 1 on
    failure.
    """
    syms = atoms.get_chemical_symbols()
    if syms[h_index] != "H":
        print(f"[region-auto] FATAL: --h-index {h_index} is {syms[h_index]}, not H",
              flush=True)
        sys.exit(1)
    region = [h_index]
    for i in range(len(atoms)):
        if i == h_index:
            continue
        if syms[i] == "H":
            continue
        d = float(atoms.get_distance(h_index, i, mic=True))
        if d <= cutoff:
            region.append(i)
    region = sorted(region)
    n_s = sum(1 for i in region if syms[i] == "S")
    n_fe = sum(1 for i in region if syms[i] == "Fe")
    print(f"[region-auto] cutoff={cutoff} Ang from H idx {h_index}: "
          f"{len(region)} atoms (S={n_s}, Fe={n_fe})", flush=True)
    for i in region:
        if i == h_index:
            print(f"    atom {i} = {syms[i]} (proton, d=0.000)", flush=True)
        else:
            d = float(atoms.get_distance(h_index, i, mic=True))
            print(f"    atom {i} = {syms[i]}  d_H={d:.3f} Ang", flush=True)
    if n_s < 2 or n_fe < 1:
        print(f"[region-auto] FATAL: region needs H + >=2 S + >=1 Fe, "
              f"got S={n_s} Fe={n_fe}", flush=True)
        sys.exit(1)
    if not (8 <= len(region) <= 15):
        print(f"[region-auto] FATAL: region size {len(region)} not in [8, 15]; "
              f"adjust --auto-region-cutoff", flush=True)
        sys.exit(1)
    return region


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saddle-xyz", required=True,
                    help="mack V_Fe saddle geometry (NEB image_04 extxyz)")
    ap.add_argument("--work-dir", default="/workspace/mack_vfe_freq_dft")
    ap.add_argument("--output-dir", default="/workspace/results")
    ap.add_argument("--indices", default="35,36,37,42,43,59,61,67,71",
                    help="comma atom indices for partial Hessian (used if "
                         "--auto-region-cutoff=0)")
    ap.add_argument("--auto-region-cutoff", type=float, default=0.0,
                    help="if >0 and --h-index set: region = H + all non-H atoms "
                         "within this many Angstrom of H (mic). 0 = use --indices.")
    ap.add_argument("--h-index", type=int, default=None,
                    help="index of the migrating proton (required for auto-region)")
    ap.add_argument("--delta", type=float, default=0.01, help="displacement (Ang)")
    ap.add_argument("--conv-thr", type=float, default=1.0e-11,
                    help="QE conv_thr; 1e-11 for soft-mode Hessian SNR")
    ap.add_argument("--nfree", type=int, default=2, choices=(2, 4),
                    help="finite-diff points: 2=central(cheap), 4=5pt(flat-top anharm)")
    ap.add_argument("--imag-thresh-cm1", type=float, default=20.0,
                    help="|imag| below this = numerical-zero (translational), NOT a real "
                         "imaginary mode. partial Hessian keeps translations")
    ap.add_argument("--ecutrho", type=float, default=240.0,
                    help="charge-density cutoff (Ry); 240 matches mack NEB")
    ap.add_argument("--kpts", default="2,2,2")
    ap.add_argument("--mpi-np", type=int, default=1)
    ap.add_argument("--nspin", type=int, default=1)
    ap.add_argument("--pseudo-dir", default=PSEUDO_DIR,
                    help="QE pseudo dir (default from ESPRESSO_PSEUDO/PSEUDO_DIR)")
    ap.add_argument("--pp-fe", default="Fe.upf")
    ap.add_argument("--pp-s", default="S.upf")
    ap.add_argument("--pp-h", default="H.upf")
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    out_dir = Path(args.output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    kpts = tuple(int(x) for x in args.kpts.split(","))
    pseudo_files = {"Fe": args.pp_fe, "S": args.pp_s, "H": args.pp_h}

    atoms = read(args.saddle_xyz)
    print(f"[load] {args.saddle_xyz}: {len(atoms)} atoms, pbc={atoms.get_pbc()}",
          flush=True)

    # --- parity sanity (M0 guard, smearing -> WARN only) ---
    n_e = n_valence_electrons(atoms)
    par = "ODD" if n_e % 2 else "EVEN"
    print(f"[parity] N_e={n_e} ({par}), nspin={args.nspin}, occ=smearing", flush=True)
    if n_e % 2 == 1 and args.nspin == 1:
        print("[parity] odd-e + nspin=1 + smearing -> half-e at E_F is OK "
              "(mack V_Fe confirmed non-magnetic). Proceeding.", flush=True)

    # --- region selection ---
    if args.auto_region_cutoff > 0:
        if args.h_index is None:
            print("[region-auto] FATAL: --auto-region-cutoff>0 requires --h-index",
                  flush=True)
            sys.exit(1)
        indices = select_region_by_cutoff(atoms, args.h_index, args.auto_region_cutoff)
    else:
        indices = [int(x) for x in args.indices.split(",")]
        for i in indices:
            print(f"[region] atom {i} = {atoms[i].symbol}", flush=True)
    print(f"[region] final partial-Hessian region ({len(indices)} atoms): {indices}",
          flush=True)

    atoms.calc = make_calc(work_dir, "freq", kpts=kpts, ecutrho=args.ecutrho,
                           conv_thr=args.conv_thr, mpi_np=args.mpi_np,
                           nspin=args.nspin, pseudo_files=pseudo_files,
                           pseudo_dir=args.pseudo_dir)

    vibname = str(work_dir / "vib")
    vib = Vibrations(atoms, indices=indices, name=vibname, delta=args.delta,
                     nfree=args.nfree)
    print(f"[vib] running partial Hessian: {len(indices)} atoms x 3 x {args.nfree} = "
          f"{len(indices)*3*args.nfree} SCF (delta={args.delta} Ang, nfree={args.nfree})",
          flush=True)
    vib.run()
    vib.summary(log=sys.stdout)

    freqs = vib.get_frequencies()  # complex ndarray, cm^-1
    # ASE convention: imaginary frequencies returned as complex (0 + b*1j).
    # NOTE: a PARTIAL Hessian does NOT project out the region's translational modes
    # -> 2-3 near-zero modes survive; numerical noise can give them a tiny imaginary
    # part and trip abs(imag)>abs(real). So a mode counts as IMAGINARY only if
    # abs(imag) > imag_thresh_cm1 (default 20). Sub-threshold near-zero modes are
    # reported separately (translational/spurious), NOT counted.
    thr = args.imag_thresh_cm1
    n_imag = 0
    imag_list, real_list, nearzero_list = [], [], []
    imag_indices = []
    for k, f in enumerate(freqs):
        f = complex(f)
        is_imag = abs(f.imag) > abs(f.real)
        mag = abs(f.imag) if is_imag else abs(f.real)
        if mag < thr:
            nearzero_list.append(round((-mag if is_imag else mag), 2))  # signed-ish tag
            continue
        if is_imag:
            n_imag += 1
            imag_list.append(round(abs(f.imag), 2))
            imag_indices.append(k)
        else:
            real_list.append(round(abs(f.real), 2))
    print(f"[modes] n_imag(>{thr}cm-1)={n_imag} imag={imag_list} | "
          f"near-zero(<{thr})={nearzero_list} (translational/spurious, excluded)",
          flush=True)

    # imaginary mode eigenvector(s): get_mode returns (natoms,3) full array.
    imag_modes = {}
    for k in imag_indices:
        mode = np.asarray(vib.get_mode(k))  # (natoms, 3)
        nz = [[int(i), float(mode[i, 0]), float(mode[i, 1]), float(mode[i, 2])]
              for i in indices]  # only region atoms are nonzero
        imag_modes[str(k)] = {"freq_cm1_imag": round(abs(complex(freqs[k]).imag), 2),
                              "vector_region": nz}

    # ZPE from REAL positive modes only (ASE helper handles this).
    try:
        zpe_eV = float(vib.get_zero_point_energy())
    except Exception:
        # manual: 0.5 * sum(h*nu) over real positive freqs
        from ase.units import invcm
        zpe_eV = 0.5 * sum(fr * invcm for fr in real_list if fr > 0)

    verdict = ("INDEX_1_TS" if n_imag == 1
               else ("MINIMUM_NO_IMAG" if n_imag == 0
                     else f"HIGHER_ORDER_SADDLE_{n_imag}"))

    result = {
        "tool": "mack_vfe_saddle_freq_dft",
        "level": "DFT (QE pw.x, PBE)",
        "calc": {"nspin": args.nspin, "ecutwfc": 60.0, "ecutrho": args.ecutrho,
                 "smearing": "gaussian", "degauss": 0.01, "kpts": list(kpts),
                 "conv_thr": args.conv_thr, "U": 0, "delta_Ang": args.delta,
                 "nfree": args.nfree, "imag_thresh_cm1": thr,
                 "pseudo_dir": args.pseudo_dir, "pseudo_files": pseudo_files},
        "saddle_xyz": args.saddle_xyz,
        "hessian_region": indices,
        "n_atoms": len(atoms),
        "n_valence_e": n_e,
        "n_imaginary": n_imag,
        "imaginary_freqs_cm1": imag_list,
        "real_freqs_cm1": real_list,
        "nearzero_freqs_cm1": nearzero_list,
        "imaginary_modes": imag_modes,
        "zpe_real_modes_eV": round(zpe_eV, 5),
        "verdict": verdict,
        "note": ("DFT-grade partial-Hessian over reactive region (H + S anchors + "
                 "1st shell) for mackinawite V_Fe, apples-to-apples with the mack "
                 "V_Fe NEB (nspin=1, no U, ecutrho 240). ZPE is reactive-subsystem "
                 "only (partial region); DeltaZPE-dd = ZPE_saddle - ZPE_endA."),
    }

    out_json = out_dir / "mack_vfe_saddle_freq_dft.json"
    out_json.write_text(json.dumps(result, indent=2, cls=NumpyEncoder))
    print(f"\n[result] {out_json}", flush=True)
    print(json.dumps(result, indent=2, cls=NumpyEncoder), flush=True)

    # write imaginary mode as an extxyz with the mode vector in an array
    if imag_indices:
        k0 = imag_indices[0]
        mode = np.asarray(vib.get_mode(k0))
        a2 = atoms.copy()
        a2.arrays["imag_mode"] = mode
        write(str(out_dir / "mack_vfe_imag_mode.extxyz"), a2)
        print(f"[result] {out_dir/'mack_vfe_imag_mode.extxyz'} (reactive mode vector)",
              flush=True)

    (work_dir / "DONE_freq_dft").write_text(verdict + "\n")
    print(f"[DONE] verdict={verdict} n_imag={n_imag} imag={imag_list} cm-1", flush=True)


if __name__ == "__main__":
    main()
