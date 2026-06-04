#!/usr/bin/env python3
"""DFT partial-Hessian frequency at the pyrite V_Fe+H NEB saddle (v5 image 4).

PURPOSE (s158 -> v6 dimer prep; data-gathering BEFORE chemist+physicist consilium):
  (a) confirm index-1 TS at DFT level (exactly 1 imaginary mode),
  (b) extract the reactive negative-mode eigenvector = SEED for the v6 dimer/Sella,
  (c) ZPE of the reactive subsystem.

APPLES-TO-APPLES with the v5 constrained NEB (so the freq is comparable to the
E_a=268.9 meV band number): IDENTICAL QE calculator --
  nspin=1, ecutwfc=60 Ry, ecutrho=480 Ry, gaussian smearing degauss=0.01 Ry,
  kpts 2x2x2, ONCV-SR PBE (/opt/pp/oncv_pbe), NO Hubbard U, david, plain mix 0.3.
ONE deliberate change: conv_thr 1e-8 -> 1e-10. A Hessian is a 2nd derivative of
energy (1st of forces); SCF/force noise is amplified, so the frequency run needs a
tighter electronic convergence than the geometry optimisation did. (Flagged for
consilium -- 1e-10 is the conservative choice; 1e-9 likely also fine.)

REGION (default [95(H), 18(S_i), 71(S_k)]): the migrating proton + its two S
anchors -- SAME 3-atom partial Hessian as the MLIP MACE check (n_imag=1, 209.4i),
for a direct DFT-vs-MLIP comparison. Configurable via --indices. (Consilium may
expand to include nearest Fe / 2nd-shell S for a fuller ZPE; this run is the
reactive-subsystem baseline.)

Cost: 3 atoms x 3 dirs x 2 (central diff, nfree=2) = 18 SCF on 96-atom A100.
~150-300 s/SCF -> ~1-1.5 h, ~$1-2.

Output: <output_dir>/pyr_vfe_saddle_freq_dft.json (freqs cm-1 + n_imaginary +
imaginary eigenvector as [idx,dx,dy,dz] + ZPE + real_freqs) + vib cache pckls
(full Hessian, recoverable) + imaginary_mode.extxyz (mode vector in an array).
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
# QE relax write -> SinglePointCalculator AssertionError). s158 Phase-2 fix: the
# re-relaxed endA xyz is written with a calc attached; strip unknown props on read.
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


def make_calc(work_dir, label, kpts=(2, 2, 2), ecutwfc=60.0, ecutrho=480.0,
              mpi_np=1, conv_thr=1.0e-10, degauss=0.01, nspin=1):
    """IDENTICAL to v5 make_calc (smearing branch) except conv_thr default 1e-10."""
    pseudo_files = {"Fe": "Fe.upf", "S": "S.upf", "H": "H.upf"}
    cmd = f"mpirun --allow-run-as-root --bind-to none -np {mpi_np} {PW_BIN}"
    profile = EspressoProfile(command=cmd, pseudo_dir=PSEUDO_DIR)
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
            # NO Hubbard U (community consensus, matches v5)
        },
        "electrons": {
            "conv_thr": conv_thr,
            "mixing_mode": "plain",
            "mixing_beta": 0.3,
            "electron_maxstep": 200,
            "diagonalization": "david",
        },
    }
    print(f"[make_calc] label={label} system={input_data['system']} conv_thr={conv_thr}",
          flush=True)
    return Espresso(profile=profile, directory=str(Path(work_dir) / label),
                    input_data=input_data, pseudopotentials=pseudo_files,
                    kpts=tuple(kpts), koffset=(0, 0, 0))


def n_valence_electrons(atoms):
    # ONCV-SR PBE valence: Fe 16, S 6, H 1 (matches v5 parity logic).
    val = {"Fe": 16, "S": 6, "H": 1}
    return sum(val[s] for s in atoms.get_chemical_symbols())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saddle-xyz", required=True,
                    help="v5 saddle geometry (image 4 extxyz)")
    ap.add_argument("--work-dir", default="/workspace/pyr_vfe_freq_dft")
    ap.add_argument("--output-dir", default="/workspace/results")
    ap.add_argument("--indices", default="95,18,71",
                    help="comma atom indices for partial Hessian (default H,S_i,S_k)")
    ap.add_argument("--delta", type=float, default=0.01, help="displacement (Ang)")
    ap.add_argument("--conv-thr", type=float, default=1.0e-11,
                    help="QE conv_thr; 1e-11 for soft-mode Hessian SNR (physicist s158)")
    ap.add_argument("--nfree", type=int, default=2, choices=(2, 4),
                    help="finite-diff points: 2=central(cheap seed), 4=5pt(flat-top anharm)")
    ap.add_argument("--imag-thresh-cm1", type=float, default=20.0,
                    help="|imag| below this = numerical-zero (translational), NOT a real "
                         "imaginary mode. partial Hessian keeps translations (s158 phys+chem)")
    ap.add_argument("--kpts", default="2,2,2")
    ap.add_argument("--mpi-np", type=int, default=1)
    ap.add_argument("--nspin", type=int, default=1)
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    out_dir = Path(args.output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = [int(x) for x in args.indices.split(",")]
    kpts = tuple(int(x) for x in args.kpts.split(","))

    atoms = read(args.saddle_xyz)
    print(f"[load] {args.saddle_xyz}: {len(atoms)} atoms, pbc={atoms.get_pbc()}",
          flush=True)

    # --- parity sanity (M0 guard, smearing -> WARN only) ---
    n_e = n_valence_electrons(atoms)
    par = "ODD" if n_e % 2 else "EVEN"
    print(f"[parity] N_e={n_e} ({par}), nspin={args.nspin}, occ=smearing", flush=True)
    if n_e % 2 == 1 and args.nspin == 1:
        print("[parity] odd-e + nspin=1 + smearing -> half-e at E_F is OK "
              "(s158 pyrite confirmed non-magnetic). Proceeding.", flush=True)

    # --- region geometry echo (sanity) ---
    for i in indices:
        sym = atoms[i].symbol
        print(f"[region] atom {i} = {sym}", flush=True)
    if 95 in indices and 18 in indices and 71 in indices:
        print(f"[region] d(H95-S18)={atoms.get_distance(95,18,mic=True):.3f}  "
              f"d(H95-S71)={atoms.get_distance(95,71,mic=True):.3f}", flush=True)

    atoms.calc = make_calc(work_dir, "freq", kpts=kpts, conv_thr=args.conv_thr,
                           mpi_np=args.mpi_np, nspin=args.nspin)

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
    # NOTE (s158 phys+chem QA): a PARTIAL Hessian does NOT project out the region's
    # translational modes -> 2-3 near-zero modes survive; numerical noise can give
    # them a tiny imaginary part and trip abs(imag)>abs(real). So a mode counts as
    # IMAGINARY only if abs(imag) > imag_thresh_cm1 (default 20). Sub-threshold
    # near-zero modes are reported separately (translational/spurious), NOT counted.
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
        "tool": "pyr_vfe_saddle_freq_dft",
        "level": "DFT (QE pw.x, ONCV-SR PBE)",
        "calc": {"nspin": args.nspin, "ecutwfc": 60.0, "ecutrho": 480.0,
                 "smearing": "gaussian", "degauss": 0.01, "kpts": list(kpts),
                 "conv_thr": args.conv_thr, "U": 0, "delta_Ang": args.delta,
                 "nfree": args.nfree, "imag_thresh_cm1": thr},
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
        "note": ("DFT-grade partial-Hessian over reactive triad (H + 2 S anchors), "
                 "apples-to-apples with v5 NEB calc. Imaginary eigenvector = dimer "
                 "seed for v6. ZPE is reactive-subsystem only (partial region)."),
    }

    out_json = out_dir / "pyr_vfe_saddle_freq_dft.json"
    out_json.write_text(json.dumps(result, indent=2, cls=NumpyEncoder))
    print(f"\n[result] {out_json}", flush=True)
    print(json.dumps(result, indent=2, cls=NumpyEncoder), flush=True)

    # write imaginary mode as an extxyz with the mode vector in an array
    if imag_indices:
        k0 = imag_indices[0]
        mode = np.asarray(vib.get_mode(k0))
        a2 = atoms.copy()
        a2.arrays["imag_mode"] = mode
        write(str(out_dir / "pyr_vfe_imag_mode.extxyz"), a2)
        print(f"[result] {out_dir/'pyr_vfe_imag_mode.extxyz'} (dimer seed vector)",
              flush=True)

    (work_dir / "DONE_freq_dft").write_text(verdict + "\n")
    print(f"[DONE] verdict={verdict} n_imag={n_imag} imag={imag_list} cm-1", flush=True)


if __name__ == "__main__":
    main()
