#!/usr/bin/env python3
"""DFT partial-Hessian frequency at the marcasite V_Fe+H NEB saddle.

PURPOSE (s161 ZPE campaign; apples-to-apples with the marc V_Fe NEB v4c):
  (a) confirm index-1 TS at DFT level (exactly 1 imaginary mode),
  (b) extract the reactive negative-mode eigenvector,
  (c) ZPE of the reactive subsystem (DeltaZPE-double-dagger = ZPE_saddle - ZPE_endA).

APPLES-TO-APPLES with the marc V_Fe NEB (paper-quotable 177 meV): REUSE the
production calculator machinery (make_calc, apply_uniform_init_marc,
MarcasiteEspresso) imported from neb_canonical_marc_96at_qe_VFe.py so the freq SCF
uses the SAME settings -- nspin=2 (defect), PBE+U=2.0 Dudarev on Fe 3d (QE 7.x
HUBBARD card), Yang uniform 0.065 muB/Fe init, mv (cold/Marzari-Vanderbilt)
smearing degauss=0.015, ecutwfc=60 Ry, ecutrho=240 Ry, kpts 2x2x3, mixing
local-TF (the prod-v4c image_04 pwi used 2 2 3 anisotropic Pnnm + local-TF).

IMPORT MODEL: the production NEB script neb_canonical_marc_96at_qe_VFe.py must sit
next to this file in the working directory on the instance. We sys.path.insert the
script's own directory so the import resolves from cwd.

PARITY: marc defect Fe31S64H1 = 881 valence e (ODD) -> nspin=2 is correct (no warn).

REGION: H + the reactive S anchors + 1st shell. Either explicit --indices, or
auto-selected via --auto-region-cutoff (all non-H atoms within cutoff Angstrom of
--h-index, mic=True), asserting H + >=2 S + >=1 Fe and size in [8, 15]
(feedback_partial_hessian_region_too_small).

Output: <output_dir>/marc_vfe_saddle_freq_dft.json + vib cache pckls +
marc_vfe_imag_mode.extxyz (mode vector in an array).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import ase
from ase.io import read, write
from ase.vibrations import Vibrations

# Make the production NEB script importable from the working directory on the
# instance (it is expected to sit next to this file).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neb_canonical_marc_96at_qe_VFe import (  # noqa: E402
    make_calc,
    apply_uniform_init_marc,
    MarcasiteEspresso,  # noqa: F401  (re-exported / referenced for clarity)
)

# FIX 4 (s161 /test): eq-point magnetic report via the dft-neb parser. The module
# is expected next to this file on the instance (same dir we sys.path.insert'd).
# Wrap in try/except so a missing file degrades gracefully (no Mtot report) rather
# than killing the whole freq run.
try:
    from magnetic_output_parser import parse_qe_output, read_text  # noqa: E402
    _HAVE_MAG_PARSER = True
except Exception as _e:  # ImportError or anything else
    parse_qe_output = None  # type: ignore
    read_text = None  # type: ignore
    _HAVE_MAG_PARSER = False
    _MAG_PARSER_ERR = _e

# tolerate extxyz carrying QE calc results ASE rejects (e.g. 'nspins' from a prior
# QE relax write -> SinglePointCalculator AssertionError). strip unknown props on read.
from ase.calculators.singlepoint import SinglePointCalculator as _SPC  # noqa: E402
import ase.calculators.singlepoint as _spmod  # noqa: E402
_spc_orig_init = _SPC.__init__
def _spc_tolerant_init(self, atoms, **results):
    results = {k: v for k, v in results.items() if k in _spmod.all_properties}
    _spc_orig_init(self, atoms, **results)
_SPC.__init__ = _spc_tolerant_init


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


def n_valence_electrons(atoms):
    # ONCV-SR PBE valence: Fe 16, S 6, H 1 (matches marc NEB parity logic).
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
                    help="marc V_Fe saddle geometry (NEB image_04 extxyz)")
    ap.add_argument("--work-dir", default="/workspace/marc_vfe_freq_dft")
    ap.add_argument("--output-dir", default="/workspace/results")
    ap.add_argument("--indices", default="95,15,35",
                    help="comma atom indices for partial Hessian (used if "
                         "--auto-region-cutoff=0)")
    ap.add_argument("--auto-region-cutoff", type=float, default=0.0,
                    help="if >0 and --h-index set: region = H + all non-H atoms "
                         "within this many Angstrom of H (mic). 0 = use --indices.")
    ap.add_argument("--h-index", type=int, default=None,
                    help="index of the migrating proton (required for auto-region)")
    ap.add_argument("--delta", type=float, default=0.01, help="displacement (Ang)")
    ap.add_argument("--nfree", type=int, default=2, choices=(2, 4),
                    help="finite-diff points: 2=central(cheap), 4=5pt(flat-top anharm)")
    ap.add_argument("--imag-thresh-cm1", type=float, default=20.0,
                    help="|imag| below this = numerical-zero (translational), NOT a real "
                         "imaginary mode. partial Hessian keeps translations")
    ap.add_argument("--kpts", default="2,2,3",
                    help="Monkhorst-Pack grid. Default 2,2,3 matches the prod-v4c "
                         "image_04 pwi (anisotropic Pnnm c~13.5 vs a~8.9). NOT 2,2,2.")
    ap.add_argument("--mpi-np", type=int, default=1)
    # FIX 2 (s161 /test BLOCKER 2): pin the magnetic sheet for ALL displacement SCF.
    ap.add_argument("--tot-magnetization", type=float, default=None,
                    help="if set, inject tot_magnetization into the QE system namelist "
                         "(single magnetic sheet across all displacements; use the "
                         "settled saddle/endA GS value, ~1.1 uB per the s161 pre-flight). "
                         "None = free magnetization (legacy).")
    # FIX 4 (s161 /test BLOCKER 2 verify): eq-point magnetic sanity vs v4c saddle.
    ap.add_argument("--expect-mtot", type=float, default=None,
                    help="if set, WARN (not abort) when |eq Mtot - expect| > --mtot-tol. "
                         "v4c saddle Mtot ~ 1.09 uB.")
    ap.add_argument("--mtot-tol", type=float, default=0.3,
                    help="tolerance for --expect-mtot sanity warning (uB)")
    ap.add_argument("--conv-thr", type=float, default=1.0e-9,
                    help="QE conv_thr; 1e-9 default -- nspin=2 Fe-S metal will not "
                         "converge at 1e-10/1e-11. Flag exposed for /test.")
    ap.add_argument("--ecutwfc", type=float, default=60.0)
    ap.add_argument("--ecutrho", type=float, default=240.0)
    ap.add_argument("--degauss", type=float, default=0.015)
    ap.add_argument("--smearing", default="mv",
                    help="mv = Marzari-Vanderbilt (cold), matches marc NEB v4c")
    ap.add_argument("--hubbard-u-fe", type=float, default=2.0,
                    help="Dudarev U_eff on Fe 3d (eV); matches marc NEB v4c")
    ap.add_argument("--magmom-per-fe", type=float, default=0.065,
                    help="Yang uniform init muB/Fe (net ~2.0 muB over 31 Fe)")
    ap.add_argument("--pseudo-dir", default=None,
                    help="QE pseudo dir (default: production PSEUDO_DIR)")
    ap.add_argument("--pp-fe", default="Fe.upf")
    ap.add_argument("--pp-s", default="S.upf")
    ap.add_argument("--pp-h", default="H.upf")
    args = ap.parse_args()

    # FIX 1 (s161 /test BLOCKER 1): log ASE version -- the HUBBARD guard below
    # depends on whether MarcasiteEspresso.write_input is actually called.
    print(f"[env] ase.__version__ = {ase.__version__}", flush=True)
    if not _HAVE_MAG_PARSER:
        print(f"[mag-parser] WARNING: magnetic_output_parser import failed "
              f"({_MAG_PARSER_ERR!r}); eq-point Mtot/Mabs report disabled. "
              f"Place magnetic_output_parser.py next to this script on the instance.",
              flush=True)

    work_dir = Path(args.work_dir)
    out_dir = Path(args.output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    kpts = tuple(int(x) for x in args.kpts.split(","))
    pseudo_files = {"Fe": args.pp_fe, "S": args.pp_s, "H": args.pp_h}

    atoms = read(args.saddle_xyz)
    print(f"[load] {args.saddle_xyz}: {len(atoms)} atoms, pbc={atoms.get_pbc()}",
          flush=True)

    # --- parity sanity: marc defect Fe31S64H1 = 881 e (ODD) -> nspin=2 correct ---
    n_e = n_valence_electrons(atoms)
    par = "ODD" if n_e % 2 else "EVEN"
    print(f"[parity] N_e={n_e} ({par}); marc V_Fe defect -> nspin=2 (correct, "
          f"Wang 2019 magnetic metal). No warn.", flush=True)

    # --- Yang uniform magmom init BEFORE attaching the calculator ---
    apply_uniform_init_marc(atoms, magmom_per_fe=args.magmom_per_fe)

    # is_defect=True -> make_calc sets nspin=2 itself.
    # FIX 3 (s161 /test): mixing_mode="local-TF" matches prod-v4c (make_calc default
    # is 'plain'); apples-to-apples for the nspin=2 Fe-S metallic-defect SCF.
    atoms.calc = make_calc(
        work_dir, "freq", kpts=kpts,
        ecutwfc=args.ecutwfc, ecutrho=args.ecutrho,
        mpi_np=args.mpi_np, conv_thr=args.conv_thr,
        mixing_mode="local-TF",
        occupations="smearing", smearing=args.smearing, degauss=args.degauss,
        is_defect=True,
        hubbard_u_fe=args.hubbard_u_fe, hubbard_u_kind="dudarev",
        pseudo_files=pseudo_files, pseudo_dir=args.pseudo_dir,
    )

    # FIX 2 (s161 /test BLOCKER 2): pin the magnetic sheet via tot_magnetization.
    # make_calc (production) does NOT accept a tot_magnetization kwarg, so we inject
    # it directly into the calculator's input_data. ASE Espresso (GenericFileIOCalculator)
    # stores its constructor kwargs in self.parameters, so input_data is reachable at
    # calc.parameters['input_data']; mutate the 'system' sub-dict in place. This is the
    # exact dict the writer serializes, so the value lands in the QE &system namelist.
    if args.tot_magnetization is not None:
        params = atoms.calc.parameters
        idata = params.get("input_data")
        if idata is None or "system" not in idata:
            print("[tot-mag] FATAL: could not locate input_data['system'] on calc "
                  "(ASE Espresso layout changed); cannot enforce single sheet.",
                  flush=True)
            sys.exit(1)
        idata["system"]["tot_magnetization"] = float(args.tot_magnetization)
        print(f"[tot-mag] tot_magnetization = {args.tot_magnetization} uB injected into "
              f"QE &system (single magnetic sheet pinned for ALL displacement SCF).",
              flush=True)
    else:
        print("[tot-mag] tot_magnetization NOT set (free magnetization, legacy).",
              flush=True)

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

    # --- equilibrium-point SCF (one undisplaced SCF) ---
    # Serves three purposes (s161 /test): (1) FIX 1 HUBBARD guard -- check the
    # actually-written pwi carries the HUBBARD card; (2) FIX 4 eq magnetic report
    # via the dft-neb parser; (3) warms a converged state. Vibrations only runs
    # DISPLACED points, so this eq SCF is not otherwise computed. It writes into the
    # same calc directory (work_dir/freq) which the first displacement will overwrite;
    # the Vibrations cache lives in separate <vibname>.*.pckl files -> no conflict.
    print("[eq] running equilibrium-point SCF (guard + magnetic report)...", flush=True)
    eq_energy = float(atoms.get_potential_energy())
    print(f"[eq] eq SCF done, E = {eq_energy:.6f} eV", flush=True)

    freq_dir = work_dir / "freq"
    pwi_path = freq_dir / "espresso.pwi"
    pwo_path = freq_dir / "espresso.pwo"

    # --- FIX 1 (BLOCKER 1): HUBBARD card must be present if U requested ---
    # On modern ASE (>=3.24) Espresso is a GenericFileIOCalculator and never calls
    # MarcasiteEspresso.write_input -> the HUBBARD card injection is dead code and U
    # is silently dropped (calc runs plain PBE). Fail-fast by grepping the pwi.
    if args.hubbard_u_fe > 1e-6:
        if not pwi_path.exists():
            print(f"[hubbard-guard] FATAL: expected pwi not found at {pwi_path}; "
                  f"cannot verify HUBBARD card.", flush=True)
            sys.exit(1)
        pwi_text = pwi_path.read_text(encoding="utf-8", errors="replace")
        if "hubbard" not in pwi_text.lower():
            print(f"[hubbard-guard] FATAL: --hubbard-u-fe={args.hubbard_u_fe} requested "
                  f"but NO 'HUBBARD' card in the written pwi ({pwi_path}). U was silently "
                  f"DROPPED -> this would run plain PBE, NOT apples-to-apples with the "
                  f"marc V_Fe NEB v4c. Cause: ASE {ase.__version__} >= 3.24 does not call "
                  f"MarcasiteEspresso.write_input (GenericFileIOCalculator). Fix: port the "
                  f"injection to ase.io.espresso.write_espresso_in + explicit HUBBARD append "
                  f"(see knowledge/S152_LESSONS.md Lesson 3), or use ASE <= 3.23.",
                  flush=True)
            sys.exit(1)
        print(f"[hubbard-guard] OK: HUBBARD card present in pwi (U_Fe={args.hubbard_u_fe}).",
              flush=True)

    # --- tot_magnetization guard: verify the injection actually reached the pwi ---
    # The calc.parameters['input_data'] mutation is ASE-version-dependent; a silent
    # miss would un-pin the magnetic sheet (same silent-failure class as HUBBARD).
    # Grep the written pwi to be version-agnostic.
    if args.tot_magnetization is not None:
        if not pwi_path.exists():
            print(f"[tot-mag-guard] FATAL: pwi not found at {pwi_path}; cannot verify "
                  f"tot_magnetization.", flush=True)
            sys.exit(1)
        pwi_text_tm = pwi_path.read_text(encoding="utf-8", errors="replace")
        if "tot_magnetization" not in pwi_text_tm.lower():
            print(f"[tot-mag-guard] FATAL: --tot-magnetization={args.tot_magnetization} "
                  f"requested but NO 'tot_magnetization' in the written pwi ({pwi_path}). "
                  f"Injection into calc.parameters['input_data']['system'] did NOT reach "
                  f"the &system namelist (ASE {ase.__version__} layout mismatch) -> the "
                  f"magnetic sheet is NOT pinned. Fix injection path before trusting ZPE.",
                  flush=True)
            sys.exit(1)
        print(f"[tot-mag-guard] OK: tot_magnetization in pwi "
              f"({args.tot_magnetization} uB pinned).", flush=True)

    # --- FIX 4 (BLOCKER 2 verify): eq-point magnetic report via dft-neb parser ---
    eq_Mtot = eq_Mabs = None
    eq_mag_settled = None
    if _HAVE_MAG_PARSER and pwo_path.exists():
        try:
            mag = parse_qe_output(pwo_path, read_text(pwo_path))
            eq_Mtot = mag.total_magnetization_uB
            eq_Mabs = mag.absolute_magnetization_uB
            eq_mag_settled = mag.magnetization_settled
            print(f"[eq-mag] Mtot={eq_Mtot} uB Mabs={eq_Mabs} uB "
                  f"dMtot={mag.total_magnetization_drift_uB} dMabs={mag.absolute_magnetization_drift_uB} "
                  f"settled={eq_mag_settled} nspin={mag.nspin}", flush=True)
            if eq_mag_settled is False:
                print("[eq-mag] WARNING: magnetization NOT settled at eq SCF end; "
                      "consider tighter conv_thr or more electron_maxstep before trusting "
                      "the Hessian.", flush=True)
            if args.expect_mtot is not None and eq_Mtot is not None:
                if abs(eq_Mtot - args.expect_mtot) > args.mtot_tol:
                    print(f"[eq-mag] WARNING: eq Mtot={eq_Mtot} uB deviates from "
                          f"--expect-mtot={args.expect_mtot} by "
                          f"{abs(eq_Mtot - args.expect_mtot):.3f} > tol {args.mtot_tol} uB "
                          f"(v4c saddle Mtot ~ 1.09 uB).", flush=True)
        except Exception as e:
            print(f"[eq-mag] WARNING: magnetic parse failed: {e!r}", flush=True)
    elif _HAVE_MAG_PARSER:
        print(f"[eq-mag] WARNING: pwo not found at {pwo_path}; skipping magnetic report.",
              flush=True)

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
        "tool": "marc_vfe_saddle_freq_dft",
        "level": "DFT (QE pw.x, PBE+U Dudarev)",
        "calc": {"nspin": 2, "ecutwfc": args.ecutwfc, "ecutrho": args.ecutrho,
                 "smearing": args.smearing, "degauss": args.degauss,
                 "kpts": list(kpts), "conv_thr": args.conv_thr,
                 "mixing_mode": "local-TF",
                 "tot_magnetization": args.tot_magnetization,
                 "U": args.hubbard_u_fe, "U_kind": "dudarev",
                 "magmom_per_fe": args.magmom_per_fe,
                 "delta_Ang": args.delta, "nfree": args.nfree,
                 "imag_thresh_cm1": thr,
                 "pseudo_dir": args.pseudo_dir, "pseudo_files": pseudo_files},
        "saddle_xyz": args.saddle_xyz,
        "hessian_region": indices,
        "n_atoms": len(atoms),
        "n_valence_e": n_e,
        "eq_energy_eV": round(eq_energy, 6),
        "eq_Mtot_uB": eq_Mtot,
        "eq_Mabs_uB": eq_Mabs,
        "eq_mag_settled": eq_mag_settled,
        "n_imaginary": n_imag,
        "imaginary_freqs_cm1": imag_list,
        "real_freqs_cm1": real_list,
        "nearzero_freqs_cm1": nearzero_list,
        "imaginary_modes": imag_modes,
        "zpe_real_modes_eV": round(zpe_eV, 5),
        "verdict": verdict,
        "note": ("DFT-grade partial-Hessian over reactive region (H + S anchors + "
                 "1st shell) for marcasite V_Fe, apples-to-apples with the marc V_Fe "
                 "NEB v4c (nspin=2, U=2 Dudarev, Yang init, mv smearing, ecutrho 240). "
                 "ZPE is reactive-subsystem only; DeltaZPE-dd = ZPE_saddle - ZPE_endA."),
    }

    out_json = out_dir / "marc_vfe_saddle_freq_dft.json"
    out_json.write_text(json.dumps(result, indent=2, cls=NumpyEncoder))
    print(f"\n[result] {out_json}", flush=True)
    print(json.dumps(result, indent=2, cls=NumpyEncoder), flush=True)

    # write imaginary mode as an extxyz with the mode vector in an array
    if imag_indices:
        k0 = imag_indices[0]
        mode = np.asarray(vib.get_mode(k0))
        a2 = atoms.copy()
        a2.arrays["imag_mode"] = mode
        write(str(out_dir / "marc_vfe_imag_mode.extxyz"), a2)
        print(f"[result] {out_dir/'marc_vfe_imag_mode.extxyz'} (reactive mode vector)",
              flush=True)

    (work_dir / "DONE_freq_dft").write_text(verdict + "\n")
    print(f"[DONE] verdict={verdict} n_imag={n_imag} imag={imag_list} cm-1", flush=True)


if __name__ == "__main__":
    main()
