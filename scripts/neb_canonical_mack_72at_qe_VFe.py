#!/usr/bin/env python3
"""
Canonical V_Fe + H lateral hop NEB -- mackinawite FeS, 3x3x2 = 71 atoms (after V_Fe).

Adapted from neb_canonical_mack_72at_qe.py (V_S+H variant) per s132 chemist+physicist
joint verdict + MLIP topology validation:
  - V_S+H lateral hop confirmed BROKEN (single-broad-Fe-pocket-well, 43/48 MACE collapse,
    DFT smoke endA==endB single basin, pent V_S+H confirmed same artifact at BFGS step 14)
  - V_Fe+H MLIP scan (s132 gomer) shows 3 distinct S-H basins in MACE (54% S-H, 24% Fe collapse;
    cf. V_S MACE 8% S-H / 90% Fe collapse). CHGNet AMBIGUOUS but partial S-H.
  - Liu 2021 ACS Omega L-650: V_Fe surface E_a = 0.26 eV (literature anchor).

Reaction coordinate (this script):
  - Build mack 3x3x2 (72 atoms pristine)
  - Pick central Fe atom (V_Fe = vacancy_atom).
  - Find 4 S neighbours of V_Fe within fe_s_max ~ 2.7 Å (Fe-S nominal 2.25 Å in mack).
  - Pick 2 S atoms (S_i, S_k) from the 4 V_Fe neighbours: choose the pair with
    maximum d(S_i, S_k) within the pocket (maximizes lateral hop length, gives
    distinct S-H basins on either side of pocket).
  - Build endA: H placed 1.35 Å from S_i toward V_Fe (S-H configuration in pocket).
  - Build endB: H placed 1.35 Å from S_k toward V_Fe (S-H on other side).
  - Relax pristine + endA + endB BFGS, then CI-NEB FIRE 9 images.

Expected E_a: 0.2-0.5 eV (Liu 2021 surface 0.26, bulk should be similar +/- 0.1).

s129/s132 patches applied (all of):
  - canonical_triple.json save/load (REUSE bypasses re-picker)
  - try/finally on SinglePointCalculator monkey-patch
  - assert REUSE atom count = expected (4 atoms/primitive × supercell - 1 Fe)
  - post-prewrap recompute e_B + dE_endpoints
  - PAW pseudo detection
  - --skip-endpoints / --skip-neb / --disk-io-neb / --wfc-reuse / --reuse-relaxed
  - --idpp-prewrap (default ON, MANDATORY per ASE issue #1130)
  - Initial path E_a sanity check (>5 eV → abort FIRE)
  - Sanity: post-relax expect d_H_min ~ 1.34-1.50 Å (S-H covalent), warn if Fe-H signature

References:
  - knowledge/MACK_NEB_LIT_REVIEW_2026-05-03.md (Liu 2021, V_Fe RC literature)
  - knowledge/MACK_PENT_NEB_PROTOCOL_BUG_2026-05-02.md (V_S+H artifact forensic)
  - paper/SI/SI_vfe_mlip_scan.md (s132 MLIP topology validation, 80 candidates)
  - knowledge/QE_BATCH_LESSONS.md #40, #43, #50, #54, #61, #62
  - knowledge/Q115_DFT_PROTOCOL_FINAL_2026-04-28.md §2.2 (PWSCF, nspin=1, ecutwfc=60, smearing)
"""

import warnings
warnings.filterwarnings("ignore")

# Q-115 ERRATA #6 part D (CS s132 evening): seed numpy random for IDPP
# bit-reproducibility. ASE IDPP uses np.random для path perturbation when
# linear init близок к saddle. Two independent runs c same script otherwise
# may pick different mirror branches due BLAS thread order numerics.
import os as _os
_os.environ.setdefault("PYTHONHASHSEED", "0")
import numpy as _np_seed
_np_seed.random.seed(42)

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from ase import Atom
from ase.spacegroup import crystal
from ase.optimize import BFGS, FIRE
from ase.mep import DyNEB, NEB
from ase.io import read, write
from ase.geometry import find_mic
from ase.calculators.espresso import Espresso, EspressoProfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_neb_band(images, args):
    """Create production NEB band using the selected optimizer policy."""
    if args.dyneb:
        neb = DyNEB(
            images,
            climb=True,
            method="improvedtangent",
            k=args.k_spring,
            fmax=args.fmax_neb,
            dynamic_relaxation=True,
            scale_fmax=args.dyneb_scale_fmax,
        )
        policy = "DyNEB"
    else:
        neb = NEB(images, climb=True, method="improvedtangent", k=args.k_spring)
        policy = "NEB"
    print(
        f"[NEB POLICY] {policy}: dynamic_relaxation={bool(args.dyneb)}, "
        f"scale_fmax={args.dyneb_scale_fmax}, fmax={args.fmax_neb}",
        flush=True,
    )
    return neb, {
        "neb_policy": policy,
        "dyneb_enabled": bool(args.dyneb),
        "dyneb_dynamic_relaxation": bool(args.dyneb),
        "dyneb_scale_fmax": float(args.dyneb_scale_fmax),
    }


MINERAL_TAG = "mack_72at_qe_VFe"
LOCK_FILE = Path("/workspace/.neb_canonical_mack_qe_VFe.lock")

PW_BIN = os.environ.get("PW_BIN", "pw.x")
PSEUDO_DIR = os.environ.get("ESPRESSO_PSEUDO",
                            os.environ.get("PSEUDO_DIR", "/opt/pp/oncv_pbe"))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def acquire_singleton():
    """python3-only singleton matched by THIS script name."""
    try:
        out = subprocess.check_output(
            ["ps", "-C", "python3", "-o", "pid=,args="], text=True
        )
        my_pid = os.getpid()
        for line in out.strip().splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) < 2:
                continue
            pid_str, args_str = parts
            if "neb_canonical_mack_72at_qe_VFe.py" not in args_str:
                continue
            try:
                pid = int(pid_str)
            except ValueError:
                continue
            if pid == my_pid:
                continue
            print(f"[SINGLETON] another instance pid={pid}: {args_str}", flush=True)
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
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
    except Exception:
        pass


# ============================================================
# Builder -- mackinawite 3x3x2 = 72 atoms
# ============================================================

def build_mackinawite(repeat=(3, 3, 2)):
    """Mackinawite FeS, P4/nmm (129), a=3.674 Å, c=5.033 Å."""
    unit = crystal(
        symbols=["Fe", "S"],
        basis=[(0.0, 0.0, 0.0), (0.0, 0.5, 0.2602)],
        spacegroup=129,
        cellpar=[3.674, 3.674, 5.033, 90, 90, 90],
    )
    return unit.repeat(repeat)


# ============================================================
# V_Fe vacancy picker + 2 S neighbour selector
# ============================================================

def pick_fe_vacancy_and_two_s_neighbours(atoms, fe_s_max=2.7, mineral_name=""):
    """Pick a central Fe atom (V_Fe) and TWO of its 4 S neighbours (S_i, S_k).

    H starts on S_i (endA) and moves to S_k (endB). NEB measures lateral hop
    barrier through the V_Fe pocket.

    Selection criteria:
      - Fe nearest to cell centre (deterministic between MACE/CHGNet/DFT runs).
      - Must have ≥4 S neighbours within fe_s_max (typical Fe-S in mack ~2.25 Å,
        2nd-nearest S ~3.5 Å; threshold 2.7 Å captures only first shell).
      - From those 4 S, pick the pair (S_i, S_k) with MAXIMUM d(S_i, S_k)
        within the pocket. This:
          (a) maximizes lateral hop length → distinct H-position basins;
          (b) minimizes endA / endB H-position overlap;
          (c) consistent with MLIP scan finding (s132): 3 distinct S-H basins
              ~ 3.6-3.9 Å apart in MACE V_Fe scan.

    Returns: (fe_v_idx, s_i_idx, s_k_idx, hop_distance_si_sk_A,
              all_4_s_neighbours_list)
    """
    syms = atoms.get_chemical_symbols()
    fe_indices = [i for i, s in enumerate(syms) if s == "Fe"]
    if not fe_indices:
        raise RuntimeError("No Fe atoms found")

    # Pick Fe nearest cell centre (deterministic)
    cell_centre = atoms.cell.array.sum(axis=0) / 2
    fe_centre_d = [(float(np.linalg.norm(atoms.positions[i] - cell_centre)), i)
                   for i in fe_indices]
    fe_centre_d.sort()
    fe_v = fe_centre_d[0][1]

    # Find S neighbours of V_Fe within fe_s_max
    s_indices = [i for i, s in enumerate(syms) if s == "S"]
    s_with_d = []
    for s_idx in s_indices:
        d = float(atoms.get_distance(fe_v, s_idx, mic=True))
        if d <= fe_s_max:
            s_with_d.append((d, s_idx))
    s_with_d.sort()
    if len(s_with_d) < 4:
        raise RuntimeError(
            f"V_Fe at idx {fe_v} has only {len(s_with_d)} S within {fe_s_max} Å; "
            f"expected 4 (mack tetrahedral coordination)."
        )
    s_neighbours = [s_idx for _, s_idx in s_with_d[:4]]

    # From 4 S, find pair with MAXIMUM d(S, S')
    best_pair = None
    best_d = -1.0
    candidates = []
    for a in range(4):
        for b in range(a + 1, 4):
            si, sk = s_neighbours[a], s_neighbours[b]
            d_si_sk = float(atoms.get_distance(si, sk, mic=True))
            candidates.append((si, sk, d_si_sk))
            if d_si_sk > best_d:
                best_d = d_si_sk
                best_pair = (si, sk, d_si_sk)

    if best_pair is None:
        raise RuntimeError(
            f"V_Fe at {fe_v}: no valid S-S pair among 4 neighbours {s_neighbours}"
        )

    s_i, s_k, hop_d = best_pair

    # Diagnostic (chemist QA: log dz to flag intra-layer vs cross-layer hop)
    fe_z = float(atoms.positions[fe_v, 2])
    print(f"[V_Fe pick mineral={mineral_name}] V_Fe={fe_v} z={fe_z:.3f} Å",
          flush=True)
    print(f"  4 S neighbours within {fe_s_max} Å: {s_neighbours}", flush=True)
    # FIX: s_with_d list contains (d, s_idx) tuples — unpack correctly
    for dist, s_idx in s_with_d[:4]:
        sz = float(atoms.positions[s_idx, 2])
        print(f"    S idx {s_idx}: d_FeS={dist:.3f} Å, z_S={sz:.3f} Å, "
              f"dz={sz-fe_z:+.3f} Å", flush=True)
    print(f"  S-S pair candidates (sorted by d_si_sk descending):", flush=True)
    for si_c, sk_c, d_c in sorted(candidates, key=lambda c: -c[2]):
        tag = " [SELECTED]" if (si_c, sk_c) == (s_i, s_k) else ""
        dz_pair = abs(float(atoms.positions[si_c, 2] - atoms.positions[sk_c, 2]))
        layer_kind = "cross-layer" if dz_pair > 1.0 else "in-layer"
        print(f"    S {si_c} <-> S {sk_c}: d={d_c:.3f} Å, "
              f"|dz|={dz_pair:.3f} Å [{layer_kind}]{tag}", flush=True)

    return fe_v, s_i, s_k, hop_d, s_neighbours


def place_h_on_neighbour_s(atoms, v_idx, h_anchor_idx, bond_length=1.35):
    """Remove atom at v_idx (Fe in this script's V_Fe variant), place H 1.35 Å
    from h_anchor_idx (S) along (S_anchor → V_Fe) direction.

    Same function as V_S variant — works for either vacancy species since it
    only does del+append+placement. Direction always points "into the pocket".
    """
    pos_v = atoms.positions[v_idx].copy()
    pos_a = atoms.positions[h_anchor_idx].copy()
    dvec = pos_v - pos_a
    cell = atoms.cell.array
    fc = np.linalg.solve(cell.T, dvec)
    fc -= np.round(fc)
    dvec_mic = cell.T @ fc
    dn = np.linalg.norm(dvec_mic)
    if dn < 1e-6:
        raise RuntimeError("Anchor and vacancy coincide")
    unit = dvec_mic / dn
    pos_h = pos_a + bond_length * unit
    new_atoms = atoms.copy()
    del new_atoms[v_idx]
    new_atoms.append(Atom("H", position=pos_h))
    return new_atoms


def prewrap_endpoint_for_idpp(initial, final, label="endB"):
    """Fix ASE GitLab issue #1130 (verbatim from V_S variant)."""
    assert all(initial.pbc), (
        f"prewrap_endpoint_for_idpp requires full 3D PBC, got pbc={initial.pbc}."
    )
    disp_naive = final.positions - initial.positions
    disp_mic, _ = find_mic(disp_naive, initial.cell, initial.pbc)
    diffs = np.linalg.norm(disp_naive - disp_mic, axis=1)
    wrapped_idx = np.where(diffs > 0.5)[0]
    summary = {
        "n_unwrapped":   int(len(wrapped_idx)),
        "max_disp_naive_A": float(np.linalg.norm(disp_naive, axis=1).max()),
        "max_disp_mic_A":   float(np.linalg.norm(disp_mic, axis=1).max()),
        "wrapped_atoms": [
            {"idx": int(i), "symbol": final.get_chemical_symbols()[i],
             "naive_A": float(np.linalg.norm(disp_naive[i])),
             "mic_A":   float(np.linalg.norm(disp_mic[i]))}
            for i in wrapped_idx
        ],
    }
    if len(wrapped_idx) == 0:
        return final, 0, summary
    out = final.copy()
    out.positions = initial.positions + disp_mic
    print(f"[IDPP-PREWRAP {label}] {len(wrapped_idx)} atom(s) unwrapped: "
          f"{[(int(i), final.get_chemical_symbols()[i]) for i in wrapped_idx]}",
          flush=True)
    for i in wrapped_idx:
        sym = final.get_chemical_symbols()[i]
        print(f"  atom {i} ({sym}): naive |Δ|={np.linalg.norm(disp_naive[i]):.3f} Å "
              f"→ mic |Δ|={np.linalg.norm(disp_mic[i]):.3f} Å", flush=True)
    return out, len(wrapped_idx), summary


# ============================================================
# Sanity diagnostic: H bonding signature (S-H vs Fe-H)
# ============================================================

def h_bonding_signature(atoms, label=""):
    """Returns dict with d(H, nearest S), d(H, nearest Fe), classification.

    For V_Fe + H endpoint we EXPECT S-H (covalent, d ~ 1.34-1.50 Å).
    Fe-H signature (d_Fe ~ 1.7-1.9 Å, d_S > 2.0 Å) indicates artifact
    (similar to mack V_S+H Fe-bridge collapse — see s130 forensic).
    """
    h_idx = len(atoms) - 1
    syms = atoms.get_chemical_symbols()
    if syms[h_idx] != "H":
        raise RuntimeError(f"last atom expected H, got {syms[h_idx]}")
    fe_d = []
    s_d = []
    for i in range(len(atoms)):
        if i == h_idx:
            continue
        d = float(atoms.get_distance(h_idx, i, mic=True))
        if syms[i] == "Fe":
            fe_d.append((d, i))
        elif syms[i] == "S":
            s_d.append((d, i))
    fe_d.sort()
    s_d.sort()
    d_S, idx_S = (s_d[0] if s_d else (float("inf"), -1))
    d_Fe, idx_Fe = (fe_d[0] if fe_d else (float("inf"), -1))

    if d_S < 1.55 and d_S < d_Fe:
        kind = "S-H (S covalent)"
        artifact = False
    elif d_Fe < 2.0 and d_Fe < d_S:
        kind = "Fe-H (Fe-bridge artifact)"
        artifact = True
    else:
        kind = "intermediate"
        artifact = False
    print(f"[H-SIG {label}] d_H_S={d_S:.3f} Å (S idx {idx_S}), "
          f"d_H_Fe={d_Fe:.3f} Å (Fe idx {idx_Fe}) → {kind}",
          flush=True)
    return {
        "d_H_S": d_S, "nearest_S_idx": idx_S,
        "d_H_Fe": d_Fe, "nearest_Fe_idx": idx_Fe,
        "kind": kind, "artifact_flag": bool(artifact),
    }


# ============================================================
# QE PWSCF calculator (verbatim from V_S variant)
# ============================================================

def make_calc(work_dir, label, kpts=(2, 2, 2),
              ecutwfc=60.0, ecutrho=240.0, mpi_np=1,
              conv_thr=1.0e-8, mixing_beta=0.2, mixing_mode="plain",
              disk_io="medium",
              occupations="smearing", smearing="gaussian", degauss=0.01,
              pseudo_files=None, pseudo_dir=None,
              wfc_reuse=False):
    if pseudo_files is None:
        pseudo_files = {"Fe": "Fe.upf", "S": "S.upf", "H": "H.upf"}
    if pseudo_dir is None:
        pseudo_dir = PSEUDO_DIR

    cmd = f"mpirun --allow-run-as-root --bind-to none -np {mpi_np} {PW_BIN}"
    profile = EspressoProfile(command=cmd, pseudo_dir=pseudo_dir)
    restart_mode = "restart" if wfc_reuse else "from_scratch"

    input_data = {
        "control": {
            "calculation":  "scf",
            "restart_mode": restart_mode,
            "tprnfor":      True,
            "tstress":      False,
            "verbosity":    "high",
            "disk_io":      disk_io,
            "outdir":       str(Path(work_dir) / label / "tmp"),
            "prefix":       label,
        },
        "system": {
            "ecutwfc":     ecutwfc,
            "ecutrho":     ecutrho,
            "occupations": occupations,
            "nspin":       1,
        },
        "electrons": {
            "conv_thr":         conv_thr,
            "mixing_mode":      mixing_mode,
            "mixing_beta":      mixing_beta,
            "electron_maxstep": 200,
            "diagonalization":  "david",
        },
    }

    if wfc_reuse:
        input_data["electrons"]["startingwfc"] = "file"

    if occupations == "smearing":
        input_data["system"]["smearing"] = smearing
        input_data["system"]["degauss"] = degauss

    print(f"[DEBUG make_calc] label={label} system={input_data['system']}", flush=True)

    return Espresso(
        profile=profile,
        directory=str(Path(work_dir) / label),
        input_data=input_data,
        pseudopotentials=pseudo_files,
        kpts=tuple(kpts),
        koffset=(0, 0, 0),
    )


# ============================================================
# Pipeline
# ============================================================

def run_mackinawite_VFe(args):
    t_start = time.time()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pseudo_files = {"Fe": args.pp_fe, "S": args.pp_s, "H": args.pp_h}

    calc_kwargs = dict(
        kpts=tuple(args.kpts),
        ecutwfc=args.ecutwfc,
        ecutrho=args.ecutrho,
        mpi_np=args.mpi_np,
        mixing_beta=args.mixing_beta,
        mixing_mode=args.mixing_mode,
        disk_io="medium",
        occupations="smearing",
        smearing=args.smearing,
        degauss=args.degauss,
        pseudo_files=pseudo_files,
        pseudo_dir=args.pseudo_dir,
        wfc_reuse=args.wfc_reuse,
    )
    neb_calc_kwargs = dict(calc_kwargs, disk_io=args.disk_io_neb)

    result = {
        "mineral":   "mackinawite",
        "vacancy_kind": "V_Fe",
        "cell_spec": f"{args.supercell[0]}x{args.supercell[1]}x{args.supercell[2]}",
        "code":      "QE PWSCF",
        "method":    f"PBE PWFFT, nspin=1 (PM surrogate), ecutwfc={args.ecutwfc} Ry",
        "protocol":  "canonical V_Fe + H lateral S-S hop (s132 chemist+physicist verdict)",
        "kpts":      list(args.kpts),
        "ecutwfc":   args.ecutwfc,
        "ecutrho":   args.ecutrho,
        "smearing":  args.smearing,
        "degauss":   args.degauss,
        "n_images":  args.n_images,
        "fmax_neb":  args.fmax_neb,
        "fmax_endpoint": args.fmax_endpoint,
        "fe_s_max":  args.fe_s_max,
        "decision_ref": (
            "RESHENIE-079 (Q-115 §2.2 PWSCF nspin=1) + s132 V_Fe pivot "
            "(MACK_NEB_LIT_REVIEW_2026-05-03.md, V_S+H broken artifact, "
            "MLIP V_Fe scan PASS)"
        ),
        "literature_anchor": "Liu 2021 ACS Omega L-650: V_Fe surface E_a = 0.26 eV",
    }

    # ---------- build + (relax | reuse) pristine ----------
    REUSE = args.reuse_relaxed is not None
    reused_triple = None
    if REUSE:
        import ase.calculators.singlepoint as _sp
        from ase.calculators.calculator import all_properties as _all_props
        _orig_spc_init = _sp.SinglePointCalculator.__init__
        def _patched_spc_init(self, atoms_obj, **results):
            filtered = {k: v for k, v in results.items() if k in _all_props}
            _orig_spc_init(self, atoms_obj, **filtered)
        _sp.SinglePointCalculator.__init__ = _patched_spc_init
        try:
            src = Path(args.reuse_relaxed)
            for fn in ("relaxed_pristine.xyz", "relaxed_endA.xyz", "relaxed_endB.xyz"):
                if not (src / fn).exists():
                    raise FileNotFoundError(f"--reuse-relaxed missing: {src / fn}")
            print(f"[REUSE] loading relaxed endpoints from {src}", flush=True)
            atoms = read(str(src / "relaxed_pristine.xyz"))
            pre_endA = read(str(src / "relaxed_endA.xyz"))
            pre_endB = read(str(src / "relaxed_endB.xyz"))
            atoms.calc = None
            pre_endA.calc = None
            pre_endB.calc = None
        finally:
            _sp.SinglePointCalculator.__init__ = _orig_spc_init
        # Mack 4 atoms per primitive (Fe2S2). Pristine = 4 × supercell_prod.
        # Endpoints = pristine_count - 1 (Fe removed) + 1 (H added) = pristine_count.
        expected_n_pristine = (args.supercell[0] * args.supercell[1] *
                               args.supercell[2] * 4)
        if len(atoms) != expected_n_pristine:
            raise RuntimeError(
                f"REUSE pristine atom count {len(atoms)} != expected "
                f"{expected_n_pristine}. Either supercell flag differs or XYZ wrong."
            )
        if len(pre_endA) != expected_n_pristine:
            raise RuntimeError(
                f"REUSE endA atom count {len(pre_endA)} != expected "
                f"{expected_n_pristine} (= pristine - 1 Fe + 1 H)."
            )
        if len(pre_endB) != expected_n_pristine:
            raise RuntimeError(
                f"REUSE endB atom count {len(pre_endB)} != expected "
                f"{expected_n_pristine}."
            )
        print(f"  pristine={atoms.get_chemical_formula()} {len(atoms)}at, "
              f"endA={pre_endA.get_chemical_formula()} {len(pre_endA)}at, "
              f"endB={pre_endB.get_chemical_formula()} {len(pre_endB)}at", flush=True)
        result["reuse_relaxed_from"] = str(src)

        triple_path = src / "canonical_triple.json"
        if triple_path.exists():
            with open(triple_path) as f:
                reused_triple = json.load(f)
            if reused_triple.get("vacancy_kind") not in (None, "V_Fe"):
                raise RuntimeError(
                    f"REUSE vacancy_kind mismatch: stored "
                    f"{reused_triple.get('vacancy_kind')!r}, expected 'V_Fe'"
                )
            if reused_triple.get("n_atoms_pristine") not in (None, len(atoms)):
                raise RuntimeError(
                    f"REUSE atom-count mismatch: stored "
                    f"{reused_triple['n_atoms_pristine']}, loaded {len(atoms)}"
                )
            print(f"[REUSE TRIPLE] loaded canonical_triple.json: "
                  f"V_Fe={reused_triple.get('V_Fe_index', reused_triple.get('V_atom_index'))}, "
                  f"S_i={reused_triple['S_i_index']}, "
                  f"S_k={reused_triple['S_k_index']}", flush=True)
        else:
            print(f"[REUSE TRIPLE WARN] no canonical_triple.json in {src}; "
                  f"will re-run picker on relaxed pristine.", flush=True)
    else:
        print("[1/6] Build mackinawite "
              f"{args.supercell[0]}x{args.supercell[1]}x{args.supercell[2]}", flush=True)
        atoms = build_mackinawite(repeat=tuple(args.supercell))

    result["n_atoms_pristine"] = len(atoms)
    result["formula_pristine"] = atoms.get_chemical_formula()
    result["cell_A"] = atoms.cell.lengths().tolist()
    print(f"  {atoms.get_chemical_formula()}, {len(atoms)} atoms, "
          f"cell={[f'{x:.3f}' for x in atoms.cell.lengths()]} Å", flush=True)

    if REUSE:
        print("[2/6] Single-point pristine (REUSE — skip BFGS)", flush=True)
    else:
        print(f"[2/6] Relax pristine (BFGS fmax={args.fmax_pristine})", flush=True)
    t0 = time.time()
    label_p = "sp_pristine" if REUSE else "relax_pristine"
    atoms.calc = make_calc(work_dir, label_p,
                           conv_thr=args.conv_thr_endpoint,
                           **calc_kwargs)
    if REUSE:
        try:
            e_pristine = float(atoms.get_potential_energy())
            fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
            conv = True
            nsteps_p = 0
        except RuntimeError as exc:
            raise RuntimeError(f"pristine single-point failed: {exc}") from exc
    else:
        opt = BFGS(atoms, logfile=str(work_dir / "relax_pristine.log"))
        try:
            conv = bool(opt.run(fmax=args.fmax_pristine, steps=args.max_steps_relax))
            e_pristine = float(atoms.get_potential_energy())
            fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
            nsteps_p = int(opt.nsteps)
        except RuntimeError as exc:
            raise RuntimeError(
                f"pristine relax failed: {exc}. Check {work_dir}/relax_pristine/."
            ) from exc
    result["E_pristine_eV"] = e_pristine
    result["relax_pristine_steps"] = nsteps_p
    result["relax_pristine_converged"] = conv
    result["relax_pristine_fmax"] = fmax
    result["t_relax_pristine_s"] = time.time() - t0
    print(f"  E={e_pristine:.4f} eV, steps={nsteps_p}, fmax={fmax:.4f}, "
          f"conv={conv}, {result['t_relax_pristine_s']:.0f}s", flush=True)
    write(str(work_dir / "relaxed_pristine.xyz"), atoms)

    if args.skip_endpoints:
        print("[SKIP] endpoints + NEB skipped (--skip-endpoints, pristine-only smoke)",
              flush=True)
        result["skip_endpoints"] = True
        result["skip_neb"] = True
        result["t_total_s"] = time.time() - t_start
        return result

    # ---------- pick (or load) V_Fe + S_i + S_k triple ----------
    if REUSE and reused_triple is not None:
        fe_v = int(reused_triple.get("V_Fe_index",
                                     reused_triple.get("V_atom_index")))
        s_i = int(reused_triple["S_i_index"])
        s_k = int(reused_triple["S_k_index"])
        hop_d = float(reused_triple.get("hop_distance_A",
                                        atoms.get_distance(s_i, s_k, mic=True)))
        s_neighbours = reused_triple.get("S_neighbours", [s_i, s_k])
        print(f"[3/6] REUSE V_Fe triple from smoke JSON: "
              f"V_Fe={fe_v}, S_i={s_i}, S_k={s_k}", flush=True)
    else:
        print(f"[3/6] Pick canonical V_Fe + S_i + S_k triple "
              f"(fe_s_max={args.fe_s_max})", flush=True)
        fe_v, s_i, s_k, hop_d, s_neighbours = pick_fe_vacancy_and_two_s_neighbours(
            atoms, fe_s_max=args.fe_s_max, mineral_name="mackinawite"
        )

    d_fe_si = float(atoms.get_distance(fe_v, s_i, mic=True))
    d_fe_sk = float(atoms.get_distance(fe_v, s_k, mic=True))
    print(f"  V_Fe={fe_v}, S_i={s_i} (d_FeS={d_fe_si:.3f}), "
          f"S_k={s_k} (d_FeS={d_fe_sk:.3f}), hop d_si_sk={hop_d:.3f} Å", flush=True)
    result["V_Fe_index"] = int(fe_v)
    result["S_i_index"] = int(s_i)
    result["S_k_index"] = int(s_k)
    result["S_neighbours_all"] = [int(x) for x in s_neighbours]
    result["d_VFe_Si_A"] = d_fe_si
    result["d_VFe_Sk_A"] = d_fe_sk
    result["hop_distance_A"] = float(hop_d)

    # Persist triple metadata (s132 schema: vacancy_kind=V_Fe)
    triple_meta = {
        "vacancy_kind": "V_Fe",
        "V_Fe_index": int(fe_v),
        "S_i_index": int(s_i),
        "S_k_index": int(s_k),
        "S_neighbours": [int(x) for x in s_neighbours],
        "hop_distance_A": float(hop_d),
        "fe_s_max": float(args.fe_s_max),
        "n_atoms_pristine": int(len(atoms)),
        "mineral": "mackinawite",
    }
    with open(work_dir / "canonical_triple.json", "w") as _tf:
        json.dump(triple_meta, _tf, indent=2)

    # ---------- endpoint A ----------
    t0 = time.time()
    if REUSE:
        print("[4/6] Single-point endA (REUSE)", flush=True)
        endA = pre_endA
    else:
        print(f"[4/6] Relax endpoint A (V_Fe={fe_v}, H on S_i={s_i}) BFGS "
              f"fmax={args.fmax_endpoint}", flush=True)
        endA = place_h_on_neighbour_s(atoms, v_idx=fe_v, h_anchor_idx=s_i)
    label_A = "sp_endA" if REUSE else "relax_endA"
    endA.calc = make_calc(work_dir, label_A,
                          conv_thr=args.conv_thr_endpoint,
                          **calc_kwargs)
    if REUSE:
        try:
            e_A = float(endA.get_potential_energy())
            fmaxA = float(np.linalg.norm(endA.get_forces(), axis=1).max())
            convA = True
            nsteps_A = 0
        except RuntimeError as exc:
            raise RuntimeError(f"endA single-point failed: {exc}") from exc
    else:
        optA = BFGS(endA, logfile=str(work_dir / "relax_endA.log"))
        try:
            convA = bool(optA.run(fmax=args.fmax_endpoint, steps=args.max_steps_endpoint))
            e_A = float(endA.get_potential_energy())
            fmaxA = float(np.linalg.norm(endA.get_forces(), axis=1).max())
            nsteps_A = int(optA.nsteps)
        except RuntimeError as exc:
            raise RuntimeError(f"endA relax failed: {exc}") from exc
    sigA = h_bonding_signature(endA, label="endA")
    result["E_endpointA_eV"] = e_A
    result["relax_endA_steps"] = nsteps_A
    result["endA_converged"] = convA
    result["endA_fmax"] = fmaxA
    result["endA_h_signature"] = sigA
    result["t_relax_endA_s"] = time.time() - t0
    print(f"  E_A={e_A:.4f} eV, steps={nsteps_A}, fmax={fmaxA:.4f}, "
          f"d_H_S={sigA['d_H_S']:.3f} Å, d_H_Fe={sigA['d_H_Fe']:.3f} Å, "
          f"kind={sigA['kind']}, conv={convA}, {result['t_relax_endA_s']:.0f}s",
          flush=True)
    write(str(work_dir / "relaxed_endA.xyz"), endA)

    if sigA["artifact_flag"]:
        warn = (
            f"WARN endA artifact: H bonded to Fe (d_H_Fe={sigA['d_H_Fe']:.3f} Å) "
            f"instead of S (d_H_S={sigA['d_H_S']:.3f} Å). "
            f"V_Fe protocol may have collapsed to Fe-bridge basin (mack-style "
            f"V_S+H artifact carryover). Continuing for diagnostic — review final."
        )
        print(f"  {warn}", flush=True)
        result["endA_artifact_warning"] = warn

    # ---------- endpoint B ----------
    t0 = time.time()
    if REUSE:
        print("[5/6] Single-point endB (REUSE)", flush=True)
        endB = pre_endB
    else:
        print(f"[5/6] Relax endpoint B (V_Fe={fe_v}, H on S_k={s_k}) BFGS "
              f"fmax={args.fmax_endpoint}", flush=True)
        endB = place_h_on_neighbour_s(atoms, v_idx=fe_v, h_anchor_idx=s_k)
    label_B = "sp_endB" if REUSE else "relax_endB"
    endB.calc = make_calc(work_dir, label_B,
                          conv_thr=args.conv_thr_endpoint,
                          **calc_kwargs)
    if REUSE:
        try:
            e_B = float(endB.get_potential_energy())
            fmaxB = float(np.linalg.norm(endB.get_forces(), axis=1).max())
            convB = True
            nsteps_B = 0
        except RuntimeError as exc:
            raise RuntimeError(f"endB single-point failed: {exc}") from exc
    else:
        optB = BFGS(endB, logfile=str(work_dir / "relax_endB.log"))
        try:
            convB = bool(optB.run(fmax=args.fmax_endpoint, steps=args.max_steps_endpoint))
            e_B = float(endB.get_potential_energy())
            fmaxB = float(np.linalg.norm(endB.get_forces(), axis=1).max())
            nsteps_B = int(optB.nsteps)
        except RuntimeError as exc:
            raise RuntimeError(f"endB relax failed: {exc}") from exc
    sigB = h_bonding_signature(endB, label="endB")
    result["E_endpointB_eV"] = e_B
    result["relax_endB_steps"] = nsteps_B
    result["endB_converged"] = convB
    result["endB_fmax"] = fmaxB
    result["endB_h_signature"] = sigB
    result["t_relax_endB_s"] = time.time() - t0
    print(f"  E_B={e_B:.4f} eV, steps={nsteps_B}, fmax={fmaxB:.4f}, "
          f"d_H_S={sigB['d_H_S']:.3f} Å, d_H_Fe={sigB['d_H_Fe']:.3f} Å, "
          f"kind={sigB['kind']}, conv={convB}, {result['t_relax_endB_s']:.0f}s",
          flush=True)
    write(str(work_dir / "relaxed_endB.xyz"), endB)

    if sigB["artifact_flag"]:
        warn = (
            f"WARN endB artifact: H bonded to Fe (d_H_Fe={sigB['d_H_Fe']:.3f} Å). "
            f"V_Fe protocol may have collapsed."
        )
        print(f"  {warn}", flush=True)
        result["endB_artifact_warning"] = warn

    if len(endA) != len(endB):
        raise RuntimeError(f"endA/endB atom-count mismatch: {len(endA)} vs {len(endB)}")

    dE_endpoints = abs(e_A - e_B)
    result["dE_endpoints_eV"] = dE_endpoints
    print(f"  |E_A-E_B|={dE_endpoints:.4f} eV", flush=True)

    # SCF noise floor check (s130 deadly mistake): if dE_endpoints < 5× SCF noise,
    # endpoints are bit-equivalent within DFT precision → likely same basin.
    scf_noise_eV = 5 * 1.36e-6  # 1e-7 Ry conv_thr → 1.36e-6 eV; 5× safety
    if dE_endpoints < scf_noise_eV and not (sigA["artifact_flag"] and sigB["artifact_flag"]):
        warn = (
            f"WARN dE_endpoints={dE_endpoints:.2e} eV < 5× SCF noise floor "
            f"({scf_noise_eV:.2e} eV) — possible same-basin signature even if "
            f"H bonded to S (e.g., symmetric S_i ↔ S_k pair)."
        )
        print(f"  {warn}", flush=True)
        result["dE_endpoints_warning"] = warn

    if dE_endpoints > 0.20:
        warn = (f"WARN endpoints asymmetric: |dE|={dE_endpoints:.4f} eV > 0.20 — "
                f"V_Fe pocket S sites may be inequivalent. Acceptable for asymmetric "
                f"hop, but flag for paper.")
        print(f"  {warn}", flush=True)
        result["endpoints_symmetric"] = False
        result["endpoints_warning"] = warn
    else:
        result["endpoints_symmetric"] = True

    if args.skip_neb:
        print("[SKIP] NEB skipped (--skip-neb, endpoints-only smoke)", flush=True)
        result["skip_neb"] = True
        result["t_total_s"] = time.time() - t_start
        return result

    # ---------- Test A diagnostic (s132 mandatory pre-NEB gate) ----------
    h_idx_A = len(endA) - 1
    h_idx_B = len(endB) - 1
    disp_naive_H = endB.positions[h_idx_B] - endA.positions[h_idx_A]
    disp_mic_H, _ = find_mic(disp_naive_H[None, :], endA.cell, endA.pbc)
    h_disp = float(np.linalg.norm(disp_mic_H[0]))

    nearest_A = sigA["nearest_S_idx"] if not sigA["artifact_flag"] else sigA["nearest_Fe_idx"]
    nearest_B = sigB["nearest_S_idx"] if not sigB["artifact_flag"] else sigB["nearest_Fe_idx"]
    same_nearest = (nearest_A == nearest_B and nearest_A >= 0)

    # PATCH-1 (physicist QA, s132): for V_Fe pocket the MAX-d S-S pair is
    # mirror-equivalent through V_Fe plane (C2v point symmetry after Fe
    # removal). Symmetric S_i ↔ S_k → dE_endpoints ~ 0 by physics, NOT
    # artifact. Therefore dE_below_scf_noise is WARNING-ONLY, not abort.
    #
    # Q-115 ERRATA #6 (s132 day 2 evening, after W3 abort + harvest re-analysis):
    # h_disp threshold 1.0 was too strict for in-pocket mirror endpoints. Real
    # mack V_Fe in-layer endA/endB are mirror images через V_Fe pocket center,
    # both at canonical d_H_S=1.432 Å + d_to_V_Fe_pos=1.222 Å. Geometrically
    # h_disp = 2 × Δy_VFe ≈ 0.5-0.7 Å (NOT 3.674 hop distance).
    # PRIMARY criterion: same_nearest_host == False (different S anchors).
    # SECONDARY: h_disp ≥ 0.5 (relaxed from 1.0). Both must pass.
    # Reference: knowledge/MACK_VFE_NEB_REINTERP_2026-05-03.md, post-W3 harvest.
    H_DISP_MIN = 0.5  # was 1.0; mirror endpoints have small geometric h_disp
    test_a = {
        "h_displacement_A": h_disp,
        "h_displacement_min_A": H_DISP_MIN,
        "h_displacement_too_small": bool(h_disp < H_DISP_MIN),
        "nearest_endA": nearest_A,
        "nearest_endB": nearest_B,
        "same_nearest_host": bool(same_nearest),
        "dE_endpoints_eV": float(dE_endpoints),
        "dE_below_scf_noise": bool(dE_endpoints < scf_noise_eV),
    }
    # Strict gates: only structural same-basin signatures abort. dE noise → warn.
    test_a_pass = (not test_a["h_displacement_too_small"]
                   and not test_a["same_nearest_host"])
    test_a["pass"] = test_a_pass
    if test_a["dE_below_scf_noise"]:
        test_a["dE_warning"] = (
            "dE_endpoints below SCF noise floor. Likely C2v mirror-equivalent S "
            "pair (expected for V_Fe pocket, symmetric hop). "
            f"NOT a same-basin signature when h_disp ≥ {H_DISP_MIN} Å AND distinct nearest hosts."
        )
        print(f"[TEST A] dE_endpoints={dE_endpoints:.2e} eV < SCF noise floor "
              f"({scf_noise_eV:.2e}) — accepted (likely symmetric pair).",
              flush=True)
    result["test_a"] = test_a
    print(f"[TEST A] h_disp={h_disp:.3f} Å (≥{H_DISP_MIN}?), same_nearest={same_nearest} "
          f"→ {'PASS' if test_a_pass else 'FAIL'}", flush=True)
    if not test_a_pass:
        warn = (
            f"WARN Test A FAIL: endpoints likely in same basin "
            f"(h_disp={h_disp:.3f}<{H_DISP_MIN} OR same_nearest_host={same_nearest}). "
            f"NEB result would be artifact. Aborting."
        )
        print(f"  {warn}", flush=True)
        raise RuntimeError(warn)

    # ---------- IDPP prewrap ----------
    if args.idpp_prewrap:
        endB_unwrapped, n_unwrapped, prewrap_summary = prewrap_endpoint_for_idpp(
            endA, endB, label="endB"
        )
        result["idpp_prewrap_n_unwrapped"] = int(n_unwrapped)
        result["idpp_prewrap_summary"] = prewrap_summary
        if n_unwrapped > 0:
            endB_unwrapped.calc = endB.calc
            endB = endB_unwrapped
            try:
                e_B_pp = float(endB.get_potential_energy())
                fmaxB_pp = float(np.linalg.norm(endB.get_forces(), axis=1).max())
            except RuntimeError as exc:
                raise RuntimeError(
                    f"endB SCF post-prewrap failed: {exc}."
                ) from exc
            print(f"[POST-PREWRAP] endB recomputed: E_B={e_B_pp:.4f} eV "
                  f"(was {e_B:.4f}), fmax={fmaxB_pp:.4f}", flush=True)
            result["E_endpointB_eV_pre_prewrap"] = e_B
            result["E_endpointB_eV"] = e_B_pp
            result["endB_fmax_pre_prewrap"] = fmaxB
            result["endB_fmax"] = fmaxB_pp
            e_B = e_B_pp
            fmaxB = fmaxB_pp
            dE_endpoints = abs(e_A - e_B)
            result["dE_endpoints_eV"] = dE_endpoints
            print(f"[POST-PREWRAP] |E_A-E_B|={dE_endpoints:.4f} eV", flush=True)
    else:
        result["idpp_prewrap_n_unwrapped"] = 0
        result["idpp_prewrap_summary"] = {"n_unwrapped": 0, "disabled": True}
        print("[IDPP-PREWRAP] disabled via --no-idpp-prewrap (legacy)", flush=True)

    # ---------- CI-NEB ----------
    t0 = time.time()
    print(f"[6/6] CI-NEB ({args.n_images} images, IDPP, FIRE, "
          f"fmax={args.fmax_neb}, k={args.k_spring})", flush=True)
    n_intermediate = args.n_images - 2
    images = [endA]
    for i in range(n_intermediate):
        img = endA.copy()
        img.calc = make_calc(work_dir, f"image_{i+1:02d}",
                             conv_thr=args.conv_thr_neb,
                             **neb_calc_kwargs)
        images.append(img)
    images.append(endB)

    neb, neb_policy = make_neb_band(images, args)
    result.update(neb_policy)
    try:
        neb.interpolate("idpp")
    except Exception as e:
        print(f"  IDPP failed ({e}), fallback linear", flush=True)
        neb.interpolate()

    # Q-115 ERRATA #6 part B: asymmetric IDPP nudge for mirror-symmetric
    # endpoints (chemist + physicist HIGH s132 evening). Mirror configurations
    # risk "degenerate band failure" — IDPP through y_VFe-plane gives all
    # images at same E. Nudge intermediate images to break exact mirror,
    # allowing NEB to find one branch of saddle path cleanly.
    #
    # Physicist refinement: original (x, z) nudge was invariant под y-mirror
    # reflection (mirror plane = y=y_VFe). Added y-component so nudge truly
    # breaks mirror symmetry, not just quasi-degeneracy.
    #   image mid:    +0.20 Å z + 0.05 Å y (out-of-plane + mirror break)
    #   images mid±1: +0.10 Å x (in-plane support, biases path direction)
    if args.asymmetric_idpp_nudge and n_intermediate >= 5:
        h_idx = len(images[0]) - 1  # H atom is last
        mid = 1 + n_intermediate // 2  # absolute image index (skip endA at 0)
        if 1 <= mid <= len(images) - 2:
            images[mid].positions[h_idx, 1] += 0.05  # y-nudge — true mirror break
            images[mid].positions[h_idx, 2] += 0.20  # z-nudge — out of S layer
            print(f"[IDPP-NUDGE] image {mid}: +0.05 Å y + 0.20 Å z "
                  f"(break y-mirror + out-of-plane)", flush=True)
            for off in (-1, +1):
                idx = mid + off
                if 1 <= idx <= len(images) - 2:
                    images[idx].positions[h_idx, 0] += 0.10  # x-nudge neighbours
                    print(f"[IDPP-NUDGE] image {idx}: +0.10 Å in x", flush=True)
        result["asymmetric_idpp_nudged"] = True
    else:
        result["asymmetric_idpp_nudged"] = False
        if not args.asymmetric_idpp_nudge:
            print("[IDPP-NUDGE] disabled via --no-asymmetric-idpp-nudge", flush=True)
        else:
            print(f"[IDPP-NUDGE] skipped: only {n_intermediate} intermediate "
                  f"images (need ≥5)", flush=True)

    min_interatomic = float("inf")
    for img in images:
        dmat = img.get_all_distances(mic=True)
        np.fill_diagonal(dmat, np.inf)
        mi = float(dmat.min())
        if mi < min_interatomic:
            min_interatomic = mi
    result["min_interatomic_post_idpp_A"] = min_interatomic
    if min_interatomic < 0.8:
        raise RuntimeError(
            f"IDPP overlapping atoms: min_d={min_interatomic:.3f} Å < 0.8")
    print(f"  min interatomic dist (post-IDPP) = {min_interatomic:.3f} Å", flush=True)

    print("[INITIAL PATH SANITY] computing E_a from IDPP-interpolated images...",
          flush=True)
    initial_energies = [e_A]
    for i, img in enumerate(images[1:-1]):
        try:
            initial_energies.append(float(img.get_potential_energy()))
            print(f"  image {i+1:02d}: E={initial_energies[-1]:.4f} eV", flush=True)
        except RuntimeError as exc:
            raise RuntimeError(f"initial SCF on image {i+1} failed: {exc}") from exc
    initial_energies.append(e_B)
    e_a_initial = max(initial_energies) - e_A
    result["E_a_initial_post_idpp_eV"] = float(e_a_initial)
    print(f"[INITIAL PATH SANITY] E_a (post-IDPP, pre-FIRE) = "
          f"{e_a_initial:.3f} eV", flush=True)
    if e_a_initial > 5.0:
        raise RuntimeError(
            f"INITIAL PATH SANITY FAIL: E_a (post-IDPP) = {e_a_initial:.3f} eV "
            f">5 eV. Likely broken initial path. Aborting FIRE."
        )
    print(f"  → SANE (< 5 eV), proceeding to FIRE", flush=True)

    opt_neb = FIRE(neb, logfile=str(work_dir / "neb.log"),
                   trajectory=str(work_dir / "neb.traj"))
    try:
        neb_conv = bool(opt_neb.run(fmax=args.fmax_neb, steps=args.max_steps_neb))
        energies = [float(img.get_potential_energy()) for img in images]
    except RuntimeError as exc:
        raise RuntimeError(
            f"NEB run/harvest failed after {opt_neb.nsteps} steps: {exc}") from exc

    e_ref = energies[0]
    rel = [e - e_ref for e in energies]
    e_a = max(rel)
    e_rxn = rel[-1]

    max_idx = int(np.argmax(energies))
    # CS MEDIUM patch: warn if saddle on endpoint (no real barrier — degenerate
    # path или endpoint is itself maximum)
    if max_idx in (0, len(images) - 1):
        result["saddle_on_endpoint_warning"] = True
        print(f"[SADDLE WARN] argmax at endpoint (idx={max_idx}) — no real barrier, "
              f"likely degenerate path or convergence issue", flush=True)
    try:
        fci = images[max_idx].get_forces()
        fmax_ci = float(np.linalg.norm(fci, axis=1).max())
    except Exception:
        fmax_ci = float("nan")

    # Q-115 ERRATA #6 part C (physicist s132 evening): log saddle H position +
    # nearest Fe identity для cross-run reproducibility. If two independent
    # runs pick different mirror branches due BLAS thread numerics, E_a should
    # match ±5 meV but y-sign of saddle H may flip.
    try:
        saddle_h_idx = len(images[max_idx]) - 1
        saddle_h_pos = images[max_idx].positions[saddle_h_idx].tolist()
        # Find nearest Fe in saddle image
        saddle_syms = images[max_idx].get_chemical_symbols()
        saddle_fe_indices = [i for i, s in enumerate(saddle_syms) if s == "Fe"]
        saddle_s_indices = [i for i, s in enumerate(saddle_syms) if s == "S"]
        d_h_fe_saddle = sorted(
            (float(images[max_idx].get_distance(saddle_h_idx, i, mic=True)), i)
            for i in saddle_fe_indices)[:1]
        d_h_s_saddle = sorted(
            (float(images[max_idx].get_distance(saddle_h_idx, i, mic=True)), i)
            for i in saddle_s_indices)[:2]
        result["saddle_h_position_A"] = saddle_h_pos
        result["saddle_d_H_Fe_nearest_A"] = d_h_fe_saddle[0][0] if d_h_fe_saddle else None
        result["saddle_Fe_nearest_idx"] = d_h_fe_saddle[0][1] if d_h_fe_saddle else -1
        result["saddle_d_H_S_nearest_A"] = d_h_s_saddle[0][0] if d_h_s_saddle else None
        result["saddle_d_H_S_2nd_A"] = d_h_s_saddle[1][0] if len(d_h_s_saddle) >= 2 else None
        result["saddle_image_idx"] = max_idx
        print(f"[SADDLE] image {max_idx}: H pos={[f'{x:.3f}' for x in saddle_h_pos]}, "
              f"d_H_S_nearest={d_h_s_saddle[0][0]:.3f} Å, "
              f"d_H_S_2nd={d_h_s_saddle[1][0]:.3f} Å, "
              f"d_H_Fe_nearest={d_h_fe_saddle[0][0]:.3f} Å (Fe idx {d_h_fe_saddle[0][1]})",
              flush=True)
    except Exception as exc:
        print(f"[SADDLE] logging failed (non-blocking): {exc}", flush=True)

    # NEB step floor (s132 universal gate): ≥20 FIRE iterations expected for real
    # saddle. 0 iter = endpoints in same basin (mack V_S artifact signature).
    min_neb_steps_ok = int(opt_neb.nsteps) >= int(args.min_neb_steps)

    result["neb_k_spring"] = float(args.k_spring)
    result.update(neb_policy)
    result["neb_converged"] = neb_conv
    result["neb_steps"] = int(opt_neb.nsteps)
    result["min_neb_steps"] = int(args.min_neb_steps)
    result["min_neb_steps_ok"] = bool(min_neb_steps_ok)
    result["neb_final_fmax_CI"] = fmax_ci
    result["neb_energies_rel_eV"] = rel
    result["E_a_eV"] = float(e_a)
    # PATCH-1: dE_below_scf_noise no longer disqualifies paper_quotable
    # (symmetric C2v pair expected for V_Fe). Paper quotability needs
    # NEB convergence + min steps + non-trivial path.
    paper_ok = bool(neb_conv and min_neb_steps_ok)
    result["E_a_paper_quotable"] = float(e_a) if paper_ok else None
    result["E_rxn_eV"] = float(e_rxn)
    result["t_neb_s"] = time.time() - t0

    # PATCH-2 (physicist QA, s132): intermediate well detection. Real V_Fe
    # lateral hop = single barrier (Liu 2021). If middle of NEB path lies
    # below endpoints → either physical intermediate (rare) OR Fe-collapse
    # mid-path (artifact carryover from V_S protocol bug).
    if len(rel) >= 5:
        middle_rel = rel[1:-1]
        min_middle = float(min(middle_rel))
        endpoint_floor = min(rel[0], rel[-1])
        if min_middle < endpoint_floor - 0.05:
            depth = float(endpoint_floor - min_middle)
            result["intermediate_well_min_eV"] = min_middle
            result["intermediate_well_depth_eV"] = depth
            result["intermediate_well_warning"] = (
                f"NEB path has intermediate min {min_middle:.3f} eV "
                f"({depth:.3f} eV below endpoint floor {endpoint_floor:.3f} eV). "
                f"Either physical intermediate state or Fe-collapse mid-path. "
                f"Inspect final_NN.xyz visually."
            )
            print(f"[INTERMEDIATE WELL] depth={depth:.3f} eV — flagged for review",
                  flush=True)
        else:
            result["intermediate_well_min_eV"] = min_middle
            result["intermediate_well_depth_eV"] = 0.0

    print(f"  E_a={e_a:.4f} eV (paper_quotable={paper_ok}), "
          f"E_rxn={e_rxn:.4f} eV, steps={opt_neb.nsteps}, conv={neb_conv}, "
          f"min_steps_ok={min_neb_steps_ok}, fmax_CI={fmax_ci:.4f}, "
          f"{result['t_neb_s']:.0f}s", flush=True)

    for k, img in enumerate(images):
        write(str(work_dir / f"final_{k:02d}.xyz"), img)

    result["cross_references"] = {
        "Liu_2021_VFe_surface_E_a_eV": 0.26,
        "MACE_VFe_MLIP_scan_s132": (
            "3 distinct S-H basins, 54% S-H, 24% Fe collapse"
        ),
        "V_S_artifact_E_a_eV": 8e-7,  # mack V_S+H artifact reference
    }
    result["NOTE"] = (
        "Canonical V_Fe + H lateral S-S hop (s132 pivot from broken V_S+H protocol). "
        "nspin=1 PM/itinerant primary per Q-115 RESHENIE-079. MLIP topology validated "
        "via 80-candidate scan (paper/SI/SI_vfe_mlip_scan.md). Literature anchor: "
        "Liu 2021 ACS Omega L-650 V_Fe surface 0.26 eV."
    )

    try:
        png = output_dir / "neb_canonical_mack_72at_qe_VFe.png"
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.linspace(0, 1, len(rel))
        ax.plot(x, rel, "go-", linewidth=2, markersize=7)
        ax.set_xlabel("Reaction coordinate")
        ax.set_ylabel("Energy (eV)")
        ax.set_title(f"Mackinawite FeS V_Fe + H lateral S-S hop (QE PWSCF, s132)\n"
                     f"E_a={e_a:.3f} eV, n_atoms={len(endA)}, conv={neb_conv}")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(png, dpi=130)
        plt.close(fig)
        result["png_path"] = str(png)
        print(f"  saved {png}", flush=True)
    except Exception as e:
        print(f"  PNG save failed: {e}", flush=True)

    result["t_total_s"] = time.time() - t_start
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir",   default="/workspace/neb_canonical_mack_72at_qe_VFe")
    parser.add_argument("--output-dir", default="/workspace/results")
    parser.add_argument("--supercell", type=int, nargs=3, default=[3, 3, 2],
                        help="mack supercell (default 3x3x2 = 72 atoms pristine)")
    parser.add_argument("--kpts", type=int, nargs=3, default=[2, 2, 2])
    parser.add_argument("--n-images", type=int, default=9)
    parser.add_argument("--fmax-pristine", type=float, default=0.03)
    parser.add_argument("--fmax-endpoint", type=float, default=0.03)
    parser.add_argument("--fmax-neb", type=float, default=0.05)
    parser.add_argument("--max-steps-relax",    type=int, default=200)
    parser.add_argument("--max-steps-endpoint", type=int, default=500)
    parser.add_argument("--max-steps-neb",      type=int, default=500)
    parser.add_argument("--min-neb-steps",      type=int, default=20,
                        help="Minimum FIRE iterations for paper-quotable NEB. "
                             "Below this → likely same-basin artifact (V_S signature).")
    parser.add_argument("--k-spring", type=float, default=0.3)
    parser.add_argument("--dyneb", dest="dyneb", action="store_true", default=True,
                        help="Use ASE DyNEB dynamic relaxation for selective image updates (default)")
    parser.add_argument("--no-dyneb", dest="dyneb", action="store_false",
                        help="Use standard ASE NEB; keep for A/B baseline comparisons")
    parser.add_argument("--dyneb-scale-fmax", type=float, default=0.0,
                        help="ASE DyNEB scaled convergence factor. Default 0.0 keeps the same "
                             "fmax threshold for all images; tune only after a reference run.")
    # V_Fe picker
    parser.add_argument("--fe-s-max", type=float, default=2.7,
                        help="Max Fe-S distance to count S as V_Fe neighbour. "
                             "Default 2.7 Å (mack Fe-S nominal 2.25, 2nd shell ~3.5).")
    # QE
    parser.add_argument("--ecutwfc", type=float, default=60.0)
    parser.add_argument("--ecutrho", type=float, default=240.0)
    parser.add_argument("--mixing-beta", type=float, default=0.2)
    parser.add_argument("--mixing-mode", default="plain",
                        choices=["plain", "TF", "local-TF"])
    parser.add_argument("--smearing", default="gaussian",
                        choices=["gaussian", "mp", "mv", "fd"])
    parser.add_argument("--degauss", type=float, default=0.01)
    parser.add_argument("--conv-thr-endpoint", type=float, default=1.0e-8)
    parser.add_argument("--conv-thr-neb", type=float, default=1.0e-7)
    parser.add_argument("--pseudo-dir", default=PSEUDO_DIR)
    parser.add_argument("--pp-fe", default="Fe.upf")
    parser.add_argument("--pp-s", default="S.upf")
    parser.add_argument("--pp-h", default="H.upf")
    default_omp = int(os.environ.get("OMP_NUM_THREADS", 8))
    parser.add_argument("--omp",    type=int, default=default_omp)
    parser.add_argument("--mpi-np", type=int, default=1)
    parser.add_argument("--skip-endpoints", action="store_true")
    parser.add_argument("--skip-neb", action="store_true")
    parser.add_argument("--disk-io-neb", default="high",
                        choices=["low", "medium", "high"])
    parser.add_argument("--wfc-reuse", action="store_true")
    parser.add_argument("--reuse-relaxed", default=None)
    parser.add_argument("--idpp-prewrap", action="store_true", default=True,
                        help="Pre-wrap endB relative to endA via find_mic before NEB. "
                             "Default ON (paper-grade fix for ASE issue #1130).")
    parser.add_argument("--no-idpp-prewrap", dest="idpp_prewrap",
                        action="store_false")
    # Q-115 ERRATA #6 part B (s132 day 2 evening): asymmetric IDPP nudge for
    # mirror-symmetric V_Fe endpoints, prevents "degenerate band failure" в
    # CI-NEB. Default ON for V_Fe canonical с reuse_relaxed.
    parser.add_argument("--asymmetric-idpp-nudge", action="store_true",
                        default=True,
                        help="Nudge middle NEB image +0.2 Å z, neighbours ±0.1 Å x "
                             "to break mirror symmetry (avoid degenerate band failure). "
                             "Default ON for V_Fe canonical.")
    parser.add_argument("--no-asymmetric-idpp-nudge", dest="asymmetric_idpp_nudge",
                        action="store_false")
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"]      = str(args.omp)
    os.environ["MKL_NUM_THREADS"]      = str(args.omp)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.omp)
    os.environ.setdefault("OMP_STACKSIZE", "256M")

    # PAW pseudo detection (physicist C3 s126)
    missing_pp = []
    paw_pp = []
    for sym, fn in (("Fe", args.pp_fe), ("S", args.pp_s), ("H", args.pp_h)):
        full = Path(args.pseudo_dir) / fn
        if not full.exists():
            missing_pp.append(f"{sym}: {full}")
            continue
        try:
            head = full.read_text(errors="ignore")[:4000]
        except Exception:
            continue
        if ('pseudo_type="PAW"' in head or
            'pseudo_type=\'PAW\'' in head or
            'is_paw="T"' in head or
            'is_paw=\'T\'' in head):
            paw_pp.append(f"{sym}: {full}")
    if missing_pp:
        print("[FATAL] Missing pseudopotential files:", flush=True)
        for m in missing_pp:
            print(f"  {m}", flush=True)
        sys.exit(3)
    if paw_pp:
        print("[FATAL] PAW pseudo detected — ecutrho=240 (4×) inadequate "
              "(need 480 8× for PAW). Use ONCV pseudo or bump ecutrho.", flush=True)
        for m in paw_pp:
            print(f"  {m}", flush=True)
        sys.exit(4)

    acquire_singleton()
    try:
        try:
            result = run_mackinawite_VFe(args)
            result["status"] = "success"
        except KeyboardInterrupt:
            print("[INTERRUPT] killed by user", flush=True)
            raise
        except Exception as exc:
            print(f"[FAIL] {exc}", flush=True)
            traceback.print_exc()
            result = {
                "status": "fail",
                "error":  str(exc),
                "traceback": traceback.format_exc(),
                "mineral": "mackinawite",
                "vacancy_kind": "V_Fe",
            }

        out = Path(args.output_dir) / "neb_canonical_mack_72at_qe_VFe.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2, cls=NumpyEncoder)
        print(f"[saved] {out}", flush=True)
    finally:
        release_singleton()


if __name__ == "__main__":
    main()
