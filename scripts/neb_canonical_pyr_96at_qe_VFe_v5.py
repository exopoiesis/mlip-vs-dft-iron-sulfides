#!/usr/bin/env python3
"""
Canonical V_Fe (octahedral 4a) + H lateral hop NEB -- pyrite FeS2 (Pa-3, #205),
conventional cell * 2x2x2 = 96 atoms pristine (Fe32 S64); after V_Fe + H = 96 atoms
(Fe31 S64 H1).

Adapted from neighbouring templates per Q-115 + РЕШЕНИЕ-082 (chemistry-signature):
  - structure builder + ONCV PP paths + 2x2x2 cell from neb_canonical_pyr_96at_qe.py
    (V_S2 paper anchor, paper-grade 94.6 meV s129)
  - V_Fe defect logic (canonical_triple JSON save/load, S_i + S_k anchors,
    endA/endB constructor with H placed at S anchor, H1/H2/H3 endpoint geometry
    classification, Test A diagnostic, BFGS trajectory= mandatory) from
    neb_canonical_marc_96at_qe.py
  - cubic V_Fe + H reference (octahedral picker, mirror endpoints expectation
    dE ~= 0, REUSE_PRISTINE_ONLY mode, sp_tmp cleanup before NEB, pre-flight
    gates G1/G2/G3) from neb_canonical_greigite_56at_qe_VFe.py

Pyrite specifics (KEEP, do not simplify):
  - nspin=1 baseline (diamagnetic Fe2+ LS d6; MP mp-226, Brik 2021, Macke-Timrov 2024)
  - NO Hubbard U (pyrite consensus; Tier 1 +U=2 sensitivity in SI optional)
  - PBE plain, ecutwfc=60 Ry, ecutrho=480 Ry (ONCV 8x for paper-grade
    rho-precision; same as greigite +U setup, more conservative than 4x default)
  - gaussian smearing degauss=0.01 Ry (Q-115 errata; wider than 0.005 for
    V_Fe-introduced holes near Fermi level)
  - mixing_mode='plain' beta=0.3 (homogeneous diamagnetic; не 0.05 как greigite)
  - 2x2x2 k-mesh
  - 9-image CI-NEB, FIRE optimizer, IDPP-prewrap (ASE #1130 fix mandatory)
  - fmax_endpoint=0.03, fmax_neb=0.05 (РЕШЕНИЕ-079 paper-grade)

Reaction coordinate:
  1. Build pyrite Pa-3 conv 2x2x2 (96 atoms).
  2. Pick central Fe(4a) atom (V_Fe = vacancy_atom) -- Wyckoff 4a, octahedral 6-S.
  3. Find 6 S anchors within fe_s_max ~2.85 A (Fe-S nominal ~2.27 A).
  4. Pick 2 S atoms (S_i, S_k) as adjacent edge pair (S-S edge d ~3.20-3.85 A).
  5. Build endA: H placed 1.35 A from S_i toward V_Fe (S-H в pocket).
  6. Build endB: H placed 1.35 A from S_k toward V_Fe.
  7. PRE-FLIGHT GATES (s148, mandatory): G1 Wyckoff, G2 parity, G3 pocket-radius.
  8. Relax pristine (REUSE skip if --reuse-pristine) + endA + endB BFGS.
  9. CI-NEB FIRE 9 images.

Expected E_a: 150-400 meV (cross-mineral scaling, см. CROSS_MINERAL_VFE_BARRIER_PATTERN.md).
Endpoint dE_predicted: ~0 (Pa-3 cubic mirror symmetry).

REFERENCE DOCS:
  - knowledge/PYR_VFE_NOMAD_REFERENCE_2026-05-28.md
  - knowledge/PYR_VFE_EXPERIMENT_PLAN.md
  - knowledge/Q115_DFT_PROTOCOL_FINAL_2026-04-28.md
  - knowledge/CROSS_MINERAL_VFE_BARRIER_PATTERN.md

LITERATURE GAP: V_Fe + S-H migration barrier in cubic FeS2 -- NOT FOUND in
public DFT literature 2014-2026. This script produces the first paper-grade
quantitative anchor.

Per РЕШЕНИЕ-079 (s125) + РЕШЕНИЕ-082 (s148 chemistry-signature scope):
  - QE PWSCF (no Pulay forces, paper-grade fmax achievable)
  - V_Fe pivot (V_S+H deprecated for thiospinels/all-equivalent-S sulfides)
  - canonical_triple.json save/load (s129: avoid silent index flip on REUSE)
  - BFGS endpoints + trajectory= mandatory (s134)
  - Test A diagnostic (s130/132: not-same-basin gate)
  - Disk check at start (s153: >=100 GB)
  - IDPP-prewrap (s148: ASE issue #1130 fix)
  - Em-dash -> -- in strings (s50: ASCII container UnicodeError)

References:
  - infra/gpu_scripts/neb_canonical_pyr_96at_qe.py (V_S2 paper anchor parent)
  - infra/gpu_scripts/neb_canonical_marc_96at_qe.py (V_Fe defect logic parent)
  - infra/gpu_scripts/neb_canonical_greigite_56at_qe_VFe.py (cubic V_Fe + H ref)
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from ase import Atom
from ase.spacegroup import crystal
from ase.optimize import BFGS, LBFGS, LBFGSLineSearch, FIRE
from ase.mep import DyNEB, NEB
try:
    from ase.mep import NEBOptimizer
    _HAS_NEB_OPTIMIZER = True
except ImportError:
    _HAS_NEB_OPTIMIZER = False
from ase.io import read, write
from ase.geometry import find_mic
from ase.calculators.espresso import Espresso, EspressoProfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MINERAL_TAG = "pyrite_96at_qe_VFe"
LOCK_FILE = Path("/workspace/.neb_canonical_pyr_qe_VFe.lock")

PW_BIN = os.environ.get("PW_BIN", "pw.x")
# ONCV-SR PBE (Pseudo Dojo nc-sr-04 standard) per Q115 protocol section 1.
# Cross-code validated for pyrite (GPAW 0.181 vs QE 0.190 vs ABACUS 0.187,
# SI_I_pyr_1x1x1_crosscode_2026-05-16.md).
PSEUDO_DIR = os.environ.get("ESPRESSO_PSEUDO",
                            os.environ.get("PSEUDO_DIR", "/opt/pp/oncv_pbe"))


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy scalars / arrays."""
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


# ============================================================
# Disk check (s153 lesson: >=100 GB mandatory for 9-image NEB)
# ============================================================

def check_disk_space(work_dir: Path, min_gb: int = 100) -> dict:
    """Check available disk space in workspace; warn if below min_gb.

    Returns dict {available_gb, total_gb, used_gb, ok_for_neb}.
    s153 precedent: 50 GB instance crashed на image_02 exit 127, $9 + 12 hr lost.
    Recommended: >= 150 GB для 9-image NEB w/ disk_io='high' (~80-100 GB peak).
    """
    try:
        stat = shutil.disk_usage(str(work_dir))
        avail_gb = stat.free / (1024 ** 3)
        total_gb = stat.total / (1024 ** 3)
        used_gb = stat.used / (1024 ** 3)
        ok = avail_gb >= min_gb
        print(f"[DISK CHECK] {work_dir}: avail={avail_gb:.1f} GB, "
              f"total={total_gb:.1f} GB, used={used_gb:.1f} GB",
              flush=True)
        if not ok:
            print(f"[DISK WARN] available {avail_gb:.1f} GB < min {min_gb} GB. "
                  f"9-image NEB requires >=100 GB (rec 150 GB) per s153 lesson. "
                  f"Pre-flight: increase disk allocation or risk crash mid-NEB.",
                  flush=True)
        return {
            "available_gb": float(avail_gb),
            "total_gb": float(total_gb),
            "used_gb": float(used_gb),
            "ok_for_neb": bool(ok),
            "min_required_gb": int(min_gb),
        }
    except Exception as e:
        print(f"[DISK CHECK WARN] {e}", flush=True)
        return {"error": str(e), "ok_for_neb": False}


# ============================================================
# NEB band factory
# ============================================================

def make_neb_band(images, args):
    """Create production NEB band using the selected optimizer policy.

    Returns (neb_band, policy_dict). DyNEB default (selective image updates;
    A/B baseline comparison available via --no-dyneb).
    """
    if args.dyneb:
        neb = DyNEB(
            images,
            climb=args.climb,
            method="improvedtangent",
            k=args.k_spring,
            fmax=args.fmax_neb,
            dynamic_relaxation=True,
            scale_fmax=args.dyneb_scale_fmax,
        )
        policy = "DyNEB"
    else:
        neb = NEB(images, climb=args.climb, method="improvedtangent", k=args.k_spring)
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


# ============================================================
# Singleton guard (rule #31: ps -C filter, not pgrep -f self-match bug)
# ============================================================

def acquire_singleton():
    """python3-only singleton (s130 W2 v1 kill failure precedent: pgrep -f self-match).

    Uses ps -C python3 + args filter on script filename. Refuses second instance.
    """
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
            if "neb_canonical_pyr_96at_qe_VFe.py" not in args_str:
                continue
            try:
                pid = int(pid_str)
            except ValueError:
                continue
            if pid == my_pid:
                continue
            print(f"[SINGLETON] another instance pid={pid}: {args_str}",
                  flush=True)
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
    """Best-effort lock cleanup on shutdown."""
    try:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
    except Exception:
        pass


# ============================================================
# Builder -- canonical pyrite Pa-3 conventional * 2x2x2 = 96 atoms
# (Fe Wyckoff 4a, S Wyckoff 8c (x,x,x), x ~0.385, a=5.418 A exp/PBE)
# ============================================================

def build_pyrite(a: float = 5.418, x_S: float = 0.385,
                 repeat=(2, 2, 2)):
    """Build pyrite FeS2, Pa-3 (#205), a=5.418 A, conv cell * repeat = 96 atoms.

    VERBATIM structure builder from neb_canonical_pyr_96at_qe.py (V_S2 parent),
    но с explicit lattice/x_S params for community-verified values:
      a = 5.418 A      (Bayliss 1977 exp; PBE 5.417-5.425 A, <0.1% deviation)
      x_S = 0.385      (Brik 2021 Front Chem; MP mp-226)
      Wyckoff Fe 4a    (0, 0, 0) + symmetry
      Wyckoff S 8c     (x, x, x) + symmetry, S-S dimer ~2.18 A
    Conventional cell = 12 atoms (Fe4 S8); after repeat (2,2,2) = 96 atoms
    (Fe32 S64). 2x2x2 cell ~10.836 A -- V_Fe-V_Fe spacing ~10.84 A (well >8 A
    isolated-defect threshold).

    Reference: PYR_VFE_NOMAD_REFERENCE_2026-05-28.md.
    """
    unit = crystal(
        symbols=["Fe", "S"],
        basis=[(0, 0, 0), (x_S, x_S, x_S)],
        spacegroup=205,
        cellpar=[a, a, a, 90, 90, 90],
        primitive_cell=False,
    )
    supercell = unit.repeat(tuple(repeat))
    # Sanity check (rule #1: NE DOVERYAT KOMMENTARIYAM)
    n_fe = sum(1 for s in supercell.get_chemical_symbols() if s == "Fe")
    n_s = sum(1 for s in supercell.get_chemical_symbols() if s == "S")
    expected_fe = 4 * repeat[0] * repeat[1] * repeat[2]
    expected_s = 8 * repeat[0] * repeat[1] * repeat[2]
    assert n_fe == expected_fe and n_s == expected_s and len(supercell) == expected_fe + expected_s, (
        f"Pyrite build wrong multiplicity: Fe={n_fe} (exp {expected_fe}), "
        f"S={n_s} (exp {expected_s}), N={len(supercell)} (exp {expected_fe + expected_s})"
    )
    return supercell


# ============================================================
# V_Fe picker -- pyrite Pa-3 octahedral Fe(4a) site (s148 V_Fe pivot)
# ============================================================

def pick_vfe_oct_and_two_s_neighbours(atoms, fe_s_max: float = 2.85,
                                      expected_n_s: int = 6,
                                      ss_min: float = 2.30,
                                      ss_max: float = 4.20):
    """Pick V_Fe = central Fe(4a) + 2 S anchors forming adjacent edge pair.

    Pyrite Pa-3 (#205) octahedral Fe(4a) site has 6 S neighbours at d ~2.27 A.
    From 6 S, pick (S_i, S_k) edge pair (d_S-S in [ss_min, ss_max] A,
    typical octahedron edge S-S 3.20-3.85 A; ss_min excludes intra-dimer
    S-S ~2.18 A -- pyrite dimers must NOT be hop pairs by chemistry).

    Tie-break for cell embedding (s148 chemistry-signature):
      - Among 12 edges (octahedron edges of 6 anchors), pick pair с minimum
        deviation from Pa-3 mirror plane intersecting V_Fe -- i.e. pair с
        max ||midpoint_S_i_S_k - V_Fe_pos||_mic but bounded; deterministic
        secondary tie-break by (S_i_idx, S_k_idx) ascending.

    Returns: (fe_v_idx, s_i_idx, s_k_idx, hop_d, s_neighbours_list)
      hop_d = d(S_i, S_k) mic (paper-grade hop distance)

    Pyrite cubic Pa-3 expectation: 12 edges all equivalent through point group
    m-3 at vacancy site; supercell embedding breaks some operations. Picker
    enforces deterministic choice for canonical_triple.json reproducibility.
    """
    syms = atoms.get_chemical_symbols()
    fe_indices = [i for i, s in enumerate(syms) if s == "Fe"]
    s_indices = [i for i, s in enumerate(syms) if s == "S"]
    if not fe_indices:
        raise RuntimeError("No Fe atoms found in pristine supercell")
    if not s_indices:
        raise RuntimeError("No S atoms found in pristine supercell")

    # Pick Fe(4a) nearest cell centre (deterministic; central V_Fe minimises
    # V_Fe-V_Fe periodic image interaction).
    cell_centre = atoms.cell.array.sum(axis=0) / 2
    fe_d_centre = [(float(np.linalg.norm(atoms.positions[i] - cell_centre)), i)
                   for i in fe_indices]
    fe_d_centre.sort()
    fe_v = fe_d_centre[0][1]

    # Find S neighbours within fe_s_max
    s_with_d = []
    for s_idx in s_indices:
        d = float(atoms.get_distance(fe_v, s_idx, mic=True))
        if d <= fe_s_max:
            s_with_d.append((d, s_idx))
    s_with_d.sort()
    if len(s_with_d) < expected_n_s:
        raise RuntimeError(
            f"V_Fe at idx {fe_v} has only {len(s_with_d)} S neighbours within "
            f"{fe_s_max} A; expected >={expected_n_s} for octahedral coordination. "
            f"Distances: {[f'{d:.3f}' for d, _ in s_with_d]}"
        )
    s_neighbours = [s_idx for _, s_idx in s_with_d[:expected_n_s]]

    # All pair candidates (S_i, S_k) with d_S-S filter
    pos_v = atoms.positions[fe_v]
    candidates = []
    n_neigh = len(s_neighbours)
    for ia in range(n_neigh):
        for ib in range(ia + 1, n_neigh):
            si, sk = s_neighbours[ia], s_neighbours[ib]
            d_si_sk = float(atoms.get_distance(si, sk, mic=True))
            if d_si_sk < ss_min:
                # Exclude pyrite S-S dimer pair (~2.18 A) -- those are NOT
                # hop targets (intra-dimer chemistry; partner S NOT in pocket
                # but in dimer with one of anchors).
                continue
            if d_si_sk > ss_max:
                continue
            # Compute midpoint S_i--S_k (mic) and its distance to V_Fe (mic)
            pos_si = atoms.positions[si]
            pos_sk = atoms.positions[sk]
            d_sk_from_si = pos_sk - pos_si
            cell = atoms.cell.array
            fc = np.linalg.solve(cell.T, d_sk_from_si)
            fc -= np.round(fc)
            d_sk_mic = cell.T @ fc
            midpoint = pos_si + 0.5 * d_sk_mic
            d_mid_v_vec = pos_v - midpoint
            fc2 = np.linalg.solve(cell.T, d_mid_v_vec)
            fc2 -= np.round(fc2)
            d_mid_v_mic = float(np.linalg.norm(cell.T @ fc2))
            candidates.append({
                "si": int(si),
                "sk": int(sk),
                "d_si_sk": float(d_si_sk),
                "d_mid_to_vfe": d_mid_v_mic,
            })
    if not candidates:
        raise RuntimeError(
            f"No valid (S_i, S_k) edge pair within ss_min={ss_min}--"
            f"ss_max={ss_max} A among {len(s_neighbours)} S neighbours of V_Fe={fe_v}. "
            f"Check fe_s_max / ss bounds; pyrite octahedron edge S-S expected ~3.20-3.85 A."
        )

    # Sort: ascending d_si_sk (prefer shorter, closer-to-octahedron-edge
    # pairs over face-diagonals); secondary by d_mid_to_vfe (prefer mid
    # close to V_Fe pos -- straight hop through pocket centre).
    # Tertiary tie-break: deterministic (si, sk) index order
    candidates.sort(key=lambda c: (c["d_si_sk"], c["d_mid_to_vfe"], c["si"], c["sk"]))
    best = candidates[0]

    print(f"[V_Fe pick mineral=pyrite] V_Fe(4a)={fe_v}", flush=True)
    print(f"  {len(s_neighbours)} S neighbours within {fe_s_max} A (octahedral): "
          f"{s_neighbours}", flush=True)
    for dist, s_idx in s_with_d[:expected_n_s]:
        print(f"    S idx {s_idx}: d_FeS={dist:.3f} A", flush=True)
    print(f"  top-5 S-S edge candidates (by d_si_sk, d_mid_to_vfe):",
          flush=True)
    for rank, c in enumerate(candidates[:5], start=1):
        tag = " [SELECTED]" if rank == 1 else ""
        print(f"    {rank}. S_i={c['si']} S_k={c['sk']} "
              f"d_si_sk={c['d_si_sk']:.3f} d_mid_to_vfe={c['d_mid_to_vfe']:.3f}{tag}",
              flush=True)
    return fe_v, best["si"], best["sk"], best["d_si_sk"], s_neighbours


# ============================================================
# canonical_triple.json save/load (s129 lesson: avoid silent index flip)
# ============================================================

def save_canonical_triple(path: Path, fe_v: int, s_i: int, s_k: int,
                          hop_d: float, s_neighbours: list,
                          extra: dict = None):
    """Persist V_Fe / S_i / S_k indices + provenance to canonical_triple.json.

    s129 lesson: on REUSE_PRISTINE re-loading, picker may pick DIFFERENT triple
    if pristine atoms ordering changed (BFGS relax + write/read xyz roundtrip
    can permute order). Save once after first pick; reload on REUSE to keep
    indices stable across smoke -> production runs.
    """
    payload = {
        "V_Fe_index": int(fe_v),
        "S_i_index": int(s_i),
        "S_k_index": int(s_k),
        "hop_distance_A": float(hop_d),
        "s_neighbours": [int(x) for x in s_neighbours],
        "saved_epoch": float(time.time()),
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, cls=NumpyEncoder))
    print(f"[canonical_triple] saved {path}: V_Fe={fe_v}, "
          f"S_i={s_i}, S_k={s_k}, hop_d={hop_d:.3f} A", flush=True)


def load_canonical_triple(path: Path):
    """Load V_Fe / S_i / S_k triple from canonical_triple.json (REUSE path).

    Returns (fe_v, s_i, s_k, hop_d, s_neighbours) tuple or None if missing.
    Validates indices are non-negative ints.
    """
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    fe_v = int(payload["V_Fe_index"])
    s_i = int(payload["S_i_index"])
    s_k = int(payload["S_k_index"])
    hop_d = float(payload["hop_distance_A"])
    s_neighbours = [int(x) for x in payload.get("s_neighbours", [])]
    if min(fe_v, s_i, s_k) < 0:
        raise ValueError(f"canonical_triple.json has negative indices: {payload}")
    print(f"[canonical_triple] loaded {path}: V_Fe={fe_v}, "
          f"S_i={s_i}, S_k={s_k}, hop_d={hop_d:.3f} A", flush=True)
    return fe_v, s_i, s_k, hop_d, s_neighbours


# ============================================================
# IDPP prewrap (ASE GitLab issue #1130 -- mandatory s148)
# ============================================================

def prewrap_endpoint_for_idpp(initial, final, label: str = "endB"):
    """Fix ASE GitLab issue #1130: pre-IDPP linear interp lacks MIC.

    Atoms whose endpoint position wraps across PBC get linearly interpolated
    через WHOLE cell (not the short hop) -> broken initial NEB path с overlaps.

    Fix: pre-wrap final endpoint relative to initial via find_mic, so
    final.positions = initial.positions + mic_displacement. After this,
    naive linear interp `initial + i*disp` gives correct path.

    Empirically verified s128 (pyr 96at V_S2): only H atom wraps across
    z-boundary (dz_naive=10.0 A vs dz_mic=-0.84 A). Heavy atoms unaffected.

    Returns: (final_unwrapped, n_unwrapped, summary_dict)
    """
    # Physicist condition: defensive guard against slab/2D systems where
    # find_mic for non-PBC axis silently returns naive displacement.
    assert all(initial.pbc), (
        f"prewrap_endpoint_for_idpp requires full 3D PBC, got pbc={initial.pbc}. "
        f"For slabs, manual unwrap needed for non-PBC axis."
    )
    disp_naive = final.positions - initial.positions
    disp_mic, _ = find_mic(disp_naive, initial.cell, initial.pbc)
    diffs = np.linalg.norm(disp_naive - disp_mic, axis=1)
    wrapped_idx = np.where(diffs > 0.5)[0]
    summary = {
        "n_unwrapped": int(len(wrapped_idx)),
        "max_disp_naive_A": float(np.linalg.norm(disp_naive, axis=1).max()),
        "max_disp_mic_A": float(np.linalg.norm(disp_mic, axis=1).max()),
        "wrapped_atoms": [
            {"idx": int(i), "symbol": final.get_chemical_symbols()[i],
             "naive_A": float(np.linalg.norm(disp_naive[i])),
             "mic_A": float(np.linalg.norm(disp_mic[i]))}
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
        print(f"  atom {i} ({sym}): naive |D|={np.linalg.norm(disp_naive[i]):.3f} A "
              f"-> mic |D|={np.linalg.norm(disp_mic[i]):.3f} A",
              flush=True)
    return out, len(wrapped_idx), summary


# ============================================================
# H placer for V_Fe + H endpoint construction
# ============================================================

def build_endpoint_with_H(atoms, v_idx: int, h_anchor_idx: int,
                          bond_length: float = 1.35):
    """Build V_Fe + H endpoint: remove Fe at v_idx, place H at S anchor.

    Geometry:
      1. Compute mic displacement vector S_anchor_pos -> V_Fe_pos
         (dvec = pos_v - pos_a, so points from S anchor toward V_Fe site).
      2. Place H atom at S_anchor + bond_length * unit_vector(S_anchor -> V_Fe).
         So H is INSIDE the V_Fe pocket, BETWEEN S_anchor and V_Fe site
         (S-H bond points inward, toward the vacancy cavity).
      3. Delete Fe atom at v_idx (index shift: subsequent atoms shift by -1).
      4. Append H atom (becomes last index).

    Note (chemistry):
      - bond_length=1.35 A targets covalent S-H (typical 1.34-1.55 A)
      - After BFGS relax, H may localize at S anchor (H2: S-H covalent),
        Fe nearest (H1: Fe-H hydride), or bridge (H3): structure-dependent
      - Cross-mineral pattern (mack 43 meV, greig 1861 meV): H1 likely for
        cubic FeS2 V_Fe pocket (1 hole creates Fe3+ -> H attraction?)
      - Predicted on first iteration: S-H covalent (H2-like) by chemistry,
        will see in endA relax which basin H prefers.

    Returns: new Atoms (Fe count -1, H count +1, total atoms same)
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
        raise RuntimeError(
            f"Anchor and vacancy coincide: v_idx={v_idx}, anchor={h_anchor_idx} "
            f"(d_mic={dn:.6e} A)"
        )
    unit = dvec_mic / dn
    # H is placed OPPOSITE direction (away from V_Fe), so S--H bond points
    # outward into bulk / next pocket. Sign convention: bond_length * unit
    # places H at pos_a + bond_length * (V_Fe - S_anchor) / d = H BETWEEN
    # S_anchor and V_Fe.
    # Empirical from marc/mack/greig: this initial guess relaxes correctly
    # whether final endpoint is S-H_covalent (H2) or Fe-H_hydride (H1)
    # because BFGS finds local min regardless of initial direction.
    pos_h = pos_a + bond_length * unit

    new_atoms = atoms.copy()
    del new_atoms[v_idx]
    new_atoms.append(Atom("H", position=pos_h))
    return new_atoms


# ============================================================
# QE PWSCF calculator -- pyrite (non-magnetic, no U)
# ============================================================

def make_calc(work_dir: Path, label: str, kpts=(2, 2, 2),
              ecutwfc: float = 60.0, ecutrho: float = 480.0,
              mpi_np: int = 1,
              conv_thr: float = 1.0e-8,
              mixing_beta: float = 0.3, mixing_mode: str = "plain",
              electron_maxstep: int = 200,
              disk_io: str = "medium",
              occupations: str = "smearing",
              smearing: str = "gaussian", degauss: float = 0.01,
              pseudo_files: dict = None, pseudo_dir: str = None,
              wfc_reuse: bool = False,
              nspin: int = 1):
    """Build ASE Espresso calculator for pyrite V_Fe + H (non-magnetic baseline).

    Per Q115_DFT_PROTOCOL_FINAL + PYR_VFE_NOMAD_REFERENCE_2026-05-28:
      - ecutwfc=60 Ry (Q-115 standard; pyrite community 38-90 Ry range; ours mid)
      - ecutrho=480 Ry default (8 * ecutwfc, conservative для V_Fe defect rho
        precision; ONCV typical 4x = 240 Ry also OK but 8x safer for paper)
      - nspin=1 baseline (diamagnetic Fe2+ LS d6; MP mp-226, Brik 2021 consensus)
      - NO Hubbard U (community consensus; +U=2 sensitivity in SI optional)
      - occupations='smearing' (V_Fe creates 1 hole; integer occupations would
        give Fermi-level pinning; gaussian degauss=0.01 Ry ~136 meV < gap)
      - mixing_mode='plain' beta=0.3 (homogeneous diamagnetic semiconductor;
        no metallic transitions expected; greigite-style local-TF NOT needed)
      - electron_maxstep=200 (generous for 96-atom V_Fe defect)
      - diagonalization='david'
      - mpirun ALWAYS prefixed (QE GPU pw.x silent crash without it, CLAUDE.md)

    disk_io:
      - 'medium' (default): wfc each opt step (endpoint BFGS recovery on
        SIGTERM kill window); good balance for endpoint relax
      - 'high': wfc each SCF (NEB images; recoverable mid-FIRE on kill;
        ~1 GB / image / SCF -> ~9 GB workspace для 9-image NEB)
      - 'low': no save (NOT recoverable on crash, smallest disk; risky)

    wfc_reuse: enables `restart_mode='restart'` + `startingwfc='file'` for
    SIGTERM kill recovery. Requires previous run with disk_io>='medium'.

    nspin: 1 default (diamagnetic); pass nspin=2 для Tier 1 sensitivity if
    Q1 (magnetic anomaly from V_Fe hole) materializes. If nspin=2, caller
    must set atoms.set_initial_magnetic_moments() prior to make_calc.
    """
    if pseudo_files is None:
        pseudo_files = {"Fe": "Fe.upf", "S": "S.upf", "H": "H.upf"}
    if pseudo_dir is None:
        pseudo_dir = PSEUDO_DIR

    # mpirun ALWAYS, even for np=1 (QE GPU init requirement, CLAUDE.md)
    cmd = f"mpirun --allow-run-as-root --bind-to none -np {mpi_np} {PW_BIN}"

    profile = EspressoProfile(
        command=cmd,
        pseudo_dir=pseudo_dir,
    )

    restart_mode = "restart" if wfc_reuse else "from_scratch"

    input_data = {
        "control": {
            "calculation": "scf",
            "restart_mode": restart_mode,
            "tprnfor": True,
            "tstress": False,
            "verbosity": "high",
            "disk_io": disk_io,
            "outdir": str(Path(work_dir) / label / "tmp"),
            "prefix": label,
        },
        "system": {
            "ecutwfc": ecutwfc,
            "ecutrho": ecutrho,
            "occupations": occupations,
            "nspin": nspin,
            # NO Hubbard U for pyrite primary (community consensus)
        },
        "electrons": {
            "conv_thr": conv_thr,
            "mixing_mode": mixing_mode,
            "mixing_beta": mixing_beta,
            "electron_maxstep": electron_maxstep,
            "diagonalization": "david",
        },
    }

    if wfc_reuse:
        input_data["electrons"]["startingwfc"] = "file"

    # Smearing block -- V_Fe creates 1 hole (96 valence e_pristine -> 95
    # e_endpoint; odd number requires occupations='smearing' для nspin=1
    # to avoid Fermi-level fix issues).
    if occupations == "smearing":
        input_data["system"]["smearing"] = smearing
        input_data["system"]["degauss"] = degauss

    print(f"[DEBUG make_calc] label={label} system={input_data['system']}",
          flush=True)

    return Espresso(
        profile=profile,
        directory=str(Path(work_dir) / label),
        input_data=input_data,
        pseudopotentials=pseudo_files,
        kpts=tuple(kpts),
        koffset=(0, 0, 0),
    )


# ============================================================
# Structural sanity gates (s148: endpoint geometry validation)
# ============================================================

def structural_sanity_gate(atoms, h_idx: int, label: str,
                           d_HS_lo: float = 1.30,
                           d_HS_hi: float = 1.55,
                           d_HFe_min: float = 1.50):
    """Validate endpoint geometry sanity (s148 lesson, MUST PRESERVE).

    Checks:
      G_HS:   d(H, S_nearest) in [d_HS_lo, d_HS_hi] A  -> S-H covalent OK
      G_FeH:  d(H, Fe_any) > d_HFe_min  OR  flag Fe-bridge artifact
      G_id:   nearest non-H atom is S  (not Fe) -> chemistry-sane

    Returns: dict with d_H_S, d_H_Fe, nearest_S_idx, nearest_Fe_idx, geom
    classification (H1/H2/H3 from marc convention), passes (bool).

    Note: this is a WARN gate, not abort. False geometries (Fe-bridge,
    broken bond) are diagnostic for subsequent decision making, not
    immediate stop -- run completes JSON with status='ok' but
    paper_quotable=False if gates fail.
    """
    syms = atoms.get_chemical_symbols()
    all_d = [(j, syms[j], float(atoms.get_distance(h_idx, j, mic=True)))
             for j in range(len(atoms)) if j != h_idx]
    all_d.sort(key=lambda x: x[2])
    d_H_S = min((d for j, sym, d in all_d if sym == "S"), default=float("inf"))
    d_H_Fe = min((d for j, sym, d in all_d if sym == "Fe"), default=float("inf"))
    j_S = next((j for j, sym, d in all_d if sym == "S"), -1)
    j_Fe = next((j for j, sym, d in all_d if sym == "Fe"), -1)
    nearest_j, nearest_sym, nearest_d = all_d[0]

    # H1/H2/H3 classification (chemist convention)
    S_H_covalent = bool(d_H_S < 1.65)   # covalent S-H 1.34-1.55 A
    Fe_H_bond = bool(d_H_Fe < 1.85)     # Fe-H hydride 1.50-1.75 A
    if S_H_covalent and not Fe_H_bond:
        geom = "S-H_covalent"  # H2
    elif Fe_H_bond and not S_H_covalent:
        geom = "Fe-H_hydride"  # H1
    elif S_H_covalent and Fe_H_bond:
        geom = "Fe-H-S_bridge"  # H3
    else:
        geom = "broken_no_near_bond"  # H4 artifact

    # Sanity gate evaluation
    gate_HS_pass = (d_HS_lo <= d_H_S <= d_HS_hi)
    gate_FeH_pass = (d_H_Fe > d_HFe_min)
    gate_id_pass = (nearest_sym == "S")
    passes = bool(gate_HS_pass and gate_FeH_pass and gate_id_pass)

    print(f"  [sanity {label}] d_H-S_min={d_H_S:.3f} A (S#{j_S}) "
          f"[gate {d_HS_lo}--{d_HS_hi}: {'PASS' if gate_HS_pass else 'WARN'}], "
          f"d_H-Fe_min={d_H_Fe:.3f} A (Fe#{j_Fe}) "
          f"[gate >{d_HFe_min}: {'PASS' if gate_FeH_pass else 'WARN'}]",
          flush=True)
    print(f"  [sanity {label}] nearest non-H = {nearest_sym}#{nearest_j} "
          f"@ {nearest_d:.3f} A [gate S: {'PASS' if gate_id_pass else 'WARN'}]; "
          f"geom={geom}, overall={'PASS' if passes else 'WARN'}",
          flush=True)

    return {
        "d_H_S_nearest_A": float(d_H_S),
        "d_H_Fe_nearest_A": float(d_H_Fe),
        "nearest_S_idx": int(j_S),
        "nearest_Fe_idx": int(j_Fe),
        "nearest_nonH": {
            "idx": int(nearest_j),
            "symbol": nearest_sym,
            "distance_A": float(nearest_d),
        },
        "geometry": geom,
        "gate_HS_pass": bool(gate_HS_pass),
        "gate_FeH_pass": bool(gate_FeH_pass),
        "gate_id_pass": bool(gate_id_pass),
        "all_gates_pass": passes,
    }


# ============================================================
# Test A diagnostic: not-same-basin check (s130/132 mack/pent precedent)
# ============================================================

def test_A_diagnostic(endA, endB, h_idx_A: int, h_idx_B: int,
                      e_A: float, e_B: float,
                      conv_thr_endpoint: float = 1.0e-8,
                      h_disp_min: float = 0.5):
    """Detect same-basin endpoint artifact (V_Fe+H trap, s132 precedent).

    Same-basin signature (mack canonical 3/3 ARTIFACT confirmed s130):
      - dE_endpoints < SCF noise floor (5 * conv_thr in eV)
      - h_displacement_mic < h_disp_min (default 0.5 A, paper-grade 1.0 A)

    If both conditions true -> endpoints collapsed to same well during BFGS
    relax. NEB will compute trivial barrier (E_a ~= 0 spurious).

    Returns: dict with h_disp_mic, scf_noise_floor, same_basin_risk (bool),
    nearest_class (BOTH_S / BOTH_Fe / ASYMMETRIC), paper_quotable_flag.

    For pyrite cubic Pa-3 with mirror endpoints, dE expected ~0 by symmetry
    (not artifact). Disambiguator: h_displacement must be >= hop_distance / 2
    (typical mirror hop H moves ~half hop_d). For pyrite octahedron edge
    hop ~3.85 A, h_displ ~1.6-1.9 A expected -- well above artifact threshold.
    """
    pos_h_A = endA.positions[h_idx_A]
    pos_h_B = endB.positions[h_idx_B]
    dvec = pos_h_B - pos_h_A
    cell_arr = endA.cell.array
    fc = np.linalg.solve(cell_arr.T, dvec)
    fc -= np.round(fc)
    h_disp_mic = float(np.linalg.norm(cell_arr.T @ fc))

    # SCF noise floor (R1-FIX F4 from marc lesson: conv_thr is total-E noise
    # in Ry, NOT per-electron multiplication. Old formula over-estimated
    # noise by 3 orders, made Test A blind. Drop n_elec multiplier.)
    scf_noise_floor_eV = float(conv_thr_endpoint * 13.6056)  # Ry -> eV total
    dE = abs(e_A - e_B)
    same_basin_risk = bool(
        (dE < 5 * scf_noise_floor_eV) and (h_disp_mic < 1.0)
    )

    syms_A = endA.get_chemical_symbols()
    syms_B = endB.get_chemical_symbols()
    all_d_A = sorted(
        ((j, syms_A[j], float(endA.get_distance(h_idx_A, j, mic=True)))
         for j in range(len(endA)) if j != h_idx_A),
        key=lambda x: x[2]
    )
    all_d_B = sorted(
        ((j, syms_B[j], float(endB.get_distance(h_idx_B, j, mic=True)))
         for j in range(len(endB)) if j != h_idx_B),
        key=lambda x: x[2]
    )
    nearest_A_sym = all_d_A[0][1]
    nearest_B_sym = all_d_B[0][1]

    if nearest_A_sym == "Fe" and nearest_B_sym == "Fe":
        nearest_class = "BOTH_Fe_attraction"  # H1 pattern
    elif nearest_A_sym == "S" and nearest_B_sym == "S":
        nearest_class = "BOTH_S_anchor"       # H2 pattern (expected pyrite)
    else:
        nearest_class = "ASYMMETRIC_bridge_or_broken"

    h_disp_ok = bool(h_disp_mic >= h_disp_min)
    print(f"  [Test A] h_disp_mic={h_disp_mic:.3f} A "
          f"(min {h_disp_min} A): {'PASS' if h_disp_ok else 'WARN'}",
          flush=True)
    print(f"  [Test A] nearest_class={nearest_class} "
          f"(A:{nearest_A_sym} B:{nearest_B_sym})", flush=True)
    print(f"  [Test A] dE={dE:.6f} eV vs 5x SCF noise {5 * scf_noise_floor_eV:.2e} eV; "
          f"same_basin_risk={same_basin_risk}", flush=True)

    return {
        "h_displacement_endA_to_endB_mic_A": h_disp_mic,
        "h_disp_passes_min": h_disp_ok,
        "h_disp_min_required": float(h_disp_min),
        "scf_noise_floor_estimate_eV": scf_noise_floor_eV,
        "dE_endpoints_eV": float(dE),
        "test_A_same_basin_risk": same_basin_risk,
        "test_A_nearest_class": nearest_class,
    }


# ============================================================
# REUSE helpers: load relaxed xyz with QE singlepoint kwargs patched
# ============================================================

def _patch_singlepoint_for_load():
    """Monkey-patch SinglePointCalculator to ignore non-ASE-standard results.

    ASE write() saves QE calc results (nspins, nkpts, eigenvalues, fermi_level,
    ...) in extended xyz comment line. ASE read() then fails because
    SinglePointCalculator rejects those non-standard properties.

    Workaround: filter results dict to only include all_properties before SPC
    init. Returns (original_init, restore_callable).
    """
    import ase.calculators.singlepoint as _sp
    from ase.calculators.calculator import all_properties as _all_props
    _orig_spc_init = _sp.SinglePointCalculator.__init__

    def _patched_spc_init(self, atoms=None, **results):
        # s150 fix: kw arg `atoms` not `atoms_obj` для ASE NEB compatibility
        filtered = {k: v for k, v in results.items() if k in _all_props}
        _orig_spc_init(self, atoms, **filtered)

    _sp.SinglePointCalculator.__init__ = _patched_spc_init

    def restore():
        _sp.SinglePointCalculator.__init__ = _orig_spc_init

    return restore


# ONCV PBE PseudoDojo valences (parity-critical). Mirror of dft-neb M0
# electron_parity_gate.DEFAULT_VALENCE. Most even; semicore Mn/Co/Cu are ODD.
_Z_VAL = {"H": 1, "C": 4, "N": 5, "O": 6, "Na": 9, "Mg": 10, "Al": 3, "Si": 4,
          "P": 5, "S": 6, "Cl": 7, "Mn": 15, "Fe": 16, "Co": 17, "Ni": 18,
          "Cu": 19, "Zn": 20, "Se": 16}


def assert_spin_parity(atoms, nspin, charge=0.0, label="", occupations="smearing"):
    """Runtime parity guard (s158). Odd electron count is incompatible with a
    spin-RESTRICTED (nspin=1) FIXED-occupation treatment (forces S=0). BUT with
    metallic SMEARING an odd-e system is fine at nspin=1 -- the half-electron is
    smeared at E_F (pyrite V_Fe+H: cheap nspin=2 test 2026-05-30 confirmed
    abs_mag->0, genuinely non-magnetic). So:
      * occupations='fixed'    + odd + nspin=1 -> HARD ABORT (insulator needs nspin=2).
      * occupations='smearing' + odd + nspin=1 -> WARN only (acceptable for a
        metallic defect; verify the moment collapses to 0 via a cheap nspin=2 test).
    See dft-neb M0 electron_parity_gate.py + feedback_odd_electron_nspin_vacancy_H.
    """
    syms = atoms.get_chemical_symbols()
    missing = sorted(set(s for s in syms if s not in _Z_VAL))
    if missing:
        print(f"[PARITY GUARD {label}] unknown valence for {missing}; skipping check", flush=True)
        return
    n_e = int(round(sum(_Z_VAL[s] for s in syms) - charge))
    par = "odd" if n_e % 2 else "even"
    if n_e % 2 == 1 and nspin == 1:
        base = (f"[PARITY {label}] N_e={n_e} is ODD with nspin=1: an odd-electron "
                f"cell cannot carry S=0.")
        if occupations == "smearing":
            print(f"{base} occupations='smearing' -> half-electron smeared at E_F is "
                  f"ACCEPTABLE for a metallic defect. PROCEEDING (verify M->0 via a cheap "
                  f"nspin=2 test; s158 pyrite confirmed non-magnetic).", flush=True)
            return
        raise RuntimeError(
            f"{base} occupations='{occupations}' (fixed/insulating) -> nspin=1 is "
            f"unphysical; set --nspin 2 + starting_magnetization. (s158 M0 gate)")
    print(f"[PARITY GUARD {label}] N_e={n_e} ({par}), nspin={nspin} -> OK", flush=True)


# ============================================================
# Pipeline -- run pyrite V_Fe + S-H NEB
# ============================================================

def run_pyrite_VFe(args):
    """Main pyrite V_Fe + S-H NEB pipeline.

    Phases:
      1. Build pyrite Pa-3 conv supercell (or REUSE relaxed pristine)
      2. Relax pristine BFGS (or single-point if REUSE)
      3. Pick canonical V_Fe triple (or load from canonical_triple.json on REUSE)
      4. Pre-flight gates G1/G2/G3 (s148 lesson, defer to neb_preflight_gates)
      5. Build + relax endpoint A (V_Fe + H @ S_i)
      6. Build + relax endpoint B (V_Fe + H @ S_k)
      7. Test A diagnostic + structural sanity gate
      8. IDPP-prewrap endB relative to endA (ASE #1130 fix)
      9. CI-NEB FIRE 9 images

    Returns result dict (saved to JSON by main()).
    """
    t_start = time.time()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Disk check (s153 lesson: 100 GB minimum, 150 GB recommended for 9-image NEB)
    disk_status = check_disk_space(work_dir, min_gb=100)

    pseudo_files = {
        "Fe": args.pp_fe,
        "S": args.pp_s,
        "H": args.pp_h,
    }

    # UNIFIED smearing для pristine + endpoints + NEB (per V_S2 sibling lesson):
    # - Pristine pyrite gap 0.95 eV >> degauss 0.01 Ry (~136 meV) -> smearing
    #   essentially inactive for integer fillings; effect <1 meV/atom for pristine
    # - V_Fe + H endpoint: 1 hole -> requires smearing for nspin=1 odd-electron OK
    # - Avoids fragile kwargs split (s126 v4 endpoint pwi showed 'fixed' despite
    #   spec; unified approach robust regardless of ASE Espresso write quirks).
    pristine_calc_kwargs = dict(
        kpts=tuple(args.kpts),
        ecutwfc=args.ecutwfc,
        ecutrho=args.ecutrho,
        mpi_np=args.mpi_np,
        mixing_beta=args.mixing_beta,
        mixing_mode=args.mixing_mode,
        electron_maxstep=args.electron_maxstep,
        disk_io="medium",
        occupations="smearing",
        smearing=args.smearing,
        degauss=args.degauss,
        pseudo_files=pseudo_files,
        pseudo_dir=args.pseudo_dir,
        wfc_reuse=args.wfc_reuse,
        nspin=args.nspin,
    )

    endpoint_calc_kwargs = pristine_calc_kwargs
    # disk_io_neb configurable: 'high' = robust against SIGTERM (wfc each SCF
    # to disk, ~1 GB/image * 9 images = ~9 GB workspace).
    neb_calc_kwargs = dict(endpoint_calc_kwargs, disk_io=args.disk_io_neb)

    result = {
        "mineral": "pyrite",
        "spacegroup": "Pa-3 (#205) cubic",
        "cell_spec": f"conv x {args.repeat[0]}x{args.repeat[1]}x{args.repeat[2]}",
        "code": "QE PWSCF",
        "method": (f"PBE PWFFT, nspin={args.nspin}, ecutwfc={args.ecutwfc} Ry, "
                   f"no Hubbard U (community consensus pyrite)"),
        "protocol": ("canonical V_Fe (Wyckoff 4a, octahedral) + S-H lateral hop, "
                     "s148 V_Fe pivot, РЕШЕНИЕ-082 chemistry-signature scope"),
        "kpts": list(args.kpts),
        "ecutwfc": args.ecutwfc,
        "ecutrho": args.ecutrho,
        "mixing_beta": args.mixing_beta,
        "mixing_mode": args.mixing_mode,
        "smearing": args.smearing,
        "degauss_Ry": args.degauss,
        "n_images": args.n_images,
        "fmax_pristine": args.fmax_pristine,
        "fmax_endpoint": args.fmax_endpoint,
        "fmax_neb": args.fmax_neb,
        "k_spring": args.k_spring,
        "decision_ref": ("RESHENIE-079 (Q-115 protocol s125) + "
                         "RESHENIE-082 chemistry-signature scope (s148). "
                         "knowledge/PYR_VFE_NOMAD_REFERENCE_2026-05-28.md."),
        "literature_anchor": ("V_Fe + S-H in cubic FeS2 -- LITERATURE GAP "
                              "(no DFT-NEB anchor 2014-2026). This script "
                              "produces first paper-grade quantitative anchor. "
                              "Cross-mineral scaling predicts E_a 150-400 meV "
                              "(mack 43, greig 1861, marc TBD). See "
                              "knowledge/CROSS_MINERAL_VFE_BARRIER_PATTERN.md."),
        "disk_check": disk_status,
    }

    # ----- Phase 1+2: build + relax pristine -----
    canonical_triple_path = work_dir / "canonical_triple.json"
    REUSE_FULL = args.reuse_relaxed is not None
    REUSE_PRISTINE_ONLY = (args.reuse_pristine is not None) and not REUSE_FULL

    pre_endA = None
    pre_endB = None
    if REUSE_FULL:
        restore_spc = _patch_singlepoint_for_load()
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
            restore_spc()
        print(f"  pristine={atoms.get_chemical_formula()} {len(atoms)}at, "
              f"endA={pre_endA.get_chemical_formula()} {len(pre_endA)}at, "
              f"endB={pre_endB.get_chemical_formula()} {len(pre_endB)}at",
              flush=True)
        result["reuse_relaxed_from"] = str(src)
    elif REUSE_PRISTINE_ONLY:
        restore_spc = _patch_singlepoint_for_load()
        try:
            src_pristine = Path(args.reuse_pristine)
            if src_pristine.is_dir():
                src_pristine = src_pristine / "relaxed_pristine.xyz"
            if not src_pristine.exists():
                raise FileNotFoundError(
                    f"--reuse-pristine xyz missing: {src_pristine}"
                )
            print(f"[REUSE-PRISTINE] loading {src_pristine}", flush=True)
            atoms = read(str(src_pristine))
            atoms.calc = None
        finally:
            restore_spc()
        result["reuse_pristine_from"] = str(src_pristine)
        print(f"  {atoms.get_chemical_formula()} {len(atoms)}at -- pristine reused",
              flush=True)
    else:
        print(f"[1/6] Build pyrite Pa-3 conv x "
              f"{args.repeat[0]}x{args.repeat[1]}x{args.repeat[2]}",
              flush=True)
        atoms = build_pyrite(
            a=args.lattice_a,
            x_S=args.wyckoff_x_S,
            repeat=tuple(args.repeat),
        )

    result["n_atoms_pristine"] = len(atoms)
    result["formula_pristine"] = atoms.get_chemical_formula()
    result["cell_A"] = atoms.cell.lengths().tolist()
    print(f"  {atoms.get_chemical_formula()}, {len(atoms)} atoms, "
          f"cell={[f'{x:.3f}' for x in atoms.cell.lengths()]} A",
          flush=True)

    # Phase 2: relax pristine (or single-point on REUSE)
    if REUSE_FULL or REUSE_PRISTINE_ONLY:
        print(f"[2/6] Single-point pristine (REUSE -- skip BFGS)", flush=True)
    else:
        print(f"[2/6] Relax pristine (BFGS fmax={args.fmax_pristine})",
              flush=True)
    t0 = time.time()
    label_p = ("sp_pristine" if (REUSE_FULL or REUSE_PRISTINE_ONLY)
               else "relax_pristine")
    # s113 guard: break spin symmetry if nspin=2 (silent SCF plateau prevention)
    if args.nspin == 2:
        atoms.set_initial_magnetic_moments(
            [0.3 if s == "Fe" else 0.0 for s in atoms.get_chemical_symbols()]
        )
    atoms.calc = make_calc(work_dir, label_p,
                           conv_thr=args.conv_thr_endpoint,
                           **pristine_calc_kwargs)
    if REUSE_FULL or REUSE_PRISTINE_ONLY:
        try:
            e_pristine = float(atoms.get_potential_energy())
            fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
            conv = True
            nsteps_p = 0
        except RuntimeError as exc:
            raise RuntimeError(f"pristine single-point failed: {exc}") from exc
    else:
        # BFGS with trajectory= (s134 lesson: mandatory for kill-recovery)
        opt = BFGS(atoms, logfile=str(work_dir / "relax_pristine.log"),
                   trajectory=str(work_dir / "relax_pristine.traj"))
        try:
            conv = bool(opt.run(fmax=args.fmax_pristine,
                                 steps=args.max_steps_relax))
            e_pristine = float(atoms.get_potential_energy())
            fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
            nsteps_p = int(opt.nsteps)
        except RuntimeError as exc:
            raise RuntimeError(
                f"pristine relax failed (likely SCF non-conv): {exc}. "
                f"Check {work_dir}/relax_pristine/. "
                f"Try mixing_beta=0.1 or check pseudo files."
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

    # ----- Phase 3: pick canonical V_Fe triple (or load from JSON) -----
    print("[3/6] Pick canonical V_Fe (Wyckoff 4a, octahedral) + S_i + S_k triple",
          flush=True)
    loaded = None
    if canonical_triple_path.exists():
        try:
            loaded = load_canonical_triple(canonical_triple_path)
        except Exception as e:
            print(f"  [WARN] canonical_triple load failed: {e}; re-picking",
                  flush=True)
            loaded = None
    if loaded is not None:
        sv, si, sk, hop_d, s_neighbours = loaded
        # Validate indices in current pristine atoms
        if not (0 <= sv < len(atoms) and 0 <= si < len(atoms)
                and 0 <= sk < len(atoms)):
            print(f"  [WARN] loaded triple out-of-range for pristine "
                  f"(len={len(atoms)}); re-picking", flush=True)
            loaded = None
    if loaded is None:
        sv, si, sk, hop_d, s_neighbours = pick_vfe_oct_and_two_s_neighbours(
            atoms,
            fe_s_max=args.fe_s_max,
            expected_n_s=args.expected_n_s,
            ss_min=args.ss_min,
            ss_max=args.ss_max,
        )
        save_canonical_triple(
            canonical_triple_path,
            fe_v=sv, s_i=si, s_k=sk,
            hop_d=hop_d, s_neighbours=s_neighbours,
            extra={"mineral": "pyrite", "wyckoff": "Fe(4a) V_Fe + S(8c) anchors"},
        )

    d_sv_si = float(atoms.get_distance(sv, si, mic=True))
    d_sv_sk = float(atoms.get_distance(sv, sk, mic=True))
    print(f"  V_Fe={sv} (Wyckoff 4a), S_i={si} (d_FeS={d_sv_si:.3f}), "
          f"S_k={sk} (d_FeS={d_sv_sk:.3f}), hop d_si_sk={hop_d:.3f} A",
          flush=True)
    result["V_Fe_index"] = int(sv)
    result["V_Fe_wyckoff"] = "4a (octahedral)"
    result["S_i_index"] = int(si)
    result["S_k_index"] = int(sk)
    result["S_neighbours_all"] = [int(x) for x in s_neighbours]
    result["d_VFe_Si_A"] = d_sv_si
    result["d_VFe_Sk_A"] = d_sv_sk
    result["hop_distance_A"] = float(hop_d)

    # ----- Phase 4: pre-flight gates (s148 -- defer to neb_preflight_gates) -----
    # G1 Wyckoff inequivalence, G2 parity, G3 pocket-radius
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from neb_preflight_gates import run_pre_deploy_gates  # type: ignore
        gates_diag = run_pre_deploy_gates(
            atoms,
            V_S_index=sv,           # V_Fe here, not V_S; gate API generic
            S_i_index=si,
            S_k_index=sk,
            nspin=args.nspin,
            metal_symbol="Fe",
            pocket_threshold_A=1.60,
            vacancy_is_metal=True,  # V_Fe RC
        )
        result["preflight_gates"] = gates_diag
    except ImportError as exc:
        print(f"[PRE-FLIGHT WARN] neb_preflight_gates import failed: {exc}. "
              f"Continuing without gate check (NOT recommended).", flush=True)
        result["preflight_gates"] = {"error": str(exc), "skipped": True}
    except Exception as exc:
        print(f"[PRE-FLIGHT WARN] gates run failed: {exc}", flush=True)
        result["preflight_gates"] = {"error": str(exc)}

    # ----- Phase 5: endpoint A (V_Fe + H @ S_i) -----
    t0 = time.time()
    if REUSE_FULL:
        print(f"[4/6] Single-point endA (REUSE from {args.reuse_relaxed})",
              flush=True)
        endA = pre_endA
    else:
        print(f"[4/6] Relax endpoint A (V_Fe={sv}, H on S_i={si}) "
              f"BFGS fmax={args.fmax_endpoint}", flush=True)
        endA = build_endpoint_with_H(
            atoms, v_idx=sv, h_anchor_idx=si,
            bond_length=args.h_bond_length,
        )
    # M0 runtime guard: abort BEFORE SCF if nspin is inconsistent with electron parity
    assert_spin_parity(endA, args.nspin, charge=getattr(args, "tot_charge", 0.0), label="endA")
    label_A = "sp_endA" if REUSE_FULL else "relax_endA"
    # s113 guard: break spin symmetry if nspin=2 (silent SCF plateau prevention)
    if args.nspin == 2:
        endA.set_initial_magnetic_moments(
            [0.3 if s == "Fe" else 0.0 for s in endA.get_chemical_symbols()]
        )
    endA.calc = make_calc(work_dir, label_A,
                          conv_thr=args.conv_thr_endpoint,
                          **endpoint_calc_kwargs)
    if REUSE_FULL:
        try:
            e_A = float(endA.get_potential_energy())
            fmaxA = float(np.linalg.norm(endA.get_forces(), axis=1).max())
            convA = True
            nsteps_A = 0
        except RuntimeError as exc:
            raise RuntimeError(f"endA single-point failed: {exc}") from exc
    else:
        optA = BFGS(endA, logfile=str(work_dir / "relax_endA.log"),
                    trajectory=str(work_dir / "relax_endA.traj"))
        try:
            convA = bool(optA.run(fmax=args.fmax_endpoint,
                                   steps=args.max_steps_endpoint))
            e_A = float(endA.get_potential_energy())
            fmaxA = float(np.linalg.norm(endA.get_forces(), axis=1).max())
            nsteps_A = int(optA.nsteps)
        except RuntimeError as exc:
            raise RuntimeError(f"endA relax failed: {exc}") from exc

    h_idx_A = len(endA) - 1
    sanity_A = structural_sanity_gate(
        endA, h_idx_A, label="endA",
        d_HS_lo=args.d_HS_lo, d_HS_hi=args.d_HS_hi,
        d_HFe_min=args.d_HFe_min,
    )
    result["E_endpointA_eV"] = e_A
    result["relax_endA_steps"] = nsteps_A
    result["endA_converged"] = convA
    result["endA_fmax"] = fmaxA
    result["sanity_endA"] = sanity_A
    # backward-compat keys (match marc schema)
    result["d_H_nearest_endA"] = sanity_A["nearest_nonH"]["distance_A"]
    result["d_H_S_nearest_endA"] = sanity_A["d_H_S_nearest_A"]
    result["d_H_Fe_nearest_endA"] = sanity_A["d_H_Fe_nearest_A"]
    result["nearest_S_idx_endA"] = sanity_A["nearest_S_idx"]
    result["nearest_Fe_idx_endA"] = sanity_A["nearest_Fe_idx"]
    result["nearest_nonH_endA"] = sanity_A["nearest_nonH"]
    result["endA_geometry"] = sanity_A["geometry"]
    result["t_relax_endA_s"] = time.time() - t0
    print(f"  E_A={e_A:.4f} eV, steps={nsteps_A}, fmax={fmaxA:.4f}, conv={convA}, "
          f"{result['t_relax_endA_s']:.0f}s", flush=True)
    write(str(work_dir / "relaxed_endA.xyz"), endA)

    # ----- Phase 6: endpoint B (V_Fe + H @ S_k) -----
    t0 = time.time()
    if REUSE_FULL:
        print(f"[5/6] Single-point endB (REUSE)", flush=True)
        endB = pre_endB
    else:
        print(f"[5/6] Relax endpoint B (V_Fe={sv}, H on S_k={sk}) "
              f"BFGS fmax={args.fmax_endpoint}", flush=True)
        endB = build_endpoint_with_H(
            atoms, v_idx=sv, h_anchor_idx=sk,
            bond_length=args.h_bond_length,
        )
    label_B = "sp_endB" if REUSE_FULL else "relax_endB"
    # s113 guard: break spin symmetry if nspin=2 (silent SCF plateau prevention)
    if args.nspin == 2:
        endB.set_initial_magnetic_moments(
            [0.3 if s == "Fe" else 0.0 for s in endB.get_chemical_symbols()]
        )
    endB.calc = make_calc(work_dir, label_B,
                          conv_thr=args.conv_thr_endpoint,
                          **endpoint_calc_kwargs)
    if REUSE_FULL:
        try:
            e_B = float(endB.get_potential_energy())
            fmaxB = float(np.linalg.norm(endB.get_forces(), axis=1).max())
            convB = True
            nsteps_B = 0
        except RuntimeError as exc:
            raise RuntimeError(f"endB single-point failed: {exc}") from exc
    else:
        optB = BFGS(endB, logfile=str(work_dir / "relax_endB.log"),
                    trajectory=str(work_dir / "relax_endB.traj"))
        try:
            convB = bool(optB.run(fmax=args.fmax_endpoint,
                                   steps=args.max_steps_endpoint))
            e_B = float(endB.get_potential_energy())
            fmaxB = float(np.linalg.norm(endB.get_forces(), axis=1).max())
            nsteps_B = int(optB.nsteps)
        except RuntimeError as exc:
            raise RuntimeError(f"endB relax failed: {exc}") from exc

    h_idx_B = len(endB) - 1
    sanity_B = structural_sanity_gate(
        endB, h_idx_B, label="endB",
        d_HS_lo=args.d_HS_lo, d_HS_hi=args.d_HS_hi,
        d_HFe_min=args.d_HFe_min,
    )
    result["E_endpointB_eV"] = e_B
    result["relax_endB_steps"] = nsteps_B
    result["endB_converged"] = convB
    result["endB_fmax"] = fmaxB
    result["sanity_endB"] = sanity_B
    result["d_H_nearest_endB"] = sanity_B["nearest_nonH"]["distance_A"]
    result["d_H_S_nearest_endB"] = sanity_B["d_H_S_nearest_A"]
    result["d_H_Fe_nearest_endB"] = sanity_B["d_H_Fe_nearest_A"]
    result["nearest_S_idx_endB"] = sanity_B["nearest_S_idx"]
    result["nearest_Fe_idx_endB"] = sanity_B["nearest_Fe_idx"]
    result["nearest_nonH_endB"] = sanity_B["nearest_nonH"]
    result["endB_geometry"] = sanity_B["geometry"]
    result["t_relax_endB_s"] = time.time() - t0
    print(f"  E_B={e_B:.4f} eV, steps={nsteps_B}, fmax={fmaxB:.4f}, conv={convB}, "
          f"{result['t_relax_endB_s']:.0f}s", flush=True)
    write(str(work_dir / "relaxed_endB.xyz"), endB)

    if len(endA) != len(endB):
        raise RuntimeError(
            f"endA/endB atom-count mismatch: {len(endA)} vs {len(endB)}"
        )

    dE_endpoints = abs(e_A - e_B)
    result["dE_endpoints_eV"] = dE_endpoints
    print(f"  |E_A - E_B|={dE_endpoints:.4f} eV "
          f"(should be ~0 for sym hop, pyrite Pa-3 cubic mirror)",
          flush=True)
    if dE_endpoints > 0.05:
        warn = (f"WARNING endpoints asymmetric: |dE|={dE_endpoints:.4f} eV > 0.05 "
                f"-> possible wrong triple / SCF drift / local-min issue")
        print(f"  {warn}", flush=True)
        result["endpoints_symmetric"] = False
        result["endpoints_warning"] = warn
    else:
        result["endpoints_symmetric"] = True

    # ----- Phase 7: Test A diagnostic (s130/132 lesson) -----
    test_A = test_A_diagnostic(
        endA, endB, h_idx_A, h_idx_B,
        e_A=e_A, e_B=e_B,
        conv_thr_endpoint=args.conv_thr_endpoint,
        h_disp_min=0.5,
    )
    result.update(test_A)
    if test_A["test_A_same_basin_risk"]:
        warn2 = (f"CRITICAL Test A: same-basin trap signature "
                 f"(dE<5*noise AND h_disp<1.0 A). "
                 f"Endpoints likely collapsed to same well. "
                 f"s130/132 mack/pent precedent.")
        print(f"  {warn2}", flush=True)
        result["test_A_warning"] = warn2
        result["paper_quotable"] = None  # block paper-quotable claim

    # H1/H2/H3 verdict combining endA + endB geometry
    geom_A = sanity_A["geometry"]
    geom_B = sanity_B["geometry"]
    if geom_A == "S-H_covalent" and geom_B == "S-H_covalent":
        hypothesis_verdict = (
            "H2: clean S-H both endpoints (expected pyrite chemistry; "
            "Fe2+ LS d6 + V_Fe creates 1 hole stabilized by S-H covalent)"
        )
    elif geom_A == "Fe-H_hydride" and geom_B == "Fe-H_hydride":
        hypothesis_verdict = (
            "H1: Fe-H both endpoints (genuine Fe-attraction; "
            "V_Fe-adjacent Fe localizes hole, attracts H hydride-like)"
        )
    elif geom_A == "Fe-H-S_bridge" and geom_B == "Fe-H-S_bridge":
        hypothesis_verdict = (
            "H3: bridge geometry stable both endpoints "
            "(Fe-H-S three-center bond, less common but seen in marc)"
        )
    else:
        hypothesis_verdict = f"asymmetric/mixed: A={geom_A}, B={geom_B}"
    result["hypothesis_verdict"] = hypothesis_verdict
    print(f"  [Hypothesis] {hypothesis_verdict}", flush=True)

    if args.skip_neb:
        print("[SKIP] NEB skipped (--skip-neb, endpoints-only smoke)",
              flush=True)
        result["skip_neb"] = True
        result["t_total_s"] = time.time() - t_start
        return result

    # Defensive: endA.calc and endB.calc must be set before NEB
    if endA.calc is None:
        raise RuntimeError("endA.calc is None before NEB -- calculator detached")
    if endB.calc is None:
        raise RuntimeError("endB.calc is None before NEB -- calculator detached")

    # ----- Phase 8: IDPP-prewrap endB relative to endA (ASE #1130 fix) -----
    if args.idpp_prewrap:
        endB_unwrapped, n_unwrapped, prewrap_summary = prewrap_endpoint_for_idpp(
            endA, endB, label="endB"
        )
        result["idpp_prewrap_n_unwrapped"] = int(n_unwrapped)
        result["idpp_prewrap_summary"] = prewrap_summary
        if n_unwrapped > 0:
            # CRITICAL: ASE Atoms.copy() does NOT preserve calc; restore ref
            # (energy invariant under integer lattice translation, cached
            # results valid).
            endB_unwrapped.calc = endB.calc
            endB = endB_unwrapped
    else:
        result["idpp_prewrap_n_unwrapped"] = 0
        result["idpp_prewrap_summary"] = {"n_unwrapped": 0, "disabled": True}
        print("[IDPP-PREWRAP] disabled via --no-idpp-prewrap (legacy)",
              flush=True)

    # ----- Pre-NEB cleanup: drop sp_*/tmp wfc (greigite s150 lesson) -----
    # Single-point reference results already in *.pwo files; tmp wfc not
    # needed for NEB phase. Saves ~25 GB on small-disk instances.
    for sp_subdir in ("sp_pristine", "sp_endA", "sp_endB"):
        tmp_path = work_dir / sp_subdir / "tmp"
        if tmp_path.exists():
            try:
                size_mb = sum(
                    f.stat().st_size for f in tmp_path.rglob('*') if f.is_file()
                ) / 1024**2
                shutil.rmtree(tmp_path)
                print(f"[CLEANUP-pre-NEB] removed {tmp_path} ({size_mb:.0f} MB)",
                      flush=True)
            except Exception as e:
                print(f"[CLEANUP-pre-NEB] WARN failed to remove {tmp_path}: {e}",
                      flush=True)

    # ----- Phase 9: CI-NEB FIRE -----
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
    # s154 fix: force mic=True in IDPP to avoid stock-ASE mic=False bug.
    # Stock ASE on fresh vastai ships idpp_interpolate mic=False default →
    # IDPP path crosses PBC linearly → image_NN bad geometry → SCF non-conv.
    # marc W4 cost $4+5h before fix discovered (s154). Hot-fix here вместо
    # ASE source patching: pass mic=True kwarg explicitly.
    idpp_mode = None
    if args.init_band_xyz:
        # v4 (s158) warm-start: load a pre-converged band (e.g. MLIP) as the
        # initial path. Skips IDPP entirely. Endpoints (images[0], images[-1])
        # stay pinned to the DFT-relaxed endA/endB; only interior images are
        # repositioned from the warm-start band. Satisfies convergence condition
        # C4 (initial path in the basin of the TRUE MEP) for degenerate-saddle
        # systems where IDPP-linear slides off-ridge (pyrite V_Fe m-3bar pocket).
        # MLIP band xyz carries non-ASE-standard props (nspins, nkpts, ...) in the
        # comment line -> SinglePointCalculator rejects them on read. Reuse the same
        # monkey-patch used for endpoint reuse (s158 fix: was missing here -> crashed).
        _restore_spc = _patch_singlepoint_for_load()
        try:
            warm = read(str(args.init_band_xyz), index=":")
        finally:
            _restore_spc()
        if len(warm) != len(images):
            raise RuntimeError(
                f"--init-band-xyz has {len(warm)} frames, need {len(images)} "
                f"(--n-images {args.n_images})")
        for i in range(1, len(images) - 1):
            images[i].set_positions(warm[i].get_positions())
        idpp_mode = f"warmstart:{Path(args.init_band_xyz).name}"
        print(f"[WARMSTART] band initialized from {args.init_band_xyz} "
              f"({len(warm)} images, IDPP skipped) -- s158 C4 fix", flush=True)
    else:
        try:
            neb.interpolate("idpp", mic=True)
            idpp_mode = "idpp_mic_true"
            print("[IDPP] mic=True used (s154 fix)", flush=True)
        except TypeError as te:
            # Older ASE doesn't accept mic kwarg in interpolate
            print(f"  [IDPP] mic kwarg not supported ({te}); using default", flush=True)
            try:
                neb.interpolate("idpp")
                idpp_mode = "idpp_default"
            except Exception as e:
                print(f"  IDPP failed ({e}), fallback linear", flush=True)
                neb.interpolate()
                idpp_mode = "linear_fallback"
        except Exception as e:
            print(f"  IDPP failed ({e}), fallback linear", flush=True)
            neb.interpolate()
            idpp_mode = "linear_fallback"
    result["idpp_mode"] = idpp_mode

    # IDPP sanity: min interatomic distance check
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
            f"IDPP produced overlapping atoms: min_d={min_interatomic:.3f} A < 0.8. "
            f"Retry with linear interp or adjust endpoints."
        )
    print(f"  min interatomic dist (post-IDPP, all images) = "
          f"{min_interatomic:.3f} A", flush=True)

    # v5 (s158) physicist fix: FixedLine constraint on the migrating H along the
    # S_i->S_k axis. The 5 prior runs stalled because the band rolled OFF the
    # S18->S71 ridge onto third pocket sulfurs (S20/S43). FixedLine lets H move
    # only parallel to S_i->S_k, freezing the perpendicular component at its
    # (good) warm-start value -> no roll-off. Interior images only (NEB fixes
    # endpoints). Slight over-constraint of the small off-line arc -> near-upper-
    # bound barrier; relax for an exact saddle in a follow-up if needed.
    if getattr(args, "constrain_h_line", False):
        # chemist gate (s158): FixedLine freezes H's PERPENDICULAR component at its
        # current value -> only valid on a sane warm-start ARC, NOT IDPP-linear
        # (perp~0 would force H through the vacancy centre = artifact barrier).
        if not args.init_band_xyz:
            raise RuntimeError(
                "[CONSTRAINT] --constrain-h-line REQUIRES --init-band-xyz (a sane MLIP "
                "warm-start arc). With IDPP-linear it freezes H on the straight S_i-S_k "
                "line through the vacancy centre -> artifact. (s158 chemist gate)")
        # tester gate (s158): pristine->endpoint index map is identity only if both
        # S anchors precede the deleted V_Fe; otherwise the direction would be wrong.
        if not (si < sv and sk < sv):
            raise RuntimeError(
                f"[CONSTRAINT] S_i={si}/S_k={sk} not both < V_Fe={sv}: pristine indices "
                f"shift in the endpoint frame -> FixedLine direction would be wrong. "
                f"Remap indices before using --constrain-h-line. (s158 tester gate)")
        from ase.constraints import FixedLine
        h_idx = len(images[0]) - 1
        cell = images[0].cell.array
        d = images[0].positions[sk] - images[0].positions[si]
        fc = np.linalg.solve(cell.T, d)
        fc -= np.round(fc)
        direction = cell.T @ fc
        direction = direction / np.linalg.norm(direction)
        for im in images[1:-1]:
            im.set_constraint(FixedLine(h_idx, direction=direction.tolist()))
        result["h_line_constraint"] = {
            "si": int(si), "sk": int(sk), "h_idx": int(h_idx),
            "direction": [float(x) for x in direction],
        }
        print(f"[CONSTRAINT] FixedLine on H#{h_idx} along S{si}->S{sk} "
              f"dir={[round(float(x),3) for x in direction]} "
              f"(interior images; anti-roll-off s158)", flush=True)

    # Initial path sanity: E_a from IDPP-interpolated images BEFORE FIRE
    # (physicist condition: catch broken path early, avoid wasted hours)
    print("[INITIAL PATH SANITY] computing E_a from IDPP-interpolated images...",
          flush=True)
    initial_energies = [e_A]
    for i, img in enumerate(images[1:-1]):
        try:
            initial_energies.append(float(img.get_potential_energy()))
            print(f"  image {i+1:02d}: E={initial_energies[-1]:.4f} eV",
                  flush=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"initial SCF on image {i+1} failed: {exc}"
            ) from exc
    initial_energies.append(e_B)
    e_a_initial = max(initial_energies) - e_A
    result["E_a_initial_post_idpp_eV"] = float(e_a_initial)
    print(f"[INITIAL PATH SANITY] E_a (post-IDPP, pre-FIRE) = "
          f"{e_a_initial:.3f} eV", flush=True)
    if e_a_initial > 5.0:
        raise RuntimeError(
            f"INITIAL PATH SANITY FAIL: E_a (post-IDPP) = {e_a_initial:.3f} eV "
            f">5 eV threshold. Likely broken initial path (atom overlap, "
            f"wrong basin, or PBC wrap not handled). "
            f"Check IDPP-prewrap status. "
            f"Aborting FIRE -- would waste hours on bad path."
        )
    print(f"  -> SANE (< 5 eV), proceeding to FIRE", flush=True)

    # Optimizer selection — FIRE default, LBFGS / ODE for stiff systems (s156 pyrite fix)
    # trajectory= mandatory for recovery (s134 lesson)
    # gomer agent recommendation hierarchy (preferred → fallback):
    #   1. NEBOptimizer(method='ode')  — adaptive ODE integrator, designed for NEB
    #   2. plain LBFGS                 — Hessian-based, no line search
    #   3. FIRE                        — momentum (default, fails on stiff systems)
    if args.optimizer == "ode":
        if not _HAS_NEB_OPTIMIZER:
            raise RuntimeError(
                "ase.mep.NEBOptimizer not available -- requires ASE >= 3.23. "
                "Fallback: --optimizer lbfgs or fire."
            )
        opt_neb = NEBOptimizer(neb, logfile=str(work_dir / "neb.log"),
                                trajectory=str(work_dir / "neb.traj"),
                                method='ode')
        print(f"[OPTIMIZER] NEBOptimizer(method='ode') (s156 fix v3.2: gomer "
              f"agent preferred -- adaptive ODE integrator on non-conservative "
              f"NEB spring forces)", flush=True)
    elif args.optimizer == "lbfgs":
        opt_neb = LBFGS(neb, logfile=str(work_dir / "neb.log"),
                        trajectory=str(work_dir / "neb.traj"))
        print(f"[OPTIMIZER] LBFGS plain (s156 fix v3.1: plain LBFGS -- line search "
              f"variant stuck on NEB non-conservative forces; pyrite divergence precedent)",
              flush=True)
    else:  # fire
        opt_neb = FIRE(neb, logfile=str(work_dir / "neb.log"),
                       trajectory=str(work_dir / "neb.traj"))
        print(f"[OPTIMIZER] FIRE (default)", flush=True)
    try:
        neb_conv = bool(opt_neb.run(fmax=args.fmax_neb,
                                     steps=args.max_steps_neb))
        energies = [float(img.get_potential_energy()) for img in images]
    except RuntimeError as exc:
        raise RuntimeError(
            f"NEB run/harvest failed after {opt_neb.nsteps} steps: {exc}"
        ) from exc

    e_ref = energies[0]
    rel = [e - e_ref for e in energies]
    e_a = max(rel)
    e_rxn = rel[-1]

    max_idx = int(np.argmax(energies))
    try:
        fci = images[max_idx].get_forces()
        fmax_ci = float(np.linalg.norm(fci, axis=1).max())
    except Exception:
        fmax_ci = float("nan")

    result["neb_k_spring"] = float(args.k_spring)
    result.update(neb_policy)
    result["neb_converged"] = neb_conv
    result["neb_steps"] = int(opt_neb.nsteps)
    result["neb_final_fmax_CI"] = fmax_ci
    result["neb_energies_rel_eV"] = rel
    result["E_a_eV"] = float(e_a)
    result["E_a_paper_quotable"] = (
        float(e_a) if (neb_conv and not result.get("test_A_same_basin_risk"))
        else None
    )
    result["E_rxn_eV"] = float(e_rxn)
    result["t_neb_s"] = time.time() - t0
    print(f"  E_a={e_a:.4f} eV (quotable={result['E_a_paper_quotable'] is not None}), "
          f"E_rxn={e_rxn:.4f} eV, steps={opt_neb.nsteps}, conv={neb_conv}, "
          f"fmax_CI={fmax_ci:.4f}, {result['t_neb_s']:.0f}s", flush=True)

    for k, img in enumerate(images):
        write(str(work_dir / f"final_{k:02d}.xyz"), img)

    # ----- Cross-references and plot -----
    result["cross_mineral_anchors"] = {
        "mack_V_Fe_E_a_meV": 43,
        "greigite_V_Fe_E_a_meV": 1861,
        "marc_V_Fe_status": "4 NEB fails as of s148",
        "pyrite_V_S2_anchor_meV": 94.6,
        "predicted_pyrite_V_Fe_meV_range": [150, 400],
        "note": ("Cross-mineral V_Fe pattern: cubic Pa-3 expected to give "
                 "easier NEB convergence than ortho marc (greigite cubic "
                 "spinel precedent: converged with U=1.0). Pyrite gives "
                 "diamagnetic V_Fe (no Fe magnetic complication unlike "
                 "greigite ferri+U).")
    }
    result["NOTE"] = (
        "Canonical V_Fe (Wyckoff 4a, octahedral) + S-H lateral hop on pyrite "
        "FeS2 (Pa-3, #205). Conventional 12-atom cell * 2x2x2 = 96 atoms "
        "(Fe32 S64 pristine; Fe31 S64 H1 endpoint). Diamagnetic Fe2+ LS d6 "
        "(nspin=1 baseline; community consensus MP mp-226, Brik 2021, "
        "Macke-Timrov 2024). NO Hubbard U primary. Recipe per Q-115 "
        "protocol + s148 V_Fe pivot. Full relax pristine + endpoints "
        "(NO FixAtoms). Phase 1: 1x1x1 conv*2x2x2 96at smoke. "
        "Literature gap: V_Fe + S-H in cubic FeS2 -- first paper-grade "
        "anchor. Cross-mineral scaling predicts 150-400 meV."
    )

    # Plot NEB path
    try:
        png = output_dir / "neb_canonical_pyr_96at_qe_VFe.png"
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.linspace(0, 1, len(rel))
        ax.plot(x, rel, "bo-", linewidth=2, markersize=7)
        ax.set_xlabel("Reaction coordinate")
        ax.set_ylabel("Energy (eV)")
        ax.set_title(
            f"Pyrite FeS2 canonical V_Fe (4a) + S-H hop (QE PWSCF, nspin=1)\n"
            f"E_a={e_a:.3f} eV, n_atoms={len(endA)}, conv={neb_conv}"
        )
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


# ============================================================
# main() -- argparse + dispatch
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=("Canonical V_Fe (4a octahedral) + S-H lateral hop NEB "
                     "for pyrite FeS2 (Pa-3, #205, conv*2x2x2 = 96 at). "
                     "Q-115 protocol, s148 V_Fe pivot.")
    )

    # ---------- paths ----------
    parser.add_argument("--work-dir",
                        default="/workspace/neb_canonical_pyr_96at_qe_VFe")
    parser.add_argument("--output-dir", default="/workspace/results")

    # ---------- cell + supercell ----------
    parser.add_argument("--lattice-a", type=float, default=5.418,
                        help="pyrite lattice constant A (Bayliss 1977 exp; "
                             "PBE 5.417-5.425 A consistent <0.1%% deviation)")
    parser.add_argument("--wyckoff-x-S", type=float, default=0.385,
                        help="S Wyckoff 8c (x,x,x) free param "
                             "(Brik 2021 / MP mp-226: 0.385)")
    parser.add_argument("--repeat", type=int, nargs=3, default=[2, 2, 2],
                        help="pyrite supercell repeat nx ny nz "
                             "(default 2 2 2 = 96 atoms; "
                             "primitive unit = 12 atoms Fe4 S8)")
    parser.add_argument("--kpts", type=int, nargs=3, default=[2, 2, 2],
                        help="k-mesh nx ny nz (default 2 2 2 for "
                             "a~10.84 A conv*2x2x2 cell)")

    # ---------- NEB ----------
    parser.add_argument("--n-images", type=int, default=9,
                        help="total images inc. endpoints")
    parser.add_argument("--k-spring", type=float, default=0.3,
                        help="NEB spring constant (Q-115 default 0.3)")
    parser.add_argument("--dyneb", dest="dyneb", action="store_true",
                        default=True,
                        help="Use ASE DyNEB dynamic relaxation (default)")
    parser.add_argument("--no-dyneb", dest="dyneb", action="store_false",
                        help="Use standard ASE NEB for A/B baseline comparison")
    parser.add_argument("--dyneb-scale-fmax", type=float, default=0.0,
                        help="DyNEB scaled convergence factor (default 0.0 "
                             "keeps same fmax for all images)")
    # Climb/optimizer selection (s156 pyrite fix: 2-stage settle then CI + LBFGS)
    parser.add_argument("--climb", dest="climb", action="store_true",
                        default=True,
                        help="CI-NEB climbing image (default)")
    parser.add_argument("--no-climb", dest="climb", action="store_false",
                        help="Plain NEB (climb=False) — settle band before "
                             "CI. Recommended for stiff systems where FIRE "
                             "diverges with climb=True (pyrite s156 case).")
    parser.add_argument("--optimizer", choices=["fire", "lbfgs", "ode"], default="fire",
                        help="NEB optimizer. fire (default) = momentum-based; "
                             "lbfgs = plain LBFGS (Hessian-based fixed step); "
                             "ode = NEBOptimizer(method='ode') — gomer agent's "
                             "preferred for NEB (adaptive ODE integrator on "
                             "non-conservative spring forces, robust).")
    # v4 (s158): warm-start the band from a pre-converged MLIP band instead of
    # IDPP-linear. Fixes the symmetry-degenerate-saddle slide (pyrite V_Fe: IDPP
    # crosses the m-3bar degenerate region -> band slides off-ridge onto S20/S43).
    # The MLIP-converged band already sits in the convergence basin of the TRUE
    # MEP (THEORY_PENT_CONVERGENCE condition C4). Pass the 9-image .xyz written by
    # neb_pyr_vfe_validate_mlip.py (pyr_vfe_mlip_band_<model>.xyz).
    parser.add_argument("--init-band-xyz", default=None,
                        help="Path to a pre-converged N-image band .xyz (e.g. MLIP "
                             "band) used as the NEB initial path INSTEAD of IDPP. "
                             "Must have exactly --n-images frames. v4 s158 fix for "
                             "degenerate-saddle systems where IDPP-linear slides off.")
    parser.add_argument("--constrain-h-line", action="store_true",
                        help="v5 s158 fix: FixedLine constraint on the migrating H "
                             "along S_i->S_k, blocking the roll-off onto third pocket "
                             "sulfurs that stalled the 5 prior runs (interior images "
                             "only). Near-upper-bound barrier; relax for exact saddle.")

    # ---------- convergence ----------
    # Per РЕШЕНИЕ-079 (Q115 protocol): paper-grade fmax for QE PWSCF
    parser.add_argument("--fmax-pristine", type=float, default=0.03,
                        help="pristine relax fmax (eV/A) -- РЕШЕНИЕ-079: 0.03")
    parser.add_argument("--fmax-endpoint", type=float, default=0.03,
                        help="endpoint relax fmax (eV/A) -- РЕШЕНИЕ-079: 0.03")
    parser.add_argument("--fmax-neb", type=float, default=0.05,
                        help="NEB fmax (eV/A) -- РЕШЕНИЕ-079: 0.05")
    parser.add_argument("--max-steps-relax", type=int, default=200)
    parser.add_argument("--max-steps-endpoint", type=int, default=500)
    parser.add_argument("--max-steps-neb", type=int, default=500)
    parser.add_argument("--conv-thr-endpoint", type=float, default=1.0e-8,
                        help="SCF threshold for endpoint relax (paper-grade)")
    parser.add_argument("--conv-thr-neb", type=float, default=1.0e-7,
                        help="SCF threshold for NEB images (slightly relaxed)")

    # ---------- QE specific ----------
    parser.add_argument("--ecutwfc", type=float, default=60.0,
                        help="plane-wave cutoff Ry "
                             "(Q-115 default; pyrite community 38-90 Ry)")
    parser.add_argument("--ecutrho", type=float, default=480.0,
                        help="density cutoff Ry "
                             "(8 * ecutwfc for ONCV paper-grade rho-precision)")
    parser.add_argument("--mixing-beta", type=float, default=0.3,
                        help="electron mixing (0.3 for pyrite homogeneous "
                             "diamagnetic semiconductor; NOT 0.05 like greigite)")
    parser.add_argument("--mixing-mode", default="plain",
                        choices=["plain", "TF", "local-TF"],
                        help="electron mixing mode (plain default for "
                             "homogeneous diamagnetic pyrite)")
    parser.add_argument("--electron-maxstep", type=int, default=200,
                        help="QE electron_maxstep (200 generous for 96at)")
    parser.add_argument("--nspin", type=int, default=1, choices=[1, 2],
                        help="nspin (1 baseline; 2 if Q1 magnetic anomaly "
                             "from V_Fe hole materializes -- Tier 1 sensitivity)")

    # ---------- smearing ----------
    parser.add_argument("--smearing", default="gaussian",
                        choices=["gaussian", "mp", "mv", "fd"],
                        help="smearing type (gaussian default; "
                             "do NOT use mp/mv for systems with band gap)")
    parser.add_argument("--degauss", type=float, default=0.01,
                        help="smearing width in Ry "
                             "(0.01 Q-115 errata; wider than 0.005 для "
                             "V_Fe-introduced holes near Fermi level)")

    # ---------- pseudopotentials ----------
    parser.add_argument("--pseudo-dir", default=PSEUDO_DIR,
                        help="QE pseudopotential directory")
    # ONCV-SR PBE Pseudo Dojo nc-sr-04 standard
    parser.add_argument("--pp-fe", default="Fe.upf",
                        help="Fe ONCV-SR PBE pseudopotential")
    parser.add_argument("--pp-s", default="S.upf",
                        help="S ONCV-SR PBE pseudopotential")
    parser.add_argument("--pp-h", default="H.upf",
                        help="H ONCV-SR PBE pseudopotential")

    # ---------- V_Fe picker tuning ----------
    parser.add_argument("--fe-s-max", type=float, default=2.85,
                        help="Fe-S neighbour cutoff (pyrite Fe-S nominal "
                             "~2.27 A, 2.85 safe for post-relax thermal exp)")
    parser.add_argument("--expected-n-s", type=int, default=6,
                        help="Expected S neighbours per V_Fe (pyrite "
                             "octahedral = 6)")
    parser.add_argument("--ss-min", type=float, default=2.30,
                        help="Min S-S pair distance for hop pair "
                             "(must exclude S-S dimer ~2.18 A)")
    parser.add_argument("--ss-max", type=float, default=4.20,
                        help="Max S-S pair distance for hop pair "
                             "(octahedron edge ~3.20-3.85 A, plus margin)")

    # ---------- H placement ----------
    parser.add_argument("--h-bond-length", type=float, default=1.35,
                        help="Initial S-H bond length A (target covalent "
                             "S-H 1.34-1.55 A; BFGS will relax to local min)")

    # ---------- structural sanity gates ----------
    parser.add_argument("--d-HS-lo", type=float, default=1.30,
                        help="Sanity gate: min d(H, S_nearest) A")
    parser.add_argument("--d-HS-hi", type=float, default=1.55,
                        help="Sanity gate: max d(H, S_nearest) A")
    parser.add_argument("--d-HFe-min", type=float, default=1.50,
                        help="Sanity gate: min d(H, Fe_any) A "
                             "(below this = Fe-bridge artifact)")

    # ---------- threads ----------
    default_omp = int(os.environ.get("OMP_NUM_THREADS", 8))
    parser.add_argument("--omp", type=int, default=default_omp)
    parser.add_argument("--mpi-np", type=int, default=1,
                        help="MPI ranks (always wrapped in mpirun for QE GPU)")

    # ---------- skip / reuse flags ----------
    parser.add_argument("--skip-endpoints", action="store_true",
                        help="Skip endpoint A/B + NEB (pristine-only smoke; "
                             "fastest sanity check ~2 min A100)")
    parser.add_argument("--skip-neb", action="store_true",
                        help="Skip NEB after endpoint relax "
                             "(paper-grade smoke ~30-90 min A100)")
    parser.add_argument("--disk-io-neb", default="high",
                        choices=["low", "medium", "high"],
                        help="NEB image disk_io: 'high' = wfc each SCF for "
                             "SIGTERM recovery (~1 GB/image * 9 = ~9 GB); "
                             "'medium' = wfc each opt step; "
                             "'low' = no save (NOT recoverable on crash)")
    parser.add_argument("--wfc-reuse", action="store_true",
                        help="Set restart_mode='restart' + startingwfc='file' "
                             "for ALL calcs (resume from saved wfc after "
                             "SIGTERM kill). Requires prior run disk_io>='medium'. "
                             "~3-4x SCF speedup. WARNING: crashes if no save dir.")
    parser.add_argument("--reuse-relaxed", default=None,
                        help="Path к dir with relaxed_pristine.xyz + "
                             "relaxed_endA.xyz + relaxed_endB.xyz. "
                             "Skip BFGS, do single-point SCF on each, "
                             "then jump to NEB phase.")
    parser.add_argument("--reuse-pristine", default=None,
                        help="Path к relaxed_pristine.xyz (file or dir). "
                             "Skip pristine BFGS only; endA/endB built fresh "
                             "via V_Fe picker. NEW s148.")

    # ---------- IDPP prewrap (s148 mandatory) ----------
    parser.add_argument("--idpp-prewrap", action="store_true", default=True,
                        help="Pre-wrap endB relative to endA via find_mic "
                             "before NEB (fixes ASE issue #1130). "
                             "Default ON (paper-grade fix, s148 mandatory).")
    parser.add_argument("--no-idpp-prewrap", dest="idpp_prewrap",
                        action="store_false",
                        help="Disable IDPP prewrap (legacy s127 behavior; "
                             "broken for atoms wrapped across PBC).")

    args = parser.parse_args()

    # ---------- environment ----------
    os.environ["OMP_NUM_THREADS"] = str(args.omp)
    os.environ["MKL_NUM_THREADS"] = str(args.omp)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.omp)
    os.environ.setdefault("OMP_STACKSIZE", "256M")

    # ---------- pre-flight: pseudopotential files exist ----------
    # Chemist Smoke #1: verify pseudo files exist BEFORE singleton + SCF spawn
    missing_pp = []
    for sym, fn in (("Fe", args.pp_fe), ("S", args.pp_s), ("H", args.pp_h)):
        full = Path(args.pseudo_dir) / fn
        if not full.exists():
            missing_pp.append(f"{sym}: {full}")
    if missing_pp:
        print("[FATAL] Missing pseudopotential files:", flush=True)
        for m in missing_pp:
            print(f"  {m}", flush=True)
        print(f"  Available in {args.pseudo_dir}:", flush=True)
        try:
            for f in sorted(Path(args.pseudo_dir).iterdir()):
                print(f"    {f.name}", flush=True)
        except Exception:
            pass
        sys.exit(3)

    # ---------- singleton lock ----------
    acquire_singleton()
    try:
        # ---------- banner ----------
        print("=" * 70, flush=True)
        print("Canonical V_Fe (4a, oct) + S-H NEB: pyrite FeS2 96 at "
              "(QE PWSCF, s148 V_Fe pivot)", flush=True)
        print(f"pw.x={PW_BIN}", flush=True)
        print(f"pseudo_dir={args.pseudo_dir}", flush=True)
        print(f"pseudos: Fe={args.pp_fe}, S={args.pp_s}, H={args.pp_h}",
              flush=True)
        print(f"lattice a={args.lattice_a} A, x_S={args.wyckoff_x_S}",
              flush=True)
        print(f"repeat={args.repeat}, kpts={args.kpts}", flush=True)
        print(f"ecutwfc={args.ecutwfc} Ry, ecutrho={args.ecutrho} Ry "
              f"(ratio {args.ecutrho/args.ecutwfc:.1f}x)", flush=True)
        print(f"n_images={args.n_images}, fmax_neb={args.fmax_neb}, "
              f"k_spring={args.k_spring}", flush=True)
        print(f"fmax_pristine={args.fmax_pristine}, "
              f"fmax_endpoint={args.fmax_endpoint}", flush=True)
        print(f"conv_thr endpoint={args.conv_thr_endpoint}, "
              f"neb={args.conv_thr_neb}", flush=True)
        print(f"mixing_mode={args.mixing_mode} beta={args.mixing_beta}",
              flush=True)
        print(f"smearing={args.smearing} degauss={args.degauss} Ry "
              f"(~ {args.degauss*13.6056:.3f} eV)", flush=True)
        print(f"nspin={args.nspin} (no Hubbard U; pyrite community consensus)",
              flush=True)
        print(f"omp={args.omp}, mpi_np={args.mpi_np}", flush=True)
        print(f"skip_endpoints={args.skip_endpoints}, "
              f"skip_neb={args.skip_neb}, "
              f"disk_io_neb={args.disk_io_neb}, wfc_reuse={args.wfc_reuse}",
              flush=True)
        if args.reuse_relaxed:
            print(f"REUSE_RELAXED={args.reuse_relaxed} -- "
                  f"skip BFGS, single-point + NEB", flush=True)
        elif args.reuse_pristine:
            print(f"REUSE_PRISTINE={args.reuse_pristine} -- "
                  f"skip pristine BFGS, build endA/endB fresh", flush=True)
        print(f"idpp_prewrap={args.idpp_prewrap} (ASE issue #1130 fix)",
              flush=True)
        print(f"РЕШЕНИЕ-079 (Q115 protocol) + РЕШЕНИЕ-082 (s148 V_Fe pivot)",
              flush=True)
        print("=" * 70, flush=True)

        try:
            res = run_pyrite_VFe(args)
            status = "ok"
        except Exception as e:
            res = {
                "mineral": "pyrite",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            status = "fail"
            print(f"[FAIL] {e}", flush=True)
            print(traceback.format_exc(), flush=True)

        res["status"] = status
        res["t_end_epoch"] = time.time()

        out_json = Path(args.output_dir) / "neb_canonical_pyr_96at_qe_VFe.json"
        with open(out_json, "w") as f:
            json.dump(res, f, indent=2, cls=NumpyEncoder)
        print(f"\nSaved {out_json} (status={status})", flush=True)
        sys.exit(0 if status == "ok" else 1)
    finally:
        release_singleton()


if __name__ == "__main__":
    # V_Fe pivot per РЕШЕНИЕ-082 (s148, 2026-05-20).
    # V_S+H deprecated for pyrite: S all-equivalent under Pa-3 Wyckoff 8c,
    # picker hop_mode returned symmetry-equivalent triples -> endpoints
    # by-construction identical (Wigner-Bloch). V_Fe (Wyckoff 4a, octahedral)
    # is the chemistry-correct RC for cubic FeS2 with diamagnetic baseline.
    # See: knowledge/PYR_VFE_NOMAD_REFERENCE_2026-05-28.md
    #      knowledge/PYR_VFE_EXPERIMENT_PLAN.md
    #      knowledge/DECISIONS.md РЕШЕНИЕ-082
    #      knowledge/RC_SELECTION_RULES.md
    main()






