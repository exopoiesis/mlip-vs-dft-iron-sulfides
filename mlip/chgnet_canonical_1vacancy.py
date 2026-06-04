#!/usr/bin/env python3
"""
Canonical 1-vacancy S-hop NEB benchmark: CHGNet v0.3.0 foundation model.

Twin of mace_canonical_1vacancy.py. Same supercells, same endpoints, same NEB protocol.
Only the calculator loader differs: CHGNet instead of MACE-MP-0.

Plan v2 (PAPER_REVISION_PLAN_V2 §2.1) requires pure endmember compositions
(Fe9S8 or Ni9S8) for apples-to-apples with DFT canonical 1-vacancy results.

Paper section: "Limitations of foundation MLIPs for TM sulfide kinetics".
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from ase import Atom
from ase.spacegroup import crystal
from ase.optimize import LBFGS, FIRE
from ase.mep import NEB

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_TAG = "chgnet_v030"
LOCK_FILE = Path("/workspace/.mlip_canonical_chgnet.lock")


def chgnet_pretrained_path(model_name="0.3.0"):
    from chgnet.model import model as chgnet_model_module

    rel_paths = {
        "0.3.0": "../pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar",
        "0.2.0": "../pretrained/0.2.0/chgnet_0.2.0_e30f77s348m32.pth.tar",
        "r2scan": "../pretrained/r2scan/chgnet_r2scan_transfer_learning_e15f36s161m23.pth.tar",
    }
    rel_path = rel_paths.get(model_name)
    if rel_path is None:
        raise ValueError(f"Unknown CHGNet model_name={model_name!r}")
    return str((Path(chgnet_model_module.module_dir) / rel_path).resolve())


def assign_default_magmoms(atoms, mineral_name, mode="default", fe_moment=3.5, ni_moment=2.0):
    """Attach signed collinear moments for the local spin-aware CHGNet fork.

    The model consumes the sign through pymatgen site.properties["magmom"],
    which AseAtomsAdaptor gets from ASE initial_magnetic_moments.

    Modes:
      * "none"     — zero out all magmoms (paramagnetic/non-spin)
      * "preserve" — keep whatever build_*() set (T11 pyrrhotite ferrimag
                     chequerboard; consilium 2026-05-05). Logs current state.
      * "fm"       — uniform ferromagnetic on Fe + Ni
      * "default" / "afm-index" / "afm-layer" — mineral-specific heuristics
    """
    if mode == "none":
        moments = np.zeros(len(atoms), dtype=float)
        atoms.set_initial_magnetic_moments(moments)
        return {
            "magmom_mode": mode,
            "magmom_abs_sum_muB": 0.0,
            "magmom_net_muB": 0.0,
            "magmom_nonzero_count": 0,
        }

    if mode == "preserve":
        # Keep whatever the builder already set (e.g. pyrrhotite ferrimag
        # chequerboard from build_pyrrhotite_stoichiometric). No mutation.
        existing = atoms.get_initial_magnetic_moments()
        return {
            "magmom_mode": mode,
            "magmom_abs_sum_muB": float(np.abs(existing).sum()),
            "magmom_net_muB": float(existing.sum()),
            "magmom_nonzero_count": int(np.count_nonzero(np.abs(existing) > 1e-12)),
        }

    symbols = atoms.get_chemical_symbols()
    moments = np.zeros(len(atoms), dtype=float)
    if mode == "fm":
        for i, sym in enumerate(symbols):
            if sym == "Fe":
                moments[i] = fe_moment
            elif sym == "Ni":
                moments[i] = ni_moment
    elif mode in {"default", "afm-index", "afm-layer"}:
        if mineral_name == "pyrite" and mode == "default":
            # Pyrite reference is low-spin/non-magnetic in this benchmark.
            pass
        elif mineral_name == "mackinawite" and mode in {"default", "afm-layer"}:
            scaled = atoms.get_scaled_positions(wrap=True)
            metal_idx = [i for i, sym in enumerate(symbols) if sym in {"Fe", "Ni"}]
            if metal_idx and mode == "afm-layer":
                z_vals = np.array([scaled[i, 2] for i in metal_idx], dtype=float)
                if float(np.ptp(z_vals)) > 1e-4:
                    z_mid = float(np.median(z_vals))
                    for i in metal_idx:
                        sign = 1.0 if scaled[i, 2] >= z_mid else -1.0
                        moments[i] = sign * (fe_moment if symbols[i] == "Fe" else ni_moment)
                else:
                    mode = "default"
            if metal_idx and mode == "default":
                ordered = sorted(
                    metal_idx,
                    key=lambda i: (round(float(scaled[i, 2]), 6),
                                   round(float(scaled[i, 1]), 6),
                                   round(float(scaled[i, 0]), 6)),
                )
                for rank, i in enumerate(ordered):
                    sign = 1.0 if rank % 2 == 0 else -1.0
                    moments[i] = sign * (fe_moment if symbols[i] == "Fe" else ni_moment)
        else:
            sign = 1.0
            for i, sym in enumerate(symbols):
                if sym == "Fe":
                    moments[i] = sign * fe_moment
                    sign *= -1.0
                elif sym == "Ni":
                    moments[i] = sign * ni_moment
                    sign *= -1.0
    else:
        raise ValueError(f"Unknown magmom mode {mode!r}")

    atoms.set_initial_magnetic_moments(moments)
    return {
        "magmom_mode": mode,
        "magmom_abs_sum_muB": float(np.abs(moments).sum()),
        "magmom_net_muB": float(moments.sum()),
        "magmom_nonzero_count": int(np.count_nonzero(np.abs(moments) > 1e-12)),
    }


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


def check_vram(min_free_gb=2.0, label=""):
    if not torch.cuda.is_available():
        print(f"[VRAM {label}] CUDA not available (CPU fallback)", flush=True)
        return None, None
    torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    alloc = torch.cuda.memory_allocated(0) / 1e9
    free = total - alloc
    print(f"[VRAM {label}] total={total:.2f} GB alloc={alloc:.2f} GB free={free:.2f} GB", flush=True)
    if free < min_free_gb:
        print(f"[VRAM {label}] WARNING only {free:.2f} GB free, need {min_free_gb} GB", flush=True)
    return total, free


def acquire_singleton():
    """Atomic lockfile + python3-only process check (rule #31).

    Uses `ps -C python3 -o pid,args` to match ONLY python interpreter processes
    running this script — NOT bash wrappers that happen to have the filename
    in their argv (feedback_pgrep_self_match_3rd_time.md).
    """
    import subprocess
    try:
        out = subprocess.check_output(
            ["ps", "-C", "python3", "-o", "pid=,args="], text=True
        )
        matches = []
        my_pid = os.getpid()
        for line in out.strip().splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) < 2:
                continue
            pid_str, args_str = parts[0], parts[1]
            if "chgnet_canonical_1vacancy.py" not in args_str:
                continue
            try:
                pid = int(pid_str)
            except ValueError:
                continue
            if pid == my_pid:
                continue
            matches.append(f"{pid} {args_str}")
        if matches:
            print(f"[SINGLETON] another python3 chgnet_canonical_1vacancy.py is running:\n  {matches}", flush=True)
            sys.exit(2)
    except subprocess.CalledProcessError:
        pass  # ps returns 1 if no python3 processes
    if LOCK_FILE.exists():
        try:
            old_pid = int(LOCK_FILE.read_text().strip())
            os.kill(old_pid, 0)
            print(f"[SINGLETON] lock held by PID {old_pid}, exiting", flush=True)
            sys.exit(2)
        except (OSError, ValueError):
            print(f"[SINGLETON] stale lock, overwriting", flush=True)
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.write_text(str(os.getpid()))


def release_singleton():
    try:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
    except Exception:
        pass


# ============================================================
# BUILDERS (copy from MACE twin for apples-to-apples)
# ============================================================

def build_pentlandite(composition="fe"):
    """Pentlandite endmember, Fm-3m (225), a=10.07 A, primitive * 2x2x2 = 136 atoms.

    Plan v2 PAPER_REVISION_PLAN_V2 §2.1 requires pure endmember compositions for
    apples-to-apples cross-validation with DFT canonical 1-vacancy results.

    composition options:
      * "fe"    — pure Fe9S8  (ALL M-sites = Fe).  DEFAULT (canonical endmember).
      * "ni"    — pure Ni9S8  (ALL M-sites = Ni).
      * "mixed" — legacy Fe3Ni6S8 (ASE basis gives 2 Ni + 6 Fe per prim.; replace
                  first 5 Fe -> Ni to reach Ni:Fe = 7:3). PRESERVED for comparison
                  with the historical MACE 1.29 eV value (old protocol). NOT to be
                  used for plan v2 DFT apples-to-apples.
    """
    atoms = crystal(
        symbols=["Ni", "Fe", "S"],
        basis=[(0, 0, 0), (0.356, 0.356, 0.356), (0.118, 0.118, 0.118)],
        spacegroup=225,
        cellpar=[10.07, 10.07, 10.07, 90, 90, 90],
        primitive_cell=True,
    )
    symbols = atoms.get_chemical_symbols()
    if composition == "fe":
        symbols = ["Fe" if s == "Ni" else s for s in symbols]
    elif composition == "ni":
        symbols = ["Ni" if s == "Fe" else s for s in symbols]
    elif composition == "mixed":
        fe_count = 0
        for i, s in enumerate(symbols):
            if s == "Fe" and fe_count < 5:
                symbols[i] = "Ni"
                fe_count += 1
    else:
        raise ValueError(f"build_pentlandite: unknown composition '{composition}' "
                         f"(expected 'fe'|'ni'|'mixed')")
    atoms.set_chemical_symbols(symbols)
    return atoms.repeat((2, 2, 2))


def build_pyrite():
    unit = crystal(
        symbols=["Fe", "S"],
        basis=[(0, 0, 0), (0.385, 0.385, 0.385)],
        spacegroup=205,
        cellpar=[5.416, 5.416, 5.416, 90, 90, 90],
    )
    return unit.repeat((2, 2, 2))


def build_mackinawite(repeat=(2, 2, 2)):
    """Mackinawite FeS, P4/nmm (129), a=3.674 A, c=5.033 A, * 2x2x2 = 32 atoms.

    TODO Mack cross-layer path (plan v2 §2.2: "intra-layer + cross-layer dry").
    Current run with z_tol=1.0 (mineral_map) filters to intra-layer only. For
    cross-layer exploration: relax/remove z_tolerance so sv, si, sk may span
    the vdW gap (c=5.033 A). Expected cross-layer E_a >> intra-layer.
    """
    unit = crystal(
        symbols=["Fe", "S"],
        basis=[(0.0, 0.0, 0.0), (0.0, 0.5, 0.2602)],
        spacegroup=129,
        cellpar=[3.674, 3.674, 5.033, 90, 90, 90],
    )
    return unit.repeat(repeat)


def build_pyrrhotite_stoichiometric(repeat=(2, 2, 2), fe_moment=3.5):
    """Stoichiometric NiAs-type FeS supercell as pyrrhotite Fe7S8 OOD probe (T11).

    Builds troilite-NiAs P6_3/mmc (#194, a=3.446 A, c=5.877 A, Liles &
    de Villiers 2012 American Mineralogist, in `data/pyrrhotite/`) repeated
    to (2,2,2) -> 32 atoms (16 Fe + 16 S). NO Fe removal here — the V_Fe is
    created downstream by the canonical scan driver
    (`mlip_pyrr_endpoint_scan_vfe.py`), exactly mirroring mack/pent V_Fe
    flow for apples-to-apples MLIP cross-architecture comparison.

    Why stoichiometric NiAs and not Fe14S16 with structural vacancies:
    chemist+physicist consilium (2026-05-05) verdict — Fe14S16 introduces
    pre-existing voids that break apples-to-apples with mack/pent
    (full-Fe -> V_Fe events). Stoichiometric Fe16S16 is the cleanest OOD
    probe: same V_Fe-creation protocol as mack/pent, but in a NiAs
    framework absent from MPtrj training distribution.

    Magnetic init: AFM-collinear chequerboard by z-layer, ±fe_moment
    (default 3.5 muB). Stoichiometric Fe16S16 NiAs = troilite (FeS,
    T_N=590K, well above 298K). Pyrrhotite Fe7S8 ferrimagnetism arises
    from V_Fe ordering — here V_Fe is created downstream by the scan
    driver, so init = troilite AFM. Mack treats Fe as PM/itinerant
    (T_N=65K << 298K, REŠENIE-079); NiAs FeS is a different magnetic
    regime — preserve-mode init is canonical.

    Returns ase.Atoms with initial_magnetic_moments set.
    """
    unit = crystal(
        symbols=["Fe", "S"],
        basis=[(0.0, 0.0, 0.0), (1 / 3, 2 / 3, 0.25)],
        spacegroup=194,
        cellpar=[3.446, 3.446, 5.877, 90, 90, 120],
        primitive_cell=True,
    )
    sc = unit.repeat(repeat)

    # Ferrimagnetic chequerboard init: alternating sign by z-layer.
    syms = np.array(sc.get_chemical_symbols())
    fe_mask = syms == "Fe"
    fe_z = sc.positions[fe_mask, 2]
    z_med = float(np.median(fe_z))
    magmoms = np.zeros(len(sc))
    fe_indices = np.where(fe_mask)[0]
    for i in fe_indices:
        magmoms[i] = fe_moment if sc.positions[i, 2] < z_med else -fe_moment
    sc.set_initial_magnetic_moments(magmoms)

    return sc


def pick_vacancy_and_two_neighbours(atoms, d_min=2.5, d_max=4.5, z_tolerance=None,
                                     mineral_name="", hop_mode="diagonal"):
    """
    Pick vacancy V_S (= sv) and TWO of its S neighbours (si, sk). H starts on si
    (endpoint A) and moves to sk (endpoint B). NEB then follows this hop path.

    Selection criteria (canonical 1-vacancy S-hop):
      * si, sk both S neighbours of sv in coordination shell [d_min, d_max].
      * d_si_sk unconstrained except MIC sanity (hop distance emerges from geometry).
      * z_tolerance (mackinawite): sv, si, sk all in the same FeS z-layer.

    hop_mode controls triple ranking:
      * "diagonal"  (default): minimise d_si_sk, subject to d_mid_sv <= 3.0 A.
      * "antipodal": V_S strictly between si and sk. Minimise d_mid_sv.
      * "nearest":  no geometric constraint on d_mid_sv — minimise d_si_sk.
    """
    s_indices = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "S"]

    def _within_layer(a, b):
        if z_tolerance is None:
            return True
        cz = atoms.cell.lengths()[2]
        dz = abs(atoms.positions[a, 2] - atoms.positions[b, 2])
        dz = min(dz, cz - dz)
        return dz <= z_tolerance

    def _search(lo, hi):
        candidates = []
        for sv in s_indices:
            pos_sv = atoms.positions[sv]
            nbrs = []
            for si in s_indices:
                if si == sv:
                    continue
                d_sv_si = atoms.get_distance(sv, si, mic=True)
                if lo < d_sv_si < hi and _within_layer(sv, si):
                    nbrs.append((si, d_sv_si))
            if len(nbrs) < 2:
                continue
            for ia in range(len(nbrs)):
                for ib in range(ia + 1, len(nbrs)):
                    si, _ = nbrs[ia]
                    sk, _ = nbrs[ib]
                    d_si_sk = atoms.get_distance(si, sk, mic=True)
                    if d_si_sk < 2.3:
                        continue
                    if not _within_layer(si, sk):
                        continue
                    pos_si = atoms.positions[si]
                    pos_sk = atoms.positions[sk]
                    d_sk_from_si = pos_sk - pos_si
                    cell = atoms.cell.array
                    fc = np.linalg.solve(cell.T, d_sk_from_si)
                    fc -= np.round(fc)
                    d_sk_mic = cell.T @ fc
                    midpoint = pos_si + 0.5 * d_sk_mic
                    d_mid_sv_vec = pos_sv - midpoint
                    fc2 = np.linalg.solve(cell.T, d_mid_sv_vec)
                    fc2 -= np.round(fc2)
                    d_mid_sv_mic = cell.T @ fc2
                    d_mid_sv = float(np.linalg.norm(d_mid_sv_mic))
                    if hop_mode != "nearest" and d_mid_sv > 3.0:
                        continue
                    candidates.append({
                        "sv": sv, "si": si, "sk": sk,
                        "d_si_sk": float(d_si_sk),
                        "d_mid_sv": float(d_mid_sv),
                    })
        if not candidates:
            return None, []
        if hop_mode == "antipodal":
            candidates.sort(key=lambda c: (c["d_mid_sv"], c["d_si_sk"]))
        else:
            candidates.sort(key=lambda c: (c["d_si_sk"], c["d_mid_sv"]))
        return candidates[0], candidates

    best, candidates = _search(d_min, d_max)
    if best is None:
        best, candidates = _search(1.5, 5.5)
    if best is None:
        raise RuntimeError("No valid (V_S, S_i, S_k) triple found for canonical hop")

    # Diagnostic: top-5 triples
    label = mineral_name or "unknown"
    print(f"[hop_diag mineral={label} mode={hop_mode}] candidate triples "
          f"(top 5 by {'d_mid_sv,d_si_sk' if hop_mode == 'antipodal' else 'd_si_sk,d_mid_sv'}):",
          flush=True)
    for rank, c in enumerate(candidates[:5], start=1):
        tag = " [SELECTED]" if rank == 1 else ""
        print(f"  {rank}. sv={c['sv']} si={c['si']} sk={c['sk']} "
              f"d_si_sk={c['d_si_sk']:.3f} d_mid_sv={c['d_mid_sv']:.3f}{tag}",
              flush=True)

    return best["sv"], best["si"], best["sk"], best["d_si_sk"]


def place_h_on_neighbour_s(atoms, v_idx, h_anchor_idx, bond_length=1.35):
    pos_v = atoms.positions[v_idx].copy()
    pos_a = atoms.positions[h_anchor_idx].copy()
    dvec = pos_v - pos_a
    cell = atoms.cell.array
    fcoord = np.linalg.solve(cell.T, dvec)
    fcoord -= np.round(fcoord)
    dvec_mic = cell.T @ fcoord
    dn = np.linalg.norm(dvec_mic)
    if dn < 1e-6:
        raise RuntimeError("Anchor and vacancy are at the same position")
    unit = dvec_mic / dn
    pos_h = pos_a + bond_length * unit

    new_atoms = atoms.copy()
    del new_atoms[v_idx]
    new_atoms.append(Atom("H", position=pos_h))
    return new_atoms


def mic_vector(atoms, pos_a, pos_b):
    """Minimum-image vector from pos_a to pos_b."""
    dvec = np.asarray(pos_b) - np.asarray(pos_a)
    cell = atoms.cell.array
    fcoord = np.linalg.solve(cell.T, dvec)
    fcoord -= np.round(fcoord)
    return cell.T @ fcoord


def nearest_host_basin(atoms, atom_idx, host_species=("Fe", "Ni", "S")):
    best = None
    for idx, atom in enumerate(atoms):
        if idx == atom_idx or atom.symbol not in host_species:
            continue
        dist = float(atoms.get_distance(atom_idx, idx, mic=True))
        if best is None or dist < best["distance_A"]:
            best = {
                "index": int(idx),
                "symbol": atom.symbol,
                "distance_A": dist,
            }
    if best is None:
        return {"index": None, "symbol": None, "distance_A": None}
    return best


def endpoint_same_basin_diagnostic(endA, endB, migrating_idx=None):
    """Test A: detect endpoints that relaxed to the same local basin."""
    if len(endA) != len(endB):
        raise RuntimeError("endpoint_same_basin_diagnostic: atom count mismatch")
    if migrating_idx is None:
        migrating_idx = len(endA) - 1

    h_disp = float(np.linalg.norm(
        mic_vector(endA, endA.positions[migrating_idx], endB.positions[migrating_idx])
    ))
    non_migrating = []
    for idx in range(len(endA)):
        if idx == migrating_idx:
            continue
        non_migrating.append(float(np.linalg.norm(
            mic_vector(endA, endA.positions[idx], endB.positions[idx])
        )))
    max_non_h = float(max(non_migrating)) if non_migrating else 0.0
    nearest_A = nearest_host_basin(endA, migrating_idx)
    nearest_B = nearest_host_basin(endB, migrating_idx)
    same_nearest = (
        nearest_A["index"] == nearest_B["index"]
        and nearest_A["symbol"] == nearest_B["symbol"]
    )
    h_displacement_too_small = bool(h_disp < 0.5)
    same_basin = bool(h_displacement_too_small and max_non_h < 0.5)
    return {
        "migrating_atom_index": int(migrating_idx),
        "h_displacement_A": h_disp,
        "h_displacement_min_A": 0.5,
        "h_displacement_too_small": h_displacement_too_small,
        "max_non_h_displacement_A": max_non_h,
        "nearest_host_endA": nearest_A,
        "nearest_host_endB": nearest_B,
        "same_nearest_host": bool(same_nearest),
        "same_basin_flag": same_basin,
    }


def place_h_at_metal_bridge(atoms, v_idx, metal_pair, offset_A=0.8):
    """Create V_S + H at a metal-metal bridge near the vacancy."""
    i, j = metal_pair
    pos_i = atoms.positions[i]
    pair_vec = mic_vector(atoms, pos_i, atoms.positions[j])
    midpoint = pos_i + 0.5 * pair_vec
    pos_v = atoms.positions[v_idx]
    away = mic_vector(atoms, pos_v, midpoint)
    norm = np.linalg.norm(away)
    if norm < 1e-6:
        z_axis = np.array([0.0, 0.0, 1.0])
        away = np.cross(pair_vec, z_axis)
        norm = np.linalg.norm(away)
        if norm < 1e-6:
            away = np.array([1.0, 0.0, 0.0])
            norm = 1.0
    pos_h = midpoint + offset_A * away / norm
    new_atoms = atoms.copy()
    del new_atoms[v_idx]
    new_atoms.append(Atom("H", position=pos_h))
    return new_atoms


def build_mack_fe_bridge_endpoints(atoms, v_idx, offset_A=0.8):
    """Pick two distinct Fe/Ni-Fe/Ni bridge basins around V_S and build endpoints."""
    metal_indices = [
        i for i, s in enumerate(atoms.get_chemical_symbols())
        if s in ("Fe", "Ni") and atoms.get_distance(v_idx, i, mic=True) < 4.0
    ]
    candidates = []
    for a, i in enumerate(metal_indices):
        for j in metal_indices[a + 1:]:
            d_pair = float(atoms.get_distance(i, j, mic=True))
            if not (1.8 <= d_pair <= 4.0):
                continue
            endpoint = place_h_at_metal_bridge(atoms, v_idx, (i, j), offset_A=offset_A)
            h_idx = len(endpoint) - 1
            pos_v = atoms.positions[v_idx]
            pos_h = endpoint.positions[h_idx]
            d_from_v = float(np.linalg.norm(mic_vector(atoms, pos_v, pos_h)))
            candidates.append({
                "pair": (int(i), int(j)),
                "d_pair_A": d_pair,
                "d_H_from_Vs_A": d_from_v,
                "endpoint": endpoint,
            })
    if len(candidates) < 2:
        raise RuntimeError(
            f"mack-fe-bridge endpoint selection failed: only {len(candidates)} candidates"
        )

    best = None
    for a, ca in enumerate(candidates):
        for cb in candidates[a + 1:]:
            h_a = ca["endpoint"].positions[-1]
            h_b = cb["endpoint"].positions[-1]
            h_disp = float(np.linalg.norm(mic_vector(ca["endpoint"], h_a, h_b)))
            if best is None or h_disp > best["h_displacement_initial_A"]:
                best = {
                    "A": ca,
                    "B": cb,
                    "h_displacement_initial_A": h_disp,
                }
    if best is None or best["h_displacement_initial_A"] < 2.0:
        raise RuntimeError(
            "mack-fe-bridge endpoint selection failed: no pair with H displacement > 2 A"
        )
    meta = {
        "endpoint_mode_resolved": "mack-fe-bridge",
        "Fe_bridge_A": list(best["A"]["pair"]),
        "Fe_bridge_B": list(best["B"]["pair"]),
        "Fe_bridge_A_distance_A": best["A"]["d_pair_A"],
        "Fe_bridge_B_distance_A": best["B"]["d_pair_A"],
        "H_bridge_initial_displacement_A": best["h_displacement_initial_A"],
        "H_bridge_offset_A": float(offset_A),
    }
    print("[mack-fe-bridge] selected endpoints:", meta, flush=True)
    return best["A"]["endpoint"], best["B"]["endpoint"], meta


# ============================================================
# run_mineral
# ============================================================

def run_mineral(mineral_name, build_fn, calc, args, z_tol=None, d_range=(2.5, 4.5),
                k_spring=0.1):
    """
    Canonical 1-vacancy S-hop protocol (twin of MACE). k_spring per-mineral:
    mackinawite long ~5.2 A hop needs stiffer 0.2 to prevent image bunching.
    """
    t_start = time.time()
    result = {
        "mineral": mineral_name,
        "model": MODEL_TAG,
        "supercell": None,
        "n_atoms_pristine": None,
    }

    t0 = time.time()
    print(f"\n[{mineral_name}] build supercell", flush=True)
    if mineral_name == "pentlandite":
        atoms = build_fn(composition=args.pent_composition)
    else:
        atoms = build_fn()
    magmom_meta = assign_default_magmoms(
        atoms,
        mineral_name,
        mode=args.magmom_mode,
        fe_moment=args.fe_moment,
        ni_moment=args.ni_moment,
    )
    result.update(magmom_meta)
    result["spin_aware_chgnet"] = bool(args.spin_aware_chgnet)
    result["supercell"] = list(atoms.cell.lengths())
    result["n_atoms_pristine"] = len(atoms)
    result["pent_composition"] = args.pent_composition if mineral_name == "pentlandite" else None
    print(f"  {atoms.get_chemical_formula()}, {len(atoms)} atoms", flush=True)
    print(f"  cell lengths: {[f'{x:.3f}' for x in atoms.cell.lengths()]} A", flush=True)
    print(f"  magmoms: mode={magmom_meta['magmom_mode']}, "
          f"net={magmom_meta['magmom_net_muB']:.3f} muB, "
          f"abs_sum={magmom_meta['magmom_abs_sum_muB']:.3f} muB", flush=True)

    check_vram(min_free_gb=2.0, label=f"{mineral_name}_start")
    atoms.calc = calc
    print(f"[{mineral_name}] relax pristine (LBFGS fmax=0.01, 500)", flush=True)
    opt = LBFGS(atoms, logfile=None)
    opt.run(fmax=0.01, steps=500)
    e_pristine = float(atoms.get_potential_energy())
    result["E_pristine_eV"] = e_pristine
    result["relax_pristine_steps"] = int(opt.nsteps)
    result["t_relax_pristine_s"] = time.time() - t0
    print(f"  E_pristine={e_pristine:.4f} eV, {opt.nsteps} steps, {result['t_relax_pristine_s']:.0f}s", flush=True)

    print(f"[{mineral_name}] pick V_S + 2 S neighbours (canonical hop)", flush=True)
    sv_idx, si_idx, sk_idx, hop_d = pick_vacancy_and_two_neighbours(
        atoms, d_min=d_range[0], d_max=d_range[1], z_tolerance=z_tol,
        mineral_name=mineral_name, hop_mode=args.mack_hop_mode,
    )
    d_sv_si = float(atoms.get_distance(sv_idx, si_idx, mic=True))
    d_sv_sk = float(atoms.get_distance(sv_idx, sk_idx, mic=True))
    print(f"  V_S at {sv_idx}; S_i={si_idx} (d_Vs_Si={d_sv_si:.3f} A); "
          f"S_k={sk_idx} (d_Vs_Sk={d_sv_sk:.3f} A); hop d(S_i-S_k)={hop_d:.3f} A", flush=True)
    result["V_S_index"] = int(sv_idx)
    result["S_i_index"] = int(si_idx)
    result["S_k_index"] = int(sk_idx)
    result["d_Vs_Si_A"] = d_sv_si
    result["d_Vs_Sk_A"] = d_sv_sk
    result["hop_distance_A"] = float(hop_d)
    result["mack_hop_mode"] = args.mack_hop_mode
    endpoint_mode = args.endpoint_mode
    if endpoint_mode == "auto":
        endpoint_mode = "mack-fe-bridge" if mineral_name == "mackinawite" else "s-anchor"
    result["endpoint_mode"] = endpoint_mode

    # Aligned endpoint A: V at sv, H on si
    t0 = time.time()
    print(f"[{mineral_name}] endpoint A/B mode={endpoint_mode}", flush=True)
    if endpoint_mode == "mack-fe-bridge":
        endA, endB_initial, endpoint_meta = build_mack_fe_bridge_endpoints(
            atoms, sv_idx, offset_A=args.fe_bridge_offset
        )
        assign_default_magmoms(
            endA, mineral_name, mode=args.magmom_mode,
            fe_moment=args.fe_moment, ni_moment=args.ni_moment,
        )
        assign_default_magmoms(
            endB_initial, mineral_name, mode=args.magmom_mode,
            fe_moment=args.fe_moment, ni_moment=args.ni_moment,
        )
        result.update(endpoint_meta)
        print(f"[{mineral_name}] endpoint A (V_S at {sv_idx}, H at Fe bridge "
              f"{endpoint_meta['Fe_bridge_A']})", flush=True)
    elif endpoint_mode == "s-anchor":
        print(f"[{mineral_name}] endpoint A (V_S at {sv_idx}, H on {si_idx})", flush=True)
        endA = place_h_on_neighbour_s(atoms, v_idx=sv_idx, h_anchor_idx=si_idx)
        assign_default_magmoms(
            endA, mineral_name, mode=args.magmom_mode,
            fe_moment=args.fe_moment, ni_moment=args.ni_moment,
        )
        endB_initial = None
    else:
        raise ValueError(f"Unknown endpoint_mode {endpoint_mode!r}")
    endA.calc = calc
    optA = LBFGS(endA, logfile=None)
    converged_A = bool(optA.run(fmax=0.01, steps=args.max_steps_endpoint))
    e_A = float(endA.get_potential_energy())
    f_A = endA.get_forces()
    fmax_A = float(np.linalg.norm(f_A, axis=1).max())
    result["E_endpointA_eV"] = e_A
    result["relax_endA_steps"] = int(optA.nsteps)
    result["endA_converged"] = converged_A
    result["endA_final_fmax"] = fmax_A
    result["t_relax_endA_s"] = time.time() - t0
    h_idx_A = len(endA) - 1
    result["d_H_nearest_endA"] = float(min(
        endA.get_distance(h_idx_A, j, mic=True) for j in range(len(endA)) if j != h_idx_A
    ))
    print(f"  E_A={e_A:.4f} eV, {optA.nsteps} steps, d_H_min={result['d_H_nearest_endA']:.3f} A, "
          f"{result['t_relax_endA_s']:.0f}s, conv={converged_A}, fmax={fmax_A:.3f}", flush=True)
    if not converged_A:
        print(f"  [{mineral_name}] WARNING endpoint A NOT converged after {args.max_steps_endpoint} steps "
              f"(final fmax={fmax_A:.3f} eV/A)", flush=True)

    # Aligned endpoint B: V at sv, H on sk (same atom ordering as A)
    t0 = time.time()
    if endpoint_mode == "mack-fe-bridge":
        print(f"[{mineral_name}] endpoint B (V_S at {sv_idx}, H at Fe bridge "
              f"{result['Fe_bridge_B']})", flush=True)
        endB = endB_initial
    else:
        print(f"[{mineral_name}] endpoint B (V_S at {sv_idx}, H on {sk_idx})", flush=True)
        endB = place_h_on_neighbour_s(atoms, v_idx=sv_idx, h_anchor_idx=sk_idx)
        assign_default_magmoms(
            endB, mineral_name, mode=args.magmom_mode,
            fe_moment=args.fe_moment, ni_moment=args.ni_moment,
        )
    endB.calc = calc
    optB = LBFGS(endB, logfile=None)
    converged_B = bool(optB.run(fmax=0.01, steps=args.max_steps_endpoint))
    e_B = float(endB.get_potential_energy())
    f_B = endB.get_forces()
    fmax_B = float(np.linalg.norm(f_B, axis=1).max())
    result["E_endpointB_eV"] = e_B
    result["relax_endB_steps"] = int(optB.nsteps)
    result["endB_converged"] = converged_B
    result["endB_final_fmax"] = fmax_B
    result["t_relax_endB_s"] = time.time() - t0
    h_idx_B = len(endB) - 1
    result["d_H_nearest_endB"] = float(min(
        endB.get_distance(h_idx_B, j, mic=True) for j in range(len(endB)) if j != h_idx_B
    ))
    print(f"  E_B={e_B:.4f} eV, {optB.nsteps} steps, d_H_min={result['d_H_nearest_endB']:.3f} A, "
          f"{result['t_relax_endB_s']:.0f}s, conv={converged_B}, fmax={fmax_B:.3f}", flush=True)
    if not converged_B:
        print(f"  [{mineral_name}] WARNING endpoint B NOT converged after {args.max_steps_endpoint} steps "
              f"(final fmax={fmax_B:.3f} eV/A)", flush=True)

    if len(endA) != len(endB):
        raise RuntimeError(f"{mineral_name}: endA/endB atom count mismatch {len(endA)} vs {len(endB)}")

    endpoint_diag = endpoint_same_basin_diagnostic(endA, endB)
    result.update(endpoint_diag)
    print(f"[{mineral_name}] endpoint Test A: H_disp={endpoint_diag['h_displacement_A']:.3f} A, "
          f"max_non_H={endpoint_diag['max_non_h_displacement_A']:.3f} A, "
          f"nearestA={endpoint_diag['nearest_host_endA']}, "
          f"nearestB={endpoint_diag['nearest_host_endB']}, "
          f"same_basin={endpoint_diag['same_basin_flag']}", flush=True)

    if endpoint_diag["same_basin_flag"] and args.abort_same_basin:
        warn_msg = "endpoint_same_basin: endpoints relaxed to same local basin; NEB skipped"
        print(f"  [{mineral_name}] {warn_msg}", flush=True)
        result["failure_mode"] = "endpoint_same_basin"
        result["paper_quotable"] = False
        result["E_a_eV"] = None
        result["E_a_paper_quotable"] = None
        result["neb_converged"] = False
        result["neb_steps"] = 0
        result["neb_skipped"] = True
        result["neb_skip_reason"] = warn_msg
        result["t_total_s"] = time.time() - t_start
        return result

    # CI-NEB
    t0 = time.time()
    print(f"[{mineral_name}] CI-NEB ({args.n_images} images, IDPP, FIRE, "
          f"fmax={args.fmax_neb}, k={k_spring})", flush=True)

    n_intermediate = args.n_images - 2
    start = endA.copy()
    start.calc = calc
    end = endB.copy()
    end.calc = calc
    images = [start]
    for _ in range(n_intermediate):
        img = endA.copy()
        img.calc = calc
        images.append(img)
    images.append(end)

    neb = NEB(
        images,
        climb=True,
        method="improvedtangent",
        allow_shared_calculator=True,
        k=k_spring,
    )
    try:
        neb.interpolate("idpp")
    except Exception as e:
        print(f"  IDPP interpolate failed ({e}), fallback linear", flush=True)
        neb.interpolate()

    # VRAM check with all N images loaded, before NEB optimisation begins.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_pre = torch.cuda.max_memory_allocated(0) / 1e9
        print(f"  [VRAM pre-NEB] peak after loading {args.n_images} images: "
              f"{peak_pre:.2f} GB", flush=True)
        if peak_pre > 9.0:  # RTX 4070 has 12 GB
            print(f"  [VRAM] WARNING: high usage, RTX 4070 limit 12 GB", flush=True)
        result["vram_pre_neb_gb"] = float(peak_pre)
    check_vram(min_free_gb=2.0, label=f"{mineral_name}_pre_neb")

    opt_neb = FIRE(neb, logfile=None)
    conv = bool(opt_neb.run(fmax=args.fmax_neb, steps=args.max_steps_neb))
    neb_energies = [float(img.get_potential_energy()) for img in images]
    e_ref = neb_energies[0]
    neb_rel = [e - e_ref for e in neb_energies]
    e_a = max(neb_rel)
    e_rxn = neb_rel[-1]

    # Max fmax on CI image (max E) — kinetic bottleneck.
    try:
        max_e_idx = int(np.argmax(neb_energies))
        f_ci = images[max_e_idx].get_forces()
        fmax_ci = float(np.linalg.norm(f_ci, axis=1).max())
    except Exception as _e:
        fmax_ci = float("nan")

    result["neb_n_images"] = int(args.n_images)
    result["neb_k_spring"] = float(k_spring)
    result["neb_converged"] = conv
    result["neb_final_fmax"] = fmax_ci
    result["neb_steps"] = int(opt_neb.nsteps)
    result["E_a_eV"] = float(e_a)
    result["E_rxn_eV"] = float(e_rxn)
    result["neb_energies_rel_eV"] = [float(e) for e in neb_rel]
    min_intermediate = min(neb_rel[1:-1]) if len(neb_rel) > 2 else 0.0
    intermediate_well_depth = max(0.0, -float(min_intermediate))
    result["min_neb_rel_eV"] = float(min(neb_rel))
    result["intermediate_well_depth_eV"] = float(intermediate_well_depth)
    result["intermediate_well_flag"] = bool(intermediate_well_depth > 0.15)
    result["t_neb_s"] = time.time() - t0

    # Fix #5: E_rxn sanity — endpoints should be near-degenerate.
    if abs(e_rxn) > 0.15:
        warn_msg = (f"WARNING endpoints asymmetric: |E_rxn|={abs(e_rxn):.3f} eV "
                    f"> 0.15 eV threshold")
        print(f"  {warn_msg}", flush=True)
        result["endpoints_symmetric"] = False
        result["endpoints_warning"] = warn_msg
    else:
        result["endpoints_symmetric"] = True

    min_steps_ok = int(opt_neb.nsteps) >= int(args.min_neb_steps)
    result["min_neb_steps"] = int(args.min_neb_steps)
    result["min_neb_steps_ok"] = bool(min_steps_ok)
    paper_quotable = bool(
        conv
        and converged_A
        and converged_B
        and result["endpoints_symmetric"]
        and not result["same_basin_flag"]
        and not result["intermediate_well_flag"]
        and min_steps_ok
    )
    result["paper_quotable"] = paper_quotable
    result["E_a_paper_quotable"] = float(e_a) if paper_quotable else None
    print(f"  E_a={e_a:.4f} eV (paper_quotable={result['E_a_paper_quotable'] is not None}), "
          f"E_rxn={e_rxn:.4f} eV, steps={opt_neb.nsteps}, conv={conv}, "
          f"fmax_CI={fmax_ci:.3f}, {result['t_neb_s']:.0f}s", flush=True)
    if not min_steps_ok:
        result["neb_steps_warning"] = (
            f"NEB steps {opt_neb.nsteps} < min_neb_steps {args.min_neb_steps}"
        )

    try:
        png_path = Path(args.output_dir) / f"{MODEL_TAG}_{mineral_name}_neb.png"
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.linspace(0, 1, len(neb_rel))
        ax.plot(x, neb_rel, "ro-", linewidth=2, markersize=7)
        ax.set_xlabel("Reaction coordinate")
        ax.set_ylabel("Energy (eV)")
        ax.set_title(f"{mineral_name} canonical 1-vacancy S-hop ({MODEL_TAG})\n"
                     f"E_a={e_a:.3f} eV, n_atoms={len(endA)}, converged={conv}")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(png_path, dpi=130)
        plt.close(fig)
        result["png_path"] = str(png_path)
        print(f"  saved {png_path}", flush=True)
    except Exception as e:
        print(f"  PNG save failed: {e}", flush=True)

    result["t_total_s"] = time.time() - t_start
    if torch.cuda.is_available():
        result["peak_vram_gb"] = float(torch.cuda.max_memory_allocated(0) / 1e9)
        torch.cuda.reset_peak_memory_stats()
    return result


# ============================================================
# Model loader
# ============================================================

def load_calculator(args):
    # CHGNet calculator API -- v0.3.0 provides CHGNetCalculator
    from chgnet.model import CHGNet
    from chgnet.model.dynamics import CHGNetCalculator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] CHGNet {args.chgnet_model_name} on {device} "
          f"(spin_aware={args.spin_aware_chgnet})", flush=True)
    model = None
    if args.spin_aware_chgnet:
        try:
            model = CHGNet.from_file(
                chgnet_pretrained_path(args.chgnet_model_name),
                mlp_out_bias=args.chgnet_model_name == "0.2.0",
                version=args.chgnet_model_name,
                spin_aware=True,
            )
        except TypeError as exc:
            raise RuntimeError(
                "Spin-aware CHGNet requires the local Third Matter CHGNet fork. "
                "Deploy git/chgnet on gomer and put it first in PYTHONPATH."
            ) from exc
    else:
        model = CHGNet.load(
            model_name=args.chgnet_model_name,
            verbose=False,
            use_device=device,
        )
    try:
        calc = CHGNetCalculator(model=model, use_device=device)
    except TypeError:
        # older API fallback
        calc = CHGNetCalculator(model=model)
    # CHGNet model defaults to float32; ASE LBFGS / NEB math is float64
    # CHGNet forward pass stays float32, but stresses/forces returned as numpy float64
    # No dtype toggle exposed in v0.3.0 API.
    return calc


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minerals", default="pent,pyr,mack",
                        help="comma-separated: pent,pyr,mack (default all)")
    parser.add_argument("--output-dir", default="/workspace/results")
    parser.add_argument("--chgnet-model-name", default="0.3.0",
                        choices=["0.3.0", "0.2.0", "r2scan"],
                        help="CHGNet pretrained checkpoint")
    parser.add_argument("--spin-aware-chgnet", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use the local Third Matter CHGNet fork with signed "
                             "magmom input channel")
    parser.add_argument("--magmom-mode", default="default",
                        choices=["default", "none", "fm", "afm-index", "afm-layer"],
                        help="Initial signed magmom pattern for spin-aware CHGNet")
    parser.add_argument("--fe-moment", type=float, default=3.5,
                        help="Initial Fe moment magnitude for spin-aware CHGNet")
    parser.add_argument("--ni-moment", type=float, default=2.0,
                        help="Initial Ni moment magnitude for spin-aware CHGNet")
    parser.add_argument("--fmax-neb", type=float, default=0.03)
    parser.add_argument("--n-images", type=int, default=9)
    parser.add_argument("--max-steps-neb", type=int, default=500)
    parser.add_argument("--pent-composition", default="fe",
                        choices=["fe", "ni", "mixed"],
                        help="pentlandite composition: fe=Fe9S8 (default, plan v2), "
                             "ni=Ni9S8, mixed=legacy Fe3Ni6S8 (historical)")
    parser.add_argument("--mack-hop-mode", default="diagonal",
                        choices=["diagonal", "antipodal", "nearest"],
                        help="S_i-S_k ranking: diagonal (default), antipodal, nearest")
    parser.add_argument("--endpoint-mode", default="auto",
                        choices=["auto", "s-anchor", "mack-fe-bridge"],
                        help="Endpoint construction. auto uses mack-fe-bridge for "
                             "mackinawite and s-anchor for other minerals.")
    parser.add_argument("--fe-bridge-offset", type=float, default=0.8,
                        help="H offset from metal-metal bridge midpoint for "
                             "mack-fe-bridge endpoint mode")
    parser.add_argument("--max-steps-endpoint", type=int, default=500,
                        help="Max LBFGS steps for endpoint A/B relaxation "
                             "(s122 control run: 2000 for pent extended relax)")
    parser.add_argument("--min-neb-steps", type=int, default=0,
                        help="Minimum NEB optimizer steps required for paper_quotable; "
                             "use 20-50 for paper runs to catch IDPP early-exit artifacts")
    parser.add_argument("--abort-same-basin", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Skip NEB and mark result non-quotable if endpoint Test A "
                             "detects the same local basin")
    parser.add_argument("--mack-supercell", default="2x2x2",
                        help="Mack supercell repeat NxMxP (default 2x2x2=32 at, "
                             "use 3x3x2=72 at for finite-size check s122)")
    args = parser.parse_args()

    try:
        mack_rep = tuple(int(x) for x in args.mack_supercell.split("x"))
        if len(mack_rep) != 3:
            raise ValueError
    except ValueError:
        raise SystemExit(f"--mack-supercell must be NxMxP format, got {args.mack_supercell!r}")

    acquire_singleton()
    try:
        args_output = Path(args.output_dir)
        args_output.mkdir(parents=True, exist_ok=True)

        print("=" * 70, flush=True)
        print(f"Canonical 1-vacancy S-hop NEB: {MODEL_TAG}", flush=True)
        print(f"Minerals: {args.minerals}", flush=True)
        print(f"n_images={args.n_images}, fmax_neb={args.fmax_neb}, max_steps={args.max_steps_neb}", flush=True)
        print("=" * 70, flush=True)

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
            check_vram(min_free_gb=2.0, label="init")

        calc = load_calculator(args)

        mineral_map = {
            # (name, build_fn, z_tol, (d_min_VS_coord, d_max_VS_coord), k_spring)
            # k_spring: NEB spring const. Mack long-hop (~5.2 A) needs stiffer 0.2.
            # TODO Mack cross-layer path (plan v2 §2.2): current z_tol=1.0 filters
            # to intra-layer. For cross-layer: z_tol=None (or >= c=5.033).
            "pent": ("pentlandite", build_pentlandite, None, (2.0, 4.5), 0.1),
            "pyr":  ("pyrite",      build_pyrite,      None, (2.5, 4.5), 0.1),
            "mack": ("mackinawite", lambda: build_mackinawite(repeat=mack_rep), 1.0,  (3.0, 4.5), 0.2),
        }
        selected = [m.strip() for m in args.minerals.split(",") if m.strip()]

        all_results = {
            "model_tag": MODEL_TAG,
            "fmax_neb": args.fmax_neb,
            "n_images": args.n_images,
            "max_steps_neb": args.max_steps_neb,
            "pent_composition": args.pent_composition,
            "mack_hop_mode": args.mack_hop_mode,
            "chgnet_model_name": args.chgnet_model_name,
            "spin_aware_chgnet": bool(args.spin_aware_chgnet),
            "magmom_mode": args.magmom_mode,
            "t_start_epoch": time.time(),
            "runs": {},
        }

        n_ok = 0
        for key in selected:
            if key not in mineral_map:
                print(f"[skip] unknown mineral '{key}'", flush=True)
                continue
            name, build_fn, z_tol, d_range, k_spring = mineral_map[key]
            print("\n" + "#" * 70, flush=True)
            print(f"# {name}", flush=True)
            print("#" * 70, flush=True)
            try:
                res = run_mineral(name, build_fn, calc, args, z_tol=z_tol,
                                  d_range=d_range, k_spring=k_spring)
                all_results["runs"][name] = res
                n_ok += 1
            except Exception as e:
                print(f"[{name}] FAILED: {e}", flush=True)
                all_results["runs"][name] = {
                    "mineral": name,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }

        all_results["n_ok"] = n_ok
        all_results["n_total"] = len(selected)
        all_results["t_total_s"] = time.time() - all_results["t_start_epoch"]

        json_path = args_output / f"{MODEL_TAG}_canonical_1vacancy.json"
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2, cls=NumpyEncoder)
        print(f"\n[DONE] saved {json_path} ({n_ok}/{len(selected)} minerals OK)", flush=True)
        print(f"Total: {all_results['t_total_s']:.0f}s", flush=True)

        if n_ok > 0:
            done_flag = Path(f"/workspace/DONE_{MODEL_TAG}")
            done_flag.write_text(f"ok={n_ok}/{len(selected)}\njson={json_path}\n")
            print(f"[DONE] flag: {done_flag}", flush=True)
            sys.exit(0)
        else:
            print("[FAIL] all minerals failed", flush=True)
            sys.exit(1)
    finally:
        release_singleton()


if __name__ == "__main__":
    main()
