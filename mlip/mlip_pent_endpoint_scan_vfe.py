#!/usr/bin/env python3
"""V_Fe + H endpoint basin scan for PENTLANDITE (Fe9S8).

Adapted from mlip_mack_endpoint_scan_vfe.py (s132): same V_Fe+H methodology,
swapped builder + cubane-Fe filter (pent has 4-coord cubane Fe AND 6-coord
octahedral Fe; we target tetrahedrally-coordinated cubane for apples-to-apples
with mack tetrahedral V_Fe pocket).

Hypothesis: pent V_S+H confirmed broken (s132 W3 mid-relax: H migrated to
Fe-Fe bridge at 1.72 Å, same artifact as mack). Test V_Fe+H as alternative.

Pentlandite Fe9S8 builder: ASE crystal Fm-3m (225), a=10.07 Å, 2x2x2 supercell
gives 136 atoms (Fe72S64 with composition='fe').
  - 4 Fe per primitive in cubane site Wyckoff 8c (4-coordinate by S, ~2.10 Å)
  - 5 Fe per primitive in octahedral 4b (6-coordinate by S, ~2.50 Å)

For V_Fe scan we filter to CUBANE Fe (apples-to-apples with mack tetrahedral),
selecting Fe with exactly 4 S within fe_s_max = 2.4 Å.

Outputs JSON identical schema to mack V_Fe scan, plus mineral=pentlandite.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from ase import Atom
from ase.mep import NEB
from ase.optimize import LBFGS
from ase.optimize import FIRE


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


def acquire_singleton(output_json: str):
    lock_path = Path(output_json).with_suffix(Path(output_json).suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    my_pid = os.getpid()
    if lock_path.exists():
        try:
            old_pid = int(lock_path.read_text().strip())
            os.kill(old_pid, 0)
            print(
                f"[singleton] active lock {lock_path} held by PID {old_pid}; exiting",
                flush=True,
            )
            sys.exit(2)
        except (OSError, ValueError):
            print(f"[singleton] stale lock {lock_path}, overwriting", flush=True)
    lock_path.write_text(str(my_pid))
    return lock_path


def release_singleton(lock_path):
    if lock_path is None:
        return
    try:
        if lock_path.exists() and lock_path.read_text().strip() == str(os.getpid()):
            lock_path.unlink()
    except Exception:
        pass


def load_runner(path: str):
    spec = importlib.util.spec_from_file_location("canonical_runner", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Cannot import runner: {path}")
    spec.loader.exec_module(module)
    return module


def mic_vector(atoms, pos_a, pos_b):
    dvec = np.asarray(pos_b) - np.asarray(pos_a)
    cell = atoms.cell.array
    fcoord = np.linalg.solve(cell.T, dvec)
    fcoord -= np.round(fcoord)
    return cell.T @ fcoord


def pick_cubane_fe_vacancy_and_neighbour_s(atoms, fe_s_max=2.55):
    """Pick an Fe atom in pentlandite for V_Fe placement.

    s132-fix v3: после MACE/CHGNet diagnostic выяснилось что foundation MLIP
    relaxed pent НЕ воспроизводит canonical cubane 4-coord (получается 3 close
    S at 2.34 + 3 far S at 3.57 — distorted). Octahedral Fe (Wyckoff 4b в
    pent) однако сохраняет clean 6-coord (6 S at 2.478 Å MACE / 2.387 CHGNet).

    Pivot strategy: pick the most **regular 6-coord (octahedral)** Fe by finding
    Fe with smallest spread among 6 nearest S (max - min < 0.15 Å). After V_Fe
    removal, the 6 S form pocket; we take the 4 nearest as S_neighbours for
    candidate generation (or all 6 if argument explicitly wants).

    Returns (fe_idx, [s_idx_1..s_idx_4]) — 4 nearest S to selected Fe.
    """
    syms = atoms.get_chemical_symbols()
    fe_indices = [i for i, s in enumerate(syms) if s == "Fe"]
    if not fe_indices:
        raise RuntimeError("No Fe atoms found in pentlandite")
    s_indices = [i for i, s in enumerate(syms) if s == "S"]

    print(f"[pick] {len(fe_indices)} Fe, {len(s_indices)} S in pristine cell",
          flush=True)

    # For each Fe, compute 6 nearest S distances + spread + cell-centre distance.
    cell_centre = atoms.cell.array.sum(axis=0) / 2
    fe_diag = []
    for fe_idx in fe_indices:
        ds = sorted(float(atoms.get_distance(fe_idx, s_idx, mic=True))
                    for s_idx in s_indices)
        nearest6 = ds[:6]
        spread6 = nearest6[-1] - nearest6[0]
        spread4 = nearest6[3] - nearest6[0]
        d_centre = float(np.linalg.norm(atoms.positions[fe_idx] - cell_centre))
        fe_diag.append({
            "fe_idx": fe_idx,
            "d_centre": d_centre,
            "nearest6": nearest6,
            "spread6": spread6,
            "spread4": spread4,
        })

    # Rank Fe by (spread6 small = octahedral, then nearest centre)
    fe_diag_sorted = sorted(fe_diag, key=lambda r: (r["spread6"], r["d_centre"]))

    print(f"[pick] Top 5 Fe ranked by 6-NN S spread (octahedral if spread<0.1):",
          flush=True)
    for r in fe_diag_sorted[:5]:
        print(f"  Fe {r['fe_idx']:3d}: spread6={r['spread6']:.3f} Å, "
              f"spread4={r['spread4']:.3f} Å, d_centre={r['d_centre']:.3f} Å, "
              f"nearest6={[f'{d:.3f}' for d in r['nearest6']]}",
              flush=True)

    # Pick top-ranked (most octahedral, nearest centre)
    selected = fe_diag_sorted[0]
    fe_v_idx = selected["fe_idx"]
    if selected["spread6"] > 0.20:
        print(f"[pick] WARN: best Fe spread6={selected['spread6']:.3f} Å > 0.20 — "
              f"pent structure may be heavily distorted by MLIP. Proceeding "
              f"but expect possible artifact.", flush=True)
    else:
        print(f"[pick] Fe {fe_v_idx} selected (octahedral-like, spread6="
              f"{selected['spread6']:.3f} Å)", flush=True)

    # Take 4 nearest S to selected Fe within fe_s_max (or relax to top-4 nearest)
    s_with_d = sorted(
        (float(atoms.get_distance(fe_v_idx, s_idx, mic=True)), s_idx)
        for s_idx in s_indices
    )
    s_within = [(d, idx) for d, idx in s_with_d if d <= fe_s_max]
    if len(s_within) >= 4:
        s_neighbours = [idx for _, idx in s_within[:4]]
        print(f"[pick] {len(s_within)} S within fe_s_max={fe_s_max} Å, "
              f"taking 4 nearest", flush=True)
    else:
        # Fallback: just take 4 nearest S regardless of cutoff
        s_neighbours = [idx for _, idx in s_with_d[:4]]
        max_d = s_with_d[3][0]
        print(f"[pick] only {len(s_within)} S within {fe_s_max} Å, "
              f"falling back to 4 nearest (max d={max_d:.3f} Å)", flush=True)

    return fe_v_idx, s_neighbours


def place_h_on_s_after_fe_removal(atoms, fe_idx, s_anchor_idx,
                                  direction_mode="toward_fe",
                                  bond_length=1.35):
    """Same as mack V_Fe variant: remove Fe, place H 1.35 Å from S anchor.

    Direction modes (s132 patches): toward_fe / away_fe (1.0 Å) /
    tangent (1.0 Å) / lateral / lateral2.
    """
    pos_fe = atoms.positions[fe_idx].copy()
    pos_s = atoms.positions[s_anchor_idx].copy()
    dvec = mic_vector(atoms, pos_s, pos_fe)
    dn = np.linalg.norm(dvec)
    if dn < 1e-6:
        raise RuntimeError("Anchor S and Fe at same position")
    unit_to_fe = dvec / dn

    effective_bond = bond_length
    if direction_mode == "toward_fe":
        unit = unit_to_fe
    elif direction_mode == "away_fe":
        unit = -unit_to_fe
        effective_bond = 1.0
    elif direction_mode == "tangent":
        z = np.array([0.0, 0.0, 1.0])
        unit = z - np.dot(z, unit_to_fe) * unit_to_fe
        nrm = np.linalg.norm(unit)
        if nrm < 1e-6:
            unit = np.cross(unit_to_fe, np.array([1.0, 0.0, 0.0]))
            nrm = np.linalg.norm(unit)
        unit = unit / nrm
        effective_bond = 1.0
    elif direction_mode == "lateral":
        any_vec = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(any_vec, unit_to_fe)) > 0.95:
            any_vec = np.array([0.0, 1.0, 0.0])
        unit = any_vec - np.dot(any_vec, unit_to_fe) * unit_to_fe
        nrm = np.linalg.norm(unit)
        unit = unit / nrm
    elif direction_mode == "lateral2":
        any_vec = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(any_vec, unit_to_fe)) > 0.95:
            any_vec = np.array([0.0, 1.0, 0.0])
        in_plane = any_vec - np.dot(any_vec, unit_to_fe) * unit_to_fe
        in_plane = in_plane / np.linalg.norm(in_plane)
        unit = np.cross(unit_to_fe, in_plane)
        unit = unit / np.linalg.norm(unit)
    else:
        raise ValueError(f"unknown direction_mode={direction_mode}")

    pos_h = pos_s + effective_bond * unit
    new_atoms = atoms.copy()
    del new_atoms[fe_idx]
    new_atoms.append(Atom("H", position=pos_h))
    return new_atoms


def place_h_at_ss_bridge(atoms, fe_idx, s_pair, offset_A=0.8, sign=1.0):
    """Place H above/below S-S edge after Fe removal."""
    i, j = s_pair
    pos_i = atoms.positions[i]
    pair_vec = mic_vector(atoms, pos_i, atoms.positions[j])
    midpoint = pos_i + 0.5 * pair_vec
    pos_fe = atoms.positions[fe_idx]
    away = mic_vector(atoms, pos_fe, midpoint)
    norm = np.linalg.norm(away)
    if norm < 1e-6:
        away = np.cross(pair_vec, np.array([0.0, 0.0, 1.0]))
        norm = np.linalg.norm(away)
        if norm < 1e-6:
            away = np.array([1.0, 0.0, 0.0])
            norm = 1.0
    pos_h = midpoint + float(sign) * float(offset_A) * away / norm
    new_atoms = atoms.copy()
    del new_atoms[fe_idx]
    new_atoms.append(Atom("H", position=pos_h))
    return new_atoms


def candidate_endpoints_vfe(atoms, fe_idx, s_neighbours, offsets, max_candidates):
    """Generate V_Fe + H candidates (verbatim from mack V_Fe scan, s132 patches).

    Total: 4 S × 5 directions = 20 anchor + 6 S-S pairs × len(offsets) × 2 signs.
    Bridge pairs sorted by d_pair (nearest first).
    """
    candidates = []
    directions = ["toward_fe", "away_fe", "tangent", "lateral", "lateral2"]

    for s_idx in s_neighbours:
        for direction in directions:
            try:
                endpoint = place_h_on_s_after_fe_removal(atoms, fe_idx, s_idx,
                                                         direction_mode=direction)
            except Exception:
                continue
            candidates.append({
                "kind": "s_anchor",
                "anchor": int(s_idx),
                "direction": direction,
                "seed_distance_A": float(atoms.get_distance(fe_idx, s_idx, mic=True)),
                "atoms": endpoint,
            })

    bridge_rows = []
    for a in range(len(s_neighbours)):
        for b in range(a + 1, len(s_neighbours)):
            si, sj = s_neighbours[a], s_neighbours[b]
            d_pair = float(atoms.get_distance(si, sj, mic=True))
            bridge_rows.append((d_pair, si, sj))
    bridge_rows.sort()

    for d_pair, si, sj in bridge_rows:
        for offset in offsets:
            for sign in (1.0, -1.0):
                endpoint = place_h_at_ss_bridge(atoms, fe_idx, (si, sj),
                                                offset_A=offset, sign=sign)
                candidates.append({
                    "kind": "s_bridge",
                    "pair": [int(si), int(sj)],
                    "offset_A": float(offset),
                    "sign": float(sign),
                    "s_pair_distance_A": d_pair,
                    "atoms": endpoint,
                })

    return candidates[:max_candidates]


def maybe_assign_magmoms(runner, atoms, mineral_name, args):
    if hasattr(runner, "assign_default_magmoms"):
        runner.assign_default_magmoms(
            atoms,
            mineral_name,
            mode=args.magmom_mode,
            fe_moment=args.fe_moment,
            ni_moment=args.ni_moment,
        )


def relax_endpoint(runner, atoms, calc, args):
    endpoint = atoms.copy()
    maybe_assign_magmoms(runner, endpoint, "pentlandite", args)
    endpoint.calc = calc
    opt = LBFGS(endpoint, logfile=None)
    converged = bool(opt.run(fmax=args.fmax_endpoint, steps=args.max_steps_endpoint))
    forces = endpoint.get_forces()
    h_idx = len(endpoint) - 1
    return endpoint, {
        "converged": converged,
        "steps": int(opt.nsteps),
        "energy_eV": float(endpoint.get_potential_energy()),
        "fmax_eVA": float(np.linalg.norm(forces, axis=1).max()),
        "h_nearest": runner.nearest_host_basin(endpoint, h_idx),
    }


def cluster_relaxed(rows, threshold_A):
    clusters = []
    for row in rows:
        endpoint = row.pop("_endpoint")
        h_pos = endpoint.positions[-1]
        placed = False
        for cluster in clusters:
            rep = cluster["_rep_endpoint"]
            dist = float(np.linalg.norm(mic_vector(endpoint, h_pos, rep.positions[-1])))
            if dist < threshold_A:
                cluster["members"].append(row)
                cluster["size"] += 1
                placed = True
                break
        if not placed:
            clusters.append({
                "cluster_id": len(clusters),
                "size": 1,
                "_rep_endpoint": endpoint,
                "rep_h_nearest": row["h_nearest"],
                "members": [row],
            })
    for cluster in clusters:
        cluster.pop("_rep_endpoint", None)
    clusters.sort(key=lambda c: c["size"], reverse=True)
    return clusters


def run_neb_from_selected_clusters(runner, rows, calc, args):
    endpoint_symbol = args.neb_endpoint_symbol
    selected_rows = [
        row for row in rows
        if row.get("converged")
        and row.get("h_nearest", {}).get("symbol") == endpoint_symbol
        and "_endpoint" in row
    ]
    if len(selected_rows) < 2:
        return {
            "attempted": False,
            "endpoint_symbol": endpoint_symbol,
            "reason": (
                f"need >=2 converged {endpoint_symbol}-H endpoints, "
                f"found {len(selected_rows)}"
            ),
        }

    e_min = min(row["energy_eV"] for row in selected_rows)
    pool = [
        row for row in selected_rows
        if row["energy_eV"] <= e_min + args.s_endpoint_energy_window_eV
    ]
    if len(pool) < 2:
        pool = sorted(selected_rows, key=lambda row: row["energy_eV"])[:2]

    best = None
    for a, row_a in enumerate(pool):
        for row_b in pool[a + 1:]:
            disp = float(np.linalg.norm(mic_vector(
                row_a["_endpoint"],
                row_a["_endpoint"].positions[-1],
                row_b["_endpoint"].positions[-1],
            )))
            if best is None or disp > best["h_displacement_A"]:
                best = {
                    "row_a": row_a,
                    "row_b": row_b,
                    "h_displacement_A": disp,
                }
    if best is None:
        return {"attempted": False, "reason": f"no {endpoint_symbol}-H endpoint pair"}

    end_a = best["row_a"]["_endpoint"].copy()
    end_b = best["row_b"]["_endpoint"].copy()
    end_a.calc = calc
    end_b.calc = calc
    endpoint_diag = runner.endpoint_same_basin_diagnostic(end_a, end_b)
    if endpoint_diag["same_basin_flag"]:
        return {
            "attempted": False,
            "endpoint_symbol": endpoint_symbol,
            "reason": f"selected {endpoint_symbol}-H endpoints still fail Test A",
            "candidate_a": best["row_a"]["candidate_id"],
            "candidate_b": best["row_b"]["candidate_id"],
            **endpoint_diag,
        }

    n_intermediate = args.n_images - 2
    images = [end_a]
    for _ in range(n_intermediate):
        img = end_a.copy()
        img.calc = calc
        images.append(img)
    images.append(end_b)
    neb = NEB(
        images,
        climb=True,
        method="improvedtangent",
        allow_shared_calculator=True,
        k=args.neb_k,
    )
    try:
        neb.interpolate("idpp")
    except Exception:
        neb.interpolate()
    opt = FIRE(neb, logfile=None)
    conv = bool(opt.run(fmax=args.fmax_neb, steps=args.max_steps_neb))
    energies = [float(img.get_potential_energy()) for img in images]
    rel = [e - energies[0] for e in energies]
    e_a = max(rel)
    e_rxn = rel[-1]
    min_intermediate = min(rel[1:-1]) if len(rel) > 2 else 0.0
    intermediate_well_depth = max(0.0, -float(min_intermediate))
    intermediate_well_flag = intermediate_well_depth > 0.15
    endpoints_symmetric = abs(e_rxn) <= args.endpoint_energy_symmetry_eV
    min_steps_ok = int(opt.nsteps) >= int(args.min_neb_steps)
    paper_quotable = bool(
        conv
        and endpoints_symmetric
        and min_steps_ok
        and not endpoint_diag["same_basin_flag"]
        and not intermediate_well_flag
    )
    return {
        "attempted": True,
        "endpoint_symbol": endpoint_symbol,
        "candidate_a": int(best["row_a"]["candidate_id"]),
        "candidate_b": int(best["row_b"]["candidate_id"]),
        "candidate_a_energy_eV": float(best["row_a"]["energy_eV"]),
        "candidate_b_energy_eV": float(best["row_b"]["energy_eV"]),
        "candidate_a_nearest": best["row_a"]["h_nearest"],
        "candidate_b_nearest": best["row_b"]["h_nearest"],
        **endpoint_diag,
        "n_images": int(args.n_images),
        "neb_converged": conv,
        "neb_steps": int(opt.nsteps),
        "neb_final_fmax": None,
        "E_a_eV": float(e_a),
        "E_rxn_eV": float(e_rxn),
        "neb_energies_rel_eV": [float(x) for x in rel],
        "min_neb_rel_eV": float(min(rel)),
        "intermediate_well_depth_eV": float(intermediate_well_depth),
        "intermediate_well_flag": bool(intermediate_well_flag),
        "endpoints_symmetric": bool(endpoints_symmetric),
        "min_neb_steps": int(args.min_neb_steps),
        "min_neb_steps_ok": bool(min_steps_ok),
        "paper_quotable": paper_quotable,
        "E_a_paper_quotable": float(e_a) if paper_quotable else None,
    }


def load_calculator(runner, backend, args):
    if backend == "mace":
        return runner.load_calculator()
    load_args = SimpleNamespace(
        chgnet_model_name=args.chgnet_model_name,
        spin_aware_chgnet=args.spin_aware_chgnet,
    )
    return runner.load_calculator(load_args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True)
    parser.add_argument("--backend", required=True, choices=["mace", "chgnet"])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--pent-composition", default="fe",
                        choices=["fe", "ni", "mixed"],
                        help="pentlandite composition (default fe = Fe9S8)")
    parser.add_argument("--offsets", default="0.3,0.6,0.9,1.2,1.6")
    parser.add_argument("--max-candidates", type=int, default=80)
    parser.add_argument("--max-steps-pristine", type=int, default=500)
    parser.add_argument("--max-steps-endpoint", type=int, default=250)
    parser.add_argument("--fmax-pristine", type=float, default=0.01)
    parser.add_argument("--fmax-endpoint", type=float, default=0.02)
    parser.add_argument("--cluster-threshold-A", type=float, default=0.5)
    parser.add_argument("--run-neb", action="store_true")
    parser.add_argument("--neb-endpoint-symbol", default="S",
                        choices=["S", "Fe", "Ni"])
    parser.add_argument("--s-endpoint-energy-window-eV", type=float, default=0.2)
    parser.add_argument("--n-images", type=int, default=9)
    parser.add_argument("--max-steps-neb", type=int, default=300)
    parser.add_argument("--fmax-neb", type=float, default=0.03)
    parser.add_argument("--min-neb-steps", type=int, default=20)
    parser.add_argument("--neb-k", type=float, default=0.2)
    parser.add_argument("--endpoint-energy-symmetry-eV", type=float, default=0.15)
    parser.add_argument("--chgnet-model-name", default="0.3.0")
    parser.add_argument("--spin-aware-chgnet", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--magmom-mode", default="default")
    parser.add_argument("--fe-moment", type=float, default=3.5)
    parser.add_argument("--ni-moment", type=float, default=2.0)
    parser.add_argument("--fe-s-max", type=float, default=2.55,
                        help="Max Fe-S distance to count S as cubane Fe neighbour. "
                             "Default 2.55 Å (foundation MLIP overstructures Fe-S; "
                             "cubane nominal 2.10 Å → relaxed 2.40-2.55 Å). "
                             "Auto-fallback up to 2.80 if no cubane Fe found.")
    args = parser.parse_args()

    lock_path = acquire_singleton(args.output_json)
    try:
        runner = load_runner(args.runner)
        calc = load_calculator(runner, args.backend, args)
        offsets = [float(x) for x in args.offsets.split(",") if x.strip()]

        t0 = time.time()
        atoms = runner.build_pentlandite(composition=args.pent_composition)
        maybe_assign_magmoms(runner, atoms, "pentlandite", args)
        atoms.calc = calc
        opt = LBFGS(atoms, logfile=None)
        pristine_converged = bool(opt.run(fmax=args.fmax_pristine,
                                          steps=args.max_steps_pristine))
        e_pristine = float(atoms.get_potential_energy())

        n_fe_pristine = sum(1 for s in atoms.get_chemical_symbols() if s == "Fe")
        n_ni_pristine = sum(1 for s in atoms.get_chemical_symbols() if s == "Ni")
        n_s_pristine = sum(1 for s in atoms.get_chemical_symbols() if s == "S")
        print(f"[pent] pristine relaxed: {atoms.get_chemical_formula()}, "
              f"Fe={n_fe_pristine} Ni={n_ni_pristine} S={n_s_pristine} "
              f"total={len(atoms)}, E={e_pristine:.4f} eV, "
              f"steps={int(opt.nsteps)} conv={pristine_converged}",
              flush=True)

        fe_v_idx, s_neighbours = pick_cubane_fe_vacancy_and_neighbour_s(
            atoms, fe_s_max=args.fe_s_max)
        fe_pos = atoms.positions[fe_v_idx]
        print(f"[vfe] V_Fe at idx {fe_v_idx} pos=({fe_pos[0]:.3f},"
              f"{fe_pos[1]:.3f},{fe_pos[2]:.3f}); 4 S neighbours: {s_neighbours}",
              flush=True)
        for s_idx in s_neighbours:
            d = float(atoms.get_distance(fe_v_idx, s_idx, mic=True))
            sp = atoms.positions[s_idx]
            print(f"  S idx {s_idx}: d_FeS={d:.3f} Å, pos=({sp[0]:.3f},"
                  f"{sp[1]:.3f},{sp[2]:.3f})", flush=True)

        candidates = candidate_endpoints_vfe(
            atoms, fe_v_idx, s_neighbours, offsets, args.max_candidates
        )
        print(f"[scan] generated {len(candidates)} V_Fe candidates", flush=True)

        rows = []
        for idx, cand in enumerate(candidates):
            seed = cand.pop("atoms")
            seed_h_pos = seed.positions[-1].copy()
            try:
                relaxed, metrics = relax_endpoint(runner, seed, calc, args)
                final_h_pos = relaxed.positions[-1].copy()
                row = {
                    "candidate_id": idx,
                    **cand,
                    "seed_to_final_h_displacement_A": float(
                        np.linalg.norm(mic_vector(seed, seed_h_pos, final_h_pos))
                    ),
                    **metrics,
                    "_endpoint": relaxed,
                }
                print(
                    f"[scan] {idx:03d} {cand['kind']} E={row['energy_eV']:.4f} "
                    f"steps={row['steps']} conv={row['converged']} "
                    f"H_near={row['h_nearest']}",
                    flush=True,
                )
            except Exception as exc:
                row = {
                    "candidate_id": idx,
                    **cand,
                    "error": str(exc),
                    "_endpoint": seed,
                }
                print(f"[scan] {idx:03d} FAILED {exc}", flush=True)
            rows.append(row)

        valid_rows = [row for row in rows if "error" not in row]
        neb_result = None
        if args.run_neb:
            neb_result = run_neb_from_selected_clusters(runner, valid_rows, calc, args)
        clusters = cluster_relaxed(valid_rows, args.cluster_threshold_A)
        result = {
            "backend": args.backend,
            "runner": args.runner,
            "mineral": "pentlandite",
            "pent_composition": args.pent_composition,
            "vacancy_kind": "V_Fe",
            "vacancy_site_kind": "octahedral_or_distorted_4coord_subset",
            "pristine_converged": pristine_converged,
            "pristine_steps": int(opt.nsteps),
            "E_pristine_eV": e_pristine,
            "n_atoms_pristine": len(atoms),
            "V_Fe_index": int(fe_v_idx),
            "S_neighbours": [int(x) for x in s_neighbours],
            "fe_s_max_A": float(args.fe_s_max),
            "n_candidates": len(candidates),
            "n_valid": len(valid_rows),
            "cluster_threshold_A": float(args.cluster_threshold_A),
            "n_clusters": len(clusters),
            "clusters": clusters,
            "selected_cluster_neb": neb_result,
            "s_cluster_neb": neb_result if args.neb_endpoint_symbol == "S" else None,
            "t_total_s": time.time() - t0,
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as fh:
            json.dump(result, fh, indent=2, cls=NumpyEncoder)
        print(f"[done] saved {args.output_json}", flush=True)
    finally:
        release_singleton(lock_path)


if __name__ == "__main__":
    main()
