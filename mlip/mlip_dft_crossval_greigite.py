#!/usr/bin/env python3
"""MLIP vs DFT cross-validation on greigite training dataset (s132 day 2).

Reads `greigite_w3.xyz` (extxyz from GPAW PBE single-points, 24 configs),
runs MACE-MP-0 large + CHGNet v0.3.0 single-points and (optional) relax
on each frame, computes ΔE / ΔF metrics for paper benchmark.

Apples-to-apples: preserves DFT initial_magmoms (ferrimagnetic AFM
sub-lattice ordering 8a tet=+3.5, 16d oct=-3.1 μB) for CHGNet via
`spin-aware` mode. MACE doesn't use magmoms at inference time
(architecture spin-agnostic) — so MACE numbers are direct PES comparison
without spin protocol consistency.

Output: one JSON per backend with per-frame {E_DFT, E_MLIP, ΔE, F_RMSE,
optional E_relaxed, n_steps, fmax_final}. Aggregate stats: ΔE rMSE,
F_RMSE, R² for E.

Usage on gomer:
  python mlip_dft_crossval_greigite.py \
    --backend mace \
    --runner /workspace/mace_canonical_1vacancy.py \
    --xyz /workspace/data/greigite_w3.xyz \
    --output-json /workspace/results_crossval/mace_greigite_crossval.json \
    --do-relax  # optional, adds 30s/config
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from ase.io import read
from ase.optimize import LBFGS, FIRE
from ase.calculators.singlepoint import SinglePointCalculator


def numpy_default(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(f"unsupported type: {type(o)}")


def load_runner(path: str):
    spec = importlib.util.spec_from_file_location("runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_dft_energy_forces(atoms):
    """Extract energy + forces from extxyz frame.

    ASE attaches a SinglePointCalculator; we pull from there.
    Falls back to atoms.info["energy"] and arrays["forces"] if older format.
    """
    try:
        e = atoms.get_potential_energy()
    except Exception:
        e = atoms.info.get("energy", None)
    try:
        f = atoms.get_forces()
    except Exception:
        f = atoms.arrays.get("forces", None)
    return e, f


def force_rmse(F_a, F_b):
    """Component-wise RMSE between force arrays (eV/Å)."""
    F_a = np.asarray(F_a)
    F_b = np.asarray(F_b)
    if F_a.shape != F_b.shape:
        return None
    return float(np.sqrt(((F_a - F_b) ** 2).mean()))


def force_max_abs_err(F_a, F_b):
    F_a = np.asarray(F_a); F_b = np.asarray(F_b)
    if F_a.shape != F_b.shape: return None
    return float(np.abs(F_a - F_b).max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["mace", "chgnet"])
    parser.add_argument("--runner", required=True,
                        help="Path to runner script (mace_canonical_1vacancy.py "
                             "or chgnet_canonical_1vacancy.py)")
    parser.add_argument("--xyz", required=True,
                        help="Path to greigite_w3.xyz")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--do-relax", action="store_true",
                        help="Also relax each config (LBFGS+FIRE fallback, "
                             "fmax=0.005, max 200 steps); else single-point only")
    parser.add_argument("--magmom-strategy", default="preserve",
                        choices=["preserve", "none", "default"],
                        help="preserve = use DFT initial magmoms from xyz "
                             "(apples-to-apples); none = zero out; "
                             "default = let runner.apply_magmoms decide")
    parser.add_argument("--chgnet-model-name", default="0.3.0")
    parser.add_argument("--spin-aware-chgnet", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--filter-config-prefix", default=None,
                        help="If set, only process frames whose config_type starts с "
                             "this prefix (e.g. greigite_bulk for primitive only)")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] reading {args.xyz}", flush=True)
    frames = read(args.xyz, index=":", format="extxyz")
    print(f"[load] {len(frames)} frames", flush=True)

    if args.filter_config_prefix:
        frames = [f for f in frames if f.info.get("config_type", "").startswith(
            args.filter_config_prefix)]
        print(f"[filter] {len(frames)} frames after prefix filter", flush=True)
    if args.max_frames:
        frames = frames[:args.max_frames]
        print(f"[filter] truncated to {len(frames)} frames", flush=True)

    print(f"[load] runner: {args.runner}", flush=True)
    runner = load_runner(args.runner)

    if args.backend == "mace":
        # MACE runner: load_calculator() — no args
        try:
            calc = runner.load_calculator()
        except TypeError:
            # New MACE runner signatures may take args namespace too
            from types import SimpleNamespace
            calc = runner.load_calculator(SimpleNamespace())
    else:  # chgnet
        # CHGNet runner: load_calculator(args) where args is argparse Namespace
        # with .chgnet_model_name and .spin_aware_chgnet attributes
        from types import SimpleNamespace
        chgnet_args = SimpleNamespace(
            chgnet_model_name=args.chgnet_model_name,
            spin_aware_chgnet=args.spin_aware_chgnet,
        )
        calc = runner.load_calculator(chgnet_args)
    print(f"[load] {args.backend} calculator ready", flush=True)

    apply_magmoms = getattr(runner, "apply_magmoms", None)

    results = []
    n_skipped = 0
    t_start = time.time()
    for i, atoms_dft in enumerate(frames):
        cfg = atoms_dft.info.get("config_type", f"frame_{i}")
        e_dft, f_dft = get_dft_energy_forces(atoms_dft)
        if e_dft is None or f_dft is None:
            print(f"[{i:3d}] {cfg}: DFT data missing — skip", flush=True)
            n_skipped += 1
            continue

        # Build fresh atoms for MLIP — DON'T attach DFT calc, only coords/cell/magmoms
        atoms = atoms_dft.copy()
        atoms.calc = None  # detach SinglePointCalculator

        # Magmom strategy
        if args.magmom_strategy == "preserve":
            # Already on atoms (via .copy()) — just leave it
            magmom_meta = {
                "strategy": "preserve",
                "abs_sum_muB": float(abs(atoms.get_initial_magnetic_moments()).sum()),
                "n_nonzero": int((abs(atoms.get_initial_magnetic_moments()) > 0.01).sum()),
            }
        elif args.magmom_strategy == "none":
            atoms.set_initial_magnetic_moments(np.zeros(len(atoms)))
            magmom_meta = {"strategy": "none", "abs_sum_muB": 0.0, "n_nonzero": 0}
        else:  # "default"
            if apply_magmoms is not None:
                magmom_meta = apply_magmoms(atoms, mode="default", fe_moment=3.5,
                                            ni_moment=2.0)
                magmom_meta["strategy"] = "default"
            else:
                magmom_meta = {"strategy": "default-noop"}

        atoms.calc = calc

        # Single-point
        t0 = time.time()
        try:
            e_mlip = float(atoms.get_potential_energy())
            f_mlip = atoms.get_forces()
        except Exception as exc:
            print(f"[{i:3d}] {cfg}: MLIP single-point FAILED — {exc}", flush=True)
            n_skipped += 1
            continue
        t_sp = time.time() - t0

        de = e_mlip - e_dft
        f_rmse = force_rmse(f_mlip, f_dft)
        f_maxerr = force_max_abs_err(f_mlip, f_dft)

        entry = {
            "frame_index": i,
            "config_type": cfg,
            "n_atoms": len(atoms),
            "E_DFT_eV": float(e_dft),
            "E_MLIP_eV": float(e_mlip),
            "delta_E_eV": float(de),
            "delta_E_per_atom_meV": float(de / len(atoms) * 1000),
            "force_rmse_eVA": f_rmse,
            "force_max_abs_err_eVA": f_maxerr,
            "magmom_meta": magmom_meta,
            "t_singlepoint_s": t_sp,
        }

        if args.do_relax:
            t0 = time.time()
            atoms_r = atoms.copy()
            atoms_r.calc = calc
            opt_l = LBFGS(atoms_r, logfile=None)
            conv = bool(opt_l.run(fmax=0.005, steps=150))
            n_l = int(opt_l.nsteps)
            n_f = 0
            if not conv:
                opt_f = FIRE(atoms_r, logfile=None)
                opt_f.run(fmax=0.005, steps=100)
                n_f = int(opt_f.nsteps)
            e_relax = float(atoms_r.get_potential_energy())
            f_relax_max = float(np.linalg.norm(atoms_r.get_forces(), axis=1).max())
            t_rx = time.time() - t0
            entry["relax"] = {
                "E_relaxed_eV": e_relax,
                "fmax_final_eVA": f_relax_max,
                "delta_E_to_singlepoint_eV": e_relax - e_mlip,
                "n_steps_lbfgs": n_l,
                "n_steps_fire": n_f,
                "t_relax_s": t_rx,
            }

        results.append(entry)
        relax_extra = (f", E_rx-E_sp={entry['relax']['delta_E_to_singlepoint_eV']:.4f} eV "
                       f"({entry['relax']['n_steps_lbfgs']}+{entry['relax']['n_steps_fire']} steps)") if args.do_relax else ""
        print(f"[{i:3d}] {cfg}: ΔE={de:+.4f} eV ({de/len(atoms)*1000:+.2f} meV/atom), "
              f"F_RMSE={f_rmse:.4f}, max|F_err|={f_maxerr:.4f}{relax_extra}", flush=True)

    # Aggregate
    n_ok = len(results)
    if n_ok > 0:
        de_arr = np.array([r["delta_E_eV"] for r in results])
        de_pa_arr = np.array([r["delta_E_per_atom_meV"] for r in results])
        f_arr = np.array([r["force_rmse_eVA"] for r in results])
        # R² for E_MLIP vs E_DFT
        e_dft_arr = np.array([r["E_DFT_eV"] for r in results])
        e_mlip_arr = np.array([r["E_MLIP_eV"] for r in results])
        ss_res = float(((e_mlip_arr - e_dft_arr) ** 2).sum())
        ss_tot = float(((e_dft_arr - e_dft_arr.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    else:
        de_arr = de_pa_arr = f_arr = np.array([])
        r2 = None

    summary = {
        "backend": args.backend,
        "runner": args.runner,
        "xyz": args.xyz,
        "magmom_strategy": args.magmom_strategy,
        "n_frames_total": len(frames),
        "n_frames_processed": n_ok,
        "n_frames_skipped": n_skipped,
        "delta_E_mean_eV": float(de_arr.mean()) if n_ok else None,
        "delta_E_std_eV": float(de_arr.std()) if n_ok else None,
        "delta_E_rmse_eV": float(np.sqrt((de_arr ** 2).mean())) if n_ok else None,
        "delta_E_per_atom_mean_meV": float(de_pa_arr.mean()) if n_ok else None,
        "delta_E_per_atom_rmse_meV": float(np.sqrt((de_pa_arr ** 2).mean())) if n_ok else None,
        "force_rmse_mean_eVA": float(f_arr.mean()) if n_ok else None,
        "force_rmse_max_eVA": float(f_arr.max()) if n_ok else None,
        "R2_E_MLIP_vs_DFT": r2,
        "do_relax": args.do_relax,
        "t_total_s": time.time() - t_start,
        "frames": results,
    }

    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2, default=numpy_default)
    print(f"\n[done] saved {args.output_json}", flush=True)
    print(f"  n_frames: {n_ok}/{len(frames)}", flush=True)
    print(f"  ΔE rMSE: {summary['delta_E_rmse_eV']:.4f} eV "
          f"(per atom: {summary['delta_E_per_atom_rmse_meV']:.2f} meV)" if n_ok else "  no frames", flush=True)
    print(f"  F_RMSE: {summary['force_rmse_mean_eVA']:.4f} eV/Å mean, "
          f"{summary['force_rmse_max_eVA']:.4f} max" if n_ok else "", flush=True)
    print(f"  R² (E_MLIP vs E_DFT): {r2:.4f}" if r2 is not None else "", flush=True)


if __name__ == "__main__":
    main()
