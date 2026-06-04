#!/usr/bin/env python3
"""
Multi-Endpoint v3: MACE relax all candidate H sites for V_Fe pocket.

Input: candidate xyz files (from enumeration script) + manifest.
Output: relaxed xyz files + energy/fmax/displacement summary per candidate.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read, write
from ase.optimize import BFGS

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")


class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, Path):        return str(obj)
        return super().default(obj)


def load_calculator():
    from mace.calculators import mace_mp
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] MACE-MP-0 large on {device}", flush=True)
    return mace_mp(model="large", device=device, default_dtype="float64")


def relax_one(atoms_path, out_path, log_path, calc, fmax=0.05, max_steps=300):
    atoms = read(atoms_path)
    atoms.calc = calc
    pos_init = atoms.get_positions().copy()

    t0 = time.time()
    opt = BFGS(atoms, logfile=str(log_path))
    try:
        opt.run(fmax=fmax, steps=max_steps)
    except Exception as e:
        print(f"  [relax] error: {e}", flush=True)

    elapsed = time.time() - t0
    e_final = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    fmax_final = float(np.linalg.norm(forces, axis=1).max())

    delta = atoms.get_positions() - pos_init
    max_disp = float(np.linalg.norm(delta, axis=1).max())

    write(out_path, atoms, format="extxyz")

    return {
        "out_xyz": str(out_path),
        "n_steps": int(opt.nsteps),
        "fmax_final": fmax_final,
        "E_final_eV": e_final,
        "max_disp_A": max_disp,
        "elapsed_s": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=300)
    args = ap.parse_args()

    cand_dir = Path(args.candidate_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = cand_dir / "enumeration_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"=== Multi-endpoint MACE relax ===", flush=True)
    print(f"  Candidates: {manifest['n_candidates']} from {cand_dir}", flush=True)
    print(f"  Output: {out_dir}", flush=True)

    calc = load_calculator()

    summary = []
    t_global = time.time()
    for i, cand in enumerate(manifest["candidates"]):
        cand_id = cand["id"]
        in_xyz = cand_dir / f"{cand_id}.xyz"
        out_xyz = out_dir / f"relaxed_{cand_id}.xyz"
        log = out_dir / f"relax_{cand_id}.log"
        print(f"\n[{i+1}/{len(manifest['candidates'])}] {cand_id} ({cand['class']})", flush=True)
        try:
            result = relax_one(in_xyz, out_xyz, log, calc, args.fmax, args.max_steps)
            result["candidate_id"] = cand_id
            result["class"] = cand["class"]
            result["anchor"] = cand["anchor"]
            print(f"  done in {result['elapsed_s']:.1f}s  steps={result['n_steps']}  "
                  f"fmax={result['fmax_final']:.4f}  E={result['E_final_eV']:.3f}  "
                  f"max_disp={result['max_disp_A']:.3f}", flush=True)
            summary.append(result)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            summary.append({"candidate_id": cand_id, "class": cand["class"], "error": str(e)})

    successful = [s for s in summary if "E_final_eV" in s]
    successful.sort(key=lambda x: x["E_final_eV"])

    print(f"\n{'='*70}\nRanked by energy (lowest first):\n{'='*70}", flush=True)
    for i, s in enumerate(successful):
        print(f"  {i+1:2d}. E={s['E_final_eV']:.4f}  fmax={s['fmax_final']:.4f}  "
              f"{s['candidate_id']} ({s['class']})", flush=True)

    out_summary = {
        "candidate_dir": str(cand_dir),
        "n_candidates": manifest["n_candidates"],
        "n_relaxed": len(successful),
        "fmax_target": args.fmax,
        "results": summary,
        "total_elapsed_s": time.time() - t_global,
    }
    with open(out_dir / "relax_summary.json", "w") as f:
        json.dump(out_summary, f, indent=2, cls=SafeJSONEncoder)

    with open(out_dir / "DONE_multi_endpoint_relax", "w") as f:
        f.write(f"Relaxed {len(successful)}/{manifest['n_candidates']} candidates in "
                f"{time.time()-t_global:.0f}s\n")
        for s in successful[:5]:
            f.write(f"  {s['candidate_id']}: E={s['E_final_eV']:.4f} eV\n")

    print(f"\n[DONE] {len(successful)}/{manifest['n_candidates']} in "
          f"{time.time()-t_global:.0f}s.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
