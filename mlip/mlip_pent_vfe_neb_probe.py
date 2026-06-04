#!/usr/bin/env python3
"""
Pent V_Fe S-H lateral-hop MLIP NEB pre-flight PROBE (Phase 4, $0 gomer).

Purpose: de-risk the production DFT pent V_Fe NEB BEFORE spending A100.
This is NOT a paper-grade barrier -- it is a PARAMETER-TUNING probe answering:
  1. E_a/|dE| ratio  -> string-method vs CI-NEB decision (gate G_v3.3)
  2. k_spring robustness -> does k=0.3 roll the band off the ridge (pyrite s156
     lesson) while k=1.5-3.0 holds it? Measured by a band-roll-off detector.
  3. n_images sufficiency.
  4. MACE vs CHGNet agreement (foundation-MLIP OOD check on cubane Fe4S4).

Endpoints are LOADED (the v3 enumeration already produced the correct V_Fe + H-on-S
geometries), NOT rebuilt. Canonical pent V_Fe hop = H on S98 <-> H on S62
(V_Fe index 119, manifest hop_distance 3.868 A). Both in the SAME S-coordinated
SOAP cluster -> no Fe-hydride OOD contamination.

Run ONE model per invocation (single consumer GPU, no MPS). Deploy script calls
this twice: --model mace, then --model chgnet.

Rule #0 (CLAUDE.md): launched via a deploy script, never bare docker exec.
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

from ase.io import read
from ase.optimize import LBFGS, FIRE
from ase.mep import NEB

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --- numpy safe JSON ---
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


def check_vram(label=""):
    if not torch.cuda.is_available():
        print(f"[VRAM {label}] CUDA not available (CPU)", flush=True)
        return
    torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    alloc = torch.cuda.memory_allocated(0) / 1e9
    print(f"[VRAM {label}] total={total:.2f} GB alloc={alloc:.2f} GB free={total-alloc:.2f} GB",
          flush=True)


# --- geometry helpers (MIC) ---
def mic_vec(cell, p_from, p_to):
    d = np.asarray(p_to) - np.asarray(p_from)
    fc = np.linalg.solve(cell.T, d)
    fc -= np.round(fc)
    return cell.T @ fc


def nearest_S(atoms, h_idx):
    """Return (index, distance) of S atom nearest to atom h_idx (MIC)."""
    cell = atoms.cell.array
    syms = atoms.get_chemical_symbols()
    best_i, best_d = None, 1e9
    for j in range(len(atoms)):
        if syms[j] != "S":
            continue
        d = float(np.linalg.norm(mic_vec(cell, atoms.positions[h_idx], atoms.positions[j])))
        if d < best_d:
            best_i, best_d = j, d
    return best_i, best_d


def point_to_line_dist(cell, p, a, b):
    """Perp distance of point p from the MIC line a->b (all positions)."""
    ab = mic_vec(cell, a, b)
    ap = mic_vec(cell, a, p)
    L = np.linalg.norm(ab)
    if L < 1e-9:
        return float(np.linalg.norm(ap))
    t = np.dot(ap, ab) / (L * L)
    closest = ap - t * ab
    return float(np.linalg.norm(closest))


# --- calculators ---
def load_mace():
    from mace.calculators import mace_mp
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] MACE-MP-0 large on {device}, float64", flush=True)
    return mace_mp(model="large", device=device, default_dtype="float64")


def load_chgnet():
    from chgnet.model import CHGNet
    from chgnet.model.dynamics import CHGNetCalculator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] CHGNet pretrained on {device}", flush=True)
    model = CHGNet.load()
    try:
        return CHGNetCalculator(model=model, use_device=device)
    except TypeError:
        return CHGNetCalculator(model=model)


# --- endpoint relax ---
def relax_endpoint(atoms, calc, tag, fmax, max_steps):
    a = atoms.copy()
    a.calc = calc
    opt = LBFGS(a, logfile=None)
    conv = bool(opt.run(fmax=fmax, steps=max_steps))
    e = float(a.get_potential_energy())
    fmx = float(np.linalg.norm(a.get_forces(), axis=1).max())
    h_idx = len(a) - 1
    sN, sD = nearest_S(a, h_idx)
    print(f"  [{tag}] E={e:.4f} eV steps={opt.nsteps} conv={conv} fmax={fmx:.3f} "
          f"nearestS=#{sN} d={sD:.3f}", flush=True)
    return a, dict(E_eV=e, steps=int(opt.nsteps), converged=conv, fmax=fmx,
                   nearest_S=int(sN), d_S_H=sD, h_idx=int(h_idx))


# --- one NEB run ---
def run_neb(endA, endB, calc, k_spring, climb, n_images, fmax_neb, max_steps_neb,
            s_A, s_B):
    cell = endA.cell.array
    h_idx = len(endA) - 1
    n_inter = n_images - 2
    images = [endA.copy()]
    for _ in range(n_inter):
        images.append(endA.copy())
    images.append(endB.copy())
    for img in images:
        img.calc = calc

    neb = NEB(images, climb=climb, method="improvedtangent",
              allow_shared_calculator=True, k=k_spring)
    idpp_ok = True
    try:
        neb.interpolate("idpp")
    except Exception as e:
        idpp_ok = False
        print(f"    IDPP failed ({e}); linear fallback", flush=True)
        neb.interpolate()

    t0 = time.time()
    opt = FIRE(neb, logfile=None)
    conv = bool(opt.run(fmax=fmax_neb, steps=max_steps_neb))
    dt = time.time() - t0

    energies = np.array([float(im.get_potential_energy()) for im in images])
    rel = energies - energies[0]
    e_a = float(rel.max())
    e_rxn = float(rel[-1])
    ratio = e_a / max(abs(e_rxn), 1e-3)

    # CI image fmax
    ci_idx = int(np.argmax(energies))
    fmx_ci = float(np.linalg.norm(images[ci_idx].get_forces(), axis=1).max())

    # --- band-roll-off detector (the pyrite failure signature) ---
    # Track H's nearest-S per image. On-path = nearest S in {s_A, s_B}.
    # A third-S visit on an interior image = band slid off the proton-transfer ridge.
    near_seq = []
    perp_seq = []
    third_visits = []
    for i, im in enumerate(images):
        sN, sD = nearest_S(im, h_idx)
        near_seq.append(int(sN))
        perp = point_to_line_dist(cell, im.positions[h_idx],
                                  endA.positions[h_idx], endB.positions[h_idx])
        perp_seq.append(perp)
        if 0 < i < len(images) - 1 and sN not in (s_A, s_B):
            third_visits.append(dict(image=i, nearest_S=int(sN), d_S_H=float(sD)))
    rolled_off = len(third_visits) > 0
    max_perp = float(max(perp_seq))

    return dict(
        k_spring=float(k_spring), climb=bool(climb), n_images=int(n_images),
        idpp_ok=idpp_ok, converged=conv, steps=int(opt.nsteps),
        E_a_eV=e_a, E_rxn_eV=e_rxn, E_a_over_dE=float(ratio),
        fmax_CI=fmx_ci, ci_image=ci_idx, t_s=dt,
        band_rel_eV=[float(x) for x in rel],
        nearest_S_per_image=near_seq,
        h_perp_off_line_A=perp_seq, max_h_perp_A=max_perp,
        rolled_off_ridge=rolled_off, third_S_visits=third_visits,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mace", "chgnet"])
    ap.add_argument("--endpoint-a", required=True)
    ap.add_argument("--endpoint-b", required=True)
    ap.add_argument("--output-dir", default="/workspace/results")
    ap.add_argument("--k-springs", default="0.3,1.0,3.0")
    ap.add_argument("--climb-modes", default="plain,ci")
    ap.add_argument("--n-images", type=int, default=7)
    ap.add_argument("--fmax-neb", type=float, default=0.05)
    ap.add_argument("--max-steps-neb", type=int, default=300)
    ap.add_argument("--endpoint-fmax", type=float, default=0.05)
    ap.add_argument("--max-steps-endpoint", type=int, default=300)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    k_list = [float(x) for x in args.k_springs.split(",") if x.strip()]
    climb_list = [("ci" if c.strip() == "ci" else "plain") for c in args.climb_modes.split(",") if c.strip()]

    print("=" * 70, flush=True)
    print(f"PENT V_Fe NEB PROBE  model={args.model}", flush=True)
    print(f"k_springs={k_list}  climb={climb_list}  n_images={args.n_images}", flush=True)
    print("=" * 70, flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    check_vram("init")

    endA_raw = read(args.endpoint_a)
    endB_raw = read(args.endpoint_b)
    assert len(endA_raw) == len(endB_raw), "atom count mismatch"
    assert endA_raw.get_chemical_symbols()[-1] == "H", "H must be last atom"

    calc = load_mace() if args.model == "mace" else load_chgnet()

    print(f"\n[relax endpoints with {args.model}]", flush=True)
    endA, infoA = relax_endpoint(endA_raw, calc, "endA", args.endpoint_fmax, args.max_steps_endpoint)
    endB, infoB = relax_endpoint(endB_raw, calc, "endB", args.endpoint_fmax, args.max_steps_endpoint)

    s_A, s_B = infoA["nearest_S"], infoB["nearest_S"]
    h_idx = len(endA) - 1
    h_disp = float(np.linalg.norm(
        mic_vec(endA.cell.array, endA.positions[h_idx], endB.positions[h_idx])))
    dE_endpoints = infoB["E_eV"] - infoA["E_eV"]
    same_basin = bool(h_disp < 0.5 and s_A == s_B)
    print(f"\n[endpoint summary {args.model}] s_A=#{s_A} s_B=#{s_B} "
          f"H_disp={h_disp:.3f} A  dE={dE_endpoints*1000:.1f} meV  same_basin={same_basin}",
          flush=True)

    result = dict(
        model=args.model,
        endpoint_a=args.endpoint_a, endpoint_b=args.endpoint_b,
        n_atoms=len(endA),
        endpoint_A=infoA, endpoint_B=infoB,
        s_A=int(s_A), s_B=int(s_B),
        H_disp_AB_A=h_disp, dE_endpoints_eV=dE_endpoints,
        endpoint_same_basin=same_basin,
        t_start=time.time(), runs=[],
    )

    if same_basin:
        print("[ABORT] endpoints collapsed to same basin under this calc; "
              "NEB scan skipped", flush=True)
        result["aborted"] = "endpoint_same_basin_under_calc"
    else:
        for k in k_list:
            for climb_tag in climb_list:
                climb = (climb_tag == "ci")
                print(f"\n[NEB] k={k} climb={climb_tag} n={args.n_images}", flush=True)
                try:
                    r = run_neb(endA, endB, calc, k, climb, args.n_images,
                                args.fmax_neb, args.max_steps_neb, s_A, s_B)
                    r["climb_tag"] = climb_tag
                    flag = "ROLLED-OFF" if r["rolled_off_ridge"] else "on-path"
                    print(f"    E_a={r['E_a_eV']*1000:.0f} meV  E_rxn={r['E_rxn_eV']*1000:.0f} meV  "
                          f"E_a/|dE|={r['E_a_over_dE']:.2f}  conv={r['converged']} "
                          f"steps={r['steps']} fmaxCI={r['fmax_CI']:.3f}  "
                          f"max_perp={r['max_h_perp_A']:.2f}A  [{flag}]", flush=True)
                    result["runs"].append(r)
                except Exception as e:
                    print(f"    FAILED: {e}", flush=True)
                    result["runs"].append(dict(k_spring=k, climb_tag=climb_tag,
                                               error=str(e),
                                               traceback=traceback.format_exc()))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    result["t_total_s"] = time.time() - result["t_start"]
    json_path = out / f"pent_vfe_neb_probe_{args.model}.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    print(f"\n[DONE] saved {json_path}  ({len(result['runs'])} runs, "
          f"{result['t_total_s']:.0f}s)", flush=True)

    # quick band overlay PNG
    try:
        ok = [r for r in result["runs"] if "band_rel_eV" in r]
        if ok:
            fig, ax = plt.subplots(figsize=(8, 5))
            for r in ok:
                x = np.linspace(0, 1, len(r["band_rel_eV"]))
                ls = "-" if r["climb"] else "--"
                lbl = f"k={r['k_spring']} {r['climb_tag']}" + (" ROLLED" if r["rolled_off_ridge"] else "")
                ax.plot(x, np.array(r["band_rel_eV"]) * 1000, ls, marker="o", label=lbl, alpha=0.8)
            ax.set_xlabel("reaction coordinate")
            ax.set_ylabel("E (meV)")
            ax.set_title(f"pent V_Fe S-H hop probe ({args.model}) S{s_A}<->S{s_B}")
            ax.axhline(0, color="gray", ls=":", alpha=0.4)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out / f"pent_vfe_neb_probe_{args.model}.png", dpi=130)
            plt.close(fig)
    except Exception as e:
        print(f"PNG failed: {e}", flush=True)

    Path(f"/workspace/DONE_pent_vfe_probe_{args.model}").write_text(str(json_path))
    sys.exit(0)


if __name__ == "__main__":
    main()
